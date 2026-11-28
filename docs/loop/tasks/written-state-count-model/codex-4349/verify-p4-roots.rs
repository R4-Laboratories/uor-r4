//! Read the frozen P4 codebook without evaluating any model transitions.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use uor_r4_integer::discrete_d11::{IntegerDiscreteModel, MAGIC};

const P4_SHA: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const TRUNK_SHA: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
const P4_BYTES: u64 = 3_987_940_843;
const TRUNK_BYTES: u64 = 86_536_132;

struct HashingReader<R> {
    reader: R,
    hash: Sha256,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let n = self.reader.read(buffer)?;
        self.hash.update(&buffer[..n]);
        Ok(n)
    }
}

fn verified_p4_trunk(path: &Path) -> Result<IntegerDiscreteModel, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    if file.metadata()?.len() != P4_BYTES {
        return Err("wrong P4 byte length".into());
    }
    let mut source = HashingReader {
        reader: BufReader::with_capacity(1 << 20, file),
        hash: Sha256::new(),
    };
    let mut magic = [0; 8];
    let mut header_len = [0; 8];
    source.read_exact(&mut magic)?;
    source.read_exact(&mut header_len)?;
    let header_len = u64::from_le_bytes(header_len);
    if &magic != MAGIC || !(1..=4096).contains(&header_len) {
        return Err("invalid P4 preamble".into());
    }
    let mut header = vec![0; usize::try_from(header_len)?];
    source.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header)?;
    let expected_state_len = P4_BYTES - 16 - header_len - TRUNK_BYTES;
    if header["schema"] != "uor-r4.lut-discrete-p4/1"
        || header["trunk_sha256"] != TRUNK_SHA
        || header["trunk_bytes"].as_u64() != Some(TRUNK_BYTES)
        || header["state_bytes"].as_u64() != Some(expected_state_len)
    {
        return Err("wrong P4 embedded trunk binding".into());
    }

    let mut limit = (&mut source).take(TRUNK_BYTES);
    let trunk = IntegerDiscreteModel::from_reader(&mut limit)?;
    if limit.limit() != 0 || trunk.artifact_sha256() != TRUNK_SHA {
        return Err("wrong embedded trunk SHA-256 or length".into());
    }
    let tail = io::copy(&mut source, &mut io::sink())?;
    if tail != expected_state_len || format!("{:x}", source.hash.finalize()) != P4_SHA {
        return Err("wrong P4 parent SHA-256 or length".into());
    }
    Ok(trunk)
}

fn coarse(fingerprint: u32) -> u32 {
    ((fingerprint >> 8) & 0xf0) | (fingerprint & 0x0f)
}

fn unordered_pairs(n: usize) -> usize {
    n.saturating_sub(1) * n / 2
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("usage: verify-p4-roots P4.d11")?;
    if args.next().is_some() {
        return Err("usage: verify-p4-roots P4.d11".into());
    }
    let trunk = verified_p4_trunk(Path::new(&path))?;
    let roots = trunk.packed_codebook_roots();
    if roots.len() != 4096 || roots.iter().any(|&key| key > 0xffff) {
        return Err("expected a full 4096-ID four-root codebook".into());
    }

    let mut full_buckets = BTreeMap::<u32, Vec<usize>>::new();
    let mut coarse_buckets = BTreeMap::<u32, Vec<usize>>::new();
    for (id, &key) in roots.iter().enumerate().skip(1) {
        full_buckets.entry(key).or_default().push(id);
        coarse_buckets.entry(coarse(key)).or_default().push(id);
    }
    let mut full_histogram = BTreeMap::<usize, usize>::new();
    for ids in full_buckets.values() {
        *full_histogram.entry(ids.len()).or_default() += 1;
    }
    let full_alias_pairs: usize = full_buckets
        .values()
        .map(|ids| unordered_pairs(ids.len()))
        .sum();
    let same_coarse_pairs: usize = coarse_buckets
        .values()
        .map(|ids| unordered_pairs(ids.len()))
        .sum();
    let distinct_full = |ids: &[usize]| {
        ids.iter()
            .map(|&id| roots[id])
            .collect::<BTreeSet<_>>()
            .len()
    };
    let separable_coarse = coarse_buckets.values().filter(|ids| distinct_full(ids) > 1);
    let unique_full_ids = full_histogram.get(&1).copied().unwrap_or_default();
    let watched = |id: usize| {
        let key = roots[id];
        let low_high = coarse(key);
        let coarse_members = &coarse_buckets[&low_high];
        json!({
            "id": id,
            "roots": [key & 15, (key >> 4) & 15, (key >> 8) & 15, key >> 12],
            "fingerprint_hex": format!("{key:04x}"),
            "full_members": full_buckets[&key],
            "coarse_hex": format!("{low_high:02x}"),
            "coarse_occupancy": coarse_members.len(),
            "coarse_distinct_full": distinct_full(coarse_members),
            "different_full_rivals_in_coarse": coarse_members.len() - full_buckets[&key].len(),
        })
    };
    let report = json!({
        "p4_bytes": P4_BYTES,
        "p4_sha256": P4_SHA,
        "trunk_bytes": TRUNK_BYTES,
        "trunk_sha256": TRUNK_SHA,
        "legal_ids": roots.len() - 1,
        "distinct_full": full_buckets.len(),
        "unique_full_ids": unique_full_ids,
        "ids_with_full_aliases": roots.len() - 1 - unique_full_ids,
        "full_bucket_size_histogram": full_histogram,
        "peak_full_bucket_size": full_buckets.values().map(Vec::len).max(),
        "distinct_coarse": coarse_buckets.len(),
        "coarse_buckets_with_a_separable_pair": separable_coarse.clone().count(),
        "ids_in_those_coarse_buckets": separable_coarse.map(Vec::len).sum::<usize>(),
        "same_coarse_unordered_pairs": same_coarse_pairs,
        "same_coarse_pairs_separable_by_full_roots": same_coarse_pairs - full_alias_pairs,
        "full_alias_pairs": full_alias_pairs,
        "space": watched(223),
        "eos": watched(1),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
