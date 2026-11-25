//! Resident UORKNC03 CUDA target feature capturer, host oracle, and native replay verifier.
#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]

mod cuda;

use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};
use uor_r4_count_runtime::{kn_score::KnDiscounts, CompactCountLookup};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub(crate) type Pin = [u8; 32];
pub(crate) const ORDERS: usize = 11;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const PROBE_WORDS: usize = 12;
const MAX_BATCH: usize = 65_536;
const MAX_UORT_BYTES: u64 = 64 << 20;
const MAX_ARTIFACT_BYTES: u64 = 4_000_000_000;
const FINITE_ARTIFACT_BYTES: u64 = 32 << 20;
const FINITE_TOKENS: usize = 256;
const FEATURE_ROW_BYTES: usize = ORDERS * (4 + 8) + 8;

pub(crate) struct Input {
    pub(crate) ids: Vec<u16>,
    pub(crate) targets: usize,
    pub(crate) pin: Pin,
}

fn pin(value: &str) -> Result<Pin> {
    let mut pin = [0; 32];
    hex::decode_to_slice(value, &mut pin)?;
    Ok(pin)
}

pub(crate) fn sha(bytes: &[u8]) -> Pin {
    Sha256::digest(bytes).into()
}

fn uint32(bytes: &[u8]) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes.try_into()?))
}

fn uint64(bytes: &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes.try_into()?))
}

fn parse_input(bytes: &[u8], expected: Pin) -> Result<Input> {
    if sha(bytes) != expected {
        return Err("UORT digest mismatch".into());
    }
    let (header, ids) = bytes.split_at_checked(64).ok_or("truncated UORT header")?;
    if &header[..4] != b"UORT" || uint32(&header[4..8])? != 1 || uint32(&header[16..20])? != 4096 {
        return Err("wrong UORT v1 header or token domain".into());
    }
    if uint64(&header[8..16])?.checked_mul(2) != Some(ids.len() as u64) || ids.is_empty() {
        return Err("wrong UORT payload length".into());
    }
    let mut tokens = Vec::with_capacity(ids.len() / 2);
    let mut previous = 1;
    let mut targets = 0;
    for pair in ids.chunks_exact(2) {
        let id = u16::from_le_bytes([pair[0], pair[1]]);
        if id >= 4096 || (id == 0) != (previous == 1) {
            return Err(format!("invalid frame or token ID at UORT index {}", tokens.len()).into());
        }
        targets += usize::from(id != 0);
        previous = id;
        tokens.push(id);
    }
    if previous != 1 || targets == 0 {
        return Err("UORT must end at EOS and include scored targets".into());
    }
    Ok(Input {
        ids: tokens,
        targets,
        pin: expected,
    })
}

fn bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len > max {
        return Err(format!("{} exceeds the configured byte bound", path.display()).into());
    }
    let mut bytes = Vec::with_capacity(len as usize);
    (&mut file).take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != len {
        return Err("file length changed during bounded read".into());
    }
    Ok(bytes)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct Rows<'a> {
    ids: &'a [u16],
    at: usize,
    frame: usize,
    ordinal: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl<'a> Rows<'a> {
    pub(crate) fn new(input: &'a Input) -> Self {
        Self {
            ids: &input.ids,
            at: 0,
            frame: 0,
            ordinal: 0,
        }
    }

    pub(crate) fn fill(&mut self, batch: usize, probes: &mut Vec<u16>) -> usize {
        probes.clear();
        let first = self.ordinal;
        while self.at < self.ids.len() && self.ordinal - first < batch {
            let at = self.at;
            self.at += 1;
            if self.ids[at] == 0 {
                self.frame = at;
                continue;
            }
            let causal = &self.ids[self.frame.max(at.saturating_sub(10))..=at];
            let start = probes.len();
            probes.resize(start + PROBE_WORDS, 0);
            probes[start] = causal.len() as u16;
            probes[start + 1..start + 1 + causal.len()].copy_from_slice(causal);
            self.ordinal += 1;
        }
        self.ordinal - first
    }
}

pub(crate) struct Metadata {
    pub(crate) descriptors: Vec<u64>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    headers: Vec<(usize, Vec<u8>)>,
}

impl Metadata {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn check_chunk(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let end = offset + bytes.len();
        for (start, header) in &self.headers {
            let lo = offset.max(*start);
            let hi = end.min(*start + header.len());
            if lo < hi && bytes[lo - offset..hi - offset] != header[lo - start..hi - start] {
                return Err("compact v3 header bytes changed during upload".into());
            }
        }
        Ok(())
    }
}

pub(crate) fn metadata(file: &mut File, length: u64, train: Pin) -> Result<Metadata> {
    if !(44..=MAX_ARTIFACT_BYTES).contains(&length) {
        return Err("invalid UORKNC03 byte length".into());
    }
    let mut main = [0u8; 44];
    file.read_exact(&mut main)?;
    if &main[..8] != b"UORKNC03" || uint32(&main[8..12])? != 10 || main[12..] != train {
        return Err("wrong UORKNC03 format, maximum order or TRAIN digest".into());
    }
    let mut data = Vec::with_capacity(ORDERS * 18);
    let mut headers = vec![(0, main.to_vec())];
    let mut at = main.len() as u64;
    let mut parent_contexts = 0u32;
    for k in 0..ORDERS {
        file.seek(SeekFrom::Start(at))?;
        let mut header = [0u8; 88];
        file.read_exact(&mut header)?;
        headers.push((at as usize, header.to_vec()));
        let followers = uint32(&header[0..4])?;
        let contexts = uint32(&header[4..8])?;
        let parent_deg_esc_len = uint32(&header[72..76])?;
        let fol_deg_esc_len = uint32(&header[76..80])?;
        let esc16_len = uint32(&header[80..84])?;
        let esc32_len = uint32(&header[84..88])?;
        if (followers == 0) != (contexts == 0) || (k == 0 && (contexts > 1 || parent_deg_esc_len != 0)) {
            return Err("invalid UORKNC03 section header".into());
        }
        let shared = k == ORDERS - 1;
        let mut spectra = [0u64; 4];
        for (count, bytes) in spectra.iter_mut().zip(header[40..72].chunks_exact(8)) {
            *count = uint64(bytes)?;
        }
        let (discounts, _) = KnDiscounts::with_fallback(spectra)?;

        let mut p = at + 88;
        let (parent_sb, parent_deg, parent_deg_esc, child_toks) = if k == 0 {
            (0, 0, 0, 0)
        } else {
            let psb = p;
            p += (u64::from(parent_contexts).div_ceil(64) + 1) * 4;
            let pdeg = p;
            p += u64::from(parent_contexts);
            let pdesc = p;
            p += u64::from(parent_deg_esc_len) * 6;
            let ctoks = p;
            p += (u64::from(contexts) * 3).div_ceil(2);
            (psb, pdeg, pdesc, ctoks)
        };
        let fol_sb = p;
        p += (u64::from(contexts).div_ceil(64) + 1) * 4;
        let fol_deg = p;
        p += u64::from(contexts);
        let fol_deg_esc = p;
        p += u64::from(fol_deg_esc_len) * 6;
        let fol_words = p;
        p += u64::from(followers) * 2;
        let esc_sb = p;
        p += (u64::from(followers).div_ceil(64) + 1) * 4;
        let esc16 = p;
        p += u64::from(esc16_len) * if shared { 2 } else { 4 };
        let esc32 = p;
        p += u64::from(esc32_len) * if shared { 8 } else { 12 };
        if p > length {
            return Err("UORKNC03 section exceeds file length".into());
        }
        at = p;
        data.extend([
            parent_sb,
            parent_deg,
            parent_deg_esc,
            child_toks,
            fol_sb,
            fol_deg,
            fol_deg_esc,
            fol_words,
            esc_sb,
            esc16,
            esc32,
            u64::from(contexts) | (u64::from(followers) << 32),
            u64::from(parent_contexts) | (u64::from(parent_deg_esc_len) << 32),
            u64::from(fol_deg_esc_len) | (u64::from(esc16_len) << 32),
            u64::from(esc32_len),
        ]);
        data.extend(discounts.dq);
        parent_contexts = contexts;
    }
    if at != length {
        return Err("UORKNC03 artifact has trailing bytes".into());
    }
    file.rewind()?;
    Ok(Metadata {
        descriptors: data,
        headers,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Feature {
    pub(crate) counts: [u32; ORDERS],
    pub(crate) totals: [u64; ORDERS],
    pub(crate) mass: u64,
}

impl Feature {
    pub(crate) fn write(&self, output: &mut impl Write) -> Result<()> {
        if !(1..=(1 << 48)).contains(&self.mass)
            || self
                .counts
                .iter()
                .zip(self.totals)
                .any(|(&count, total)| u64::from(count) > total)
        {
            return Err("invalid feature count, total or Q48 word".into());
        }
        let mut row = [0u8; FEATURE_ROW_BYTES];
        for (dst, count) in row[..44].chunks_exact_mut(4).zip(self.counts) {
            dst.copy_from_slice(&count.to_le_bytes());
        }
        for (dst, total) in row[44..132].chunks_exact_mut(8).zip(self.totals) {
            dst.copy_from_slice(&total.to_le_bytes());
        }
        row[132..].copy_from_slice(&self.mass.to_le_bytes());
        output.write_all(&row)?;
        Ok(())
    }
}

pub(crate) struct Output {
    partial: OsString,
    pub(crate) writer: BufWriter<File>,
}

impl Output {
    pub(crate) fn new(path: &Path, input: &Input, train: Pin, compact: Pin) -> Result<Self> {
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
            Ok(_) => return Err("refusing to overwrite existing output".into()),
        }
        let mut partial = path.as_os_str().to_owned();
        partial.push(".partial");
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&partial)?;
        let mut writer = BufWriter::with_capacity(1 << 20, file);
        writer.write_all(b"UORCBF01")?;
        writer.write_all(&(input.targets as u64).to_le_bytes())?;
        writer.write_all(&train)?;
        writer.write_all(&compact)?;
        writer.write_all(&input.pin)?;
        Ok(Self { partial, writer })
    }

    pub(crate) fn finish(mut self, path: &Path, targets: usize) -> Result<()> {
        self.writer.flush()?;
        let file = self.writer.into_inner()?;
        if file.metadata()?.len() != 112 + (targets * FEATURE_ROW_BYTES) as u64 {
            return Err("feature output length disagrees with admitted target count".into());
        }
        file.sync_all()?;
        drop(file);
        fs::hard_link(&self.partial, path)?;
        fs::remove_file(&self.partial)?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        File::open(dir)?.sync_all()?;
        Ok(())
    }
}

fn finite_oracle(
    artifact: &Path,
    train: Pin,
    compact: Pin,
    input: &Input,
    dest: &Path,
) -> Result<()> {
    if input.ids.len() > FINITE_TOKENS {
        return Err("host oracle is restricted to the finite fixture".into());
    }
    let bytes = bounded(artifact, FINITE_ARTIFACT_BYTES)?;
    let native = CompactCountLookup::from_bytes(&bytes, train, compact)?;
    let mut output = Output::new(dest, input, train, compact)?;
    for record in input.ids.split_inclusive(|&id| id == 1) {
        for at in 1..record.len() {
            let history = &record[at.saturating_sub(10)..=at];
            let mut feature = Feature {
                counts: [0; ORDERS],
                totals: [0; ORDERS],
                mass: 0,
            };
            feature.mass = native.lookup(
                history,
                history.len() - 1,
                &mut feature.counts,
                &mut feature.totals,
            )?;
            feature.write(&mut output.writer)?;
        }
    }
    output.finish(dest, input.targets)
}

fn keep_probe_ordinals(ids: &[u16], total_targets: usize) -> Vec<usize> {
    let mut kept = Vec::new();
    let (mut frame, mut pos, mut ord) = (0usize, 0usize, 0usize);
    for &id in ids {
        if id == 0 {
            frame += 1;
            pos = 0;
            continue;
        }
        pos += 1;
        let seam = [65_536usize, 65_521usize, 8_192usize]
            .iter()
            .any(|&w| ord % w == 0 || ord % w == w - 1);
        if (frame <= 64 && (pos <= 8 || pos == 192 || id == 1))
            || seam
            || ord % 1009 == 197
            || ord + 2 >= total_targets
        {
            kept.push(ord);
        }
        ord += 1;
    }
    kept
}

fn verify_tape(
    artifact: &Path,
    train: Pin,
    compact: Pin,
    input: &Input,
    tape_path: &Path,
    all_rows: bool,
) -> Result<()> {
    let bytes = bounded(artifact, MAX_ARTIFACT_BYTES)?;
    let native = CompactCountLookup::from_bytes(&bytes, train, compact)?;
    let file = File::open(tape_path)?;
    if file.metadata()?.len() != (112 + input.targets * FEATURE_ROW_BYTES) as u64 {
        return Err("wrong UORCBF01 byte length".into());
    }
    let mut r = BufReader::with_capacity(4 << 20, file);
    let mut hdr = [0u8; 112];
    r.read_exact(&mut hdr)?;
    if &hdr[..8] != b"UORCBF01"
        || uint64(&hdr[8..16])? != input.targets as u64
        || hdr[16..48] != train
        || hdr[48..80] != compact
        || hdr[80..112] != input.pin
    {
        return Err("UORCBF01 header identity mismatch".into());
    }
    let mut hasher = Sha256::new();
    hasher.update(hdr);
    let kept = if all_rows {
        (0..input.targets).collect()
    } else {
        keep_probe_ordinals(&input.ids, input.targets)
    };
    let mut kept_pos = 0usize;
    let mut ord = 0usize;
    let mut row_buf = [0u8; FEATURE_ROW_BYTES];
    let min_probes = if all_rows || input.targets <= 1_000_000 {
        kept.len()
    } else {
        12_500
    };
    if kept.len() < min_probes {
        return Err(format!(
            "insufficient probe coverage: {} < required {}",
            kept.len(),
            min_probes
        )
        .into());
    }
    for record in input.ids.split_inclusive(|&id| id == 1) {
        for at in 1..record.len() {
            r.read_exact(&mut row_buf)?;
            hasher.update(&row_buf);
            let mut counts = [0u32; ORDERS];
            let mut totals = [0u64; ORDERS];
            for i in 0..ORDERS {
                counts[i] = uint32(&row_buf[4 * i..4 * (i + 1)])?;
                totals[i] = uint64(&row_buf[44 + 8 * i..44 + 8 * (i + 1)])?;
                if u64::from(counts[i]) > totals[i] || (i > at && (counts[i] != 0 || totals[i] != 0)) {
                    return Err(format!("invalid count/total or unused order row at ordinal {ord}").into());
                }
            }
            let mass = uint64(&row_buf[132..140])?;
            if totals[0] == 0 || !(1..=(1 << 48)).contains(&mass) {
                return Err(format!("invalid KN row range at ordinal {ord}").into());
            }
            if kept.get(kept_pos) == Some(&ord) {
                let history = &record[at.saturating_sub(10)..=at];
                let mut exp_c = [0u32; ORDERS];
                let mut exp_t = [0u64; ORDERS];
                let exp_m = native.lookup(history, history.len() - 1, &mut exp_c, &mut exp_t)?;
                if (exp_m, exp_c, exp_t) != (mass, counts, totals) {
                    return Err(format!("CUDA vs CPU UORKNC03 mismatch at ordinal {ord}").into());
                }
                kept_pos += 1;
            }
            ord += 1;
        }
    }
    if ord != input.targets || kept_pos != kept.len() || r.read(&mut [0u8; 1])? != 0 {
        return Err("incomplete or oversized UORCBF01 tape".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "rows": input.targets,
            "probes_checked": kept.len(),
            "divergent_rows": 0,
            "tape_sha256": hex::encode(hasher.finalize()),
        }))?
    );
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, artifact, train, compact, uort, uort_pin, dest, extra @ ..] => {
            let (train_p, compact_p) = (pin(train)?, pin(compact)?);
            let input = parse_input(&bounded(Path::new(uort), MAX_UORT_BYTES)?, pin(uort_pin)?)?;
            let (artifact_p, dest_p) = (Path::new(artifact), Path::new(dest));
            match (mode.as_str(), extra) {
                ("oracle", []) => finite_oracle(artifact_p, train_p, compact_p, &input, dest_p),
                ("cuda", [batch]) => {
                    let batch: usize = batch.parse()?;
                    if !(1..=MAX_BATCH).contains(&batch) {
                        return Err("batch rows must be 1..=65536".into());
                    }
                    cuda::run(artifact_p, train_p, compact_p, &input, dest_p, batch)
                }
                ("verify", []) => verify_tape(artifact_p, train_p, compact_p, &input, dest_p, false),
                ("verify-all", []) => verify_tape(artifact_p, train_p, compact_p, &input, dest_p, true),
                _ => Err("invalid cuda-v3 command".into()),
            }
        }
        _ => Err("usage: cuda-v3 (oracle|cuda|verify|verify-all) COMPACT_V3 TRAIN_SHA COMPACT_SHA UORT UORT_SHA PATH [BATCH]".into()),
    }
}

fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_metadata_and_oracle_roundtrip_synthetic_v3() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("cuda-v3-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;

        // Build minimal valid order 0..=10 UORKNC03 with empty order 2..=10 contexts.
        let train = [0x11u8; 32];
        let mut v3 = Vec::new();
        v3.extend_from_slice(b"UORKNC03");
        v3.extend_from_slice(&10u32.to_le_bytes());
        v3.extend_from_slice(&train);
        for k in 0..=10usize {
            let shared = k == 10;
            if k == 0 {
                v3.extend_from_slice(&1u32.to_le_bytes());
                v3.extend_from_slice(&1u32.to_le_bytes());
                let mut rspec = [0u64; 4];
                rspec[0] = 1;
                for w in rspec.into_iter().chain(rspec) { v3.extend_from_slice(&w.to_le_bytes()); }
                for _ in 0..4 { v3.extend_from_slice(&0u32.to_le_bytes()); }
                v3.extend_from_slice(&0u32.to_le_bytes());
                v3.extend_from_slice(&1u32.to_le_bytes());
                v3.push(0); // fol_deg = 1 - 1
                v3.extend_from_slice(&((1u16 << 4) | 0u16).to_le_bytes()); // next=1, (1,1)
                v3.extend_from_slice(&0u32.to_le_bytes());
                v3.extend_from_slice(&0u32.to_le_bytes());
            } else {
                v3.extend_from_slice(&0u32.to_le_bytes());
                v3.extend_from_slice(&0u32.to_le_bytes());
                for _ in 0..8 { v3.extend_from_slice(&0u64.to_le_bytes()); }
                for _ in 0..4 { v3.extend_from_slice(&0u32.to_le_bytes()); }
                let p_ctx = if k == 1 { 1 } else { 0 };
                if p_ctx > 0 {
                    v3.extend_from_slice(&0u32.to_le_bytes());
                    v3.extend_from_slice(&0u32.to_le_bytes());
                    v3.push(0);
                } else {
                    v3.extend_from_slice(&0u32.to_le_bytes());
                }
                v3.extend_from_slice(&0u32.to_le_bytes());
                v3.extend_from_slice(&0u32.to_le_bytes());
            }
            let _ = shared;
        }
        let art_path = dir.join("tiny.v3");
        fs::write(&art_path, &v3)?;
        let art_sha = sha(&v3);
        let mut f = File::open(&art_path)?;
        let meta = metadata(&mut f, v3.len() as u64, train)?;
        assert_eq!(meta.descriptors.len(), ORDERS * 18);

        // Build tiny valid UORT: [0, 1].
        let mut uort = vec![0u8; 64];
        uort[..4].copy_from_slice(b"UORT");
        uort[4..8].copy_from_slice(&1u32.to_le_bytes());
        uort[8..16].copy_from_slice(&2u64.to_le_bytes());
        uort[16..20].copy_from_slice(&4096u32.to_le_bytes());
        uort.extend_from_slice(&0u16.to_le_bytes());
        uort.extend_from_slice(&1u16.to_le_bytes());
        let input = parse_input(&uort, sha(&uort))?;
        let tape_path = dir.join("tiny.features");
        finite_oracle(&art_path, train, art_sha, &input, &tape_path)?;
        verify_tape(&art_path, train, art_sha, &input, &tape_path, true)?;

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
