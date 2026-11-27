use crate::{require, CorpusResult};
use d87_split_audit::{plan, Fate, Split, Splitter};
use m2_student_corpus_ingest::{read_shard, Coder};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

const SPLIT_SHA: &str = "5146fd2755fde3beacfdde6d80c51f31433f0fa73221e34299d48d669999bebd";
const R_SHA: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
const TRAIN_SHA: &str = "c6cba29ead4c745fb47b3f87007d982e72defc9f04102a268709ba0d34d51460";
const TRAIN_IDS_SHA: &str = "5b9c3f76c77d20d533a798a403a857454a7a0981dac7e32183d7de1bf1993a6c";
const D87_BPE_SHA: &str = "d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89";
const P4_BPE_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const SPLIT_PATH: &str = "/workspace/in/d87/";
const TRAIN_GROUPS: usize = 519_580;
const STRIDE: usize = 128;
const SHARDS: usize = 262;

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_sha(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    io::copy(&mut BufReader::new(File::open(path)?), &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn pinned(path: &Path, expected: &str) -> CorpusResult<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        sha(&bytes) == expected,
        format!("pin mismatch: {}", path.display()),
    )?;
    Ok(bytes)
}

pub struct Inputs {
    pub raw: PathBuf,
    pub split: PathBuf,
    pub d87_bpe: PathBuf,
    pub p4_bpe: PathBuf,
    pub train: PathBuf,
    pub r: PathBuf,
}

impl Inputs {
    pub fn verify(&self) -> CorpusResult<(Coder, Vec<u8>)> {
        let d87 = pinned(&self.d87_bpe, D87_BPE_SHA)?;
        let p4 = pinned(&self.p4_bpe, P4_BPE_SHA)?;
        let d87_model: Value = serde_json::from_slice(&d87)?;
        let p4_model: Value = serde_json::from_slice(&p4)?;
        for field in ["vocab", "merges"] {
            let field = format!("/model/{field}");
            require(
                d87_model.pointer(&field).is_some()
                    && d87_model.pointer(&field) == p4_model.pointer(&field),
                "D87 and P4 tokenizer vocabularies or merge ordering differ",
            )?;
        }
        let r = pinned(&self.r, R_SHA)?;
        verify_train_extract(&self.train, &r)?;
        Ok((Coder::new(&d87)?, r))
    }

    pub fn each_record(
        &self,
        threads: usize,
        on_record: &mut dyn FnMut(usize, &str) -> CorpusResult<()>,
    ) -> CorpusResult<()> {
        let summary: Value = serde_json::from_slice(&pinned(&self.split, SPLIT_SHA)?)?;
        let rows = summary["inputs"].as_array().ok_or("missing split inputs")?;
        require(rows.len() == SHARDS, "unexpected split input count")?;
        let paths = rows
            .iter()
            .map(|row| {
                let path = row["path"]
                    .as_str()
                    .and_then(|path| path.strip_prefix(SPLIT_PATH))
                    .ok_or("invalid split input path")?;
                Ok(self.raw.join(path))
            })
            .collect::<CorpusResult<Vec<_>>>()?;
        let splitter = Splitter::from_config_json(&summary["config"])?;
        let rebuilt = plan(&splitter, &paths, threads)?;
        let mut rebuilt_summary = rebuilt.summary();
        let rebuilt_rows = rebuilt_summary["inputs"]
            .as_array_mut()
            .ok_or("missing rebuilt inputs")?;
        require(
            rebuilt_rows.len() == rows.len(),
            "wrong rebuilt input count",
        )?;
        for (row, path) in rebuilt_rows.iter_mut().zip(&paths) {
            require(
                row["path"] == json!(path),
                "rebuilt split shard order differs",
            )?;
            row["path"] = json!(format!(
                "{SPLIT_PATH}{}",
                path.strip_prefix(&self.raw)?.display()
            ));
        }
        require(
            rebuilt_summary == summary
                && sha(rebuilt.manifest(Split::Train).as_bytes()) == TRAIN_IDS_SHA,
            "raw shards did not reproduce the pinned split and TRAIN membership",
        )?;
        eprintln!("verified 262 raw shard hashes and the original D87 split membership");
        let mut train = 0;
        for ((path, expected), fates) in rebuilt.shards.iter().zip(&rebuilt.fate) {
            let mut fates = fates.iter();
            let digest = read_shard(path, &mut |line| {
                if *fates.next().ok_or("new shard line")? == Fate::Keep(Split::Train) {
                    if train % STRIDE == 0 {
                        on_record(train, line).map_err(|error| error.to_string())?;
                    }
                    train += 1;
                }
                Ok(())
            })?;
            require(
                fates.next().is_none() && &digest == expected,
                "raw shard changed during extraction",
            )?;
        }
        require(train == TRAIN_GROUPS, "unexpected TRAIN group count")
    }

    pub fn binding(&self) -> Value {
        json!({
            "raw_shards": SHARDS,
            "split_summary_sha256": SPLIT_SHA,
            "train_groups": TRAIN_GROUPS,
            "train_ids_sha256": TRAIN_IDS_SHA,
            "train_uort_sha256": TRAIN_SHA,
            "original_r_uort_sha256": R_SHA,
            "d87_tokenizer_sha256": D87_BPE_SHA,
            "p4_tokenizer_sha256": P4_BPE_SHA,
            "train_group_selection": {"stride": STRIDE, "offset": 0},
            "identical_d87_p4_vocabulary_and_ordered_merges": true,
        })
    }
}

fn verify_train_extract(path: &Path, r: &[u8]) -> CorpusResult<()> {
    let mut train = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut header = [0; 64];
    train.read_exact(&mut header)?;
    let expected = 2_091_560_001;
    require(
        &header[..8] == b"UORT\x01\x00\x00\x00"
            && u64::from_le_bytes(header[8..16].try_into()?) == expected
            && header[16..20] == 4096u32.to_le_bytes()
            && header[20..].iter().all(|&byte| byte == 0),
        "unexpected full TRAIN header",
    )?;
    let mut hash = Sha256::new();
    hash.update(header);
    let mut group = 0;
    let mut current = 64;
    let mut remaining = expected;
    let mut bytes = vec![0u8; 1 << 20];
    while remaining != 0 {
        let count = remaining.min(bytes.len() as u64 / 2) as usize;
        let block = &mut bytes[..2 * count];
        train.read_exact(block)?;
        hash.update(&*block);
        for pair in block.chunks_exact(2) {
            if group % STRIDE == 0 {
                require(
                    r.get(current..current + 2) == Some(pair),
                    "pinned R differs from the full TRAIN stride selection",
                )?;
                current += 2;
            }
            group += usize::from(pair == [1, 0]);
        }
        remaining -= count as u64;
    }
    require(
        group == TRAIN_GROUPS
            && current == r.len()
            && train.read(&mut header)? == 0
            && format!("{:x}", hash.finalize()) == TRAIN_SHA,
        "full TRAIN source identity or selection coverage differs",
    )
}
