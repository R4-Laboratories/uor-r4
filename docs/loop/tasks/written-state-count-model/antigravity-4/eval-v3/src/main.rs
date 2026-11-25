//! Parameterized UORKNC03 reserved-R S5 fit and frozen DEV-tune evaluation.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use uor_r4_discrete::kn_ceiling::PickClassRule;

#[path = "../../../../../../labs/kn-reference/instrument/src/pick_arms.rs"]
mod pick_arms;
#[cfg(feature = "cuda")]
#[path = "../../../../../../labs/kn-reference/instrument/src/pick_device.rs"]
mod pick_device;

mod fit;
mod tune;

pub(crate) type Failure = Box<dyn std::error::Error>;
pub(crate) type Result<T> = std::result::Result<T, Failure>;

pub(crate) const CLASS_NOVEL: u8 = 1;
pub(crate) const N: u32 = 10;
pub(crate) const WIDTH: usize = 11;
pub(crate) const EXPERTS: usize = 14;
pub(crate) const PASSES: u32 = 96;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const CONTEXT: usize = 192;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const CAPTURE_BATCH: usize = 65_536;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const TAPE_HEADER: usize = 112;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const COUNT_ROW: usize = 140;
pub(crate) const COMPACT_CAP_BYTES: u64 = 3_912_936_331;

pub(crate) const C_SHA: &str = "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
pub(crate) const R_SHA: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub(crate) const TRUNK_SHA: &str =
    "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub(crate) const CENSUS_SHA: &str =
    "90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467";
pub(crate) const ID_BYTES_SHA: &str =
    "befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9";
pub(crate) const TOKENIZER_SHA: &str =
    "d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89";

pub(crate) fn fail(message: impl Into<String>) -> Failure {
    message.into().into()
}

pub(crate) fn require(condition: bool, detail: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(fail(detail))
    }
}

pub(crate) fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn file_sha(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn pin(value: &str) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(value, &mut out)?;
    Ok(out)
}

pub(crate) fn word(bytes: &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes.try_into()?))
}

pub(crate) fn string(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| fail("missing string field"))
}

pub(crate) fn pinned(path: &Path, expected: &str, len: usize) -> Result<Vec<u8>> {
    let mut file = fs::File::open(path)?;
    require(
        file.metadata()?.len() == len as u64,
        "wrong pinned input length",
    )?;
    let mut bytes = Vec::with_capacity(len);
    Read::by_ref(&mut file)
        .take(len as u64 + 1)
        .read_to_end(&mut bytes)?;
    require(
        bytes.len() == len && sha(&bytes) == expected,
        "pinned input SHA-256 mismatch",
    )?;
    Ok(bytes)
}

pub(crate) fn new_file(path: &Path) -> Result<fs::File> {
    Ok(fs::File::options()
        .write(true)
        .create_new(true)
        .open(path)?)
}

pub(crate) fn writer(path: &Path) -> Result<BufWriter<fs::File>> {
    Ok(BufWriter::with_capacity(1 << 20, new_file(path)?))
}

pub(crate) fn finish_writer(mut w: BufWriter<fs::File>) -> Result<()> {
    w.flush()?;
    Ok(w.get_ref().sync_all()?)
}

pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = new_file(path)?;
    file.write_all(bytes)?;
    Ok(file.sync_all()?)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn validate_compact(compact: &Path, compact_sha: &str) -> Result<u64> {
    pin(compact_sha)?;
    let len = fs::metadata(compact)?.len();
    if len == 0 || len > COMPACT_CAP_BYTES {
        return Err(fail(format!(
            "compact artifact {} ({len} B) outside 1..={COMPACT_CAP_BYTES}",
            compact.display()
        )));
    }
    if file_sha(compact)? != compact_sha {
        return Err(fail("compact artifact SHA-256 mismatch"));
    }
    Ok(len)
}

pub(crate) fn buckets(rule: PickClassRule) -> usize {
    176 * usize::from(rule.classes())
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Tally {
    pub(crate) targets: u64,
    pub(crate) bits_q12: u128,
    pub(crate) bytes: u64,
}

impl Tally {
    pub(crate) fn add(&mut self, bits: u64, bytes: u8) {
        self.targets += 1;
        self.bits_q12 += u128::from(bits);
        self.bytes += u64::from(bytes);
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn json(self) -> Value {
        let micro =
            (self.bytes > 0).then(|| self.bits_q12 * 1_000_000 / (4096 * u128::from(self.bytes)));
        json!({
            "targets": self.targets,
            "bytes": self.bytes,
            "bits_q12": self.bits_q12.to_string(),
            "bpb_micro": micro.map(|x| x.to_string()),
        })
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).map(PathBuf::from).collect();
    match args.as_slice() {
        [mode, dir] if mode == Path::new("finite-fit") => fit::finite(dir, false),
        [mode, dir] if mode == Path::new("verify-finite-fit") => fit::finite(dir, true),
        #[cfg(feature = "cuda")]
        [mode, pick_mode, r, trunk, compact, compact_sha, tape, tape_sha, out]
            if mode == Path::new("prepare-r") =>
        {
            fit::prepare(
                string_arg(pick_mode, "pick_mode")?,
                r,
                trunk,
                compact,
                string_arg(compact_sha, "compact_sha")?,
                tape,
                string_arg(tape_sha, "tape_sha")?,
                out,
            )
        }
        [mode, dir, pick_mode, arm] if mode == Path::new("seal-r") => fit::seal(
            dir,
            string_arg(pick_mode, "pick_mode")?,
            string_arg(arm, "arm")?,
        ),
        [mode, runner] if mode == Path::new("finite-tune") => tune::finite(runner),
        #[cfg(feature = "cuda")]
        [mode, split_root, compact, compact_sha, trunk, tape, tape_sha, runner, fit_root, out]
            if mode == Path::new("score-tune") =>
        {
            tune::run(
                split_root,
                compact,
                string_arg(compact_sha, "compact_sha")?,
                trunk,
                tape,
                string_arg(tape_sha, "tape_sha")?,
                runner,
                fit_root,
                out,
            )
        }
        _ => Err(fail(concat!(
            "usage:\n",
            "  eval-v3 finite-fit DIR | verify-finite-fit DIR\n",
            "  eval-v3 prepare-r (disabled|identity) R_UORT TRUNK COMPACT COMPACT_SHA TAPE TAPE_SHA OUT_DIR\n",
            "  eval-v3 seal-r DIR (disabled|identity) (a1|a1s)\n",
            "  eval-v3 finite-tune RUNNER\n",
            "  eval-v3 score-tune SPLIT_ROOT COMPACT COMPACT_SHA TRUNK TAPE TAPE_SHA RUNNER FIT_ROOT OUT_DIR",
        ))),
    }
}

fn string_arg<'a>(p: &'a Path, label: &str) -> Result<&'a str> {
    p.to_str()
        .ok_or_else(|| fail(format!("{label} must be valid UTF-8")))
}
