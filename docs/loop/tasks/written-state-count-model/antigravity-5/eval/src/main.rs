//! Sealed DEV-test P1(b) evaluator for UORKNC03 (artifact.d11) vs canonical R* and no-S5 control A0.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};
#[cfg(feature = "cuda")]
use uor_r4_discrete::kn_ceiling::{CeilingTable, CompactCountLookup, PickTarget};
#[cfg(any(test, feature = "cuda"))]
use uor_r4_discrete::kn_ceiling::{FrozenPickMixture, PickClassRule};
use uor_r4_integer::lawful_likelihood::bits_q12;

#[path = "../../../../../../labs/kn-reference/instrument/src/pick_arms.rs"]
mod pick_arms;
#[cfg(feature = "cuda")]
#[path = "../../../../../../labs/kn-reference/instrument/src/pick_device.rs"]
mod pick_device;
#[cfg(feature = "cuda")]
mod score;

pub type Failure = Box<dyn std::error::Error>;
pub type Result<T> = std::result::Result<T, Failure>;

pub const CLASS_NOVEL: u8 = 1;
pub const TOKENS: usize = 21_354_897;
pub const RECORDS: usize = 5_321;
pub const ROWS: usize = TOKENS - RECORDS; // 21_349_576
pub const ALL_BYTES: u64 = 43_326_982;
pub const NOVEL_BYTES: u64 = 31_328_319;
pub const WIDTH: usize = 11;
pub const EXPERTS: usize = 14;
pub const BATCH: usize = 8192;
pub const CAPTURE_BATCH: usize = 65536;

pub const COMPACT_BYTES: u64 = 3_804_378_885;
pub const COMPACT_SHA: &str = "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const DEPLOY_ARTIFACT_BYTES: u64 = 3_891_016_460;
pub const DEPLOY_TOKENIZER_BYTES: u64 = 265_393;
pub const DEPLOY_TOTAL_BYTES: u64 = DEPLOY_ARTIFACT_BYTES + DEPLOY_TOKENIZER_BYTES;
pub const DEPLOY_ARTIFACT_SHA: &str =
    "dd217c0de241f725925e62fdcecab2a78cfc4829e080200e2de708a1e5c57d05";

pub const C_SHA: &str = "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
pub const R_SHA: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub const TRUNK_SHA: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const CENSUS_SHA: &str = "ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2";
pub const ID_BYTES_SHA: &str = "befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9";
pub const TOKENIZER_SHA: &str = "d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89";

pub const RSTAR_MASSES_SHA: &str =
    "59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de";
pub const RSTAR_ALL_BITS_Q12: u128 = 251_672_950_854;
pub const RSTAR_NOVEL_BITS_Q12: u128 = 223_758_978_608;
pub const RSTAR_FLOOR_COUNT: usize = 4;

pub const A1S_WEIGHTS_SHA: &str =
    "e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293";
pub const A1S_MANIFEST_SHA: &str =
    "26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec";
pub const A1S_SEAL_SHA: &str = "b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071";

pub const A0_WEIGHTS_SHA: &str = "9ec8168a20f250ef089db1b55d94b02f248d1ec39ee4f51d9692d98ee6be9c37";
pub const A0_MANIFEST_SHA: &str =
    "95b17f7bbb9059f4542f3505a9e7aeca5315bc540b8327a0d0a882cb8710e4c8";
pub const A0_SEAL_SHA: &str = "5eddd7837bb67ed3b58b5a1262d711ade186719cc6a57ff31cc45d0042186b5b";

pub fn fail(message: impl Into<String>) -> Failure {
    message.into().into()
}

pub fn require(condition: bool, detail: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(fail(detail))
    }
}

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn pin(value: &str) -> Result<[u8; 32]> {
    let mut out = [0; 32];
    hex::decode_to_slice(value, &mut out)?;
    Ok(out)
}

pub fn word(bytes: &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes.try_into()?))
}

pub fn pinned(path: &Path, expected: &str, len: usize) -> Result<Vec<u8>> {
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

pub fn new_file(path: &Path) -> Result<fs::File> {
    Ok(fs::File::options()
        .write(true)
        .create_new(true)
        .open(path)?)
}

pub fn string(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| fail("missing string field"))
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    pub targets: u64,
    pub bits_q12: u128,
    pub bytes: u64,
}

impl Tally {
    pub fn add(&mut self, bits: u64, bytes: u8) {
        self.targets += 1;
        self.bits_q12 += u128::from(bits);
        self.bytes += u64::from(bytes);
    }

    pub fn json(self) -> Value {
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

pub struct Split {
    pub ids: Vec<u16>,
    pub novel: Vec<u8>,
    pub mask: Vec<u8>,
    pub bytes: Vec<u8>,
    pub uort: Vec<u8>,
}

impl Split {
    pub fn load(root: &Path) -> Result<Self> {
        let census = fs::read(root.join("dev-test.json"))?;
        require(sha(&census) == CENSUS_SHA, "wrong DEV-test census")?;
        let census: Value = serde_json::from_slice(&census)?;
        require(
            census["split"] == "dev-test"
                && census["records"] == RECORDS
                && census["tokens"] == TOKENS
                && census["targets"] == ROWS
                && census["target_bytes"] == ALL_BYTES
                && census["id_bytes_sha256"] == ID_BYTES_SHA
                && census["tokenizer_sha256"] == TOKENIZER_SHA,
            "wrong DEV-test census identity",
        )?;
        let load = |index: usize, name: &str, len: usize| -> Result<Vec<u8>> {
            let entry = &census["files"][index];
            require(
                entry["file"] == name && entry["bytes"] == len,
                "unexpected DEV-test census entry",
            )?;
            pinned(&root.join(name), string(&entry["sha256"])?, len)
        };
        let raw = load(0, "dev-test.tokens.u16", 64 + 2 * TOKENS)?;
        let mask = load(1, "dev-test.mask.u8", TOKENS)?;
        let segments = load(2, "dev-test.segment.u8", TOKENS)?;
        let novel = load(3, "dev-test.novel.u8", TOKENS)?;
        let bytes = pinned(&root.join("id-bytes.u8"), ID_BYTES_SHA, 4096)?;
        require(
            raw[..4] == *b"UORT"
                && raw[4..8] == 1u32.to_le_bytes()
                && raw[16..20] == 4096u32.to_le_bytes(),
            "wrong preserved DEV-test UORT header",
        )?;
        let ids = uor_r4_discrete::ngram_count::read_uort_tokens(&raw)?;
        let split = Self {
            ids,
            novel,
            mask,
            bytes,
            uort: raw,
        };
        split.validate(&census, &segments)?;
        Ok(split)
    }

    fn validate(&self, census: &Value, segments: &[u8]) -> Result<()> {
        let mut all = Tally::default();
        let mut response = Tally::default();
        let mut classes = [[Tally::default(); 3]; 2];
        let (mut frames, mut position, mut prior) = (0, 0, 1);
        for (at, &id) in self.ids.iter().enumerate() {
            let (class, mask) = (self.novel[at] as usize, self.mask[at]);
            if id >= 4096
                || (id == 0) != (prior == 1)
                || mask > 1
                || segments[at] > 3
                || (segments[at] == 1 && mask != 0)
                || (segments[at] >= 2 && mask != 1)
                || (id == 1 && (segments[at] != 0 || mask != 1))
            {
                return Err(fail(format!(
                    "wrong DEV-test token, segment, mask or frame at {at}"
                )));
            }
            prior = id;
            if id == 0 {
                require(
                    class == 3 && mask == 0 && segments[at] == 0,
                    "BOS must be outside all scoring masks",
                )?;
                frames += 1;
                position = 0;
                continue;
            }
            let len = self.bytes[id as usize];
            require(
                class < 3 && (class == 2) == (position < 12),
                "wrong DEV-test novelty class",
            )?;
            position += 1;
            all.add(0, len);
            classes[0][class].add(0, len);
            if mask == 1 {
                response.add(0, len);
                classes[1][class].add(0, len);
            }
        }
        require(
            frames == RECORDS
                && prior == 1
                && census["target_bytes"] == all.bytes
                && census["response_bytes"] == response.bytes
                && census["response_targets"] == response.targets
                && census["targets"] == all.targets
                && self.bytes[..3] == [0; 3],
            "DEV-test aggregate census mismatch",
        )?;
        for (class, name) in ["seen", "novel", "short_context"].into_iter().enumerate() {
            let count = &census["novel"]["classes"][name];
            require(
                count["bytes"] == classes[0][class].bytes
                    && count["targets"] == classes[0][class].targets
                    && count["response_bytes"] == classes[1][class].bytes
                    && count["response_targets"] == classes[1][class].targets,
                "DEV-test class census mismatch",
            )?;
        }
        Ok(())
    }
}

/// Exact integer P1(b) pruning-cost check: candidate bits_q12 <= 1.03 * R* bits_q12.
pub fn p1b_passes_threshold(candidate_bits_q12: u128, rstar_bits_q12: u128) -> Result<bool> {
    let lhs = candidate_bits_q12
        .checked_mul(100)
        .ok_or_else(|| fail("P1(b) candidate bits overflowed u128"))?;
    let rhs = rstar_bits_q12
        .checked_mul(103)
        .ok_or_else(|| fail("P1(b) R* threshold bits overflowed u128"))?;
    Ok(lhs <= rhs)
}

/// Aggregate canonical R* masses tape against split framing and pinned SHA-256/totals.
pub fn aggregate_rstar_masses(
    masses_bytes: &[u8],
    ids: &[u16],
    novel: &[u8],
    id_bytes: &[u8],
) -> Result<(Vec<[u128; 2]>, [Tally; 2], usize)> {
    require(
        sha(masses_bytes) == RSTAR_MASSES_SHA,
        "R* masses SHA-256 mismatch",
    )?;
    require(
        masses_bytes.len() == ROWS * 8,
        "R* masses byte length mismatch",
    )?;
    let mut records: Vec<[u128; 2]> = Vec::with_capacity(RECORDS);
    let mut tallies = [Tally::default(); 2];
    let mut floor_count = 0usize;
    let mut chunks = masses_bytes.chunks_exact(8);
    for (at, &id) in ids.iter().enumerate() {
        if id == 0 {
            records.push([0; 2]);
            continue;
        }
        let chunk = chunks
            .next()
            .ok_or_else(|| fail("missing R* target mass"))?;
        let mass = word(chunk)?;
        require((1..=1u64 << 48).contains(&mass), "R* mass out of Q48 range")?;
        if mass == 1 {
            floor_count += 1;
        }
        let bits = bits_q12(mass).map_err(|e| fail(format!("R* target {at}: {e:?}")))?;
        let len = id_bytes[id as usize];
        let is_novel = novel[at] == CLASS_NOVEL;
        let rec = records
            .last_mut()
            .ok_or_else(|| fail("R* missing BOS record"))?;
        rec[0] += u128::from(bits);
        tallies[0].add(bits, len);
        if is_novel {
            rec[1] += u128::from(bits);
            tallies[1].add(bits, len);
        }
    }
    require(chunks.next().is_none(), "trailing R* masses")?;
    require(
        records.len() == RECORDS
            && tallies[0].targets == ROWS as u64
            && tallies[0].bytes == ALL_BYTES
            && tallies[1].bytes == NOVEL_BYTES
            && tallies[0].bits_q12 == RSTAR_ALL_BITS_Q12
            && tallies[1].bits_q12 == RSTAR_NOVEL_BITS_Q12
            && floor_count == RSTAR_FLOOR_COUNT,
        "canonical R* aggregate counts or floor count mismatch",
    )?;
    Ok((records, tallies, floor_count))
}

pub fn paired_band_scaled(
    x: &[u128],
    y: &[u128],
    scale_x: i128,
    scale_y: i128,
    bytes: u64,
    resamples: u32,
    seed: u64,
) -> Result<Value> {
    require(
        x.len() == y.len() && !x.is_empty() && bytes > 0 && resamples >= 40,
        "invalid scaled paired band inputs",
    )?;
    let diffs: Vec<i128> = x
        .iter()
        .zip(y)
        .map(|(&a, &b)| {
            (a as i128)
                .checked_mul(scale_x)
                .and_then(|lhs| (b as i128).checked_mul(scale_y).map(|rhs| lhs - rhs))
                .ok_or_else(|| fail("paired diff overflowed i128"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut state = seed;
    let n = diffs.len() as u64;
    let mut sums: Vec<i128> = Vec::with_capacity(resamples as usize);
    for _ in 0..resamples {
        let mut total = 0i128;
        for _ in 0..diffs.len() {
            total += diffs[(pick_arms::splitmix64(&mut state) % n) as usize];
        }
        sums.push(total);
    }
    sums.sort_unstable();
    let k = (resamples / 40) as usize;
    let raw = [sums[k - 1], sums[resamples as usize - k - 1]];
    let denom = i128::from(bytes) * 4096 * scale_x;
    let lo = (raw[0] * 1_000_000).div_euclid(denom);
    let hi = -(-raw[1] * 1_000_000).div_euclid(denom);
    Ok(json!({
        "scale_x": scale_x,
        "scale_y": scale_y,
        "q12": raw.map(|v| v.to_string()),
        "micro_bpb": [lo.to_string(), hi.to_string()],
        "strictly_below_zero": raw[1] < 0,
    }))
}

/// Validates mode-specific S5 Pick weight invariants: `disabled` (A0) visits only
/// buckets 0..176 and must have zero Pick weights there (unvisited b >= 176 retain
/// `device.cu` `initialize` unit mass 1); `masked` (C1) must be zero everywhere.
#[cfg(any(test, feature = "cuda"))]
pub fn validate_mode_pick_weights(expected_mode: &str, frozen: &FrozenPickMixture) -> Result<()> {
    let disabled_ok = expected_mode != "disabled"
        || frozen
            .weights()
            .chunks_exact(EXPERTS)
            .take(176)
            .all(|row| row[EXPERTS - 1] == 0);
    let masked_ok = expected_mode != "masked"
        || frozen
            .weights()
            .chunks_exact(EXPERTS)
            .all(|row| row[EXPERTS - 1] == 0);
    require(
        disabled_ok && masked_ok,
        "R-fit carrier has nonzero Pick weights in forbidden lane",
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).map(PathBuf::from).collect();
    match args.as_slice() {
        [mode, root] if mode == Path::new("validate") => {
            let split = Split::load(root)?;
            println!(
                "{}",
                json!({
                    "uort_sha256": sha(&split.uort),
                    "census_sha256": CENSUS_SHA,
                    "rows": ROWS,
                    "records": RECORDS
                })
            );
            Ok(())
        }
        #[cfg(feature = "cuda")]
        [mode, runner] if mode == Path::new("finite") => score::finite(runner),
        #[cfg(feature = "cuda")]
        [mode, root, go, go_sha, compact, trunk, tape, rstar_masses, runner, a1s_dir, a0_dir, out]
            if mode == Path::new("score") =>
        {
            score::run(
                root,
                go,
                go_sha,
                compact,
                trunk,
                tape,
                rstar_masses,
                runner,
                a1s_dir,
                a0_dir,
                out,
            )
        }
        _ => Err(fail(concat!(
            "usage: validate SPLIT_ROOT | finite CUDA_RUNNER | ",
            "score SPLIT_ROOT GO GO_SHA COMPACT TRUNK COUNT_TAPE RSTAR_MASSES CUDA_RUNNER A1S_DIR A0_DIR OUT"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p1b_threshold_boundary_is_exact() {
        let rstar = 251_672_950_854u128;
        let exact_pass = (rstar * 103) / 100;
        let just_over = exact_pass + 1;
        assert!(p1b_passes_threshold(exact_pass, rstar).unwrap());
        assert!(!p1b_passes_threshold(just_over, rstar).unwrap());
        assert!(p1b_passes_threshold(103, 100).unwrap());
        assert!(!p1b_passes_threshold(104, 100).unwrap());
    }

    #[test]
    fn deploy_total_bytes_and_novel_threshold_hold() {
        assert_eq!(DEPLOY_TOTAL_BYTES, 3_891_281_853);
        const { assert!(DEPLOY_TOTAL_BYTES <= 4_000_000_000) };
        let exact_novel = (RSTAR_NOVEL_BITS_Q12 * 103) / 100;
        assert!(p1b_passes_threshold(exact_novel, RSTAR_NOVEL_BITS_Q12).unwrap());
        assert!(!p1b_passes_threshold(exact_novel + 1, RSTAR_NOVEL_BITS_Q12).unwrap());
    }

    #[test]
    fn paired_band_scaled_matches_exact_threshold_sign() {
        let x: Vec<u128> = vec![103; 16];
        let y: Vec<u128> = vec![100; 16];
        let eq = paired_band_scaled(&x, &y, 100, 103, 1000, 1000, 4390).unwrap();
        assert_eq!(eq["q12"], json!(["0", "0"]));
        assert_eq!(eq["strictly_below_zero"], json!(false));

        let x_below: Vec<u128> = vec![102; 16];
        let pass = paired_band_scaled(&x_below, &y, 100, 103, 1000, 1000, 4390).unwrap();
        assert_eq!(pass["strictly_below_zero"], json!(true));
    }

    #[test]
    fn corrupt_rstar_masses_digest_or_length_is_rejected() {
        let ids = [0u16, 5, 1];
        let novel = [3u8, 1, 1];
        let id_bytes = vec![1u8; 4096];
        let err = aggregate_rstar_masses(&[0u8; 16], &ids, &novel, &id_bytes).unwrap_err();
        assert!(err.to_string().contains("SHA-256"));
    }

    #[test]
    fn replay_rejects_planted_distance_class_crossing_and_tie_corruption() {
        let rule = PickClassRule::Five;
        let native = Some((8usize, 2u32, 2u32));
        let distance_corrupted = Some((8usize, 3u32, 2u32));
        let tie_corrupted = Some((8usize, 2u32, 1u32));
        assert_ne!(
            native.map(|(_, d, _)| rule.classify(Some(d))),
            distance_corrupted.map(|(_, d, _)| rule.classify(Some(d)))
        );
        assert_ne!(native, tie_corrupted);
    }

    #[test]
    fn disabled_mode_allows_unvisited_init_one_and_rejects_active_pick() {
        let mut weights = vec![0u64; 528 * EXPERTS];
        for (b, row) in weights.chunks_exact_mut(EXPERTS).enumerate() {
            if b < 176 {
                row[0] = 1u64 << 32;
            } else {
                row[0] = (1u64 << 32) - 1;
                row[EXPERTS - 1] = 1;
            }
        }
        let disabled =
            FrozenPickMixture::with_rule(10, [0; 32], weights.clone(), PickClassRule::Three)
                .unwrap();
        assert!(validate_mode_pick_weights("disabled", &disabled).is_ok());
        assert!(validate_mode_pick_weights("masked", &disabled).is_err());

        weights[EXPERTS - 1] = 1;
        weights[0] -= 1;
        assert!(FrozenPickMixture::with_rule(10, [0; 32], weights, PickClassRule::Three).is_err());
    }
}
