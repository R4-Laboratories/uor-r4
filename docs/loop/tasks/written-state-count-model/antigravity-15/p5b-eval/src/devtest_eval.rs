#![forbid(unsafe_code)]

use crate::cache_eval::compute_causal_cache_features;
use crate::s3_decode::{
    stream_golden_and_skip_keys, GoldenShellStateTable, S2SkipTable, S3_Q48_ONE, S3_VOCAB,
};
use crate::{fail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};
use uor_r4_count_runtime::p3_state::{
    brier_mix_q48, integer_brier_target_q32, project_simplex_q32, BrierS6Weights, P3_Q32_ONE,
};
use uor_r4_discrete::kn_ceiling::{FrozenPickMixture, PickClassRule, PickTarget};
use uor_r4_discrete::ngram_count::read_uort_tokens;
use uor_r4_integer::lawful_likelihood::bits_q12;

#[cfg(feature = "cuda")]
use crate::cuda::P4BrierGpuEngine;
#[cfg(feature = "cuda")]
use crate::pick_device::{windows, DeviceTape, PickDevice, READ_BATCH};

pub const C_CORPUS_SHA256: &str =
    "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
pub const GOLDEN_SHELL_SHA256: &str =
    "e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c";
pub const S2_SKIP_SHA256: &str = "85307b88f6bbc43d57fe8fe896789f9c8ea97e8e25bc24b074a3932e9332c777";
pub const COMPACT_V3_SHA256: &str =
    "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const TRUNK_SHA256: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const R_UORT_SHA256: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";

pub const DEV_TEST_CENSUS_SHA256: &str =
    "ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2";
pub const DEV_TEST_UORT_SHA256: &str =
    "023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b";
pub const DEV_TEST_TAPE_SHA256: &str =
    "f0e3432d3ecede6b54ae1d4782dd6a9ea87da280d2a8d04f456f88b2fc6a9951";
pub const RSTAR_MASSES_SHA256: &str =
    "59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de";

pub const B4_SKIP_STATIC_SHA256: &str =
    "e0d5a65774d38279b84a5ccc56aa2966098f0eb6e7706e57dbc2be560518b79b";
pub const B4_SKIP_ERRWRITE_SHA256: &str =
    "a0d50fdcf3971d793de42159f4271975ec43cc8e2c0cb3bec743ef4f187c71fb";
pub const B4_CACHE_STATIC_SHA256: &str =
    "fe7f5d06a9717aaee5eb3eeac31b35eb8ce319072810a83ded4e27701333d285";
pub const B4_FULL_ERRWRITE_SHA256: &str =
    "5783ba3b8b03cb4e9c35ed335b89c8ff36797175e9e1b8d9db40ddfd64f46a2d";

pub const RSTAR_ALL_BITS_Q12: u128 = 251_672_950_854;
pub const RSTAR_NOVEL_BITS_Q12: u128 = 223_758_978_608;
pub const RSTAR_DEVTEST_BPB_MICRO: u128 = 1_418_136;
pub const P1_A1S_DEVTEST_ALL_Q12: u128 = 246_099_676_775;
pub const P1_A1S_DEVTEST_NOVEL_Q12: u128 = 215_795_693_587;

pub const R_TARGETS: usize = 15_947_584;
pub const DEV_TEST_RECORDS: usize = 5_321;
pub const DEV_TEST_TOKENS: usize = 21_354_897;
pub const DEV_TEST_TARGETS: usize = 21_349_576;
pub const P1_EXPERTS: usize = 14;
pub const S6_BUCKETS: usize = 880;
pub const S6_CACHE_BUCKETS: usize = S6_BUCKETS * 3;
pub const KN_BASE_MASS: u64 = 1u64 << 36;
#[cfg(feature = "cuda")]
const CAPTURE_BATCH: usize = 65_536;
const E18: usize = 18;
const E22: usize = 22;
const CUDA_CHUNK: usize = 65_536;
const MACRO_BATCH: usize = 524_288;
const WARM_EXTRA_SHARE_Q32: u64 = P3_Q32_ONE >> 7;
const ERR_LR_SHIFT: u32 = 7;

fn decode_hex32(hex_str: &str) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut out)?;
    Ok(out)
}

#[derive(Default, Clone, Copy)]
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

    pub fn micro_bpb(self) -> Option<u128> {
        (self.bytes > 0).then(|| self.bits_q12 * 1_000_000 / (4096 * u128::from(self.bytes)))
    }

    pub fn json(self) -> Value {
        json!({
            "targets": self.targets,
            "bytes": self.bytes,
            "bits_q12": self.bits_q12.to_string(),
            "bpb_micro": self.micro_bpb().map(|x| x.to_string()),
        })
    }
}

pub struct DevTestData {
    pub ids: Vec<u16>,
    pub novel: Vec<u8>,
    pub mask: Vec<u8>,
    pub id_bytes: Vec<u8>,
}

fn load_pinned_entry(
    dir: &Path,
    entry: &Value,
    expected_name: &str,
    expected_len: usize,
) -> Result<Vec<u8>> {
    let fname = entry["file"]
        .as_str()
        .ok_or_else(|| fail("missing census file"))?;
    let fbytes = entry["bytes"]
        .as_u64()
        .ok_or_else(|| fail("missing census bytes"))?;
    let fsha = entry["sha256"]
        .as_str()
        .ok_or_else(|| fail("missing census sha256"))?;
    if fname != expected_name || fbytes != expected_len as u64 {
        return Err(fail(format!("unexpected census entry {fname}")));
    }
    let raw = std::fs::read(dir.join(expected_name))?;
    if raw.len() != expected_len || hex::encode(Sha256::digest(&raw)) != fsha {
        return Err(fail(format!("{expected_name} length/SHA mismatch")));
    }
    Ok(raw)
}

impl DevTestData {
    pub fn load_checked(dir: &Path) -> Result<Self> {
        let census_bytes = std::fs::read(dir.join("dev-test.json"))?;
        let csha = hex::encode(Sha256::digest(&census_bytes));
        if csha != DEV_TEST_CENSUS_SHA256 {
            return Err(fail(format!(
                "DEV-test census SHA mismatch: {csha} != {DEV_TEST_CENSUS_SHA256}"
            )));
        }
        let census: Value = serde_json::from_slice(&census_bytes)?;
        if census["split"] != "dev-test"
            || census["records"] != DEV_TEST_RECORDS
            || census["tokens"] != DEV_TEST_TOKENS as u64
            || census["targets"] != DEV_TEST_TARGETS as u64
        {
            return Err(fail("DEV-test census identity mismatch"));
        }
        let uort_raw = load_pinned_entry(
            dir,
            &census["files"][0],
            "dev-test.tokens.u16",
            64 + 2 * DEV_TEST_TOKENS,
        )?;
        if hex::encode(Sha256::digest(&uort_raw)) != DEV_TEST_UORT_SHA256 {
            return Err(fail("DEV-test UORT SHA-256 mismatch"));
        }
        let mask = load_pinned_entry(
            dir,
            &census["files"][1],
            "dev-test.mask.u8",
            DEV_TEST_TOKENS,
        )?;
        let _seg = load_pinned_entry(
            dir,
            &census["files"][2],
            "dev-test.segment.u8",
            DEV_TEST_TOKENS,
        )?;
        let novel = load_pinned_entry(
            dir,
            &census["files"][3],
            "dev-test.novel.u8",
            DEV_TEST_TOKENS,
        )?;
        let id_bytes = std::fs::read(dir.join("id-bytes.u8"))?;
        let id_sha = census["id_bytes_sha256"]
            .as_str()
            .ok_or_else(|| fail("missing id_bytes_sha256"))?;
        if id_bytes.len() != S3_VOCAB || hex::encode(Sha256::digest(&id_bytes)) != id_sha {
            return Err(fail("id-bytes.u8 mismatch"));
        }
        let ids = read_uort_tokens(&uort_raw)?;
        if ids.len() != DEV_TEST_TOKENS {
            return Err(fail("DEV-test token len mismatch"));
        }
        Ok(Self {
            ids,
            novel,
            mask,
            id_bytes,
        })
    }
}

pub fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn paired_record_ci(x: &[u128], y: &[u128], resamples: u32, seed: u64) -> Result<[i128; 2]> {
    let n = x.len();
    if n == 0 || x.len() != y.len() || resamples < 40 {
        return Err(fail("invalid paired_record_ci inputs"));
    }
    let diffs: Vec<i128> = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| a as i128 - b as i128)
        .collect();
    let mut rng = seed;
    let mut sums = Vec::with_capacity(resamples as usize);
    for _ in 0..resamples {
        let mut s = 0i128;
        for _ in 0..n {
            s += diffs[(splitmix64_step(&mut rng) % (n as u64)) as usize];
        }
        sums.push(s);
    }
    sums.sort_unstable();
    let k = (resamples / 40) as usize;
    Ok([sums[k - 1], sums[resamples as usize - k - 1]])
}

pub type LoadedP1Tape = (Vec<u16>, Vec<u64>, Vec<u64>, Vec<u64>);

pub fn load_p1_tape_and_verify_a1s(
    tape_path: &Path,
    pick_rows_path: &Path,
    frozen: &FrozenPickMixture,
    expected_uort_sha: [u8; 32],
    rows: usize,
) -> Result<LoadedP1Tape> {
    let tape_file = File::open(tape_path)?;
    if tape_file.metadata()?.len() != (112 + 140 * rows) as u64 {
        return Err(fail("UORCBF01 file length mismatch"));
    }
    let mut tape = BufReader::with_capacity(1 << 20, tape_file);
    let mut hdr = [0u8; 112];
    tape.read_exact(&mut hdr)?;
    if &hdr[..8] != b"UORCBF01"
        || u64::from_le_bytes(hdr[8..16].try_into()?) != rows as u64
        || hdr[16..48] != decode_hex32(C_CORPUS_SHA256)?
        || hdr[48..80] != decode_hex32(COMPACT_V3_SHA256)?
        || hdr[80..112] != expected_uort_sha
    {
        return Err(fail("UORCBF01 header/SHA binding mismatch"));
    }

    let pk_file = File::open(pick_rows_path)?;
    if pk_file.metadata()?.len() != (80 + 16 * rows) as u64 {
        return Err(fail("R37PK001 length mismatch"));
    }
    let mut pk = BufReader::with_capacity(1 << 20, pk_file);
    let mut pk_hdr = [0u8; 80];
    pk.read_exact(&mut pk_hdr)?;
    if &pk_hdr[..8] != b"R37PK001"
        || u64::from_le_bytes(pk_hdr[8..16].try_into()?) != rows as u64
        || pk_hdr[16..48] != expected_uort_sha
        || pk_hdr[48..80] != decode_hex32(TRUNK_SHA256)?
    {
        return Err(fail("R37PK001 header/SHA mismatch"));
    }

    let mut buckets = Vec::with_capacity(rows);
    let mut p1_experts = Vec::with_capacity(rows * P1_EXPERTS);
    let mut a1s_masses = Vec::with_capacity(rows);
    let mut crow = [0u8; 140];
    let mut prow = [0u8; 16];
    let mut counts = [0u32; 11];
    let mut totals = [0u64; 11];
    for r in 0..rows {
        tape.read_exact(&mut crow)?;
        pk.read_exact(&mut prow)?;
        let mut h = 0usize;
        for i in 0..11 {
            counts[i] = u32::from_le_bytes(crow[4 * i..4 * (i + 1)].try_into()?);
            totals[i] = u64::from_le_bytes(crow[44 + 8 * i..44 + 8 * (i + 1)].try_into()?);
            if totals[i] > 0 {
                h = i;
            }
        }
        let kn = u64::from_le_bytes(crow[132..].try_into()?);
        let pred = u32::from_le_bytes(prow[0..4].try_into()?);
        let dist = u32::from_le_bytes(prow[4..8].try_into()?);
        let hit = u32::from_le_bytes(prow[12..16].try_into()?) != 0;
        let pick = PickTarget {
            class: PickClassRule::Five.classify(if pred == u32::MAX { None } else { Some(dist) }),
            hit,
        };
        let log2 = 63 - totals[h].leading_zeros() as usize;
        let plain_b = h * 16 + log2.min(15);
        let bkt = u16::try_from(usize::from(pick.class) * 176 + plain_b)?;
        buckets.push(bkt);

        let base_off = p1_experts.len();
        p1_experts.push(kn);
        for i in 0..11 {
            let m = if totals[i] == 0 {
                0
            } else {
                ((u128::from(counts[i]) << 48) / u128::from(totals[i])) as u64
            };
            p1_experts.push(m);
        }
        p1_experts.push(KN_BASE_MASS);
        p1_experts.push(if pick.hit { S3_Q48_ONE } else { 0 });

        let (fb, fm) = frozen.predict_mass(kn, &counts, &totals, pick)?;
        if r < 256 || r % 16384 == 0 {
            let start = usize::from(fb) * P1_EXPERTS;
            let bm = brier_mix_q48(
                &frozen.weights()[start..start + P1_EXPERTS],
                &p1_experts[base_off..base_off + P1_EXPERTS],
            )?;
            if fb != bkt || bm != fm {
                return Err(fail(format!(
                    "A1s/P1 14-expert decomposition diverged at row {r}"
                )));
            }
        }
        a1s_masses.push(fm);
    }
    Ok((buckets, p1_experts, a1s_masses, frozen.weights().to_vec()))
}

pub type EpochEvalOut = (Vec<u64>, Vec<i128>, Vec<u64>, u64);
pub type EmEpochOut = (Vec<u64>, Vec<u128>, Vec<u64>, u64);

#[allow(clippy::too_many_arguments)]
fn gpu_or_cpu_eval_epoch(
    #[cfg(feature = "cuda")] gpu: &P4BrierGpuEngine,
    buckets_count: usize,
    experts: usize,
    row_buckets: &[u16],
    expert_masses: &[u64],
    weights_q32: &[u64],
    batch_size: usize,
) -> Result<EpochEvalOut> {
    let rows = row_buckets.len();
    let mut mixed_all = vec![0u64; rows];
    let mut grads = vec![0i128; buckets_count * experts];
    let mut touched = vec![0u64; buckets_count];
    let mut brier_sum = 0u128;

    let mut pos = 0usize;
    while pos < rows {
        let end = (pos + batch_size).min(rows);
        let len = end - pos;
        let zeros = vec![0u64; len];
        #[cfg(feature = "cuda")]
        let (cmix, cbri, cgrad, ccnt) = gpu.eval_and_accum_batch(
            buckets_count,
            experts,
            &row_buckets[pos..end],
            &expert_masses[pos * experts..end * experts],
            &zeros,
            weights_q32,
        )?;
        #[cfg(not(feature = "cuda"))]
        let (cmix, cbri, cgrad, ccnt) = {
            let mut m = Vec::with_capacity(len);
            let mut b = Vec::with_capacity(len);
            let mut g = vec![0u64; buckets_count * experts];
            let mut c = vec![0u64; buckets_count];
            for (idx, &z) in zeros.iter().enumerate().take(len) {
                let r = pos + idx;
                let bk = usize::from(row_buckets[r]);
                let w = &weights_q32[bk * experts..(bk + 1) * experts];
                let row_m = &expert_masses[r * experts..(r + 1) * experts];
                let mx = brier_mix_q48(w, row_m)?;
                m.push(mx);
                b.push(integer_brier_target_q32(mx, z));
                c[bk] += 1;
                let p_q16 = (mx >> 32).min(1u64 << 16);
                let err = (1u64 << 16) - p_q16;
                for e in 0..experts {
                    let mq = row_m[e] >> 32;
                    g[bk * experts + e] += err * mq;
                }
            }
            (m, b, g, c)
        };
        mixed_all[pos..end].copy_from_slice(&cmix);
        for &bv in &cbri {
            brier_sum += u128::from(bv);
        }
        for i in 0..touched.len() {
            touched[i] += ccnt[i];
        }
        for i in 0..grads.len() {
            grads[i] += i128::from(cgrad[i]);
        }
        pos = end;
    }
    let mean_brier = if rows == 0 {
        0
    } else {
        (brier_sum / (rows as u128)) as u64
    };
    Ok((mixed_all, grads, touched, mean_brier))
}

#[allow(clippy::too_many_arguments)]
fn gpu_or_cpu_em_epoch(
    #[cfg(feature = "cuda")] gpu: &P4BrierGpuEngine,
    buckets_count: usize,
    experts: usize,
    row_buckets: &[u16],
    expert_masses: &[u64],
    weights_q32: &[u64],
    batch_size: usize,
) -> Result<EmEpochOut> {
    let rows = row_buckets.len();
    let mut mixed_all = vec![0u64; rows];
    let mut resp_sum = vec![0u128; buckets_count * experts];
    let mut touched = vec![0u64; buckets_count];
    let mut brier_sum = 0u128;

    let mut pos = 0usize;
    while pos < rows {
        let end = (pos + batch_size).min(rows);
        #[cfg(feature = "cuda")]
        let (cmix, cresp, ccnt) = gpu.em_eval_and_accum_batch(
            buckets_count,
            experts,
            &row_buckets[pos..end],
            &expert_masses[pos * experts..end * experts],
            weights_q32,
        )?;
        #[cfg(not(feature = "cuda"))]
        let (cmix, cresp, ccnt) = {
            let len = end - pos;
            let mut m = Vec::with_capacity(len);
            let mut rp = vec![0u64; buckets_count * experts];
            let mut c = vec![0u64; buckets_count];
            for idx in 0..len {
                let r = pos + idx;
                let bk = usize::from(row_buckets[r]);
                let w = &weights_q32[bk * experts..(bk + 1) * experts];
                let row_m = &expert_masses[r * experts..(r + 1) * experts];
                let mx = brier_mix_q48(w, row_m)?;
                m.push(mx);
                c[bk] += 1;
                for e in 0..experts {
                    let wm_q48 = ((u128::from(w[e]) * u128::from(row_m[e])) >> 32) as u64;
                    let rq = ((u128::from(wm_q48) << 32) / u128::from(mx)) as u64;
                    rp[bk * experts + e] += rq;
                }
            }
            (m, rp, c)
        };
        for &mv in &cmix {
            brier_sum += u128::from(integer_brier_target_q32(mv, 0));
        }
        mixed_all[pos..end].copy_from_slice(&cmix);
        for i in 0..touched.len() {
            touched[i] += ccnt[i];
        }
        for i in 0..resp_sum.len() {
            resp_sum[i] += u128::from(cresp[i]);
        }
        pos = end;
    }
    let mean_brier = if rows == 0 {
        0
    } else {
        (brier_sum / (rows as u128)) as u64
    };
    Ok((mixed_all, resp_sum, touched, mean_brier))
}

pub fn init_a1s_warmstart_weights_tiled(
    a1s_weights: &[u64],
    base_buckets: usize,
    tiles: usize,
    experts: usize,
    extra_share_q32: u64,
    train_sha: [u8; 32],
) -> Result<BrierS6Weights> {
    if a1s_weights.len() != base_buckets * P1_EXPERTS || experts < P1_EXPERTS || tiles == 0 {
        return Err(fail("init_a1s_warmstart_weights_tiled dimension mismatch"));
    }
    let total_buckets = base_buckets * tiles;
    let extra = experts - P1_EXPERTS;
    let reserve = (extra as u64).saturating_mul(extra_share_q32);
    if reserve >= P3_Q32_ONE - (P1_EXPERTS as u64) {
        return Err(fail("extra_share_q32 reserve exceeds Q32 simplex"));
    }
    let keep_budget = P3_Q32_ONE - reserve;
    let mut s6 = BrierS6Weights::uniform(total_buckets, experts, train_sha)?;
    for b in 0..total_buckets {
        let src_b = b % base_buckets;
        let dst = &mut s6.weights_q32[b * experts..(b + 1) * experts];
        if extra == 0 {
            dst.copy_from_slice(&a1s_weights[src_b * P1_EXPERTS..(src_b + 1) * P1_EXPERTS]);
            continue;
        }
        let mut sum_keep = 0u64;
        let mut best_e = 0usize;
        let mut best_w = 0u64;
        for e in 0..P1_EXPERTS {
            let aw = a1s_weights[src_b * P1_EXPERTS + e];
            let scaled = ((u128::from(aw) * u128::from(keep_budget)) >> 32).max(1) as u64;
            dst[e] = scaled;
            sum_keep += scaled;
            if scaled >= best_w {
                best_w = scaled;
                best_e = e;
            }
        }
        if sum_keep < keep_budget {
            dst[best_e] += keep_budget - sum_keep;
        } else if sum_keep > keep_budget {
            let excess = sum_keep - keep_budget;
            if dst[best_e] > excess {
                dst[best_e] -= excess;
            }
        }
        dst[P1_EXPERTS..experts].fill(extra_share_q32);
        let check_sum: u64 = dst.iter().copied().sum();
        if check_sum != P3_Q32_ONE {
            let mut raw: Vec<i128> = dst.iter().map(|&v| i128::from(v)).collect();
            raw[best_e] += i128::from(P3_Q32_ONE) - i128::from(check_sum);
            let proj = project_simplex_q32(&raw, 1)?;
            dst.copy_from_slice(&proj);
        }
    }
    Ok(s6)
}

fn step_s6_em_weights(
    weights: &mut [u64],
    buckets_count: usize,
    experts: usize,
    resp_sum: &[u128],
    touched: &[u64],
    prior_weight: u128,
) -> Result<usize> {
    let mut updated = 0usize;
    for (b, &cnt) in touched.iter().enumerate().take(buckets_count) {
        if cnt == 0 {
            continue;
        }
        let start = b * experts;
        let denom = prior_weight + u128::from(cnt);
        let mut raw = Vec::with_capacity(experts);
        for e in 0..experts {
            let num = prior_weight * u128::from(weights[start + e]) + resp_sum[start + e];
            raw.push((num / denom) as i128);
        }
        let proj = project_simplex_q32(&raw, 1 << 10)?;
        weights[start..start + experts].copy_from_slice(&proj);
        updated += 1;
    }
    Ok(updated)
}

fn total_bits_q12(masses: &[u64]) -> Result<u128> {
    let mut sum = 0u128;
    for &m in masses {
        sum += u128::from(bits_q12(m)?);
    }
    Ok(sum)
}

#[derive(Default)]
struct SplitAcc {
    all: Tally,
    novel: Tally,
    response: Tally,
    novel_response: Tally,
    rec_all: Vec<u128>,
    rec_novel: Vec<u128>,
}

impl SplitAcc {
    fn new(records: usize) -> Self {
        Self {
            rec_all: Vec::with_capacity(records),
            rec_novel: Vec::with_capacity(records),
            ..Self::default()
        }
    }

    fn accumulate_chunk(
        &mut self,
        chunk_ids: &[u16],
        chunk_novel: &[u8],
        chunk_mask: &[u8],
        id_bytes: &[u8],
        masses: &[u64],
    ) -> Result<()> {
        let mut ord = 0usize;
        for (at, &id) in chunk_ids.iter().enumerate() {
            if id == 0 {
                self.rec_all.push(0);
                self.rec_novel.push(0);
                continue;
            }
            let bq12 = bits_q12(masses[ord])?;
            ord += 1;
            let is_novel = chunk_novel[at] == 1;
            let is_resp = chunk_mask[at] == 1;
            let len = id_bytes[usize::from(id)];
            if let Some(last) = self.rec_all.last_mut() {
                *last += u128::from(bq12);
            }
            if is_novel {
                if let Some(last) = self.rec_novel.last_mut() {
                    *last += u128::from(bq12);
                }
            }
            self.all.add(bq12, len);
            if is_novel {
                self.novel.add(bq12, len);
            }
            if is_resp {
                self.response.add(bq12, len);
            }
            if is_novel && is_resp {
                self.novel_response.add(bq12, len);
            }
        }
        if ord != masses.len() {
            return Err(fail(format!(
                "chunk mass len mismatch: consumed {ord} vs {}",
                masses.len()
            )));
        }
        Ok(())
    }
}

pub fn build_record_chunks(
    ids: &[u16],
    max_chunk_toks: usize,
) -> Result<Vec<(usize, usize, usize, usize)>> {
    if ids.first() != Some(&0) {
        return Err(fail("token stream must start with BOS=0"));
    }
    let mut chunks = Vec::new();
    let mut tok_s = 0usize;
    let mut ord_s = 0usize;
    let mut cur_toks = 0usize;
    let mut cur_ords = 0usize;
    let mut last_bos = 0usize;
    let mut last_ord_at_bos = 0usize;
    for (i, &t) in ids.iter().enumerate() {
        if t == 0 {
            if i > tok_s && cur_toks >= max_chunk_toks {
                chunks.push((tok_s, last_bos, ord_s, last_ord_at_bos));
                tok_s = last_bos;
                ord_s = last_ord_at_bos;
                cur_toks = i - tok_s;
            }
            last_bos = i;
            last_ord_at_bos = cur_ords;
        } else {
            cur_ords += 1;
        }
        cur_toks += 1;
    }
    if tok_s < ids.len() {
        chunks.push((tok_s, ids.len(), ord_s, cur_ords));
    }
    Ok(chunks)
}

struct ReplayedB4 {
    w_skip_static: BrierS6Weights,
    w_skip_err: BrierS6Weights,
    w_cache_static: BrierS6Weights,
    w_full: BrierS6Weights,
    arm2_p_errs: Vec<i8>,
    arm2_b_errs: Vec<i8>,
    arm2_s2_errs: [Vec<i8>; 3],
    arm4_p_errs: Vec<i8>,
    arm4_b_errs: Vec<i8>,
    arm4_s2_errs: [Vec<i8>; 3],
    replay_shas: Value,
    full_epochs: Vec<Value>,
    skip_err_epochs: Vec<Value>,
}

#[allow(clippy::too_many_arguments)]
fn replay_stage_b4_on_r(
    r_uort_path: &Path,
    b1a_r_tape: &Path,
    b1a_r_pick: &Path,
    a1s: &FrozenPickMixture,
    binding: [u8; 32],
    gs_table: &mut GoldenShellStateTable,
    s2_skip: &mut S2SkipTable,
    #[cfg(feature = "cuda")] gpu: &P4BrierGpuEngine,
    #[cfg(feature = "cuda")] gpu_gs: &mut crate::cuda::GpuS3Table,
    #[cfg(feature = "cuda")] gpu_s2s: &mut [crate::cuda::GpuS2Table; 3],
    out_dir: &Path,
) -> Result<ReplayedB4> {
    let r_uort_bytes = std::fs::read(r_uort_path)?;
    let r_uort_sha: [u8; 32] = Sha256::digest(&r_uort_bytes).into();
    if hex::encode(r_uort_sha) != R_UORT_SHA256 {
        return Err(fail("R UORT SHA-256 mismatch"));
    }
    let r_ids = read_uort_tokens(&r_uort_bytes)?;
    drop(r_uort_bytes);

    let (r_buckets, r_p1, _r_a1s, _) =
        load_p1_tape_and_verify_a1s(b1a_r_tape, b1a_r_pick, a1s, r_uort_sha, R_TARGETS)?;
    let a1s_weights = a1s.weights();
    let _ = binding;
    let (r_targets, r_gkeys, r_skip_keys) = stream_golden_and_skip_keys(&r_ids, &gs_table.token_h4);
    let (r_cache_states, r_cache_cols) = compute_causal_cache_features(&r_ids);
    drop(r_ids);
    if r_targets.len() != R_TARGETS || r_cache_cols.len() != R_TARGETS {
        return Err(fail("R target/cache length mismatch"));
    }
    let r_buckets_cache: Vec<u16> = r_buckets
        .iter()
        .zip(&r_cache_states)
        .map(|(&b, &hs)| u16::from(hs) * (S6_BUCKETS as u16) + b)
        .collect();
    drop(r_cache_states);

    #[cfg(feature = "cuda")]
    let r_s3_init = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 7)?.0;
    #[cfg(not(feature = "cuda"))]
    let r_s3_init: Vec<u64> = r_gkeys
        .iter()
        .zip(&r_targets)
        .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
        .collect();

    let mut r_skip_inits: Vec<Vec<u64>> = Vec::with_capacity(3);
    for (fam, sk) in r_skip_keys.iter().enumerate() {
        #[cfg(feature = "cuda")]
        r_skip_inits.push(gpu_s2s[fam].lookup_and_accum(sk, &r_targets, None, 7)?.0);
        #[cfg(not(feature = "cuda"))]
        r_skip_inits.push(
            sk.iter()
                .zip(&r_targets)
                .map(|(&k, &t)| s2_skip.families[fam].lookup_modulated_q48(k, t).0)
                .collect(),
        );
    }

    let mut r_22 = Vec::with_capacity(R_TARGETS * E22);
    for i in 0..R_TARGETS {
        r_22.extend_from_slice(&r_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
        r_22.push(r_s3_init[i]);
        for col in &r_skip_inits {
            r_22.push(col[i]);
        }
        r_22.extend_from_slice(&r_cache_cols[i]);
    }
    drop(r_p1);
    drop(r_s3_init);
    drop(r_skip_inits);
    drop(r_cache_cols);

    let slice_range = |src: &[u64], rows: usize, c_start: usize, c_end: usize| -> Vec<u64> {
        let mut out = Vec::with_capacity(rows * (P1_EXPERTS + c_end - c_start));
        for r in 0..rows {
            let base = r * E22;
            out.extend_from_slice(&src[base..base + P1_EXPERTS]);
            out.extend_from_slice(&src[base + c_start..base + c_end]);
        }
        out
    };

    // Arm 1: p4_c_skip_static
    let mut r_18_skip = slice_range(&r_22, R_TARGETS, 14, 18);
    let mut w_skip_static = init_a1s_warmstart_weights_tiled(
        a1s_weights,
        S6_BUCKETS,
        1,
        E18,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (sk_st_init, _, _, mut sk_st_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        gpu,
        S6_BUCKETS,
        E18,
        &r_buckets,
        &r_18_skip,
        &w_skip_static.weights_q32,
        CUDA_CHUNK,
    )?;
    drop(sk_st_init);
    for ep in 0..3u32 {
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_static.weights_q32,
            CUDA_CHUNK,
        )?;
        let _ = step_s6_em_weights(
            &mut w_skip_static.weights_q32,
            S6_BUCKETS,
            E18,
            &resp,
            &touched,
            64,
        )?;
        let (rm2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_static.weights_q32,
            CUDA_CHUNK,
        )?;
        drop(rm2);
        sk_st_prev_mb = mb_after;
        w_skip_static.epoch = ep + 1;
        w_skip_static.train_brier_q32_mean = mb_after;
    }
    let b1 = w_skip_static.encode_bytes()?;
    let sha_skip_static = hex::encode(Sha256::digest(&b1));
    if sha_skip_static != B4_SKIP_STATIC_SHA256 {
        return Err(fail(format!(
            "Stage B4 replay drift p4_c_skip_static: {sha_skip_static} != {B4_SKIP_STATIC_SHA256}"
        )));
    }
    std::fs::write(out_dir.join("p4_c_skip_static.uors6b01"), &b1)?;

    // Arm 2: p4_c_skip_errwrite
    let mut w_skip_err = init_a1s_warmstart_weights_tiled(
        a1s_weights,
        S6_BUCKETS,
        1,
        E18,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (sk_er_init, _, _, mut sk_er_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        gpu,
        S6_BUCKETS,
        E18,
        &r_buckets,
        &r_18_skip,
        &w_skip_err.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut sk_er_prev_bits = total_bits_q12(&sk_er_init)?;
    drop(sk_er_init);
    let mut skip_err_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = sk_er_prev_mb;
        let bits_before = sk_er_prev_bits;
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_err.weights_q32,
            CUDA_CHUNK,
        )?;
        let buckets_upd = step_s6_em_weights(
            &mut w_skip_err.weights_q32,
            S6_BUCKETS,
            E18,
            &resp,
            &touched,
            64,
        )?;
        let (rmix_post, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_err.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let (s3_upd, s2_upds) = {
            let (_m3, pd, ph, bd, bh) =
                gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, Some(&rmix_post), ERR_LR_SHIFT)?;
            let su3 = gs_table.apply_zero_centered_deltas(&pd, &ph, &bd, &bh);
            gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
            let (r3u, ..) = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, ERR_LR_SHIFT)?;
            for (i, &m) in r3u.iter().enumerate() {
                r_18_skip[i * E18 + 14] = m;
            }
            let mut su2 = [0usize; 3];
            for fam in 0..3usize {
                let (_m2, sd, sh) = gpu_s2s[fam].lookup_and_accum(
                    &r_skip_keys[fam],
                    &r_targets,
                    Some(&rmix_post),
                    ERR_LR_SHIFT,
                )?;
                su2[fam] = s2_skip.families[fam].apply_zero_centered_deltas(&sd, &sh);
                gpu_s2s[fam].update_errs(&s2_skip.families[fam].errs())?;
                let (r2u, ..) = gpu_s2s[fam].lookup_and_accum(
                    &r_skip_keys[fam],
                    &r_targets,
                    None,
                    ERR_LR_SHIFT,
                )?;
                for (i, &m) in r2u.iter().enumerate() {
                    r_18_skip[i * E18 + 15 + fam] = m;
                }
            }
            (su3, su2)
        };
        #[cfg(not(feature = "cuda"))]
        let (s3_upd, s2_upds) = {
            let su3 = gs_table.apply_epoch_brier_error_writes(
                &r_gkeys,
                &r_targets,
                &rmix_post,
                ERR_LR_SHIFT,
            );
            for i in 0..R_TARGETS {
                r_18_skip[i * E18 + 14] = gs_table.lookup_modulated_q48(r_gkeys[i], r_targets[i]).0;
            }
            let mut su2 = [0usize; 3];
            for fam in 0..3usize {
                su2[fam] = s2_skip.families[fam].apply_epoch_brier_error_writes(
                    &r_skip_keys[fam],
                    &r_targets,
                    &rmix_post,
                    ERR_LR_SHIFT,
                );
                for i in 0..R_TARGETS {
                    r_18_skip[i * E18 + 15 + fam] = s2_skip.families[fam]
                        .lookup_modulated_q48(r_skip_keys[fam][i], r_targets[i])
                        .0;
                }
            }
            (su3, su2)
        };
        drop(rmix_post);
        if ep == 2 {
            let (_rm_fin, resp_fin, touched_fin, _) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                gpu,
                S6_BUCKETS,
                E18,
                &r_buckets,
                &r_18_skip,
                &w_skip_err.weights_q32,
                CUDA_CHUNK,
            )?;
            let _ = step_s6_em_weights(
                &mut w_skip_err.weights_q32,
                S6_BUCKETS,
                E18,
                &resp_fin,
                &touched_fin,
                64,
            )?;
        }
        let (rm2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_err.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rm2)?;
        drop(rm2);
        if mb_after >= mb_before {
            return Err(fail("p4_c_skip_errwrite Brier failed to fall on epoch"));
        }
        sk_er_prev_mb = mb_after;
        sk_er_prev_bits = bits_after;
        w_skip_err.epoch = ep + 1;
        w_skip_err.train_brier_q32_mean = mb_after;
        skip_err_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "s3_cells_updated": s3_upd,
            "s2_cells_updated_by_family": s2_upds,
            "brier_fell": true,
        }));
    }
    drop(r_18_skip);
    drop(r_buckets);
    let arm2_p_errs = gs_table.p_errs();
    let arm2_b_errs = gs_table.b_errs();
    let arm2_s2_errs = [
        s2_skip.families[0].errs(),
        s2_skip.families[1].errs(),
        s2_skip.families[2].errs(),
    ];
    let b2 = w_skip_err.encode_bytes()?;
    let sha_skip_errwrite = hex::encode(Sha256::digest(&b2));
    if sha_skip_errwrite != B4_SKIP_ERRWRITE_SHA256 {
        return Err(fail(format!(
            "Stage B4 replay drift p4_c_skip_errwrite: {sha_skip_errwrite} != {B4_SKIP_ERRWRITE_SHA256}"
        )));
    }
    std::fs::write(out_dir.join("p4_c_skip_errwrite.uors6b01"), &b2)?;

    // Reset S3 & S2-skip errors cleanly back to 0 for Arm 3/4.
    for r in &mut gs_table.primary_rows {
        r.err_i8 = 0;
    }
    for r in &mut gs_table.backoff_rows {
        r.err_i8 = 0;
    }
    s2_skip.reset_errs();
    #[cfg(feature = "cuda")]
    {
        gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
        for (fam, gs2) in gpu_s2s.iter_mut().enumerate() {
            gs2.update_errs(&s2_skip.families[fam].errs())?;
        }
    }

    // Arm 3: p4_cache_static
    let r_18_cache = slice_range(&r_22, R_TARGETS, 18, 22);
    let mut w_cache_static = init_a1s_warmstart_weights_tiled(
        a1s_weights,
        S6_BUCKETS,
        3,
        E18,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (ca_init, _, _, mut ca_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        gpu,
        S6_CACHE_BUCKETS,
        E18,
        &r_buckets_cache,
        &r_18_cache,
        &w_cache_static.weights_q32,
        CUDA_CHUNK,
    )?;
    drop(ca_init);
    for ep in 0..3u32 {
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_CACHE_BUCKETS,
            E18,
            &r_buckets_cache,
            &r_18_cache,
            &w_cache_static.weights_q32,
            CUDA_CHUNK,
        )?;
        let _ = step_s6_em_weights(
            &mut w_cache_static.weights_q32,
            S6_CACHE_BUCKETS,
            E18,
            &resp,
            &touched,
            64,
        )?;
        let (rm2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_CACHE_BUCKETS,
            E18,
            &r_buckets_cache,
            &r_18_cache,
            &w_cache_static.weights_q32,
            CUDA_CHUNK,
        )?;
        drop(rm2);
        ca_prev_mb = mb_after;
        w_cache_static.epoch = ep + 1;
        w_cache_static.train_brier_q32_mean = mb_after;
    }
    drop(r_18_cache);
    let b3 = w_cache_static.encode_bytes()?;
    let sha_cache_static = hex::encode(Sha256::digest(&b3));
    if sha_cache_static != B4_CACHE_STATIC_SHA256 {
        return Err(fail(format!(
            "Stage B4 replay drift p4_cache_static: {sha_cache_static} != {B4_CACHE_STATIC_SHA256}"
        )));
    }
    std::fs::write(out_dir.join("p4_cache_static.uors6b01"), &b3)?;

    // Arm 4: p4_full_errwrite
    let mut w_full = init_a1s_warmstart_weights_tiled(
        a1s_weights,
        S6_BUCKETS,
        3,
        E22,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (fu_init, _, _, mut fu_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        gpu,
        S6_CACHE_BUCKETS,
        E22,
        &r_buckets_cache,
        &r_22,
        &w_full.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut fu_prev_bits = total_bits_q12(&fu_init)?;
    drop(fu_init);
    let mut full_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = fu_prev_mb;
        let bits_before = fu_prev_bits;
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_CACHE_BUCKETS,
            E22,
            &r_buckets_cache,
            &r_22,
            &w_full.weights_q32,
            CUDA_CHUNK,
        )?;
        let buckets_upd = step_s6_em_weights(
            &mut w_full.weights_q32,
            S6_CACHE_BUCKETS,
            E22,
            &resp,
            &touched,
            64,
        )?;
        let (rmix_post, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_CACHE_BUCKETS,
            E22,
            &r_buckets_cache,
            &r_22,
            &w_full.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let (s3_upd, s2_upds) = {
            let (_m3, pd, ph, bd, bh) =
                gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, Some(&rmix_post), ERR_LR_SHIFT)?;
            let su3 = gs_table.apply_zero_centered_deltas(&pd, &ph, &bd, &bh);
            gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
            let (r3u, ..) = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, ERR_LR_SHIFT)?;
            for (i, &m) in r3u.iter().enumerate() {
                r_22[i * E22 + 14] = m;
            }
            let mut su2 = [0usize; 3];
            for fam in 0..3usize {
                let (_m2, sd, sh) = gpu_s2s[fam].lookup_and_accum(
                    &r_skip_keys[fam],
                    &r_targets,
                    Some(&rmix_post),
                    ERR_LR_SHIFT,
                )?;
                su2[fam] = s2_skip.families[fam].apply_zero_centered_deltas(&sd, &sh);
                gpu_s2s[fam].update_errs(&s2_skip.families[fam].errs())?;
                let (r2u, ..) = gpu_s2s[fam].lookup_and_accum(
                    &r_skip_keys[fam],
                    &r_targets,
                    None,
                    ERR_LR_SHIFT,
                )?;
                for (i, &m) in r2u.iter().enumerate() {
                    r_22[i * E22 + 15 + fam] = m;
                }
            }
            (su3, su2)
        };
        #[cfg(not(feature = "cuda"))]
        let (s3_upd, s2_upds) = {
            let su3 = gs_table.apply_epoch_brier_error_writes(
                &r_gkeys,
                &r_targets,
                &rmix_post,
                ERR_LR_SHIFT,
            );
            for i in 0..R_TARGETS {
                r_22[i * E22 + 14] = gs_table.lookup_modulated_q48(r_gkeys[i], r_targets[i]).0;
            }
            let mut su2 = [0usize; 3];
            for fam in 0..3usize {
                su2[fam] = s2_skip.families[fam].apply_epoch_brier_error_writes(
                    &r_skip_keys[fam],
                    &r_targets,
                    &rmix_post,
                    ERR_LR_SHIFT,
                );
                for i in 0..R_TARGETS {
                    r_22[i * E22 + 15 + fam] = s2_skip.families[fam]
                        .lookup_modulated_q48(r_skip_keys[fam][i], r_targets[i])
                        .0;
                }
            }
            (su3, su2)
        };
        drop(rmix_post);
        if ep == 2 {
            let (_rm_fin, resp_fin, touched_fin, _) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                gpu,
                S6_CACHE_BUCKETS,
                E22,
                &r_buckets_cache,
                &r_22,
                &w_full.weights_q32,
                CUDA_CHUNK,
            )?;
            let _ = step_s6_em_weights(
                &mut w_full.weights_q32,
                S6_CACHE_BUCKETS,
                E22,
                &resp_fin,
                &touched_fin,
                64,
            )?;
        }
        let (rm2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            gpu,
            S6_CACHE_BUCKETS,
            E22,
            &r_buckets_cache,
            &r_22,
            &w_full.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rm2)?;
        drop(rm2);
        if mb_after >= mb_before {
            return Err(fail("p4_full_errwrite Brier failed to fall on epoch"));
        }
        fu_prev_mb = mb_after;
        fu_prev_bits = bits_after;
        w_full.epoch = ep + 1;
        w_full.train_brier_q32_mean = mb_after;
        full_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "s3_cells_updated": s3_upd,
            "s2_cells_updated_by_family": s2_upds,
            "brier_fell": true,
        }));
    }
    let _ = sk_st_prev_mb;
    let _ = ca_prev_mb;
    let arm4_p_errs = gs_table.p_errs();
    let arm4_b_errs = gs_table.b_errs();
    let arm4_s2_errs = [
        s2_skip.families[0].errs(),
        s2_skip.families[1].errs(),
        s2_skip.families[2].errs(),
    ];
    let b4 = w_full.encode_bytes()?;
    let sha_full_errwrite = hex::encode(Sha256::digest(&b4));
    if sha_full_errwrite != B4_FULL_ERRWRITE_SHA256 {
        return Err(fail(format!(
            "Stage B4 replay drift p4_full_errwrite: {sha_full_errwrite} != {B4_FULL_ERRWRITE_SHA256}"
        )));
    }
    std::fs::write(out_dir.join("p4_full_errwrite.uors6b01"), &b4)?;

    Ok(ReplayedB4 {
        w_skip_static,
        w_skip_err,
        w_cache_static,
        w_full,
        arm2_p_errs,
        arm2_b_errs,
        arm2_s2_errs,
        arm4_p_errs,
        arm4_b_errs,
        arm4_s2_errs,
        replay_shas: json!({
            "p4_c_skip_static": sha_skip_static,
            "p4_c_skip_errwrite": sha_skip_errwrite,
            "p4_cache_static": sha_cache_static,
            "p4_full_errwrite": sha_full_errwrite,
        }),
        full_epochs,
        skip_err_epochs,
    })
}

fn aggregate_rstar_devtest(rstar_path: &Path, dev: &DevTestData) -> Result<SplitAcc> {
    let raw = std::fs::read(rstar_path)?;
    if raw.len() != DEV_TEST_TARGETS * 8 || hex::encode(Sha256::digest(&raw)) != RSTAR_MASSES_SHA256
    {
        return Err(fail("rstar-devtest.masses.u64 size or SHA-256 mismatch"));
    }
    let mut acc = SplitAcc::new(DEV_TEST_RECORDS);
    let mut masses = Vec::with_capacity(DEV_TEST_TARGETS);
    for ch in raw.chunks_exact(8) {
        masses.push(u64::from_le_bytes(ch.try_into()?));
    }
    drop(raw);
    acc.accumulate_chunk(&dev.ids, &dev.novel, &dev.mask, &dev.id_bytes, &masses)?;
    if acc.rec_all.len() != DEV_TEST_RECORDS
        || acc.all.bits_q12 != RSTAR_ALL_BITS_Q12
        || acc.novel.bits_q12 != RSTAR_NOVEL_BITS_Q12
    {
        return Err(fail(format!(
            "R* aggregation mismatch: all={} (want {RSTAR_ALL_BITS_Q12}) novel={} (want {RSTAR_NOVEL_BITS_Q12})",
            acc.all.bits_q12, acc.novel.bits_q12
        )));
    }
    Ok(acc)
}

#[allow(clippy::too_many_arguments)]
pub fn run_devtest_eval(
    devtest_dir: &Path,
    r_uort_path: &Path,
    trunk_path: &Path,
    b1a_r_tape: &Path,
    b1a_r_pick: &Path,
    devtest_tape_path: &Path,
    rstar_masses_path: &Path,
    a1s_weights_path: &Path,
    a1s_seal_json: &Path,
    golden_shell_path: &Path,
    s2_skip_path: &Path,
    out_dir: &Path,
    out_json_path: &Path,
) -> Result<Value> {
    let trunk_model = uor_r4_integer::discrete_d11::IntegerDiscreteModel::load(trunk_path)?;
    if trunk_model.artifact_sha256() != TRUNK_SHA256 {
        return Err(fail("seed1.d11 SHA-256 mismatch"));
    }
    let seal: Value = serde_json::from_slice(&std::fs::read(a1s_seal_json)?)?;
    let binding_hex = seal["manifest_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing manifest_sha256"))?;
    let binding = decode_hex32(binding_hex)?;
    let frozen_a1s = FrozenPickMixture::from_bytes(&std::fs::read(a1s_weights_path)?, binding)?;
    if frozen_a1s.rule() != PickClassRule::Five {
        return Err(fail("expected A1s PickClassRule::Five"));
    }
    let _ = frozen_a1s.weights();

    let c_corpus_sha = decode_hex32(C_CORPUS_SHA256)?;
    let mut gs_table = GoldenShellStateTable::load_checked(
        &std::fs::read(golden_shell_path)?,
        c_corpus_sha,
        decode_hex32(GOLDEN_SHELL_SHA256)?,
    )?;
    let s2_bytes = std::fs::read(s2_skip_path)?;
    if hex::encode(Sha256::digest(&s2_bytes)) != S2_SKIP_SHA256 {
        return Err(fail("s2_skip.uors2s01 SHA-256 mismatch"));
    }
    let mut s2_skip = S2SkipTable::from_bytes(&s2_bytes, c_corpus_sha)?;
    drop(s2_bytes);

    #[cfg(feature = "cuda")]
    let gpu = P4BrierGpuEngine::new()?;
    #[cfg(feature = "cuda")]
    let mut gpu_gs = gs_table.upload_gpu(&gpu)?;
    #[cfg(feature = "cuda")]
    let mut gpu_s2s = [
        s2_skip.families[0].upload_gpu(&gpu)?,
        s2_skip.families[1].upload_gpu(&gpu)?,
        s2_skip.families[2].upload_gpu(&gpu)?,
    ];

    // Phase 1: Zero-drift Stage B4 R replay BEFORE reading DEV-test.
    let replayed = replay_stage_b4_on_r(
        r_uort_path,
        b1a_r_tape,
        b1a_r_pick,
        &frozen_a1s,
        binding,
        &mut gs_table,
        &mut s2_skip,
        #[cfg(feature = "cuda")]
        &gpu,
        #[cfg(feature = "cuda")]
        &mut gpu_gs,
        #[cfg(feature = "cuda")]
        &mut gpu_s2s,
        out_dir,
    )?;

    // Save frozen post-epoch-3 error cells for P4 serving integration.
    let to_u8_slice = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
    let p_err_u8 = to_u8_slice(&replayed.arm4_p_errs);
    let b_err_u8 = to_u8_slice(&replayed.arm4_b_errs);
    let mut s2_err_u8 = Vec::with_capacity(
        replayed.arm4_s2_errs[0].len()
            + replayed.arm4_s2_errs[1].len()
            + replayed.arm4_s2_errs[2].len(),
    );
    for fam in 0..3usize {
        s2_err_u8.extend(replayed.arm4_s2_errs[fam].iter().map(|&x| x as u8));
    }
    std::fs::write(out_dir.join("p5b_s3_err_p.i8"), &p_err_u8)?;
    std::fs::write(out_dir.join("p5b_s3_err_b.i8"), &b_err_u8)?;
    std::fs::write(out_dir.join("p5b_s2_skip_errs.i8"), &s2_err_u8)?;
    let p_err_sha = hex::encode(Sha256::digest(&p_err_u8));
    let b_err_sha = hex::encode(Sha256::digest(&b_err_u8));
    let s2_err_sha = hex::encode(Sha256::digest(&s2_err_u8));

    // Phase 2: Load sealed DEV-test and aggregate canonical R*.
    let dev = DevTestData::load_checked(devtest_dir)?;
    let rstar_acc = aggregate_rstar_devtest(rstar_masses_path, &dev)?;

    // Run Pick over sealed DEV-test with >= 13,000 native CPU verification probes.
    let mut pick_classes = Vec::with_capacity(DEV_TEST_TARGETS);
    let mut pick_hits = Vec::with_capacity(DEV_TEST_TARGETS);
    let mut pick_probes_checked = 0usize;
    #[cfg(feature = "cuda")]
    {
        let wins = windows(&dev.ids, 192)?;
        if wins.len() != DEV_TEST_TARGETS {
            return Err(fail("DEV-test window count mismatch"));
        }
        let mut frame = 0usize;
        let mut kept = Vec::new();
        for (ordinal, &(_at, prefix)) in wins.iter().enumerate() {
            frame += usize::from(prefix.len() == 1);
            let seam = [8192usize, READ_BATCH, CAPTURE_BATCH].iter().any(|w| {
                let off = ordinal % w;
                off == 0 || off + 1 == *w
            });
            if (frame <= 64 && (prefix.len() <= 8 || dev.ids[_at] == 1))
                || seam
                || ordinal % 8191 == 197
                || ordinal + 2 >= DEV_TEST_TARGETS
            {
                kept.push(ordinal);
            }
        }
        let pdev = PickDevice::new(&trunk_model, None)?;
        let DeviceTape { rows: prows, words } =
            pdev.rows(&dev.ids, 192, &|i| kept.binary_search(&i).is_ok())?;
        if prows.len() != DEV_TEST_TARGETS || words.len() != kept.len() {
            return Err(fail("DEV-test PickDevice row/probe count mismatch"));
        }
        for (&ordinal, observed) in kept.iter().zip(words) {
            let (at, prefix) = wins[ordinal];
            let pr = prows[ordinal];
            if observed.ordinal != ordinal {
                return Err(fail("DEV-test Pick ordinal drift"));
            }
            let ids_u32: Vec<u32> = prefix.iter().map(|&x| u32::from(x)).collect();
            let native = trunk_model.observe_identity_pick(&ids_u32)?;
            if native.block_words.as_slice() != observed.blocks.as_slice() {
                return Err(fail(format!("Pick native parity diverged at {ordinal}")));
            }
            let exp_hit = pr
                .pick
                .is_some_and(|p| prefix.get(p as usize + 1) == Some(&dev.ids[at]));
            if pr.hit != exp_hit {
                return Err(fail(format!("Pick hit diverged at {ordinal}")));
            }
            pick_probes_checked += 1;
        }
        for pr in prows {
            pick_classes.push(PickClassRule::Five.classify(pr.distance));
            pick_hits.push(u8::from(pr.hit));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut rec_s = 0usize;
        for (at, &tok) in dev.ids.iter().enumerate() {
            if tok == 0 {
                rec_s = at;
                continue;
            }
            let ws = rec_s.max(at.saturating_sub(192));
            let pfx: Vec<u32> = dev.ids[ws..at].iter().map(|&x| u32::from(x)).collect();
            let diag = trunk_model.observe_identity_pick(&pfx)?;
            let class = if diag.block_words[0][0] == u32::MAX {
                0
            } else {
                PickClassRule::Five.classify(Some(diag.block_words[0][1]))
            };
            pick_classes.push(class);
            pick_hits.push(u8::from(diag.successor == Some(u32::from(tok))));
            pick_probes_checked += 1;
        }
    }
    drop(trunk_model);

    // Open sealed 65536.features tape for single sequential streaming SHA-256 verify + read.
    let tape_file = File::open(devtest_tape_path)?;
    if tape_file.metadata()?.len() != (112 + 140 * DEV_TEST_TARGETS) as u64 {
        return Err(fail("DEV-test 65536.features byte len mismatch"));
    }
    let mut tape_reader = BufReader::with_capacity(1 << 20, tape_file);
    let mut tape_hasher = Sha256::new();
    let mut hdr = [0u8; 112];
    tape_reader.read_exact(&mut hdr)?;
    tape_hasher.update(hdr);
    if &hdr[..8] != b"UORCBF01"
        || u64::from_le_bytes(hdr[8..16].try_into()?) != DEV_TEST_TARGETS as u64
        || hdr[16..48] != c_corpus_sha
        || hdr[48..80] != decode_hex32(COMPACT_V3_SHA256)?
        || hdr[80..112] != decode_hex32(DEV_TEST_UORT_SHA256)?
    {
        return Err(fail("DEV-test UORCBF01 header/SHA binding mismatch"));
    }

    let chunks = build_record_chunks(&dev.ids, 2_097_152)?;
    let mut acc_p1 = SplitAcc::new(DEV_TEST_RECORDS);
    let mut acc_skip_static = SplitAcc::new(DEV_TEST_RECORDS);
    let mut acc_skip_err = SplitAcc::new(DEV_TEST_RECORDS);
    let mut acc_cache_static = SplitAcc::new(DEV_TEST_RECORDS);
    let mut acc_full = SplitAcc::new(DEV_TEST_RECORDS);

    let zero_p_errs = vec![0i8; gs_table.primary_rows.len()];
    let zero_b_errs = vec![0i8; gs_table.backoff_rows.len()];
    let zero_s2_errs = [
        vec![0i8; s2_skip.families[0].entry_count()],
        vec![0i8; s2_skip.families[1].entry_count()],
        vec![0i8; s2_skip.families[2].entry_count()],
    ];

    let mut spot_checked = 0usize;
    let mut spot_mix_div = 0usize;
    let mut spot_s3_div = 0usize;
    let mut spot_s2_div = 0usize;
    let mut shifted_parity_ok = false;

    let mut crow = [0u8; 140];
    let mut counts = [0u32; 11];
    let mut totals = [0u64; 11];

    for (chunk_idx, &(tok_s, tok_e, ord_s, ord_e)) in chunks.iter().enumerate() {
        let clen = ord_e - ord_s;
        let mut c_buckets = Vec::with_capacity(clen);
        let mut c_p1 = Vec::with_capacity(clen * P1_EXPERTS);
        let mut c_a1s_masses = Vec::with_capacity(clen);

        for local_i in 0..clen {
            let ord = ord_s + local_i;
            tape_reader.read_exact(&mut crow)?;
            tape_hasher.update(crow);
            let mut h = 0usize;
            for j in 0..11 {
                counts[j] = u32::from_le_bytes(crow[4 * j..4 * (j + 1)].try_into()?);
                totals[j] = u64::from_le_bytes(crow[44 + 8 * j..44 + 8 * (j + 1)].try_into()?);
                if totals[j] > 0 {
                    h = j;
                }
            }
            let kn = u64::from_le_bytes(crow[132..].try_into()?);
            let pick = PickTarget {
                class: pick_classes[ord],
                hit: pick_hits[ord] != 0,
            };
            let log2 = 63 - totals[h].leading_zeros() as usize;
            let plain_b = h * 16 + log2.min(15);
            let bkt = u16::try_from(usize::from(pick.class) * 176 + plain_b)?;
            c_buckets.push(bkt);

            c_p1.push(kn);
            for j in 0..11 {
                let m = if totals[j] == 0 {
                    0
                } else {
                    ((u128::from(counts[j]) << 48) / u128::from(totals[j])) as u64
                };
                c_p1.push(m);
            }
            c_p1.push(KN_BASE_MASS);
            c_p1.push(if pick.hit { S3_Q48_ONE } else { 0 });

            let (fb, fm) = frozen_a1s.predict_mass(kn, &counts, &totals, pick)?;
            if fb != bkt {
                return Err(fail(format!("P1 bucket mismatch at ord {ord}")));
            }
            c_a1s_masses.push(fm);
        }

        acc_p1.accumulate_chunk(
            &dev.ids[tok_s..tok_e],
            &dev.novel[tok_s..tok_e],
            &dev.mask[tok_s..tok_e],
            &dev.id_bytes,
            &c_a1s_masses,
        )?;
        drop(c_a1s_masses);

        let (c_cache_states, c_cache_cols) = compute_causal_cache_features(&dev.ids[tok_s..tok_e]);
        let (c_targets, c_gkeys, c_skip_keys) =
            stream_golden_and_skip_keys(&dev.ids[tok_s..tok_e], &gs_table.token_h4);
        if c_targets.len() != clen || c_cache_cols.len() != clen {
            return Err(fail("chunk DEV-test target/cache length mismatch"));
        }
        let c_buckets_cache: Vec<u16> = c_buckets
            .iter()
            .zip(&c_cache_states)
            .map(|(&b, &hs)| u16::from(hs) * (S6_BUCKETS as u16) + b)
            .collect();
        drop(c_cache_states);

        // Helper to assemble 18-col skip matrix ([14 P1 | S3 | skip0..2]) for given S3+S2 errors.
        let eval_s3_s2 = |gs: &mut GoldenShellStateTable,
                          s2: &mut S2SkipTable,
                          #[cfg(feature = "cuda")] g_gs: &mut crate::cuda::GpuS3Table,
                          #[cfg(feature = "cuda")] g_s2s: &mut [crate::cuda::GpuS2Table; 3],
                          pe: &[i8],
                          be: &[i8],
                          s2e: &[Vec<i8>; 3]|
         -> Result<(Vec<u64>, [Vec<u64>; 3])> {
            gs.set_errs(pe, be)?;
            for (fam, s2_arr) in s2e.iter().enumerate().take(3) {
                s2.families[fam].set_errs(s2_arr)?;
            }
            #[cfg(feature = "cuda")]
            {
                g_gs.update_errs(pe, be)?;
                for (fam, s2_arr) in s2e.iter().enumerate().take(3) {
                    g_s2s[fam].update_errs(s2_arr)?;
                }
                let s3m = g_gs
                    .lookup_and_accum(&c_gkeys, &c_targets, None, ERR_LR_SHIFT)?
                    .0;
                let s0 = g_s2s[0]
                    .lookup_and_accum(&c_skip_keys[0], &c_targets, None, ERR_LR_SHIFT)?
                    .0;
                let s1 = g_s2s[1]
                    .lookup_and_accum(&c_skip_keys[1], &c_targets, None, ERR_LR_SHIFT)?
                    .0;
                let s2v = g_s2s[2]
                    .lookup_and_accum(&c_skip_keys[2], &c_targets, None, ERR_LR_SHIFT)?
                    .0;
                Ok((s3m, [s0, s1, s2v]))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let s3m: Vec<u64> = c_gkeys
                    .iter()
                    .zip(&c_targets)
                    .map(|(&k, &t)| gs.lookup_modulated_q48(k, t).0)
                    .collect();
                let s0: Vec<u64> = c_skip_keys[0]
                    .iter()
                    .zip(&c_targets)
                    .map(|(&k, &t)| s2.families[0].lookup_modulated_q48(k, t).0)
                    .collect();
                let s1: Vec<u64> = c_skip_keys[1]
                    .iter()
                    .zip(&c_targets)
                    .map(|(&k, &t)| s2.families[1].lookup_modulated_q48(k, t).0)
                    .collect();
                let s2v: Vec<u64> = c_skip_keys[2]
                    .iter()
                    .zip(&c_targets)
                    .map(|(&k, &t)| s2.families[2].lookup_modulated_q48(k, t).0)
                    .collect();
                Ok((s3m, [s0, s1, s2v]))
            }
        };

        // Arm 1: p4_c_skip_static (err=0)
        let (z_s3, z_sk) = eval_s3_s2(
            &mut gs_table,
            &mut s2_skip,
            #[cfg(feature = "cuda")]
            &mut gpu_gs,
            #[cfg(feature = "cuda")]
            &mut gpu_s2s,
            &zero_p_errs,
            &zero_b_errs,
            &zero_s2_errs,
        )?;
        let mut c18_zero = Vec::with_capacity(clen * E18);
        for i in 0..clen {
            c18_zero.extend_from_slice(&c_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
            c18_zero.push(z_s3[i]);
            c18_zero.push(z_sk[0][i]);
            c18_zero.push(z_sk[1][i]);
            c18_zero.push(z_sk[2][i]);
        }
        let (m_sk_st, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E18,
            &c_buckets,
            &c18_zero,
            &replayed.w_skip_static.weights_q32,
            CUDA_CHUNK,
        )?;
        if chunk_idx == 0 && clen >= MACRO_BATCH {
            let (_m2, rp2, t2, b2) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &c_buckets[..MACRO_BATCH],
                &c18_zero[..MACRO_BATCH * E18],
                &replayed.w_skip_static.weights_q32,
                65_521,
            )?;
            let (_m1, rp1, t1, b1) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &c_buckets[..MACRO_BATCH],
                &c18_zero[..MACRO_BATCH * E18],
                &replayed.w_skip_static.weights_q32,
                CUDA_CHUNK,
            )?;
            if rp2 != rp1 || t2 != t1 || b2 != b1 {
                return Err(fail("DEV-test shifted EM batch parity failed (880 bkts)"));
            }
            let (_bm2, g2, tg2, bb2) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &c_buckets[..MACRO_BATCH],
                &c18_zero[..MACRO_BATCH * E18],
                &replayed.w_skip_static.weights_q32,
                65_521,
            )?;
            let (_bm1, g1, tg1, bb1) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &c_buckets[..MACRO_BATCH],
                &c18_zero[..MACRO_BATCH * E18],
                &replayed.w_skip_static.weights_q32,
                CUDA_CHUNK,
            )?;
            if g2 != g1 || tg2 != tg1 || bb2 != bb1 {
                return Err(fail(
                    "DEV-test shifted Brier batch parity failed (880 bkts)",
                ));
            }
        }
        acc_skip_static.accumulate_chunk(
            &dev.ids[tok_s..tok_e],
            &dev.novel[tok_s..tok_e],
            &dev.mask[tok_s..tok_e],
            &dev.id_bytes,
            &m_sk_st,
        )?;
        drop(c18_zero);
        drop(m_sk_st);
        drop(z_s3);
        drop(z_sk);

        // Arm 2: p4_c_skip_errwrite
        let (e2_s3, e2_sk) = eval_s3_s2(
            &mut gs_table,
            &mut s2_skip,
            #[cfg(feature = "cuda")]
            &mut gpu_gs,
            #[cfg(feature = "cuda")]
            &mut gpu_s2s,
            &replayed.arm2_p_errs,
            &replayed.arm2_b_errs,
            &replayed.arm2_s2_errs,
        )?;
        let mut c18_e2 = Vec::with_capacity(clen * E18);
        for i in 0..clen {
            c18_e2.extend_from_slice(&c_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
            c18_e2.push(e2_s3[i]);
            c18_e2.push(e2_sk[0][i]);
            c18_e2.push(e2_sk[1][i]);
            c18_e2.push(e2_sk[2][i]);
        }
        let (m_sk_er, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E18,
            &c_buckets,
            &c18_e2,
            &replayed.w_skip_err.weights_q32,
            CUDA_CHUNK,
        )?;
        acc_skip_err.accumulate_chunk(
            &dev.ids[tok_s..tok_e],
            &dev.novel[tok_s..tok_e],
            &dev.mask[tok_s..tok_e],
            &dev.id_bytes,
            &m_sk_er,
        )?;
        drop(c18_e2);
        drop(m_sk_er);
        drop(e2_s3);
        drop(e2_sk);

        // Arm 3: p4_cache_static
        let mut c18_ca = Vec::with_capacity(clen * E18);
        for i in 0..clen {
            c18_ca.extend_from_slice(&c_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
            c18_ca.extend_from_slice(&c_cache_cols[i]);
        }
        let (m_ca_st, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_CACHE_BUCKETS,
            E18,
            &c_buckets_cache,
            &c18_ca,
            &replayed.w_cache_static.weights_q32,
            CUDA_CHUNK,
        )?;
        acc_cache_static.accumulate_chunk(
            &dev.ids[tok_s..tok_e],
            &dev.novel[tok_s..tok_e],
            &dev.mask[tok_s..tok_e],
            &dev.id_bytes,
            &m_ca_st,
        )?;
        drop(c18_ca);
        drop(m_ca_st);

        // Arm 4: p4_full_errwrite + >= 25,000 CPU parity probes
        let (e4_s3, e4_sk) = eval_s3_s2(
            &mut gs_table,
            &mut s2_skip,
            #[cfg(feature = "cuda")]
            &mut gpu_gs,
            #[cfg(feature = "cuda")]
            &mut gpu_s2s,
            &replayed.arm4_p_errs,
            &replayed.arm4_b_errs,
            &replayed.arm4_s2_errs,
        )?;
        let mut c22 = Vec::with_capacity(clen * E22);
        for i in 0..clen {
            c22.extend_from_slice(&c_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
            c22.push(e4_s3[i]);
            c22.push(e4_sk[0][i]);
            c22.push(e4_sk[1][i]);
            c22.push(e4_sk[2][i]);
            c22.extend_from_slice(&c_cache_cols[i]);
        }
        let (m_full, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_CACHE_BUCKETS,
            E22,
            &c_buckets_cache,
            &c22,
            &replayed.w_full.weights_q32,
            CUDA_CHUNK,
        )?;
        if chunk_idx == 0 && clen >= MACRO_BATCH {
            let (_m2, rp2, t2, b2) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_CACHE_BUCKETS,
                E22,
                &c_buckets_cache[..MACRO_BATCH],
                &c22[..MACRO_BATCH * E22],
                &replayed.w_full.weights_q32,
                65_521,
            )?;
            let (_m1, rp1, t1, b1) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_CACHE_BUCKETS,
                E22,
                &c_buckets_cache[..MACRO_BATCH],
                &c22[..MACRO_BATCH * E22],
                &replayed.w_full.weights_q32,
                CUDA_CHUNK,
            )?;
            if rp2 != rp1 || t2 != t1 || b2 != b1 {
                return Err(fail("DEV-test shifted batch parity failed (2640 bkts)"));
            }
            shifted_parity_ok = true;
        }
        for i in 0..clen {
            let ord = ord_s + i;
            if ord < 256 || ord % 839 == 197 || ord + 16 >= DEV_TEST_TARGETS {
                spot_checked += 1;
                if gs_table.lookup_modulated_q48(c_gkeys[i], c_targets[i]).0 != e4_s3[i] {
                    spot_s3_div += 1;
                }
                for fam in 0..3usize {
                    if s2_skip.families[fam]
                        .lookup_modulated_q48(c_skip_keys[fam][i], c_targets[i])
                        .0
                        != e4_sk[fam][i]
                    {
                        spot_s2_div += 1;
                    }
                }
                let b = usize::from(c_buckets_cache[i]);
                let exp_m = brier_mix_q48(
                    &replayed.w_full.weights_q32[b * E22..(b + 1) * E22],
                    &c22[i * E22..(i + 1) * E22],
                )?;
                if exp_m != m_full[i] {
                    spot_mix_div += 1;
                }
            }
        }
        acc_full.accumulate_chunk(
            &dev.ids[tok_s..tok_e],
            &dev.novel[tok_s..tok_e],
            &dev.mask[tok_s..tok_e],
            &dev.id_bytes,
            &m_full,
        )?;
    }

    if tape_reader.read(&mut [0u8])? != 0
        || hex::encode(tape_hasher.finalize()) != DEV_TEST_TAPE_SHA256
    {
        return Err(fail("DEV-test 65536.features EOF/SHA-256 mismatch"));
    }
    if acc_p1.all.bits_q12 != P1_A1S_DEVTEST_ALL_Q12
        || acc_p1.novel.bits_q12 != P1_A1S_DEVTEST_NOVEL_Q12
    {
        return Err(fail(format!(
            "P1(A1s) DEV-test mismatch: {} / {} != {P1_A1S_DEVTEST_ALL_Q12} / {P1_A1S_DEVTEST_NOVEL_Q12}",
            acc_p1.all.bits_q12, acc_p1.novel.bits_q12
        )));
    }
    if !shifted_parity_ok
        || pick_probes_checked < 13_000
        || spot_checked < 25_000
        || spot_mix_div != 0
        || spot_s3_div != 0
        || spot_s2_div != 0
    {
        return Err(fail(format!(
            "DEV-test parity failed: shifted={shifted_parity_ok} pick_probes={pick_probes_checked} spot={spot_checked} mix={spot_mix_div} s3={spot_s3_div} s2={spot_s2_div}"
        )));
    }

    let ci_skip_err_vs_static =
        paired_record_ci(&acc_skip_err.rec_all, &acc_skip_static.rec_all, 1000, 4390)?;
    let ci_full_vs_cache_static =
        paired_record_ci(&acc_full.rec_all, &acc_cache_static.rec_all, 1000, 4390)?;

    let summarize_arm = |name: &str, acc: &SplitAcc| -> Result<Value> {
        let ci_vs_p1_all = paired_record_ci(&acc.rec_all, &acc_p1.rec_all, 1000, 4390)?;
        let ci_vs_p1_nov = paired_record_ci(&acc.rec_novel, &acc_p1.rec_novel, 1000, 4390)?;
        let ci_vs_rstar_all = paired_record_ci(&acc.rec_all, &rstar_acc.rec_all, 1000, 4390)?;
        let micro_all = acc.all.micro_bpb().unwrap_or(0);
        let pct_vs_p1_bps = (acc.all.bits_q12 as i128 * 10_000) / (P1_A1S_DEVTEST_ALL_Q12 as i128);
        let gate_1pct_fewer_bits =
            acc.all.bits_q12 * 100 <= P1_A1S_DEVTEST_ALL_Q12 * 99 && ci_vs_p1_all[1] < 0;
        let ratio_vs_rstar_bps = (micro_all * 10_000) / RSTAR_DEVTEST_BPB_MICRO;
        Ok(json!({
            "arm": name,
            "all": acc.all.json(),
            "novel": acc.novel.json(),
            "response": acc.response.json(),
            "novel_response": acc.novel_response.json(),
            "ci_minus_p1_all_q12": [ci_vs_p1_all[0].to_string(), ci_vs_p1_all[1].to_string()],
            "ci_minus_p1_novel_q12": [ci_vs_p1_nov[0].to_string(), ci_vs_p1_nov[1].to_string()],
            "ci_minus_rstar_all_q12": [ci_vs_rstar_all[0].to_string(), ci_vs_rstar_all[1].to_string()],
            "pct_of_p1_bps": pct_vs_p1_bps,
            "ratio_vs_rstar_bps": ratio_vs_rstar_bps,
            "clears_095_rstar": micro_all * 100 <= RSTAR_DEVTEST_BPB_MICRO * 95,
            "clears_090_rstar": micro_all * 100 <= RSTAR_DEVTEST_BPB_MICRO * 90,
            "clears_080_rstar": micro_all * 100 <= RSTAR_DEVTEST_BPB_MICRO * 80,
            "gate_brier_1pct_fewer_bits": gate_1pct_fewer_bits,
        }))
    };

    let report = json!({
        "phase": "Stage B5b Sealed D87 DEV-test Single-Shot P3 Qualification",
        "dev_test_records": DEV_TEST_RECORDS,
        "dev_test_tokens": DEV_TEST_TOKENS,
        "dev_test_targets": DEV_TEST_TARGETS,
        "b4_replay_uors6b01_sha256": replayed.replay_shas,
        "b4_skip_err_epochs": replayed.skip_err_epochs,
        "b4_full_epochs": replayed.full_epochs,
        "exported_error_cells": {
            "p5b_s3_err_p_sha256": p_err_sha,
            "p5b_s3_err_b_sha256": b_err_sha,
            "p5b_s2_skip_errs_sha256": s2_err_sha,
        },
        "rstar_baseline": {
            "all": rstar_acc.all.json(),
            "novel": rstar_acc.novel.json(),
            "micro_bpb_all": RSTAR_DEVTEST_BPB_MICRO,
            "thresh_095_rstar_micro": (RSTAR_DEVTEST_BPB_MICRO * 95) / 100,
            "thresh_090_rstar_micro": (RSTAR_DEVTEST_BPB_MICRO * 90) / 100,
            "thresh_080_rstar_micro": (RSTAR_DEVTEST_BPB_MICRO * 80) / 100,
        },
        "p1_a1s_baseline": {
            "all": acc_p1.all.json(),
            "novel": acc_p1.novel.json(),
            "response": acc_p1.response.json(),
            "novel_response": acc_p1.novel_response.json(),
        },
        "pick_probes_checked": pick_probes_checked,
        "spot_check_probes": spot_checked,
        "spot_check_divergences": spot_mix_div,
        "s3_spot_check_divergences": spot_s3_div,
        "s2_spot_check_divergences": spot_s2_div,
        "shifted_batch_parity": shifted_parity_ok,
        "controlled_error_write_deltas": {
            "p4_c_skip_errwrite_minus_static_all_q12": [
                ci_skip_err_vs_static[0].to_string(),
                ci_skip_err_vs_static[1].to_string(),
            ],
            "p4_full_errwrite_minus_cache_static_all_q12": [
                ci_full_vs_cache_static[0].to_string(),
                ci_full_vs_cache_static[1].to_string(),
            ],
        },
        "arms": [
            summarize_arm("p4_c_skip_static", &acc_skip_static)?,
            summarize_arm("p4_c_skip_errwrite", &acc_skip_err)?,
            summarize_arm("p4_cache_static", &acc_cache_static)?,
            summarize_arm("p4_full_errwrite", &acc_full)?,
        ],
    });
    std::fs::write(out_json_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_chunked_cache_matches_unchunked_bit_for_bit() {
        let mut ids = Vec::new();
        for rec in 0..8usize {
            ids.push(0u16);
            for j in 0..14usize {
                ids.push(((rec * 7 + j * 3) % 40 + 2) as u16);
            }
        }
        let chunks = build_record_chunks(&ids, 35).unwrap();
        assert!(chunks.len() >= 3);

        let (full_st, full_cols) = compute_causal_cache_features(&ids);
        let (_t_f, full_gk, full_sk) = stream_golden_and_skip_keys(&ids, &[3u8; S3_VOCAB]);

        let mut chunked_st = Vec::new();
        let mut chunked_cols = Vec::new();
        let mut chunked_gk = Vec::new();
        let mut chunked_sk0 = Vec::new();
        for (ts, te, os, oe) in chunks {
            let (st, cols) = compute_causal_cache_features(&ids[ts..te]);
            let (_tg, gk, sk) = stream_golden_and_skip_keys(&ids[ts..te], &[3u8; S3_VOCAB]);
            assert_eq!(st.len(), oe - os);
            chunked_st.extend(st);
            chunked_cols.extend(cols);
            chunked_gk.extend(gk);
            chunked_sk0.extend_from_slice(&sk[0]);
        }
        assert_eq!(full_st, chunked_st);
        assert_eq!(full_cols, chunked_cols);
        assert_eq!(full_gk, chunked_gk);
        assert_eq!(full_sk[0], chunked_sk0);
    }

    #[test]
    fn devtest_census_rejects_tampered_split() {
        let tmp = std::env::temp_dir().join(format!("p5b-devtest-chk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dev-test.json"), b"{\"split\":\"wrong\"}").unwrap();
        assert!(DevTestData::load_checked(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
