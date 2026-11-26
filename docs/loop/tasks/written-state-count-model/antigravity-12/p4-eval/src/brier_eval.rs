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

pub const C_CORPUS_SHA256: &str =
    "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
pub const GOLDEN_SHELL_SHA256: &str =
    "e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c";
pub const COMPACT_V3_SHA256: &str =
    "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const TRUNK_SHA256: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const R_UORT_SHA256: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub const DEV_TUNE_UORT_SHA256: &str =
    "75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a";
pub const DEV_TUNE_CENSUS_SHA256: &str =
    "90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467";
pub const SEALED_DEV_TEST_CENSUS_SHA256: &str =
    "ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2";
pub const SEALED_DEV_TEST_UORT_SHA256: &str =
    "023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b";

pub const P1_A1S_DEVTUNE_Q12: u128 = 125_750_699_315;
pub const RSTAR_DEVTUNE_BPB_MICRO: u128 = 1_432_318;
pub const R_TARGETS: usize = 15_947_584;
pub const DEV_TUNE_RECORDS: u64 = 2_670;
pub const DEV_TUNE_TOKENS: usize = 10_810_087;
pub const DEV_TUNE_TARGETS: usize = 10_807_417;
pub const P1_EXPERTS: usize = 14;
pub const S6_BUCKETS: usize = 880;
pub const S6_CACHE_BUCKETS: usize = S6_BUCKETS * 3;
pub const KN_BASE_MASS: u64 = 1u64 << 36;

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

pub struct DevTuneData {
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

impl DevTuneData {
    pub fn load_checked(dir: &Path) -> Result<Self> {
        let census_bytes = std::fs::read(dir.join("dev-tune.json"))?;
        let csha = hex::encode(Sha256::digest(&census_bytes));
        if csha == SEALED_DEV_TEST_CENSUS_SHA256 || csha != DEV_TUNE_CENSUS_SHA256 {
            return Err(fail(format!(
                "refusing census SHA {csha} (DEV-test locked)"
            )));
        }
        let census: Value = serde_json::from_slice(&census_bytes)?;
        if census["split"] != "dev-tune"
            || census["records"] != DEV_TUNE_RECORDS
            || census["tokens"] != DEV_TUNE_TOKENS as u64
            || census["targets"] != DEV_TUNE_TARGETS as u64
        {
            return Err(fail("DEV-tune census identity mismatch"));
        }
        let uort_raw = load_pinned_entry(
            dir,
            &census["files"][0],
            "dev-tune.tokens.u16",
            64 + 2 * DEV_TUNE_TOKENS,
        )?;
        if hex::encode(Sha256::digest(&uort_raw)) == SEALED_DEV_TEST_UORT_SHA256 {
            return Err(fail("DEV-test UORT forbidden in Stage B4"));
        }
        let mask = load_pinned_entry(
            dir,
            &census["files"][1],
            "dev-tune.mask.u8",
            DEV_TUNE_TOKENS,
        )?;
        let _seg = load_pinned_entry(
            dir,
            &census["files"][2],
            "dev-tune.segment.u8",
            DEV_TUNE_TOKENS,
        )?;
        let novel = load_pinned_entry(
            dir,
            &census["files"][3],
            "dev-tune.novel.u8",
            DEV_TUNE_TOKENS,
        )?;
        let id_bytes = std::fs::read(dir.join("id-bytes.u8"))?;
        let id_sha = census["id_bytes_sha256"]
            .as_str()
            .ok_or_else(|| fail("missing id_bytes_sha256"))?;
        if id_bytes.len() != S3_VOCAB || hex::encode(Sha256::digest(&id_bytes)) != id_sha {
            return Err(fail("id-bytes.u8 mismatch"));
        }
        let ids = read_uort_tokens(&uort_raw)?;
        if ids.len() != DEV_TUNE_TOKENS {
            return Err(fail("DEV-tune token len mismatch"));
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
    a1s_weights_path: &Path,
    expected_binding: [u8; 32],
    expected_uort_sha: [u8; 32],
    rows: usize,
) -> Result<LoadedP1Tape> {
    let w_bytes = std::fs::read(a1s_weights_path)?;
    let frozen = FrozenPickMixture::from_bytes(&w_bytes, expected_binding)?;
    if frozen.rule() != PickClassRule::Five {
        return Err(fail("expected A1s PickClassRule::Five"));
    }
    let a1s_weights = frozen.weights().to_vec();
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
                    "A1s/P1 14-expert decomposition diverged at row {r}: ({bkt}, {bm}) vs ({fb}, {fm})"
                )));
            }
        }
        a1s_masses.push(fm);
    }
    Ok((buckets, p1_experts, a1s_masses, a1s_weights))
}

pub type DevTuneScore = (Tally, Tally, Tally, Tally, Vec<u128>, Vec<u128>);
pub type EpochEvalOut = (Vec<u64>, Vec<i128>, Vec<u64>, u64);
pub type EmEpochOut = (Vec<u64>, Vec<u128>, Vec<u64>, u64);

pub fn score_dev_tune(tune: &DevTuneData, masses: &[u64]) -> Result<DevTuneScore> {
    let non_bos = tune.ids.iter().filter(|&&id| id != 0).count();
    if non_bos != masses.len() {
        return Err(fail(format!(
            "DEV-tune target mismatch: expected {non_bos}, got {}",
            masses.len()
        )));
    }
    let mut t_all = Tally::default();
    let mut t_novel = Tally::default();
    let mut t_resp = Tally::default();
    let mut t_nov_resp = Tally::default();
    let mut rec_all = Vec::new();
    let mut rec_novel = Vec::new();
    let mut ord = 0usize;
    for (at, &id) in tune.ids.iter().enumerate() {
        if id == 0 {
            rec_all.push(0);
            rec_novel.push(0);
            continue;
        }
        let bq12 = bits_q12(masses[ord])?;
        ord += 1;
        let novel = tune.novel[at] == 1;
        let resp = tune.mask[at] == 1;
        let len = tune.id_bytes[usize::from(id)];
        if let Some(last) = rec_all.last_mut() {
            *last += u128::from(bq12);
        }
        if novel {
            if let Some(last) = rec_novel.last_mut() {
                *last += u128::from(bq12);
            }
        }
        t_all.add(bq12, len);
        if novel {
            t_novel.add(bq12, len);
        }
        if resp {
            t_resp.add(bq12, len);
        }
        if novel && resp {
            t_nov_resp.add(bq12, len);
        }
    }
    Ok((t_all, t_novel, t_resp, t_nov_resp, rec_all, rec_novel))
}

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

#[allow(clippy::too_many_arguments)]
pub fn run_brier_eval(
    tune_dir: &Path,
    r_uort_path: &Path,
    trunk_path: &Path,
    b1a_r_tape: &Path,
    b1a_r_pick: &Path,
    b1a_tune_tape: &Path,
    b2_tune_pick: &Path,
    a1s_weights_path: &Path,
    a1s_seal_json: &Path,
    golden_shell_path: &Path,
    s2_skip_path: &Path,
    out_dir: &Path,
    out_json_path: &Path,
) -> Result<Value> {
    let trunk = uor_r4_integer::discrete_d11::IntegerDiscreteModel::load(trunk_path)?;
    if trunk.artifact_sha256() != TRUNK_SHA256 {
        return Err(fail("seed1.d11 SHA-256 mismatch in run_brier_eval"));
    }
    drop(trunk);
    let tune = DevTuneData::load_checked(tune_dir)?;
    let r_uort_bytes = std::fs::read(r_uort_path)?;
    let r_uort_sha: [u8; 32] = Sha256::digest(&r_uort_bytes).into();
    if hex::encode(r_uort_sha) != R_UORT_SHA256 {
        return Err(fail("R UORT SHA-256 mismatch"));
    }
    let r_ids = read_uort_tokens(&r_uort_bytes)?;

    let seal: Value = serde_json::from_slice(&std::fs::read(a1s_seal_json)?)?;
    let binding_hex = seal["manifest_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing manifest_sha256"))?;
    let binding = decode_hex32(binding_hex)?;
    let tune_uort_sha = decode_hex32(DEV_TUNE_UORT_SHA256)?;

    let (r_buckets, r_p1, _r_a1s, a1s_weights) = load_p1_tape_and_verify_a1s(
        b1a_r_tape,
        b1a_r_pick,
        a1s_weights_path,
        binding,
        r_uort_sha,
        R_TARGETS,
    )?;
    let (tune_buckets, tune_p1, tune_a1s, _) = load_p1_tape_and_verify_a1s(
        b1a_tune_tape,
        b2_tune_pick,
        a1s_weights_path,
        binding,
        tune_uort_sha,
        DEV_TUNE_TARGETS,
    )?;

    let (base_all, base_novel, base_resp, base_nov_resp, base_rec_all, base_rec_novel) =
        score_dev_tune(&tune, &tune_a1s)?;
    if base_all.bits_q12 != P1_A1S_DEVTUNE_Q12 {
        return Err(fail(format!(
            "P1(A1s) baseline bits_q12 mismatch: {} != {P1_A1S_DEVTUNE_Q12}",
            base_all.bits_q12
        )));
    }

    let gs_bytes = std::fs::read(golden_shell_path)?;
    let expected_gs_sha = decode_hex32(GOLDEN_SHELL_SHA256)?;
    let c_corpus_sha = decode_hex32(C_CORPUS_SHA256)?;
    let mut gs_table =
        GoldenShellStateTable::load_checked(&gs_bytes, c_corpus_sha, expected_gs_sha)?;

    let s2_skip_bytes = std::fs::read(s2_skip_path)?;
    let s2_skip_sha = hex::encode(Sha256::digest(&s2_skip_bytes));
    let mut s2_skip = S2SkipTable::from_bytes(&s2_skip_bytes, c_corpus_sha)?;
    drop(s2_skip_bytes);
    if s2_skip.per_family_bytes()[0] > 20_000_000
        || s2_skip.per_family_bytes()[1] > 30_000_000
        || s2_skip.per_family_bytes()[2] > 30_000_000
        || s2_skip.total_encoded_bytes() > 80_000_000
    {
        return Err(fail("S2-skip exceeds byte budgets (20/30/30 MB, <=80 MB)"));
    }

    let (r_targets, r_gkeys, r_skip_keys) = stream_golden_and_skip_keys(&r_ids, &gs_table.token_h4);
    let (tune_targets, tune_gkeys, tune_skip_keys) =
        stream_golden_and_skip_keys(&tune.ids, &gs_table.token_h4);
    if r_targets.len() != R_TARGETS || tune_targets.len() != DEV_TUNE_TARGETS {
        return Err(fail("stream target length mismatch"));
    }

    let (r_cache_states, r_cache_cols) = compute_causal_cache_features(&r_ids);
    let (tune_cache_states, tune_cache_cols) = compute_causal_cache_features(&tune.ids);
    if r_cache_cols.len() != R_TARGETS || tune_cache_cols.len() != DEV_TUNE_TARGETS {
        return Err(fail("causal cache column length mismatch"));
    }
    let r_buckets_cache: Vec<u16> = r_buckets
        .iter()
        .zip(&r_cache_states)
        .map(|(&b, &hs)| u16::from(hs) * (S6_BUCKETS as u16) + b)
        .collect();
    let tune_buckets_cache: Vec<u16> = tune_buckets
        .iter()
        .zip(&tune_cache_states)
        .map(|(&b, &hs)| u16::from(hs) * (S6_BUCKETS as u16) + b)
        .collect();

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

    #[cfg(feature = "cuda")]
    let (r_s3_init, tune_s3_init) = (
        gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 7)?.0,
        gpu_gs
            .lookup_and_accum(&tune_gkeys, &tune_targets, None, 7)?
            .0,
    );
    #[cfg(not(feature = "cuda"))]
    let (r_s3_init, tune_s3_init) = (
        r_gkeys
            .iter()
            .zip(&r_targets)
            .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
            .collect::<Vec<_>>(),
        tune_gkeys
            .iter()
            .zip(&tune_targets)
            .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
            .collect::<Vec<_>>(),
    );

    let mut r_skip_inits: Vec<Vec<u64>> = Vec::with_capacity(3);
    let mut tune_skip_inits: Vec<Vec<u64>> = Vec::with_capacity(3);
    for fam in 0..3usize {
        #[cfg(feature = "cuda")]
        {
            r_skip_inits.push(
                gpu_s2s[fam]
                    .lookup_and_accum(&r_skip_keys[fam], &r_targets, None, 7)?
                    .0,
            );
            tune_skip_inits.push(
                gpu_s2s[fam]
                    .lookup_and_accum(&tune_skip_keys[fam], &tune_targets, None, 7)?
                    .0,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            r_skip_inits.push(
                r_skip_keys[fam]
                    .iter()
                    .zip(&r_targets)
                    .map(|(&k, &t)| s2_skip.families[fam].lookup_modulated_q48(k, t).0)
                    .collect(),
            );
            tune_skip_inits.push(
                tune_skip_keys[fam]
                    .iter()
                    .zip(&tune_targets)
                    .map(|(&k, &t)| s2_skip.families[fam].lookup_modulated_q48(k, t).0)
                    .collect(),
            );
        }
    }

    const E18: usize = 18;
    const E22: usize = 22;
    const CUDA_CHUNK: usize = 65_536;
    const MACRO_BATCH: usize = 524_288;
    const WARM_EXTRA_SHARE_Q32: u64 = P3_Q32_ONE >> 7;
    const ERR_LR_SHIFT: u32 = 7;

    // Build unified 22-expert matrices: [14 P1 | S3 | skip0..2 | cache0..3].
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

    let mut tune_22 = Vec::with_capacity(DEV_TUNE_TARGETS * E22);
    for i in 0..DEV_TUNE_TARGETS {
        tune_22.extend_from_slice(&tune_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
        tune_22.push(tune_s3_init[i]);
        for col in &tune_skip_inits {
            tune_22.push(col[i]);
        }
        tune_22.extend_from_slice(&tune_cache_cols[i]);
    }
    drop(tune_p1);
    drop(r_cache_cols);
    drop(tune_cache_cols);

    // Slice 18-expert submatrix `[14 P1 | S3 | skip0..2]` for Arms 1 & 2, `[14 P1 | cache0..3]` for Arm 3.
    let slice_range = |src: &[u64], rows: usize, c_start: usize, c_end: usize| -> Vec<u64> {
        let keep = c_end - c_start;
        let mut out = Vec::with_capacity(rows * keep);
        for r in 0..rows {
            let base = r * E22;
            out.extend_from_slice(&src[base..base + P1_EXPERTS]);
            out.extend_from_slice(&src[base + c_start..base + c_end]);
        }
        out
    };

    // ---------------- Arm 1: p4_c_skip_static (18 experts, 880 buckets, EM S6, err_i8=0) ----------------
    let mut r_18_skip = slice_range(&r_22, R_TARGETS, 14, 18);
    let mut tune_18_skip = slice_range(&tune_22, DEV_TUNE_TARGETS, 14, 18);
    let mut w_skip_static = init_a1s_warmstart_weights_tiled(
        &a1s_weights,
        S6_BUCKETS,
        1,
        E18,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (sk_st_init, _, _, mut sk_st_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E18,
        &r_buckets,
        &r_18_skip,
        &w_skip_static.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut sk_st_prev_bits = total_bits_q12(&sk_st_init)?;
    drop(sk_st_init);
    let mut skip_static_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = sk_st_prev_mb;
        let bits_before = sk_st_prev_bits;
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_static.weights_q32,
            CUDA_CHUNK,
        )?;
        if ep == 0 {
            let (_m2, rp2, t2, b2) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &r_buckets[..MACRO_BATCH],
                &r_18_skip[..MACRO_BATCH * E18],
                &w_skip_static.weights_q32,
                65_521,
            )?;
            let (_m1, rp1, t1, b1) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &r_buckets[..MACRO_BATCH],
                &r_18_skip[..MACRO_BATCH * E18],
                &w_skip_static.weights_q32,
                CUDA_CHUNK,
            )?;
            if rp2 != rp1 || t2 != t1 || b2 != b1 {
                return Err(fail(
                    "shifted EM batch parity (65536 vs 65521) failed (880 bkts)",
                ));
            }
            let (_bm2, g2, tg2, bb2) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &r_buckets[..MACRO_BATCH],
                &r_18_skip[..MACRO_BATCH * E18],
                &w_skip_static.weights_q32,
                65_521,
            )?;
            let (_bm1, g1, tg1, bb1) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E18,
                &r_buckets[..MACRO_BATCH],
                &r_18_skip[..MACRO_BATCH * E18],
                &w_skip_static.weights_q32,
                CUDA_CHUNK,
            )?;
            if g2 != g1 || tg2 != tg1 || bb2 != bb1 {
                return Err(fail(
                    "shifted Brier batch parity (65536 vs 65521) failed (880 bkts)",
                ));
            }
        }
        let buckets_upd = step_s6_em_weights(
            &mut w_skip_static.weights_q32,
            S6_BUCKETS,
            E18,
            &resp,
            &touched,
            64,
        )?;
        let (rm2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_static.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rm2)?;
        drop(rm2);
        sk_st_prev_mb = mb_after;
        sk_st_prev_bits = bits_after;
        w_skip_static.epoch = ep + 1;
        w_skip_static.train_brier_q32_mean = mb_after;
        skip_static_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    let (tune_mix_skip_static, _, _, _) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E18,
        &tune_buckets,
        &tune_18_skip,
        &w_skip_static.weights_q32,
        CUDA_CHUNK,
    )?;
    std::fs::write(
        out_dir.join("p4_c_skip_static.uors6b01"),
        w_skip_static.encode_bytes()?,
    )?;

    // ---------------- Arm 2: p4_c_skip_errwrite (18 experts, 880 buckets, 3-epoch GPU Brier error-write lr7 + EM S6) ----------------
    let mut w_skip_err = init_a1s_warmstart_weights_tiled(
        &a1s_weights,
        S6_BUCKETS,
        1,
        E18,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (sk_er_init, _, _, mut sk_er_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
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
            &gpu,
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
            &gpu,
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
                &gpu,
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
            &gpu,
            S6_BUCKETS,
            E18,
            &r_buckets,
            &r_18_skip,
            &w_skip_err.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rm2)?;
        drop(rm2);
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
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    drop(r_18_skip);
    #[cfg(feature = "cuda")]
    let tune_s3_er1 = gpu_gs
        .lookup_and_accum(&tune_gkeys, &tune_targets, None, ERR_LR_SHIFT)?
        .0;
    #[cfg(not(feature = "cuda"))]
    let tune_s3_er1: Vec<u64> = tune_gkeys
        .iter()
        .zip(&tune_targets)
        .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
        .collect();
    for (i, &m) in tune_s3_er1.iter().enumerate() {
        tune_18_skip[i * E18 + 14] = m;
    }
    drop(tune_s3_er1);
    for fam in 0..3usize {
        #[cfg(feature = "cuda")]
        let ts2 = gpu_s2s[fam]
            .lookup_and_accum(&tune_skip_keys[fam], &tune_targets, None, ERR_LR_SHIFT)?
            .0;
        #[cfg(not(feature = "cuda"))]
        let ts2: Vec<u64> = tune_skip_keys[fam]
            .iter()
            .zip(&tune_targets)
            .map(|(&k, &t)| s2_skip.families[fam].lookup_modulated_q48(k, t).0)
            .collect();
        for (i, &m) in ts2.iter().enumerate() {
            tune_18_skip[i * E18 + 15 + fam] = m;
        }
    }
    let (tune_mix_skip_errwrite, _, _, _) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E18,
        &tune_buckets,
        &tune_18_skip,
        &w_skip_err.weights_q32,
        CUDA_CHUNK,
    )?;
    drop(tune_18_skip);
    std::fs::write(
        out_dir.join("p4_c_skip_errwrite.uors6b01"),
        w_skip_err.encode_bytes()?,
    )?;

    // Reset S3 and S2-skip errors cleanly back to 0 before running cache / full arms.
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
        for (fam, gpu_s2) in gpu_s2s.iter_mut().enumerate() {
            gpu_s2.update_errs(&s2_skip.families[fam].errs())?;
        }
    }

    // ---------------- Arm 3: p4_cache_static (18 experts: 14 P1 + 4 causal cache experts, 2,640 buckets, EM S6) ----------------
    let r_18_cache = slice_range(&r_22, R_TARGETS, 18, 22);
    let tune_18_cache = slice_range(&tune_22, DEV_TUNE_TARGETS, 18, 22);
    let mut w_cache_static = init_a1s_warmstart_weights_tiled(
        &a1s_weights,
        S6_BUCKETS,
        3,
        E18,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (ca_init, _, _, mut ca_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_CACHE_BUCKETS,
        E18,
        &r_buckets_cache,
        &r_18_cache,
        &w_cache_static.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut ca_prev_bits = total_bits_q12(&ca_init)?;
    drop(ca_init);
    let mut cache_static_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = ca_prev_mb;
        let bits_before = ca_prev_bits;
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_CACHE_BUCKETS,
            E18,
            &r_buckets_cache,
            &r_18_cache,
            &w_cache_static.weights_q32,
            CUDA_CHUNK,
        )?;
        let buckets_upd = step_s6_em_weights(
            &mut w_cache_static.weights_q32,
            S6_CACHE_BUCKETS,
            E18,
            &resp,
            &touched,
            64,
        )?;
        let (rm2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_CACHE_BUCKETS,
            E18,
            &r_buckets_cache,
            &r_18_cache,
            &w_cache_static.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rm2)?;
        drop(rm2);
        ca_prev_mb = mb_after;
        ca_prev_bits = bits_after;
        w_cache_static.epoch = ep + 1;
        w_cache_static.train_brier_q32_mean = mb_after;
        cache_static_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    drop(r_18_cache);
    let (tune_mix_cache_static, _, _, _) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_CACHE_BUCKETS,
        E18,
        &tune_buckets_cache,
        &tune_18_cache,
        &w_cache_static.weights_q32,
        CUDA_CHUNK,
    )?;
    drop(tune_18_cache);
    std::fs::write(
        out_dir.join("p4_cache_static.uors6b01"),
        w_cache_static.encode_bytes()?,
    )?;

    // ---------------- Arm 4: p4_full_errwrite (22 experts, 2,640 buckets, disjoint-C S2-skip + S3 error-write + 4 cache) ----------------
    let mut w_full = init_a1s_warmstart_weights_tiled(
        &a1s_weights,
        S6_BUCKETS,
        3,
        E22,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (fu_init, _, _, mut fu_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
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
            &gpu,
            S6_CACHE_BUCKETS,
            E22,
            &r_buckets_cache,
            &r_22,
            &w_full.weights_q32,
            CUDA_CHUNK,
        )?;
        if ep == 0 {
            let (_m2, rp2, t2, b2) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_CACHE_BUCKETS,
                E22,
                &r_buckets_cache[..MACRO_BATCH],
                &r_22[..MACRO_BATCH * E22],
                &w_full.weights_q32,
                65_521,
            )?;
            let (_m1, rp1, t1, b1) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_CACHE_BUCKETS,
                E22,
                &r_buckets_cache[..MACRO_BATCH],
                &r_22[..MACRO_BATCH * E22],
                &w_full.weights_q32,
                CUDA_CHUNK,
            )?;
            if rp2 != rp1 || t2 != t1 || b2 != b1 {
                return Err(fail(
                    "shifted EM batch parity (65536 vs 65521) failed (2640 bkts)",
                ));
            }
            let (_bm2, g2, tg2, bb2) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_CACHE_BUCKETS,
                E22,
                &r_buckets_cache[..MACRO_BATCH],
                &r_22[..MACRO_BATCH * E22],
                &w_full.weights_q32,
                65_521,
            )?;
            let (_bm1, g1, tg1, bb1) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_CACHE_BUCKETS,
                E22,
                &r_buckets_cache[..MACRO_BATCH],
                &r_22[..MACRO_BATCH * E22],
                &w_full.weights_q32,
                CUDA_CHUNK,
            )?;
            if g2 != g1 || tg2 != tg1 || bb2 != bb1 {
                return Err(fail(
                    "shifted Brier batch parity (65536 vs 65521) failed (2640 bkts)",
                ));
            }
        }
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
            &gpu,
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
                &gpu,
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
            &gpu,
            S6_CACHE_BUCKETS,
            E22,
            &r_buckets_cache,
            &r_22,
            &w_full.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rm2)?;
        drop(rm2);
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
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    drop(r_22);

    // Update & spot-check DEV-tune dynamic S3, S2-skip, and 22-expert mixture (>= 12,500 probes, 0 divs).
    #[cfg(feature = "cuda")]
    let tune_s3_full = gpu_gs
        .lookup_and_accum(&tune_gkeys, &tune_targets, None, ERR_LR_SHIFT)?
        .0;
    #[cfg(not(feature = "cuda"))]
    let tune_s3_full: Vec<u64> = tune_gkeys
        .iter()
        .zip(&tune_targets)
        .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
        .collect();
    let mut s3_spot_div = 0usize;
    let mut spot_checked = 0usize;
    for (i, &m) in tune_s3_full.iter().enumerate() {
        if i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS {
            spot_checked += 1;
            if gs_table
                .lookup_modulated_q48(tune_gkeys[i], tune_targets[i])
                .0
                != m
            {
                s3_spot_div += 1;
            }
        }
        tune_22[i * E22 + 14] = m;
    }
    drop(tune_s3_full);

    let mut s2_spot_div = 0usize;
    for fam in 0..3usize {
        #[cfg(feature = "cuda")]
        let ts2 = gpu_s2s[fam]
            .lookup_and_accum(&tune_skip_keys[fam], &tune_targets, None, ERR_LR_SHIFT)?
            .0;
        #[cfg(not(feature = "cuda"))]
        let ts2: Vec<u64> = tune_skip_keys[fam]
            .iter()
            .zip(&tune_targets)
            .map(|(&k, &t)| s2_skip.families[fam].lookup_modulated_q48(k, t).0)
            .collect();
        for (i, &m) in ts2.iter().enumerate() {
            if (i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS)
                && s2_skip.families[fam]
                    .lookup_modulated_q48(tune_skip_keys[fam][i], tune_targets[i])
                    .0
                    != m
            {
                s2_spot_div += 1;
            }
            tune_22[i * E22 + 15 + fam] = m;
        }
    }
    let (tune_mix_full, _, _, _) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_CACHE_BUCKETS,
        E22,
        &tune_buckets_cache,
        &tune_22,
        &w_full.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut spot_div = 0usize;
    for i in 0..DEV_TUNE_TARGETS {
        if i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS {
            let b = usize::from(tune_buckets_cache[i]);
            let exp_m = brier_mix_q48(
                &w_full.weights_q32[b * E22..(b + 1) * E22],
                &tune_22[i * E22..(i + 1) * E22],
            )?;
            if exp_m != tune_mix_full[i] {
                spot_div += 1;
            }
        }
    }
    if spot_checked < 12_500 || spot_div != 0 || s3_spot_div != 0 || s2_spot_div != 0 {
        return Err(fail(format!(
            "DEV-tune spot-check failed: probes={spot_checked} mix_div={spot_div} s3_div={s3_spot_div} s2_div={s2_spot_div}"
        )));
    }
    drop(tune_22);
    std::fs::write(
        out_dir.join("p4_full_errwrite.uors6b01"),
        w_full.encode_bytes()?,
    )?;

    let (_a1, _n1, _r1, _nr1, rec_all_sk_st, _) = score_dev_tune(&tune, &tune_mix_skip_static)?;
    let (_a2, _n2, _r2, _nr2, rec_all_sk_er, _) = score_dev_tune(&tune, &tune_mix_skip_errwrite)?;
    let (_a3, _n3, _r3, _nr3, rec_all_ca_st, _) = score_dev_tune(&tune, &tune_mix_cache_static)?;
    let (_a4, _n4, _r4, _nr4, rec_all_full, _) = score_dev_tune(&tune, &tune_mix_full)?;

    let ci_skip_err_minus_static = paired_record_ci(&rec_all_sk_er, &rec_all_sk_st, 1000, 4390)?;
    let ci_full_minus_cache_static = paired_record_ci(&rec_all_full, &rec_all_ca_st, 1000, 4390)?;

    let summarize_arm =
        |name: &str, masses: &[u64], epochs: Value, extra: Value| -> Result<Value> {
            let (all, nov, resp, nresp, rec_a, rec_n) = score_dev_tune(&tune, masses)?;
            let ci_vs_p1_all = paired_record_ci(&rec_a, &base_rec_all, 1000, 4390)?;
            let ci_vs_p1_nov = paired_record_ci(&rec_n, &base_rec_novel, 1000, 4390)?;
            let micro_all = all.micro_bpb().unwrap_or(0);
            let pct_vs_p1_bp = (all.bits_q12 as i128 * 10_000) / (P1_A1S_DEVTUNE_Q12 as i128);
            let gate_1pct_fewer_bits =
                all.bits_q12 * 100 <= P1_A1S_DEVTUNE_Q12 * 99 && ci_vs_p1_all[1] < 0;
            let ratio_vs_rstar_bp = (micro_all * 10_000) / RSTAR_DEVTUNE_BPB_MICRO;
            Ok(json!({
                "arm": name,
                "all": all.json(),
                "novel": nov.json(),
                "response": resp.json(),
                "novel_response": nresp.json(),
                "ci_minus_p1_all_q12": [ci_vs_p1_all[0].to_string(), ci_vs_p1_all[1].to_string()],
                "ci_minus_p1_novel_q12": [ci_vs_p1_nov[0].to_string(), ci_vs_p1_nov[1].to_string()],
                "pct_of_p1_bps": pct_vs_p1_bp,
                "ratio_vs_rstar_bps": ratio_vs_rstar_bp,
                "clears_095_rstar": micro_all * 100 <= RSTAR_DEVTUNE_BPB_MICRO * 95,
                "clears_090_rstar": micro_all * 100 <= RSTAR_DEVTUNE_BPB_MICRO * 90,
                "clears_080_rstar": micro_all * 100 <= RSTAR_DEVTUNE_BPB_MICRO * 80,
                "gate_brier_1pct_fewer_bits": gate_1pct_fewer_bits,
                "epochs": epochs,
                "extra": extra,
            }))
        };

    let s2_skip_meta = json!({
        "families": ["skip_lag2", "skip_3_1", "skip_4_1"],
        "family_ids": s2_skip.families.iter().map(|f| f.family()).collect::<Vec<_>>(),
        "s2_skip_sha256": s2_skip_sha,
        "per_family_bytes": s2_skip.per_family_bytes(),
        "per_family_ctxs": s2_skip.per_family_ctxs(),
        "per_family_entries": s2_skip.per_family_entries(),
        "total_bytes": s2_skip.total_encoded_bytes(),
        "max_bytes_budget": 80_000_000usize,
    });
    let report = json!({
        "phase": "Stage B4 P3 DEV-tune Disjoint-C S2-skip + Error-Write + Causal Session Cache",
        "dev_tune_targets": DEV_TUNE_TARGETS,
        "p1_a1s_baseline": {
            "all": base_all.json(),
            "novel": base_novel.json(),
            "response": base_resp.json(),
            "novel_response": base_nov_resp.json(),
            "rstar_devtune_bpb_micro": RSTAR_DEVTUNE_BPB_MICRO.to_string(),
        },
        "s2_skip_table": s2_skip_meta,
        "spot_check_probes": spot_checked,
        "spot_check_divergences": spot_div,
        "s3_spot_check_divergences": s3_spot_div,
        "s2_spot_check_divergences": s2_spot_div,
        "shifted_batch_parity": true,
        "controlled_error_write_deltas": {
            "p4_c_skip_errwrite_minus_static_all_q12": [
                ci_skip_err_minus_static[0].to_string(),
                ci_skip_err_minus_static[1].to_string(),
            ],
            "p4_full_errwrite_minus_cache_static_all_q12": [
                ci_full_minus_cache_static[0].to_string(),
                ci_full_minus_cache_static[1].to_string(),
            ],
        },
        "arms": [
            summarize_arm("p4_c_skip_static", &tune_mix_skip_static, json!(skip_static_epochs), json!({"experts": E18, "buckets": S6_BUCKETS, "err_write": false, "tables": "c_skip_plus_s3"}))?,
            summarize_arm("p4_c_skip_errwrite", &tune_mix_skip_errwrite, json!(skip_err_epochs), json!({"experts": E18, "buckets": S6_BUCKETS, "err_write": true, "lr_shift": ERR_LR_SHIFT, "tables": "c_skip_plus_s3"}))?,
            summarize_arm("p4_cache_static", &tune_mix_cache_static, json!(cache_static_epochs), json!({"experts": E18, "buckets": S6_CACHE_BUCKETS, "err_write": false, "tables": "causal_cache_only"}))?,
            summarize_arm("p4_full_errwrite", &tune_mix_full, json!(full_epochs), json!({"experts": E22, "buckets": S6_CACHE_BUCKETS, "err_write": true, "lr_shift": ERR_LR_SHIFT, "tables": "c_skip_plus_s3_plus_cache"}))?,
        ],
    });
    std::fs::write(out_json_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devtest_refused_and_tiled_warmstart_simplex_holds() {
        let tmp = std::env::temp_dir().join(format!("p4-brier-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dev-tune.json"), b"{\"split\":\"dev-test\"}").unwrap();
        assert!(DevTuneData::load_checked(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);

        let a1s_mock = {
            let mut v = vec![1u64 << 24; 14];
            v[0] = P3_Q32_ONE - 13 * (1u64 << 24);
            v
        };
        let ws = init_a1s_warmstart_weights_tiled(&a1s_mock, 1, 3, 22, P3_Q32_ONE >> 7, [0u8; 32])
            .unwrap();
        assert_eq!(ws.buckets, 3);
        assert_eq!(ws.experts, 22);
        for b in 0..3 {
            assert_eq!(
                ws.weights_q32[b * 22..(b + 1) * 22]
                    .iter()
                    .copied()
                    .sum::<u64>(),
                P3_Q32_ONE
            );
        }
    }
}
