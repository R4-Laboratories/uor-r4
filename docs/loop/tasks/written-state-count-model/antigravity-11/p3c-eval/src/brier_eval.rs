#![forbid(unsafe_code)]

use crate::s3_decode::{
    stream_golden_and_multi_s2_keys, GoldenShellStateTable, S2MultiOrderErrorTable, S3_Q48_ONE,
    S3_VOCAB,
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
use crate::cuda::P3BrierGpuEngine;

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
            return Err(fail("DEV-test UORT forbidden in Stage B3c"));
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
                return Err(fail(format!("A1s/P1 14-expert decomposition diverged at row {r}: ({bkt}, {bm}) vs ({fb}, {fm})")));
            }
        }
        a1s_masses.push(fm);
    }
    Ok((buckets, p1_experts, a1s_masses, a1s_weights))
}

pub type DevTuneScore = (Tally, Tally, Tally, Tally, Vec<u128>, Vec<u128>);
pub type EpochEvalOut = (Vec<u64>, Vec<i128>, Vec<u64>, u64);

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
    #[cfg(feature = "cuda")] gpu: &P3BrierGpuEngine,
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
    let mean_brier = (brier_sum / (rows as u128)) as u64;
    Ok((mixed_all, grads, touched, mean_brier))
}

pub fn init_a1s_warmstart_weights(
    a1s_weights: &[u64],
    buckets: usize,
    experts: usize,
    extra_share_q32: u64,
    train_sha: [u8; 32],
) -> Result<BrierS6Weights> {
    if a1s_weights.len() != buckets * P1_EXPERTS || experts < P1_EXPERTS {
        return Err(fail("init_a1s_warmstart_weights dimension mismatch"));
    }
    let extra = experts - P1_EXPERTS;
    let reserve = (extra as u64).saturating_mul(extra_share_q32);
    if reserve >= P3_Q32_ONE - (P1_EXPERTS as u64) {
        return Err(fail("extra_share_q32 reserve exceeds Q32 simplex"));
    }
    let keep_budget = P3_Q32_ONE - reserve;
    let mut s6 = BrierS6Weights::uniform(buckets, experts, train_sha)?;
    for b in 0..buckets {
        let dst = &mut s6.weights_q32[b * experts..(b + 1) * experts];
        if extra == 0 {
            dst.copy_from_slice(&a1s_weights[b * P1_EXPERTS..(b + 1) * P1_EXPERTS]);
            continue;
        }
        let mut sum_keep = 0u64;
        let mut best_e = 0usize;
        let mut best_w = 0u64;
        for e in 0..P1_EXPERTS {
            let aw = a1s_weights[b * P1_EXPERTS + e];
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

pub type EmEpochOut = (Vec<u64>, Vec<u128>, Vec<u64>, u64);

#[allow(clippy::too_many_arguments)]
fn gpu_or_cpu_em_epoch(
    #[cfg(feature = "cuda")] gpu: &P3BrierGpuEngine,
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

fn step_s6_brier_weights(
    weights: &mut [u64],
    buckets_count: usize,
    experts: usize,
    grads: &[i128],
    touched: &[u64],
    lr_shift: u32,
) -> Result<usize> {
    let mut updated = 0usize;
    for (b, &cnt) in touched.iter().enumerate().take(buckets_count) {
        if cnt == 0 {
            continue;
        }
        let start = b * experts;
        let denom = i128::from(cnt);
        let mut mean_g = Vec::with_capacity(experts);
        let mut sum_g = 0i128;
        for e in 0..experts {
            let mg = grads[start + e] / denom;
            sum_g += mg;
            mean_g.push(mg);
        }
        let avg_g = sum_g / (experts as i128);
        let mut raw = Vec::with_capacity(experts);
        for e in 0..experts {
            let step = (mean_g[e] - avg_g) >> lr_shift.min(31);
            raw.push(i128::from(weights[start + e]) + step);
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

fn slice_first_k_experts(
    src: &[u64],
    stride: usize,
    keep: usize,
    start: usize,
    end: usize,
) -> Vec<u64> {
    let mut out = Vec::with_capacity((end - start) * keep);
    for r in start..end {
        let base = r * stride;
        out.extend_from_slice(&src[base..base + keep]);
    }
    out
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
    let gs_train_sha = decode_hex32(C_CORPUS_SHA256)?;
    let mut gs_table =
        GoldenShellStateTable::load_checked(&gs_bytes, gs_train_sha, expected_gs_sha)?;

    let (r_targets, r_gkeys, r_s2keys) =
        stream_golden_and_multi_s2_keys(&r_ids, &gs_table.token_h4);
    let (tune_targets, tune_gkeys, tune_s2keys) =
        stream_golden_and_multi_s2_keys(&tune.ids, &gs_table.token_h4);
    if r_targets.len() != R_TARGETS || tune_targets.len() != DEV_TUNE_TARGETS {
        return Err(fail("stream target length mismatch"));
    }

    // 4 bounded S2 orders (1..=4): <= 25,000,000 B per order, <= 100,000,000 B total (`ORDER.md:9`).
    let mut s2_multi = S2MultiOrderErrorTable::build_from_r_stream(&r_ids, 25_000_000)?;
    if s2_multi.total_encoded_bytes() > 100_000_000 {
        return Err(fail("S2 multi-order total encoded bytes > 100,000,000"));
    }
    #[cfg(feature = "cuda")]
    let gpu = P3BrierGpuEngine::new()?;
    #[cfg(feature = "cuda")]
    let mut gpu_gs = gs_table.upload_gpu(&gpu)?;
    #[cfg(feature = "cuda")]
    let mut gpu_s2s = [
        s2_multi.orders[0].upload_gpu(&gpu)?,
        s2_multi.orders[1].upload_gpu(&gpu)?,
        s2_multi.orders[2].upload_gpu(&gpu)?,
        s2_multi.orders[3].upload_gpu(&gpu)?,
    ];

    // Pre-extract initial static S3 and S2(1..=4) expert columns into unified 19-expert matrices
    // (14 P1 + S3 + 4 S2 orders), then drop initial P1 tapes immediately.
    #[cfg(feature = "cuda")]
    let (r_s3_init, tune_s3_init) = (
        gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 8)?.0,
        gpu_gs
            .lookup_and_accum(&tune_gkeys, &tune_targets, None, 8)?
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

    let mut r_s2_inits: Vec<Vec<u64>> = Vec::with_capacity(4);
    let mut tune_s2_inits: Vec<Vec<u64>> = Vec::with_capacity(4);
    for ord in 0..4usize {
        #[cfg(feature = "cuda")]
        {
            r_s2_inits.push(
                gpu_s2s[ord]
                    .lookup_and_accum(&r_s2keys[ord], &r_targets, None, 8)?
                    .0,
            );
            tune_s2_inits.push(
                gpu_s2s[ord]
                    .lookup_and_accum(&tune_s2keys[ord], &tune_targets, None, 8)?
                    .0,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            r_s2_inits.push(
                r_s2keys[ord]
                    .iter()
                    .zip(&r_targets)
                    .map(|(&k, &t)| s2_multi.orders[ord].lookup_modulated_q48(k, t).0)
                    .collect(),
            );
            tune_s2_inits.push(
                tune_s2keys[ord]
                    .iter()
                    .zip(&tune_targets)
                    .map(|(&k, &t)| s2_multi.orders[ord].lookup_modulated_q48(k, t).0)
                    .collect(),
            );
        }
    }

    const E15: usize = 15;
    const E19: usize = 19;
    let mut r_19 = Vec::with_capacity(R_TARGETS * E19);
    for i in 0..R_TARGETS {
        r_19.extend_from_slice(&r_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
        r_19.push(r_s3_init[i]);
        for col in &r_s2_inits {
            r_19.push(col[i]);
        }
    }
    drop(r_p1);

    let mut tune_19 = Vec::with_capacity(DEV_TUNE_TARGETS * E19);
    for i in 0..DEV_TUNE_TARGETS {
        tune_19.extend_from_slice(&tune_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
        tune_19.push(tune_s3_init[i]);
        for col in &tune_s2_inits {
            tune_19.push(col[i]);
        }
    }
    drop(tune_p1);

    let r_15_static = slice_first_k_experts(&r_19, E19, E15, 0, R_TARGETS);
    let tune_15_static = slice_first_k_experts(&tune_19, E19, E15, 0, DEV_TUNE_TARGETS);

    const MACRO_BATCH: usize = 524_288;
    const CUDA_CHUNK: usize = 65_536;
    const WARM_EXTRA_SHARE_Q32: u64 = P3_Q32_ONE >> 7;

    // Arm 1: p3c_a1s_brier_s6 (15 experts, A1s warm-start + conservative Brier S6, lr_shift=7, chunk parity 65536==65521).
    let mut w_brier_s6 = init_a1s_warmstart_weights(
        &a1s_weights,
        S6_BUCKETS,
        E15,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (m_init, _g_init, _t_init, mut s6_prev_mb) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E15,
        &r_buckets,
        &r_15_static,
        &w_brier_s6.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut s6_prev_bits = total_bits_q12(&m_init)?;
    drop(m_init);
    let mut s6_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = s6_prev_mb;
        let bits_before = s6_prev_bits;
        let mut buckets_upd = 0usize;
        let mut pos = 0usize;
        while pos < R_TARGETS {
            let end = (pos + MACRO_BATCH).min(R_TARGETS);
            let (_bmix, grads, touched, mb_batch) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E15,
                &r_buckets[pos..end],
                &r_15_static[pos * E15..end * E15],
                &w_brier_s6.weights_q32,
                CUDA_CHUNK,
            )?;
            if ep == 0 && pos == 0 {
                let (_m2, g2, t2, b2) = gpu_or_cpu_eval_epoch(
                    #[cfg(feature = "cuda")]
                    &gpu,
                    S6_BUCKETS,
                    E15,
                    &r_buckets[pos..end],
                    &r_15_static[pos * E15..end * E15],
                    &w_brier_s6.weights_q32,
                    65_521,
                )?;
                if g2 != grads || t2 != touched || b2 != mb_batch {
                    return Err(fail(
                        "shifted Brier batch parity (65536 vs 65521) failed in p3c_a1s_brier_s6",
                    ));
                }
            }
            buckets_upd += step_s6_brier_weights(
                &mut w_brier_s6.weights_q32,
                S6_BUCKETS,
                E15,
                &grads,
                &touched,
                7,
            )?;
            pos = end;
        }
        let (rmix2, _g_after, _t_after, mb_after) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E15,
            &r_buckets,
            &r_15_static,
            &w_brier_s6.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rmix2)?;
        drop(rmix2);
        s6_prev_mb = mb_after;
        s6_prev_bits = bits_after;
        w_brier_s6.epoch = ep + 1;
        w_brier_s6.train_brier_q32_mean = mb_after;
        s6_epochs.push(json!({
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
    let (tune_mix_s6, _tg, _tt, _tbm) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E15,
        &tune_buckets,
        &tune_15_static,
        &w_brier_s6.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut spot_checked = 0usize;
    let mut spot_div = 0usize;
    for i in 0..DEV_TUNE_TARGETS {
        if i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS {
            spot_checked += 1;
            let b = usize::from(tune_buckets[i]);
            let exp_m = brier_mix_q48(
                &w_brier_s6.weights_q32[b * E15..(b + 1) * E15],
                &tune_15_static[i * E15..(i + 1) * E15],
            )?;
            if exp_m != tune_mix_s6[i] {
                spot_div += 1;
            }
        }
    }
    if spot_checked < 12_500 || spot_div != 0 {
        return Err(fail(format!(
            "DEV-tune S6 Brier spot-check failed: checked={spot_checked} div={spot_div}"
        )));
    }
    std::fs::write(
        out_dir.join("p3c_a1s_brier_s6.uors6b01"),
        w_brier_s6.encode_bytes()?,
    )?;

    // Arm 2: p3c_a1s_em_s3_brier (15 experts, A1s warm-start + EM S6 refit + GPU S3 Brier error write).
    let mut r_15_s3 = r_15_static;
    let mut tune_15_s3 = tune_15_static;
    let mut w_em_s3 = init_a1s_warmstart_weights(
        &a1s_weights,
        S6_BUCKETS,
        E15,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (em_s3_init_m, _, _, mut em_s3_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E15,
        &r_buckets,
        &r_15_s3,
        &w_em_s3.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut em_s3_prev_bits = total_bits_q12(&em_s3_init_m)?;
    drop(em_s3_init_m);
    let mut em_s3_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = em_s3_prev_mb;
        let bits_before = em_s3_prev_bits;
        let (_rm0, resp, touched, mb_check) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E15,
            &r_buckets,
            &r_15_s3,
            &w_em_s3.weights_q32,
            CUDA_CHUNK,
        )?;
        if ep == 0 {
            let (_m2, rp2, t2, b2) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E15,
                &r_buckets[..MACRO_BATCH],
                &r_15_s3[..MACRO_BATCH * E15],
                &w_em_s3.weights_q32,
                65_521,
            )?;
            let (_m1, rp1, t1, b1) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E15,
                &r_buckets[..MACRO_BATCH],
                &r_15_s3[..MACRO_BATCH * E15],
                &w_em_s3.weights_q32,
                CUDA_CHUNK,
            )?;
            if rp2 != rp1 || t2 != t1 || b2 != b1 || mb_check == u64::MAX {
                return Err(fail("shifted EM batch parity (65536 vs 65521) failed"));
            }
        }
        let buckets_upd = step_s6_em_weights(
            &mut w_em_s3.weights_q32,
            S6_BUCKETS,
            E15,
            &resp,
            &touched,
            64,
        )?;
        let (rmix_post_s6, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E15,
            &r_buckets,
            &r_15_s3,
            &w_em_s3.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let cells_upd = {
            let (_m_cur, pd, ph, bd, bh) =
                gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, Some(&rmix_post_s6), 9)?;
            let upd = gs_table.apply_zero_centered_deltas(&pd, &ph, &bd, &bh);
            gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
            let (r_s3_upd, ..) = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 9)?;
            for (i, &m) in r_s3_upd.iter().enumerate() {
                r_15_s3[i * E15 + 14] = m;
            }
            upd
        };
        #[cfg(not(feature = "cuda"))]
        let cells_upd = {
            let upd =
                gs_table.apply_epoch_brier_error_writes(&r_gkeys, &r_targets, &rmix_post_s6, 9);
            for i in 0..R_TARGETS {
                r_15_s3[i * E15 + 14] = gs_table.lookup_modulated_q48(r_gkeys[i], r_targets[i]).0;
            }
            upd
        };
        drop(rmix_post_s6);
        if ep == 2 {
            let (_rm_fin, resp_fin, touched_fin, _) = gpu_or_cpu_em_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E15,
                &r_buckets,
                &r_15_s3,
                &w_em_s3.weights_q32,
                CUDA_CHUNK,
            )?;
            let _ = step_s6_em_weights(
                &mut w_em_s3.weights_q32,
                S6_BUCKETS,
                E15,
                &resp_fin,
                &touched_fin,
                64,
            )?;
        }
        let (rmix2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E15,
            &r_buckets,
            &r_15_s3,
            &w_em_s3.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rmix2)?;
        drop(rmix2);
        em_s3_prev_mb = mb_after;
        em_s3_prev_bits = bits_after;
        w_em_s3.epoch = ep + 1;
        w_em_s3.train_brier_q32_mean = mb_after;
        em_s3_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "s3_cells_updated": cells_upd,
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    #[cfg(feature = "cuda")]
    let tune_s3_upd = gpu_gs
        .lookup_and_accum(&tune_gkeys, &tune_targets, None, 8)?
        .0;
    #[cfg(not(feature = "cuda"))]
    let tune_s3_upd: Vec<u64> = tune_gkeys
        .iter()
        .zip(&tune_targets)
        .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
        .collect();
    let mut s3_spot_div = 0usize;
    for i in 0..DEV_TUNE_TARGETS {
        if (i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS)
            && gs_table
                .lookup_modulated_q48(tune_gkeys[i], tune_targets[i])
                .0
                != tune_s3_upd[i]
        {
            s3_spot_div += 1;
        }
        tune_15_s3[i * E15 + 14] = tune_s3_upd[i];
    }
    if s3_spot_div != 0 {
        return Err(fail(format!(
            "DEV-tune S3 GPU-vs-CPU spot-check failed: div={s3_spot_div}"
        )));
    }
    drop(tune_s3_upd);
    let (tune_mix_em_s3, _tg3, _tt3, _tbm3) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E15,
        &tune_buckets,
        &tune_15_s3,
        &w_em_s3.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut em_spot_div = 0usize;
    for i in 0..DEV_TUNE_TARGETS {
        if i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS {
            let b = usize::from(tune_buckets[i]);
            let exp_m = brier_mix_q48(
                &w_em_s3.weights_q32[b * E15..(b + 1) * E15],
                &tune_15_s3[i * E15..(i + 1) * E15],
            )?;
            if exp_m != tune_mix_em_s3[i] {
                em_spot_div += 1;
            }
        }
    }
    if em_spot_div != 0 {
        return Err(fail(format!("DEV-tune EM S3 spot-check div={em_spot_div}")));
    }
    drop(r_15_s3);
    drop(tune_15_s3);
    std::fs::write(
        out_dir.join("p3c_a1s_em_s3_brier.uors6b01"),
        w_em_s3.encode_bytes()?,
    )?;

    // Helper to reset S3 & S2 error cells cleanly to 0 before running an independent 19-expert arm.
    let reset_dynamic_tables = |gs_t: &mut GoldenShellStateTable,
                                s2_m: &mut S2MultiOrderErrorTable,
                                r_mat: &mut [u64],
                                t_mat: &mut [u64],
                                #[cfg(feature = "cuda")] g_gs: &mut crate::cuda::GpuS3Table,
                                #[cfg(feature = "cuda")] g_s2s: &mut [crate::cuda::GpuS2Table;
                                         4]|
     -> Result<()> {
        for r in &mut gs_t.primary_rows {
            r.err_i8 = 0;
        }
        for r in &mut gs_t.backoff_rows {
            r.err_i8 = 0;
        }
        s2_m.reset_errs();
        #[cfg(feature = "cuda")]
        {
            g_gs.update_errs(&gs_t.p_errs(), &gs_t.b_errs())?;
            for ord in 0..4usize {
                g_s2s[ord].update_errs(&s2_m.orders[ord].errs())?;
            }
        }
        for i in 0..R_TARGETS {
            r_mat[i * E19 + 14] = r_s3_init[i];
            for ord in 0..4usize {
                r_mat[i * E19 + 15 + ord] = r_s2_inits[ord][i];
            }
        }
        for i in 0..DEV_TUNE_TARGETS {
            t_mat[i * E19 + 14] = tune_s3_init[i];
            for ord in 0..4usize {
                t_mat[i * E19 + 15 + ord] = tune_s2_inits[ord][i];
            }
        }
        Ok(())
    };

    // Arm 3: p3c_a1s_em_s2_multi_brier (19 experts, A1s warm-start + EM S6 + GPU S3 & S2(1..=4) Brier error cells).
    reset_dynamic_tables(
        &mut gs_table,
        &mut s2_multi,
        &mut r_19,
        &mut tune_19,
        #[cfg(feature = "cuda")]
        &mut gpu_gs,
        #[cfg(feature = "cuda")]
        &mut gpu_s2s,
    )?;
    let mut w_em_s2 = init_a1s_warmstart_weights(
        &a1s_weights,
        S6_BUCKETS,
        E19,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (em_s2_init_m, _, _, mut em_s2_prev_mb) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E19,
        &r_buckets,
        &r_19,
        &w_em_s2.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut em_s2_prev_bits = total_bits_q12(&em_s2_init_m)?;
    drop(em_s2_init_m);
    let mut em_s2_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = em_s2_prev_mb;
        let bits_before = em_s2_prev_bits;
        let (_rm, resp, touched, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E19,
            &r_buckets,
            &r_19,
            &w_em_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        let buckets_upd = step_s6_em_weights(
            &mut w_em_s2.weights_q32,
            S6_BUCKETS,
            E19,
            &resp,
            &touched,
            64,
        )?;
        let (rmix_post, _, _, _) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E19,
            &r_buckets,
            &r_19,
            &w_em_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let (s3_upd, s2_upds) = {
            let (_m3, pd, ph, bd, bh) =
                gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, Some(&rmix_post), 9)?;
            let su3 = gs_table.apply_zero_centered_deltas(&pd, &ph, &bd, &bh);
            gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
            let (r3u, ..) = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 9)?;
            for (i, &m) in r3u.iter().enumerate() {
                r_19[i * E19 + 14] = m;
            }
            let mut su2 = [0usize; 4];
            for ord in 0..4usize {
                let (_m2, sd, sh) = gpu_s2s[ord].lookup_and_accum(
                    &r_s2keys[ord],
                    &r_targets,
                    Some(&rmix_post),
                    9,
                )?;
                su2[ord] = s2_multi.orders[ord].apply_zero_centered_deltas(&sd, &sh);
                gpu_s2s[ord].update_errs(&s2_multi.orders[ord].errs())?;
                let (r2u, ..) =
                    gpu_s2s[ord].lookup_and_accum(&r_s2keys[ord], &r_targets, None, 9)?;
                for (i, &m) in r2u.iter().enumerate() {
                    r_19[i * E19 + 15 + ord] = m;
                }
            }
            (su3, su2)
        };
        #[cfg(not(feature = "cuda"))]
        let (s3_upd, s2_upds) = {
            let su3 = gs_table.apply_epoch_brier_error_writes(&r_gkeys, &r_targets, &rmix_post, 9);
            for i in 0..R_TARGETS {
                r_19[i * E19 + 14] = gs_table.lookup_modulated_q48(r_gkeys[i], r_targets[i]).0;
            }
            let mut su2 = [0usize; 4];
            for ord in 0..4usize {
                su2[ord] = s2_multi.orders[ord].apply_epoch_brier_error_writes(
                    &r_s2keys[ord],
                    &r_targets,
                    &rmix_post,
                    9,
                );
                for i in 0..R_TARGETS {
                    r_19[i * E19 + 15 + ord] = s2_multi.orders[ord]
                        .lookup_modulated_q48(r_s2keys[ord][i], r_targets[i])
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
                E19,
                &r_buckets,
                &r_19,
                &w_em_s2.weights_q32,
                CUDA_CHUNK,
            )?;
            let _ = step_s6_em_weights(
                &mut w_em_s2.weights_q32,
                S6_BUCKETS,
                E19,
                &resp_fin,
                &touched_fin,
                64,
            )?;
        }
        let (rmix2, _, _, mb_after) = gpu_or_cpu_em_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E19,
            &r_buckets,
            &r_19,
            &w_em_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rmix2)?;
        drop(rmix2);
        em_s2_prev_mb = mb_after;
        em_s2_prev_bits = bits_after;
        w_em_s2.epoch = ep + 1;
        w_em_s2.train_brier_q32_mean = mb_after;
        em_s2_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "s3_cells_updated": s3_upd,
            "s2_cells_updated_by_order": s2_upds,
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    #[cfg(feature = "cuda")]
    let tune_s3_em2 = gpu_gs
        .lookup_and_accum(&tune_gkeys, &tune_targets, None, 8)?
        .0;
    #[cfg(not(feature = "cuda"))]
    let tune_s3_em2: Vec<u64> = tune_gkeys
        .iter()
        .zip(&tune_targets)
        .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
        .collect();
    for (i, &m) in tune_s3_em2.iter().enumerate() {
        tune_19[i * E19 + 14] = m;
    }
    drop(tune_s3_em2);
    let mut s2_spot_div = 0usize;
    for ord in 0..4usize {
        #[cfg(feature = "cuda")]
        let ts2 = gpu_s2s[ord]
            .lookup_and_accum(&tune_s2keys[ord], &tune_targets, None, 8)?
            .0;
        #[cfg(not(feature = "cuda"))]
        let ts2: Vec<u64> = tune_s2keys[ord]
            .iter()
            .zip(&tune_targets)
            .map(|(&k, &t)| s2_multi.orders[ord].lookup_modulated_q48(k, t).0)
            .collect();
        for (i, &m) in ts2.iter().enumerate() {
            if (i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS)
                && s2_multi.orders[ord]
                    .lookup_modulated_q48(tune_s2keys[ord][i], tune_targets[i])
                    .0
                    != m
            {
                s2_spot_div += 1;
            }
            tune_19[i * E19 + 15 + ord] = m;
        }
    }
    if s2_spot_div != 0 {
        return Err(fail(format!(
            "DEV-tune S2 multi GPU-vs-CPU spot-check failed: div={s2_spot_div}"
        )));
    }
    let (tune_mix_em_s2, _tg2, _tt2, _tbm2) = gpu_or_cpu_em_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E19,
        &tune_buckets,
        &tune_19,
        &w_em_s2.weights_q32,
        CUDA_CHUNK,
    )?;
    std::fs::write(
        out_dir.join("p3c_a1s_em_s2_multi_brier.uors6b01"),
        w_em_s2.encode_bytes()?,
    )?;

    // Arm 4: p3c_a1s_brier_s2_multi (19 experts, A1s warm-start + conservative Brier S6 + GPU S3 & S2(1..=4)).
    reset_dynamic_tables(
        &mut gs_table,
        &mut s2_multi,
        &mut r_19,
        &mut tune_19,
        #[cfg(feature = "cuda")]
        &mut gpu_gs,
        #[cfg(feature = "cuda")]
        &mut gpu_s2s,
    )?;
    let mut w_brier_s2 = init_a1s_warmstart_weights(
        &a1s_weights,
        S6_BUCKETS,
        E19,
        WARM_EXTRA_SHARE_Q32,
        r_uort_sha,
    )?;
    let (br_s2_init_m, _, _, mut br_s2_prev_mb) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E19,
        &r_buckets,
        &r_19,
        &w_brier_s2.weights_q32,
        CUDA_CHUNK,
    )?;
    let mut br_s2_prev_bits = total_bits_q12(&br_s2_init_m)?;
    drop(br_s2_init_m);
    let mut br_s2_epochs = Vec::new();
    for ep in 0..3u32 {
        let mb_before = br_s2_prev_mb;
        let bits_before = br_s2_prev_bits;
        let mut buckets_upd = 0usize;
        let mut pos = 0usize;
        while pos < R_TARGETS {
            let end = (pos + MACRO_BATCH).min(R_TARGETS);
            let (_bmix, grads, touched, _mbb) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                E19,
                &r_buckets[pos..end],
                &r_19[pos * E19..end * E19],
                &w_brier_s2.weights_q32,
                CUDA_CHUNK,
            )?;
            buckets_upd += step_s6_brier_weights(
                &mut w_brier_s2.weights_q32,
                S6_BUCKETS,
                E19,
                &grads,
                &touched,
                7,
            )?;
            pos = end;
        }
        let (rmix_post, _gp, _tp, _mbp) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E19,
            &r_buckets,
            &r_19,
            &w_brier_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let (s3_upd, s2_upds) = {
            let (_m3, pd, ph, bd, bh) =
                gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, Some(&rmix_post), 9)?;
            let su3 = gs_table.apply_zero_centered_deltas(&pd, &ph, &bd, &bh);
            gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
            let (r3u, ..) = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 9)?;
            for (i, &m) in r3u.iter().enumerate() {
                r_19[i * E19 + 14] = m;
            }
            let mut su2 = [0usize; 4];
            for ord in 0..4usize {
                let (_m2, sd, sh) = gpu_s2s[ord].lookup_and_accum(
                    &r_s2keys[ord],
                    &r_targets,
                    Some(&rmix_post),
                    9,
                )?;
                su2[ord] = s2_multi.orders[ord].apply_zero_centered_deltas(&sd, &sh);
                gpu_s2s[ord].update_errs(&s2_multi.orders[ord].errs())?;
                let (r2u, ..) =
                    gpu_s2s[ord].lookup_and_accum(&r_s2keys[ord], &r_targets, None, 9)?;
                for (i, &m) in r2u.iter().enumerate() {
                    r_19[i * E19 + 15 + ord] = m;
                }
            }
            (su3, su2)
        };
        #[cfg(not(feature = "cuda"))]
        let (s3_upd, s2_upds) = {
            let su3 = gs_table.apply_epoch_brier_error_writes(&r_gkeys, &r_targets, &rmix_post, 9);
            for i in 0..R_TARGETS {
                r_19[i * E19 + 14] = gs_table.lookup_modulated_q48(r_gkeys[i], r_targets[i]).0;
            }
            let mut su2 = [0usize; 4];
            for ord in 0..4usize {
                su2[ord] = s2_multi.orders[ord].apply_epoch_brier_error_writes(
                    &r_s2keys[ord],
                    &r_targets,
                    &rmix_post,
                    9,
                );
                for i in 0..R_TARGETS {
                    r_19[i * E19 + 15 + ord] = s2_multi.orders[ord]
                        .lookup_modulated_q48(r_s2keys[ord][i], r_targets[i])
                        .0;
                }
            }
            (su3, su2)
        };
        drop(rmix_post);
        let (rmix2, _ga, _ta, mb_after) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            E19,
            &r_buckets,
            &r_19,
            &w_brier_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        let bits_after = total_bits_q12(&rmix2)?;
        drop(rmix2);
        br_s2_prev_mb = mb_after;
        br_s2_prev_bits = bits_after;
        w_brier_s2.epoch = ep + 1;
        w_brier_s2.train_brier_q32_mean = mb_after;
        br_s2_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "train_bits_before_q12": bits_before.to_string(),
            "train_bits_after_q12": bits_after.to_string(),
            "buckets_updated": buckets_upd,
            "s3_cells_updated": s3_upd,
            "s2_cells_updated_by_order": s2_upds,
            "brier_fell": mb_after < mb_before,
            "bits_fell": bits_after < bits_before,
        }));
    }
    drop(r_19);
    #[cfg(feature = "cuda")]
    let tune_s3_br2 = gpu_gs
        .lookup_and_accum(&tune_gkeys, &tune_targets, None, 8)?
        .0;
    #[cfg(not(feature = "cuda"))]
    let tune_s3_br2: Vec<u64> = tune_gkeys
        .iter()
        .zip(&tune_targets)
        .map(|(&k, &t)| gs_table.lookup_modulated_q48(k, t).0)
        .collect();
    for (i, &m) in tune_s3_br2.iter().enumerate() {
        tune_19[i * E19 + 14] = m;
    }
    drop(tune_s3_br2);
    for ord in 0..4usize {
        #[cfg(feature = "cuda")]
        let ts2 = gpu_s2s[ord]
            .lookup_and_accum(&tune_s2keys[ord], &tune_targets, None, 8)?
            .0;
        #[cfg(not(feature = "cuda"))]
        let ts2: Vec<u64> = tune_s2keys[ord]
            .iter()
            .zip(&tune_targets)
            .map(|(&k, &t)| s2_multi.orders[ord].lookup_modulated_q48(k, t).0)
            .collect();
        for (i, &m) in ts2.iter().enumerate() {
            tune_19[i * E19 + 15 + ord] = m;
        }
    }
    let (tune_mix_brier_s2, _tg4, _tt4, _tbm4) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        E19,
        &tune_buckets,
        &tune_19,
        &w_brier_s2.weights_q32,
        CUDA_CHUNK,
    )?;
    drop(tune_19);
    std::fs::write(
        out_dir.join("p3c_a1s_brier_s2_multi.uors6b01"),
        w_brier_s2.encode_bytes()?,
    )?;

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

    let s2_meta = json!({
        "orders": s2_multi.orders.iter().map(|o| o.order()).collect::<Vec<_>>(),
        "per_order_bytes": s2_multi.per_order_bytes(),
        "per_order_ctxs": s2_multi.per_order_ctxs(),
        "per_order_entries": s2_multi.per_order_entries(),
        "total_bytes": s2_multi.total_encoded_bytes(),
        "max_bytes_budget": 100_000_000usize,
    });
    let report = json!({
        "phase": "Stage B3c P3 DEV-tune A1s-warm-start S6 + EM + multi-order S2",
        "dev_tune_targets": DEV_TUNE_TARGETS,
        "p1_a1s_baseline": {
            "all": base_all.json(),
            "novel": base_novel.json(),
            "response": base_resp.json(),
            "novel_response": base_nov_resp.json(),
            "rstar_devtune_bpb_micro": RSTAR_DEVTUNE_BPB_MICRO.to_string(),
        },
        "s2_multi_table": s2_meta,
        "spot_check_probes": spot_checked,
        "spot_check_divergences": spot_div,
        "em_spot_check_divergences": em_spot_div,
        "s3_spot_check_divergences": s3_spot_div,
        "s2_spot_check_divergences": s2_spot_div,
        "shifted_batch_parity": true,
        "arms": [
            summarize_arm("p3c_a1s_brier_s6", &tune_mix_s6, json!(s6_epochs), json!({"experts": E15, "warmstart": "a1s", "optimizer": "tangent_brier_lr7"}))?,
            summarize_arm("p3c_a1s_em_s3_brier", &tune_mix_em_s3, json!(em_s3_epochs), json!({"experts": E15, "warmstart": "a1s", "optimizer": "integer_em_prior64_s3_brier_lr9", "golden_shell_sha": GOLDEN_SHELL_SHA256}))?,
            summarize_arm("p3c_a1s_em_s2_multi_brier", &tune_mix_em_s2, json!(em_s2_epochs), json!({"experts": E19, "warmstart": "a1s", "optimizer": "integer_em_prior64_s3_s2_multi_lr9", "s2_total_bytes": s2_multi.total_encoded_bytes()}))?,
            summarize_arm("p3c_a1s_brier_s2_multi", &tune_mix_brier_s2, json!(br_s2_epochs), json!({"experts": E19, "warmstart": "a1s", "optimizer": "tangent_brier_lr7_s3_s2_multi_lr9", "s2_total_bytes": s2_multi.total_encoded_bytes()}))?,
        ],
    });
    std::fs::write(out_json_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_shell_modulated_distribution_stays_normalized_and_devtest_refused() {
        let tmp = std::env::temp_dir().join(format!("p3-brier-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dev-tune.json"), b"{\"split\":\"dev-test\"}").unwrap();
        assert!(DevTuneData::load_checked(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn brier_s6_step_s2_normalization_and_paired_ci_work() {
        let mut weights = vec![P3_Q32_ONE / 4; 4];
        let grads = vec![P3_Q32_ONE as i128, -(P3_Q32_ONE as i128), 0, 0];
        let touched = vec![10u64];
        let upd = step_s6_brier_weights(&mut weights, 1, 4, &grads, &touched, 6).unwrap();
        assert_eq!(upd, 1);
        assert_eq!(weights.iter().copied().sum::<u64>(), P3_Q32_ONE);
        assert!(weights[0] > weights[1]);

        let a1s_mock = {
            let mut v = vec![1u64 << 24; 14];
            v[0] = P3_Q32_ONE - 13 * (1u64 << 24);
            v
        };
        let ws_s6 =
            init_a1s_warmstart_weights(&a1s_mock, 1, 19, P3_Q32_ONE >> 7, [0u8; 32]).unwrap();
        assert_eq!(ws_s6.weights_q32.iter().copied().sum::<u64>(), P3_Q32_ONE);
        assert!(ws_s6.weights_q32[0] > (P3_Q32_ONE * 9) / 10);

        let mut s2_multi =
            S2MultiOrderErrorTable::build_from_r_stream(&[0, 5, 9, 10, 5, 9, 10, 5, 9, 11], 256)
                .unwrap();
        let k2 = crate::s3_decode::pack_s2_key(2, 9, 5, 0, 0);
        let (mass_hit, _) = s2_multi.orders[1].lookup_modulated_q48(k2, 10);
        let (mass_rival, _) = s2_multi.orders[1].lookup_modulated_q48(k2, 11);
        assert!(mass_hit > 0 && mass_hit <= S3_Q48_ONE);
        assert!(mass_hit + mass_rival <= S3_Q48_ONE);
        let upd_s2 = s2_multi.orders[1].apply_epoch_brier_error_writes(
            &[k2, k2],
            &[10, 10],
            &[mass_hit, mass_hit],
            8,
        );
        assert!(upd_s2 >= 1);
        let (mass_hit2, _) = s2_multi.orders[1].lookup_modulated_q48(k2, 10);
        let (mass_rival2, _) = s2_multi.orders[1].lookup_modulated_q48(k2, 11);
        assert!(mass_hit2 > mass_hit);
        assert!(mass_rival2 < mass_rival);
        assert!(mass_hit2 + mass_rival2 <= S3_Q48_ONE);

        // Verify both conservative Brier LMS and pure-integer EM refit drop metrics on 19-expert stream.
        #[cfg(not(feature = "cuda"))]
        {
            let mut w_em = ws_s6.weights_q32.clone();
            let rows_syn = 128usize;
            let bkts = vec![0u16; rows_syn];
            let mut masses = vec![S3_Q48_ONE / 64; rows_syn * 19];
            for r in 0..rows_syn {
                if r % 3 != 0 {
                    masses[r * 19] = S3_Q48_ONE / 3;
                } else {
                    masses[r * 19 + 15] = S3_Q48_ONE / 2;
                }
            }
            let (m0, _, _, mut prev_mb_em) =
                gpu_or_cpu_em_epoch(1, 19, &bkts, &masses, &w_em, 64).unwrap();
            let mut prev_bits = total_bits_q12(&m0).unwrap();
            for _ep in 0..3 {
                let (_, resp, touched, _) =
                    gpu_or_cpu_em_epoch(1, 19, &bkts, &masses, &w_em, 32).unwrap();
                step_s6_em_weights(&mut w_em, 1, 19, &resp, &touched, 64).unwrap();
                let (ma, _, _, after_mb) =
                    gpu_or_cpu_em_epoch(1, 19, &bkts, &masses, &w_em, 64).unwrap();
                let after_bits = total_bits_q12(&ma).unwrap();
                assert!(after_mb < prev_mb_em);
                assert!(after_bits < prev_bits);
                prev_mb_em = after_mb;
                prev_bits = after_bits;
            }

            let mut w_syn = vec![0u64; 15];
            let q = P3_Q32_ONE / 15;
            let rem = (P3_Q32_ONE % 15) as usize;
            for (e, slot) in w_syn.iter_mut().enumerate().take(15) {
                *slot = q + if e < rem { 1 } else { 0 };
            }
            let mut masses15 = vec![S3_Q48_ONE / 64; rows_syn * 15];
            for r in 0..rows_syn {
                if r % 3 != 0 {
                    masses15[r * 15] = S3_Q48_ONE / 3;
                } else {
                    masses15[r * 15 + 1] = S3_Q48_ONE / 3;
                }
            }
            let (_, _, _, mut prev_mb) =
                gpu_or_cpu_eval_epoch(1, 15, &bkts, &masses15, &w_syn, 64).unwrap();
            for _ep in 0..3 {
                for mb in bkts.chunks(32).enumerate() {
                    let s = mb.0 * 32;
                    let e = s + mb.1.len();
                    let (_, g, t, _) = gpu_or_cpu_eval_epoch(
                        1,
                        15,
                        &bkts[s..e],
                        &masses15[s * 15..e * 15],
                        &w_syn,
                        32,
                    )
                    .unwrap();
                    step_s6_brier_weights(&mut w_syn, 1, 15, &g, &t, 7).unwrap();
                }
                let (_, _, _, after_mb) =
                    gpu_or_cpu_eval_epoch(1, 15, &bkts, &masses15, &w_syn, 64).unwrap();
                assert!(
                    after_mb < prev_mb,
                    "synthetic Brier did not drop: {after_mb} >= {prev_mb}"
                );
                prev_mb = after_mb;
            }
        }

        let x = vec![100u128; 20];
        let y = vec![120u128; 20];
        let ci = paired_record_ci(&x, &y, 200, 4390).unwrap();
        assert_eq!(ci, [-400, -400]);
    }
}
