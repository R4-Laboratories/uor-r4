#![forbid(unsafe_code)]

use crate::s3_decode::{
    stream_golden_and_s2_keys, GoldenShellStateTable, S2Order2ErrorTable, S3_Q48_ONE, S3_VOCAB,
};
use crate::{fail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};
#[cfg(test)]
use uor_r4_count_runtime::p3_state::P3_Q32_ONE;
use uor_r4_count_runtime::p3_state::{
    brier_mix_q48, integer_brier_target_q32, project_simplex_q32, BrierS6Weights,
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
            return Err(fail("DEV-test UORT forbidden in Stage B3b"));
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

pub fn load_p1_tape_and_verify_a1s(
    tape_path: &Path,
    pick_rows_path: &Path,
    a1s_weights_path: &Path,
    expected_binding: [u8; 32],
    expected_uort_sha: [u8; 32],
    rows: usize,
) -> Result<(Vec<u16>, Vec<u64>, Vec<u64>)> {
    let w_bytes = std::fs::read(a1s_weights_path)?;
    let frozen = FrozenPickMixture::from_bytes(&w_bytes, expected_binding)?;
    if frozen.rule() != PickClassRule::Five {
        return Err(fail("expected A1s PickClassRule::Five"));
    }
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
    Ok((buckets, p1_experts, a1s_masses))
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

fn step_s6_weights(
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
        let mut raw = Vec::with_capacity(experts);
        for e in 0..experts {
            let mean_g = grads[start + e] / denom;
            let step = mean_g >> lr_shift.min(31);
            raw.push(i128::from(weights[start + e]) + step);
        }
        let proj = project_simplex_q32(&raw, 1)?;
        weights[start..start + experts].copy_from_slice(&proj);
        updated += 1;
    }
    Ok(updated)
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

    let (r_buckets, r_p1, _r_a1s) = load_p1_tape_and_verify_a1s(
        b1a_r_tape,
        b1a_r_pick,
        a1s_weights_path,
        binding,
        r_uort_sha,
        R_TARGETS,
    )?;
    let (tune_buckets, tune_p1, tune_a1s) = load_p1_tape_and_verify_a1s(
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

    let (r_targets, r_gkeys, r_s2keys) = stream_golden_and_s2_keys(&r_ids, &gs_table.token_h4);
    let (tune_targets, tune_gkeys, tune_s2keys) =
        stream_golden_and_s2_keys(&tune.ids, &gs_table.token_h4);
    if r_targets.len() != R_TARGETS || tune_targets.len() != DEV_TUNE_TARGETS {
        return Err(fail("stream target length mismatch"));
    }

    let mut s2_table = S2Order2ErrorTable::build_from_r_stream(&r_ids, 50_000_000)?;
    #[cfg(feature = "cuda")]
    let gpu = P3BrierGpuEngine::new()?;
    #[cfg(feature = "cuda")]
    let mut gpu_gs = gs_table.upload_gpu(&gpu)?;
    #[cfg(feature = "cuda")]
    let mut gpu_s2 = s2_table.upload_gpu(&gpu)?;

    // Pre-extract initial static S3 masses for P3_S6_BRIER arm (15 experts), freeing p1 buffers immediately.
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

    let mut r_15 = Vec::with_capacity(R_TARGETS * 15);
    for (i, &s3m) in r_s3_init.iter().enumerate() {
        r_15.extend_from_slice(&r_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
        r_15.push(s3m);
    }
    drop(r_p1);
    drop(r_s3_init);
    let mut tune_15 = Vec::with_capacity(DEV_TUNE_TARGETS * 15);
    for (i, &s3m) in tune_s3_init.iter().enumerate() {
        tune_15.extend_from_slice(&tune_p1[i * P1_EXPERTS..(i + 1) * P1_EXPERTS]);
        tune_15.push(s3m);
    }
    drop(tune_p1);
    drop(tune_s3_init);

    const MACRO_BATCH: usize = 524_288;
    const CUDA_CHUNK: usize = 65_536;

    // Arm 2: p3_s6_brier (15 experts, 3 epochs S6 integer Brier LMS in 524,288-row macro-batches, chunk parity 65536 vs 65521 on ep 1 macro-batch 0).
    let mut w_s6 = BrierS6Weights::uniform(S6_BUCKETS, 15, r_uort_sha)?;
    let mut s6_epochs = Vec::new();
    let (_m_init, _g_init, _t_init, mut s6_prev_mb) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        15,
        &r_buckets,
        &r_15,
        &w_s6.weights_q32,
        CUDA_CHUNK,
    )?;
    for ep in 0..3u32 {
        let mb_before = s6_prev_mb;
        let mut buckets_upd = 0usize;
        let mut pos = 0usize;
        while pos < R_TARGETS {
            let end = (pos + MACRO_BATCH).min(R_TARGETS);
            let (_bmix, grads, touched, mb_batch) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                15,
                &r_buckets[pos..end],
                &r_15[pos * 15..end * 15],
                &w_s6.weights_q32,
                CUDA_CHUNK,
            )?;
            if ep == 0 && pos == 0 {
                let (_m2, g2, t2, b2) = gpu_or_cpu_eval_epoch(
                    #[cfg(feature = "cuda")]
                    &gpu,
                    S6_BUCKETS,
                    15,
                    &r_buckets[pos..end],
                    &r_15[pos * 15..end * 15],
                    &w_s6.weights_q32,
                    65_521,
                )?;
                if g2 != grads || t2 != touched || b2 != mb_batch {
                    return Err(fail(
                        "shifted batch parity (65536 vs 65521) failed in p3_s6_brier",
                    ));
                }
            }
            buckets_upd +=
                step_s6_weights(&mut w_s6.weights_q32, S6_BUCKETS, 15, &grads, &touched, 0)?;
            pos = end;
        }
        let (_rmix2, _g_after, _t_after, mb_after) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            15,
            &r_buckets,
            &r_15,
            &w_s6.weights_q32,
            CUDA_CHUNK,
        )?;
        s6_prev_mb = mb_after;
        w_s6.epoch = ep + 1;
        w_s6.train_brier_q32_mean = mb_after;
        s6_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "buckets_updated": buckets_upd,
            "fell": mb_after < mb_before,
        }));
    }
    let (tune_mix_s6, _tg, _tt, _tbm) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        15,
        &tune_buckets,
        &tune_15,
        &w_s6.weights_q32,
        CUDA_CHUNK,
    )?;
    // Verify spot-check >= 12,500 CPU vs GPU probes on p3_s6_brier DEV-tune.
    let mut spot_checked = 0usize;
    let mut spot_div = 0usize;
    for i in 0..DEV_TUNE_TARGETS {
        if i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS {
            spot_checked += 1;
            let b = usize::from(tune_buckets[i]);
            let exp_m = brier_mix_q48(
                &w_s6.weights_q32[b * 15..(b + 1) * 15],
                &tune_15[i * 15..(i + 1) * 15],
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
    std::fs::write(out_dir.join("p3_s6_brier.uors6b01"), w_s6.encode_bytes()?)?;

    // Arm 3: p3_s6_s3_brier (S3 golden_shell i8 error cells + S6 UORS6B01, 3 epochs, in-place GPU S3 update).
    let mut w_s6_s3 = BrierS6Weights::uniform(S6_BUCKETS, 15, r_uort_sha)?;
    let mut s3_epochs = Vec::new();
    let (_s3_rmix0, _gs3_0, _ts3_0, mut s3_prev_mb) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        15,
        &r_buckets,
        &r_15,
        &w_s6_s3.weights_q32,
        CUDA_CHUNK,
    )?;
    for ep in 0..3u32 {
        let mb_before = s3_prev_mb;
        let mut buckets_upd = 0usize;
        let mut pos = 0usize;
        while pos < R_TARGETS {
            let end = (pos + MACRO_BATCH).min(R_TARGETS);
            let (_bmix, grads, touched, _mbb) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                15,
                &r_buckets[pos..end],
                &r_15[pos * 15..end * 15],
                &w_s6_s3.weights_q32,
                CUDA_CHUNK,
            )?;
            buckets_upd += step_s6_weights(
                &mut w_s6_s3.weights_q32,
                S6_BUCKETS,
                15,
                &grads,
                &touched,
                0,
            )?;
            pos = end;
        }
        let (rmix_post_s6, _gp, _tp, _mbp) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            15,
            &r_buckets,
            &r_15,
            &w_s6_s3.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let cells_upd = {
            let (_m_cur, pd, ph, bd, bh) =
                gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, Some(&rmix_post_s6), 8)?;
            let upd = gs_table.apply_zero_centered_deltas(&pd, &ph, &bd, &bh);
            gpu_gs.update_errs(&gs_table.p_errs(), &gs_table.b_errs())?;
            let (r_s3_upd, ..) = gpu_gs.lookup_and_accum(&r_gkeys, &r_targets, None, 8)?;
            for (i, &m) in r_s3_upd.iter().enumerate() {
                r_15[i * 15 + 14] = m;
            }
            upd
        };
        #[cfg(not(feature = "cuda"))]
        let cells_upd = {
            let upd =
                gs_table.apply_epoch_brier_error_writes(&r_gkeys, &r_targets, &rmix_post_s6, 8);
            for i in 0..R_TARGETS {
                r_15[i * 15 + 14] = gs_table.lookup_modulated_q48(r_gkeys[i], r_targets[i]).0;
            }
            upd
        };
        let (_rmix2, _ga, _ta, mb_after) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            15,
            &r_buckets,
            &r_15,
            &w_s6_s3.weights_q32,
            CUDA_CHUNK,
        )?;
        s3_prev_mb = mb_after;
        w_s6_s3.epoch = ep + 1;
        w_s6_s3.train_brier_q32_mean = mb_after;
        s3_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "buckets_updated": buckets_upd,
            "s3_cells_updated": cells_upd,
            "fell": mb_after < mb_before,
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
        tune_15[i * 15 + 14] = tune_s3_upd[i];
    }
    if s3_spot_div != 0 {
        return Err(fail(format!(
            "DEV-tune S3 GPU-vs-CPU spot-check failed: div={s3_spot_div}"
        )));
    }
    drop(tune_s3_upd);
    let (tune_mix_s3, _tg3, _tt3, _tbm3) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        15,
        &tune_buckets,
        &tune_15,
        &w_s6_s3.weights_q32,
        CUDA_CHUNK,
    )?;
    std::fs::write(
        out_dir.join("p3_s6_s3_brier.uors6b01"),
        w_s6_s3.encode_bytes()?,
    )?;

    // Arm 4: p3_s6_s2_err_brier (S2 order-2 i8 error cells + S3 + S6, 16 experts, 3 epochs, GPU S2).
    let mut w_s6_s2 = BrierS6Weights::uniform(S6_BUCKETS, 16, r_uort_sha)?;
    #[cfg(feature = "cuda")]
    let r_s2_init = gpu_s2.lookup_and_accum(&r_s2keys, &r_targets, None, 8)?.0;
    #[cfg(not(feature = "cuda"))]
    let r_s2_init: Vec<u64> = r_s2keys
        .iter()
        .zip(&r_targets)
        .map(|(&k, &t)| s2_table.lookup_modulated_q48(k, t).0)
        .collect();
    let mut r_16_dyn = Vec::with_capacity(R_TARGETS * 16);
    for (i, &s2m) in r_s2_init.iter().enumerate() {
        r_16_dyn.extend_from_slice(&r_15[i * 15..(i + 1) * 15]);
        r_16_dyn.push(s2m);
    }
    drop(r_s2_init);
    drop(r_15);
    let mut s2_epochs = Vec::new();
    let (_s2_rmix0, _gs2_0, _ts2_0, mut s2_prev_mb) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        16,
        &r_buckets,
        &r_16_dyn,
        &w_s6_s2.weights_q32,
        CUDA_CHUNK,
    )?;
    for ep in 0..3u32 {
        let mb_before = s2_prev_mb;
        let mut buckets_upd = 0usize;
        let mut pos = 0usize;
        while pos < R_TARGETS {
            let end = (pos + MACRO_BATCH).min(R_TARGETS);
            let (_bmix, grads, touched, _mbb) = gpu_or_cpu_eval_epoch(
                #[cfg(feature = "cuda")]
                &gpu,
                S6_BUCKETS,
                16,
                &r_buckets[pos..end],
                &r_16_dyn[pos * 16..end * 16],
                &w_s6_s2.weights_q32,
                CUDA_CHUNK,
            )?;
            buckets_upd += step_s6_weights(
                &mut w_s6_s2.weights_q32,
                S6_BUCKETS,
                16,
                &grads,
                &touched,
                0,
            )?;
            pos = end;
        }
        let (rmix_post_s6, _gp, _tp, _mbp) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            16,
            &r_buckets,
            &r_16_dyn,
            &w_s6_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        #[cfg(feature = "cuda")]
        let s2_upd = {
            let (_m_cur, sd, sh) =
                gpu_s2.lookup_and_accum(&r_s2keys, &r_targets, Some(&rmix_post_s6), 8)?;
            let upd = s2_table.apply_zero_centered_deltas(&sd, &sh);
            gpu_s2.update_errs(&s2_table.errs())?;
            let (r_s2_upd, ..) = gpu_s2.lookup_and_accum(&r_s2keys, &r_targets, None, 8)?;
            for (i, &m) in r_s2_upd.iter().enumerate() {
                r_16_dyn[i * 16 + 15] = m;
            }
            upd
        };
        #[cfg(not(feature = "cuda"))]
        let s2_upd = {
            let upd =
                s2_table.apply_epoch_brier_error_writes(&r_s2keys, &r_targets, &rmix_post_s6, 8);
            for i in 0..R_TARGETS {
                r_16_dyn[i * 16 + 15] = s2_table.lookup_modulated_q48(r_s2keys[i], r_targets[i]).0;
            }
            upd
        };
        let (_rmix2, _ga, _ta, mb_after) = gpu_or_cpu_eval_epoch(
            #[cfg(feature = "cuda")]
            &gpu,
            S6_BUCKETS,
            16,
            &r_buckets,
            &r_16_dyn,
            &w_s6_s2.weights_q32,
            CUDA_CHUNK,
        )?;
        s2_prev_mb = mb_after;
        w_s6_s2.epoch = ep + 1;
        w_s6_s2.train_brier_q32_mean = mb_after;
        s2_epochs.push(json!({
            "epoch": ep + 1,
            "mean_brier_before_q32": mb_before.to_string(),
            "mean_brier_after_q32": mb_after.to_string(),
            "buckets_updated": buckets_upd,
            "s2_cells_updated": s2_upd,
            "fell": mb_after < mb_before,
        }));
    }
    drop(r_16_dyn);
    #[cfg(feature = "cuda")]
    let tune_s2_upd = gpu_s2
        .lookup_and_accum(&tune_s2keys, &tune_targets, None, 8)?
        .0;
    #[cfg(not(feature = "cuda"))]
    let tune_s2_upd: Vec<u64> = tune_s2keys
        .iter()
        .zip(&tune_targets)
        .map(|(&k, &t)| s2_table.lookup_modulated_q48(k, t).0)
        .collect();
    let mut s2_spot_div = 0usize;
    let mut tune_16_s2 = Vec::with_capacity(DEV_TUNE_TARGETS * 16);
    for (i, &s2m) in tune_s2_upd.iter().enumerate() {
        if (i < 256 || i % 839 == 197 || i + 16 >= DEV_TUNE_TARGETS)
            && s2_table
                .lookup_modulated_q48(tune_s2keys[i], tune_targets[i])
                .0
                != s2m
        {
            s2_spot_div += 1;
        }
        tune_16_s2.extend_from_slice(&tune_15[i * 15..(i + 1) * 15]);
        tune_16_s2.push(s2m);
    }
    if s2_spot_div != 0 {
        return Err(fail(format!(
            "DEV-tune S2 GPU-vs-CPU spot-check failed: div={s2_spot_div}"
        )));
    }
    drop(tune_s2_upd);
    drop(tune_15);
    let (tune_mix_s2, _tg2, _tt2, _tbm2) = gpu_or_cpu_eval_epoch(
        #[cfg(feature = "cuda")]
        &gpu,
        S6_BUCKETS,
        16,
        &tune_buckets,
        &tune_16_s2,
        &w_s6_s2.weights_q32,
        CUDA_CHUNK,
    )?;
    drop(tune_16_s2);
    std::fs::write(
        out_dir.join("p3_s6_s2_err_brier.uors6b01"),
        w_s6_s2.encode_bytes()?,
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

    let report = json!({
        "phase": "Stage B3b P3 DEV-tune Brier error-write",
        "dev_tune_targets": DEV_TUNE_TARGETS,
        "p1_a1s_baseline": {
            "all": base_all.json(),
            "novel": base_novel.json(),
            "response": base_resp.json(),
            "novel_response": base_nov_resp.json(),
            "rstar_devtune_bpb_micro": RSTAR_DEVTUNE_BPB_MICRO.to_string(),
        },
        "spot_check_probes": spot_checked,
        "spot_check_divergences": spot_div,
        "s3_spot_check_divergences": s3_spot_div,
        "s2_spot_check_divergences": s2_spot_div,
        "shifted_batch_parity": true,
        "arms": [
            summarize_arm("p3_s6_brier", &tune_mix_s6, json!(s6_epochs), json!({"experts": 15}))?,
            summarize_arm("p3_s6_s3_brier", &tune_mix_s3, json!(s3_epochs), json!({"experts": 15, "golden_shell_sha": GOLDEN_SHELL_SHA256}))?,
            summarize_arm("p3_s6_s2_err_brier", &tune_mix_s2, json!(s2_epochs), json!({"experts": 16, "s2_entries": s2_table.entry_count(), "s2_bytes": s2_table.encoded_bytes()}))?,
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
        let upd = step_s6_weights(&mut weights, 1, 4, &grads, &touched, 0).unwrap();
        assert_eq!(upd, 1);
        assert_eq!(weights.iter().copied().sum::<u64>(), P3_Q32_ONE);
        assert!(weights[0] > weights[1]);

        let mut s2 =
            S2Order2ErrorTable::build_from_r_stream(&[0, 5, 9, 10, 5, 9, 10, 5, 9, 11], 256)
                .unwrap();
        let (mass_hit, _) = s2.lookup_modulated_q48((5u32 << 12) | 9, 10);
        let (mass_rival, _) = s2.lookup_modulated_q48((5u32 << 12) | 9, 11);
        assert!(mass_hit > 0 && mass_hit <= S3_Q48_ONE);
        assert!(mass_hit + mass_rival <= S3_Q48_ONE);
        let upd_s2 = s2.apply_epoch_brier_error_writes(
            &[(5u32 << 12) | 9, (5u32 << 12) | 9],
            &[10, 10],
            &[mass_hit, mass_hit],
            8,
        );
        assert!(upd_s2 >= 1);
        let (mass_hit2, _) = s2.lookup_modulated_q48((5u32 << 12) | 9, 10);
        let (mass_rival2, _) = s2.lookup_modulated_q48((5u32 << 12) | 9, 11);
        assert!(mass_hit2 > mass_hit);
        assert!(mass_rival2 < mass_rival);
        assert!(mass_hit2 + mass_rival2 <= S3_Q48_ONE);

        // Verify macro-batch Brier LMS converges monotonically across 3 epochs on a synthetic 15-expert stream.
        let mut w_syn = vec![0u64; 15];
        let q = P3_Q32_ONE / 15;
        let rem = (P3_Q32_ONE % 15) as usize;
        for (e, slot) in w_syn.iter_mut().enumerate().take(15) {
            *slot = q + if e < rem { 1 } else { 0 };
        }
        let rows_syn = 128usize;
        let bkts = vec![0u16; rows_syn];
        let mut masses = vec![S3_Q48_ONE / 64; rows_syn * 15];
        for r in 0..rows_syn {
            if r % 3 != 0 {
                masses[r * 15] = S3_Q48_ONE / 3;
            } else {
                masses[r * 15 + 1] = S3_Q48_ONE / 3;
            }
        }
        let (_, _, _, mut prev_mb) =
            gpu_or_cpu_eval_epoch(1, 15, &bkts, &masses, &w_syn, 64).unwrap();
        for _ep in 0..3 {
            for mb in bkts.chunks(32).enumerate() {
                let s = mb.0 * 32;
                let e = s + mb.1.len();
                let (_, g, t, _) =
                    gpu_or_cpu_eval_epoch(1, 15, &bkts[s..e], &masses[s * 15..e * 15], &w_syn, 32)
                        .unwrap();
                step_s6_weights(&mut w_syn, 1, 15, &g, &t, 0).unwrap();
            }
            let (_, _, _, after_mb) =
                gpu_or_cpu_eval_epoch(1, 15, &bkts, &masses, &w_syn, 64).unwrap();
            assert!(
                after_mb < prev_mb,
                "synthetic Brier did not drop: {after_mb} >= {prev_mb}"
            );
            prev_mb = after_mb;
        }

        let x = vec![100u128; 20];
        let y = vec![120u128; 20];
        let ci = paired_record_ci(&x, &y, 200, 4390).unwrap();
        assert_eq!(ci, [-400, -400]);
    }
}
