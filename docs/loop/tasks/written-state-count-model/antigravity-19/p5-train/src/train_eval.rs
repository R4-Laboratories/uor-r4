#![forbid(unsafe_code)]

use crate::{fail, verify_file_sha256, Result};
#[cfg(feature = "cuda")]
use crate::{lawful_likelihood::bits_q12, load_uort_tokens};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use std::time::Instant;
use std::{fs, path::Path};
#[cfg(feature = "cuda")]
use uor_r4_count_runtime::p4_serving::{P4Session, P4_MAX_CONTEXT, P4_VOCAB};
use uor_r4_count_runtime::FrozenPickMixture;
#[cfg(feature = "cuda")]
use uor_r4_discrete::p5_brier_core::{
    extract_s1_from_uorlut01, full_vocab_brier_q32, p5_random_block_coords, p5_schedule_block,
    P5_BLOCKS, P5_S1_COORDS, P5_VOCAB,
};
use uor_r4_discrete::p5_brier_core::{
    p5_orbit_tables, rewrite_uorlut01_s1, P5BlockRecord, P5CoordScore, P5S1State, P5_CANDIDATES,
    P5_MAX_BATCH_ROWS, P5_MAX_K,
};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, write_p4_deployment, ClientArtifactFile, DiscreteCountModel,
};

#[cfg(feature = "cuda")]
use crate::cuda::{P5GpuEngine, P5_MAX_SLOTS};
use crate::cuda::{split_uorp4a01_sections, RowStepCmdHost};

// ---------- Frozen Pinned SHA-256 Digests (verified fail-closed before compute) ----------

pub const PIN_P4_D11_SHA256: &str =
    "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
pub const PIN_SEED1_D11_SHA256: &str =
    "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const PIN_TOKENIZER_SHA256: &str =
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
pub const PIN_R_UORT_SHA256: &str =
    "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub const PIN_DEV_TUNE_UORT_SHA256: &str =
    "75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a";
pub const PIN_DEV_TEST_CENSUS_SHA256: &str =
    "ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2";
pub const PIN_DEV_TEST_UORT_SHA256: &str =
    "023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b";
pub const PIN_DEV_TEST_NOVEL_SHA256: &str =
    "5bf3881bff8fb5abe5c15da2b6dfdfe60f4eaa709ee061d9d0405d618af10471";
pub const PIN_ID_BYTES_SHA256: &str =
    "befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9";
pub const PIN_RSTAR_MASSES_SHA256: &str =
    "59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de";

pub const R_EXPECTED_RECORDS: usize = 4_060;
pub const R_EXPECTED_TOKENS: usize = 15_951_644;
pub const R_EXPECTED_TARGETS: usize = 15_947_584;

pub const DEV_TUNE_EXPECTED_RECORDS: usize = 2_670;
pub const DEV_TUNE_EXPECTED_TOKENS: usize = 10_810_087;
pub const DEV_TUNE_EXPECTED_TARGETS: usize = 10_807_417;

pub const DEV_TEST_EXPECTED_RECORDS: usize = 5_321;
pub const DEV_TEST_EXPECTED_TOKENS: usize = 21_354_897;
pub const DEV_TEST_EXPECTED_TARGETS: usize = 21_349_576;

pub const RSTAR_DEVTEST_ALL_Q12: u128 = 251_672_950_854;
pub const RSTAR_DEVTEST_BPB_MICRO: u128 = 1_418_136;
pub const P1_A1S_DEVTEST_ALL_Q12: u128 = 246_099_676_775;

pub const SEED_RESEED_CANONICAL: u64 = 4390;
pub const SEED_RESEED_RANDOM: u64 = 4391;
pub const SEED_RANDOM_COORD_RNG: u64 = 4392;
pub const SEED_BOOTSTRAP_CI: u64 = 4393;

/// Convergence / saturation threshold on `DEV-tune` (`200` ppm = `0.02%` epoch-over-epoch)
/// used strictly on `DEV-tune` after `min_epochs = 3` (never peeking `DEV-test`).
pub const DEVTUNE_CONTINUE_EPOCH_PPM: u128 = 200;
pub const DEVTUNE_CONTINUE_TOTAL_PPM: u128 = 1_000;

pub const MAX_DEPLOYED_BYTES: u64 = 4_000_000_000;

/// Split a `UORT` token stream into contiguous causal records starting at `BOS = 0`.
pub fn split_records_on_bos(tokens: &[u16]) -> Result<Vec<(usize, usize)>> {
    if tokens.is_empty() || tokens[0] != 0 {
        return Err(fail("UORT token stream must begin with BOS = 0"));
    }
    let mut recs = Vec::new();
    let mut start = 0usize;
    for (i, &tok) in tokens.iter().enumerate().skip(1) {
        if tok == 0 {
            recs.push((start, i - start));
            start = i;
        }
    }
    recs.push((start, tokens.len() - start));
    Ok(recs)
}

/// Deterministic causal row batch scheduler that visits EVERY non-BOS target
/// `(rec_off, pos, target)` across all records in an epoch **exactly once** in canonical
/// record order, packing contiguous `<= 4,096`-row batches (`3,893 * 4,096 + 1,856 = 15,947,584`
/// on `R`) with zero sparse straggler tail batches.
pub struct RowBatchScheduler<'a> {
    tokens: &'a [u16],
    recs: &'a [(usize, usize)],
    batch_rows: usize,
    rec_idx: usize,
    pos: usize,
}

impl<'a> RowBatchScheduler<'a> {
    pub fn new(tokens: &'a [u16], recs: &'a [(usize, usize)], batch_rows: usize) -> Result<Self> {
        if batch_rows == 0 || batch_rows > P5_MAX_BATCH_ROWS {
            return Err(fail("invalid batch_rows for RowBatchScheduler"));
        }
        Ok(Self {
            tokens,
            recs,
            batch_rows,
            rec_idx: 0,
            pos: 1,
        })
    }

    pub fn next_batch(&mut self) -> Option<Vec<RowStepCmdHost>> {
        let mut cmds = Vec::with_capacity(self.batch_rows);
        while cmds.len() < self.batch_rows && self.rec_idx < self.recs.len() {
            let (ro, rl) = self.recs[self.rec_idx];
            if self.pos >= rl {
                self.rec_idx += 1;
                self.pos = 1;
                continue;
            }
            let tgt = self.tokens[ro + self.pos];
            if tgt != 0 {
                cmds.push(RowStepCmdHost {
                    rec_off: ro as u32,
                    pos: self.pos as u32,
                    target: tgt,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
            self.pos += 1;
        }
        if cmds.is_empty() {
            None
        } else {
            Some(cmds)
        }
    }
}

/// Pure-integer exact `BVG-K` choose/decide/apply logic given `rows-K` `scores`
/// and a callback `eval_joint(&scratch_roots) -> Result<u64>` invoked ONLY when `|S_t| >= 2`.
/// Matches `uor_r4_discrete::p5_brier_core::p5_bvgk_step_cpu` word-for-word.
pub fn bvgk_decide_and_apply<F>(
    state: &mut P5S1State,
    coords: &[u32],
    scores: &[P5CoordScore],
    mut eval_joint: F,
) -> Result<P5BlockRecord>
where
    F: FnMut(&[u8]) -> Result<u64>,
{
    state.validate()?;
    if scores.len() != coords.len() || coords.is_empty() || coords.len() > P5_MAX_K {
        return Err(fail("bvgk_decide_and_apply score/coord length mismatch"));
    }
    let (move_root, _) = p5_orbit_tables();
    let k = coords.len();
    let l_before = scores
        .iter()
        .find(|s| s.status == 0)
        .map(|s| s.losses[0])
        .ok_or_else(|| fail("all coordinates refused in P5 BVG-K scoring"))?;

    let mut chosen_lanes = vec![0u8; k];
    let mut members: Vec<usize> = Vec::new();
    let mut best_ordinal: Option<usize> = None;
    let mut best_loss = u64::MAX;

    for (c, score) in scores.iter().enumerate() {
        if score.coord != coords[c] {
            return Err(fail("bvgk_decide_and_apply coord mismatch"));
        }
        if score.status != 0 {
            continue;
        }
        let cur_root = usize::from(state.roots[score.coord as usize]);
        let mut best_lane = 0usize;
        let mut lane = 1usize;
        while lane < P5_CANDIDATES {
            let tie = best_lane != 0
                && score.losses[lane] == score.losses[best_lane]
                && move_root[cur_root][lane] < move_root[cur_root][best_lane];
            if score.losses[lane] < score.losses[best_lane] || tie {
                best_lane = lane;
            }
            lane += 1;
        }
        chosen_lanes[c] = best_lane as u8;
        if best_lane != 0 && score.losses[best_lane] < score.losses[0] {
            members.push(c);
            let loss = score.losses[best_lane];
            if loss < best_loss {
                best_loss = loss;
                best_ordinal = Some(c);
            }
        }
    }

    if members.is_empty() {
        return Ok(P5BlockRecord {
            k: k as u32,
            s_len: 0,
            joint: false,
            l_before,
            l_joint: None,
            l_best: l_before,
            committed_loss: l_before,
            commit_ordinals: Vec::new(),
            commit_coords: Vec::new(),
            commit_lanes: Vec::new(),
            epoch_after: state.epoch,
        });
    }

    let best_ord = best_ordinal.ok_or_else(|| fail("missing BVG-K best ordinal"))?;
    let l_joint = if members.len() >= 2 {
        let mut scratch_roots = state.roots.clone();
        for &m in &members {
            let coord = coords[m] as usize;
            let lane = usize::from(chosen_lanes[m]);
            scratch_roots[coord] = move_root[usize::from(scratch_roots[coord])][lane];
        }
        Some(eval_joint(&scratch_roots)?)
    } else {
        None
    };

    let joint = l_joint.is_some_and(|lj| lj < best_loss);
    let commit_ords: Vec<usize> = if joint {
        members.clone()
    } else {
        vec![best_ord]
    };

    let mut commit_coords = Vec::with_capacity(commit_ords.len());
    let mut commit_lanes = Vec::with_capacity(commit_ords.len());
    for &ord in &commit_ords {
        let coord = coords[ord];
        let lane = chosen_lanes[ord];
        state.roots[coord as usize] =
            move_root[usize::from(state.roots[coord as usize])][usize::from(lane)];
        commit_coords.push(coord);
        commit_lanes.push(lane);
    }
    state.epoch = state
        .epoch
        .checked_add(1)
        .ok_or_else(|| fail("P5S1State epoch overflow"))?;
    let committed_loss = if joint {
        l_joint.unwrap_or(best_loss)
    } else {
        best_loss
    };
    if committed_loss >= l_before {
        return Err(fail("P5 BVG-K commit failed strict improvement invariant"));
    }
    Ok(P5BlockRecord {
        k: k as u32,
        s_len: members.len() as u32,
        joint,
        l_before,
        l_joint,
        l_best: best_loss,
        committed_loss,
        commit_ordinals: commit_ords.into_iter().map(|o| o as u32).collect(),
        commit_coords,
        commit_lanes,
        epoch_after: state.epoch,
    })
}

#[cfg(feature = "cuda")]
pub fn bvgk_step_from_gpu_scores(
    eng: &mut P5GpuEngine,
    num_rows: u32,
    state: &mut P5S1State,
    coords: &[u32],
) -> Result<P5BlockRecord> {
    let scores = eng.score_rowsk_on_batch(num_rows, &state.roots, coords)?;
    bvgk_decide_and_apply(state, coords, &scores, |scratch| {
        eng.eval_roots_on_batch(num_rows, scratch)
    })
}

fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 1,000-draw paired record-level bootstrap 95% CI on `sum(x - y)` (pure integer `i128`).
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

/// Pure-integer mean Brier scaled by `10^6` (`micro-Brier`, where `1,000,000` = `1.000000`).
pub fn mean_brier_micro(total_brier_q32: u128, targets: u64) -> u128 {
    if targets == 0 {
        return 0;
    }
    (total_brier_q32.saturating_mul(1_000_000)) / (u128::from(targets) << 32)
}

/// Pure-integer signed ppm change `(cand - base) * 1_000_000 / base` (negative = reduction).
pub fn rel_change_ppm(cand: u128, base: u128) -> i128 {
    if base == 0 {
        return 0;
    }
    if cand >= base {
        (((cand - base) * 1_000_000) / base) as i128
    } else {
        -((((base - cand) * 1_000_000) / base) as i128)
    }
}

/// Pure-integer micro-BPB (`bits_q12 * 1_000_000 / (4096 * total_bytes)`).
pub fn micro_bpb(bits_q12: u128, bytes: u64) -> u128 {
    if bytes == 0 {
        return 0;
    }
    (bits_q12 * 1_000_000) / (4096 * u128::from(bytes))
}

/// Non-extrapolating `DECISION.md:342` Phase `P5` qualification gate evaluator:
/// - If `train_elapsed_secs <= 86,400` (`<= 1` GPU-day): requires actual observed Brier drop
///   `>= 1.00%` (`100 * (base - s1) >= base`) with ZERO linear upward extrapolation from short runs.
/// - If `train_elapsed_secs > 86,400`: requires `>= 1.00% * (elapsed / 86,400)`.
/// - Requires beating BOTH equal-trial re-seeded controls (`ctrl_canon`, `ctrl_rand`) AND
///   static re-seed (`ctrl_static`), with paired 95% CI upper bound `< 0` vs `baseline_seed1`
///   and vs `ctrl_rand`.
/// - Requires `DEV-test` BPB not rising (`bits_s1 <= bits_base`, since total bytes are identical).
/// - Requires `0` CPU spot-check diffs and `0` determinism repeat diffs.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_p5_gate(
    brier_base: u128,
    brier_s1: u128,
    brier_ctrl_canon: u128,
    brier_ctrl_rand: u128,
    brier_ctrl_static: u128,
    bits_base: u128,
    bits_s1: u128,
    ci_s1_vs_base: [i128; 2],
    ci_s1_vs_rand: [i128; 2],
    train_elapsed_secs: u64,
    cpu_spot_diffs: usize,
    det_diffs: usize,
) -> (bool, &'static str, Value) {
    let brier_drop = brier_base.saturating_sub(brier_s1);
    let clears_1pct_non_extrap = if train_elapsed_secs <= 86_400 {
        brier_s1 < brier_base && brier_drop.saturating_mul(100) >= brier_base
    } else {
        brier_s1 < brier_base
            && brier_drop.saturating_mul(100).saturating_mul(86_400)
                >= brier_base.saturating_mul(u128::from(train_elapsed_secs))
    };
    let beats_base_ci = brier_s1 < brier_base && ci_s1_vs_base[1] < 0;
    let beats_ctrl_canon = brier_s1 < brier_ctrl_canon;
    let beats_ctrl_rand_ci = brier_s1 < brier_ctrl_rand && ci_s1_vs_rand[1] < 0;
    let beats_ctrl_static = brier_s1 < brier_ctrl_static;
    let bpb_not_worse = bits_s1 <= bits_base;
    let instrument_clean = cpu_spot_diffs == 0 && det_diffs == 0;

    let pass = clears_1pct_non_extrap
        && beats_base_ci
        && beats_ctrl_canon
        && beats_ctrl_rand_ci
        && beats_ctrl_static
        && bpb_not_worse
        && instrument_clean;
    let verdict = if !instrument_clean {
        "INSTRUMENT_FAILED"
    } else if pass {
        "P5_QUALIFIED"
    } else {
        "P5_S1_FROZEN_AT_DEFAULT"
    };
    let details = json!({
        "clears_1pct_per_gpu_day_non_extrapolating": clears_1pct_non_extrap,
        "beats_baseline_ci95_strictly_negative": beats_base_ci,
        "beats_ctrl_reseed_canonical": beats_ctrl_canon,
        "beats_ctrl_reseed_random_ci95_strictly_negative": beats_ctrl_rand_ci,
        "beats_ctrl_reseed_static": beats_ctrl_static,
        "devtest_bpb_not_worse": bpb_not_worse,
        "instrument_clean": instrument_clean,
        "s1_vs_base_brier_ppm": rel_change_ppm(brier_s1, brier_base),
        "s1_vs_ctrl_rand_brier_ppm": rel_change_ppm(brier_s1, brier_ctrl_rand),
        "s1_vs_ctrl_canon_brier_ppm": rel_change_ppm(brier_s1, brier_ctrl_canon),
        "s1_vs_base_bits_ppm": rel_change_ppm(bits_s1, bits_base),
        "train_elapsed_secs_single_arm_equiv": train_elapsed_secs,
        "verdict": verdict,
    });
    (pass, verdict, details)
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct ArmTrainStats {
    batches: u64,
    trials: u64,
    commits: u64,
    joint_attempts: u64,
    joint_accepts: u64,
    coords_moved: u64,
    sum_l_before: u128,
    sum_l_after: u128,
}

#[cfg(feature = "cuda")]
impl ArmTrainStats {
    fn record(&mut self, rec: &P5BlockRecord) {
        self.batches += 1;
        self.trials += u64::from(rec.k) * 15;
        if !rec.commit_coords.is_empty() {
            self.commits += 1;
            self.coords_moved += rec.commit_coords.len() as u64;
        }
        if rec.l_joint.is_some() {
            self.joint_attempts += 1;
            if rec.joint {
                self.joint_accepts += 1;
            }
        }
        self.sum_l_before += u128::from(rec.l_before);
        self.sum_l_after += u128::from(rec.committed_loss);
    }

    fn to_json(&self, state: &P5S1State, baseline_roots: &[u8]) -> Value {
        let diff_coords = state
            .roots
            .iter()
            .zip(baseline_roots)
            .filter(|(a, b)| a != b)
            .count();
        json!({
            "batches": self.batches,
            "trials": self.trials,
            "epoch": state.epoch,
            "commits": self.commits,
            "joint_attempts": self.joint_attempts,
            "joint_accepts": self.joint_accepts,
            "coords_moved_total": self.coords_moved,
            "distinct_coords_changed_vs_seed1": diff_coords,
            "batch_brier_before_sum": self.sum_l_before.to_string(),
            "batch_brier_after_sum": self.sum_l_after.to_string(),
            "batch_brier_delta_ppm": rel_change_ppm(self.sum_l_after, self.sum_l_before),
        })
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn eval_stream_all_arms(
    eng: &mut P5GpuEngine,
    tokens: &[u16],
    recs: &[(usize, usize)],
    novel: Option<&[u8]>,
    arm_roots: &[&[u8]],
    spot_stride: u32,
    spot_rec_limit: u32,
    max_spots: usize,
) -> Result<(
    Vec<crate::cuda::EvalArmRecordOutHost>,
    Vec<crate::cuda::EvalSpotOutHost>,
)> {
    eng.eval_stream_multi_arm(
        tokens,
        recs,
        novel,
        arm_roots,
        spot_stride,
        spot_rec_limit,
        max_spots,
    )
}

#[cfg(feature = "cuda")]
fn sum_arm_eval(
    rec_outs: &[crate::cuda::EvalArmRecordOutHost],
    num_recs: usize,
    num_arms: usize,
    arm: usize,
) -> (u128, u128, u64) {
    let mut b = 0u128;
    let mut q = 0u128;
    let mut t = 0u64;
    for r in 0..num_recs {
        let o = &rec_outs[r * num_arms + arm];
        b += o.all_brier_u128();
        q += u128::from(o.all_bits_q12);
        t += o.targets;
    }
    (b, q, t)
}

pub struct TrainArgs<'a> {
    pub p4_path: &'a Path,
    pub seed1_path: &'a Path,
    pub r_path: &'a Path,
    pub devtune_path: &'a Path,
    pub out_dir: &'a Path,
    pub report_path: &'a Path,
    pub min_epochs: usize,
    pub max_epochs: usize,
    pub k_block: usize,
}

pub fn run_train(args: TrainArgs<'_>) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        Err(fail("p5-train train requires --features cuda"))
    }
    #[cfg(feature = "cuda")]
    {
        if args.min_epochs == 0
            || args.max_epochs < args.min_epochs
            || args.k_block == 0
            || args.k_block > P5_MAX_K
        {
            return Err(fail("invalid min_epochs/max_epochs/k_block"));
        }
        fs::create_dir_all(args.out_dir)?;
        let p4_file_bytes = verify_file_sha256(args.p4_path, PIN_P4_D11_SHA256, "p4.d11")?;
        drop(p4_file_bytes);
        let seed1_bytes = verify_file_sha256(args.seed1_path, PIN_SEED1_D11_SHA256, "seed1.d11")?;
        let r_raw = verify_file_sha256(args.r_path, PIN_R_UORT_SHA256, "r.tokens.u16")?;
        let dt_raw = verify_file_sha256(
            args.devtune_path,
            PIN_DEV_TUNE_UORT_SHA256,
            "dev-tune.tokens.u16",
        )?;

        let s1_seed1 = extract_s1_from_uorlut01(&seed1_bytes)?;
        if s1_seed1.vocab != P5_VOCAB || s1_seed1.blocks != P5_BLOCKS {
            return Err(fail("seed1.d11 S1 shape != (4096, 4)"));
        }
        drop(seed1_bytes);

        // Load frozen P4 embedded sections directly from p4.d11 and verify its S1 codes match seed1.
        let reloaded = load_client_artifact(args.p4_path, false)
            .map_err(|e| fail(format!("load p4.d11: {e}")))?;
        let (state_bytes, model_pick_codes) = match reloaded {
            ClientArtifactFile::Embedded {
                model,
                state_bytes,
                expected_state_sha256,
                ..
            } => {
                if hex::encode(Sha256::digest(&state_bytes)) != expected_state_sha256 {
                    return Err(fail("p4.d11 embedded state SHA-256 mismatch"));
                }
                let codes = model
                    .identity_pick_codes()
                    .map_err(|e| fail(format!("identity_pick_codes: {e}")))?;
                (state_bytes, codes)
            }
            ClientArtifactFile::Legacy(_) => {
                return Err(fail("p4.d11 parsed as Legacy instead of Embedded"));
            }
        };
        if model_pick_codes != s1_seed1.derive_pick_codes_4()? {
            return Err(fail(
                "p4.d11 embedded identity_pick_codes disagree with seed1.d11 S1 state",
            ));
        }

        let r_tokens = load_uort_tokens(&r_raw)?;
        drop(r_raw);
        let r_recs = split_records_on_bos(&r_tokens)?;
        let r_targets: usize = r_recs.iter().map(|&(_, l)| l.saturating_sub(1)).sum();
        if r_recs.len() != R_EXPECTED_RECORDS
            || r_tokens.len() != R_EXPECTED_TOKENS
            || r_targets != R_EXPECTED_TARGETS
        {
            return Err(fail(format!(
                "R census mismatch: recs={} toks={} tgts={}",
                r_recs.len(),
                r_tokens.len(),
                r_targets
            )));
        }

        let dt_tokens = load_uort_tokens(&dt_raw)?;
        drop(dt_raw);
        let dt_recs = split_records_on_bos(&dt_tokens)?;
        let dt_targets: usize = dt_recs.iter().map(|&(_, l)| l.saturating_sub(1)).sum();
        if dt_recs.len() != DEV_TUNE_EXPECTED_RECORDS
            || dt_tokens.len() != DEV_TUNE_EXPECTED_TOKENS
            || dt_targets != DEV_TUNE_EXPECTED_TARGETS
        {
            return Err(fail(format!(
                "DEV-tune census mismatch: recs={} toks={} tgts={}",
                dt_recs.len(),
                dt_tokens.len(),
                dt_targets
            )));
        }

        let raw = split_uorp4a01_sections(&state_bytes)?;
        let mut eng = P5GpuEngine::new(0, &raw, P5_MAX_BATCH_ROWS, P5_MAX_SLOTS)?;
        eng.upload_train_tokens(&r_tokens)?;
        let cubin_sha = eng.cubin_sha256().to_owned();

        // Initialize all 5 arms (2 frozen baselines + 3 trained arms with exact equal trial counts).
        let base_seed1 = s1_seed1.clone();
        let mut s1_from_seed1 = s1_seed1.clone();
        let mut ctrl_canon = P5S1State::reseeded(
            P5_VOCAB,
            P5_BLOCKS,
            SEED_RESEED_CANONICAL,
            s1_seed1.parent_trunk_sha256,
        )?;
        let ctrl_static = ctrl_canon.clone();
        let mut ctrl_rand = P5S1State::reseeded(
            P5_VOCAB,
            P5_BLOCKS,
            SEED_RESEED_RANDOM,
            s1_seed1.parent_trunk_sha256,
        )?;
        let mut rng_rand_coords = SEED_RANDOM_COORD_RNG;

        let mut st_s1 = ArmTrainStats::default();
        let mut st_canon = ArmTrainStats::default();
        let mut st_rand = ArmTrainStats::default();

        // Full DEV-tune Epoch 0 baseline (zero DEV-test peeks!).
        let eval0_t0 = Instant::now();
        let (dt_ep0_outs, _) = eval_stream_all_arms(
            &mut eng,
            &dt_tokens,
            &dt_recs,
            None,
            &[
                &base_seed1.roots,
                &s1_from_seed1.roots,
                &ctrl_canon.roots,
                &ctrl_rand.roots,
                &ctrl_static.roots,
            ],
            0,
            0,
            0,
        )?;
        let (dt0_base_b, dt0_base_q, dt0_tgts) = sum_arm_eval(&dt_ep0_outs, dt_recs.len(), 5, 0);
        let (dt0_canon_b, dt0_canon_q, _) = sum_arm_eval(&dt_ep0_outs, dt_recs.len(), 5, 2);
        let (dt0_rand_b, dt0_rand_q, _) = sum_arm_eval(&dt_ep0_outs, dt_recs.len(), 5, 3);
        eprintln!(
            "[train] DEV-tune ep0 ({}s, tgts={dt0_tgts}): base_micro_brier={} reseed_canon={} reseed_rand={}",
            eval0_t0.elapsed().as_secs(),
            mean_brier_micro(dt0_base_b, dt0_tgts),
            mean_brier_micro(dt0_canon_b, dt0_tgts),
            mean_brier_micro(dt0_rand_b, dt0_tgts),
        );

        let mut epoch_logs = Vec::new();
        epoch_logs.push(json!({
            "epoch_pass": 0,
            "dev_tune_targets": dt0_tgts,
            "dev_tune_brier": {
                "baseline_seed1": dt0_base_b.to_string(),
                "s1_p5_from_seed1": dt0_base_b.to_string(),
                "ctrl_reseed_canonical": dt0_canon_b.to_string(),
                "ctrl_reseed_random": dt0_rand_b.to_string(),
                "ctrl_reseed_static": dt0_canon_b.to_string(),
            },
            "dev_tune_mean_brier_micro": {
                "baseline_seed1": mean_brier_micro(dt0_base_b, dt0_tgts).to_string(),
                "ctrl_reseed_canonical": mean_brier_micro(dt0_canon_b, dt0_tgts).to_string(),
                "ctrl_reseed_random": mean_brier_micro(dt0_rand_b, dt0_tgts).to_string(),
            },
            "dev_tune_bits_q12": {
                "baseline_seed1": dt0_base_q.to_string(),
                "ctrl_reseed_canonical": dt0_canon_q.to_string(),
                "ctrl_reseed_random": dt0_rand_q.to_string(),
            },
        }));

        let train_t0 = Instant::now();
        let mut batch_idx = 0u64;
        let mut prev_dt_s1_brier = dt0_base_b;
        let mut epochs_completed = 0usize;
        let mut stop_reason = "max_epochs_reached".to_owned();
        let mut total_materialize_ms = 0u128;
        let mut total_s1_step_ms = 0u128;
        let mut total_canon_step_ms = 0u128;
        let mut total_rand_step_ms = 0u128;

        for ep in 1..=args.max_epochs {
            let ep_t0 = Instant::now();
            let mut sched = RowBatchScheduler::new(&r_tokens, &r_recs, P5_MAX_BATCH_ROWS)?;
            let mut ep_rows = 0u64;
            let mut ep_batches = 0u64;

            while let Some(cmds) = sched.next_batch() {
                let num_rows = cmds.len() as u32;
                let tm = Instant::now();
                eng.materialize_batch(&cmds)?;
                total_materialize_ms += tm.elapsed().as_millis();

                let canon_coords = p5_schedule_block(P5_S1_COORDS, args.k_block, batch_idx)?;
                let rand_coords =
                    p5_random_block_coords(P5_S1_COORDS, canon_coords.len(), &mut rng_rand_coords)?;

                let t1 = Instant::now();
                let rec_s1 = bvgk_step_from_gpu_scores(
                    &mut eng,
                    num_rows,
                    &mut s1_from_seed1,
                    &canon_coords,
                )?;
                total_s1_step_ms += t1.elapsed().as_millis();
                st_s1.record(&rec_s1);

                let t2 = Instant::now();
                let rec_c =
                    bvgk_step_from_gpu_scores(&mut eng, num_rows, &mut ctrl_canon, &canon_coords)?;
                total_canon_step_ms += t2.elapsed().as_millis();
                st_canon.record(&rec_c);

                let t3 = Instant::now();
                let rec_r =
                    bvgk_step_from_gpu_scores(&mut eng, num_rows, &mut ctrl_rand, &rand_coords)?;
                total_rand_step_ms += t3.elapsed().as_millis();
                st_rand.record(&rec_r);

                ep_rows += u64::from(num_rows);
                ep_batches += 1;
                batch_idx += 1;
                if ep_batches.is_multiple_of(250) {
                    eprintln!(
                        "[train] ep {ep} batch {ep_batches} rows={ep_rows} s1_commits={} canon_commits={} rand_commits={} elapsed={}s",
                        st_s1.commits,
                        st_canon.commits,
                        st_rand.commits,
                        ep_t0.elapsed().as_secs()
                    );
                }
            }
            if ep_rows != R_EXPECTED_TARGETS as u64 {
                return Err(fail(format!(
                    "epoch {ep} visited {ep_rows} R targets != expected {R_EXPECTED_TARGETS}"
                )));
            }
            epochs_completed = ep;

            // Full DEV-tune evaluation after each epoch (zero DEV-test peeks!).
            let (dt_outs, _) = eval_stream_all_arms(
                &mut eng,
                &dt_tokens,
                &dt_recs,
                None,
                &[
                    &base_seed1.roots,
                    &s1_from_seed1.roots,
                    &ctrl_canon.roots,
                    &ctrl_rand.roots,
                    &ctrl_static.roots,
                ],
                0,
                0,
                0,
            )?;
            let (dt_base_b, dt_base_q, dt_t) = sum_arm_eval(&dt_outs, dt_recs.len(), 5, 0);
            let (dt_s1_b, dt_s1_q, _) = sum_arm_eval(&dt_outs, dt_recs.len(), 5, 1);
            let (dt_canon_b, dt_canon_q, _) = sum_arm_eval(&dt_outs, dt_recs.len(), 5, 2);
            let (dt_rand_b, dt_rand_q, _) = sum_arm_eval(&dt_outs, dt_recs.len(), 5, 3);
            let (dt_stat_b, dt_stat_q, _) = sum_arm_eval(&dt_outs, dt_recs.len(), 5, 4);
            if dt_base_b != dt0_base_b || dt_base_q != dt0_base_q {
                return Err(fail("DEV-tune frozen baseline drifted between epochs"));
            }

            let step_drop_ppm = -rel_change_ppm(dt_s1_b, prev_dt_s1_brier);
            let total_drop_ppm = -rel_change_ppm(dt_s1_b, dt0_base_b);
            eprintln!(
                "[train] ep {ep} DONE ({}s): R_batches={ep_batches} DEV-tune s1_vs_base={}ppm step_drop={}ppm canon_vs_base={}ppm rand_vs_base={}ppm",
                ep_t0.elapsed().as_secs(),
                rel_change_ppm(dt_s1_b, dt0_base_b),
                step_drop_ppm,
                rel_change_ppm(dt_canon_b, dt0_base_b),
                rel_change_ppm(dt_rand_b, dt0_base_b),
            );
            epoch_logs.push(json!({
                "epoch_pass": ep,
                "r_batches": ep_batches,
                "r_targets": ep_rows,
                "epoch_wall_secs": ep_t0.elapsed().as_secs(),
                "dev_tune_targets": dt_t,
                "dev_tune_brier": {
                    "baseline_seed1": dt_base_b.to_string(),
                    "s1_p5_from_seed1": dt_s1_b.to_string(),
                    "ctrl_reseed_canonical": dt_canon_b.to_string(),
                    "ctrl_reseed_random": dt_rand_b.to_string(),
                    "ctrl_reseed_static": dt_stat_b.to_string(),
                },
                "dev_tune_mean_brier_micro": {
                    "baseline_seed1": mean_brier_micro(dt_base_b, dt_t).to_string(),
                    "s1_p5_from_seed1": mean_brier_micro(dt_s1_b, dt_t).to_string(),
                    "ctrl_reseed_canonical": mean_brier_micro(dt_canon_b, dt_t).to_string(),
                    "ctrl_reseed_random": mean_brier_micro(dt_rand_b, dt_t).to_string(),
                    "ctrl_reseed_static": mean_brier_micro(dt_stat_b, dt_t).to_string(),
                },
                "dev_tune_brier_ppm_vs_base": {
                    "s1_p5_from_seed1": rel_change_ppm(dt_s1_b, dt0_base_b),
                    "ctrl_reseed_canonical": rel_change_ppm(dt_canon_b, dt0_base_b),
                    "ctrl_reseed_random": rel_change_ppm(dt_rand_b, dt0_base_b),
                    "ctrl_reseed_static": rel_change_ppm(dt_stat_b, dt0_base_b),
                },
                "dev_tune_bits_q12": {
                    "baseline_seed1": dt_base_q.to_string(),
                    "s1_p5_from_seed1": dt_s1_q.to_string(),
                    "ctrl_reseed_canonical": dt_canon_q.to_string(),
                    "ctrl_reseed_random": dt_rand_q.to_string(),
                    "ctrl_reseed_static": dt_stat_q.to_string(),
                },
                "s1_step_drop_ppm": step_drop_ppm,
                "s1_total_drop_ppm": total_drop_ppm,
                "arms": {
                    "s1_p5_from_seed1": st_s1.to_json(&s1_from_seed1, &base_seed1.roots),
                    "ctrl_reseed_canonical": st_canon.to_json(&ctrl_canon, &base_seed1.roots),
                    "ctrl_reseed_random": st_rand.to_json(&ctrl_rand, &base_seed1.roots),
                },
            }));
            prev_dt_s1_brier = dt_s1_b;

            if ep >= args.min_epochs {
                let continue_ep = step_drop_ppm >= (DEVTUNE_CONTINUE_EPOCH_PPM as i128)
                    && total_drop_ppm >= (DEVTUNE_CONTINUE_TOTAL_PPM as i128);
                if !continue_ep {
                    stop_reason = format!(
                        "dev_tune_saturated_after_epoch_{ep}_step_drop_{step_drop_ppm}ppm_total_drop_{total_drop_ppm}ppm"
                    );
                    break;
                }
            }
        }

        // Write self-describing UORP5S10 checkpoints for all 5 frozen arms + round-trip check.
        let ckpts = [
            ("baseline_seed1", &base_seed1),
            ("s1_p5_from_seed1", &s1_from_seed1),
            ("ctrl_reseed_canonical", &ctrl_canon),
            ("ctrl_reseed_random", &ctrl_rand),
            ("ctrl_reseed_static", &ctrl_static),
        ];
        let mut ckpt_shas = serde_json::Map::new();
        for (name, st) in ckpts {
            let bytes = st.encode_bytes()?;
            let reloaded_s1 = P5S1State::from_bytes(&bytes)?;
            if reloaded_s1 != *st {
                return Err(fail(format!("UORP5S10 round-trip mismatch for {name}")));
            }
            let sha = hex::encode(Sha256::digest(&bytes));
            let path = args.out_dir.join(format!("{name}.uorp5s10"));
            fs::write(&path, &bytes)?;
            ckpt_shas.insert(name.to_owned(), json!(sha));
        }

        // Equal-trial invariant check across all 3 trained arms.
        if st_s1.trials != st_canon.trials || st_s1.trials != st_rand.trials {
            return Err(fail(format!(
                "equal-trial invariant violated: s1={} canon={} rand={}",
                st_s1.trials, st_canon.trials, st_rand.trials
            )));
        }

        let train_wall_secs = train_t0.elapsed().as_secs();
        let single_arm_equiv_secs = ((total_materialize_ms + total_s1_step_ms) / 1000) as u64;
        let report = json!({
            "schema": "uor-r4.p5-train-report/1",
            "cubin_sha256": cubin_sha,
            "p4_d11_sha256": PIN_P4_D11_SHA256,
            "seed1_d11_sha256": PIN_SEED1_D11_SHA256,
            "r_uort_sha256": PIN_R_UORT_SHA256,
            "dev_tune_uort_sha256": PIN_DEV_TUNE_UORT_SHA256,
            "s1_coords": P5_S1_COORDS,
            "batch_rows_cap": P5_MAX_BATCH_ROWS,
            "k_block": args.k_block,
            "min_epochs": args.min_epochs,
            "max_epochs": args.max_epochs,
            "epochs_completed": epochs_completed,
            "stop_reason": stop_reason,
            "equal_trials_per_arm": st_s1.trials,
            "total_batches_per_arm": st_s1.batches,
            "train_wall_secs_3_arms_plus_devtune": train_wall_secs,
            "single_arm_equiv_train_secs": single_arm_equiv_secs,
            "timing_ms": {
                "materialize_shared": total_materialize_ms,
                "s1_p5_from_seed1": total_s1_step_ms,
                "ctrl_reseed_canonical": total_canon_step_ms,
                "ctrl_reseed_random": total_rand_step_ms,
            },
            "arms": {
                "s1_p5_from_seed1": st_s1.to_json(&s1_from_seed1, &base_seed1.roots),
                "ctrl_reseed_canonical": st_canon.to_json(&ctrl_canon, &base_seed1.roots),
                "ctrl_reseed_random": st_rand.to_json(&ctrl_rand, &base_seed1.roots),
            },
            "checkpoints_sha256": ckpt_shas,
            "epoch_logs": epoch_logs,
            "pass": true,
        });
        fs::write(args.report_path, serde_json::to_vec_pretty(&report)?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }
}

pub struct EvalDevTestArgs<'a> {
    pub p4_path: &'a Path,
    pub ckpt_dir: &'a Path,
    pub train_report_path: &'a Path,
    pub devtest_dir: &'a Path,
    pub rstar_path: &'a Path,
    pub out_report: &'a Path,
}

pub fn run_eval_devtest(args: EvalDevTestArgs<'_>) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        Err(fail("p5-train eval-devtest requires --features cuda"))
    }
    #[cfg(feature = "cuda")]
    {
        let eval_t0 = Instant::now();
        let _ = verify_file_sha256(args.p4_path, PIN_P4_D11_SHA256, "p4.d11")?;
        let train_rep: Value = serde_json::from_slice(&fs::read(args.train_report_path)?)?;
        let single_arm_secs = train_rep["single_arm_equiv_train_secs"]
            .as_u64()
            .ok_or_else(|| fail("train_report missing single_arm_equiv_train_secs"))?;
        let equal_trials = train_rep["equal_trials_per_arm"]
            .as_u64()
            .ok_or_else(|| fail("train_report missing equal_trials_per_arm"))?;

        // Verify all sealed DEV-test input SHA-256s fail-closed!
        let census_bytes = verify_file_sha256(
            &args.devtest_dir.join("dev-test.json"),
            PIN_DEV_TEST_CENSUS_SHA256,
            "dev-test.json",
        )?;
        let census: Value = serde_json::from_slice(&census_bytes)?;
        if census["split"] != "dev-test"
            || census["records"] != DEV_TEST_EXPECTED_RECORDS
            || census["tokens"] != DEV_TEST_EXPECTED_TOKENS as u64
            || census["targets"] != DEV_TEST_EXPECTED_TARGETS as u64
        {
            return Err(fail("DEV-test census identity mismatch"));
        }
        let dt_raw = verify_file_sha256(
            &args.devtest_dir.join("dev-test.tokens.u16"),
            PIN_DEV_TEST_UORT_SHA256,
            "dev-test.tokens.u16",
        )?;
        let dt_novel = verify_file_sha256(
            &args.devtest_dir.join("dev-test.novel.u8"),
            PIN_DEV_TEST_NOVEL_SHA256,
            "dev-test.novel.u8",
        )?;
        let id_bytes = verify_file_sha256(
            &args.devtest_dir.join("id-bytes.u8"),
            PIN_ID_BYTES_SHA256,
            "id-bytes.u8",
        )?;
        let _rstar_raw = verify_file_sha256(
            args.rstar_path,
            PIN_RSTAR_MASSES_SHA256,
            "rstar-devtest.masses.u64",
        )?;
        if id_bytes.len() != P4_VOCAB || dt_novel.len() != DEV_TEST_EXPECTED_TOKENS {
            return Err(fail("DEV-test id-bytes or novel length mismatch"));
        }

        let dt_tokens = load_uort_tokens(&dt_raw)?;
        drop(dt_raw);
        let dt_recs = split_records_on_bos(&dt_tokens)?;
        let dt_targets: usize = dt_recs.iter().map(|&(_, l)| l.saturating_sub(1)).sum();
        if dt_recs.len() != DEV_TEST_EXPECTED_RECORDS
            || dt_tokens.len() != DEV_TEST_EXPECTED_TOKENS
            || dt_targets != DEV_TEST_EXPECTED_TARGETS
        {
            return Err(fail("DEV-test record/token/target count mismatch"));
        }

        // Exact byte totals across DEV-test (shared by all 5 arms).
        let mut total_all_bytes = 0u64;
        let mut total_novel_bytes = 0u64;
        let mut total_novel_targets = 0u64;
        for &(ro, rl) in &dt_recs {
            for pos in 1..rl {
                let tgt = dt_tokens[ro + pos];
                if tgt == 0 {
                    continue;
                }
                let b = u64::from(id_bytes[usize::from(tgt)]);
                total_all_bytes += b;
                if dt_novel[ro + pos] == 1 {
                    total_novel_bytes += b;
                    total_novel_targets += 1;
                }
            }
        }

        // Load all 5 frozen UORP5S10 checkpoints and verify SHA-256 against train_report.json.
        let arm_names = [
            "baseline_seed1",
            "s1_p5_from_seed1",
            "ctrl_reseed_canonical",
            "ctrl_reseed_random",
            "ctrl_reseed_static",
        ];
        let mut arm_states = Vec::with_capacity(arm_names.len());
        for &name in &arm_names {
            let p = args.ckpt_dir.join(format!("{name}.uorp5s10"));
            let exp_sha = train_rep["checkpoints_sha256"][name]
                .as_str()
                .ok_or_else(|| fail(format!("missing checkpoint sha for {name}")))?;
            let bytes = verify_file_sha256(&p, exp_sha, name)?;
            arm_states.push(P5S1State::from_bytes(&bytes)?);
        }

        // Load frozen P4 embedded artifact (once for GPU and for CPU spot-check verification!).
        let reloaded = load_client_artifact(args.p4_path, false)
            .map_err(|e| fail(format!("load p4.d11: {e}")))?;
        let (model, state_bytes, expected_state_sha256, identity) = match reloaded {
            ClientArtifactFile::Embedded {
                model,
                state_bytes,
                expected_state_sha256,
                identity,
            } => (model, state_bytes, expected_state_sha256, identity),
            ClientArtifactFile::Legacy(_) => return Err(fail("p4.d11 parsed as Legacy")),
        };
        let raw = split_uorp4a01_sections(&state_bytes)?;
        let mut eng = P5GpuEngine::new(0, &raw, 16, P5_MAX_SLOTS)?;
        let cubin_sha = eng.cubin_sha256().to_owned();

        // Single-shot GPU evaluation over ALL 5,321 sealed DEV-test records across ALL 5 arms!
        // Capture spot checks (`stride = 197` on first `64` records across all 5 arms => >1,000 spots).
        let arm_refs: Vec<&[u8]> = arm_states.iter().map(|s| s.roots.as_slice()).collect();
        let (rec_outs, spots) = eval_stream_all_arms(
            &mut eng,
            &dt_tokens,
            &dt_recs,
            Some(&dt_novel),
            &arm_refs,
            197,
            64,
            1500,
        )?;
        eprintln!(
            "[eval-devtest] GPU 5-arm pass complete in {}s (spots captured={})",
            eval_t0.elapsed().as_secs(),
            spots.len()
        );

        // Determinism repeat on first 64 records across all 5 arms (require 0 diffs!).
        let (det_outs, _) = eval_stream_all_arms(
            &mut eng,
            &dt_tokens,
            &dt_recs[..64],
            Some(&dt_novel),
            &arm_refs,
            0,
            0,
            0,
        )?;
        let mut det_diffs = 0usize;
        for i in 0..(64 * arm_names.len()) {
            let a = &rec_outs[i];
            let b = &det_outs[i];
            if a.all_brier_u128() != b.all_brier_u128()
                || a.all_bits_q12 != b.all_bits_q12
                || a.novel_brier_u128() != b.novel_brier_u128()
                || a.novel_bits_q12 != b.novel_bits_q12
                || a.targets != b.targets
                || a.novel_targets != b.novel_targets
            {
                det_diffs += 1;
            }
        }
        drop(eng);

        // Live CPU spot-check verification using `DiscreteCountModel::bind_embedded` + `P4Session`!
        let mut bound = DiscreteCountModel::bind_embedded(
            model,
            &state_bytes,
            &expected_state_sha256,
            identity,
        )
        .map_err(|e| fail(format!("bind_embedded for CPU spot checks: {e}")))?;
        let p4_ref = bound
            .p4
            .as_mut()
            .ok_or_else(|| fail("bound p4.d11 missing P4State"))?;
        let arm_codes: Vec<Vec<[u16; 4]>> = arm_states
            .iter()
            .map(|s| s.derive_pick_codes_4())
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Check >= 512 captured spots (up to 640 stratified across all 5 arms).
        let spots_to_check = spots.len().min(640);
        if spots_to_check < 512 {
            return Err(fail(format!(
                "insufficient captured CPU spot checks: {spots_to_check} < 512"
            )));
        }
        let mut cpu_spot_diffs = 0usize;
        let mut sess = P4Session::new();
        let mut last_arm = usize::MAX;
        // Group checks by arm to swap `p4_ref.codes` at most 5 times.
        let mut spot_indices: Vec<usize> = (0..spots_to_check).collect();
        spot_indices.sort_by_key(|&i| (spots[i].arm, spots[i].rec_idx, spots[i].pos));
        for idx in spot_indices {
            let sp = &spots[idx];
            let arm = sp.arm as usize;
            if arm != last_arm {
                p4_ref.inner.codes.clone_from(&arm_codes[arm]);
                last_arm = arm;
            }
            let (ro, rl) = dt_recs[sp.rec_idx as usize];
            let pos = sp.pos as usize;
            if pos == 0 || pos >= rl {
                return Err(fail("spot check pos out of record bounds"));
            }
            let start = pos.saturating_sub(P4_MAX_CONTEXT);
            let win = &dt_tokens[ro + start..ro + pos];
            let target = dt_tokens[ro + pos];
            if u32::from(target) != sp.target {
                return Err(fail("spot check target mismatch"));
            }
            sess.sync_prompt(win)
                .map_err(|e| fail(format!("CPU spot sync_prompt: {e}")))?;
            let (_, tr) = sess
                .step(&p4_ref.inner, true, true)
                .map_err(|e| fail(format!("CPU spot step: {e}")))?;
            let tr = tr.ok_or_else(|| fail("CPU spot missing trace"))?;
            let sc = sess.last_scores_q48();
            let cb = full_vocab_brier_q32(sc, target)?;
            let cq = bits_q12(sc[usize::from(target)])?;
            let cp = tr.pick_pos.map(|p| {
                (
                    p,
                    tr.pick_distance.unwrap_or(0),
                    tr.pick_class,
                    tr.pick_successor.unwrap_or(0),
                )
            });
            let gp = (sp.pick_has != 0).then_some((
                sp.pick_pos,
                sp.pick_dist,
                sp.pick_class as u8,
                sp.pick_succ as u16,
            ));
            if cb != sp.brier_q32 || cq != sp.bits_q12 || cp != gp {
                cpu_spot_diffs += 1;
            }
        }
        eprintln!(
            "[eval-devtest] CPU spot checks={spots_to_check} diffs={cpu_spot_diffs}, det_diffs={det_diffs}"
        );

        // Aggregate exact u128 totals & per-record vectors for paired 1,000-draw bootstrap 95% CIs.
        let n_recs = dt_recs.len();
        let n_arms = arm_names.len();
        let mut all_brier = vec![0u128; n_arms];
        let mut all_bits = vec![0u128; n_arms];
        let mut nov_brier = vec![0u128; n_arms];
        let mut nov_bits = vec![0u128; n_arms];
        let mut per_rec_brier: Vec<Vec<u128>> = vec![Vec::with_capacity(n_recs); n_arms];
        let mut per_rec_bits: Vec<Vec<u128>> = vec![Vec::with_capacity(n_recs); n_arms];

        for r in 0..n_recs {
            for a in 0..n_arms {
                let o = &rec_outs[r * n_arms + a];
                let ab = o.all_brier_u128();
                let aq = u128::from(o.all_bits_q12);
                all_brier[a] += ab;
                all_bits[a] += aq;
                nov_brier[a] += o.novel_brier_u128();
                nov_bits[a] += u128::from(o.novel_bits_q12);
                per_rec_brier[a].push(ab);
                per_rec_bits[a].push(aq);
            }
        }

        // Verify exact target counts across all 5 arms.
        for a in 0..n_arms {
            let tgts: u64 = (0..n_recs).map(|r| rec_outs[r * n_arms + a].targets).sum();
            let ntgts: u64 = (0..n_recs)
                .map(|r| rec_outs[r * n_arms + a].novel_targets)
                .sum();
            if tgts != DEV_TEST_EXPECTED_TARGETS as u64 || ntgts != total_novel_targets {
                return Err(fail(format!(
                    "arm {a} target count mismatch: tgts={tgts} ntgts={ntgts}"
                )));
            }
        }

        let ci_s1_vs_base_brier = paired_record_ci(
            &per_rec_brier[1],
            &per_rec_brier[0],
            1000,
            SEED_BOOTSTRAP_CI,
        )?;
        let ci_s1_vs_rand_brier = paired_record_ci(
            &per_rec_brier[1],
            &per_rec_brier[3],
            1000,
            SEED_BOOTSTRAP_CI + 1,
        )?;
        let ci_s1_vs_canon_brier = paired_record_ci(
            &per_rec_brier[1],
            &per_rec_brier[2],
            1000,
            SEED_BOOTSTRAP_CI + 2,
        )?;
        let ci_s1_vs_base_bits = paired_record_ci(
            &per_rec_bits[1],
            &per_rec_bits[0],
            1000,
            SEED_BOOTSTRAP_CI + 3,
        )?;
        let ci_canon_vs_static_brier = paired_record_ci(
            &per_rec_brier[2],
            &per_rec_brier[4],
            1000,
            SEED_BOOTSTRAP_CI + 4,
        )?;
        let ci_rand_vs_static_brier = paired_record_ci(
            &per_rec_brier[3],
            &per_rec_brier[4],
            1000,
            SEED_BOOTSTRAP_CI + 5,
        )?;

        let (pass, verdict, gate_details) = evaluate_p5_gate(
            all_brier[0],
            all_brier[1],
            all_brier[2],
            all_brier[3],
            all_brier[4],
            all_bits[0],
            all_bits[1],
            ci_s1_vs_base_brier,
            ci_s1_vs_rand_brier,
            single_arm_secs,
            cpu_spot_diffs,
            det_diffs,
        );

        let mut arms_json = serde_json::Map::new();
        for (a, &name) in arm_names.iter().enumerate() {
            let mbpb_all = micro_bpb(all_bits[a], total_all_bytes);
            let mbpb_nov = micro_bpb(nov_bits[a], total_novel_bytes);
            arms_json.insert(
                name.to_owned(),
                json!({
                    "epoch": arm_states[a].epoch,
                    "all_brier_q32_sum": all_brier[a].to_string(),
                    "all_mean_brier_micro": mean_brier_micro(all_brier[a], DEV_TEST_EXPECTED_TARGETS as u64).to_string(),
                    "all_brier_ppm_vs_base": rel_change_ppm(all_brier[a], all_brier[0]),
                    "novel_brier_q32_sum": nov_brier[a].to_string(),
                    "novel_mean_brier_micro": mean_brier_micro(nov_brier[a], total_novel_targets).to_string(),
                    "all_bits_q12": all_bits[a].to_string(),
                    "all_bpb_micro": mbpb_all.to_string(),
                    "all_bits_ppm_vs_base": rel_change_ppm(all_bits[a], all_bits[0]),
                    "all_pct_p1_a1s_bps": (all_bits[a] * 10_000 / P1_A1S_DEVTEST_ALL_Q12).to_string(),
                    "all_pct_rstar_bps": (all_bits[a] * 10_000 / RSTAR_DEVTEST_ALL_Q12).to_string(),
                    "novel_bits_q12": nov_bits[a].to_string(),
                    "novel_bpb_micro": mbpb_nov.to_string(),
                }),
            );
        }

        let report = json!({
            "schema": "uor-r4.p5-devtest-report/1",
            "cubin_sha256": cubin_sha,
            "p4_d11_sha256": PIN_P4_D11_SHA256,
            "dev_test_uort_sha256": PIN_DEV_TEST_UORT_SHA256,
            "dev_test_novel_sha256": PIN_DEV_TEST_NOVEL_SHA256,
            "id_bytes_sha256": PIN_ID_BYTES_SHA256,
            "rstar_masses_sha256": PIN_RSTAR_MASSES_SHA256,
            "records": DEV_TEST_EXPECTED_RECORDS,
            "targets": DEV_TEST_EXPECTED_TARGETS,
            "novel_targets": total_novel_targets,
            "total_all_bytes": total_all_bytes,
            "total_novel_bytes": total_novel_bytes,
            "context_window": P4_MAX_CONTEXT,
            "equal_trials_per_arm": equal_trials,
            "reference_baselines": {
                "rstar_all_bits_q12": RSTAR_DEVTEST_ALL_Q12.to_string(),
                "rstar_bpb_micro": RSTAR_DEVTEST_BPB_MICRO.to_string(),
                "p1_a1s_all_bits_q12": P1_A1S_DEVTEST_ALL_Q12.to_string(),
            },
            "arms": arms_json,
            "paired_bootstrap_1000_ci95": {
                "s1_minus_base_brier_q32": [ci_s1_vs_base_brier[0].to_string(), ci_s1_vs_base_brier[1].to_string()],
                "s1_minus_ctrl_rand_brier_q32": [ci_s1_vs_rand_brier[0].to_string(), ci_s1_vs_rand_brier[1].to_string()],
                "s1_minus_ctrl_canon_brier_q32": [ci_s1_vs_canon_brier[0].to_string(), ci_s1_vs_canon_brier[1].to_string()],
                "s1_minus_base_bits_q12": [ci_s1_vs_base_bits[0].to_string(), ci_s1_vs_base_bits[1].to_string()],
                "ctrl_canon_minus_static_brier_q32": [ci_canon_vs_static_brier[0].to_string(), ci_canon_vs_static_brier[1].to_string()],
                "ctrl_rand_minus_static_brier_q32": [ci_rand_vs_static_brier[0].to_string(), ci_rand_vs_static_brier[1].to_string()],
            },
            "cpu_spot_checks": {
                "positions_checked": spots_to_check,
                "diffs": cpu_spot_diffs,
            },
            "determinism_repeat": {
                "records_checked": 64,
                "arms_checked": n_arms,
                "diffs": det_diffs,
            },
            "gate": gate_details,
            "p5_qualified": pass,
            "verdict": verdict,
            "eval_wall_secs": eval_t0.elapsed().as_secs(),
        });
        fs::write(args.out_report, serde_json::to_vec_pretty(&report)?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);

        if cpu_spot_diffs != 0 || det_diffs != 0 {
            return Err(fail(format!(
                "Gate 4 instrument check failed: cpu_spot_diffs={cpu_spot_diffs} det_diffs={det_diffs}"
            )));
        }
        Ok(())
    }
}

/// Rewrite `seed1_bytes` (`UORLUT01`) with `s1_state`, re-bind `fit_json`'s
/// `prepared.trunk_sha256` and `UORS5W02` header digest (without altering any
/// weight words or `S0`/`S2`/`S3`/`S4`/`S6` payloads), pack `p5.d11` via
/// `write_p4_deployment`, and cold-reload verify `identity_pick_codes()`.
pub fn rebind_and_pack_p5_deployment(
    p4_state_bytes: &[u8],
    seed1_bytes: &[u8],
    s1_state: &P5S1State,
    out_p5_path: &Path,
) -> Result<(String, String, u64)> {
    if out_p5_path.exists() {
        return Err(fail("out_p5_path already exists (L-002 refuse overwrite)"));
    }
    let p5_trunk_bytes = rewrite_uorlut01_s1(seed1_bytes, s1_state)?;
    let p5_trunk_sha = hex::encode(Sha256::digest(&p5_trunk_bytes));

    let raw = split_uorp4a01_sections(p4_state_bytes)?;
    let orig_fit_sha: [u8; 32] = Sha256::digest(raw.fit_json).into();
    let orig_a1s = FrozenPickMixture::from_bytes(raw.a1s_weights, orig_fit_sha)
        .map_err(|e| fail(format!("decode existing A1s weights: {e}")))?;
    if raw.a1s_weights.len() < 12 {
        return Err(fail("truncated A1s weights header"));
    }
    let highest = u32::from_le_bytes(raw.a1s_weights[8..12].try_into()?);

    let mut fit_val: Value = serde_json::from_slice(raw.fit_json)?;
    let prep_obj = fit_val
        .get_mut("prepared")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| fail("fit_json missing prepared object"))?;
    prep_obj.insert("trunk_sha256".to_owned(), json!(p5_trunk_sha));
    let p5_fit_json = serde_json::to_vec(&fit_val)?;
    let p5_fit_sha: [u8; 32] = Sha256::digest(&p5_fit_json).into();

    let p5_a1s_bytes = FrozenPickMixture::with_rule(
        highest,
        p5_fit_sha,
        orig_a1s.weights().to_vec(),
        orig_a1s.rule(),
    )
    .map_err(|e| fail(format!("re-bind A1s weights to P5 trunk: {e}")))?
    .to_bytes();

    write_p4_deployment(
        &p5_trunk_bytes,
        &p5_fit_json,
        &p5_a1s_bytes,
        raw.counts_v3,
        raw.s3_compact,
        raw.s2_skip_compact,
        raw.s4_table,
        raw.s6_weights,
        out_p5_path,
    )
    .map_err(|e| fail(format!("write_p4_deployment p5.d11: {e}")))?;

    let p5_bytes = fs::metadata(out_p5_path)?.len();
    let cold = load_client_artifact(out_p5_path, false)
        .map_err(|e| fail(format!("cold reload p5.d11: {e}")))?;
    let artifact_sha = match cold {
        ClientArtifactFile::Embedded {
            model,
            state_bytes: sb,
            expected_state_sha256: es,
            identity: id,
        } => {
            if model
                .identity_pick_codes()
                .map_err(|e| fail(format!("cold p5 pick codes: {e}")))?
                != s1_state.derive_pick_codes_4()?
            {
                return Err(fail("cold p5.d11 identity_pick_codes != trained S1"));
            }
            let bound = DiscreteCountModel::bind_embedded(model, &sb, &es, id)
                .map_err(|e| fail(format!("cold bind p5.d11: {e}")))?;
            bound
                .identity
                .as_ref()
                .map(|x| x.artifact_sha256.clone())
                .unwrap_or_default()
        }
        ClientArtifactFile::Legacy(_) => return Err(fail("cold p5.d11 parsed as Legacy")),
    };
    Ok((p5_trunk_sha, artifact_sha, p5_bytes))
}

/// Conditional Gate 5: ONLY packs `p5.d11` (`<= 4.000 GB`) if `devtest_p5_report.json`
/// reached `verdict == "P5_QUALIFIED"`. Otherwise records an explicit skip receipt freezing
/// `S1` at `seed1.d11` (`DECISION.md:342`).
pub fn run_pack_if_pass(
    devtest_report: &Path,
    p4_path: &Path,
    seed1_path: &Path,
    tokenizer_path: &Path,
    s1_ckpt_path: &Path,
    out_p5_path: &Path,
    out_report_path: &Path,
) -> Result<()> {
    let dt: Value = serde_json::from_slice(&fs::read(devtest_report)?)?;
    let verdict = dt["verdict"]
        .as_str()
        .ok_or_else(|| fail("devtest_p5_report missing verdict"))?;
    let qualified = dt["p5_qualified"].as_bool().unwrap_or(false);
    if !qualified || verdict != "P5_QUALIFIED" {
        let rep = json!({
            "schema": "uor-r4.p5-pack-status/1",
            "packed": false,
            "verdict": verdict,
            "s1_frozen_at": "seed1.d11",
            "seed1_d11_sha256": PIN_SEED1_D11_SHA256,
            "reason": "Phase P5 DEV-test gate did not qualify; S1 remains frozen at geometric default per DECISION.md:342",
        });
        fs::write(out_report_path, serde_json::to_vec_pretty(&rep)?)?;
        println!("{}", serde_json::to_string_pretty(&rep)?);
        return Ok(());
    }

    let _ = verify_file_sha256(p4_path, PIN_P4_D11_SHA256, "p4.d11")?;
    let seed1_bytes = verify_file_sha256(seed1_path, PIN_SEED1_D11_SHA256, "seed1.d11")?;
    let tok_bytes = verify_file_sha256(tokenizer_path, PIN_TOKENIZER_SHA256, "tokenizer.json")?;
    let s1_state = P5S1State::from_bytes(&fs::read(s1_ckpt_path)?)?;

    let reloaded =
        load_client_artifact(p4_path, false).map_err(|e| fail(format!("load p4.d11: {e}")))?;
    let state_bytes = match reloaded {
        ClientArtifactFile::Embedded { state_bytes, .. } => state_bytes,
        ClientArtifactFile::Legacy(_) => return Err(fail("p4.d11 parsed as Legacy")),
    };
    let (p5_trunk_sha, artifact_sha, p5_bytes) =
        rebind_and_pack_p5_deployment(&state_bytes, &seed1_bytes, &s1_state, out_p5_path)?;
    let total_deployed = p5_bytes + (tok_bytes.len() as u64);
    let within_cap = total_deployed <= MAX_DEPLOYED_BYTES;

    let rep = json!({
        "schema": "uor-r4.p5-pack-status/1",
        "packed": true,
        "verdict": verdict,
        "p5_trunk_sha256": p5_trunk_sha,
        "p5_d11_path": out_p5_path.display().to_string(),
        "p5_d11_bytes": p5_bytes,
        "tokenizer_bytes": tok_bytes.len(),
        "total_deployed_bytes": total_deployed,
        "within_4gb_cap": within_cap,
        "artifact_sha256": artifact_sha,
    });
    fs::write(out_report_path, serde_json::to_vec_pretty(&rep)?)?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !within_cap {
        return Err(fail("p5.d11 + tokenizer exceeds 4.000 GB cap"));
    }
    Ok(())
}
