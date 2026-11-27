#[cfg(feature = "cuda")]
use crate::{
    cuda::{expand_a1s_to_s6_weights, split_uorp4a01_sections, P4GpuEngine},
    load_uort_tokens, verify_file_sha256,
};
use crate::{
    fail,
    lawful_likelihood::{bits_q12, masses_from_scores, LOG2_SCALE, PROBABILITY_ONE},
    Result,
};
use serde_json::json;
#[cfg(feature = "cuda")]
use std::fs;
use std::path::Path;
use uor_r4_count_runtime::{
    code_pick,
    kn_score::KN_BASE_MASS,
    p4_serving::{
        skip_keys_for_prefix, CausalSessionCache, P4State, P4_BASE_BUCKETS, P4_Q48_ONE, P4_VOCAB,
    },
};
#[cfg(feature = "cuda")]
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, ClientArtifactFile, DiscreteCountModel,
};
#[cfg(feature = "cuda")]
use uor_r4_tokenizer::ByteBpeTokenizer;

pub const PIN_HELDOUT_SHA256: &str =
    "e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef";
pub const PIN_HELDOUT_BYTES: usize = 12_389_242;
pub const PIN_HELDOUT_TOKENS: usize = 6_194_589;
pub const PIN_STREAM_BYTES: u128 = 17_576_697;
pub const BAR_A_CONTEXT: u32 = 384;
pub const BAR_A_MICRO_BPB: u64 = 900_000;

/// Exact integer floor(sqrt(n)) for u128.
pub fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x0 = 1u128 << ((128 - n.leading_zeros()).div_ceil(2));
    loop {
        let x1 = (x0 + n / x0) >> 1;
        if x1 >= x0 {
            return x0;
        }
        x0 = x1;
    }
}

/// Exact rounded micro-BPB on the chat-v0-p2 stream basis:
///   BPB = (total_bits_q12 / 4096) / (n_targets * 17_576_697 / 6_194_589)
///       = (total_bits_q12 * 6_194_589) / (4096 * n_targets * 17_576_697).
pub fn stream_basis_fraction(total_bits_q12: u128, n_targets: u64) -> (u128, u128, u64) {
    let num = total_bits_q12 * (PIN_HELDOUT_TOKENS as u128);
    let den = u128::from(LOG2_SCALE) * u128::from(n_targets) * PIN_STREAM_BYTES;
    let micro_bpb = ((num * 1_000_000u128) + (den / 2)) / den;
    (num, den, micro_bpb as u64)
}

pub fn sample_basis_fraction(total_bits_q12: u128, target_bytes: u64) -> (u128, u128, u64) {
    let num = total_bits_q12;
    let den = u128::from(LOG2_SCALE) * u128::from(target_bytes.max(1));
    let micro_bpb = ((num * 1_000_000u128) + (den / 2)) / den;
    (num, den, micro_bpb as u64)
}

/// Archived SE scaling from m4-lawful-likelihood-bpb-2026-10-23.md §3 (in micro-BPB).
pub fn archived_se_micro_bpb(windows: u32) -> u64 {
    match windows {
        64 => 41_083, // 14525 * sqrt(512/64) = 14525 * sqrt(8)
        512 => 14_525,
        1024 => 10_270,
        2048 => 7_262,
        other => {
            let sq = (14_525u128 * 14_525u128 * 512u128) / u128::from(other.max(1));
            isqrt_u128(sq) as u64
        }
    }
}

/// Empirical SE of the mean across per-window micro-BPBs (`sqrt(var / (W*(W-1)))`).
pub fn empirical_se_micro_bpb(window_micro_bpbs: &[u64]) -> u64 {
    let w = window_micro_bpbs.len() as u128;
    if w < 2 {
        return 0;
    }
    let sum: u128 = window_micro_bpbs.iter().map(|&x| u128::from(x)).sum();
    let mean = sum / w;
    let mut sq_diff_sum = 0u128;
    for &x in window_micro_bpbs {
        let xi = u128::from(x);
        let d = xi.abs_diff(mean);
        sq_diff_sum += d * d;
    }
    let var_of_mean = sq_diff_sum / (w * (w - 1));
    isqrt_u128(var_of_mean) as u64
}

pub fn interval_verdict(est_micro_bpb: u64, se_micro_bpb: u64) -> (u64, u64, &'static str) {
    // z = 1.96 -> (196 * se + 50) / 100
    let half_width = (196u64 * se_micro_bpb + 50) / 100;
    let lo = est_micro_bpb.saturating_sub(half_width);
    let hi = est_micro_bpb.saturating_add(half_width);
    let verdict = if hi <= BAR_A_MICRO_BPB {
        "PASS"
    } else if lo > BAR_A_MICRO_BPB {
        "FAIL"
    } else {
        "INCONCLUSIVE"
    };
    (lo, hi, verdict)
}

/// Pure-CPU evaluation of one exact Bar A window position (matching `p4_bar_a_windows_kernel`
/// and `uor_r4_count_runtime::p4_serving` operator-for-operator, including mid-dialogue EOS=1).
pub fn cpu_eval_bar_a_step(
    state: &P4State<'_>,
    override_s6: Option<&[u64]>,
    active_prefix: &[u16],
    cache: &CausalSessionCache,
    target: u16,
) -> Result<(u64, u64, u64)> {
    let pick = code_pick(&state.codes, active_prefix);
    let pick_succ = pick.map(|(p, _, _)| active_prefix[p + 1]);
    let pick_dist = pick.map(|(_, d, _)| d);
    let pick_class = state.a1s_weights.rule().classify(pick_dist);

    let mut s3_sparse = Vec::new();
    let (_s3_meta, s3_def) = state
        .s3
        .fill_sparse_masses_q48(active_prefix, &mut s3_sparse)
        .map_err(|e| fail(format!("cpu s3 error: {e}")))?;

    let skip_keys = skip_keys_for_prefix(active_prefix);
    let mut s2_sparse: [Vec<(u16, u64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut s2_def = [0u64; 3];
    for (fam, &sk) in skip_keys.iter().enumerate() {
        let (_hit, dm) = state
            .s2_skip
            .fill_sparse_family_q48(fam, sk, &mut s2_sparse[fam])
            .map_err(|e| fail(format!("cpu s2 error: {e}")))?;
        s2_def[fam] = dm;
    }

    let mut cache_sparse: [Vec<(u16, u64)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let (cache_hs, cache_def) = cache.eval_sparse_all(&mut cache_sparse);

    let mut kn_hits: Vec<(u16, u64, [u32; 11])> = Vec::with_capacity(P4_VOCAB);
    let (bucket_880, _highest, totals, default_kn) = state
        .counts
        .visit_p4_counts(active_prefix, pick_class, |id, kn, counts| {
            kn_hits.push((id, kn, *counts));
        })
        .map_err(|e| fail(format!("cpu counts error: {e}")))?;

    let bucket_2640 = usize::from(cache_hs) * P4_BASE_BUCKETS + usize::from(bucket_880);
    let w: &[u64] = match override_s6 {
        Some(flat) => &flat[bucket_2640 * 22..(bucket_2640 + 1) * 22],
        None => state
            .s6
            .bucket_weights(bucket_2640)
            .map_err(|e| fail(format!("cpu s6 error: {e}")))?,
    };

    let base_acc = u128::from(w[0]) * u128::from(default_kn)
        + u128::from(w[12]) * u128::from(KN_BASE_MASS)
        + u128::from(w[14]) * u128::from(s3_def)
        + u128::from(w[15]) * u128::from(s2_def[0])
        + u128::from(w[16]) * u128::from(s2_def[1])
        + u128::from(w[17]) * u128::from(s2_def[2])
        + u128::from(w[18]) * u128::from(cache_def[0])
        + u128::from(w[19]) * u128::from(cache_def[1])
        + u128::from(w[20]) * u128::from(cache_def[2])
        + u128::from(w[21]) * u128::from(cache_def[3]);
    let mut acc = vec![base_acc; P4_VOCAB];
    if let Some(ps) = pick_succ {
        if ps > 0 && usize::from(ps) < P4_VOCAB {
            acc[usize::from(ps)] += u128::from(w[13]) * u128::from(P4_Q48_ONE);
        }
    }
    let w0 = u128::from(w[0]);
    let d0 = w0 * u128::from(default_kn);
    for &(id, kn, ref counts) in &kn_hits {
        let idx = usize::from(id);
        let mut a = acc[idx] - d0 + w0 * u128::from(kn);
        for (j, (&cnt, &tot)) in counts.iter().zip(&totals).enumerate() {
            if tot > 0 && cnt > 0 && w[1 + j] > 0 {
                let m = ((u128::from(cnt) << 48) / u128::from(tot)) as u64;
                a += u128::from(w[1 + j]) * u128::from(m);
            }
        }
        acc[idx] = a;
    }
    if w[14] > 0 {
        let w14 = u128::from(w[14]);
        let d14 = w14 * u128::from(s3_def);
        for &(id, m) in &s3_sparse {
            let idx = usize::from(id);
            acc[idx] = acc[idx] - d14 + w14 * u128::from(m);
        }
    }
    for fam in 0..3usize {
        let wf = w[15 + fam];
        if wf > 0 {
            let wfu = u128::from(wf);
            let df = wfu * u128::from(s2_def[fam]);
            for &(id, m) in &s2_sparse[fam] {
                let idx = usize::from(id);
                acc[idx] = acc[idx] - df + wfu * u128::from(m);
            }
        }
    }
    for ci in 0..4usize {
        let wc = w[18 + ci];
        if wc > 0 {
            let wcu = u128::from(wc);
            let dc = wcu * u128::from(cache_def[ci]);
            for &(id, m) in &cache_sparse[ci] {
                let idx = usize::from(id);
                acc[idx] = acc[idx] - dc + wcu * u128::from(m);
            }
        }
    }

    let mut scores_i64 = vec![0i64; P4_VOCAB];
    for id in 1..P4_VOCAB {
        let q48 = ((acc[id] >> 32) as u64).clamp(1, P4_Q48_ONE);
        scores_i64[id] = q48 as i64;
    }
    let raw_p4_mass = scores_i64[usize::from(target)] as u64;
    let prll_masses =
        masses_from_scores(&scores_i64).map_err(|e| fail(format!("masses_from_scores: {e}")))?;
    let sum_check: u128 = prll_masses.iter().map(|&m| u128::from(m)).sum();
    if sum_check != u128::from(PROBABILITY_ONE) || prll_masses.contains(&0) {
        return Err(fail("PRLL invariant violation on CPU oracle"));
    }
    let pm = prll_masses[usize::from(target)];
    let bq = bits_q12(pm).map_err(|e| fail(format!("bits_q12: {e}")))?;
    Ok((raw_p4_mass, pm, bq))
}

fn summarize_window_run(
    tokens: &[u16],
    tok_byte_lens: &[u64],
    windows: u32,
    stride: u32,
    bits_q12_vec: &[u64],
    raw_p4_mass_vec: &[u64],
) -> Result<serde_json::Value> {
    let ctx = BAR_A_CONTEXT as usize;
    let total_targets = (windows as usize) * ctx;
    if bits_q12_vec.len() != total_targets || raw_p4_mass_vec.len() != total_targets {
        return Err(fail("bar A output length mismatch"));
    }
    let mut total_bits = 0u128;
    let mut total_sample_bytes = 0u64;
    let mut bos_targets = 0u64;
    let mut bos_bits = 0u128;
    let mut non_bos_targets = 0u64;
    let mut non_bos_prll_bits = 0u128;
    let mut non_bos_raw_bits = 0u128;
    let mut win_micro_bpbs = Vec::with_capacity(windows as usize);

    for w in 0..(windows as usize) {
        let start = w * (stride as usize);
        let mut w_bits = 0u128;
        for t in 0..ctx {
            let idx = w * ctx + t;
            let target = tokens[start + t + 1];
            let bq = u128::from(bits_q12_vec[idx]);
            w_bits += bq;
            total_bits += bq;
            total_sample_bytes += tok_byte_lens[usize::from(target)];
            if target == 0 {
                bos_targets += 1;
                bos_bits += bq;
            } else {
                non_bos_targets += 1;
                non_bos_prll_bits += bq;
                let rm = raw_p4_mass_vec[idx];
                non_bos_raw_bits += u128::from(
                    bits_q12(rm).map_err(|e| fail(format!("raw bits_q12 failed: {e}")))?,
                );
            }
        }
        let (_, _, w_mbpb) = stream_basis_fraction(w_bits, BAR_A_CONTEXT as u64);
        win_micro_bpbs.push(w_mbpb);
    }

    let (s_num, s_den, s_mbpb) = stream_basis_fraction(total_bits, total_targets as u64);
    let (t_num, t_den, t_mbpb) = sample_basis_fraction(total_bits, total_sample_bytes);
    let (_, _, non_bos_prll_mbpb) = stream_basis_fraction(non_bos_prll_bits, non_bos_targets);
    let (_, _, non_bos_raw_mbpb) = stream_basis_fraction(non_bos_raw_bits, non_bos_targets);

    let se_arch = archived_se_micro_bpb(windows);
    let se_emp = empirical_se_micro_bpb(&win_micro_bpbs);
    let (lo_arch, hi_arch, verdict_arch) = interval_verdict(s_mbpb, se_arch);
    let (lo_emp, hi_emp, verdict_emp) = interval_verdict(s_mbpb, se_emp);

    Ok(json!({
        "windows": windows,
        "context": BAR_A_CONTEXT,
        "stride": stride,
        "targets": total_targets,
        "bos_targets": bos_targets,
        "non_bos_targets": non_bos_targets,
        "total_bits_q12": total_bits.to_string(),
        "bos_bits_q12": bos_bits.to_string(),
        "non_bos_prll_bits_q12": non_bos_prll_bits.to_string(),
        "non_bos_raw_p4_bits_q12": non_bos_raw_bits.to_string(),
        "stream_basis": {
            "exact_numerator": s_num.to_string(),
            "exact_denominator": s_den.to_string(),
            "micro_bpb": s_mbpb,
            "archived_se_micro_bpb": se_arch,
            "archived_ci95_lo_micro_bpb": lo_arch,
            "archived_ci95_hi_micro_bpb": hi_arch,
            "archived_verdict": verdict_arch,
            "empirical_se_micro_bpb": se_emp,
            "empirical_ci95_lo_micro_bpb": lo_emp,
            "empirical_ci95_hi_micro_bpb": hi_emp,
            "empirical_verdict": verdict_emp,
        },
        "sample_target_byte_basis": {
            "scored_target_bytes": total_sample_bytes,
            "exact_numerator": t_num.to_string(),
            "exact_denominator": t_den.to_string(),
            "micro_bpb": t_mbpb,
        },
        "non_bos_diagnostic_stream_basis": {
            "prll_micro_bpb": non_bos_prll_mbpb,
            "raw_p4_micro_bpb": non_bos_raw_mbpb,
        }
    }))
}

#[cfg(feature = "cuda")]
pub fn run_bar_a(
    p4_path: &Path,
    heldout_path: &Path,
    tokenizer_path: &Path,
    out_path: &Path,
) -> Result<()> {
    let heldout_bytes = verify_file_sha256(heldout_path, PIN_HELDOUT_SHA256, "heldout.u16")?;
    if heldout_bytes.len() != PIN_HELDOUT_BYTES {
        return Err(fail(format!(
            "heldout.u16 length {} != {PIN_HELDOUT_BYTES}",
            heldout_bytes.len()
        )));
    }
    let tokens = load_uort_tokens(&heldout_bytes)?;
    if tokens.len() != PIN_HELDOUT_TOKENS {
        return Err(fail(format!(
            "heldout.u16 tokens {} != {PIN_HELDOUT_TOKENS}",
            tokens.len()
        )));
    }

    let tok_bytes =
        verify_file_sha256(tokenizer_path, crate::pack::PIN_TOKENIZER, "tokenizer.json")?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
        .ok_or_else(|| fail("failed to parse tokenizer.json"))?;
    let mut tok_byte_lens = vec![0u64; P4_VOCAB];
    for id in 0..P4_VOCAB {
        let piece = tokenizer.decode(&[id as u32]);
        tok_byte_lens[id] = piece.len() as u64;
    }

    let loaded = load_client_artifact(p4_path, false)
        .map_err(|e| fail(format!("load_client_artifact failed: {e}")))?;
    let (model, state_bytes, expected_state_sha, identity) = match loaded {
        ClientArtifactFile::Embedded {
            model,
            state_bytes,
            expected_state_sha256,
            identity,
        } => (model, state_bytes, expected_state_sha256, identity),
        ClientArtifactFile::Legacy(_) => return Err(fail("expected embedded P4 deployment")),
    };
    let bound =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha, identity)
            .map_err(|e| fail(format!("bind_embedded failed: {e}")))?;
    let p4 = bound
        .p4
        .as_ref()
        .ok_or_else(|| fail("artifact did not contain P4 state"))?;
    let raw = split_uorp4a01_sections(&state_bytes)?;
    let a1s_expanded = expand_a1s_to_s6_weights(p4.inner.a1s_weights.weights())?;

    let engine = P4GpuEngine::new(
        raw.counts_v3,
        raw.s3_compact,
        raw.s2_skip_compact,
        raw.s4_table,
        &p4.inner.s6.weights_q32,
        &p4.inner.codes,
    )?;

    let stride_512 = ((PIN_HELDOUT_TOKENS as u32) - BAR_A_CONTEXT - 1) / 512;
    let stride_64 = ((PIN_HELDOUT_TOKENS as u32) - BAR_A_CONTEXT - 1) / 64;
    assert_eq!(stride_512, 12_098);

    // 1. Primary 512 windows (P4)
    let p4_512 = engine.run_bar_a_windows(&tokens, 512, stride_512, BAR_A_CONTEXT, None)?;
    // 2. Determinism repeat on 512 windows (P4)
    let p4_512_repeat = engine.run_bar_a_windows(&tokens, 512, stride_512, BAR_A_CONTEXT, None)?;
    let det_diffs = p4_512
        .prll_bits_q12
        .iter()
        .zip(&p4_512_repeat.prll_bits_q12)
        .filter(|&(a, b)| a != b)
        .count();
    if det_diffs != 0 {
        return Err(fail(format!(
            "Bar A determinism repeat had {det_diffs} Q12 differences"
        )));
    }
    let p4_512_summary = summarize_window_run(
        &tokens,
        &tok_byte_lens,
        512,
        stride_512,
        &p4_512.prll_bits_q12,
        &p4_512.raw_p4_mass_q48,
    )?;

    // 3. CPU spot checks across all 512 windows (>= 512 positions, including BOS targets!)
    let mut cpu_spot_checked = 0usize;
    let mut cpu_spot_diffs = 0usize;
    let mut cpu_bos_checked = 0usize;
    let mut cpu_p4session_checked = 0usize;
    for w in 0..512usize {
        let start = w * (stride_512 as usize);
        let mut active_prefix: Vec<u16> = Vec::with_capacity(BAR_A_CONTEXT as usize);
        let mut cache = CausalSessionCache::new();
        let target_t = (w * 37) % (BAR_A_CONTEXT as usize);
        for t in 0..(BAR_A_CONTEXT as usize) {
            let tok = tokens[start + t];
            let target = tokens[start + t + 1];
            if tok == 0 {
                active_prefix.clear();
                active_prefix.push(0);
                cache.reset_record();
            } else {
                active_prefix.push(tok);
                cache.observe_token(tok);
            }
            if t == target_t || target == 0 {
                let (cpu_raw, cpu_prll, cpu_bits) =
                    cpu_eval_bar_a_step(&p4.inner, None, &active_prefix, &cache, target)?;
                let idx = w * (BAR_A_CONTEXT as usize) + t;
                if cpu_raw != p4_512.raw_p4_mass_q48[idx]
                    || cpu_prll != p4_512.prll_mass_q48[idx]
                    || cpu_bits != p4_512.prll_bits_q12[idx]
                {
                    cpu_spot_diffs += 1;
                }
                cpu_spot_checked += 1;
                if target == 0 {
                    cpu_bos_checked += 1;
                }
                if !active_prefix.contains(&1) {
                    let mut sess = uor_r4_count_runtime::p4_serving::P4Session::new();
                    sess.sync_prompt(&active_prefix)
                        .map_err(|e| fail(format!("P4Session spot sync failed: {e}")))?;
                    sess.step(&p4.inner, true, false)
                        .map_err(|e| fail(format!("P4Session spot step failed: {e}")))?;
                    let sc = sess.last_scores_q48()[usize::from(target)];
                    if sc != cpu_raw {
                        cpu_spot_diffs += 1;
                    }
                    cpu_p4session_checked += 1;
                }
            }
        }
    }
    if cpu_spot_diffs != 0 {
        return Err(fail(format!(
            "Bar A CPU spot checks found {cpu_spot_diffs} diffs out of {cpu_spot_checked}"
        )));
    }

    // 4. Secondary 64-window diagnostic (P4)
    let p4_64 = engine.run_bar_a_windows(&tokens, 64, stride_64, BAR_A_CONTEXT, None)?;
    let p4_64_summary = summarize_window_run(
        &tokens,
        &tok_byte_lens,
        64,
        stride_64,
        &p4_64.prll_bits_q12,
        &p4_64.raw_p4_mass_q48,
    )?;

    // 5. P1(A1s) comparison on identical 512 windows
    let a1s_512 =
        engine.run_bar_a_windows(&tokens, 512, stride_512, BAR_A_CONTEXT, Some(&a1s_expanded))?;
    let a1s_512_summary = summarize_window_run(
        &tokens,
        &tok_byte_lens,
        512,
        stride_512,
        &a1s_512.prll_bits_q12,
        &a1s_512.raw_p4_mass_q48,
    )?;

    // 6. Auto-extension to 1024 / 2048 windows ONLY if 512w archived CI straddles 0.90 BPB
    let mut ext_1024 = serde_json::Value::Null;
    let mut ext_2048 = serde_json::Value::Null;
    let v512 = p4_512_summary["stream_basis"]["archived_verdict"]
        .as_str()
        .unwrap_or("FAIL");
    let mut final_verdict = v512.to_owned();
    let mut final_windows = 512u32;

    if v512 == "INCONCLUSIVE" {
        let stride_1024 = ((PIN_HELDOUT_TOKENS as u32) - BAR_A_CONTEXT - 1) / 1024;
        let r1024 = engine.run_bar_a_windows(&tokens, 1024, stride_1024, BAR_A_CONTEXT, None)?;
        ext_1024 = summarize_window_run(
            &tokens,
            &tok_byte_lens,
            1024,
            stride_1024,
            &r1024.prll_bits_q12,
            &r1024.raw_p4_mass_q48,
        )?;
        let v1024 = ext_1024["stream_basis"]["archived_verdict"]
            .as_str()
            .unwrap_or("INCONCLUSIVE");
        final_verdict = v1024.to_owned();
        final_windows = 1024;
        if v1024 == "INCONCLUSIVE" {
            let stride_2048 = ((PIN_HELDOUT_TOKENS as u32) - BAR_A_CONTEXT - 1) / 2048;
            let r2048 =
                engine.run_bar_a_windows(&tokens, 2048, stride_2048, BAR_A_CONTEXT, None)?;
            ext_2048 = summarize_window_run(
                &tokens,
                &tok_byte_lens,
                2048,
                stride_2048,
                &r2048.prll_bits_q12,
                &r2048.raw_p4_mass_q48,
            )?;
            final_verdict = ext_2048["stream_basis"]["archived_verdict"]
                .as_str()
                .unwrap_or("INCONCLUSIVE")
                .to_owned();
            final_windows = 2048;
        }
    }

    // Uniform sanity floor (log2(4096) = 12 bits/tok = 49152 Q12/tok)
    let uni_bits_per_tok = bits_q12(PROBABILITY_ONE / (P4_VOCAB as u64))
        .map_err(|e| fail(format!("uniform bits_q12: {e}")))?;
    let uni_512_total = u128::from(uni_bits_per_tok) * (512u128 * u128::from(BAR_A_CONTEXT));
    let (u_num, u_den, u_mbpb) =
        stream_basis_fraction(uni_512_total, 512 * u64::from(BAR_A_CONTEXT));

    let report = json!({
        "schema": "uor-r4.p4-bar-a-prll/1",
        "cubin_sha256": engine.cubin_sha256,
        "p4_artifact_sha256": bound.identity.as_ref().map(|i| i.artifact_sha256.as_str()),
        "p4_state_sha256": p4.artifact_sha256,
        "heldout_sha256": PIN_HELDOUT_SHA256,
        "heldout_tokens": PIN_HELDOUT_TOKENS,
        "stream_bytes": PIN_STREAM_BYTES.to_string(),
        "bar_micro_bpb": BAR_A_MICRO_BPB,
        "determinism_repeat_diffs": det_diffs,
        "cpu_spot_checks": {
            "positions_checked": cpu_spot_checked,
            "bos_targets_checked": cpu_bos_checked,
            "p4session_cross_checked": cpu_p4session_checked,
            "diffs": cpu_spot_diffs,
        },
        "uniform_floor_512w": {
            "bits_q12_per_token": uni_bits_per_tok,
            "exact_numerator": u_num.to_string(),
            "exact_denominator": u_den.to_string(),
            "micro_bpb": u_mbpb,
        },
        "p4_primary_512w": p4_512_summary,
        "p4_diagnostic_64w": p4_64_summary,
        "p1_a1s_comparison_512w": a1s_512_summary,
        "extension_1024w": ext_1024,
        "extension_2048w": ext_2048,
        "final_windows": final_windows,
        "final_bar_a_verdict": final_verdict,
    });
    fs::write(out_path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run_bar_a(
    _p4_path: &Path,
    _heldout_path: &Path,
    _tokenizer_path: &Path,
    _out_path: &Path,
) -> Result<()> {
    Err(fail("bar-a requires --features cuda"))
}
