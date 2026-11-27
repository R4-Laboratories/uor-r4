#![forbid(unsafe_code)]

use crate::lawful_likelihood::bits_q12;
use crate::{driver_check, fail, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};
use uor_r4_count_runtime::{
    p3_state::{s4_pair_key, BrierS6Weights, S4ConjunctionTable, S4PointerEntry, P3_Q32_ONE},
    p4_serving::{
        encode_compact_s2_skip_from_uors2s01, encode_compact_s3_from_uors3t02, golden_shell_pkey,
        skip_keys_for_prefix, starling_7_neighbours, CompactS2SkipView, CompactS3View, P4Session,
        P4State, H4_ROOTS, P4_EXPERTS, P4_MAX_CONTEXT, P4_TOTAL_BUCKETS, P4_VOCAB,
    },
    CompactCountLookup, FrozenPickMixture, PickClassRule,
};
#[cfg(feature = "cuda")]
use uor_r4_discrete::p5_brier_core::{
    batch_brier_q32, p5_bvgk_step_cpu, p5_orbit_tables, p5_score_block_cpu, p5_step_control_cpu,
    P5BatchBrier, P5ControlMode, P5RowBrier,
};
use uor_r4_discrete::p5_brier_core::{full_vocab_brier_q32, P5S1State, P5_BLOCKS, P5_VOCAB};

#[cfg(feature = "cuda")]
use crate::{
    cuda::{P5GpuEngine, RawP4Sections, RowStepCmdHost, P5_MAX_PICK_POS},
    train_eval::{bvgk_step_from_gpu_scores, split_records_on_bos},
};

type SynthRows<'a> = &'a [(u16, u32, i8)];

fn build_synth_uors3t02(
    token_h4: &[u8; P4_VOCAB],
    train_sha: [u8; 32],
    primary: &[(u32, SynthRows<'_>)],
    backoff: &[(u16, SynthRows<'_>)],
) -> Vec<u8> {
    let mut p_rows = Vec::new();
    let mut p_ctxs = Vec::new();
    for &(k, sl) in primary {
        let rs = p_rows.len() as u32;
        let rl = sl.len() as u16;
        let tot: u64 = sl.iter().map(|&(_, c, _)| u64::from(c)).sum();
        p_ctxs.push((k, tot, rs, rl));
        p_rows.extend_from_slice(sl);
    }
    let mut b_rows = Vec::new();
    let mut b_ctxs = Vec::new();
    for &(pk, sl) in backoff {
        let rs = b_rows.len() as u32;
        let rl = sl.len() as u16;
        let tot: u64 = sl.iter().map(|&(_, c, _)| u64::from(c)).sum();
        b_ctxs.push((pk, tot, rs, rl));
        b_rows.extend_from_slice(sl);
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"UORS3T02");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.push(7);
    out.push(0);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&train_sha);
    out.extend_from_slice(&(p_ctxs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(p_rows.len() as u32).to_le_bytes());
    out.extend_from_slice(&(b_ctxs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(b_rows.len() as u32).to_le_bytes());
    out.extend_from_slice(token_h4);
    for (k, tot, rs, rl) in p_ctxs {
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&tot.to_le_bytes());
        out.extend_from_slice(&rs.to_le_bytes());
        out.extend_from_slice(&rl.to_le_bytes());
    }
    for (nt, cnt, err) in p_rows {
        out.extend_from_slice(&nt.to_le_bytes());
        out.extend_from_slice(&cnt.to_le_bytes());
        out.push(err as u8);
    }
    for (pk, tot, rs, rl) in b_ctxs {
        out.extend_from_slice(&pk.to_le_bytes());
        out.extend_from_slice(&tot.to_le_bytes());
        out.extend_from_slice(&rs.to_le_bytes());
        out.extend_from_slice(&rl.to_le_bytes());
    }
    for (nt, cnt, err) in b_rows {
        out.extend_from_slice(&nt.to_le_bytes());
        out.extend_from_slice(&cnt.to_le_bytes());
        out.push(err as u8);
    }
    out
}

fn build_synth_uors2s01(c_sha: [u8; 32], families: [&[(u64, SynthRows<'_>)]; 3]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"UORS2S01");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&c_sha);
    for (fid, fam) in families.iter().enumerate() {
        out.push(fid as u8);
        let mut rows = Vec::new();
        let mut ctxs = Vec::new();
        for &(k, sl) in *fam {
            let rs = rows.len() as u32;
            let rl = sl.len() as u16;
            let tot: u64 = sl.iter().map(|&(_, c, _)| u64::from(c)).sum();
            ctxs.push((k, rs, rl, tot));
            rows.extend_from_slice(sl);
        }
        out.extend_from_slice(&(ctxs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for (k, rs, rl, tot) in ctxs {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&rs.to_le_bytes());
            out.extend_from_slice(&rl.to_le_bytes());
            out.extend_from_slice(&tot.to_le_bytes());
        }
        for (nt, cnt, err) in rows {
            out.extend_from_slice(&nt.to_le_bytes());
            out.extend_from_slice(&cnt.to_le_bytes());
            out.push(err as u8);
        }
    }
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    out
}

fn build_synth_uorknc03(c_sha: [u8; 32]) -> Vec<u8> {
    let highest = 2u32;
    let mut out = Vec::new();
    out.extend_from_slice(b"UORKNC03");
    out.extend_from_slice(&highest.to_le_bytes());
    out.extend_from_slice(&c_sha);

    let mut o0_fols: Vec<(u16, u32, u32)> = Vec::new();
    for i in 1..=67u16 {
        o0_fols.push((i, 1 + u32::from(i % 4), 1 + u32::from(i % 2)));
    }
    o0_fols.push((100, 200, 40));
    o0_fols.push((500, 80_000, 70_000));
    o0_fols.push((4095, 3, 2));

    let push_order = |buf: &mut Vec<u8>,
                      k: u32,
                      shared: bool,
                      parent_ctxs: u32,
                      children: &[(u32, u16)],
                      ctx_fols: &[Vec<(u16, u32, u32)>]| {
        let contexts = ctx_fols.len() as u32;
        let followers: u32 = ctx_fols.iter().map(|v| v.len() as u32).sum();
        let mut rspec = [0u64; 4];
        let mut aspec = [0u64; 4];
        for list in ctx_fols {
            for &(_, r, a) in list {
                if r <= 4 {
                    rspec[(r - 1) as usize] += 1;
                }
                if a <= 4 {
                    aspec[(a - 1) as usize] += 1;
                }
            }
        }
        let mut fol_words: Vec<u16> = Vec::with_capacity(followers as usize);
        let mut esc16_bytes: Vec<u8> = Vec::new();
        let mut esc32_bytes: Vec<u8> = Vec::new();
        let mut esc_sb: Vec<u32> = Vec::new();
        let mut esc16_count = 0u32;
        let mut esc32_count = 0u32;
        let mut flat_idx = 0u32;
        for list in ctx_fols {
            for &(tok, raw, kn) in list {
                if flat_idx.is_multiple_of(64) {
                    esc_sb.push(esc16_count);
                }
                let inline_code = if shared {
                    if raw == kn && (1..=15).contains(&raw) {
                        Some(raw - 1)
                    } else {
                        None
                    }
                } else {
                    match (raw, kn) {
                        (1, 1) => Some(0),
                        (2, 1) => Some(1),
                        (2, 2) => Some(2),
                        (3, 1) => Some(3),
                        (3, 2) => Some(4),
                        (3, 3) => Some(5),
                        (4, 1) => Some(6),
                        (4, 2) => Some(7),
                        (4, 3) => Some(8),
                        (4, 4) => Some(9),
                        (5, 1) => Some(10),
                        (5, 2) => Some(11),
                        (5, 3) => Some(12),
                        (6, 1) => Some(13),
                        (6, 2) => Some(14),
                        _ => None,
                    }
                };
                if let Some(code) = inline_code {
                    fol_words.push((tok << 4) | (code as u16));
                } else {
                    fol_words.push((tok << 4) | 15u16);
                    if shared {
                        if raw <= u32::from(u16::MAX) {
                            esc16_bytes.extend_from_slice(&(raw as u16).to_le_bytes());
                        } else {
                            esc16_bytes.extend_from_slice(&0u16.to_le_bytes());
                            esc32_bytes.extend_from_slice(&flat_idx.to_le_bytes());
                            esc32_bytes.extend_from_slice(&raw.to_le_bytes());
                            esc32_count += 1;
                        }
                    } else if raw <= u32::from(u16::MAX) && kn <= u32::from(u16::MAX) {
                        esc16_bytes.extend_from_slice(&(raw as u16).to_le_bytes());
                        esc16_bytes.extend_from_slice(&(kn as u16).to_le_bytes());
                    } else {
                        esc16_bytes.extend_from_slice(&0u16.to_le_bytes());
                        esc16_bytes.extend_from_slice(&0u16.to_le_bytes());
                        esc32_bytes.extend_from_slice(&flat_idx.to_le_bytes());
                        esc32_bytes.extend_from_slice(&raw.to_le_bytes());
                        esc32_bytes.extend_from_slice(&kn.to_le_bytes());
                        esc32_count += 1;
                    }
                    esc16_count += 1;
                }
                flat_idx += 1;
            }
        }
        esc_sb.push(esc16_count);

        buf.extend_from_slice(&followers.to_le_bytes());
        buf.extend_from_slice(&contexts.to_le_bytes());
        for w in rspec.into_iter().chain(aspec) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&esc16_count.to_le_bytes());
        buf.extend_from_slice(&esc32_count.to_le_bytes());

        if k > 0 {
            let mut p_deg = vec![0u8; parent_ctxs as usize];
            for &(pidx, _) in children {
                p_deg[pidx as usize] += 1;
            }
            let mut p_sb = Vec::new();
            let mut running = 0u32;
            for (i, &d) in p_deg.iter().enumerate() {
                if i % 64 == 0 {
                    p_sb.push(running);
                }
                running += u32::from(d);
            }
            p_sb.push(running);
            for w in p_sb {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&p_deg);
            let n_c = children.len();
            let mut packed = vec![0u8; (n_c * 3).div_ceil(2)];
            for (idx, &(_, tok)) in children.iter().enumerate() {
                let b = (idx >> 1) * 3;
                if idx & 1 == 0 {
                    packed[b] = (tok & 0xff) as u8;
                    packed[b + 1] |= ((tok >> 8) & 0x0f) as u8;
                } else {
                    packed[b + 1] |= ((tok & 0x0f) << 4) as u8;
                    packed[b + 2] = (tok >> 4) as u8;
                }
            }
            buf.extend_from_slice(&packed);
        }
        let mut f_sb = Vec::new();
        let mut f_deg = Vec::with_capacity(contexts as usize);
        let mut run = 0u32;
        for (i, list) in ctx_fols.iter().enumerate() {
            if i % 64 == 0 {
                f_sb.push(run);
            }
            f_deg.push((list.len() - 1) as u8);
            run += list.len() as u32;
        }
        f_sb.push(run);
        for w in f_sb {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&f_deg);
        for w in fol_words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for w in esc_sb {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&esc16_bytes);
        buf.extend_from_slice(&esc32_bytes);
    };

    push_order(&mut out, 0, false, 0, &[], &[o0_fols]);
    push_order(
        &mut out,
        1,
        false,
        1,
        &[(0, 5), (0, 10), (0, 42)],
        &[
            vec![(7, 3, 2), (10, 5, 3), (100, 12, 8)],
            vec![(12, 2, 1), (42, 4, 2)],
            vec![(5, 1, 1), (500, 90_000, 12_000)],
        ],
    );
    push_order(
        &mut out,
        2,
        true,
        3,
        &[(1, 5), (2, 10)],
        &[
            vec![(42, 7, 7), (500, 70_000, 70_000)],
            vec![(1, 2, 2), (100, 300, 300)],
        ],
    );
    out
}

pub struct SyntheticP5Bundle {
    pub counts_v3: Vec<u8>,
    pub s3_compact: Vec<u8>,
    pub s2_compact: Vec<u8>,
    pub s4_bytes: Vec<u8>,
    pub s6_bytes: Vec<u8>,
    pub a1s_bytes: Vec<u8>,
    pub fit_digest: [u8; 32],
    pub c_sha: [u8; 32],
    pub r_sha: [u8; 32],
}

pub fn build_synthetic_p5_bundle() -> Result<SyntheticP5Bundle> {
    let c_sha = [0x33u8; 32];
    let r_sha = [0x44u8; 32];
    let fit_digest = [0x55u8; 32];

    let counts_v3 = build_synth_uorknc03(c_sha);

    let mut token_h4 = [0u8; P4_VOCAB];
    for (i, slot) in token_h4.iter_mut().enumerate() {
        *slot = (i % H4_ROOTS) as u8;
    }
    let p_prim: [u16; 6] = [0, 5, 10, 42, 10, 5];
    let pk_prim = golden_shell_pkey(&token_h4, &p_prim);
    let p_back: [u16; 6] = [0, 7, 11, 13, 17, 19];
    let pk_back = golden_shell_pkey(&token_h4, &p_back);
    let bk_back = (pk_back >> 16) as u16;
    let p_star: [u16; 6] = [0, 21, 23, 25, 27, 29];
    let pk_star = golden_shell_pkey(&token_h4, &p_star);
    let bk_star = (pk_star >> 16) as u16;
    let (nbrs, nn) = starling_7_neighbours(bk_star % (9 * 9 * 9 * 9));
    let mut b_entries: BTreeMap<u16, Vec<(u16, u32, i8)>> = BTreeMap::new();
    b_entries.insert(bk_back, vec![(7, 14, 25), (42, 300, -40)]);
    if nn >= 2 {
        b_entries.insert(nbrs[0], vec![(12, 9, 15), (100, 20, -10)]);
        b_entries.insert(nbrs[1], vec![(12, 5, -5), (500, 17, 60)]);
    }
    let b_refs: Vec<(u16, SynthRows<'_>)> =
        b_entries.iter().map(|(&k, v)| (k, v.as_slice())).collect();
    let p_rows: [(u16, u32, i8); 3] = [(10, 4, 30), (42, 70_000, -120), (4095, 1, 127)];
    let s3_raw = build_synth_uors3t02(&token_h4, c_sha, &[(pk_prim, &p_rows)], &b_refs);
    let s3_compact = encode_compact_s3_from_uors3t02(&s3_raw, None, None, c_sha)
        .map_err(|e| fail(format!("synth s3 encode: {e}")))?;

    let skeys = skip_keys_for_prefix(&p_prim);
    let sr0: [(u16, u32, i8); 2] = [(7, 10, 12), (42, 500, -20)];
    let sr1: [(u16, u32, i8); 2] = [(10, 3, -100), (100, 15, 45)];
    let sr2: [(u16, u32, i8); 2] = [(12, 2, 0), (500, 80_000, 64)];
    let sf0 = [(skeys[0], &sr0[..])];
    let sf1 = [(skeys[1], &sr1[..])];
    let sf2 = [(skeys[2], &sr2[..])];
    let s2_raw = build_synth_uors2s01(c_sha, [&sf0, &sf1, &sf2]);
    let s2_compact = encode_compact_s2_skip_from_uors2s01(&s2_raw, None, c_sha)
        .map_err(|e| fail(format!("synth s2 encode: {e}")))?;

    let k1 = s4_pair_key(5, 10).map_err(|e| fail(format!("s4 key: {e}")))?;
    let mut s4_rows_map: BTreeMap<u32, Vec<S4PointerEntry>> = BTreeMap::new();
    s4_rows_map.insert(
        k1,
        vec![S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 8,
            err_i8: 12,
        }],
    );
    let s4_rows_ref: Vec<(u32, &[S4PointerEntry])> = s4_rows_map
        .iter()
        .map(|(&k, v)| (k, v.as_slice()))
        .collect();
    let s4_table = S4ConjunctionTable::from_sorted_rows(c_sha, &s4_rows_ref)
        .map_err(|e| fail(format!("synth s4: {e}")))?;
    let s4_bytes = s4_table
        .encode_bytes()
        .map_err(|e| fail(format!("synth s4 encode: {e}")))?;

    // Strong, distinct Pick weights across classes 1..5 so S1 Pick moves cleanly alter full-vocab Brier.
    let pick_w_by_pc: [u64; 5] = [
        1,               // pc=0 (no pick): valid >= 1 simplex floor
        P3_Q32_ONE / 3,  // pc=1 (d=0): ~0.333
        P3_Q32_ONE / 6,  // pc=2 (d=1): ~0.167
        P3_Q32_ONE / 12, // pc=3 (d=2): ~0.083
        P3_Q32_ONE / 32, // pc=4 (d>=3): ~0.031
    ];
    let mut s6_weights = vec![0u64; P4_TOTAL_BUCKETS * P4_EXPERTS];
    for b in 0..P4_TOTAL_BUCKETS {
        let base = b * P4_EXPERTS;
        let pc = (b % 880) / 176;
        let w13 = pick_w_by_pc[pc];
        let rem_total = P3_Q32_ONE - w13;
        let share = rem_total / ((P4_EXPERTS - 1) as u64);
        let mut sum = w13;
        for e in 0..P4_EXPERTS {
            if e == 13 {
                s6_weights[base + e] = w13;
            } else {
                let v = share + ((b + e) as u64 % 13);
                s6_weights[base + e] = v;
                sum += v;
            }
        }
        if sum > P3_Q32_ONE {
            s6_weights[base] -= sum - P3_Q32_ONE;
        } else {
            s6_weights[base] += P3_Q32_ONE - sum;
        }
    }
    let mut s6_obj = BrierS6Weights::uniform(P4_TOTAL_BUCKETS, P4_EXPERTS, r_sha)
        .map_err(|e| fail(format!("synth s6: {e}")))?;
    s6_obj.weights_q32 = s6_weights;
    let s6_bytes = s6_obj
        .encode_bytes()
        .map_err(|e| fail(format!("synth s6 encode: {e}")))?;

    let mut a1s_w = vec![0u64; (3 * 16) * 5 * 6];
    for b in 0..(48 * 5) {
        a1s_w[b * 6] = P3_Q32_ONE;
    }
    let a1s = FrozenPickMixture::with_rule(2, fit_digest, a1s_w, PickClassRule::Five)
        .map_err(|e| fail(format!("synth a1s: {e}")))?;
    let a1s_bytes = a1s.to_bytes();

    Ok(SyntheticP5Bundle {
        counts_v3,
        s3_compact,
        s2_compact,
        s4_bytes,
        s6_bytes,
        a1s_bytes,
        fit_digest,
        c_sha,
        r_sha,
    })
}

pub fn make_p4_state_with_s1<'a>(b: &'a SyntheticP5Bundle, s1: &P5S1State) -> Result<P4State<'a>> {
    let counts_sha: [u8; 32] = Sha256::digest(&b.counts_v3).into();
    let a1s_sha: [u8; 32] = Sha256::digest(&b.a1s_bytes).into();
    let s4_sha: [u8; 32] = Sha256::digest(&b.s4_bytes).into();
    let s6_sha: [u8; 32] = Sha256::digest(&b.s6_bytes).into();
    let counts = CompactCountLookup::from_bytes(&b.counts_v3, b.c_sha, counts_sha)
        .map_err(|e| fail(format!("synth counts load: {e}")))?;
    let a1s = FrozenPickMixture::from_bytes(&b.a1s_bytes, b.fit_digest)
        .map_err(|e| fail(format!("synth a1s load: {e}")))?;
    let s3 = CompactS3View::from_bytes(&b.s3_compact)
        .map_err(|e| fail(format!("synth s3 load: {e}")))?;
    let s2 = CompactS2SkipView::from_bytes(&b.s2_compact)
        .map_err(|e| fail(format!("synth s2 load: {e}")))?;
    let s4 = S4ConjunctionTable::from_bytes(&b.s4_bytes, b.c_sha, s4_sha)
        .map_err(|e| fail(format!("synth s4 load: {e}")))?;
    let s6 = BrierS6Weights::from_bytes(&b.s6_bytes, b.r_sha, s6_sha)
        .map_err(|e| fail(format!("synth s6 load: {e}")))?;
    let codes = s1.derive_pick_codes_4()?;
    P4State::from_components(
        counts,
        counts_sha,
        a1s,
        a1s_sha,
        b.fit_digest,
        s3,
        s2,
        s4,
        s4_sha,
        s6,
        s6_sha,
        codes,
    )
    .map_err(|e| fail(format!("synth P4State: {e}")))
}

pub type CpuPickTuple = (u32, u32, u8, u16);

/// Evaluate exact CPU `P4Session::step` + `full_vocab_brier_q32` + `bits_q12` for one
/// causal history window (exact D85 serving oracle).
pub fn cpu_eval_p4_window_brier_and_bits(
    state: &P4State<'_>,
    window: &[u16],
    target: u16,
) -> Result<(u64, u64, Option<CpuPickTuple>)> {
    let mut sess = P4Session::new();
    sess.sync_prompt(window)
        .map_err(|e| fail(format!("CPU P4Session::sync_prompt: {e}")))?;
    let (_tok, tr) = sess
        .step(state, true, true)
        .map_err(|e| fail(format!("CPU P4Session::step: {e}")))?;
    let tr = tr.ok_or_else(|| fail("missing CPU P4StepTrace"))?;
    let scores = sess.last_scores_q48();
    let brier = full_vocab_brier_q32(scores, target)?;
    let bq = bits_q12(scores[usize::from(target)])?;
    let pick_info = tr.pick_pos.map(|p| {
        (
            p,
            tr.pick_distance.unwrap_or(0),
            tr.pick_class,
            tr.pick_successor.unwrap_or(0),
        )
    });
    Ok((brier, bq, pick_info))
}

#[cfg(feature = "cuda")]
fn reconstruct_batch_brier_from_gpu(
    num_rows: usize,
    wins: &[u16],
    lens: &[u32],
    tgts: &[u16],
    no_pick: &[u64],
    pick: &[u64],
) -> Result<P5BatchBrier> {
    let mut rows = Vec::with_capacity(num_rows);
    for r in 0..num_rows {
        let len = lens[r] as usize;
        let w = wins[r * P4_MAX_CONTEXT..r * P4_MAX_CONTEXT + len].to_vec();
        let pos_len = len.saturating_sub(2);
        let mut pb = vec![0u64; 4 * pos_len];
        for s in 0..4 {
            let src_off = (r * 4 + s) * P5_MAX_PICK_POS;
            pb[s * pos_len..(s + 1) * pos_len].copy_from_slice(&pick[src_off..src_off + pos_len]);
        }
        rows.push(P5RowBrier {
            window: w,
            target: tgts[r],
            no_pick_brier_q32: no_pick[r],
            pick_brier_q32: pb,
        });
    }
    Ok(P5BatchBrier::new(rows)?)
}

pub fn run_finite(out_json: &Path) -> Result<()> {
    let bundle = build_synthetic_p5_bundle()?;
    let driver_res = driver_check::run_driver_4096_check()?;

    // Build synthetic multi-length records covering L = 1, 2, 3, S3 primary/backoff/Starling/uniform,
    // S2-skip, 64-token cache decay, 2,048-token sliding window, and planted Pick copy targets.
    let mut rec_slide = vec![0u16; P4_MAX_CONTEXT + 8];
    for (i, slot) in rec_slide.iter_mut().enumerate().skip(1) {
        *slot = match i % 13 {
            0 => 5,
            1 => 10,
            2 => 42,
            3 => 7,
            4 => 12,
            5 => 100,
            6 => 500,
            7 => 11,
            8 => 13,
            9 => 17,
            10 => 19,
            11 => 21,
            _ => 23,
        };
    }
    // Planted Pick copy rows: [0, 2, 3, 4, 9, 6, 8, 3, 7] -> target 9 (merging tok 4 & 7 drops Brier!)
    let mut planted_stream: Vec<u16> = Vec::new();
    // Record 0: short S3 primary + S2 hit
    planted_stream.extend_from_slice(&[0, 5, 10, 42, 10, 5, 42, 10]);
    // Record 1: S3 backoff
    planted_stream.extend_from_slice(&[0, 7, 11, 13, 17, 19, 7]);
    // Record 2: S3 Starling-7
    planted_stream.extend_from_slice(&[0, 21, 23, 25, 27, 29, 12]);
    // Record 3: S3 uniform miss
    planted_stream.extend_from_slice(&[0, 999, 1001, 1003, 1005, 1007, 42]);
    // Records 4..11: planted Pick alignment records so K-coordinate blocks trigger |S_t| >= 2
    for rep in 0..8u16 {
        planted_stream.extend_from_slice(&[
            0,
            2,
            3,
            4,
            9,
            6,
            8,
            3,
            7,
            9,
            14,
            15,
            16,
            18,
            20,
            22,
            15,
            24,
            18,
            25 + rep,
        ]);
    }
    // Record 12: 2056-token record crossing 64-tok cache decay and 2048-tok sliding window
    planted_stream.extend_from_slice(&rec_slide);

    let s1_base = P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 4390, [0x11; 32])?;

    #[cfg(feature = "cuda")]
    {
        let raw = RawP4Sections {
            fit_json: &[],
            a1s_weights: &bundle.a1s_bytes,
            counts_v3: &bundle.counts_v3,
            s3_compact: &bundle.s3_compact,
            s2_skip_compact: &bundle.s2_compact,
            s4_table: &bundle.s4_bytes,
            s6_weights: &bundle.s6_bytes,
        };
        let mut eng = P5GpuEngine::new(0, &raw, 256, 16)?;
        eng.upload_train_tokens(&planted_stream)?;
        let cubin_sha = eng.cubin_sha256().to_owned();

        // Build stateless row commands for a rich 101-row test batch covering L = 1, 2, 3, ..., S3, S2, cache!
        let recs = split_records_on_bos(&planted_stream)?;
        let mut cmds = Vec::new();
        for &rec_i in &[0usize, 1, 2, 3, 4, 5, 6, 7] {
            let (off, len) = recs[rec_i];
            for pos in 1..len {
                if cmds.len() >= 120 {
                    break;
                }
                cmds.push(RowStepCmdHost {
                    rec_off: off as u32,
                    pos: pos as u32,
                    target: planted_stream[off + pos],
                    _pad0: 0,
                    _pad1: 0,
                });
            }
        }
        let num_rows = cmds.len();
        eng.materialize_batch(&cmds)?;
        let (wins, lens, tgts, no_pick, pick_tbl) = eng.readback_batch_brier(num_rows)?;
        let cpu_batch =
            reconstruct_batch_brier_from_gpu(num_rows, &wins, &lens, &tgts, &no_pick, &pick_tbl)?;

        // Check 1: GPU-materialized P5RowBrier vs live CPU P4Session::step + full_vocab_brier_q32
        // across EVERY row in the batch AND across 2 distinct S1 states!
        let s1_alt = P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 4399, [0x11; 32])?;
        let mut row_brier_diffs = 0usize;
        for s1_check in [&s1_base, &s1_alt] {
            let p4_state = make_p4_state_with_s1(&bundle, s1_check)?;
            let gpu_sum = eng.eval_roots_on_batch(num_rows as u32, &s1_check.roots)?;
            let cpu_sum = batch_brier_q32(s1_check, &cpu_batch)?;
            if gpu_sum != cpu_sum {
                return Err(fail(format!(
                    "Gate 2 batch_eval sum mismatch: gpu={gpu_sum} cpu={cpu_sum}"
                )));
            }
            let codes = s1_check.derive_pick_codes_4()?;
            let mut direct_sum = 0u64;
            for row in &cpu_batch.rows {
                let (cpu_brier, _, _) =
                    cpu_eval_p4_window_brier_and_bits(&p4_state, &row.window, row.target)?;
                let pick = uor_r4_discrete::kn_pick::code_pick(&codes, &row.window)
                    .map(|(p, d, _)| (p, d));
                let tbl_brier = row.brier_for_pick(pick);
                if cpu_brier != tbl_brier {
                    row_brier_diffs += 1;
                }
                direct_sum += cpu_brier;
            }
            if direct_sum != gpu_sum {
                return Err(fail(format!(
                    "Gate 2 P4Session direct sum mismatch: direct={direct_sum} gpu={gpu_sum}"
                )));
            }
        }
        if row_brier_diffs != 0 {
            return Err(fail(format!(
                "Gate 2 found {row_brier_diffs} row Brier diffs vs P4Session::step"
            )));
        }

        // Check 2: rows-K GPU vs CPU exact parity across K = 1, 8, 16, 64!
        let mut rowsk_checked_lanes = 0usize;
        let mut rowsk_diffs = 0usize;
        for &k in &[1usize, 8, 16, 64] {
            let coords: Vec<u32> = (0..k).map(|i| (4 * 4 + i) as u32).collect();
            let gpu_scores = eng.score_rowsk_on_batch(num_rows as u32, &s1_base.roots, &coords)?;
            let cpu_scores = p5_score_block_cpu(&s1_base, &cpu_batch, &coords)?;
            if gpu_scores.len() != cpu_scores.len() {
                return Err(fail("rows-K score count mismatch"));
            }
            for (g, c) in gpu_scores.iter().zip(&cpu_scores) {
                rowsk_checked_lanes += 16;
                if g != c {
                    rowsk_diffs += 1;
                }
            }
        }
        if rowsk_diffs != 0 {
            return Err(fail(format!(
                "Gate 2 rows-K GPU-vs-CPU diffs={rowsk_diffs}"
            )));
        }

        // Check 3: BVG-K exact GPU-vs-CPU step parity covering |S_t|=0, |S_t|=1, |S_t|>=2
        // (joint-accept & joint-reject), resulting_port tie-break, and equal-trial random control.
        let mut gpu_s1 = s1_base.clone();
        let mut cpu_s1 = s1_base.clone();
        let mut saw_empty = false;
        let mut saw_single = false;
        let mut saw_multi = false;
        let mut saw_joint_accept = false;
        let mut saw_joint_reject = false;
        let mut bvgk_steps_checked = 0usize;

        // First test |S_t| = 0 on unseen token coordinates (token 3000 never appears in batch)
        let unseen_coords: Vec<u32> = (3000 * 4..3000 * 4 + 8).collect();
        let g_rec0 =
            bvgk_step_from_gpu_scores(&mut eng, num_rows as u32, &mut gpu_s1, &unseen_coords)?;
        let c_rec0 = p5_bvgk_step_cpu(&mut cpu_s1, &cpu_batch, &unseen_coords)?;
        if g_rec0 != c_rec0 || gpu_s1 != cpu_s1 {
            return Err(fail("Gate 2 BVG-K |S_t|=0 mismatch"));
        }
        saw_empty |= g_rec0.s_len == 0;
        bvgk_steps_checked += 1;

        // Test single-coordinate K=1 step on token 4 (blocks 16..20)
        for coord in 16..20u32 {
            let g_rec =
                bvgk_step_from_gpu_scores(&mut eng, num_rows as u32, &mut gpu_s1, &[coord])?;
            let c_rec = p5_bvgk_step_cpu(&mut cpu_s1, &cpu_batch, &[coord])?;
            if g_rec != c_rec || gpu_s1 != cpu_s1 {
                return Err(fail("Gate 2 BVG-K K=1 mismatch"));
            }
            if g_rec.s_len == 0 {
                saw_empty = true;
            } else if g_rec.s_len == 1 {
                saw_single = true;
            }
            bvgk_steps_checked += 1;
        }

        // Sweep active blocks (tokens 2..30) with K = 16 and K = 64 to exercise multi-member S_t!
        let mut rng_ctrl_g = 4392u64;
        let mut rng_ctrl_c = 4392u64;
        let mut gpu_rand = P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 4391, [0x11; 32])?;
        let mut cpu_rand = gpu_rand.clone();
        for blk_start in [8u32, 16, 24, 32, 40, 48, 56, 64] {
            let coords: Vec<u32> = (blk_start..blk_start + 32).collect();
            let g_rec = bvgk_step_from_gpu_scores(&mut eng, num_rows as u32, &mut gpu_s1, &coords)?;
            let c_rec = p5_bvgk_step_cpu(&mut cpu_s1, &cpu_batch, &coords)?;
            if g_rec != c_rec || gpu_s1 != cpu_s1 {
                return Err(fail("Gate 2 BVG-K K=32 mismatch"));
            }
            if g_rec.s_len == 0 {
                saw_empty = true;
            } else if g_rec.s_len == 1 {
                saw_single = true;
            } else {
                saw_multi = true;
                if g_rec.joint {
                    saw_joint_accept = true;
                } else {
                    saw_joint_reject = true;
                }
            }
            bvgk_steps_checked += 1;

            // Equal-trial random control GPU vs CPU parity
            let rcoords = uor_r4_discrete::p5_brier_core::p5_random_block_coords(
                gpu_rand.roots.len(),
                16,
                &mut rng_ctrl_g,
            )?;
            let gr = bvgk_step_from_gpu_scores(&mut eng, num_rows as u32, &mut gpu_rand, &rcoords)?;
            let cr = p5_step_control_cpu(
                P5ControlMode::ReseededRandomCoords { seed: 4391 },
                &mut cpu_rand,
                &cpu_batch,
                16,
                0,
                &mut rng_ctrl_c,
            )?;
            if gr != cr || gpu_rand != cpu_rand {
                return Err(fail(
                    "Gate 2 equal-trial random control GPU-vs-CPU mismatch",
                ));
            }
            bvgk_steps_checked += 1;
        }
        // Explicitly test a multi-member block on planted tokens {4, 7, 16, 24} from fresh seed
        // to guarantee |S_t| >= 2 is exercised even if earlier K=1 steps already settled tok 4.
        let mut fresh_g = P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 7777, [0x11; 32])?;
        let mut fresh_c = fresh_g.clone();
        let multi_coords: Vec<u32> = (0..64u32).collect();
        let gm = bvgk_step_from_gpu_scores(&mut eng, num_rows as u32, &mut fresh_g, &multi_coords)?;
        let cm = p5_bvgk_step_cpu(&mut fresh_c, &cpu_batch, &multi_coords)?;
        if gm != cm || fresh_g != fresh_c {
            return Err(fail("Gate 2 K=64 fresh multi-member BVG-K mismatch"));
        }
        if gm.s_len >= 2 {
            saw_multi = true;
            if gm.joint {
                saw_joint_accept = true;
            } else {
                saw_joint_reject = true;
            }
        }
        bvgk_steps_checked += 1;

        // Check 4: Planted target shift (change row 0's target to high-mass token 500 and confirm exact shift)
        let mut shifted_cmds = cmds.clone();
        shifted_cmds[0].target = if shifted_cmds[0].target == 500 {
            42
        } else {
            500
        };
        eng.materialize_batch(&shifted_cmds)?;
        let shifted_gpu = eng.eval_roots_on_batch(num_rows as u32, &s1_base.roots)?;
        let (sw, sl, st, sn, sp) = eng.readback_batch_brier(num_rows)?;
        let shifted_batch = reconstruct_batch_brier_from_gpu(num_rows, &sw, &sl, &st, &sn, &sp)?;
        let shifted_cpu = batch_brier_q32(&s1_base, &shifted_batch)?;
        let orig_cpu = batch_brier_q32(&s1_base, &cpu_batch)?;
        if shifted_gpu != shifted_cpu || shifted_gpu == orig_cpu {
            return Err(fail(format!(
                "Gate 2 planted target shift failed: shifted_gpu={shifted_gpu} shifted_cpu={shifted_cpu} orig={orig_cpu}"
            )));
        }

        // Check 4b: True >2,048 sliding-window & 64-token decay parity on Kernels 1-4!
        let (r12_off, r12_len) = recs[12];
        if r12_len <= P4_MAX_CONTEXT + 5 {
            return Err(fail("Record 12 must exceed P4_MAX_CONTEXT + 5"));
        }
        let mut slide_row_diffs = 0usize;
        let mut slide_rowsk_diffs = 0usize;
        let slide_tail_rows = r12_len - 1 - P4_MAX_CONTEXT;
        {
            // Check 64-token decay boundaries AND all strict >2,048 slid rows in one stateless call!
            let check_positions: Vec<usize> = [64usize, 65, 128, 129, P4_MAX_CONTEXT]
                .into_iter()
                .chain((P4_MAX_CONTEXT + 1)..r12_len)
                .collect();
            let tail_cmds: Vec<RowStepCmdHost> = check_positions
                .iter()
                .map(|&p| RowStepCmdHost {
                    rec_off: r12_off as u32,
                    pos: p as u32,
                    target: planted_stream[r12_off + p],
                    _pad0: 0,
                    _pad1: 0,
                })
                .collect();
            eng.materialize_batch(&tail_cmds)?;
            let n_check = tail_cmds.len();
            let (sw, sl, st, sn, sp) = eng.readback_batch_brier(n_check)?;
            let slide_batch =
                reconstruct_batch_brier_from_gpu(n_check, &sw, &sl, &st, &sn, &sp)?;
            let p4_base = make_p4_state_with_s1(&bundle, &s1_base)?;
            let codes_base = s1_base.derive_pick_codes_4()?;
            let mut cpu_sess = P4Session::new();
            for p in 1..r12_len {
                if p == 1 {
                    cpu_sess
                        .sync_prompt(&[planted_stream[r12_off]])
                        .map_err(|e| fail(format!("slide batch cpu sync: {e}")))?;
                } else {
                    cpu_sess
                        .observe_generated(planted_stream[r12_off + p - 1])
                        .map_err(|e| fail(format!("slide batch cpu obs: {e}")))?;
                }
                if let Some(ci) = check_positions.iter().position(|&cp| cp == p) {
                    let row = &slide_batch.rows[ci];
                    if row.window.as_slice() != cpu_sess.history() {
                        slide_row_diffs += 1;
                    }
                    cpu_sess
                        .step(&p4_base, true, false)
                        .map_err(|e| fail(format!("slide batch cpu step: {e}")))?;
                    let cpu_b = full_vocab_brier_q32(cpu_sess.last_scores_q48(), row.target)?;
                    let pick = uor_r4_discrete::kn_pick::code_pick(&codes_base, &row.window)
                        .map(|(pp, d, _)| (pp, d));
                    let gpu_b = row.brier_for_pick(pick);
                    if cpu_b != gpu_b {
                        slide_row_diffs += 1;
                    }
                }
            }
            let gpu_slide_sum = eng.eval_roots_on_batch(n_check as u32, &s1_base.roots)?;
            let cpu_slide_sum = batch_brier_q32(&s1_base, &slide_batch)?;
            if gpu_slide_sum != cpu_slide_sum {
                slide_row_diffs += 1;
            }
            let slide_coords: Vec<u32> = (5 * 4..5 * 4 + 16).collect();
            let g_sk =
                eng.score_rowsk_on_batch(n_check as u32, &s1_base.roots, &slide_coords)?;
            let c_sk = p5_score_block_cpu(&s1_base, &slide_batch, &slide_coords)?;
            for (g, c) in g_sk.iter().zip(&c_sk) {
                if g != c {
                    slide_rowsk_diffs += 1;
                }
            }
        }
        if slide_row_diffs != 0 || slide_rowsk_diffs != 0 {
            return Err(fail(format!(
                "Gate 2 >2048 sliding-window Kernels 1-4 diffs: row={slide_row_diffs} rowsk={slide_rowsk_diffs}"
            )));
        }

        // Check 5: Tile-balanced multi-arm record evaluator (`p5_eval_records_multi_arm_kernel`) across 5 arms
        // on ALL 6 synthetic records including full 2,056-token > 2,048 sliding window and novel mask!
        let eval_rec_indices = [0usize, 1, 2, 3, 4, 12];
        let mut e_recs: Vec<(usize, usize)> = Vec::new();
        let mut e_toks = Vec::new();
        let mut e_nov = Vec::new();
        for &ri in &eval_rec_indices {
            let (off, len) = recs[ri];
            e_recs.push((e_toks.len(), len));
            for p in 0..len {
                e_toks.push(planted_stream[off + p]);
                e_nov.push(u8::from((p % 3) == 0));
            }
        }
        let arms_s1 = [
            s1_base.clone(),
            gpu_s1.clone(),
            s1_alt.clone(),
            gpu_rand.clone(),
            fresh_g.clone(),
        ];
        let arm_refs: Vec<&[u8]> = arms_s1.iter().map(|s| s.roots.as_slice()).collect();
        let (rec_outs, spots) = eng.eval_stream_multi_arm(
            &e_toks,
            &e_recs,
            Some(&e_nov),
            &arm_refs,
            1,
            5,
            256,
        )?;

        // Exhaustively verify ALL 6 records (including Record 12 length 2,056 crossing > 2,048 slide!)
        // against live sequential CPU P4Session::step across all 5 arms.
        let mut eval_rec_diffs = 0usize;
        let mut eval_spot_diffs = 0usize;
        let p4_arms: Vec<P4State<'_>> = arms_s1
            .iter()
            .map(|s| make_p4_state_with_s1(&bundle, s))
            .collect::<Result<Vec<_>>>()?;

        for (ri, &(r_off, r_len)) in e_recs.iter().enumerate() {
            for (a, p4s) in p4_arms.iter().enumerate() {
                let mut exp_all_b = 0u128;
                let mut exp_all_q12 = 0u64;
                let mut exp_nov_b = 0u128;
                let mut exp_nov_q12 = 0u64;
                let mut exp_tgts = 0u64;
                let mut exp_nov_tgts = 0u64;
                let mut sess = P4Session::new();
                for pos in 1..r_len {
                    let prev = e_toks[r_off + pos - 1];
                    let target = e_toks[r_off + pos];
                    if pos == 1 {
                        sess.sync_prompt(&[prev])
                            .map_err(|e| fail(format!("cpu eval sync: {e}")))?;
                    } else {
                        sess.observe_generated(prev)
                            .map_err(|e| fail(format!("cpu eval obs: {e}")))?;
                    }
                    if target == 0 {
                        continue;
                    }
                    sess.step(p4s, true, false)
                        .map_err(|e| fail(format!("cpu eval step: {e}")))?;
                    let sc = sess.last_scores_q48();
                    let br = full_vocab_brier_q32(sc, target)?;
                    let bq = bits_q12(sc[usize::from(target)])?;
                    exp_all_b += u128::from(br);
                    exp_all_q12 += bq;
                    exp_tgts += 1;
                    if e_nov[r_off + pos] == 1 {
                        exp_nov_b += u128::from(br);
                        exp_nov_q12 += bq;
                        exp_nov_tgts += 1;
                    }
                }
                let got = &rec_outs[ri * arms_s1.len() + a];
                if got.all_brier_u128() != exp_all_b
                    || got.all_bits_q12 != exp_all_q12
                    || got.novel_brier_u128() != exp_nov_b
                    || got.novel_bits_q12 != exp_nov_q12
                    || got.targets != exp_tgts
                    || got.novel_targets != exp_nov_tgts
                {
                    eval_rec_diffs += 1;
                }
            }
        }

        // Capture explicit GPU spots strictly beyond 2,048 (`pos = 2,050` via stride 205 and
        // `pos = 2,055` via stride 2055 across all 5 arms => 10 strict >2,048 spots) and verify
        // every spot against CPU P4Session::step + determinism between multi- and single-rec launch!
        let (r5_off, r5_len) = e_recs[5];
        let (s_outs_a, slide_spots_a) = eng.eval_stream_multi_arm(
            &e_toks[r5_off..r5_off + r5_len],
            &[(0, r5_len)],
            Some(&e_nov[r5_off..r5_off + r5_len]),
            &arm_refs,
            205,
            1,
            128,
        )?;
        let (_s_outs_b, slide_spots_b) = eng.eval_stream_multi_arm(
            &e_toks[r5_off..r5_off + r5_len],
            &[(0, r5_len)],
            Some(&e_nov[r5_off..r5_off + r5_len]),
            &arm_refs,
            2055,
            1,
            64,
        )?;
        let mut slide_positions_checked = 0usize;
        for sp in &spots {
            let ri = sp.rec_idx as usize;
            let (r_off, _) = e_recs[ri];
            let pos = sp.pos as usize;
            let start = pos.saturating_sub(P4_MAX_CONTEXT);
            let win = &e_toks[r_off + start..r_off + pos];
            let target = e_toks[r_off + pos];
            let (cb, cq, cp) =
                cpu_eval_p4_window_brier_and_bits(&p4_arms[sp.arm as usize], win, target)?;
            let gpu_p = (sp.pick_has != 0).then_some((
                sp.pick_pos,
                sp.pick_dist,
                sp.pick_class as u8,
                sp.pick_succ as u16,
            ));
            if cb != sp.brier_q32 || cq != sp.bits_q12 || cp != gpu_p {
                eval_spot_diffs += 1;
            }
        }
        for sp in slide_spots_a.iter().chain(slide_spots_b.iter()) {
            let pos = sp.pos as usize;
            let start = pos.saturating_sub(P4_MAX_CONTEXT);
            let win = &e_toks[r5_off + start..r5_off + pos];
            let target = e_toks[r5_off + pos];
            let (cb, cq, cp) =
                cpu_eval_p4_window_brier_and_bits(&p4_arms[sp.arm as usize], win, target)?;
            let gpu_p = (sp.pick_has != 0).then_some((
                sp.pick_pos,
                sp.pick_dist,
                sp.pick_class as u8,
                sp.pick_succ as u16,
            ));
            if cb != sp.brier_q32 || cq != sp.bits_q12 || cp != gpu_p {
                eval_spot_diffs += 1;
            }
            if pos > P4_MAX_CONTEXT {
                slide_positions_checked += 1;
            }
        }
        let mut slide_tail_diffs = 0usize;
        for a in 0..arms_s1.len() {
            let g1 = &rec_outs[5 * arms_s1.len() + a];
            let g2 = &s_outs_a[a];
            if g1.all_brier_u128() != g2.all_brier_u128()
                || g1.all_bits_q12 != g2.all_bits_q12
                || g1.novel_brier_u128() != g2.novel_brier_u128()
                || g1.novel_bits_q12 != g2.novel_bits_q12
                || g1.targets != g2.targets
            {
                slide_tail_diffs += 1;
            }
        }

        if eval_rec_diffs != 0 || eval_spot_diffs != 0 || slide_tail_diffs != 0 {
            return Err(fail(format!(
                "Gate 2 eval_records_multi_arm diffs: rec={eval_rec_diffs} spot={eval_spot_diffs} slide={slide_tail_diffs}"
            )));
        }
        if !(saw_empty && saw_single && saw_multi && saw_joint_accept && saw_joint_reject) {
            return Err(fail(format!(
                "Gate 2 BVG-K branch coverage incomplete: empty={saw_empty} single={saw_single} multi={saw_multi} joint_accept={saw_joint_accept} joint_reject={saw_joint_reject}"
            )));
        }
        if slide_positions_checked < 10 {
            return Err(fail(format!(
                "Gate 2 checked only {slide_positions_checked} < 10 strict >2048 sliding positions"
            )));
        }

        let (move_root, _) = p5_orbit_tables();
        let report = json!({
            "schema": "uor-r4.p5-finite/1",
            "cubin_sha256": cubin_sha,
            "cuda_executed": true,
            "rows_checked": num_rows,
            "row_brier_diffs_vs_p4_session": row_brier_diffs,
            "rowsk_k_values": [1, 8, 16, 64],
            "rowsk_checked_lanes": rowsk_checked_lanes,
            "rowsk_diffs": rowsk_diffs,
            "bvgk_steps_checked": bvgk_steps_checked,
            "bvgk_cases": {
                "saw_empty": saw_empty,
                "saw_single": saw_single,
                "saw_multi": saw_multi,
                "saw_joint_accept": saw_joint_accept,
                "saw_joint_reject": saw_joint_reject,
            },
            "planted_target_shift": {
                "orig_brier_q32": orig_cpu,
                "shifted_gpu_brier_q32": shifted_gpu,
                "shifted_cpu_brier_q32": shifted_cpu,
                "diff": 0,
                "changed": shifted_gpu != orig_cpu,
            },
            "slide_batch_check": {
                "record_len": r12_len,
                "slid_rows_checked": slide_tail_rows,
                "row_diffs": slide_row_diffs,
                "rowsk_k": 16,
                "rowsk_diffs": slide_rowsk_diffs,
            },
            "multi_arm_eval": {
                "arms": arms_s1.len(),
                "records_exhaustively_checked": eval_rec_indices.len(),
                "eval_rec_diffs": eval_rec_diffs,
                "spots_checked": spots.len() + slide_spots_a.len() + slide_spots_b.len(),
                "slide_positions_checked": slide_positions_checked,
                "eval_spot_diffs": eval_spot_diffs,
                "slide_tail_diffs": slide_tail_diffs,
            },
            "orbit_move_table_checksum": hex::encode(Sha256::digest(
                move_root.iter().flatten().copied().collect::<Vec<u8>>()
            )),
            "driver_4096_check": driver_res,
            "pass": true,
        });
        fs::write(out_json, serde_json::to_vec_pretty(&report)?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    {
        let p4_state = make_p4_state_with_s1(&bundle, &s1_base)?;
        let (br, bq, _) = cpu_eval_p4_window_brier_and_bits(&p4_state, &[0, 5, 10, 42, 10, 5], 42)?;
        let report = json!({
            "schema": "uor-r4.p5-finite/1",
            "cuda_executed": false,
            "cpu_smoke_brier_q32": br,
            "cpu_smoke_bits_q12": bq,
            "driver_4096_check": driver_res,
            "pass": true,
        });
        fs::write(out_json, serde_json::to_vec_pretty(&report)?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }
}
