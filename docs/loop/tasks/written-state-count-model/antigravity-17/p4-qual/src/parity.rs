#[cfg(feature = "cuda")]
use crate::{
    bar_a::{cpu_eval_bar_a_step, PIN_HELDOUT_SHA256},
    cuda::{split_uorp4a01_sections, P4GpuEngine},
    load_uort_tokens, verify_file_sha256,
};
use crate::{cuda::ParityStepOutHost, fail, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::Command};
#[cfg(feature = "cuda")]
use uor_r4_count_runtime::p4_serving::CausalSessionCache;
use uor_r4_count_runtime::{
    p3_state::{s4_pair_key, BrierS6Weights, S4ConjunctionTable, S4PointerEntry, P3_Q32_ONE},
    p4_serving::{
        encode_compact_s2_skip_from_uors2s01, encode_compact_s3_from_uors3t02, golden_shell_pkey,
        skip_keys_for_prefix, starling_7_neighbours, CompactS2SkipView, CompactS3View, P4Session,
        P4State, P4StepTrace, H4_ROOTS, P4_EXPERTS, P4_MAX_CONTEXT, P4_TOTAL_BUCKETS, P4_VOCAB,
    },
    CompactCountLookup, FrozenPickMixture, PickClassRule,
};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, ClientArtifactFile, DiscreteCountModel,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FixtureStep {
    pub token: u16,
    pub bucket: usize,
    pub highest_order: usize,
    pub mass_q48: u64,
    pub top_ties: u32,
    pub mass_sha256: String,
    pub prefix_sha256: String,
    pub context_tokens: usize,
    pub pick_pos: Option<u32>,
    pub pick_distance: Option<u32>,
    pub pick_ties: u32,
    pub pick_successor: Option<u16>,
    pub pick_class: u8,
    pub s3_pkey: u32,
    pub s3_backoff_key: u16,
    pub s3_mode: String,
    pub s2_skip_keys: [u64; 3],
    pub s2_skip_hit_mask: u8,
    pub cache_hit_state: u8,
    pub s4_role_scores: [i64; 3],
    pub s4_role_choice: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixturePrompt {
    pub id: String,
    pub stratum: String,
    pub read_enabled: bool,
    pub max_tokens: u32,
    pub prompt_ids: Vec<u16>,
    pub generated_ids: Vec<u16>,
    pub steps: Vec<FixtureStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityFixtureFile {
    pub schema: String,
    pub cubin_sha256: String,
    pub p4_artifact_sha256: String,
    pub p4_state_sha256: String,
    pub prompt_count: usize,
    pub total_steps: usize,
    pub prompts: Vec<FixturePrompt>,
}

fn s3_hit_kind_str(code: u32) -> Result<String> {
    match code {
        0 => Ok("primary".to_owned()),
        1 => Ok("backoff".to_owned()),
        2 => Ok("starling".to_owned()),
        3 => Ok("uniform_miss".to_owned()),
        _ => Err(fail(format!("invalid GPU s3_hit_kind {code}"))),
    }
}

fn sha256_scores_u64(scores_1_to_4095: &[u64]) -> String {
    let mut h = Sha256::new();
    for &w in scores_1_to_4095 {
        h.update(w.to_le_bytes());
    }
    hex::encode(h.finalize())
}

fn sha256_prefix_u16(prefix: &[u16]) -> String {
    let mut h = Sha256::new();
    for &w in prefix {
        h.update(w.to_le_bytes());
    }
    hex::encode(h.finalize())
}

fn fixture_step_from_trace(tr: &P4StepTrace) -> FixtureStep {
    FixtureStep {
        token: tr.token_id,
        bucket: tr.bucket,
        highest_order: tr.highest_order,
        mass_q48: tr.mass_q48,
        top_ties: tr.top_ties,
        mass_sha256: tr.mass_sha256.clone(),
        prefix_sha256: tr.prefix_sha256.clone(),
        context_tokens: tr.context_tokens,
        pick_pos: tr.pick_pos,
        pick_distance: tr.pick_distance,
        pick_ties: tr.pick_ties,
        pick_successor: tr.pick_successor,
        pick_class: tr.pick_class,
        s3_pkey: tr.s3_pkey,
        s3_backoff_key: tr.s3_backoff_key,
        s3_mode: tr.s3_mode.clone(),
        s2_skip_keys: tr.s2_skip_keys,
        s2_skip_hit_mask: tr.s2_skip_hit_mask,
        cache_hit_state: tr.cache_hit_state,
        s4_role_scores: tr.s4_role_scores,
        s4_role_choice: tr.s4_role_choice,
    }
}

fn fixture_step_from_gpu(
    out: &ParityStepOutHost,
    scores_1_to_4095: &[u64],
    active_history: &[u16],
) -> Result<FixtureStep> {
    if out.status != 0 {
        return Err(fail(format!("GPU step failed with status {}", out.status)));
    }
    let pick_pos = (out.pick_has != 0).then_some(out.pick_pos);
    let pick_distance = (out.pick_has != 0).then_some(out.pick_distance);
    let pick_successor = (out.pick_has != 0).then_some(out.pick_successor as u16);
    let s4_role_choice = (out.s4_role_choice != 0).then_some(out.s4_role_choice as u8);
    Ok(FixtureStep {
        token: out.token_id as u16,
        bucket: out.bucket as usize,
        highest_order: out.highest_order as usize,
        mass_q48: out.mass_q48,
        top_ties: out.top_ties,
        mass_sha256: sha256_scores_u64(scores_1_to_4095),
        prefix_sha256: sha256_prefix_u16(active_history),
        context_tokens: out.context_tokens as usize,
        pick_pos,
        pick_distance,
        pick_ties: out.pick_ties,
        pick_successor,
        pick_class: out.pick_class as u8,
        s3_pkey: out.s3_pkey,
        s3_backoff_key: out.s3_backoff_key as u16,
        s3_mode: s3_hit_kind_str(out.s3_hit_kind)?,
        s2_skip_keys: out.s2_skip_keys,
        s2_skip_hit_mask: out.s2_skip_hit_mask as u8,
        cache_hit_state: out.cache_hit_state as u8,
        s4_role_scores: out.s4_role_scores,
        s4_role_choice,
    })
}

// ---------- Synthetic Gate 2 Finite Fixture Builders ----------

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

/// Build a valid synthetic `UORKNC03` with orders 0..=2 exercising inline, esc16, and esc32 counts
/// as well as multiple contexts and both non-shared (`k=0,1`) and shared (`k=2`) orders.
fn build_synth_uorknc03(c_sha: [u8; 32]) -> Vec<u8> {
    let highest = 2u32;
    let mut out = Vec::new();
    out.extend_from_slice(b"UORKNC03");
    out.extend_from_slice(&highest.to_le_bytes());
    out.extend_from_slice(&c_sha);

    // Order 0 (non-shared): 1 context, 70 followers (crosses 64-entry superblock boundary!)
    // Includes inline, esc16 (raw=200, kn=40), and esc32 (raw=80000, kn=70000).
    let mut o0_fols: Vec<(u16, u32, u32)> = Vec::new();
    for i in 1..=67u16 {
        o0_fols.push((i, 1 + u32::from(i % 4), 1 + u32::from(i % 2)));
    }
    o0_fols.push((100, 200, 40)); // esc16
    o0_fols.push((500, 80_000, 70_000)); // esc32
    o0_fols.push((4095, 3, 2)); // inline
    let push_order = |buf: &mut Vec<u8>,
                      k: u32,
                      shared: bool,
                      parent_ctxs: u32,
                      children: &[(u32, u16)], // (parent_idx, child_tok)
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
        // Build follower words, esc_sb, esc16, esc32
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

        // Header (88 bytes)
        buf.extend_from_slice(&followers.to_le_bytes());
        buf.extend_from_slice(&contexts.to_le_bytes());
        for w in rspec.into_iter().chain(aspec) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // parent_deg_esc_len
        buf.extend_from_slice(&0u32.to_le_bytes()); // fol_deg_esc_len
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
            // pack 12-bit child tokens
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
        // fol_sb, fol_deg (bias=1, each ctx has 1..=254 followers)
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
    // Order 1: parent 0 -> child tokens 5, 10, 42
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
    // Order 2 (shared): parent 1 (prev1=10) -> child tok 5 (context [5, 10]); parent 2 (prev1=42) -> child tok 10 (context [10, 42])
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

pub struct SyntheticP4Bundle {
    pub counts_v3: Vec<u8>,
    pub s3_compact: Vec<u8>,
    pub s2_compact: Vec<u8>,
    pub s4_bytes: Vec<u8>,
    pub s6_bytes: Vec<u8>,
    pub a1s_bytes: Vec<u8>,
    pub fit_digest: [u8; 32],
    pub c_sha: [u8; 32],
    pub r_sha: [u8; 32],
    pub codes: Vec<[u16; 4]>,
    pub prompts: Vec<(String, Vec<u16>, u32, bool)>,
}

pub fn build_synthetic_p4_bundle() -> Result<SyntheticP4Bundle> {
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
    let k2 = s4_pair_key(10, 42).map_err(|e| fail(format!("s4 key: {e}")))?;
    let mut s4_rows_map: BTreeMap<u32, Vec<S4PointerEntry>> = BTreeMap::new();
    s4_rows_map.insert(
        k1,
        vec![
            S4PointerEntry {
                role_class: 1,
                span_offset: 0,
                count: 8,
                err_i8: 12,
            },
            S4PointerEntry {
                role_class: 2,
                span_offset: 0,
                count: 4,
                err_i8: -10,
            },
        ],
    );
    s4_rows_map.insert(
        k2,
        vec![S4PointerEntry {
            role_class: 3,
            span_offset: 0,
            count: 10,
            err_i8: 5,
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

    // Non-trivial S6 weights putting positive weight on ALL 22 experts across buckets
    let mut s6_weights = vec![0u64; P4_TOTAL_BUCKETS * P4_EXPERTS];
    for b in 0..P4_TOTAL_BUCKETS {
        let base = b * P4_EXPERTS;
        let share = P3_Q32_ONE / (P4_EXPERTS as u64);
        let mut sum = 0u64;
        for e in 0..P4_EXPERTS {
            let v = share + ((b + e) as u64 % 17);
            s6_weights[base + e] = v;
            sum += v;
        }
        // Adjust expert 0 so sum == P3_Q32_ONE exactly
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

    // Valid A1s 5-class FrozenPickMixture for highest = 2
    let mut a1s_w = vec![0u64; (3 * 16) * 5 * 6]; // highest=2 => 48 base buckets * 5 classes * 6 experts
    for b in 0..(48 * 5) {
        a1s_w[b * 6] = P3_Q32_ONE;
    }
    let a1s = FrozenPickMixture::with_rule(2, fit_digest, a1s_w, PickClassRule::Five)
        .map_err(|e| fail(format!("synth a1s: {e}")))?;
    let a1s_bytes = a1s.to_bytes();

    // Deterministic 4x16-bit codes spanning Hamming distances 0, 1, 2, >=3
    let mut codes = vec![[0u16; 4]; P4_VOCAB];
    for (tok, c) in codes.iter_mut().enumerate() {
        let v = tok as u16;
        c[0] = v.wrapping_mul(0x9E37);
        c[1] = v.rotate_left(3) ^ 0xA5A5;
        c[2] = v.wrapping_mul(0x3C6E);
        c[3] = v ^ 0x55AA;
    }
    // Make codes[6] differ from codes[5] by exactly 1 bit, codes[8] by 2 bits
    codes[6] = codes[5];
    codes[6][0] ^= 1;
    codes[8] = codes[5];
    codes[8][0] ^= 3;

    // 10 diverse synthetic prompts covering every edge case including 2048-token slide & 64-tok decay
    let mut long_2048 = vec![0u16; P4_MAX_CONTEXT];
    for (i, slot) in long_2048.iter_mut().enumerate().skip(1) {
        *slot = match i % 11 {
            0 => 5,
            1 => 10,
            2 => 42,
            3 => 5,
            4 => 10,
            5 => 42,
            6 => 7,
            7 => 12,
            8 => 100,
            9 => 500,
            _ => ((i % 60) + 2) as u16,
        };
    }
    let mut decay_130 = vec![0u16; 130];
    for (i, slot) in decay_130.iter_mut().enumerate().skip(1) {
        *slot = ((i * 7) % 50 + 2) as u16;
    }

    let prompts = vec![
        (
            "synth_s3_primary_s2_hit".to_owned(),
            p_prim.to_vec(),
            4,
            true,
        ),
        (
            "synth_s3_primary_noread".to_owned(),
            p_prim.to_vec(),
            2,
            false,
        ),
        ("synth_s3_backoff".to_owned(), p_back.to_vec(), 3, true),
        ("synth_s3_starling".to_owned(), p_star.to_vec(), 3, true),
        (
            "synth_s3_uniform_miss".to_owned(),
            vec![0, 999, 1001, 1003, 1005, 1007],
            2,
            true,
        ),
        (
            "synth_pick_dist0_1_2".to_owned(),
            vec![0, 5, 10, 42, 6, 10, 8, 10, 5, 10],
            4,
            true,
        ),
        (
            "synth_bos_mid_reset".to_owned(),
            vec![0, 5, 10, 0, 10, 42, 5, 10],
            3,
            true,
        ),
        ("synth_bosless".to_owned(), vec![5, 10, 42, 5, 10], 3, true),
        ("synth_cache_decay_130".to_owned(), decay_130, 3, true),
        ("synth_slide_2048".to_owned(), long_2048, 4, true),
    ];

    Ok(SyntheticP4Bundle {
        counts_v3,
        s3_compact,
        s2_compact,
        s4_bytes,
        s6_bytes,
        a1s_bytes,
        fit_digest,
        c_sha,
        r_sha,
        codes,
        prompts,
    })
}

pub fn make_synth_p4_state(b: &SyntheticP4Bundle) -> Result<P4State<'_>> {
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
        b.codes.clone(),
    )
    .map_err(|e| fail(format!("synth P4State: {e}")))
}

#[cfg(feature = "cuda")]
pub fn run_finite(out_path: &Path) -> Result<()> {
    let bundle = build_synthetic_p4_bundle()?;
    let state = make_synth_p4_state(&bundle)?;
    let engine = P4GpuEngine::new(
        &bundle.counts_v3,
        &bundle.s3_compact,
        &bundle.s2_compact,
        &bundle.s4_bytes,
        &state.s6.weights_q32,
        &state.codes,
    )?;

    let prompts: Vec<Vec<u16>> = bundle.prompts.iter().map(|p| p.1.clone()).collect();
    let steps: Vec<u32> = bundle.prompts.iter().map(|p| p.2).collect();
    let reads: Vec<bool> = bundle.prompts.iter().map(|p| p.3).collect();
    let (gpu_steps, gpu_scores) = engine.run_parity_batch(&prompts, &steps, &reads)?;

    let mut total_steps = 0usize;
    let mut total_score_words = 0usize;
    let mut step_diffs = 0usize;
    let mut score_diffs = 0usize;
    let mut s3_modes_seen = BTreeMap::<String, usize>::new();
    let mut cursor = 0usize;
    for (idx, (name, prompt, n_steps, read_en)) in bundle.prompts.iter().enumerate() {
        let mut sess = P4Session::new();
        sess.sync_prompt(prompt)
            .map_err(|e| fail(format!("finite cpu sync {name}: {e}")))?;
        for s in 0..(*n_steps as usize) {
            let gpu_out = &gpu_steps[cursor + s];
            let gpu_sc =
                &gpu_scores[(cursor + s) * (P4_VOCAB - 1)..(cursor + s + 1) * (P4_VOCAB - 1)];
            let (tok, trace_opt) = sess
                .step(&state, *read_en, true)
                .map_err(|e| fail(format!("finite cpu step {name}#{s}: {e}")))?;
            let tr = trace_opt.ok_or_else(|| fail("missing trace"))?;
            let cpu_step = fixture_step_from_trace(&tr);
            let gpu_step = fixture_step_from_gpu(gpu_out, gpu_sc, sess.history())?;
            if cpu_step != gpu_step {
                step_diffs += 1;
                eprintln!("FINITE STEP DIFF on {name}#{s} (prompt {idx}):\nCPU: {cpu_step:?}\nGPU: {gpu_step:?}");
            }
            let cpu_sc = &sess.last_scores_q48()[1..P4_VOCAB];
            for (v, (&cs, &gs)) in cpu_sc.iter().zip(gpu_sc).enumerate() {
                if cs != gs {
                    score_diffs += 1;
                    if score_diffs <= 5 {
                        eprintln!(
                            "FINITE SCORE DIFF on {name}#{s} tok {}: cpu={cs} gpu={gs}",
                            v + 1
                        );
                    }
                }
            }
            *s3_modes_seen.entry(cpu_step.s3_mode.clone()).or_insert(0) += 1;
            total_steps += 1;
            total_score_words += P4_VOCAB - 1;
            if tok == 1 {
                for rem in (s + 1)..(*n_steps as usize) {
                    if gpu_steps[cursor + rem].status != 99 {
                        step_diffs += 1;
                    }
                }
                break;
            }
            if s + 1 < (*n_steps as usize) {
                sess.observe_generated(tok)
                    .map_err(|e| fail(format!("finite observe {name}: {e}")))?;
            }
        }
        cursor += *n_steps as usize;
    }

    // Also test Bar A PRLL CUDA kernel vs CPU oracle on synthetic stream with BOS=0 targets & EOS=1 turns!
    let mut synth_stream = Vec::with_capacity(600);
    synth_stream.push(0);
    for i in 0..599usize {
        if i == 50 || i == 150 || i == 250 {
            synth_stream.push(1);
            synth_stream.push(0);
        } else if i == 100 {
            synth_stream.push(1); // mid-dialogue EOS without BOS
        } else {
            synth_stream.push(match i % 9 {
                0 => 5,
                1 => 10,
                2 => 42,
                3 => 7,
                4 => 12,
                5 => 100,
                6 => 500,
                7 => 6,
                _ => ((i % 40) + 2) as u16,
            });
        }
    }
    let prll_w = 4u32;
    let prll_ctx = 64u32;
    let prll_stride = 60u32;
    let gpu_prll = engine.run_bar_a_windows(&synth_stream, prll_w, prll_stride, prll_ctx, None)?;
    let mut prll_diffs = 0usize;
    let mut prll_bos_checked = 0usize;
    for w in 0..(prll_w as usize) {
        let start = w * (prll_stride as usize);
        let mut active = Vec::new();
        let mut cache = CausalSessionCache::new();
        for t in 0..(prll_ctx as usize) {
            let tok = synth_stream[start + t];
            let target = synth_stream[start + t + 1];
            if tok == 0 {
                active.clear();
                active.push(0);
                cache.reset_record();
            } else {
                active.push(tok);
                cache.observe_token(tok);
            }
            let (cr, cp, cb) = cpu_eval_bar_a_step(&state, None, &active, &cache, target)?;
            let idx = w * (prll_ctx as usize) + t;
            if cr != gpu_prll.raw_p4_mass_q48[idx]
                || cp != gpu_prll.prll_mass_q48[idx]
                || cb != gpu_prll.prll_bits_q12[idx]
            {
                prll_diffs += 1;
                if prll_diffs <= 5 {
                    eprintln!(
                        "FINITE PRLL DIFF w={w} t={t} target={target}: cpu=({cr},{cp},{cb}) gpu=({},{},{})",
                        gpu_prll.raw_p4_mass_q48[idx],
                        gpu_prll.prll_mass_q48[idx],
                        gpu_prll.prll_bits_q12[idx]
                    );
                }
            }
            if target == 0 {
                prll_bos_checked += 1;
            }
        }
    }

    let pass = step_diffs == 0 && score_diffs == 0 && prll_diffs == 0;
    let report = json!({
        "schema": "uor-r4.p4-finite-parity/1",
        "cubin_sha256": engine.cubin_sha256,
        "prompts_tested": bundle.prompts.len(),
        "total_steps": total_steps,
        "total_score_words": total_score_words,
        "step_diffs": step_diffs,
        "score_diffs": score_diffs,
        "s3_modes_seen": s3_modes_seen,
        "prll_positions_tested": (prll_w * prll_ctx) as usize,
        "prll_bos_targets_tested": prll_bos_checked,
        "prll_diffs": prll_diffs,
        "pass": pass,
    });
    fs::write(out_path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !pass {
        return Err(fail("Gate 2 synthetic finite parity failed"));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run_finite(_out_path: &Path) -> Result<()> {
    Err(fail("finite requires --features cuda"))
}

// ---------- Gate 3b: 96-Prompt Real Full-Dose CUDA-vs-CPU Parity Fixture ----------

/// Build 96 diverse valid open-record prompts (`!rec.contains(&1)`, `1 <= len <= 2048`)
/// spanning short, medium, long, exact `2,048`-context + 4-step sliding rollouts, BOS-less,
/// read_enabled true/false, repeated phrases (Pick dist 0 + Cache hit_state 2), and sparse keys.
fn build_96_prompts(heldout_tokens: &[u16]) -> Vec<(String, String, Vec<u16>, u32, bool)> {
    // Extract clean open records (between BOS=0 and first EOS=1) from heldout_tokens
    let mut clean_records: Vec<Vec<u16>> = Vec::new();
    let mut i = 0usize;
    while i < heldout_tokens.len() && clean_records.len() < 400 {
        if heldout_tokens[i] == 0 {
            let start = i;
            i += 1;
            while i < heldout_tokens.len() && heldout_tokens[i] != 1 && heldout_tokens[i] != 0 {
                i += 1;
            }
            let len = i - start;
            if len >= 20 {
                clean_records.push(heldout_tokens[start..i].to_vec());
            }
        } else {
            i += 1;
        }
    }

    let mut out: Vec<(String, String, Vec<u16>, u32, bool)> = Vec::with_capacity(96);
    // 1. 24 short prompts (len 2..=16, 2 steps, read_enabled=true)
    for k in 0..24usize {
        let rec = &clean_records[k % clean_records.len()];
        let take = (2 + (k * 5) % 15).min(rec.len());
        out.push((
            format!("short_{k:02}"),
            "short_2_16".to_owned(),
            rec[..take].to_vec(),
            2,
            true,
        ));
    }
    // 2. 24 medium prompts (len 32..=256, 2 steps, read_enabled=true)
    for k in 0..24usize {
        let target_len = 32 + (k * 9) % 225;
        let mut p = Vec::with_capacity(target_len);
        p.push(0);
        let mut src_idx = 24 + k * 3;
        while p.len() < target_len {
            let r = &clean_records[src_idx % clean_records.len()];
            for &tk in &r[1..] {
                if p.len() >= target_len {
                    break;
                }
                p.push(tk);
            }
            src_idx += 1;
        }
        out.push((
            format!("med_{k:02}"),
            "medium_32_256".to_owned(),
            p,
            2,
            true,
        ));
    }
    // 3. 16 long prompts (len 512..=1536, 2 steps, read_enabled=true)
    for k in 0..16usize {
        let target_len = 512 + k * 64;
        let mut p = Vec::with_capacity(target_len);
        p.push(0);
        let mut src_idx = 100 + k * 7;
        while p.len() < target_len {
            let r = &clean_records[src_idx % clean_records.len()];
            for &tk in &r[1..] {
                if p.len() >= target_len {
                    break;
                }
                p.push(tk);
            }
            src_idx += 1;
        }
        out.push((
            format!("long_{k:02}"),
            "long_512_1536".to_owned(),
            p,
            2,
            true,
        ));
    }
    // 4. 12 exact 2,048-token prompts with 4-step sliding rollouts (read_enabled=true)
    for k in 0..12usize {
        let mut p = Vec::with_capacity(P4_MAX_CONTEXT);
        p.push(0);
        let mut src_idx = 200 + k * 11;
        while p.len() < P4_MAX_CONTEXT {
            let r = &clean_records[src_idx % clean_records.len()];
            for &tk in &r[1..] {
                if p.len() >= P4_MAX_CONTEXT {
                    break;
                }
                p.push(tk);
            }
            src_idx += 1;
        }
        out.push((
            format!("ctx2048_{k:02}"),
            "ctx2048_slide4".to_owned(),
            p,
            4,
            true,
        ));
    }
    // 5. 12 read_enabled = false prompts across short/medium/2048 lengths
    for k in 0..12usize {
        let target_len = match k % 3 {
            0 => 12 + k,
            1 => 96 + k * 7,
            _ => P4_MAX_CONTEXT,
        };
        let mut p = Vec::with_capacity(target_len);
        p.push(0);
        let mut src_idx = 300 + k * 5;
        while p.len() < target_len {
            let r = &clean_records[src_idx % clean_records.len()];
            for &tk in &r[1..] {
                if p.len() >= target_len {
                    break;
                }
                p.push(tk);
            }
            src_idx += 1;
        }
        out.push((
            format!("noread_{k:02}"),
            "read_disabled".to_owned(),
            p,
            2,
            false,
        ));
    }
    // 6. 8 boundary & stress prompts (single-token BOS, BOS-less, exact repeat for Pick dist=0 & LCP, rare token misses)
    out.push((
        "edge_bos_only".to_owned(),
        "edge".to_owned(),
        vec![0],
        3,
        true,
    ));
    let r0 = &clean_records[0];
    out.push((
        "edge_bosless".to_owned(),
        "edge".to_owned(),
        r0[1..17.min(r0.len())].to_vec(),
        3,
        true,
    ));
    let mut rep_p = vec![0u16];
    let seg = &r0[1..13.min(r0.len())];
    rep_p.extend_from_slice(seg);
    rep_p.extend_from_slice(seg);
    rep_p.extend_from_slice(&seg[..6.min(seg.len())]);
    out.push((
        "edge_repeat_lcp_pick0".to_owned(),
        "edge".to_owned(),
        rep_p,
        4,
        true,
    ));
    for k in 0..5usize {
        let base = (3800 + k * 53) as u16;
        let rare = vec![0, base, base + 7, base + 13, base + 19, base + 23];
        out.push((
            format!("edge_rare_{k:02}"),
            "edge_rare".to_owned(),
            rare,
            2,
            true,
        ));
    }
    assert_eq!(out.len(), 96);
    out
}

#[cfg(feature = "cuda")]
pub fn run_parity_fixture(
    p4_path: &Path,
    heldout_path: &Path,
    out_path: &Path,
    summary_path: &Path,
) -> Result<()> {
    let heldout_bytes = verify_file_sha256(heldout_path, PIN_HELDOUT_SHA256, "heldout.u16")?;
    let heldout_tokens = load_uort_tokens(&heldout_bytes)?;

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
    let art_sha = bound
        .identity
        .as_ref()
        .ok_or_else(|| fail("missing identity"))?
        .artifact_sha256
        .clone();
    let raw = split_uorp4a01_sections(&state_bytes)?;

    let engine = P4GpuEngine::new(
        raw.counts_v3,
        raw.s3_compact,
        raw.s2_skip_compact,
        raw.s4_table,
        &p4.inner.s6.weights_q32,
        &p4.inner.codes,
    )?;

    let specs = build_96_prompts(&heldout_tokens);
    let prompts: Vec<Vec<u16>> = specs.iter().map(|s| s.2.clone()).collect();
    let steps: Vec<u32> = specs.iter().map(|s| s.3).collect();
    let reads: Vec<bool> = specs.iter().map(|s| s.4).collect();
    let (gpu_steps, gpu_scores) = engine.run_parity_batch(&prompts, &steps, &reads)?;

    let mut fixture_prompts = Vec::with_capacity(specs.len());
    let mut total_steps = 0usize;
    let mut cpu_gpu_step_diffs = 0usize;
    let mut cpu_gpu_score_diffs = 0usize;
    let mut s3_counts = BTreeMap::<String, usize>::new();
    let mut cache_counts = BTreeMap::<u8, usize>::new();
    let mut pick_counts = BTreeMap::<u8, usize>::new();
    let mut s2_mask_counts = BTreeMap::<u8, usize>::new();
    let mut s4_active_steps = 0usize;
    let mut s4_abstain_steps = 0usize;

    let mut cursor = 0usize;
    for (id, stratum, prompt_ids, max_tokens, read_en) in specs {
        let mut sess = P4Session::new();
        sess.sync_prompt(&prompt_ids)
            .map_err(|e| fail(format!("cpu sync {id}: {e}")))?;
        let mut gen_ids = Vec::new();
        let mut fsteps = Vec::new();
        for s in 0..(max_tokens as usize) {
            let g_out = &gpu_steps[cursor + s];
            let g_sc =
                &gpu_scores[(cursor + s) * (P4_VOCAB - 1)..(cursor + s + 1) * (P4_VOCAB - 1)];
            let (tok, trace_opt) = sess
                .step(&p4.inner, read_en, true)
                .map_err(|e| fail(format!("cpu step {id}#{s}: {e}")))?;
            let tr = trace_opt.ok_or_else(|| fail("missing trace"))?;
            let c_step = fixture_step_from_trace(&tr);
            let g_step = fixture_step_from_gpu(g_out, g_sc, sess.history())?;
            if c_step != g_step {
                cpu_gpu_step_diffs += 1;
                eprintln!("PARITY STEP DIFF on {id}#{s}:\nCPU: {c_step:?}\nGPU: {g_step:?}");
            }
            let c_sc = &sess.last_scores_q48()[1..P4_VOCAB];
            for (&cs, &gs) in c_sc.iter().zip(g_sc) {
                if cs != gs {
                    cpu_gpu_score_diffs += 1;
                }
            }
            *s3_counts.entry(g_step.s3_mode.clone()).or_insert(0) += 1;
            *cache_counts.entry(g_step.cache_hit_state).or_insert(0) += 1;
            *pick_counts.entry(g_step.pick_class).or_insert(0) += 1;
            *s2_mask_counts.entry(g_step.s2_skip_hit_mask).or_insert(0) += 1;
            if g_step.s4_role_choice.is_some() {
                s4_active_steps += 1;
            } else {
                s4_abstain_steps += 1;
            }
            gen_ids.push(tok);
            fsteps.push(g_step);
            total_steps += 1;
            if tok == 1 {
                break;
            }
            if s + 1 < (max_tokens as usize) {
                sess.observe_generated(tok)
                    .map_err(|e| fail(format!("cpu observe {id}: {e}")))?;
            }
        }
        cursor += max_tokens as usize;
        fixture_prompts.push(FixturePrompt {
            id,
            stratum,
            read_enabled: read_en,
            max_tokens,
            prompt_ids,
            generated_ids: gen_ids,
            steps: fsteps,
        });
    }

    if cpu_gpu_step_diffs != 0 || cpu_gpu_score_diffs != 0 {
        return Err(fail(format!(
            "Wick CUDA-vs-CPU parity fixture failed: step_diffs={cpu_gpu_step_diffs} score_diffs={cpu_gpu_score_diffs}"
        )));
    }

    let fixture = ParityFixtureFile {
        schema: "uor-r4.p4-cuda-parity-fixture/1".to_owned(),
        cubin_sha256: engine.cubin_sha256.clone(),
        p4_artifact_sha256: art_sha.clone(),
        p4_state_sha256: p4.artifact_sha256.clone(),
        prompt_count: fixture_prompts.len(),
        total_steps,
        prompts: fixture_prompts,
    };
    let fixture_bytes = serde_json::to_vec(&fixture)?;
    let fixture_sha = hex::encode(Sha256::digest(&fixture_bytes));
    fs::write(out_path, &fixture_bytes)?;

    let summary = json!({
        "schema": "uor-r4.p4-parity-fixture-summary/1",
        "cubin_sha256": engine.cubin_sha256,
        "p4_artifact_sha256": art_sha,
        "p4_state_sha256": p4.artifact_sha256,
        "fixture_sha256": fixture_sha,
        "prompt_count": fixture.prompt_count,
        "total_steps": total_steps,
        "total_raw_score_words_checked": total_steps * (P4_VOCAB - 1),
        "wick_cuda_vs_cpu_step_diffs": cpu_gpu_step_diffs,
        "wick_cuda_vs_cpu_score_word_diffs": cpu_gpu_score_diffs,
        "coverage": {
            "s3_modes": s3_counts,
            "cache_hit_states": cache_counts,
            "pick_classes": pick_counts,
            "s2_skip_hit_masks": s2_mask_counts,
            "s4_active_steps": s4_active_steps,
            "s4_abstain_steps": s4_abstain_steps,
        },
        "pass": true,
    });
    fs::write(summary_path, serde_json::to_vec_pretty(&summary)?)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run_parity_fixture(
    _p4_path: &Path,
    _heldout_path: &Path,
    _out_path: &Path,
    _summary_path: &Path,
) -> Result<()> {
    Err(fail("parity-fixture requires --features cuda"))
}

// ---------- Gate 5: Apple M1 uor-chat Verification Against Wick CUDA Fixture ----------

/// Verify all 96 prompts in `cuda_parity_96.json` on local Apple M1 (`uor-chat` CLI binary +
/// in-process `load_client_artifact` + `P4Session::step` exact check).
/// Note: spawning `uor-chat` 96 times on a 3.99 GB artifact would re-read 380 GB from disk, so
/// we verify ALL 96 prompts in-process via the exact `uor-r4-integer` `load_client_artifact` +
/// `P4Session::step` codepath AND ALSO spawn the real `uor-chat` binary on a stratified subset
/// (default `8` prompts spanning short, medium, 2048-context slide4, read_disabled, and edge cases)
/// verifying bit-identical JSON output end-to-end through the CLI executable!
pub fn run_verify_m1(
    p4_path: &Path,
    fixture_path: &Path,
    uor_chat_bin: &Path,
    cli_count: usize,
    out_path: &Path,
) -> Result<()> {
    let fixture_bytes = fs::read(fixture_path)?;
    let fixture_sha = hex::encode(Sha256::digest(&fixture_bytes));
    let fixture: ParityFixtureFile = serde_json::from_slice(&fixture_bytes)?;

    let loaded = load_client_artifact(p4_path, false)
        .map_err(|e| fail(format!("M1 load_client_artifact failed: {e}")))?;
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
            .map_err(|e| fail(format!("M1 bind_embedded failed: {e}")))?;
    let p4 = bound
        .p4
        .as_ref()
        .ok_or_else(|| fail("M1 load missing P4"))?;
    let art_sha = bound
        .identity
        .as_ref()
        .ok_or_else(|| fail("M1 missing identity"))?
        .artifact_sha256
        .clone();
    if art_sha != fixture.p4_artifact_sha256 || p4.artifact_sha256 != fixture.p4_state_sha256 {
        return Err(fail("M1 p4.d11 SHA-256 does not match Wick CUDA fixture"));
    }

    let mut inproc_steps_checked = 0usize;
    let mut inproc_diffs = 0usize;
    let mut sess = P4Session::new();
    for fp in &fixture.prompts {
        sess.sync_prompt(&fp.prompt_ids)
            .map_err(|e| fail(format!("M1 sync {}: {e}", fp.id)))?;
        let mut gen = Vec::new();
        for (sidx, exp_step) in fp.steps.iter().enumerate() {
            let (tok, trace_opt) = sess
                .step(&p4.inner, fp.read_enabled, true)
                .map_err(|e| fail(format!("M1 step {}#{sidx}: {e}", fp.id)))?;
            let tr = trace_opt.ok_or_else(|| fail("M1 missing trace"))?;
            let got_step = fixture_step_from_trace(&tr);
            if &got_step != exp_step {
                inproc_diffs += 1;
                eprintln!(
                    "M1 IN-PROCESS DIFF on {}#{sidx}:\nCUDA: {exp_step:?}\nM1:   {got_step:?}",
                    fp.id
                );
            }
            gen.push(tok);
            inproc_steps_checked += 1;
            if tok == 1 {
                break;
            }
            if sidx + 1 < fp.steps.len() {
                sess.observe_generated(tok)
                    .map_err(|e| fail(format!("M1 observe {}: {e}", fp.id)))?;
            }
        }
        if gen != fp.generated_ids {
            inproc_diffs += 1;
        }
    }
    // Drop the 4 GB in-process model before spawning `uor-chat` subprocesses so M1 16 GB RAM stays clean
    drop(bound);
    drop(state_bytes);

    // Stratified real `uor-chat` CLI binary parity checks
    let mut cli_indices: Vec<usize> = Vec::new();
    let want = cli_count.min(fixture.prompts.len());
    if want > 0 {
        // Pick representative prompts: short, med, long, ctx2048_00, noread_00, edge_bos_only, edge_bosless, edge_repeat_lcp_pick0
        let preferred = [0usize, 24, 48, 64, 76, 88, 89, 90];
        for &idx in &preferred {
            if idx < fixture.prompts.len() && cli_indices.len() < want {
                cli_indices.push(idx);
            }
        }
        let mut extra = 1usize;
        while cli_indices.len() < want && extra < fixture.prompts.len() {
            if !cli_indices.contains(&extra) {
                cli_indices.push(extra);
            }
            extra += 1;
        }
    }

    let mut cli_prompts_checked = 0usize;
    let mut cli_steps_checked = 0usize;
    let mut cli_diffs = 0usize;
    let mut cli_prompt_ids = Vec::new();
    for &idx in &cli_indices {
        let fp = &fixture.prompts[idx];
        let csv: String = fp
            .prompt_ids
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut cmd = Command::new(uor_chat_bin);
        cmd.arg("--discrete")
            .arg(p4_path)
            .arg("--prompt-ids")
            .arg(&csv)
            .arg("--max-tokens")
            .arg(fp.max_tokens.to_string())
            .arg("--p1-trace");
        if !fp.read_enabled {
            cmd.arg("--read-mode").arg("no_read");
        }
        let output = cmd.output()?;
        if !output.status.success() {
            return Err(fail(format!(
                "uor-chat failed on {}: {}",
                fp.id,
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let val: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        if val["artifact_sha256"].as_str() != Some(fixture.p4_artifact_sha256.as_str()) {
            return Err(fail(format!(
                "uor-chat artifact_sha256 mismatch on {}",
                fp.id
            )));
        }
        let got_gen: Vec<u16> = serde_json::from_value(val["generated_ids"].clone())?;
        let got_steps: Vec<FixtureStep> = serde_json::from_value(val["p4_steps"].clone())?;
        if got_gen != fp.generated_ids || got_steps != fp.steps {
            cli_diffs += 1;
            eprintln!(
                "M1 CLI DIFF on {}:\nCUDA: {:?}\nCLI:  {:?}",
                fp.id, fp.steps, got_steps
            );
        }
        cli_prompts_checked += 1;
        cli_steps_checked += got_steps.len();
        cli_prompt_ids.push(fp.id.clone());
    }

    let pass = inproc_diffs == 0 && cli_diffs == 0;
    let report = json!({
        "schema": "uor-r4.p4-m1-parity-summary/1",
        "cubin_sha256": fixture.cubin_sha256,
        "p4_artifact_sha256": fixture.p4_artifact_sha256,
        "p4_state_sha256": fixture.p4_state_sha256,
        "fixture_sha256": fixture_sha,
        "host_arch": std::env::consts::ARCH,
        "host_os": std::env::consts::OS,
        "prompts_total": fixture.prompts.len(),
        "inproc_steps_checked": inproc_steps_checked,
        "inproc_diffs": inproc_diffs,
        "uor_chat_cli_prompts_checked": cli_prompts_checked,
        "uor_chat_cli_steps_checked": cli_steps_checked,
        "uor_chat_cli_prompt_ids": cli_prompt_ids,
        "uor_chat_cli_diffs": cli_diffs,
        "pass": pass,
    });
    fs::write(out_path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !pass {
        return Err(fail("M1 parity verification failed"));
    }
    Ok(())
}
