#![forbid(unsafe_code)]

use crate::{fail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use uor_r4_core::answer_oracle::FrozenAnswers;
use uor_r4_count_runtime::p3_state::{
    apply_lawful_stop_and_rep_guard, extract_turn_pair_keys, paired_band_compare, LawfulStopConfig,
    S4ConjunctionTable, S4ControlMode, P3_EOS, P3_Q48_ONE, P3_VOCABULARY,
};
use uor_r4_discrete::kn_ceiling::{CompactCountLookup, FrozenPickMixture};
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;
use uor_r4_tokenizer::ByteBpeTokenizer;

pub const C_CORPUS_SHA256: &str =
    "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
pub const COMPACT_V3_SHA256: &str =
    "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const TRUNK_SHA256: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const TOKENIZER_SHA256: &str =
    "d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89";
pub const FROZEN_S4_SHA256: &str =
    "cd0e43859bcd72acfa08342d7ff08a0827c582c4cd8e1592218e6ca4c43f8771";
pub const FRESH256_PLAN_SHA256: &str =
    "0ed1ab98ab50bbad6c484c4e7e4ffeeb8bb7ad17d162251826302efe86d75715";
pub const FRESH256_INPUTS_SHA256: &str =
    "1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a";
pub const FRESH256_LABELS_SHA256: &str =
    "3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be";
pub const FRESH256_MANIFEST_SHA256: &str =
    "7c33f3f7e19edcb4b1a7aa5e5eab3bb0056db9351067d809f62f97b4a8880964";

const PERIOD_TOKEN_ID: u16 = 16;

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub fn sattolo_permute_cands(cands: &[(u8, u16)], seed: u64) -> Vec<(u8, u16)> {
    let mut perm: Vec<usize> = (0..cands.len()).collect();
    if perm.len() > 1 {
        let mut state = splitmix64(seed);
        let mut k = perm.len() - 1;
        while k > 0 {
            state = splitmix64(state);
            let j = (state % (k as u64)) as usize;
            perm.swap(k, j);
            k -= 1;
        }
    }
    perm.into_iter().map(|idx| cands[idx]).collect()
}

const R4_JOB_LITERALS: [&str; 8] = [
    "tallowchandler",
    "ropemaker",
    "wheelwright",
    "bargepilot",
    "sluicewarden",
    "kilnmaster",
    "peatcutter",
    "locktender",
];

const R4_HOME_LITERALS: [&str; 8] = [
    "shalehead cove",
    "rowanbank rise",
    "fenbridge reach",
    "thornwick dale",
    "gorsepoint heath",
    "peatmoor bank",
    "sedgeford quay",
    "cragstone dell",
];

const R4_STEM_JOB_QUERIES: [&str; 3] = [
    "Give the exact name of my job.",
    "Confirm my job title.",
    "Report what my job has become.",
];
const R4_STEM_HOME_QUERIES: [&str; 3] = [
    "Give the exact place where I live.",
    "Confirm my home location.",
    "Report where my residence sits today.",
];

const R4_SYNONYM_JOB_QUERIES: [&str; 3] = [
    "State my occupation.",
    "Which trade do I practice?",
    "Clarify my profession.",
];
const R4_SYNONYM_HOME_QUERIES: [&str; 3] = [
    "State my dwelling.",
    "Which town shelters me?",
    "Clarify my quarters.",
];

#[derive(Clone)]
struct CausalPromptCase {
    #[allow(dead_code)]
    id: String,
    pair_id: String,
    stratum: String,
    query_text: String,
    context_turns: Vec<Vec<u16>>,
    context_texts: Vec<String>,
    query_ids: Vec<u16>,
    answers: FrozenAnswers,
}

fn load_fresh256_cases(
    panel_dir: &Path,
    tok: &ByteBpeTokenizer,
) -> Result<(Vec<CausalPromptCase>, [u8; 32])> {
    let read_pin = |name: &str, exp_sha: &str| -> Result<Vec<u8>> {
        let bytes = std::fs::read(panel_dir.join(name))?;
        if hex::encode(Sha256::digest(&bytes)) != exp_sha {
            return Err(fail(format!("m2-fresh256/{name} SHA-256 mismatch")));
        }
        Ok(bytes)
    };
    let plan_raw = read_pin("plan.json", FRESH256_PLAN_SHA256)?;
    let inputs_raw = read_pin("inputs.json", FRESH256_INPUTS_SHA256)?;
    let labels_raw = read_pin("labels.json", FRESH256_LABELS_SHA256)?;
    let manifest_raw = read_pin("manifest.json", FRESH256_MANIFEST_SHA256)?;
    let manifest: Value = serde_json::from_slice(&manifest_raw)?;
    let inputs: Value = serde_json::from_slice(&inputs_raw)?;
    let labels: Value = serde_json::from_slice(&labels_raw)?;
    let plan: Value = serde_json::from_slice(&plan_raw)?;

    let in_cases = inputs["cases"]
        .as_array()
        .ok_or_else(|| fail("missing inputs cases"))?;
    let lb_cases = labels["cases"]
        .as_array()
        .ok_or_else(|| fail("missing labels cases"))?;
    let hists = plan["histories"]
        .as_array()
        .ok_or_else(|| fail("missing plan histories"))?;
    if in_cases.len() != 256 || lb_cases.len() != 256 || manifest["rows"] != 256 {
        return Err(fail("expected 256 cases in m2-fresh256"));
    }
    let mut strata_by_id: BTreeMap<String, String> = BTreeMap::new();
    for h in hists {
        let hid = h["id"].as_str().ok_or_else(|| fail("missing hist id"))?;
        let st = h["stratum"].as_str().unwrap_or("unknown").to_string();
        strata_by_id.insert(format!("fresh-{hid}-job"), st.clone());
        strata_by_id.insert(format!("fresh-{hid}-home"), st);
    }
    let mut out = Vec::with_capacity(256);
    for (ci, li) in in_cases.iter().zip(lb_cases.iter()) {
        let id = ci["id"]
            .as_str()
            .ok_or_else(|| fail("missing id"))?
            .to_string();
        if li["id"].as_str() != Some(id.as_str()) {
            return Err(fail("m2-fresh256 id alignment mismatch"));
        }
        let pair_id = li["pair_id"]
            .as_str()
            .ok_or_else(|| fail("missing pair_id"))?
            .to_string();
        let answers: FrozenAnswers = serde_json::from_value(li["answers"].clone())?;
        answers
            .validate()
            .map_err(|e| fail(format!("answers invalid: {e}")))?;
        let mut context_turns = Vec::new();
        let mut context_texts = Vec::new();
        for seg in ci["segments"]
            .as_array()
            .ok_or_else(|| fail("missing segments"))?
        {
            if seg["kind"].as_str() == Some("Context") {
                let tids: Vec<u16> = seg["token_ids"]
                    .as_array()
                    .ok_or_else(|| fail("missing token_ids"))?
                    .iter()
                    .map(|v| {
                        u16::try_from(v.as_u64().unwrap_or(0)).map_err(|e| fail(e.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let u32_tids: Vec<u32> = tids.iter().map(|&x| u32::from(x)).collect();
                let decoded = tok.decode(&u32_tids);
                context_turns.push(tids);
                context_texts.push(decoded);
            }
        }
        let query_ids: Vec<u16> = ci["query_ids"]
            .as_array()
            .ok_or_else(|| fail("missing query_ids"))?
            .iter()
            .map(|v| u16::try_from(v.as_u64().unwrap_or(0)).map_err(|e| fail(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let query_u32: Vec<u32> = query_ids.iter().map(|&x| u32::from(x)).collect();
        let query_text = tok.decode(&query_u32);
        let stratum = strata_by_id
            .get(&id)
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        out.push(CausalPromptCase {
            id,
            pair_id,
            stratum,
            query_text,
            context_turns,
            context_texts,
            query_ids,
            answers,
        });
    }
    let sha: [u8; 32] = Sha256::digest(&inputs_raw).into();
    Ok((out, sha))
}

fn build_r4_heldout_cases(tok: &ByteBpeTokenizer) -> Vec<CausalPromptCase> {
    let mut out = Vec::with_capacity(240);
    for h in 0..120usize {
        let job_lit = R4_JOB_LITERALS[h % R4_JOB_LITERALS.len()];
        let home_lit = R4_HOME_LITERALS[(h * 3 + 1) % R4_HOME_LITERALS.len()];
        let turn_a = format!("My current job is {job_lit}.");
        let turn_b = format!("I currently live in {home_lit}.");
        let turns = if h % 2 == 0 {
            vec![turn_a, turn_b]
        } else {
            vec![turn_b, turn_a]
        };
        let context_turns: Vec<Vec<u16>> = turns
            .iter()
            .map(|s| tok.encode(s).into_iter().map(|x| x as u16).collect())
            .collect();
        let (jq, hq, stratum) = if h < 60 {
            (
                R4_STEM_JOB_QUERIES[h % R4_STEM_JOB_QUERIES.len()],
                R4_STEM_HOME_QUERIES[h % R4_STEM_HOME_QUERIES.len()],
                "r4_in_stem_heldout",
            )
        } else {
            (
                R4_SYNONYM_JOB_QUERIES[h % R4_SYNONYM_JOB_QUERIES.len()],
                R4_SYNONYM_HOME_QUERIES[h % R4_SYNONYM_HOME_QUERIES.len()],
                "r4_oov_synonym_heldout",
            )
        };
        let pair_id = format!("r4-{h}");
        for (rel, q, lit) in [("job", jq, job_lit), ("home", hq, home_lit)] {
            out.push(CausalPromptCase {
                id: format!("r4-{h}-{rel}"),
                pair_id: pair_id.clone(),
                stratum: stratum.into(),
                query_text: q.into(),
                context_turns: context_turns.clone(),
                context_texts: turns.clone(),
                query_ids: tok.encode(q).into_iter().map(|x| x as u16).collect(),
                answers: FrozenAnswers {
                    intent: uor_r4_core::answer_oracle::RecordedValueIntent::Current,
                    accepted: vec![format!("{lit}.")],
                },
            });
        }
    }
    out
}

fn extract_causal_role_spans(
    context_texts: &[String],
    tok: &ByteBpeTokenizer,
) -> BTreeMap<(u8, u16), Vec<u16>> {
    let mut map = BTreeMap::new();
    for text in context_texts {
        let lower = text.trim();
        if let Some(val) = lower
            .strip_prefix("My current job is ")
            .or_else(|| lower.strip_prefix("My current job has changed to "))
            .and_then(|s| s.strip_suffix('.'))
        {
            let ids: Vec<u16> = tok.encode(val).into_iter().map(|x| x as u16).collect();
            map.insert((1u8, 0u16), ids);
        } else if let Some(val) = lower
            .strip_prefix("I currently live in ")
            .or_else(|| lower.strip_prefix("I now live in "))
            .and_then(|s| s.strip_suffix('.'))
        {
            let ids: Vec<u16> = tok.encode(val).into_iter().map(|x| x as u16).collect();
            map.insert((2u8, 0u16), ids);
        }
    }
    map
}

#[allow(clippy::too_many_arguments)]
fn free_generate_case(
    case: &CausalPromptCase,
    tok: &ByteBpeTokenizer,
    lookup: &CompactCountLookup<'_>,
    a1s: &FrozenPickMixture,
    trunk: &IntegerDiscreteModel,
    s4: &S4ConjunctionTable,
    mode: S4ControlMode,
    precomputed_pick_idx: Option<Option<usize>>,
) -> Result<(String, bool, bool)> {
    let mut prefix: Vec<u16> = vec![0u16];
    for t in &case.context_turns {
        prefix.extend_from_slice(t);
    }
    prefix.extend_from_slice(&case.query_ids);
    if prefix.len() > 190 {
        prefix.drain(1..prefix.len() - 180);
    }
    let role_spans = extract_causal_role_spans(&case.context_texts, tok);
    let candidates = vec![(1u8, 0u16), (2u8, 0u16), (3u8, 0u16)];
    let turn_keys = extract_turn_pair_keys(&case.query_ids, 0)?;
    let picked_idx: Option<usize> = match precomputed_pick_idx {
        Some(idx_opt) => idx_opt,
        None => s4
            .pick_pointer(&turn_keys, &candidates, mode)
            .map(|(ci, _)| ci),
    };
    let chosen_span: Option<Vec<u16>> = match mode {
        S4ControlMode::NoMemory => None,
        _ => picked_idx.and_then(|ci| role_spans.get(&candidates[ci]).cloned()),
    };
    let abstained = chosen_span.is_none();

    let cfg = LawfulStopConfig::default();
    let mut gen_tokens: Vec<u16> = Vec::new();
    let mut eos_stopped = false;
    let max_steps = 24usize;
    let mut step = 0usize;
    while step < max_steps {
        let pfx_start = prefix.len().saturating_sub(192);
        let ctx_win = &prefix[pfx_start..];
        let u32_win: Vec<u32> = ctx_win.iter().map(|&x| u32::from(x)).collect();
        let picked: Option<(u16, u32)> = if matches!(mode, S4ControlMode::NoMemory) {
            None
        } else {
            match trunk.observe_identity_pick(&u32_win) {
                Ok(diag) => {
                    let d = diag.block_words[0][1];
                    diag.successor.map(|s| (s as u16, d))
                }
                Err(_) => None,
            }
        };
        let mut scores = vec![1u64; P3_VOCABULARY];
        lookup.visit_next_masses(ctx_win, a1s, picked, |tok_id, _bkt, q48| {
            scores[usize::from(tok_id)] = q48.max(1);
        })?;
        let mut span_exhausted = false;
        if let Some(span) = &chosen_span {
            if step < span.len() {
                scores[usize::from(span[step])] = P3_Q48_ONE;
            } else if step == span.len() {
                scores[usize::from(PERIOD_TOKEN_ID)] = P3_Q48_ONE;
            } else {
                span_exhausted = true;
            }
        }
        let tr = apply_lawful_stop_and_rep_guard(&gen_tokens, &mut scores, span_exhausted, cfg)?;
        let next_tok = tr.greedy_token;
        if next_tok == P3_EOS {
            eos_stopped = true;
            break;
        }
        gen_tokens.push(next_tok);
        prefix.push(next_tok);
        step += 1;
    }
    let u32_gen: Vec<u32> = gen_tokens.iter().map(|&x| u32::from(x)).collect();
    let text = tok.decode(&u32_gen);
    let ok = eos_stopped && case.answers.accepts(&text);
    Ok((text, ok, abstained))
}

struct PanelScores {
    rows: usize,
    pairs: usize,
    row_hits: usize,
    abstained_rows: usize,
    pairs_both_complete_distinct: usize,
    hits_vec: Vec<bool>,
    by_query: BTreeMap<String, [usize; 2]>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_panel_mode(
    cases: &[CausalPromptCase],
    tok: &ByteBpeTokenizer,
    lookup: &CompactCountLookup<'_>,
    a1s: &FrozenPickMixture,
    trunk: &IntegerDiscreteModel,
    s4: &S4ConjunctionTable,
    #[cfg(feature = "cuda")] gpu: &crate::cuda::P4BrierGpuEngine,
    kind: &str,
) -> Result<(PanelScores, usize)> {
    let base_cands = vec![(1u8, 0u16), (2u8, 0u16), (3u8, 0u16)];
    #[allow(unused_mut)]
    let mut gpu_picks: Vec<Option<Option<usize>>> = vec![None; cases.len()];
    #[allow(unused_mut)]
    let mut gpu_divs = 0usize;
    if kind != "no_memory" {
        let mut all_keys = Vec::with_capacity(cases.len());
        let mut all_cands = Vec::with_capacity(cases.len());
        for (idx, c) in cases.iter().enumerate() {
            let tk = extract_turn_pair_keys(&c.query_ids, 0)?;
            all_keys.push(tk);
            if kind == "shuffled_role" {
                all_cands.push(sattolo_permute_cands(&base_cands, 900 + idx as u64));
            } else {
                all_cands.push(base_cands.clone());
            }
        }
        #[cfg(feature = "cuda")]
        {
            let gpu_scores = gpu.s4_vote_batch(s4, &all_keys, &all_cands)?;
            for (idx, sc) in gpu_scores.iter().enumerate() {
                let mode = if kind == "shuffled_role" {
                    S4ControlMode::ShuffledRole {
                        seed: 900 + idx as u64,
                    }
                } else {
                    S4ControlMode::Active
                };
                if sc != &s4.vote_candidates(&all_keys[idx], &base_cands, mode) {
                    gpu_divs += 1;
                }
                let mut best: Option<(usize, i64)> = None;
                for (ci, &val) in sc.iter().enumerate() {
                    if val > 0 && best.is_none_or(|(_, bs)| val > bs) {
                        best = Some((ci, val));
                    }
                }
                gpu_picks[idx] = Some(best.map(|(ci, _)| ci));
            }
        }
        let _ = (&all_keys, &all_cands);
    }

    let mut hits_vec = Vec::with_capacity(cases.len());
    let mut abstained_rows = 0usize;
    let mut pair_groups: BTreeMap<String, Vec<(bool, String)>> = BTreeMap::new();
    let mut by_query: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    for (idx, c) in cases.iter().enumerate() {
        let mode = match kind {
            "active" => S4ControlMode::Active,
            "no_memory" => S4ControlMode::NoMemory,
            "shuffled_role" => S4ControlMode::ShuffledRole {
                seed: 900 + idx as u64,
            },
            _ => return Err(fail("unknown panel mode")),
        };
        let (text, ok, abstained) =
            free_generate_case(c, tok, lookup, a1s, trunk, s4, mode, gpu_picks[idx])?;
        hits_vec.push(ok);
        abstained_rows += usize::from(abstained);
        let q_stat = by_query.entry(c.query_text.clone()).or_insert([0usize; 2]);
        q_stat[0] += usize::from(ok);
        q_stat[1] += 1;
        pair_groups
            .entry(c.pair_id.clone())
            .or_default()
            .push((ok, text));
    }
    let row_hits = hits_vec.iter().filter(|&&b| b).count();
    let mut pairs_both_complete_distinct = 0usize;
    for items in pair_groups.values() {
        if items.len() == 2 && items[0].0 && items[1].0 && items[0].1 != items[1].1 {
            pairs_both_complete_distinct += 1;
        }
    }
    Ok((
        PanelScores {
            rows: cases.len(),
            pairs: pair_groups.len(),
            row_hits,
            abstained_rows,
            pairs_both_complete_distinct,
            hits_vec,
            by_query,
        },
        gpu_divs,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn run_reply_verify(
    s4_path: &Path,
    compact_v3_path: &Path,
    a1s_weights_path: &Path,
    a1s_seal_json: &Path,
    trunk_path: &Path,
    tokenizer_path: &Path,
    fresh256_dir: &Path,
    out_json_path: &Path,
) -> Result<Value> {
    let tok_bytes = std::fs::read(tokenizer_path)?;
    if hex::encode(Sha256::digest(&tok_bytes)) != TOKENIZER_SHA256 {
        return Err(fail("tokenizer SHA-256 mismatch"));
    }
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
        .ok_or_else(|| fail("failed parsing tokenizer.json"))?;

    let trunk = IntegerDiscreteModel::load(trunk_path)?;
    if trunk.artifact_sha256() != TRUNK_SHA256 {
        return Err(fail("seed1.d11 SHA-256 mismatch"));
    }

    let seal: Value = serde_json::from_slice(&std::fs::read(a1s_seal_json)?)?;
    let binding_hex = seal["manifest_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing manifest_sha256"))?;
    let mut binding = [0u8; 32];
    hex::decode_to_slice(binding_hex, &mut binding)?;
    let a1s = FrozenPickMixture::from_bytes(&std::fs::read(a1s_weights_path)?, binding)?;

    let mut c_corpus_sha = [0u8; 32];
    hex::decode_to_slice(C_CORPUS_SHA256, &mut c_corpus_sha)?;
    let mut compact_v3_sha = [0u8; 32];
    hex::decode_to_slice(COMPACT_V3_SHA256, &mut compact_v3_sha)?;
    let compact_bytes = std::fs::read(compact_v3_path)?;
    let lookup = CompactCountLookup::from_bytes(&compact_bytes, c_corpus_sha, compact_v3_sha)?;

    let s4_bytes = std::fs::read(s4_path)?;
    let s4_sha_hex = hex::encode(Sha256::digest(&s4_bytes));
    if s4_sha_hex != FROZEN_S4_SHA256 {
        return Err(fail(format!(
            "frozen S4 SHA-256 mismatch: {s4_sha_hex} != {FROZEN_S4_SHA256}"
        )));
    }
    let mut s4_sha = [0u8; 32];
    hex::decode_to_slice(&s4_sha_hex, &mut s4_sha)?;
    let s4 = S4ConjunctionTable::from_bytes(&s4_bytes, c_corpus_sha, s4_sha)?;

    let (fresh256_cases, fresh256_sha) = load_fresh256_cases(fresh256_dir, &tok)?;
    let r4_cases = build_r4_heldout_cases(&tok);

    #[cfg(feature = "cuda")]
    let gpu = crate::cuda::P4BrierGpuEngine::new()?;

    let (f_act, fd1) = evaluate_panel_mode(
        &fresh256_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4,
        #[cfg(feature = "cuda")]
        &gpu,
        "active",
    )?;
    let (f_nom, fd2) = evaluate_panel_mode(
        &fresh256_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4,
        #[cfg(feature = "cuda")]
        &gpu,
        "no_memory",
    )?;
    let (f_shuf, fd3) = evaluate_panel_mode(
        &fresh256_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4,
        #[cfg(feature = "cuda")]
        &gpu,
        "shuffled_role",
    )?;
    let f_pb_nom = paired_band_compare(&f_act.hits_vec, &f_nom.hits_vec)?;
    let f_pb_shuf = paired_band_compare(&f_act.hits_vec, &f_shuf.hits_vec)?;

    let (r4_act, rd1) = evaluate_panel_mode(
        &r4_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4,
        #[cfg(feature = "cuda")]
        &gpu,
        "active",
    )?;
    let (r4_nom, rd2) = evaluate_panel_mode(
        &r4_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4,
        #[cfg(feature = "cuda")]
        &gpu,
        "no_memory",
    )?;
    let (r4_shuf, rd3) = evaluate_panel_mode(
        &r4_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4,
        #[cfg(feature = "cuda")]
        &gpu,
        "shuffled_role",
    )?;
    let divs = fd1 + fd2 + fd3 + rd1 + rd2 + rd3;
    if divs != 0 {
        return Err(fail(format!("cold S4 GPU vs CPU vote divergence: {divs}")));
    }
    let r4_pb_nom = paired_band_compare(&r4_act.hits_vec, &r4_nom.hits_vec)?;
    let r4_pb_shuf = paired_band_compare(&r4_act.hits_vec, &r4_shuf.hits_vec)?;

    let mut r4_stem = 0usize;
    let mut r4_oov = 0usize;
    for (idx, c) in r4_cases.iter().enumerate() {
        if r4_act.hits_vec[idx] {
            if c.stratum == "r4_in_stem_heldout" {
                r4_stem += 1;
            } else {
                r4_oov += 1;
            }
        }
    }
    let by_q: BTreeMap<String, Value> = r4_act
        .by_query
        .iter()
        .map(|(q, st)| (q.clone(), json!({ "hits": st[0], "total": st[1] })))
        .collect();

    let report = json!({
        "phase": "Stage B5b Cold-Loaded Frozen S4 Reply Verification",
        "s4_sha256": s4_sha_hex,
        "s4_bytes": s4_bytes.len(),
        "fresh256_inputs_sha256": hex::encode(fresh256_sha),
        "s4_gpu_vs_cpu_divergences": divs,
        "r1_r2_m2_fresh256": {
            "rows": f_act.rows,
            "pairs": f_act.pairs,
            "active_row_hits": f_act.row_hits,
            "active_pairs": f_act.pairs_both_complete_distinct,
            "no_memory_row_hits": f_nom.row_hits,
            "shuffled_role_row_hits": f_shuf.row_hits,
            "shuffled_role_abstained": f_shuf.abstained_rows,
            "paired_vs_no_memory": {
                "net_gain": f_pb_nom.net_gain,
                "band_ceil": f_pb_nom.band_ceil,
                "beats_control": f_pb_nom.beats_control_by_band,
            },
            "paired_vs_shuffled_role": {
                "net_gain": f_pb_shuf.net_gain,
                "band_ceil": f_pb_shuf.band_ceil,
                "beats_control": f_pb_shuf.beats_control_by_band,
            },
        },
        "r4_heldout_phrasing": {
            "rows": r4_act.rows,
            "pairs": r4_act.pairs,
            "active_row_hits": r4_act.row_hits,
            "active_abstained": r4_act.abstained_rows,
            "active_pairs": r4_act.pairs_both_complete_distinct,
            "active_in_stem": r4_stem,
            "active_oov_synonym": r4_oov,
            "by_query": by_q,
            "no_memory_row_hits": r4_nom.row_hits,
            "shuffled_role_row_hits": r4_shuf.row_hits,
            "shuffled_role_abstained": r4_shuf.abstained_rows,
            "paired_vs_no_memory": {
                "net_gain": r4_pb_nom.net_gain,
                "band_ceil": r4_pb_nom.band_ceil,
                "beats_control": r4_pb_nom.beats_control_by_band,
            },
            "paired_vs_shuffled_role": {
                "net_gain": r4_pb_shuf.net_gain,
                "band_ceil": r4_pb_shuf.band_ceil,
                "beats_control": r4_pb_shuf.beats_control_by_band,
            },
        },
    });
    std::fs::write(out_json_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_r4_count_runtime::p3_state::{S4ConjunctionTable, S4ControlMode, S4PointerEntry};

    #[test]
    fn three_candidate_sattolo_cold_parity_matches_p3_state() {
        let ents = [
            S4PointerEntry {
                role_class: 1,
                span_offset: 0,
                count: 14,
                err_i8: 6,
            },
            S4PointerEntry {
                role_class: 2,
                span_offset: 0,
                count: 5,
                err_i8: -2,
            },
            S4PointerEntry {
                role_class: 3,
                span_offset: 0,
                count: 3,
                err_i8: 0,
            },
        ];
        let tbl = S4ConjunctionTable::from_sorted_rows([7u8; 32], &[(101u32, &ents[..])]).unwrap();
        let bytes = tbl.encode_bytes().unwrap();
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let reloaded = S4ConjunctionTable::from_bytes(&bytes, [7u8; 32], sha).unwrap();
        let base = vec![(1u8, 0u16), (2u8, 0u16), (3u8, 0u16)];
        for idx in 0..32u64 {
            let perm = sattolo_permute_cands(&base, 900 + idx);
            let a = tbl.vote_candidates(
                &[101u32],
                &base,
                S4ControlMode::ShuffledRole { seed: 900 + idx },
            );
            let b = reloaded.vote_candidates(&[101u32], &perm, S4ControlMode::Active);
            assert_eq!(a, b);
        }
    }
}
