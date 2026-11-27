#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use m2_student_corpus_ingest::{hex, BOS, EOS, UORT_HEADER, VOCAB};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uor_r4_core::answer_oracle::FrozenAnswers;
use uor_r4_count_runtime::p4_serving::{P4Session, P4State, P4_MAX_CONTEXT, P4_VOCAB};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, ClientArtifactFile, DiscreteCountModel,
};
use crate::lawful_likelihood::{bits_q12, masses_from_scores, LOG2_SCALE, PROBABILITY_ONE};
use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

pub const FRESH256_PLAN_SHA256: &str =
    "0ed1ab98ab50bbad6c484c4e7e4ffeeb8bb7ad17d162251826302efe86d75715";
pub const FRESH256_INPUTS_SHA256: &str =
    "1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a";
pub const FRESH256_LABELS_SHA256: &str =
    "3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be";
pub const FRESH256_MANIFEST_SHA256: &str =
    "7c33f3f7e19edcb4b1a7aa5e5eab3bb0056db9351067d809f62f97b4a8880964";

/// Ordinary, strictly unwrapped token-level prediction interface.
/// Every Gate 0/1/2 model is graded through this exact interface:
/// zero regex span stripping, zero post-hoc score forcing, zero external role hints.
pub trait UnwrappedTokenModel {
    fn sync_prefix(&mut self, prefix: &[u16]) -> Result<(), String>;
    /// Compute raw integer scores for all `VOCAB` tokens and return greedy next token id.
    fn step_scores(&mut self, scores_out: &mut [i64; VOCAB as usize]) -> Result<u16, String>;
    /// Append one observed token to the live session (caller stops on EOS).
    fn observe_token(&mut self, tok: u16) -> Result<(), String>;
}

/// Wrapper executing the real frozen `P4State` + `P4Session::step` strictly unwrapped.
pub struct P4UnwrappedRunner<'a> {
    state: &'a P4State<'a>,
    session: P4Session,
    read_enabled: bool,
}

impl<'a> P4UnwrappedRunner<'a> {
    pub fn new(state: &'a P4State<'a>, read_enabled: bool) -> Self {
        Self {
            state,
            session: P4Session::new(),
            read_enabled,
        }
    }
}

impl<'a> UnwrappedTokenModel for P4UnwrappedRunner<'a> {
    fn sync_prefix(&mut self, prefix: &[u16]) -> Result<(), String> {
        let len = prefix.len();
        let slice = if len > P4_MAX_CONTEXT {
            &prefix[len - P4_MAX_CONTEXT..]
        } else {
            prefix
        };
        self.session
            .sync_prompt(slice)
            .map_err(|e| format!("P4Session::sync_prompt: {e}"))
    }

    fn step_scores(&mut self, scores_out: &mut [i64; VOCAB as usize]) -> Result<u16, String> {
        let (tok, _) = self
            .session
            .step(self.state, self.read_enabled, false)
            .map_err(|e| format!("P4Session::step: {e}"))?;
        let raw = self.session.last_scores_q48();
        scores_out[0] = 0;
        for id in 1..P4_VOCAB {
            scores_out[id] = raw[id] as i64;
        }
        Ok(tok)
    }

    fn observe_token(&mut self, tok: u16) -> Result<(), String> {
        self.session
            .observe_generated(tok)
            .map_err(|e| format!("P4Session::observe_generated({tok}): {e}"))
    }
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Exact rounded micro-BPB from `(bits_q12, bytes)`.
pub fn micro_bpb(bits_q12_sum: u128, bytes_sum: u128) -> u64 {
    if bytes_sum == 0 {
        return 0;
    }
    let den = u128::from(LOG2_SCALE) * bytes_sum;
    let num = bits_q12_sum * 1_000_000u128;
    ((num + den / 2) / den) as u64
}

/// Signed micro-BPB difference: `bpb(a) - bpb(b)` computed cleanly in integer arithmetic.
pub fn signed_micro_bpb_diff(
    a_bits: u128,
    a_bytes: u128,
    b_bits: u128,
    b_bytes: u128,
) -> i64 {
    let ma = micro_bpb(a_bits, a_bytes) as i64;
    let mb = micro_bpb(b_bits, b_bytes) as i64;
    ma - mb
}

/// Exact PRLL Q12 bits for one target token from a full-vocabulary integer score array.
pub fn prll_bits_for_target(scores: &[i64; VOCAB as usize], target: u16) -> Result<u64, String> {
    let masses = masses_from_scores(scores).map_err(|e| format!("masses_from_scores: {e}"))?;
    let sum_check: u128 = masses.iter().map(|&m| u128::from(m)).sum();
    if sum_check != u128::from(PROBABILITY_ONE) || masses.contains(&0) {
        return Err("PRLL invariant violation".into());
    }
    bits_q12(masses[usize::from(target)]).map_err(|e| format!("bits_q12: {e}"))
}

/// A parsed record sliced out of a Protocol-2 UORT stream (`tokens.u16` + sidecars).
#[derive(Clone, Debug)]
pub struct StreamRecord {
    pub tokens: Vec<u16>,
    #[allow(dead_code)]
    pub mask: Vec<u8>,
    pub novel: Vec<u8>,
    pub absent: Vec<u8>,
    /// Index of the first `mask == 1` token in the record (start of turn-1 answer).
    pub first_ans_start: usize,
    /// End index (exclusive) of the turn-1 answer (including its closing `EOS`).
    pub first_ans_end: usize,
}

/// Load records from a Protocol-2 DEV directory (`<split>.tokens.u16`, `.mask.u8`, `.novel.u8`, `.absent.u8`).
pub fn load_proto2_dev_records(dir: &Path, split: &str, max_records: usize) -> Result<Vec<StreamRecord>, String> {
    let tok_raw = std::fs::read(dir.join(format!("{split}.tokens.u16")))
        .map_err(|e| format!("{split}.tokens.u16: {e}"))?;
    if tok_raw.len() < UORT_HEADER || &tok_raw[0..4] != b"UORT" {
        return Err("invalid UORT header".into());
    }
    let total = u64::from_le_bytes(tok_raw[8..16].try_into().unwrap()) as usize;
    let body = &tok_raw[UORT_HEADER..];
    if body.len() != total * 2 {
        return Err("UORT length mismatch".into());
    }
    let mask_all = std::fs::read(dir.join(format!("{split}.mask.u8")))
        .map_err(|e| format!("{split}.mask.u8: {e}"))?;
    let novel_all = std::fs::read(dir.join(format!("{split}.novel.u8")))
        .map_err(|e| format!("{split}.novel.u8: {e}"))?;
    let absent_all = std::fs::read(dir.join(format!("{split}.absent.u8")))
        .map_err(|e| format!("{split}.absent.u8: {e}"))?;
    if mask_all.len() != total || novel_all.len() != total || absent_all.len() != total {
        return Err("sidecar length mismatch".into());
    }
    let mut tokens_all = Vec::with_capacity(total);
    for chunk in body.chunks_exact(2) {
        tokens_all.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }

    let mut recs = Vec::new();
    let mut start = 0usize;
    while start < total {
        if tokens_all[start] != BOS as u16 {
            return Err(format!("expected BOS at token {start}"));
        }
        let mut end = start + 1;
        while end < total && tokens_all[end] != BOS as u16 {
            end += 1;
        }
        let toks = tokens_all[start..end].to_vec();
        let msk = mask_all[start..end].to_vec();
        let nov = novel_all[start..end].to_vec();
        let abs = absent_all[start..end].to_vec();
        if let Some(ans_s) = msk.iter().position(|&m| m == 1) {
            let mut ans_e = ans_s;
            while ans_e < msk.len() && msk[ans_e] == 1 {
                ans_e += 1;
            }
            if ans_s >= 2 && ans_e > ans_s {
                recs.push(StreamRecord {
                    tokens: toks,
                    mask: msk,
                    novel: nov,
                    absent: abs,
                    first_ans_start: ans_s,
                    first_ans_end: ans_e,
                });
            }
        }
        start = end;
    }

    if max_records == 0 || recs.len() <= max_records {
        return Ok(recs);
    }
    // Deterministic stratified selection evenly across all records.
    let mut out = Vec::with_capacity(max_records);
    let n = recs.len();
    for i in 0..max_records {
        let idx = (i * n) / max_records;
        out.push(recs[idx].clone());
    }
    Ok(out)
}

/// Evaluate Gate E1: Answer-Only Stratified Likelihood + Counterfactual Prompt-Shuffle (`ΔBPB_prompt`).
pub fn eval_e1_likelihood_and_shuffle<M: UnwrappedTokenModel>(
    model: &mut M,
    records: &[StreamRecord],
    id_bytes_table: &[u8],
    max_answer_tokens_per_rec: usize,
) -> Result<Value, String> {
    let mut all_bits = 0u128;
    let mut all_bytes = 0u128;
    let mut all_targets = 0u64;
    let mut all_top1 = 0u64;

    let mut na_bits = 0u128;
    let mut na_bytes = 0u128;
    let mut na_targets = 0u64;
    let mut na_top1 = 0u64;

    let mut ic_bits = 0u128;
    let mut ic_bytes = 0u128;
    let mut ic_targets = 0u64;

    let mut eos_targets = 0u64;
    let mut eos_top1 = 0u64;

    // Per-record `(true_bits, swapped_bits, bytes)` for paired bootstrap CI of ΔBPB_prompt.
    let mut shuffle_pairs: Vec<(u128, u128, u128)> = Vec::with_capacity(records.len());
    let mut scores = [0i64; VOCAB as usize];
    let n = records.len();

    for (rec_idx, rec) in records.iter().enumerate() {
        let ans_s = rec.first_ans_start;
        let ans_e = rec.first_ans_end.min(ans_s + max_answer_tokens_per_rec);
        let true_prompt = &rec.tokens[..ans_s];
        model.sync_prefix(true_prompt)?;

        let mut rec_true_bits = 0u128;
        let mut rec_bytes = 0u128;
        for pos in ans_s..ans_e {
            let target = rec.tokens[pos];
            let pred = model.step_scores(&mut scores)?;
            let bq = u128::from(prll_bits_for_target(&scores, target)?);
            let nb = u128::from(id_bytes_table[usize::from(target)]);

            all_bits += bq;
            all_bytes += nb;
            all_targets += 1;
            if pred == target {
                all_top1 += 1;
            }
            if target == EOS as u16 {
                eos_targets += 1;
                if pred == EOS as u16 {
                    eos_top1 += 1;
                }
            }
            if rec.novel[pos] == 1 && rec.absent[pos] == 1 {
                na_bits += bq;
                na_bytes += nb;
                na_targets += 1;
                if pred == target {
                    na_top1 += 1;
                }
            }
            if rec.absent[pos] == 0 {
                ic_bits += bq;
                ic_bytes += nb;
                ic_targets += 1;
            }
            rec_true_bits += bq;
            rec_bytes += nb;

            if target == EOS as u16 || pos + 1 == ans_e {
                break;
            }
            model.observe_token(target)?;
        }

        // Counterfactual Prompt-Shuffle: feed a disjoint record's prompt, score the SAME gold answer.
        if n >= 2 {
            let other_idx = (rec_idx + n / 2) % n;
            let other_prompt = &records[other_idx].tokens[..records[other_idx].first_ans_start];
            model.sync_prefix(other_prompt)?;
            let mut rec_swap_bits = 0u128;
            for pos in ans_s..ans_e {
                let target = rec.tokens[pos];
                let _ = model.step_scores(&mut scores)?;
                rec_swap_bits += u128::from(prll_bits_for_target(&scores, target)?);
                if target == EOS as u16 || pos + 1 == ans_e {
                    break;
                }
                model.observe_token(target)?;
            }
            shuffle_pairs.push((rec_true_bits, rec_swap_bits, rec_bytes));
        }
    }

    let true_total_bits: u128 = shuffle_pairs.iter().map(|&(t, _, _)| t).sum();
    let swap_total_bits: u128 = shuffle_pairs.iter().map(|&(_, s, _)| s).sum();
    let total_pair_bytes: u128 = shuffle_pairs.iter().map(|&(_, _, b)| b).sum();
    let delta_micro_bpb =
        signed_micro_bpb_diff(swap_total_bits, total_pair_bytes, true_total_bits, total_pair_bytes);

    // Deterministic 1,000-draw paired bootstrap 95% CI on ΔBPB_prompt (integer micro-BPB).
    let (ci_lo, ci_hi) = if shuffle_pairs.len() >= 2 {
        let mut draws = Vec::with_capacity(1000);
        let mut rng = 0xD870_E101_2026_1127u64;
        let m = shuffle_pairs.len();
        for _ in 0..1000 {
            let mut tb = 0u128;
            let mut sb = 0u128;
            let mut by = 0u128;
            for _ in 0..m {
                rng = splitmix64(rng);
                let idx = (rng as usize) % m;
                let (t, s, b) = shuffle_pairs[idx];
                tb += t;
                sb += s;
                by += b;
            }
            draws.push(signed_micro_bpb_diff(sb, by, tb, by));
        }
        draws.sort_unstable();
        (draws[25], draws[974])
    } else {
        (delta_micro_bpb, delta_micro_bpb)
    };

    Ok(json!({
        "records": records.len(),
        "max_answer_tokens_per_rec": max_answer_tokens_per_rec,
        "all_answer": {
            "targets": all_targets,
            "bytes": all_bytes as u64,
            "bits_q12": all_bits.to_string(),
            "micro_bpb": micro_bpb(all_bits, all_bytes),
            "top1": all_top1,
            "top1_permille": (all_top1 * 1000).checked_div(all_targets).unwrap_or(0),
        },
        "novel_absent_answer": {
            "targets": na_targets,
            "bytes": na_bytes as u64,
            "bits_q12": na_bits.to_string(),
            "micro_bpb": micro_bpb(na_bits, na_bytes),
            "top1": na_top1,
            "top1_permille": (na_top1 * 1000).checked_div(na_targets).unwrap_or(0),
        },
        "in_context_answer": {
            "targets": ic_targets,
            "bytes": ic_bytes as u64,
            "micro_bpb": micro_bpb(ic_bits, ic_bytes),
        },
        "gold_eos": {
            "targets": eos_targets,
            "top1": eos_top1,
            "top1_permille": (eos_top1 * 1000).checked_div(eos_targets).unwrap_or(0),
        },
        "counterfactual_prompt_shuffle": {
            "true_prompt_micro_bpb": micro_bpb(true_total_bits, total_pair_bytes),
            "swapped_prompt_micro_bpb": micro_bpb(swap_total_bits, total_pair_bytes),
            "delta_micro_bpb": delta_micro_bpb,
            "ci95_lo_micro_bpb": ci_lo,
            "ci95_hi_micro_bpb": ci_hi,
            "prompt_conditioned_pass": ci_lo > 0,
        },
    }))
}

fn has_fourgram_loop(tokens: &[u16]) -> bool {
    if tokens.len() < 8 {
        return false;
    }
    let mut seen: BTreeSet<[u16; 4]> = BTreeSet::new();
    let mut repeats = 0usize;
    for w in tokens.windows(4) {
        let gram = [w[0], w[1], w[2], w[3]];
        if !seen.insert(gram) {
            repeats += 1;
            if repeats >= 3 {
                return true;
            }
        }
    }
    false
}

/// Generate strictly unwrapped greedy tokens up to `cap`, stopping on `EOS`.
pub fn generate_unwrapped<M: UnwrappedTokenModel>(
    model: &mut M,
    prefix: &[u16],
    cap: usize,
) -> Result<(Vec<u16>, bool), String> {
    model.sync_prefix(prefix)?;
    let mut out = Vec::with_capacity(cap);
    let mut scores = [0i64; VOCAB as usize];
    for _ in 0..cap {
        let tok = model.step_scores(&mut scores)?;
        if tok == EOS as u16 {
            return Ok((out, true));
        }
        out.push(tok);
        if tok == BOS as u16 {
            return Ok((out, false));
        }
        model.observe_token(tok)?;
    }
    Ok((out, false))
}

/// Evaluate Gate E2: Unwrapped Free-Run Collapse & Counterfactual Swap Divergence (`N <= 200`).
pub fn eval_e2_freerun_collapse<M: UnwrappedTokenModel>(
    model: &mut M,
    records: &[StreamRecord],
    tok: &ByteBpeTokenizer,
    n_prompts: usize,
    gen_cap: usize,
) -> Result<Value, String> {
    let n = records.len().min(n_prompts);
    let mut first_hist: BTreeMap<u16, u64> = BTreeMap::new();
    let mut generated: Vec<Vec<u16>> = Vec::with_capacity(n);
    let mut natural_eos = 0u64;
    let mut cap_trunc = 0u64;
    let mut premature_eos = 0u64;
    let mut loops = 0u64;

    for rec in records.iter().take(n) {
        let prefix = &rec.tokens[..rec.first_ans_start];
        let (gen, hit_eos) = generate_unwrapped(model, prefix, gen_cap)?;
        if hit_eos {
            natural_eos += 1;
            if gen.is_empty() {
                premature_eos += 1;
            }
        } else {
            cap_trunc += 1;
        }
        if has_fourgram_loop(&gen) {
            loops += 1;
        }
        if let Some(&first_tok) = gen.first() {
            *first_hist.entry(first_tok).or_insert(0) += 1;
        } else {
            *first_hist.entry(EOS as u16).or_insert(0) += 1;
        }
        generated.push(gen);
    }

    let (modal_id, modal_count) = first_hist
        .iter()
        .max_by(|&(&ka, &va), &(&kb, &vb)| va.cmp(&vb).then_with(|| kb.cmp(&ka)))
        .map(|(&k, &v)| (k, v))
        .unwrap_or((0, 0));
    let modal_surface =
        String::from_utf8_lossy(&tok.decode_bytes(&[u32::from(modal_id)])).into_owned();

    // Counterfactual swap divergence across 64 disjoint pairs sharing `\nAssistant:`.
    let n_pairs = (n / 2).min(64);
    let mut diverge_step3 = 0u64;
    let mut diverge_step1 = 0u64;
    for i in 0..n_pairs {
        let a = &generated[i];
        let b = &generated[i + n / 2];
        let p3_a = &a[..a.len().min(3)];
        let p3_b = &b[..b.len().min(3)];
        if p3_a != p3_b {
            diverge_step3 += 1;
        }
        if a.first() != b.first() {
            diverge_step1 += 1;
        }
    }

    let n_u64 = n.max(1) as u64;
    let modal_permille = (modal_count * 1000) / n_u64;
    let eos_permille = (natural_eos * 1000) / n_u64;
    let trunc_permille = (cap_trunc * 1000) / n_u64;
    let loop_permille = (loops * 1000) / n_u64;
    let div3_permille = if n_pairs > 0 {
        (diverge_step3 * 1000) / (n_pairs as u64)
    } else {
        0
    };

    Ok(json!({
        "prompts": n,
        "gen_cap": gen_cap,
        "modal_first_token_id": modal_id,
        "modal_first_token_surface": modal_surface,
        "modal_first_token_count": modal_count,
        "modal_first_token_permille": modal_permille,
        "distinct_first_tokens": first_hist.len(),
        "natural_eos": natural_eos,
        "natural_eos_permille": eos_permille,
        "cap_truncation": cap_trunc,
        "cap_truncation_permille": trunc_permille,
        "premature_eos": premature_eos,
        "fourgram_loops": loops,
        "fourgram_loop_permille": loop_permille,
        "counterfactual_pairs": n_pairs,
        "diverge_by_step1": diverge_step1,
        "diverge_by_step3": diverge_step3,
        "diverge_by_step3_permille": div3_permille,
        "gate_e2_pass": modal_permille < 150
            && first_hist.len() >= 40
            && eos_permille >= 700
            && loop_permille < 50
            && div3_permille > 800,
    }))
}

#[derive(Clone)]
struct Fresh256Case {
    #[allow(dead_code)]
    id: String,
    pair_id: String,
    context_texts: Vec<String>,
    query_text: String,
    answers: FrozenAnswers,
}

fn load_fresh256_panel(panel_dir: &Path, tok: &ByteBpeTokenizer) -> Result<Vec<Fresh256Case>, String> {
    let read_pin = |name: &str, exp_sha: &str| -> Result<Vec<u8>, String> {
        let p = panel_dir.join(name);
        let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let got = hex(&Sha256::digest(&bytes));
        if got != exp_sha {
            return Err(format!("m2-fresh256/{name} SHA-256 mismatch: {got} != {exp_sha}"));
        }
        Ok(bytes)
    };
    let _plan_raw = read_pin("plan.json", FRESH256_PLAN_SHA256)?;
    let inputs_raw = read_pin("inputs.json", FRESH256_INPUTS_SHA256)?;
    let labels_raw = read_pin("labels.json", FRESH256_LABELS_SHA256)?;
    let _manifest_raw = read_pin("manifest.json", FRESH256_MANIFEST_SHA256)?;

    let inputs: Value = serde_json::from_slice(&inputs_raw).map_err(|e| e.to_string())?;
    let labels: Value = serde_json::from_slice(&labels_raw).map_err(|e| e.to_string())?;
    let in_cases = inputs["cases"].as_array().ok_or("missing inputs.cases")?;
    let lb_cases = labels["cases"].as_array().ok_or("missing labels.cases")?;
    if in_cases.len() != 256 || lb_cases.len() != 256 {
        return Err("expected 256 cases in m2-fresh256".into());
    }
    let mut out = Vec::with_capacity(256);
    for (ci, li) in in_cases.iter().zip(lb_cases.iter()) {
        let id = ci["id"].as_str().ok_or("missing id")?.to_string();
        if li["id"].as_str() != Some(id.as_str()) {
            return Err("m2-fresh256 id alignment mismatch".into());
        }
        let pair_id = li["pair_id"].as_str().ok_or("missing pair_id")?.to_string();
        let answers: FrozenAnswers =
            serde_json::from_value(li["answers"].clone()).map_err(|e| e.to_string())?;
        let mut context_texts = Vec::new();
        for seg in ci["segments"].as_array().ok_or("missing segments")? {
            if seg["kind"].as_str() == Some("Context") {
                let ids: Vec<u32> = seg["token_ids"]
                    .as_array()
                    .ok_or("missing token_ids")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                context_texts.push(tok.decode(&ids));
            }
        }
        let q_ids: Vec<u32> = ci["query_ids"]
            .as_array()
            .ok_or("missing query_ids")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();
        let query_text = tok.decode(&q_ids);
        out.push(Fresh256Case {
            id,
            pair_id,
            context_texts,
            query_text,
            answers,
        });
    }
    Ok(out)
}

fn encode_protocol2_prompt(
    tok: &ByteBpeTokenizer,
    context_turns: &[String],
    query: &str,
) -> Result<Vec<u16>, String> {
    let proto = DialogueProtocol::literal_roles_v2(tok)
        .map_err(|e| format!("DialogueProtocol::literal_roles_v2: {e}"))?;
    let enc = proto
        .bind(tok)
        .map_err(|e| format!("DialogueProtocol::bind: {e}"))?;
    let mut msgs: Vec<Message<'_>> = Vec::with_capacity(context_turns.len() + 1);
    for t in context_turns {
        msgs.push(Message {
            role: "user",
            content: t.as_str(),
        });
    }
    msgs.push(Message {
        role: "user",
        content: query,
    });
    Ok(enc
        .encode_assistant_prefix(&msgs)
        .tokens
        .into_iter()
        .map(|x| x as u16)
        .collect())
}

fn grade_m2_arm<M: UnwrappedTokenModel>(
    model: &mut M,
    cases: &[Fresh256Case],
    tok: &ByteBpeTokenizer,
    mode: &str,
    gen_cap: usize,
) -> Result<Value, String> {
    let mut complete = 0u64;
    let mut pair_hits: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (idx, c) in cases.iter().enumerate() {
        let prompt = match mode {
            "active" => encode_protocol2_prompt(tok, &c.context_texts, &c.query_text)?,
            "no_memory" => encode_protocol2_prompt(tok, &[], &c.query_text)?,
            "role_swapped" => {
                // Swap the query with the opposite-role mate in the same pair (adjacent idx ^ 1).
                let mate = &cases[idx ^ 1];
                encode_protocol2_prompt(tok, &c.context_texts, &mate.query_text)?
            }
            _ => return Err(format!("unknown arm {mode}")),
        };
        let (gen, hit_eos) = generate_unwrapped(model, &prompt, gen_cap)?;
        let gen_u32: Vec<u32> = gen.iter().map(|&x| u32::from(x)).collect();
        let reply = tok.decode(&gen_u32).trim().to_string();
        let matched = hit_eos && c.answers.accepts(&reply);
        if matched {
            complete += 1;
            pair_hits.entry(c.pair_id.clone()).or_default().push(reply);
        }
    }
    let pairs_both_complete_distinct = pair_hits
        .values()
        .filter(|v| v.len() == 2 && v[0] != v[1])
        .count() as u64;
    Ok(json!({
        "arm": mode,
        "rows": cases.len(),
        "complete": complete,
        "pairs_both_complete_distinct": pairs_both_complete_distinct,
        "bar_52_of_128_pass": pairs_both_complete_distinct >= 52 && complete >= 104,
    }))
}

/// Evaluate Gate E3: Strictly unwrapped `m2-fresh256` (`128` pairs) + `64` abstention prompts.
pub fn eval_e3_unwrapped_m2<M: UnwrappedTokenModel>(
    model: &mut M,
    panel_dir: &Path,
    tok: &ByteBpeTokenizer,
    gen_cap: usize,
) -> Result<Value, String> {
    let cases = load_fresh256_panel(panel_dir, tok)?;
    let active = grade_m2_arm(model, &cases, tok, "active", gen_cap)?;
    let no_memory = grade_m2_arm(model, &cases, tok, "no_memory", gen_cap)?;
    let role_swapped = grade_m2_arm(model, &cases, tok, "role_swapped", gen_cap)?;

    // 64 deterministic unanswerable abstention prompts (no fact given, asks for job/home).
    let queries = [
        "What is my current job?",
        "Where do I currently live?",
        "State my occupation.",
        "State my dwelling.",
    ];
    let distractors = [
        "Today is Tuesday and the sky is clear.",
        "I bought three notebooks and a blue pen.",
        "The train arrived four minutes early.",
        "Please keep this note concise.",
    ];
    let mut abstained = 0u64;
    for i in 0..64usize {
        let ctx = vec![distractors[i % distractors.len()].to_string()];
        let q = queries[(i / 4) % queries.len()];
        let prompt = encode_protocol2_prompt(tok, &ctx, q)?;
        let (gen, hit_eos) = generate_unwrapped(model, &prompt, gen_cap)?;
        let gen_u32: Vec<u32> = gen.iter().map(|&x| u32::from(x)).collect();
        let lower = tok.decode(&gen_u32).to_ascii_lowercase();
        let signals_unknown = lower.contains("not stated")
            || lower.contains("don't know")
            || lower.contains("do not know")
            || lower.contains("unknown")
            || lower.contains("not provided")
            || lower.contains("not mentioned")
            || lower.contains("no information");
        if hit_eos && signals_unknown {
            abstained += 1;
        }
    }

    Ok(json!({
        "panel": "m2-fresh256-unwrapped",
        "no_regex_span_stripping": true,
        "no_score_forcing": true,
        "active": active,
        "no_memory": no_memory,
        "role_swapped": role_swapped,
        "abstention_64": {
            "total": 64,
            "abstained": abstained,
        },
    }))
}

/// Evaluate Gate E4: `100` deterministic checkable items (40 arithmetic, 40 long-distance QA, 20 Rust).
pub fn eval_e4_checkable_micro<M: UnwrappedTokenModel>(
    model: &mut M,
    tok: &ByteBpeTokenizer,
    gen_cap: usize,
) -> Result<Value, String> {
    // 40 deterministic arithmetic & state-tracking items.
    let mut arith_pass = 0u64;
    for i in 0..40usize {
        let a = 11 + ((i * 7) % 37);
        let b = 5 + ((i * 13) % 29);
        let (prompt_text, expected) = if i % 2 == 0 {
            (
                format!("Compute {a} + {b}. Reply with only the integer."),
                format!("{}", a + b),
            )
        } else {
            let c = 3 + (i % 9);
            (
                format!("Start with {a}. Add {b}, then subtract {c}. Reply with only the integer."),
                format!("{}", a + b - c),
            )
        };
        let prompt = encode_protocol2_prompt(tok, &[], &prompt_text)?;
        let (gen, _) = generate_unwrapped(model, &prompt, gen_cap)?;
        let gen_u32: Vec<u32> = gen.iter().map(|&x| u32::from(x)).collect();
        let reply = tok.decode(&gen_u32);
        let tokens_ws: Vec<&str> = reply.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).collect();
        if tokens_ws.first().copied() == Some(expected.as_str()) {
            arith_pass += 1;
        }
    }

    // 40 long-distance passage QA items (answer absent from final 16+ tokens).
    let entities = [
        ("cobalt", "harbor", "lantern", "cedar"),
        ("amber", "valley", "compass", "birch"),
        ("quartz", "meadow", "whistle", "maple"),
        ("basalt", "canyon", "satchel", "alder"),
        ("ochre", "ridge", "kettle", "rowan"),
        ("silver", "orchard", "journal", "linden"),
        ("crimson", "estuary", "prism", "willow"),
        ("indigo", "plateau", "cipher", "spruce"),
    ];
    let mut qa_pass = 0u64;
    for i in 0..40usize {
        let (color, place, item, tree) = entities[i % entities.len()];
        let filler = "The morning weather report noted calm winds, moderate visibility along the coastal road, and routine ferry crossings every thirty minutes throughout the afternoon.";
        let (passage, q, expected) = match i % 4 {
            0 => (
                format!("The key token is {color}. {filler}"),
                "What was the key token stated at the beginning?",
                color,
            ),
            1 => (
                format!("The meeting place is {place}. {filler}"),
                "Which meeting place was named at the beginning?",
                place,
            ),
            2 => (
                format!("The packed item is {item}. {filler}"),
                "What packed item was mentioned at the start?",
                item,
            ),
            _ => (
                format!("The landmark tree is {tree}. {filler}"),
                "Which landmark tree was mentioned at the start?",
                tree,
            ),
        };
        let prompt = encode_protocol2_prompt(tok, &[passage], q)?;
        let (gen, _) = generate_unwrapped(model, &prompt, gen_cap)?;
        let gen_u32: Vec<u32> = gen.iter().map(|&x| u32::from(x)).collect();
        let reply_lower = tok.decode(&gen_u32).to_ascii_lowercase();
        if reply_lower.contains(expected) {
            qa_pass += 1;
        }
    }

    // 20 tiny deterministic Rust completion items.
    let rust_items = [
        ("Complete the Rust expression: `fn add(a: i32, b: i32) -> i32 { `", "a + b"),
        ("Complete the Rust expression: `fn sub(a: i32, b: i32) -> i32 { `", "a - b"),
        ("Complete the Rust expression: `fn mul(a: i32, b: i32) -> i32 { `", "a * b"),
        ("Complete the Rust expression: `fn double(x: i32) -> i32 { `", "x * 2"),
        ("Complete the Rust expression: `fn is_zero(x: i32) -> bool { `", "x == 0"),
        ("Complete the Rust expression: `fn is_even(x: i32) -> bool { `", "x % 2 == 0"),
        ("Complete the Rust expression: `fn neg(x: i32) -> i32 { `", "-x"),
        ("Complete the Rust expression: `fn identity(x: u32) -> u32 { `", "x"),
        ("Complete the Rust expression: `fn min_val(a: i32, b: i32) -> i32 { `", "a.min(b)"),
        ("Complete the Rust expression: `fn max_val(a: i32, b: i32) -> i32 { `", "a.max(b)"),
    ];
    let mut rust_pass = 0u64;
    for i in 0..20usize {
        let (q, exp) = rust_items[i % rust_items.len()];
        let prompt = encode_protocol2_prompt(tok, &[], q)?;
        let (gen, _) = generate_unwrapped(model, &prompt, gen_cap)?;
        let gen_u32: Vec<u32> = gen.iter().map(|&x| u32::from(x)).collect();
        let reply = tok.decode(&gen_u32);
        if reply.contains(exp) {
            rust_pass += 1;
        }
    }

    Ok(json!({
        "total": 100,
        "arithmetic_40": arith_pass,
        "long_distance_qa_40": qa_pass,
        "rust_micro_20": rust_pass,
        "passed": arith_pass + qa_pass + rust_pass,
    }))
}

/// Run the complete unwrapped Gate 0 evaluation suite (`E1–E4`) on a frozen `p4.d11` deployment.
#[allow(clippy::too_many_arguments)]
pub fn run_eval_p4(
    p4_path: &Path,
    tokenizer_path: &Path,
    proto2_dir: &Path,
    fresh256_dir: &Path,
    out_json: &Path,
    e1_records: usize,
    e1_max_ans_tok: usize,
    e2_prompts: usize,
) -> Result<Value, String> {
    let tok_bytes = std::fs::read(tokenizer_path)
        .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
        .ok_or("invalid tokenizer.json")?;
    let tok_sha = hex(&Sha256::digest(&tok_bytes));
    let id_bytes_table = std::fs::read(proto2_dir.join("id-bytes.u8"))
        .map_err(|e| format!("id-bytes.u8: {e}"))?;
    if id_bytes_table.len() != VOCAB as usize {
        return Err("id-bytes.u8 length != 4096".into());
    }

    let loaded = load_client_artifact(p4_path, false)
        .map_err(|e| format!("load_client_artifact: {e}"))?;
    let (model, state_bytes, expected_sha, identity) = match loaded {
        ClientArtifactFile::Embedded {
            model,
            state_bytes,
            expected_state_sha256,
            identity,
        } => (model, state_bytes, expected_state_sha256, identity),
        ClientArtifactFile::Legacy(_) => return Err("expected embedded P4 deployment".into()),
    };
    let artifact_sha = identity.artifact_sha256.clone();
    let dcm = DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_sha, identity)
        .map_err(|e| format!("bind_embedded: {e}"))?;
    let p4_state = dcm
        .p4
        .as_deref()
        .map(|p| &p.inner)
        .ok_or("artifact is not a P4 deployment")?;

    let dev_tune_recs = load_proto2_dev_records(proto2_dir, "dev-tune", e1_records.max(e2_prompts))?;
    let mut runner = P4UnwrappedRunner::new(p4_state, true);

    eprintln!("gate0-eval: running E1 ({} records, cap {} ans tok)...", dev_tune_recs.len().min(e1_records), e1_max_ans_tok);
    let e1 = eval_e1_likelihood_and_shuffle(
        &mut runner,
        &dev_tune_recs[..dev_tune_recs.len().min(e1_records)],
        &id_bytes_table,
        e1_max_ans_tok,
    )?;
    eprintln!("gate0-eval: running E2 ({} free-run prompts)...", dev_tune_recs.len().min(e2_prompts));
    let e2 = eval_e2_freerun_collapse(&mut runner, &dev_tune_recs, &tok, e2_prompts, 128)?;
    eprintln!("gate0-eval: running E3 (unwrapped m2-fresh256 + 64 abstention)...");
    let e3 = eval_e3_unwrapped_m2(&mut runner, fresh256_dir, &tok, 32)?;
    eprintln!("gate0-eval: running E4 (100 deterministic checkable items)...");
    let e4 = eval_e4_checkable_micro(&mut runner, &tok, 32)?;

    let report = json!({
        "schema": "uor-r4.gate0-unwrapped-eval/1",
        "model_kind": "frozen_p4_unwrapped",
        "artifact_sha256": artifact_sha,
        "tokenizer_sha256": tok_sha,
        "e1_likelihood_and_shuffle": e1,
        "e2_freerun_collapse": e2,
        "e3_unwrapped_m2": e3,
        "e4_checkable_micro": e4,
    });
    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        out_json,
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(report)
}
