#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![cfg_attr(feature = "cuda", deny(unsafe_code))]
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

//! Gate 1 PR 2 (`antigravity-3`): Pure-integer AOT CUDA Butterfly-Rotor trainer,
//! Bar A (`chat-v0-p2`) PRLL scorer, strict unwrapped E1–E4 evaluator, and Apple M1 bench.

mod cuda_backend;

#[allow(dead_code, clippy::identity_op, clippy::manual_is_multiple_of)]
#[path = "../../../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
mod lawful_likelihood;

#[allow(dead_code, unused_imports)]
use cuda_backend::FlatModelBuffers;
#[cfg(feature = "cuda")]
use lawful_likelihood::bits_q12;
use lawful_likelihood::LOG2_SCALE;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
#[cfg(feature = "cuda")]
use uor_r4_core::answer_oracle::FrozenAnswers;
use uor_r4_native_vector::butterfly_rotor::{
    isqrt_u128, splitmix64, ButterflyRotorModel, LossKind,
};
#[cfg(feature = "cuda")]
use uor_r4_native_vector::butterfly_rotor::{ButterflyRotorConfig, HIER_VOCAB};
#[cfg(feature = "cuda")]
use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};
#[cfg(feature = "cuda")]
use uor_r4_tokenizer::ByteBpeTokenizer;

#[cfg(feature = "cuda")]
const EOS_ID: u16 = 1;
const UORT_HEADER_BYTES: usize = 64;
#[cfg(feature = "cuda")]
const PIN_HELDOUT_SHA256: &str = "e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef";
const PIN_HELDOUT_TOKENS: usize = 6_194_589;
const PIN_STREAM_BYTES: u128 = 17_576_697;
#[cfg(feature = "cuda")]
const BAR_A_CONTEXT: usize = 384;
#[cfg(feature = "cuda")]
const BAR_A_WINDOWS: usize = 512;
#[cfg(feature = "cuda")]
const BAR_A_STRIDE: usize = 12_098;
#[cfg(feature = "cuda")]
const PIN_TOKENIZER_SHA256: &str =
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
#[cfg(feature = "cuda")]
const PIN_M2_INPUTS_SHA256: &str =
    "1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a";
#[cfg(feature = "cuda")]
const PIN_M2_LABELS_SHA256: &str =
    "3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be";
#[cfg(feature = "cuda")]
const PIN_TRAIN100M_TOK_SHA256: &str =
    "c804c3b084a49d11f2bfcd2a5cf741862df29b42e3442ccb609305e049aba590";
#[cfg(feature = "cuda")]
const PIN_TRAIN100M_MASK_SHA256: &str =
    "9e8d0cb6302ea890be5847c76133a731e7b4a080cfa096049cf37323072ae1d5";
#[cfg(feature = "cuda")]
const PIN_DEVTUNE_TOK_SHA256: &str =
    "99b714a67415d92e42750577110d9ce405625d93adabbd906739bbfca60d9c78";
#[cfg(feature = "cuda")]
const PIN_DEVTUNE_MASK_SHA256: &str =
    "643fccbbd63f38f4ac61dbedff02d75964c91809d9b277e75a27e5a8fbd30199";
#[cfg(feature = "cuda")]
const PIN_DEVTUNE_NOVEL_SHA256: &str =
    "6239ac461ebb4e32fe8f170290ffd78c6d3149e0214b67e8068a485816e29fe4";
#[cfg(feature = "cuda")]
const PIN_DEVTUNE_ABSENT_SHA256: &str =
    "8860f193f6f069037ac87f2a590a3a6ddd1d3ba72c0b3a0de148a6aa08cc658b";
#[cfg(feature = "cuda")]
const PIN_ID_BYTES_SHA256: &str =
    "befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9";

type Res<T> = std::result::Result<T, String>;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn read_and_pin(path: &Path, expected_sha: Option<&str>, label: &str) -> Res<Vec<u8>> {
    let bytes = fs::read(path).map_err(|e| format!("read {label} {}: {e}", path.display()))?;
    let got = sha256_hex(&bytes);
    if let Some(exp) = expected_sha {
        if got != exp {
            return Err(format!(
                "SHA-256 mismatch on {label}: got {got}, expected {exp}"
            ));
        }
    }
    Ok(bytes)
}

fn parse_uort_u16(bytes: &[u8]) -> Res<Vec<u16>> {
    if bytes.len() < UORT_HEADER_BYTES || &bytes[..4] != b"UORT" {
        return Err("invalid UORT header".to_owned());
    }
    let body = &bytes[UORT_HEADER_BYTES..];
    if !body.len().is_multiple_of(2) {
        return Err("odd UORT payload byte count".to_owned());
    }
    Ok(body
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn micro_bpb_from_q12(bits_sum: u128, byte_count: u64) -> u64 {
    if byte_count == 0 {
        return 0;
    }
    let den = u128::from(LOG2_SCALE) * u128::from(byte_count);
    (((bits_sum * 1_000_000u128) + (den / 2)) / den) as u64
}

fn bar_a_stream_micro_bpb(bits_sum: u128, n_targets: u64) -> u64 {
    if n_targets == 0 {
        return 0;
    }
    let num = bits_sum * (PIN_HELDOUT_TOKENS as u128);
    let den = u128::from(LOG2_SCALE) * u128::from(n_targets) * PIN_STREAM_BYTES;
    (((num * 1_000_000u128) + (den / 2)) / den) as u64
}

fn empirical_se_micro_bpb(window_mbpbs: &[u64]) -> u64 {
    let w = window_mbpbs.len() as u128;
    if w < 2 {
        return 0;
    }
    let sum: u128 = window_mbpbs.iter().map(|&x| u128::from(x)).sum();
    let mean = sum / w;
    let mut sq = 0u128;
    for &x in window_mbpbs {
        let d = u128::from(x).abs_diff(mean);
        sq += d * d;
    }
    isqrt_u128(sq / (w * (w - 1))) as u64
}

fn parse_loss_kind(name: &str, k: i32) -> Res<LossKind> {
    match name {
        "hui-belkin" | "hb" => Ok(LossKind::HuiBelkinRescaledSquare {
            pos_weight_k: k.max(1),
        }),
        "sparsemax" | "sp" => Ok(LossKind::Sparsemax64),
        other => Err(format!(
            "unknown loss kind `{other}` (use hui-belkin or sparsemax)"
        )),
    }
}

fn flag_val(args: &[String], key: &str) -> Res<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .ok_or_else(|| format!("missing required flag {key}"))
}

fn opt_flag(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

// ---------------------------------------------------------------------------
// Corpus chunking & Proto-2 DEV-tune record extraction
// ---------------------------------------------------------------------------

struct SplitChunks {
    tokens: Vec<u16>,
    mask: Vec<u8>,
    starts: Vec<usize>,
    lens: Vec<usize>,
}

/// Partition a UORT stream at `BOS = 0` boundaries into deterministic chunks of length `<= seq_len + 1`.
fn build_training_chunks(
    tokens: Vec<u16>,
    mask: Vec<u8>,
    chunk_toks: usize,
    all_non_bos: bool,
) -> Res<SplitChunks> {
    if tokens.len() != mask.len() {
        return Err(format!(
            "train token/mask length mismatch: {} != {}",
            tokens.len(),
            mask.len()
        ));
    }
    let mut eff_mask = mask;
    if all_non_bos {
        for (i, &tok) in tokens.iter().enumerate() {
            eff_mask[i] = u8::from(tok != 0);
        }
    }
    let mut starts = Vec::new();
    let mut lens = Vec::new();
    let n = tokens.len();
    let mut i = 0usize;
    while i < n {
        let mut end = i + 1;
        while end < n && tokens[end] != 0 {
            end += 1;
        }
        // Slice record `[i..end)` into contiguous chunks of `<= chunk_toks` (`>= 2`).
        let mut cur = i;
        while cur + 1 < end {
            let take = (end - cur).min(chunk_toks);
            let has_sup = eff_mask[cur + 1..cur + take].contains(&1);
            if has_sup {
                starts.push(cur);
                lens.push(take);
            }
            cur += take;
        }
        i = end;
    }
    Ok(SplitChunks {
        tokens,
        mask: eff_mask,
        starts,
        lens,
    })
}

#[derive(Clone)]
struct DevRec {
    tokens: Vec<u16>,
    #[allow(dead_code)]
    mask: Vec<u8>,
    novel: Vec<u8>,
    absent: Vec<u8>,
    first_ans_start: usize,
    first_ans_end: usize,
}

#[cfg(feature = "cuda")]
fn load_proto2_dev_records(proto2_dir: &Path, split: &str, max_recs: usize) -> Res<Vec<DevRec>> {
    if split != "dev-tune" {
        return Err(format!(
            "refusing non-dev-tune split `{split}` (DEV-test stays sealed)"
        ));
    }
    let toks = parse_uort_u16(&read_and_pin(
        &proto2_dir.join("dev-tune.tokens.u16"),
        Some(PIN_DEVTUNE_TOK_SHA256),
        "dev-tune.tokens.u16",
    )?)?;
    let mask = read_and_pin(
        &proto2_dir.join("dev-tune.mask.u8"),
        Some(PIN_DEVTUNE_MASK_SHA256),
        "dev-tune.mask.u8",
    )?;
    let novel = read_and_pin(
        &proto2_dir.join("dev-tune.novel.u8"),
        Some(PIN_DEVTUNE_NOVEL_SHA256),
        "dev-tune.novel.u8",
    )?;
    let absent = read_and_pin(
        &proto2_dir.join("dev-tune.absent.u8"),
        Some(PIN_DEVTUNE_ABSENT_SHA256),
        "dev-tune.absent.u8",
    )?;
    if toks.len() != mask.len() || toks.len() != novel.len() || toks.len() != absent.len() {
        return Err("proto2 sidecar length mismatch".to_owned());
    }
    let mut recs = Vec::new();
    let mut i = 0usize;
    while i < toks.len() {
        if toks[i] != 0 {
            return Err(format!("expected BOS=0 at dev-tune token {i}"));
        }
        let mut end = i + 1;
        while end < toks.len() && toks[end] != 0 {
            end += 1;
        }
        let r_toks = toks[i..end].to_vec();
        let r_mask = mask[i..end].to_vec();
        let r_novel = novel[i..end].to_vec();
        let r_absent = absent[i..end].to_vec();
        if let Some(ans_s) = r_mask.iter().position(|&m| m == 1) {
            let mut ans_e = ans_s;
            while ans_e < r_mask.len() && r_mask[ans_e] == 1 {
                ans_e += 1;
            }
            if ans_s >= 2 && ans_e > ans_s {
                recs.push(DevRec {
                    tokens: r_toks,
                    mask: r_mask,
                    novel: r_novel,
                    absent: r_absent,
                    first_ans_start: ans_s,
                    first_ans_end: ans_e,
                });
            }
        }
        i = end;
    }
    if max_recs == 0 || recs.len() <= max_recs {
        return Ok(recs);
    }
    // Exact Gate 0 `eval.rs` deterministic stratified selection evenly across all dev-tune records.
    let n = recs.len();
    let mut out = Vec::with_capacity(max_recs);
    for idx_i in 0..max_recs {
        let pick = (idx_i * n) / max_recs;
        out.push(recs[pick].clone());
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct ShufflePair {
    true_bits: u128,
    swap_bits: u128,
    bytes: u128,
}

fn signed_micro_bpb_diff(a_bits: u128, a_bytes: u128, b_bits: u128, b_bytes: u128) -> i64 {
    let ma = micro_bpb_from_q12(a_bits, a_bytes as u64) as i64;
    let mb = micro_bpb_from_q12(b_bits, b_bytes as u64) as i64;
    ma - mb
}

/// Exact Gate 0 `eval.rs` byte-weighted ratio-of-sums 1,000-draw paired bootstrap 95% CI.
fn paired_bootstrap_ci95_weighted(pairs: &[ShufflePair], seed: u64) -> (i64, i64) {
    let m = pairs.len();
    if m == 0 {
        return (0, 0);
    }
    let mut draws = Vec::with_capacity(1000);
    let mut rng = seed;
    for _ in 0..1000 {
        let mut tb = 0u128;
        let mut sb = 0u128;
        let mut by = 0u128;
        for _ in 0..m {
            rng = splitmix64(rng);
            let idx = (rng as usize) % m;
            let p = pairs[idx];
            tb += p.true_bits;
            sb += p.swap_bits;
            by += p.bytes;
        }
        draws.push(signed_micro_bpb_diff(sb, by, tb, by));
    }
    draws.sort_unstable();
    (draws[25], draws[974])
}

// ---------------------------------------------------------------------------
// Strict Unwrapped E2 / E3 / E4 Evaluation (aligned with Gate 0 `eval.rs`)
// ---------------------------------------------------------------------------

fn generate_unwrapped_cpu(
    model: &ButterflyRotorModel,
    prompt_tokens: &[u16],
    eos_id: u16,
    cap: usize,
) -> Res<(Vec<u16>, bool)> {
    let mut session = model.new_session();
    let mut last_tok = 0u16;
    for &t in prompt_tokens {
        last_tok = model
            .step(t, &mut session)
            .map_err(|e| e.to_string())?
            .greedy_token;
    }
    let mut gen = Vec::with_capacity(cap);
    for _ in 0..cap {
        if last_tok == eos_id {
            return Ok((gen, true));
        }
        gen.push(last_tok);
        if last_tok == 0 {
            return Ok((gen, false));
        }
        last_tok = model
            .step(last_tok, &mut session)
            .map_err(|e| e.to_string())?
            .greedy_token;
    }
    Ok((gen, false))
}

/// Exact Gate 0 `eval.rs:381-397` 4-gram loop detector (`>= 3` repeated 4-grams).
fn has_fourgram_loop(tokens: &[u16]) -> bool {
    if tokens.len() < 8 {
        return false;
    }
    let mut seen: BTreeSet<[u16; 4]> = BTreeSet::new();
    let mut repeats = 0usize;
    for w in tokens.windows(4) {
        let key = [w[0], w[1], w[2], w[3]];
        if !seen.insert(key) {
            repeats += 1;
            if repeats >= 3 {
                return true;
            }
        }
    }
    false
}

#[cfg(feature = "cuda")]
fn eval_e2_freerun_cpu(
    model: &ButterflyRotorModel,
    tok: &ByteBpeTokenizer,
    recs: &[DevRec],
    n_prompts: usize,
    gen_cap: usize,
) -> Res<Value> {
    let eos_id = EOS_ID;
    let space_id = 223u16;
    let take = n_prompts.min(recs.len());
    let mut first_tok_counts: BTreeMap<u16, usize> = BTreeMap::new();
    let mut first_nonspace_counts: BTreeMap<u16, usize> = BTreeMap::new();
    let mut natural_eos = 0usize;
    let mut premature_eos = 0usize;
    let mut cap_trunc = 0usize;
    let mut loops = 0usize;
    let mut gens: Vec<Vec<u16>> = Vec::with_capacity(take);

    for rec in recs.iter().take(take) {
        // Full intact prompt ending at `\nAssistant:` (exact Gate 0 `eval.rs:439` parity).
        let prompt = &rec.tokens[..rec.first_ans_start];
        let (gen, hit_eos) = generate_unwrapped_cpu(model, prompt, eos_id, gen_cap)?;
        if let Some(&t0) = gen.first() {
            *first_tok_counts.entry(t0).or_insert(0) += 1;
        } else {
            *first_tok_counts.entry(eos_id).or_insert(0) += 1;
        }
        if let Some(&t_ns) = gen.iter().find(|&&t| t != space_id) {
            *first_nonspace_counts.entry(t_ns).or_insert(0) += 1;
        } else {
            *first_nonspace_counts.entry(eos_id).or_insert(0) += 1;
        }
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
        gens.push(gen);
    }

    let (&modal_id, &modal_cnt) = first_tok_counts
        .iter()
        .max_by(|(&ka, &va), (&kb, &vb)| va.cmp(&vb).then_with(|| kb.cmp(&ka)))
        .unwrap_or((&0, &0));
    let (&modal_ns_id, &modal_ns_cnt) = first_nonspace_counts
        .iter()
        .max_by(|(&ka, &va), (&kb, &vb)| va.cmp(&vb).then_with(|| kb.cmp(&ka)))
        .unwrap_or((&0, &0));

    let pairs = (take / 2).min(64);
    let mut div1 = 0usize;
    let mut div3 = 0usize;
    for i in 0..pairs {
        let a = &gens[i];
        let b = &gens[i + take / 2];
        if a.first() != b.first() {
            div1 += 1;
        }
        let p3_a = &a[..a.len().min(3)];
        let p3_b = &b[..b.len().min(3)];
        if p3_a != p3_b {
            div3 += 1;
        }
    }

    let modal_ns_permille = (modal_ns_cnt * 1000) / take.max(1);
    let eos_permille = (natural_eos * 1000) / take.max(1);
    let loop_permille = (loops * 1000) / take.max(1);
    let div3_permille = (div3 * 1000) / pairs.max(1);
    let gate_pass = modal_ns_permille < 250
        && first_nonspace_counts.len() >= 20
        && eos_permille > 500
        && loop_permille < 150
        && div3_permille > 750;

    Ok(json!({
        "prompts": take,
        "gen_cap": gen_cap,
        "distinct_first_tokens": first_tok_counts.len(),
        "modal_first_token_id": modal_id,
        "modal_first_token_surface": tok.decode(&[u32::from(modal_id)]),
        "modal_first_token_permille": (modal_cnt * 1000) / take.max(1),
        "distinct_first_nonspace_tokens": first_nonspace_counts.len(),
        "modal_first_nonspace_id": modal_ns_id,
        "modal_first_nonspace_surface": tok.decode(&[u32::from(modal_ns_id)]),
        "modal_first_nonspace_permille": modal_ns_permille,
        "natural_eos": natural_eos,
        "natural_eos_permille": eos_permille,
        "premature_eos": premature_eos,
        "cap_truncation": cap_trunc,
        "fourgram_loops": loops,
        "fourgram_loop_permille": loop_permille,
        "counterfactual_pairs": pairs,
        "diverge_by_step1": div1,
        "diverge_by_step3": div3,
        "diverge_by_step3_permille": div3_permille,
        "gate_e2_pass": gate_pass,
    }))
}

#[cfg(feature = "cuda")]
fn encode_proto2_prompt(
    tok: &ByteBpeTokenizer,
    context_turns: &[String],
    query: &str,
) -> Res<Vec<u16>> {
    let proto = DialogueProtocol::literal_roles_v2(tok).map_err(|e| format!("{e}"))?;
    let enc = proto.bind(tok).map_err(|e| format!("{e}"))?;
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

#[cfg(feature = "cuda")]
struct Fresh256Case {
    pair_id: String,
    context_texts: Vec<String>,
    query_text: String,
    answers: FrozenAnswers,
}

#[cfg(feature = "cuda")]
fn load_fresh256_cases(m2_dir: &Path, tok: &ByteBpeTokenizer) -> Res<Vec<Fresh256Case>> {
    let inputs_bytes = read_and_pin(
        &m2_dir.join("inputs.json"),
        Some(PIN_M2_INPUTS_SHA256),
        "m2 inputs.json",
    )?;
    let labels_bytes = read_and_pin(
        &m2_dir.join("labels.json"),
        Some(PIN_M2_LABELS_SHA256),
        "m2 labels.json",
    )?;
    let inputs: Value = serde_json::from_slice(&inputs_bytes).map_err(|e| e.to_string())?;
    let labels: Value = serde_json::from_slice(&labels_bytes).map_err(|e| e.to_string())?;
    let in_cases = inputs["cases"].as_array().ok_or("missing inputs.cases")?;
    let lb_cases = labels["cases"].as_array().ok_or("missing labels.cases")?;
    if in_cases.len() != 256 || lb_cases.len() != 256 {
        return Err("expected 256 cases in m2-fresh256".to_owned());
    }
    let mut cases = Vec::with_capacity(256);
    for (ci, li) in in_cases.iter().zip(lb_cases.iter()) {
        if ci["id"] != li["id"] {
            return Err("m2-fresh256 id alignment mismatch".to_owned());
        }
        let pair_id = li["pair_id"].as_str().ok_or("missing pair_id")?.to_owned();
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
        cases.push(Fresh256Case {
            pair_id,
            context_texts,
            query_text: tok.decode(&q_ids),
            answers,
        });
    }
    Ok(cases)
}

#[cfg(feature = "cuda")]
fn eval_e3_m2_unwrapped_cpu(
    model: &ButterflyRotorModel,
    tok: &ByteBpeTokenizer,
    cases: &[Fresh256Case],
) -> Res<Value> {
    struct PairOutcome {
        ok: bool,
        tokens: Vec<u16>,
    }
    let proto = DialogueProtocol::literal_roles_v2(tok).map_err(|e| format!("{e}"))?;
    let run_arm = |arm: &str| -> Res<Value> {
        let mut complete = 0usize;
        let mut pair_hits: BTreeMap<String, Vec<PairOutcome>> = BTreeMap::new();
        for (idx, c) in cases.iter().enumerate() {
            let prompt = match arm {
                "active" => encode_proto2_prompt(tok, &c.context_texts, &c.query_text)?,
                "no_memory" => encode_proto2_prompt(tok, &[], &c.query_text)?,
                "role_swapped" => {
                    let mate = &cases[idx ^ 1];
                    encode_proto2_prompt(tok, &c.context_texts, &mate.query_text)?
                }
                _ => return Err(format!("unknown E3 arm {arm}")),
            };
            let (gen, hit_eos) = generate_unwrapped_cpu(model, &prompt, EOS_ID, 48)?;
            let gen_u32: Vec<u32> = gen.iter().map(|&x| u32::from(x)).collect();
            let decoded = tok.decode(&gen_u32);
            // Aligned with PR #4316 note: require `hit_eos` + `DialogueProtocol::reply_text` + distinct token-ID sequence!
            let reply = proto.reply_text(&decoded);
            let ok = hit_eos && c.answers.accepts(reply);
            if ok {
                complete += 1;
            }
            pair_hits
                .entry(c.pair_id.clone())
                .or_default()
                .push(PairOutcome { ok, tokens: gen });
        }
        let pairs_both_distinct = pair_hits
            .values()
            .filter(|v| v.len() == 2 && v[0].ok && v[1].ok && v[0].tokens != v[1].tokens)
            .count();
        Ok(json!({
            "arm": arm,
            "rows": cases.len(),
            "complete": complete,
            "pairs_both_complete_distinct": pairs_both_distinct,
            "bar_52_of_128_pass": pairs_both_distinct >= 52 && complete >= 104,
        }))
    };

    Ok(json!({
        "panel": "m2-fresh256-unwrapped-strict",
        "no_regex_span_stripping": true,
        "no_score_forcing": true,
        "requires_hit_eos": true,
        "active": run_arm("active")?,
        "no_memory": run_arm("no_memory")?,
        "role_swapped": run_arm("role_swapped")?,
    }))
}

/// Strict terminal-answer + `hit_eos` check for E4 (addresses PR #4316 review note #2).
fn strict_e4_terminal_match(reply: &str, expected: &str, hit_eos: bool) -> bool {
    if !hit_eos {
        return false;
    }
    let trimmed = reply.trim().trim_end_matches('.');
    if trimmed == expected {
        return true;
    }
    // Accept `"The answer is <expected>"` ONLY if `<expected>` is the exact final token.
    trimmed
        .split_whitespace()
        .last()
        .map(|tok| tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-'))
        == Some(expected)
}

#[cfg(feature = "cuda")]
fn eval_e4_checkable_cpu(model: &ButterflyRotorModel, tok: &ByteBpeTokenizer) -> Res<Value> {
    // Exact Gate 0 `eval.rs:725-835` 100-item checkable micro-suite, reporting BOTH strict
    // (`hit_eos` + terminal/exact) and Gate 0-compatible counts for apples-to-apples p4.d11 comparison.
    let proto = DialogueProtocol::literal_roles_v2(tok).map_err(|e| format!("{e}"))?;
    let mut arith_strict = 0usize;
    let mut arith_compat = 0usize;
    let mut qa_strict = 0usize;
    let mut qa_compat = 0usize;
    let mut rust_strict = 0usize;
    let mut rust_compat = 0usize;

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
        let p = encode_proto2_prompt(tok, &[], &prompt_text)?;
        let (gen, hit_eos) = generate_unwrapped_cpu(model, &p, EOS_ID, 48)?;
        let dec = tok.decode(&gen.iter().map(|&x| u32::from(x)).collect::<Vec<_>>());
        let reply = proto.reply_text(&dec);
        if strict_e4_terminal_match(reply, &expected, hit_eos) {
            arith_strict += 1;
        }
        let tokens_ws: Vec<&str> = dec
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .collect();
        if tokens_ws.first().copied() == Some(expected.as_str()) {
            arith_compat += 1;
        }
    }

    struct QaEntity {
        color: &'static str,
        place: &'static str,
        item: &'static str,
        tree: &'static str,
    }
    let entities = [
        QaEntity {
            color: "cobalt",
            place: "harbor",
            item: "lantern",
            tree: "cedar",
        },
        QaEntity {
            color: "amber",
            place: "valley",
            item: "compass",
            tree: "birch",
        },
        QaEntity {
            color: "quartz",
            place: "meadow",
            item: "whistle",
            tree: "maple",
        },
        QaEntity {
            color: "basalt",
            place: "canyon",
            item: "satchel",
            tree: "alder",
        },
        QaEntity {
            color: "ochre",
            place: "ridge",
            item: "kettle",
            tree: "rowan",
        },
        QaEntity {
            color: "silver",
            place: "orchard",
            item: "journal",
            tree: "linden",
        },
        QaEntity {
            color: "crimson",
            place: "estuary",
            item: "prism",
            tree: "willow",
        },
        QaEntity {
            color: "indigo",
            place: "plateau",
            item: "cipher",
            tree: "spruce",
        },
    ];
    let filler = "The morning weather report noted calm winds, moderate visibility along the coastal road, and routine ferry crossings every thirty minutes throughout the afternoon.";
    for i in 0..40usize {
        let ent = &entities[i % entities.len()];
        let (passage, q, expected) = match i % 4 {
            0 => (
                format!("The key token is {}. {filler}", ent.color),
                "What was the key token stated at the beginning?",
                ent.color,
            ),
            1 => (
                format!("The meeting place is {}. {filler}", ent.place),
                "Which meeting place was named at the beginning?",
                ent.place,
            ),
            2 => (
                format!("The packed item is {}. {filler}", ent.item),
                "What packed item was mentioned at the start?",
                ent.item,
            ),
            _ => (
                format!("The landmark tree is {}. {filler}", ent.tree),
                "Which landmark tree was mentioned at the start?",
                ent.tree,
            ),
        };
        let p = encode_proto2_prompt(tok, &[passage], q)?;
        let (gen, hit_eos) = generate_unwrapped_cpu(model, &p, EOS_ID, 48)?;
        let dec = tok.decode(&gen.iter().map(|&x| u32::from(x)).collect::<Vec<_>>());
        let reply_lower = proto.reply_text(&dec).to_ascii_lowercase();
        if strict_e4_terminal_match(&reply_lower, expected, hit_eos) {
            qa_strict += 1;
        }
        if dec.to_ascii_lowercase().contains(expected) {
            qa_compat += 1;
        }
    }

    struct RustMicro {
        prompt: &'static str,
        expected: &'static str,
    }
    let rust_items = [
        RustMicro {
            prompt: "Complete the Rust expression: `fn add(a: i32, b: i32) -> i32 { `",
            expected: "a + b",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn sub(a: i32, b: i32) -> i32 { `",
            expected: "a - b",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn mul(a: i32, b: i32) -> i32 { `",
            expected: "a * b",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn double(x: i32) -> i32 { `",
            expected: "x * 2",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn is_zero(x: i32) -> bool { `",
            expected: "x == 0",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn is_even(x: i32) -> bool { `",
            expected: "x % 2 == 0",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn neg(x: i32) -> i32 { `",
            expected: "-x",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn identity(x: u32) -> u32 { `",
            expected: "x",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn min_val(a: i32, b: i32) -> i32 { `",
            expected: "a.min(b)",
        },
        RustMicro {
            prompt: "Complete the Rust expression: `fn max_val(a: i32, b: i32) -> i32 { `",
            expected: "a.max(b)",
        },
    ];
    for i in 0..20usize {
        let item = &rust_items[i % rust_items.len()];
        let p = encode_proto2_prompt(tok, &[], item.prompt)?;
        let (gen, hit_eos) = generate_unwrapped_cpu(model, &p, EOS_ID, 48)?;
        let dec = tok.decode(&gen.iter().map(|&x| u32::from(x)).collect::<Vec<_>>());
        let reply = proto.reply_text(&dec);
        if hit_eos && reply.trim().trim_end_matches('}').trim() == item.expected {
            rust_strict += 1;
        }
        if dec.contains(item.expected) {
            rust_compat += 1;
        }
    }
    Ok(json!({
        "suite": "gate0-e4-100-aligned",
        "requires_hit_eos_and_terminal_match": true,
        "arithmetic_40": arith_strict,
        "long_distance_qa_40": qa_strict,
        "rust_micro_20": rust_strict,
        "passed": arith_strict + qa_strict + rust_strict,
        "gate0_compat_arithmetic_40": arith_compat,
        "gate0_compat_long_distance_qa_40": qa_compat,
        "gate0_compat_rust_micro_20": rust_compat,
        "gate0_compat_passed": arith_compat + qa_compat + rust_compat,
        "total": 100,
    }))
}

// ---------------------------------------------------------------------------
// Gate 2: Finite GPU-vs-CPU 0-Diff Verification (`D=32` & `D=256`, both losses)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn cmd_finite(args: &[String]) -> Res<()> {
    let out_path = PathBuf::from(flag_val(args, "--out")?);
    let mut cases = Vec::new();
    let configs = [
        (
            32usize,
            2usize,
            4usize,
            LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            101u64,
        ),
        (32usize, 2usize, 4usize, LossKind::Sparsemax64, 202u64),
        (
            256usize,
            4usize,
            16usize,
            LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            303u64,
        ),
        (256usize, 4usize, 16usize, LossKind::Sparsemax64, 404u64),
    ];

    for (dim, layers, sub_keys, loss_kind, seed) in configs {
        let cfg = ButterflyRotorConfig {
            vocab: HIER_VOCAB,
            dim,
            layers,
            pkm_sub_keys: sub_keys,
            loss_kind,
            seed,
        };
        let mut cpu_model = ButterflyRotorModel::new(cfg).map_err(|e| e.to_string())?;
        let mut cpu_lion = cpu_model.new_lion_state();
        let mut gpu = cuda_backend::gpu::CudaButterflyEngine::new(&cpu_model, &cpu_lion, 4, 32)?;

        // 4 varied-length sequences including masked-out prompt + supervised answer + BOS=0.
        let seqs: Vec<(Vec<u16>, Vec<u8>)> = vec![
            (
                vec![0, 10, 64, 11, 130, 12, 10, 64, 1],
                vec![0, 0, 0, 0, 0, 0, 0, 1, 1],
            ),
            (
                vec![0, 11, 131, 10, 65, 12, 11, 131, 1],
                vec![0, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            (
                vec![0, 287, 318, 439, 799, 1042, 1],
                vec![0, 0, 1, 0, 1, 1, 1],
            ),
            (
                vec![0, 223, 598, 772, 971, 376, 1526, 1],
                vec![0, 1, 1, 1, 0, 1, 1, 1],
            ),
        ];
        let stride = 12usize;
        let mut b_toks = vec![0u16; seqs.len() * stride];
        let mut b_mask = vec![0u8; seqs.len() * stride];
        let mut b_lens = Vec::with_capacity(seqs.len());
        for (i, (t, m)) in seqs.iter().enumerate() {
            b_toks[i * stride..i * stride + t.len()].copy_from_slice(t);
            b_mask[i * stride..i * stride + m.len()].copy_from_slice(m);
            b_lens.push(t.len() as u32);
        }

        let mut total_grad_diffs = 0usize;
        let mut total_weight_diffs = 0usize;
        let mut total_mom_diffs = 0usize;
        let mut total_loss_diffs = 0usize;
        let mut checked_grads = 0usize;
        let mut checked_weights = 0usize;

        for step_idx in 0..3u64 {
            let mut cpu_grad = cpu_model.zero_grad();
            let mut cpu_loss = 0u128;
            let mut cpu_sup = 0u64;
            for (t, m) in &seqs {
                let (l, s) = cpu_model
                    .accumulate_sequence_grad(t, m, &mut cpu_grad)
                    .map_err(|e| e.to_string())?;
                cpu_loss += l;
                cpu_sup += s;
            }
            gpu.zero_grads()?;
            let (gpu_loss, gpu_sup) = gpu.accumulate_batch(&b_toks, &b_mask, &b_lens, stride)?;
            if gpu_loss != cpu_loss || gpu_sup != cpu_sup {
                total_loss_diffs += 1;
            }
            let cpu_gflat = FlatModelBuffers::pack_grad(&cpu_model, &cpu_grad);
            let gpu_gflat = gpu.download_grads()?;
            checked_grads += cpu_gflat.len();
            total_grad_diffs += cpu_gflat
                .iter()
                .zip(&gpu_gflat)
                .filter(|&(a, b)| a != b)
                .count();

            let lr_q8 = 32_768;
            cpu_model.apply_lion_step(&cpu_grad, &mut cpu_lion, lr_q8, 12);
            gpu.apply_lion_step(step_idx + 1, lr_q8, 12)?;

            let mut gpu_model = cpu_model.clone();
            let mut gpu_lion = cpu_lion.clone();
            gpu.download_into(&mut gpu_model, &mut gpu_lion)?;
            let cpu_flat = FlatModelBuffers::pack(&cpu_model, &cpu_lion);
            let gpu_flat = FlatModelBuffers::pack(&gpu_model, &gpu_lion);
            checked_weights += cpu_flat.weights.len();
            total_weight_diffs += cpu_flat
                .weights
                .iter()
                .zip(&gpu_flat.weights)
                .filter(|&(a, b)| a != b)
                .count();
            total_mom_diffs += cpu_flat
                .momentum
                .iter()
                .zip(&gpu_flat.momentum)
                .filter(|&(a, b)| a != b)
                .count();
        }

        // Verify `bfy_eval_records_kernel` (including mid-record `BOS=0` reset parity!).
        let eval_seqs = [
            vec![0u16, 10, 64, 11, 130, 0, 287, 318, 439, 1],
            vec![0u16, 223, 598, 772, 971, 376, 1526, 1],
        ];
        let mut flat_t = Vec::new();
        let mut flat_m = Vec::new();
        let mut offs = Vec::new();
        let mut lens = Vec::new();
        for s in &eval_seqs {
            offs.push(flat_t.len() as u64);
            lens.push(s.len() as u32);
            flat_t.extend_from_slice(s);
            flat_m.extend((0..s.len()).map(|idx| u8::from(idx > 0)));
        }
        let (gpu_masses, gpu_greedy) = gpu.eval_records(&flat_t, &flat_m, &offs, &lens, true)?;
        let mut eval_mass_diffs = 0usize;
        let mut eval_tok_diffs = 0usize;
        let mut eval_checked = 0usize;
        for (r_idx, s) in eval_seqs.iter().enumerate() {
            let mut sess = cpu_model.new_session();
            let off = offs[r_idx] as usize;
            for t in 0..s.len() - 1 {
                if s[t] == 0 && t > 0 {
                    sess = cpu_model.new_session();
                }
                let out = cpu_model.step(s[t], &mut sess).map_err(|e| e.to_string())?;
                let tgt_mass = cpu_model
                    .target_q48_mass_from_hidden(&out.hidden_q15, s[t + 1])
                    .map_err(|e| e.to_string())?;
                eval_checked += 1;
                if gpu_masses[off + t + 1] != tgt_mass {
                    eval_mass_diffs += 1;
                }
                if gpu_greedy[off + t + 1] != out.greedy_token {
                    eval_tok_diffs += 1;
                }
            }
        }

        // Lossless UORBFY01 round-trip
        let bytes = cpu_model.to_bytes();
        let reloaded = ButterflyRotorModel::from_bytes(&bytes).map_err(|e| e.to_string())?;
        let reload_ok = reloaded == cpu_model;

        let ok = total_loss_diffs == 0
            && total_grad_diffs == 0
            && total_weight_diffs == 0
            && total_mom_diffs == 0
            && eval_mass_diffs == 0
            && eval_tok_diffs == 0
            && reload_ok;
        cases.push(json!({
            "dim": dim,
            "layers": layers,
            "pkm_sub_keys": sub_keys,
            "loss_kind": format!("{loss_kind:?}"),
            "checked_grad_words": checked_grads,
            "checked_weight_words": checked_weights,
            "checked_eval_targets": eval_checked,
            "loss_diffs": total_loss_diffs,
            "grad_diffs": total_grad_diffs,
            "weight_diffs": total_weight_diffs,
            "momentum_diffs": total_mom_diffs,
            "eval_mass_diffs": eval_mass_diffs,
            "eval_greedy_diffs": eval_tok_diffs,
            "uorbfy01_reload_ok": reload_ok,
            "pass": ok,
        }));
        if !ok {
            let report = json!({"pass": false, "cases": cases});
            fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap()).ok();
            return Err(format!(
                "Gate 2 finite GPU-vs-CPU parity FAILED on dim={dim} {loss_kind:?}"
            ));
        }
    }

    let report = json!({
        "schema": "uor-r4.gate1-finite-parity/1",
        "pass": true,
        "cases": cases,
    });
    fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "Gate 2 finite parity PASS ({} configs, 0 diffs)",
        cases.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU Bar A (`chat-v0-p2`) and E1 (`DEV-tune` + prompt-shuffle) Evaluation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn eval_bar_a_gpu(
    gpu: &cuda_backend::gpu::CudaButterflyEngine,
    cpu_model: &ButterflyRotorModel,
    heldout_tokens: &[u16],
    tok_byte_lens: &[u64],
) -> Res<Value> {
    let mut flat_t = Vec::with_capacity(BAR_A_WINDOWS * (BAR_A_CONTEXT + 1));
    let mut flat_m = Vec::with_capacity(BAR_A_WINDOWS * (BAR_A_CONTEXT + 1));
    let mut offs = Vec::with_capacity(BAR_A_WINDOWS);
    let mut lens = Vec::with_capacity(BAR_A_WINDOWS);
    for w in 0..BAR_A_WINDOWS {
        let start = w * BAR_A_STRIDE;
        offs.push(flat_t.len() as u64);
        lens.push((BAR_A_CONTEXT + 1) as u32);
        flat_t.extend_from_slice(&heldout_tokens[start..start + BAR_A_CONTEXT + 1]);
        flat_m.push(0);
        flat_m.extend(std::iter::repeat_n(1u8, BAR_A_CONTEXT));
    }
    let (masses, greedy) = gpu.eval_records(&flat_t, &flat_m, &offs, &lens, true)?;

    // Exact PRLL bits_q12 from Q48 masses (summing to 2^48, >= 1 everywhere!).
    let mut total_bits = 0u128;
    let mut non_bos_bits = 0u128;
    let mut non_bos_targets = 0u64;
    let mut sample_bytes = 0u64;
    let mut top1_hits = 0u64;
    let mut win_mbpbs = Vec::with_capacity(BAR_A_WINDOWS);

    for w in 0..BAR_A_WINDOWS {
        let off = offs[w] as usize;
        let mut w_bits = 0u128;
        for t in 0..BAR_A_CONTEXT {
            let tgt = flat_t[off + t + 1];
            let m = masses[off + t + 1];
            let bq = u128::from(bits_q12(m).map_err(|e| format!("bits_q12: {e}"))?);
            w_bits += bq;
            total_bits += bq;
            sample_bytes += tok_byte_lens[usize::from(tgt)];
            if tgt != 0 {
                non_bos_bits += bq;
                non_bos_targets += 1;
            }
            if greedy[off + t + 1] == tgt {
                top1_hits += 1;
            }
        }
        win_mbpbs.push(bar_a_stream_micro_bpb(w_bits, BAR_A_CONTEXT as u64));
    }
    let total_targets = (BAR_A_WINDOWS * BAR_A_CONTEXT) as u64;
    let stream_mbpb = bar_a_stream_micro_bpb(total_bits, total_targets);
    let non_bos_mbpb = bar_a_stream_micro_bpb(non_bos_bits, non_bos_targets);
    let sample_mbpb = micro_bpb_from_q12(total_bits, sample_bytes);
    let se_emp = empirical_se_micro_bpb(&win_mbpbs);
    let hw = (196u64 * se_emp + 50) / 100;

    // Spot-check 16 windows × full 384 targets? Spot-check 32 stratified windows (12,288 targets) on CPU!
    let mut spot_checked = 0usize;
    let mut spot_diffs = 0usize;
    for w in (0..BAR_A_WINDOWS).step_by(64) {
        let off = offs[w] as usize;
        let mut sess = cpu_model.new_session();
        for t in 0..BAR_A_CONTEXT {
            let inp = flat_t[off + t];
            let tgt = flat_t[off + t + 1];
            if inp == 0 && t > 0 {
                sess = cpu_model.new_session();
            }
            let out = cpu_model.step(inp, &mut sess).map_err(|e| e.to_string())?;
            if t % 12 == 0 {
                let c_mass = cpu_model
                    .target_q48_mass_from_hidden(&out.hidden_q15, tgt)
                    .map_err(|e| e.to_string())?;
                spot_checked += 1;
                if c_mass != masses[off + t + 1] || out.greedy_token != greedy[off + t + 1] {
                    spot_diffs += 1;
                }
            }
        }
    }
    if spot_diffs != 0 {
        return Err(format!(
            "Bar A GPU-vs-CPU spot check found {spot_diffs}/{spot_checked} diffs!"
        ));
    }

    Ok(json!({
        "windows": BAR_A_WINDOWS,
        "targets": total_targets,
        "total_bits_q12": total_bits.to_string(),
        "stream_basis_micro_bpb": stream_mbpb,
        "stream_basis_ci95_lo": stream_mbpb.saturating_sub(hw),
        "stream_basis_ci95_hi": stream_mbpb.saturating_add(hw),
        "non_bos_stream_micro_bpb": non_bos_mbpb,
        "sample_basis_micro_bpb": sample_mbpb,
        "top1_permille": (top1_hits * 1000) / total_targets.max(1),
        "cpu_spot_checked": spot_checked,
        "cpu_spot_diffs": spot_diffs,
    }))
}

#[cfg(feature = "cuda")]
fn eval_e1_devtune_gpu(
    gpu: &cuda_backend::gpu::CudaButterflyEngine,
    recs: &[DevRec],
    id_bytes: &[u8],
    eos_id: u16,
    max_ans_per_rec: usize,
) -> Res<Value> {
    // Exact Gate 0 `eval.rs` parity: intact `\nAssistant:`-terminated true AND donor prompts,
    // stopping at EOS, with byte-weighted paired bootstrap CI95 on `ΔBPB_prompt`.
    struct AnsRange {
        ans_s: usize,
        ans_e: usize,
    }
    let n = recs.len();
    let mut flat_t = Vec::new();
    let mut flat_m = Vec::new();
    let mut offs = Vec::new();
    let mut lens = Vec::new();
    let mut ans_ranges = Vec::with_capacity(n);

    for r in recs {
        let ans_s = r.first_ans_start;
        let ans_e = r
            .first_ans_end
            .min(ans_s + max_ans_per_rec)
            .min(r.tokens.len());
        offs.push(flat_t.len() as u64);
        lens.push(ans_e as u32);
        flat_t.extend_from_slice(&r.tokens[..ans_e]);
        for p in 0..ans_e {
            flat_m.push(u8::from(p >= ans_s));
        }
        ans_ranges.push(AnsRange { ans_s, ans_e });
    }
    let (true_masses, true_greedy) = gpu.eval_records(&flat_t, &flat_m, &offs, &lens, false)?;

    // Swapped prompts: FULL intact prompt from `(i + n/2) % n` ending at `\nAssistant:` + answer `i`.
    let mut sw_t = Vec::new();
    let mut sw_m = Vec::new();
    let mut sw_offs = Vec::new();
    let mut sw_lens = Vec::new();
    let mut sw_ans_start = Vec::with_capacity(n);
    for i in 0..n {
        let donor = &recs[(i + n / 2) % n];
        let r = &recs[i];
        let ar = &ans_ranges[i];
        let p_end = donor.first_ans_start;
        let seq_len = p_end + (ar.ans_e - ar.ans_s);
        sw_offs.push(sw_t.len() as u64);
        sw_lens.push(seq_len as u32);
        sw_ans_start.push(p_end);
        sw_t.extend_from_slice(&donor.tokens[..p_end]);
        sw_t.extend_from_slice(&r.tokens[ar.ans_s..ar.ans_e]);
        for pos in 0..seq_len {
            sw_m.push(u8::from(pos >= p_end));
        }
    }
    let (sw_masses, _) = gpu.eval_records(&sw_t, &sw_m, &sw_offs, &sw_lens, false)?;

    let mut all_bits = 0u128;
    let mut all_bytes = 0u64;
    let mut all_tgts = 0u64;
    let mut all_top1 = 0u64;
    let mut nov_bits = 0u128;
    let mut nov_bytes = 0u64;
    let mut nov_tgts = 0u64;
    let mut nov_top1 = 0u64;
    let mut inc_bits = 0u128;
    let mut inc_bytes = 0u64;
    let mut inc_tgts = 0u64;
    let mut eos_tgts = 0u64;
    let mut eos_top1 = 0u64;
    let mut sw_all_bits = 0u128;
    let mut shuffle_pairs = Vec::with_capacity(n);

    for i in 0..n {
        let r = &recs[i];
        let off = offs[i] as usize;
        let ar = &ans_ranges[i];
        let sw_off = sw_offs[i] as usize;
        let sw_s = sw_ans_start[i];
        let mut r_true_bits = 0u128;
        let mut r_sw_bits = 0u128;
        let mut r_bytes = 0u128;
        for pos in ar.ans_s..ar.ans_e {
            let tgt = r.tokens[pos];
            let b_len = u64::from(id_bytes[usize::from(tgt)]);
            let tb = u128::from(bits_q12(true_masses[off + pos]).map_err(|e| e.to_string())?);
            let sb = u128::from(
                bits_q12(sw_masses[sw_off + sw_s + (pos - ar.ans_s)]).map_err(|e| e.to_string())?,
            );
            r_true_bits += tb;
            r_sw_bits += sb;
            r_bytes += u128::from(b_len);
            all_bits += tb;
            sw_all_bits += sb;
            all_bytes += b_len;
            all_tgts += 1;
            let hit = u64::from(true_greedy[off + pos] == tgt);
            all_top1 += hit;
            if tgt == eos_id {
                eos_tgts += 1;
                eos_top1 += hit;
            }
            if r.novel[pos] == 1 && r.absent[pos] == 1 {
                nov_bits += tb;
                nov_bytes += b_len;
                nov_tgts += 1;
                nov_top1 += hit;
            } else if r.absent[pos] == 0 {
                inc_bits += tb;
                inc_bytes += b_len;
                inc_tgts += 1;
            }
            if tgt == eos_id {
                break;
            }
        }
        if r_bytes > 0 {
            shuffle_pairs.push(ShufflePair {
                true_bits: r_true_bits,
                swap_bits: r_sw_bits,
                bytes: r_bytes,
            });
        }
    }

    let true_mbpb = micro_bpb_from_q12(all_bits, all_bytes);
    let sw_mbpb = micro_bpb_from_q12(sw_all_bits, all_bytes);
    let delta_mbpb = signed_micro_bpb_diff(
        sw_all_bits,
        u128::from(all_bytes),
        all_bits,
        u128::from(all_bytes),
    );
    let (ci_lo, ci_hi) = paired_bootstrap_ci95_weighted(&shuffle_pairs, 0xD870_E101_2026_1127);

    Ok(json!({
        "records": n,
        "all_answer": {
            "targets": all_tgts,
            "bytes": all_bytes,
            "bits_q12": all_bits.to_string(),
            "micro_bpb": true_mbpb,
            "top1_permille": (all_top1 * 1000) / all_tgts.max(1),
        },
        "novel_absent_answer": {
            "targets": nov_tgts,
            "bytes": nov_bytes,
            "bits_q12": nov_bits.to_string(),
            "micro_bpb": micro_bpb_from_q12(nov_bits, nov_bytes),
            "top1_permille": (nov_top1 * 1000) / nov_tgts.max(1),
        },
        "in_context_answer": {
            "targets": inc_tgts,
            "bytes": inc_bytes,
            "micro_bpb": micro_bpb_from_q12(inc_bits, inc_bytes),
        },
        "gold_eos": {
            "targets": eos_tgts,
            "top1": eos_top1,
            "top1_permille": (eos_top1 * 1000) / eos_tgts.max(1),
        },
        "counterfactual_prompt_shuffle": {
            "true_prompt_micro_bpb": true_mbpb,
            "swapped_prompt_micro_bpb": sw_mbpb,
            "delta_micro_bpb": delta_mbpb,
            "ci95_lo_micro_bpb": ci_lo,
            "ci95_hi_micro_bpb": ci_hi,
            "prompt_conditioned_pass": ci_lo > 0,
        },
    }))
}

fn synthetic_bench_prompt(ctx_len: usize) -> Vec<u16> {
    (0..ctx_len)
        .map(|i| {
            if i == 0 {
                0
            } else {
                (1 + (splitmix64(i as u64 + 17) % 4095)) as u16
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gate 3: Training Arm Driver on `train-100m`
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn cmd_train(args: &[String]) -> Res<()> {
    let corpus_dir = PathBuf::from(flag_val(args, "--corpus-dir")?);
    let heldout_path = PathBuf::from(flag_val(args, "--heldout")?);
    let tok_path = PathBuf::from(flag_val(args, "--tokenizer")?);
    let m2_dir = PathBuf::from(flag_val(args, "--m2-dir")?);
    let out_ckpt = PathBuf::from(flag_val(args, "--out-ckpt")?);
    let out_report = PathBuf::from(flag_val(args, "--out-report")?);
    let out_replay = opt_flag(args, "--out-replay").map(PathBuf::from);
    let dim: usize = opt_flag(args, "--dim")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let layers: usize = opt_flag(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let sub_keys: usize = opt_flag(args, "--pkm-sub-keys")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let loss_name = opt_flag(args, "--loss").unwrap_or_else(|| "hui-belkin".to_owned());
    let loss_k: i32 = opt_flag(args, "--loss-k")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let epochs: usize = opt_flag(args, "--epochs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let batch_size: usize = opt_flag(args, "--batch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let seq_len: usize = opt_flag(args, "--seq-len")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let lr_q8_init: i32 = opt_flag(args, "--lr-q8")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_048);
    let wd_shift: u32 = opt_flag(args, "--wd-shift")
        .and_then(|s| s.parse().ok())
        .unwrap_or(17);
    let seed: u64 = opt_flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20261127);
    let max_steps_per_epoch: usize = opt_flag(args, "--max-steps-per-epoch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let tok_bytes = read_and_pin(&tok_path, Some(PIN_TOKENIZER_SHA256), "tokenizer.json")?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
        .ok_or_else(|| "failed to load tokenizer.json".to_owned())?;
    // Fail-closed preflight verification of `m2-fresh256` BEFORE starting GPU compute!
    let m2_cases = load_fresh256_cases(&m2_dir, &tok)?;
    let mut tok_byte_lens = vec![0u64; HIER_VOCAB];
    for id in 0..HIER_VOCAB {
        tok_byte_lens[id] = tok.decode(&[id as u32]).len() as u64;
    }

    let heldout_bytes = read_and_pin(
        &heldout_path,
        Some(PIN_HELDOUT_SHA256),
        "chat-v0-p2 heldout.u16",
    )?;
    let heldout_tokens = parse_uort_u16(&heldout_bytes)?;
    let id_bytes = read_and_pin(
        &corpus_dir.join("id-bytes.u8"),
        Some(PIN_ID_BYTES_SHA256),
        "id-bytes.u8",
    )?;
    let dev_recs = load_proto2_dev_records(&corpus_dir, "dev-tune", 256)?;
    let train_toks = parse_uort_u16(&read_and_pin(
        &corpus_dir.join("train-100m.tokens.u16"),
        Some(PIN_TRAIN100M_TOK_SHA256),
        "train-100m.tokens.u16",
    )?)?;
    let train_mask = read_and_pin(
        &corpus_dir.join("train-100m.mask.u8"),
        Some(PIN_TRAIN100M_MASK_SHA256),
        "train-100m.mask.u8",
    )?;
    let chunks_all =
        build_training_chunks(train_toks.clone(), train_mask.clone(), seq_len + 1, true)?;
    let chunks_ans = build_training_chunks(train_toks, train_mask, seq_len + 1, false)?;

    let loss_kind = parse_loss_kind(&loss_name, loss_k)?;
    let cfg = ButterflyRotorConfig {
        vocab: HIER_VOCAB,
        dim,
        layers,
        pkm_sub_keys: sub_keys,
        loss_kind,
        seed,
    };
    let mut model = ButterflyRotorModel::new(cfg).map_err(|e| e.to_string())?;
    let mut lion = model.new_lion_state();
    let mut gpu = cuda_backend::gpu::CudaButterflyEngine::new(&model, &lion, batch_size, seq_len)?;

    let eos_id = EOS_ID;
    let t_start = Instant::now();
    let mut epoch_logs = Vec::new();
    let mut best_dev_mbpb = u64::MAX;
    let mut best_epoch = 0usize;
    let mut best_model = model.clone();

    // Epoch 0 baseline eval
    let e0_bar_a = eval_bar_a_gpu(&gpu, &model, &heldout_tokens, &tok_byte_lens)?;
    let e0_e1 = eval_e1_devtune_gpu(&gpu, &dev_recs, &id_bytes, eos_id, 256)?;
    println!(
        "[epoch 0] chat-v0-p2={} E1_all={} E1_nov={}",
        e0_bar_a["stream_basis_micro_bpb"],
        e0_e1["all_answer"]["micro_bpb"],
        e0_e1["novel_absent_answer"]["micro_bpb"]
    );
    epoch_logs.push(json!({
        "epoch": 0,
        "phase": "init",
        "steps": 0,
        "bar_a": e0_bar_a,
        "e1_devtune": e0_e1,
    }));

    let mut global_step = 0u64;
    let stride = seq_len + 1;
    let mut b_toks = vec![0u16; batch_size * stride];
    let mut b_mask = vec![0u8; batch_size * stride];
    let mut b_lens = vec![0u32; batch_size];

    for ep in 1..=epochs {
        let ep_t0 = Instant::now();
        // Epoch 1: full causal LM (`all_non_bos`); Epochs 2..: Protocol-2 answer-only SFT (`mask == 1`).
        let chunks = if ep == 1 { &chunks_all } else { &chunks_ans };
        let phase = if ep == 1 {
            "pretrain_all_non_bos"
        } else {
            "sft_answer_only"
        };
        let n_chunks = chunks.starts.len();
        // Deterministic Fisher-Yates permutation per epoch using `splitmix64`.
        let mut order: Vec<usize> = (0..n_chunks).collect();
        let mut rng = splitmix64(seed ^ (ep as u64).wrapping_mul(0x517C_C1B7_2722_0A95));
        for i in (1..n_chunks).rev() {
            rng = splitmix64(rng);
            let j = (rng as usize) % (i + 1);
            order.swap(i, j);
        }

        let total_batches = n_chunks.div_ceil(batch_size).min(max_steps_per_epoch);
        let mut ep_loss_sum = 0u128;
        let mut ep_sup_sum = 0u64;
        let mut ep_tok_sum = 0u64;
        let mut win_loss_sum = 0u128;
        let mut win_sup_sum = 0u64;

        for b_idx in 0..total_batches {
            let s0 = b_idx * batch_size;
            let s1 = (s0 + batch_size).min(n_chunks);
            let cur_bs = s1 - s0;
            for bi in 0..cur_bs {
                let c_id = order[s0 + bi];
                let st = chunks.starts[c_id];
                let ln = chunks.lens[c_id];
                b_toks[bi * stride..bi * stride + ln].copy_from_slice(&chunks.tokens[st..st + ln]);
                b_mask[bi * stride..bi * stride + ln].copy_from_slice(&chunks.mask[st..st + ln]);
                b_lens[bi] = ln as u32;
                ep_tok_sum += (ln - 1) as u64;
            }
            gpu.zero_grads()?;
            let (bloss, bsup) = gpu.accumulate_batch(
                &b_toks[..cur_bs * stride],
                &b_mask[..cur_bs * stride],
                &b_lens[..cur_bs],
                stride,
            )?;
            ep_loss_sum += bloss;
            ep_sup_sum += bsup;
            win_loss_sum += bloss;
            win_sup_sum += bsup;

            global_step += 1;
            // Smooth pure-integer sign-Lion schedule: 200-step linear warmup to `lr_q8_init`,
            // then smooth per-batch linear decay within each epoch:
            // Ep 1: `lr -> lr/2`, Ep 2: `lr/2 -> lr/4`, Ep 3+: `lr/4 -> lr/16` (min 128 = 1/2 LSB).
            let (ep_hi, ep_lo) = if ep == 1 {
                (lr_q8_init, (lr_q8_init >> 1).max(128))
            } else if ep == 2 {
                ((lr_q8_init >> 1).max(128), (lr_q8_init >> 2).max(128))
            } else {
                ((lr_q8_init >> 2).max(128), (lr_q8_init >> 4).max(128))
            };
            let span = i64::from(ep_hi - ep_lo);
            let denom = (total_batches.max(1)) as i64;
            let decayed = (i64::from(ep_hi) - (span * (b_idx as i64)) / denom).max(128) as i32;
            let lr_now = if global_step <= 200 {
                ((i64::from(lr_q8_init) * (global_step as i64)) / 200).max(32) as i32
            } else {
                decayed
            };
            gpu.apply_lion_step(global_step, lr_now, wd_shift)?;

            if (b_idx + 1) % 250 == 0 || b_idx + 1 == total_batches {
                let elapsed_ms = ep_t0.elapsed().as_millis().max(1);
                let tok_s = (u128::from(ep_tok_sum) * 1000) / elapsed_ms;
                let mean_loss = ep_loss_sum / u128::from(ep_sup_sum.max(1));
                let win_loss = win_loss_sum / u128::from(win_sup_sum.max(1));
                println!(
                    "  [ep {ep}/{epochs} batch {}/{total_batches} step {global_step}] win_loss_q15={win_loss} ep_loss_q15={mean_loss} tok/s={tok_s} lr_q8={lr_now}",
                    b_idx + 1
                );
                win_loss_sum = 0;
                win_sup_sum = 0;
            }
        }

        gpu.download_into(&mut model, &mut lion)?;
        model.step = global_step;
        let ep_secs = ep_t0.elapsed().as_secs();
        let bar_a = eval_bar_a_gpu(&gpu, &model, &heldout_tokens, &tok_byte_lens)?;
        let e1 = eval_e1_devtune_gpu(&gpu, &dev_recs, &id_bytes, eos_id, 256)?;
        let dev_mbpb = e1["all_answer"]["micro_bpb"].as_u64().unwrap_or(u64::MAX);
        if dev_mbpb < best_dev_mbpb {
            best_dev_mbpb = dev_mbpb;
            best_epoch = ep;
            best_model = model.clone();
        }
        println!(
            "[epoch {ep} DONE in {ep_secs}s] chat-v0-p2={} E1_all={} E1_nov={} shuf_delta={}",
            bar_a["stream_basis_micro_bpb"],
            e1["all_answer"]["micro_bpb"],
            e1["novel_absent_answer"]["micro_bpb"],
            e1["counterfactual_prompt_shuffle"]["delta_micro_bpb"]
        );
        epoch_logs.push(json!({
            "epoch": ep,
            "phase": phase,
            "steps_total": global_step,
            "batches_in_epoch": total_batches,
            "tokens_seen": ep_tok_sum,
            "supervised_targets": ep_sup_sum,
            "mean_loss_q15": (ep_loss_sum / u128::from(ep_sup_sum.max(1))) as u64,
            "wall_secs": ep_secs,
            "bar_a": bar_a,
            "e1_devtune": e1,
        }));
    }

    // Save lossless `UORBFY01` best checkpoint and verify cold reload!
    let ckpt_bytes = best_model.to_bytes();
    let ckpt_sha = sha256_hex(&ckpt_bytes);
    fs::write(&out_ckpt, &ckpt_bytes).map_err(|e| e.to_string())?;
    let cold_model =
        ButterflyRotorModel::from_bytes(&fs::read(&out_ckpt).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if cold_model != best_model {
        return Err("cold UORBFY01 reload mismatch!".to_owned());
    }

    // Emit GPU-produced CUDA=M1 bit-identical replay fixture on the exact `bench-cpu` 2,048-token prompt!
    if let Some(ref replay_path) = out_replay {
        let prompt = synthetic_bench_prompt(2048);
        let mut sess = cold_model.new_session();
        let mut last = 0u16;
        for &tok_v in &prompt {
            last = cold_model
                .step(tok_v, &mut sess)
                .map_err(|e| e.to_string())?
                .greedy_token;
        }
        let mut seq = prompt;
        for _ in 0..128 {
            seq.push(last);
            last = cold_model
                .step(last, &mut sess)
                .map_err(|e| e.to_string())?
                .greedy_token;
        }
        let mut eval_mask = vec![0u8; seq.len()];
        for m in eval_mask.iter_mut().skip(2048) {
            *m = 1;
        }
        let cold_lion = cold_model.new_lion_state();
        let cold_gpu = cuda_backend::gpu::CudaButterflyEngine::new(&cold_model, &cold_lion, 1, 1)?;
        let (gpu_masses, gpu_greedy) =
            cold_gpu.eval_records(&seq, &eval_mask, &[0u64], &[seq.len() as u32], false)?;
        let gpu_gen_tokens: Vec<u16> = gpu_greedy[2048..2048 + 128].to_vec();
        let gpu_gen_masses: Vec<u64> = gpu_masses[2048..2048 + 128].to_vec();
        if gpu_gen_tokens != seq[2048..2048 + 128] {
            return Err("CUDA=CPU replay fixture mismatch on cold UORBFY01!".to_owned());
        }
        let fixture = json!({
            "schema": "uor-r4.gate1-replay-fixture/1",
            "artifact_sha256": ckpt_sha,
            "context_len": 2048,
            "gen_len": 128,
            "gpu_greedy_tokens": gpu_gen_tokens,
            "gpu_q48_masses": gpu_gen_masses,
        });
        fs::write(replay_path, serde_json::to_string_pretty(&fixture).unwrap())
            .map_err(|e| e.to_string())?;
    }

    // Run strict unwrapped E2, E3, and E4 on cold-reloaded best model!
    let e2 = eval_e2_freerun_cpu(&cold_model, &tok, &dev_recs, 200, 128)?;
    let e3 = eval_e3_m2_unwrapped_cpu(&cold_model, &tok, &m2_cases)?;
    let e4 = eval_e4_checkable_cpu(&cold_model, &tok)?;

    let report = json!({
        "schema": "uor-r4.gate1-train-report/1",
        "config": {
            "dim": dim,
            "layers": layers,
            "pkm_sub_keys": sub_keys,
            "loss_kind": format!("{loss_kind:?}"),
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lr_q8_init": lr_q8_init,
            "wd_shift": wd_shift,
            "seed": seed,
            "param_count": best_model.param_count(),
            "artifact_bytes": ckpt_bytes.len(),
            "artifact_sha256": ckpt_sha,
        },
        "total_wall_secs": t_start.elapsed().as_secs(),
        "best_epoch": best_epoch,
        "best_dev_e1_all_micro_bpb": best_dev_mbpb,
        "epochs": epoch_logs,
        "unwrapped_eval_best_cold": {
            "e2_freerun": e2,
            "e3_m2_unwrapped": e3,
            "e4_checkable": e4,
        },
    });
    fs::write(&out_report, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "Saved {} ({} B, sha256={}) & {}",
        out_ckpt.display(),
        ckpt_bytes.len(),
        &ckpt_sha[..12],
        out_report.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Apple M1 / Host Native Cold Open, 2048-Context Warm tok/s & CUDA=M1 Replay Bench
// ---------------------------------------------------------------------------

fn cmd_bench_cpu(args: &[String]) -> Res<()> {
    let ckpt_path = PathBuf::from(flag_val(args, "--ckpt")?);
    let out_path = PathBuf::from(flag_val(args, "--out")?);
    let replay_fixture_path = opt_flag(args, "--replay-fixture").map(PathBuf::from);
    let ctx_len: usize = opt_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let gen_len: usize = opt_flag(args, "--gen")
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let runs: usize = opt_flag(args, "--runs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let t_open = Instant::now();
    let raw = fs::read(&ckpt_path).map_err(|e| e.to_string())?;
    let sha = sha256_hex(&raw);
    let model = ButterflyRotorModel::from_bytes(&raw).map_err(|e| e.to_string())?;
    let open_ms = t_open.elapsed().as_millis() as u64;

    let prompt = synthetic_bench_prompt(ctx_len);

    let mut prefill_ms = Vec::with_capacity(runs);
    let mut decode_us = Vec::with_capacity(runs);
    let mut first_gen = Vec::new();
    let mut first_masses = Vec::new();
    let mut replay_diffs = 0usize;

    for r in 0..runs {
        let mut sess = model.new_session();
        let tp = Instant::now();
        let mut last = 0u16;
        let mut last_hidden = vec![0i32; model.cfg.dim];
        for &tok in &prompt {
            let st = model.step(tok, &mut sess).map_err(|e| e.to_string())?;
            last = st.greedy_token;
            if r == 0 {
                last_hidden = st.hidden_q15;
            }
        }
        prefill_ms.push(tp.elapsed().as_millis() as u64);

        let td = Instant::now();
        let mut gen = Vec::with_capacity(gen_len);
        let mut masses = if r == 0 {
            Vec::with_capacity(gen_len)
        } else {
            Vec::new()
        };
        for _ in 0..gen_len {
            if r == 0 {
                let m = model
                    .target_q48_mass_from_hidden(&last_hidden, last)
                    .map_err(|e| e.to_string())?;
                masses.push(m);
            }
            gen.push(last);
            let st = model.step(last, &mut sess).map_err(|e| e.to_string())?;
            last = st.greedy_token;
            if r == 0 {
                last_hidden = st.hidden_q15;
            }
        }
        decode_us.push(td.elapsed().as_micros().max(1) as u64);
        if r == 0 {
            first_gen = gen;
            first_masses = masses;
        } else if gen != first_gen {
            replay_diffs += 1;
        }
    }

    // Optional fail-closed CUDA=M1 bit-identical replay check against GPU-emitted fixture.
    let mut cuda_replay_checked = false;
    let mut cuda_replay_diffs = 0usize;
    if let Some(ref fpath) = replay_fixture_path {
        let fraw = fs::read(fpath).map_err(|e| format!("read replay fixture: {e}"))?;
        let fval: Value = serde_json::from_slice(&fraw).map_err(|e| e.to_string())?;
        let fsha = fval["artifact_sha256"]
            .as_str()
            .ok_or("missing artifact_sha256")?;
        if fsha != sha {
            return Err(format!(
                "replay fixture artifact_sha256 mismatch: {fsha} != {sha}"
            ));
        }
        let fctx = fval["context_len"].as_u64().ok_or("missing context_len")? as usize;
        if fctx != ctx_len {
            return Err(format!(
                "replay fixture context_len mismatch: {fctx} != {ctx_len}"
            ));
        }
        let exp_toks: Vec<u16> = fval["gpu_greedy_tokens"]
            .as_array()
            .ok_or("missing gpu_greedy_tokens")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u16)
            .collect();
        let exp_masses: Vec<u64> = fval["gpu_q48_masses"]
            .as_array()
            .ok_or("missing gpu_q48_masses")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0))
            .collect();
        if exp_toks.len() != first_gen.len() || exp_masses.len() != first_masses.len() {
            return Err("replay fixture length mismatch against bench-cpu gen_len".to_owned());
        }
        for (a, b) in first_gen.iter().zip(&exp_toks) {
            if a != b {
                cuda_replay_diffs += 1;
            }
        }
        for (a, b) in first_masses.iter().zip(&exp_masses) {
            if a != b {
                cuda_replay_diffs += 1;
            }
        }
        cuda_replay_checked = true;
        if cuda_replay_diffs != 0 {
            return Err(format!(
                "FAIL-CLOSED CUDA=CPU replay mismatch: {cuda_replay_diffs} diffs"
            ));
        }
    }

    prefill_ms.sort_unstable();
    decode_us.sort_unstable();
    let med_prefill_ms = prefill_ms[runs / 2];
    let med_decode_us = decode_us[runs / 2];
    let med_tok_per_sec = (u128::from(gen_len as u64) * 1_000_000u128) / u128::from(med_decode_us);

    let report = json!({
        "schema": "uor-r4.gate1-cpu-bench/1",
        "artifact_bytes": raw.len(),
        "artifact_sha256": sha,
        "dim": model.cfg.dim,
        "layers": model.cfg.layers,
        "pkm_sub_keys": model.cfg.pkm_sub_keys,
        "cold_open_ms": open_ms,
        "context_len": ctx_len,
        "gen_len": gen_len,
        "runs": runs,
        "median_prefill_ms": med_prefill_ms,
        "median_decode_us": med_decode_us,
        "median_decode_tok_per_sec": med_tok_per_sec as u64,
        "determinism_replay_diffs": replay_diffs,
        "cuda_vs_cpu_replay_checked": cuda_replay_checked,
        "cuda_vs_cpu_replay_diffs": cuda_replay_diffs,
    });
    fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "CPU bench: open={}ms prefill({})={}ms decode({})={} tok/s replay_diffs={} cuda_replay_diffs={}",
        open_ms, ctx_len, med_prefill_ms, gen_len, med_tok_per_sec, replay_diffs, cuda_replay_diffs
    );
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return Err("usage: butterfly-pilot <finite|train|bench-cpu> ...".to_owned());
    }
    match args[1].as_str() {
        "bench-cpu" => cmd_bench_cpu(&args[2..]),
        #[cfg(feature = "cuda")]
        "finite" => cmd_finite(&args[2..]),
        #[cfg(feature = "cuda")]
        "train" => cmd_train(&args[2..]),
        #[cfg(not(feature = "cuda"))]
        "finite" | "train" => Err("rebuild with --features cuda to run GPU commands".to_owned()),
        other => Err(format!("unknown subcommand `{other}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_r4_native_vector::butterfly_rotor::{ButterflyRotorConfig, LossKind, HIER_VOCAB};

    #[test]
    fn flat_pack_unpack_and_slice_cursors_match_model_exactly() {
        let cfg = ButterflyRotorConfig {
            vocab: HIER_VOCAB,
            dim: 32,
            layers: 2,
            pkm_sub_keys: 4,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            seed: 42,
        };
        let model = ButterflyRotorModel::new(cfg).unwrap();
        let lion = model.new_lion_state();
        let flat = FlatModelBuffers::pack(&model, &lion);
        assert_eq!(flat.weights.len(), model.param_count());
        assert_eq!(flat.slices.offsets.len(), 3 + 21 * cfg.layers);
        let mut model2 =
            ButterflyRotorModel::new(ButterflyRotorConfig { seed: 99, ..cfg }).unwrap();
        model2.cfg.seed = cfg.seed;
        let mut lion2 = model2.new_lion_state();
        flat.unpack_into(&mut model2, &mut lion2);
        // Copy frozen PKM keys before comparing full model equality.
        for (l2, l1) in model2.layers.iter_mut().zip(&model.layers) {
            l2.pkm.key_a.copy_from_slice(&l1.pkm.key_a);
            l2.pkm.key_b.copy_from_slice(&l1.pkm.key_b);
        }
        assert_eq!(model2, model);
        let grad = model.zero_grad();
        let gflat = FlatModelBuffers::pack_grad(&model, &grad);
        assert_eq!(gflat.len(), model.param_count());
    }

    #[test]
    fn strict_e4_grader_rejects_missing_eos_and_prompt_echo() {
        assert!(!strict_e4_terminal_match("19", "19", false));
        assert!(strict_e4_terminal_match("19", "19", true));
        assert!(strict_e4_terminal_match("The answer is 19.", "19", true));
        assert!(!strict_e4_terminal_match("Compute 12 + 7 = 99", "12", true));
        assert!(!strict_e4_terminal_match("19 and then 20", "19", true));
    }

    #[test]
    fn bar_a_stream_basis_matches_p4_uniform_pin() {
        // Uniform floor: 49,152 Q12/tok over 196,608 targets -> 4,229,183 micro-BPB
        assert_eq!(
            bar_a_stream_micro_bpb(49_152u128 * 196_608u128, 196_608),
            4_229_183
        );
    }

    #[test]
    fn gate0_aligned_loop_and_weighted_bootstrap_and_chunk_validation() {
        // Fewer than 3 repeated 4-grams -> not a loop; >= 3 repeated 4-grams -> loop.
        assert!(!has_fourgram_loop(&[1, 2, 3, 4, 5, 1, 2, 3, 4]));
        assert!(has_fourgram_loop(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]));
        // Byte-weighted paired bootstrap returns strictly positive CI when swap > true on all pairs.
        let pairs = vec![
            ShufflePair {
                true_bits: 4096 * 10,
                swap_bits: 4096 * 14,
                bytes: 10,
            },
            ShufflePair {
                true_bits: 4096 * 20,
                swap_bits: 4096 * 30,
                bytes: 20,
            },
            ShufflePair {
                true_bits: 4096 * 15,
                swap_bits: 4096 * 21,
                bytes: 15,
            },
        ];
        let (lo, hi) = paired_bootstrap_ci95_weighted(&pairs, 77);
        assert!(lo >= 400_000 && hi <= 500_000, "unexpected CI [{lo}, {hi}]");
        // Length mismatch rejected fail-closed.
        assert!(build_training_chunks(vec![0, 5, 6], vec![0, 1], 16, false).is_err());
    }
}
