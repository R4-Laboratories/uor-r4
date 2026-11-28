#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use uor_r4_native_vector::butterfly_rotor::{clamp_act, HIER_FINE};
use uor_r4_native_vector::butterfly_rotor_v3::{
    ButterflyV3Model, ButterflyV3Session, ButterflyV3StepOutput,
};

const MAX_EMITTED: usize = 32;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Row {
    prefix: Vec<u16>,
    expected: Vec<u16>,
}

#[derive(Deserialize)]
struct Pair {
    pair_id: u64,
    a: Row,
    b: Row,
}

fn is_admissible(model: &ButterflyV3Model, token: u16) -> bool {
    usize::from(token) < model.cfg.vocab && (model.cfg.vocab == 1 || token != 0)
}

fn validate(model: &ButterflyV3Model, pair: &Pair, eos: u16) -> Result<()> {
    for row in [&pair.a, &pair.b] {
        if row.prefix.len() < 2
            || row
                .prefix
                .iter()
                .any(|&t| usize::from(t) >= model.cfg.vocab)
            || row.expected.last() != Some(&eos)
            || row.expected[..row.expected.len() - 1].contains(&eos)
            || row.expected.iter().any(|&t| !is_admissible(model, t))
        {
            return Err(format!(
                "pair {} has an invalid prefix or expected suffix",
                pair.pair_id
            )
            .into());
        }
    }
    if pair.a.prefix.last() != pair.b.prefix.last()
        || pair.a.prefix == pair.b.prefix
        || pair.a.expected == pair.b.expected
    {
        return Err(format!(
            "pair {} has no distinct histories and answers at a shared final input",
            pair.pair_id
        )
        .into());
    }
    Ok(())
}

fn prefill(model: &ButterflyV3Model, prefix: &[u16]) -> Result<ButterflyV3Session> {
    let mut state = model.new_session();
    for &token in &prefix[..prefix.len() - 1] {
        model.step(token, &mut state)?;
    }
    Ok(state)
}

fn words(state: &ButterflyV3Session) -> Value {
    json!({"carry": state.carry, "shift_hist": state.shift_hist, "tokens_seen": state.tokens_seen})
}

fn output_words(step: &ButterflyV3StepOutput) -> Value {
    json!({
        "greedy_token": step.greedy_token,
        "greedy_coarse": step.greedy_coarse,
        "greedy_fine": step.greedy_fine,
        "coarse_scores": step.coarse_scores.as_slice(),
        "pkm_slots": step.pkm_slots,
        "hidden_q15": step.hidden_q15,
    })
}

fn fine_score(model: &ButterflyV3Model, hidden: &[i32], slot: usize) -> i32 {
    let dim = model.cfg.dim;
    let row = &model.embeddings[slot * dim..(slot + 1) * dim];
    let acc = hidden
        .iter()
        .zip(row)
        .map(|(&h, &w)| i64::from(h) * i64::from(w))
        .sum::<i64>();
    let shift = ButterflyV3Model::head_dot_shift(dim);
    clamp_act(((acc + (1i64 << (shift - 1))) >> shift) as i32)
}

fn target_diag(
    model: &ButterflyV3Model,
    step: &ButterflyV3StepOutput,
    token: u16,
) -> Result<Value> {
    let target_slot = usize::from(model.vocab_perm[usize::from(token)]);
    let cell = target_slot / HIER_FINE;
    let target_score = fine_score(model, &step.hidden_q15, target_slot);
    let mut rivals = Vec::new();
    for slot in cell * HIER_FINE..(cell + 1) * HIER_FINE {
        let candidate = model.vocab_inv[slot];
        if slot != target_slot && is_admissible(model, candidate) {
            rivals.push((slot, candidate, fine_score(model, &step.hidden_q15, slot)));
        }
    }
    let outrankers = rivals
        .iter()
        .filter(|&&(slot, _, score)| {
            score > target_score || (score == target_score && slot < target_slot)
        })
        .count();
    // max_by_key takes the last equal key; descending priority in the slot
    // component preserves the native first-slot tie rule regardless.
    let best_other = rivals
        .into_iter()
        .max_by_key(|&(slot, _, score)| (score, std::cmp::Reverse(slot)));
    let fine_winner = best_other
        .filter(|&(slot, _, score)| {
            score > target_score || (score == target_score && slot < target_slot)
        })
        .map_or(token, |(_, candidate, _)| candidate);
    let entered = cell == usize::from(step.greedy_coarse);
    if entered && fine_winner != step.greedy_token {
        return Err("fine diagnostic winner disagrees with reference step".into());
    }
    Ok(json!({
        "target": token,
        "cell": cell,
        "coarse_score": step.coarse_scores[cell],
        "coarse_score_minus_winner": i64::from(step.coarse_scores[cell])
            - i64::from(step.coarse_scores[usize::from(step.greedy_coarse)]),
        "entered_target_cell": entered,
        "target_fine": target_slot % HIER_FINE,
        "target_score": target_score,
        "best_other": best_other,
        "target_minus_best_other": best_other.map(|(_, _, score)| i64::from(target_score) - i64::from(score)),
        "target_fine_rank": outrankers + 1,
        "fine_winner": fine_winner,
    }))
}

fn outcome(emitted: &[u16], expected: &[u16], reached_eos: bool) -> Value {
    let mismatch = emitted
        .iter()
        .zip(expected)
        .position(|(observed, gold)| observed != gold)
        .or_else(|| (emitted.len() > expected.len()).then_some(expected.len()));
    let verdict = if mismatch.is_some() {
        "first_wrong"
    } else if reached_eos && emitted == expected {
        "exact_through_eos"
    } else {
        "censored"
    };
    json!({"verdict": verdict, "first_mismatch_index": mismatch})
}

fn run_arm(
    model: &ButterflyV3Model,
    recipient: &ButterflyV3Session,
    carry: &[Vec<i32>],
    row: &Row,
    rival: &Row,
    eos: u16,
) -> Result<Value> {
    let mut state = recipient.clone();
    state.carry.clone_from_slice(carry);
    let mut step = model.step(row.prefix[row.prefix.len() - 1], &mut state)?;
    let first = output_words(&step);
    let after = words(&state);
    let mut emitted = Vec::new();
    let mut trace = Vec::new();
    for index in 0..MAX_EMITTED {
        let token = step.greedy_token;
        emitted.push(token);
        trace.push(json!({
            "index": index,
            "token": token,
            "greedy_coarse": step.greedy_coarse,
            "greedy_fine": step.greedy_fine,
            "own": row.expected.get(index).map(|&t| target_diag(model, &step, t)).transpose()?,
            "rival": rival.expected.get(index).map(|&t| target_diag(model, &step, t)).transpose()?,
        }));
        if token == eos || token == 0 {
            break;
        }
        if index + 1 < MAX_EMITTED {
            step = model.step(token, &mut state)?;
        }
    }
    let reached_eos = emitted.last() == Some(&eos);
    let stop = match emitted.last() {
        Some(&token) if token == eos => "eos",
        Some(&0) => "invalid_bos",
        _ => "cap",
    };
    Ok(json!({
        "first": first, "after": after, "emitted": emitted, "trace": trace,
        "stop": stop,
        "own_outcome": outcome(&emitted, &row.expected, reached_eos),
        "rival_outcome": outcome(&emitted, &rival.expected, reached_eos),
    }))
}

fn run_side(
    model: &ButterflyV3Model,
    row: &Row,
    rival: &Row,
    recipient: &ButterflyV3Session,
    donor: &ButterflyV3Session,
    eos: u16,
) -> Result<Value> {
    let initial = model.new_session();
    let native = run_arm(model, recipient, &recipient.carry, row, rival, eos)?;
    let swapped = run_arm(model, recipient, &donor.carry, row, rival, eos)?;
    let zeroed = run_arm(model, recipient, &initial.carry, row, rival, eos)?;
    let delta = |arm: &Value, carry_changed: bool| {
        json!({
            "carry_before": carry_changed,
            "carry_after": arm["after"]["carry"] != native["after"]["carry"],
            "hidden": arm["first"]["hidden_q15"] != native["first"]["hidden_q15"],
            "pkm_slots": arm["first"]["pkm_slots"] != native["first"]["pkm_slots"],
            "coarse_scores": arm["first"]["coarse_scores"] != native["first"]["coarse_scores"],
            "greedy": arm["first"]["greedy_token"] != native["first"]["greedy_token"],
        })
    };
    Ok(json!({
        "before": words(recipient), "native": native, "swap": swapped, "zero": zeroed,
        "changed_from_native": {"swap": delta(&swapped, recipient.carry != donor.carry), "zero": delta(&zeroed, recipient.carry != initial.carry)},
    }))
}

fn compare(model: &ButterflyV3Model, pair: Pair, eos: u16) -> Result<Value> {
    validate(model, &pair, eos)?;
    let a = prefill(model, &pair.a.prefix)?;
    let b = prefill(model, &pair.b.prefix)?;
    Ok(json!({
        "pair_id": pair.pair_id,
        "a": run_side(model, &pair.a, &pair.b, &a, &b, eos)?,
        "b": run_side(model, &pair.b, &pair.a, &b, &a, eos)?,
    }))
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [checkpoint, checkpoint_sha, pairs, pairs_sha, eos] = args.as_slice() else {
        return Err(
            "usage: compact-carry-replay CHECKPOINT CHECKPOINT_SHA256 PAIRS_JSONL PAIRS_SHA256 EOS"
                .into(),
        );
    };
    let checkpoint = fs::read(checkpoint)?;
    let pairs_file = fs::read(pairs)?;
    for (label, bytes, pin) in [
        ("checkpoint", &checkpoint, checkpoint_sha),
        ("pairs", &pairs_file, pairs_sha),
    ] {
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != *pin {
            return Err(format!("{label} SHA256 mismatch: expected {pin}, got {actual}").into());
        }
        eprintln!("{label}_sha256={actual}");
    }
    let model = ButterflyV3Model::from_bytes(&checkpoint)?;
    let eos: u16 = eos.parse()?;
    if !is_admissible(&model, eos) {
        return Err("EOS is not an admissible output token".into());
    }
    let mut seen = HashSet::new();
    let mut output = BufWriter::new(io::stdout().lock());
    for line in BufReader::new(pairs_file.as_slice()).lines() {
        let pair: Pair = serde_json::from_str(&line?)?;
        if !seen.insert(pair.pair_id) {
            return Err(format!("duplicate pair id {}", pair.pair_id).into());
        }
        serde_json::to_writer(&mut output, &compare(&model, pair, eos)?)?;
        writeln!(output)?;
    }
    if seen.is_empty() {
        return Err("frozen cohort is empty".into());
    }
    output.flush()?;
    eprintln!("pairs={} max_emitted={MAX_EMITTED} eos={eos}", seen.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_r4_native_vector::butterfly_rotor::LossKind;
    use uor_r4_native_vector::butterfly_rotor_v3::ButterflyV3Config;

    fn fixture() -> Result<ButterflyV3Model> {
        Ok(ButterflyV3Model::new(ButterflyV3Config {
            vocab: 128,
            dim: 32,
            layers: 2,
            ffn_branches: 1,
            pkm_sub_keys: 8,
            pkm_key_dim: 8,
            loss_kind: LossKind::Sparsemax64,
            seed: 576,
        })?)
    }

    #[test]
    fn identity_and_planted_carry_are_distinguished() -> Result<()> {
        let model = fixture()?;
        let own = Row {
            prefix: vec![0, 4, 7, 11, 5],
            expected: vec![5, 1],
        };
        let rival = Row {
            prefix: vec![0, 9, 2, 21, 5],
            expected: vec![77, 1],
        };
        let recipient = prefill(&model, &own.prefix)?;
        let donor = prefill(&model, &rival.prefix)?;
        let native = run_arm(&model, &recipient, &recipient.carry, &own, &rival, 1)?;
        let identity = run_arm(&model, &recipient, &recipient.carry, &own, &rival, 1)?;
        assert_eq!(native, identity);
        assert_ne!(recipient.carry, donor.carry);
        assert_ne!(recipient.shift_hist, donor.shift_hist);
        let swapped = run_arm(&model, &recipient, &donor.carry, &own, &rival, 1)?;
        let mut planted = recipient.carry.clone();
        for layer in &mut planted {
            for lane in layer.chunks_exact_mut(4) {
                lane.copy_from_slice(&[0, 32767, 0, 0]);
            }
        }
        let planted = run_arm(&model, &recipient, &planted, &own, &rival, 1)?;
        assert_ne!(
            native["first"]["hidden_q15"],
            swapped["first"]["hidden_q15"]
        );
        assert_ne!(
            native["first"]["hidden_q15"],
            planted["first"]["hidden_q15"]
        );
        assert_ne!(native["first"]["pkm_slots"], planted["first"]["pkm_slots"]);
        assert_eq!(
            native["after"]["tokens_seen"],
            swapped["after"]["tokens_seen"]
        );
        // With two layers an intervention may alter subsequent layer shift
        // history. Layer 0 only sees the unchanged embedding and its history.
        assert_eq!(
            native["after"]["shift_hist"][0],
            swapped["after"]["shift_hist"][0]
        );
        Ok(())
    }

    #[test]
    fn in_cell_winner_respects_slot_order_and_wrong_cell_stays_inaccessible() -> Result<()> {
        let mut model = fixture()?;
        model.embeddings.fill(0);
        model.coarse_centroids.fill(0);
        let output = model.step(4, &mut model.new_session())?;
        assert_eq!(output.greedy_token, 1);
        let tied_admissible = target_diag(&model, &output, 3)?;
        assert_eq!(tied_admissible["target_fine_rank"], 3);
        assert_eq!(tied_admissible["target_minus_best_other"], 0);
        let wrong_cell = target_diag(&model, &output, 64)?;
        assert_eq!(wrong_cell["target_fine_rank"], 1);
        assert_eq!(wrong_cell["coarse_score_minus_winner"], 0);
        assert_eq!(wrong_cell["entered_target_cell"], false);
        assert_eq!(wrong_cell["fine_winner"], 64);
        Ok(())
    }

    #[test]
    fn greedy_eos_mismatch_and_cap_are_distinct() -> Result<()> {
        let mut model = fixture()?;
        model.embeddings.fill(0);
        model.coarse_centroids.fill(0);
        let recipient = model.new_session();
        let run = |expected, eos| {
            let own = Row {
                prefix: vec![0, 9],
                expected,
            };
            run_arm(&model, &recipient, &recipient.carry, &own, &own, eos)
        };
        let eos = run(vec![1], 1)?;
        assert_eq!(eos["stop"], "eos");
        assert_eq!(eos["own_outcome"]["verdict"], "exact_through_eos");
        let wrong = run(vec![1, 2], 2)?;
        assert_eq!(wrong["stop"], "cap");
        assert_eq!(wrong["own_outcome"]["verdict"], "first_wrong");
        assert_eq!(wrong["own_outcome"]["first_mismatch_index"], 1);
        assert_eq!(wrong["emitted"], json!(vec![1; MAX_EMITTED]));
        let mut gold_over_cap = vec![1; MAX_EMITTED];
        gold_over_cap.push(2);
        let capped = run(gold_over_cap, 2)?;
        assert_eq!(capped["stop"], "cap");
        assert_eq!(capped["own_outcome"]["verdict"], "censored");
        assert_eq!(capped["emitted"], json!(vec![1; MAX_EMITTED]));
        Ok(())
    }
}
