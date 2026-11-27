use crate::{
    engine::P5Gpu,
    fail, measure, replay,
    rows::{NativeRow, RowsGpu, Sample},
    sample::{self, Corpus, Draw, Position},
    Result,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, process::Command, time::Instant};
use uor_r4_count_runtime::p4_serving::P4_MAX_CONTEXT;
use uor_r4_discrete::p5_brier_core::{P5BatchBrier, P5S1State};
use uor_r4_integer::discrete_d11::count_deployment::SCHEMA_P4;

const PRIOR_SEED: u64 = 4309;
const PRIOR_COUNT: usize = 512;
const FRESH_SEED: u64 = 4313;
const FRESH_COUNT: usize = 1024;
const ELIGIBLE_COUNT: usize = 2158;
const REGRESSION_INDICES: [usize; 3] = [6, 36, 413];
const FROZEN_P4_SHA: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const FROZEN_TRUNK_SHA: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
const PRIOR_STANDARD_SHA: &str = "f681fad83f59376bf5b6a3598c1af0e89834bc6b98e6645c1b918544c990c0ac";
const ARM_NAMES: [&str; 3] = ["frozen", "standard", "penalty"];

fn fresh_positions(dev: &Corpus, prior: &[Position]) -> Result<Vec<Position>> {
    let excluded: BTreeSet<_> = prior.iter().map(|p| p.record).collect();
    let mut starts: Vec<_> = dev
        .ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id == 0).then_some(i))
        .collect();
    if excluded.len() != PRIOR_COUNT
        || starts.len() != PRIOR_COUNT + ELIGIBLE_COUNT
        || excluded.iter().any(|p| starts.binary_search(p).is_err())
    {
        return Err(fail(
            "fresh DEV frame does not match the registered record counts",
        ));
    }
    starts.push(dev.ids.len());
    let remaining: Vec<_> = starts
        .windows(2)
        .filter(|ends| !excluded.contains(&ends[0]))
        .collect();
    let mut draw = Draw::new(FRESH_SEED);
    let mut selected = BTreeSet::new();
    let mut positions = Vec::with_capacity(FRESH_COUNT);
    for i in 0..FRESH_COUNT {
        let lo = i * remaining.len() / FRESH_COUNT;
        let hi = (i + 1) * remaining.len() / FRESH_COUNT;
        let ends = remaining[lo + draw.index(hi - lo)];
        let target_at = 1 + draw.index(ends[1] - ends[0] - 1);
        if !selected.insert(ends[0])
            || excluded.contains(&ends[0])
            || dev.ids[ends[0] + target_at] == 0
        {
            return Err(fail("fresh DEV draw repeated or crossed a BOS record"));
        }
        positions.push(Position {
            record: ends[0],
            target_at,
        });
    }
    Ok(positions)
}

fn active_window(dev: &Corpus, pos: Position) -> &[u16] {
    let prefix = dev.prefix(pos);
    &prefix[prefix.len().saturating_sub(P4_MAX_CONTEXT)..]
}

fn query_pair(prefix: &[u16]) -> Option<[u16; 2]> {
    let [.., left, right] = prefix else {
        return None;
    };
    Some([*left, *right])
}

fn source_pair(prefix: &[u16], prediction: &NativeRow) -> Option<[u16; 2]> {
    prediction.pick_pos.map(|p| [prefix[p - 1], prefix[p]])
}

fn pair_kind(prefix: &[u16], prediction: &NativeRow) -> &'static str {
    match source_pair(prefix, prediction) {
        None => "no_pick",
        pair if pair == query_pair(prefix) => "literal",
        _ => "nonliteral",
    }
}

fn evaluate(
    gpu: &P5Gpu,
    reader: &RowsGpu,
    state: &P5S1State,
    batch: &P5BatchBrier,
    dev: &Corpus,
    samples: &[Sample],
    codes: Option<&[[u16; 4]]>,
) -> Result<(Vec<NativeRow>, Value)> {
    let start = Instant::now();
    let ordinary = reader.evaluate(&dev.ids, samples, codes)?;
    let ordinary_us = start.elapsed().as_micros();
    let start = Instant::now();
    let p5 = gpu.evaluate(state, batch)?;
    let lookup_us = start.elapsed().as_micros();
    if ordinary.len() != samples.len() || p5.rows.len() != ordinary.len() {
        return Err(fail(
            "ordinary DEV evaluation and native lookup were not aligned",
        ));
    }
    for (i, (r, p)) in ordinary.iter().zip(&p5.rows).enumerate() {
        let s = samples[i];
        if r.target != dev.ids[s.record + s.target_at]
            || r.pick_pos.unwrap_or(0) as u32 != p.position
            || r.pick_distance.unwrap_or(u32::MAX) != p.distance
            || r.pick_successor.unwrap_or(0) != p.successor
            || r.brier_q32 != p.loss_q32
        {
            return Err(fail(format!(
                "ordinary P4 and native lookup differ on DEV sample {i}"
            )));
        }
    }
    if p5.loss_q32 != ordinary.iter().map(|r| r.brier_q32).sum::<u64>() {
        return Err(fail(
            "ordinary DEV evaluation and native lookup Brier totals differ",
        ));
    }
    Ok((
        ordinary,
        json!({"ordinary_us": ordinary_us, "native_lookup_us": lookup_us,
        "native_lookup_device_us": p5.device_micros}),
    ))
}

fn comparison(before: &[NativeRow], after: &[NativeRow]) -> Result<Value> {
    let mut result = measure::compare(before, after)?;
    let n = before.len();
    let diffs: Vec<[i64; 2]> = before
        .iter()
        .zip(after)
        .map(|(a, b)| {
            [
                i64::from(b.top1 == b.target) - i64::from(a.top1 == a.target),
                b.prll_bits_q12 as i64 - a.prll_bits_q12 as i64,
            ]
        })
        .collect();
    // Same paired record draws as the audited pilot Brier interval.
    let mut draw = Draw::new(4312);
    let mut top = Vec::with_capacity(1000);
    let mut bits = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let (mut top_delta, mut bits_delta) = (0, 0);
        for _ in 0..n {
            let [t, b] = diffs[draw.index(n)];
            top_delta += t;
            bits_delta += b;
        }
        top.push(top_delta);
        bits.push(bits_delta);
    }
    top.sort_unstable();
    bits.sort_unstable();
    result["paired_top1_delta_ci95"] = json!([top[24], top[974]]);
    result["paired_prll_delta_ci95_q12"] = json!([bits[24], bits[974]]);
    result["top1_delta"] = json!(diffs.iter().map(|d| d[0]).sum::<i64>());
    result["prll_delta_q12"] = json!(diffs.iter().map(|d| d[1]).sum::<i64>());
    result["same_address"] = json!(before
        .iter()
        .zip(after)
        .filter(|(a, b)| a.pick_pos.is_some() && a.pick_pos == b.pick_pos)
        .count());
    result["both_top1_correct"] = json!(before
        .iter()
        .zip(after)
        .filter(|(a, b)| a.top1 == a.target && b.top1 == b.target)
        .count());
    result["both_top1_wrong"] = json!(before
        .iter()
        .zip(after)
        .filter(|(a, b)| a.top1 != a.target && b.top1 != b.target)
        .count());
    Ok(result)
}

fn useful_nonliteral(
    indices: &[usize],
    dev: &Corpus,
    positions: &[Position],
    frozen: &[NativeRow],
    arm: &[NativeRow],
) -> Value {
    let count = |test: fn(&[u16], &NativeRow, &NativeRow) -> bool| {
        indices
            .iter()
            .filter(|&&i| test(active_window(dev, positions[i]), &frozen[i], &arm[i]))
            .count()
    };
    json!({
        "rows":indices.len(),
        "top1_correct":count(|_, _, b| b.top1 == b.target),
        "top1_recovered_vs_frozen":count(|_, a, b| a.top1 != a.target && b.top1 == b.target),
        "top1_lost_vs_frozen":count(|_, a, b| a.top1 == a.target && b.top1 != b.target),
        "correct_source_successor":count(|_, _, b| b.pick_successor == Some(b.target)),
        "correct_nonliteral_successor":count(|w, _, b| b.pick_successor == Some(b.target) && pair_kind(w, b) == "nonliteral"),
        "exact_frozen_source_address_retained":count(|_, a, b| a.pick_pos == b.pick_pos),
        "frozen_source_pair_retained":count(|w, a, b| source_pair(w, a) == source_pair(w, b)),
        "frozen_pick_trace_retained":count(|_, a, b| measure::same_pick(a, b)),
    })
}

fn alias_precision(
    dev: &Corpus,
    positions: &[Position],
    frozen: &[NativeRow],
    arm: &[NativeRow],
) -> Value {
    let mut literal_frozen = 0;
    let mut correct_literal_frozen = 0;
    let mut any_displacement = 0;
    let mut displacements_of_correct = 0;
    let mut wrong_successors = 0;
    let mut worse_brier = 0;
    let mut top1_losses = 0;
    let mut increased_ties = 0;
    let mut unchanged_ties = 0;
    let mut decreased_ties = 0;
    let mut tie_counts = Vec::new();
    for (i, (&pos, (a, b))) in positions.iter().zip(frozen.iter().zip(arm)).enumerate() {
        let window = active_window(dev, pos);
        if a.pick_distance != Some(0) || pair_kind(window, a) != "literal" {
            continue;
        }
        literal_frozen += 1;
        let correct = a.pick_successor == Some(a.target);
        correct_literal_frozen += usize::from(correct);
        if b.pick_distance != Some(0) || pair_kind(window, b) != "nonliteral" {
            continue;
        }
        any_displacement += 1;
        if !correct {
            continue;
        }
        displacements_of_correct += 1;
        wrong_successors += usize::from(b.pick_successor != Some(b.target));
        worse_brier += usize::from(b.brier_q32 > a.brier_q32);
        top1_losses += usize::from(a.top1 == a.target && b.top1 != b.target);
        increased_ties += usize::from(b.pick_ties > a.pick_ties);
        unchanged_ties += usize::from(b.pick_ties == a.pick_ties);
        decreased_ties += usize::from(b.pick_ties < a.pick_ties);
        tie_counts.push(json!({"fresh_index":i,"frozen":a.pick_ties,"arm":b.pick_ties}));
    }
    json!({"frozen_zero_distance_literal":literal_frozen,
        "frozen_correct_zero_distance_literal":correct_literal_frozen,
        "all_zero_distance_literal_displacements":any_displacement,
        "displacements_of_correct_literal":displacements_of_correct,
        "wrong_successor_precision": {
            "numerator":wrong_successors,"denominator":displacements_of_correct},
        "worse_brier":worse_brier,"lost_top1":top1_losses,
        "ties_increased":increased_ties,"ties_unchanged":unchanged_ties,"ties_decreased":decreased_ties,
        "ties_by_displacement":tie_counts})
}

fn row_evidence(
    dev: &Corpus,
    pos: Position,
    index: usize,
    arms: &[Vec<NativeRow>; 3],
    offset: usize,
) -> Value {
    let prefix = active_window(dev, pos);
    let window_offset = pos.target_at - prefix.len();
    let mut record = json!({"index":index,"bos_offset":pos.record,"target_offset_in_record":pos.target_at,
        "target":dev.ids[pos.record + pos.target_at],"active_prefix_sha256":hex::encode(sample::prefix_digest(dev, pos)),
        "active_window_offset_in_record":window_offset,"active_window_tokens":prefix.len(),"query_pair":query_pair(prefix)});
    for (name, rows) in ARM_NAMES.into_iter().zip(arms) {
        let prediction = &rows[offset];
        let mut row = measure::row_json(prediction);
        row["source_pair"] = json!(source_pair(prefix, prediction));
        row["source_kind"] = json!(pair_kind(prefix, prediction));
        row["source_second_token_absolute_offset"] =
            json!(prediction.pick_pos.map(|p| pos.record + window_offset + p));
        record[name] = row;
    }
    record
}

fn sha_state(state: &P5S1State) -> Result<String> {
    Ok(hex::encode(Sha256::digest(state.encode_bytes()?)))
}

fn prior_regressions(rows: &[Value]) -> Result<()> {
    let historical: Value =
        serde_json::from_str(include_str!("../codex-4308/evidence/measurement.json"))?;
    let prior = historical["posthoc_source_exactness"]["three_zero_distance_aliases"]
        .as_array()
        .ok_or_else(|| fail("missing archived pilot regression scores"))?;
    if prior.len() != rows.len() {
        return Err(fail("wrong archived pilot regression count"));
    }
    for (saved, replay) in prior.iter().zip(rows) {
        if saved["index"] != replay["index"]
            || saved["target"] != replay["target"]
            || saved["prefix_sha256"] != replay["active_prefix_sha256"]
            || saved["bos"] != replay["bos_offset"]
            || saved["depth"] != replay["target_offset_in_record"]
        {
            return Err(fail(
                "historical DEV regression sample differs from the pinned pilot",
            ));
        }
        for (old, new) in [("frozen", "frozen"), ("trained", "standard")] {
            let fields = saved[old]
                .as_object()
                .ok_or_else(|| fail("missing archived GPU row"))?;
            if fields
                .iter()
                .any(|(field, value)| replay[new][field] != *value)
            {
                return Err(fail(format!(
                    "historical regression {} arm {new} differs from the pinned pilot",
                    saved["index"]
                )));
            }
        }
    }
    Ok(())
}

pub struct Witnesses {
    fixed: Vec<Vec<u16>>,
    diagnostic: Option<Vec<u16>>,
    expected: Vec<Value>,
    p4_sha: String,
    trunk_sha: String,
    penalty_sha: String,
    frozen_epoch: u64,
    penalty_epoch: u64,
}

pub fn run(
    gpu: &P5Gpu,
    reader: &RowsGpu,
    base: &P5S1State,
    standard: &P5S1State,
    penalty: &P5S1State,
    dev: &Corpus,
    out: &Path,
) -> Result<Witnesses> {
    let standard_sha = sha_state(standard)?;
    let penalty_sha = sha_state(penalty)?;
    let trunk_sha = hex::encode(base.parent_trunk_sha256);
    if reader.artifact_sha256 != FROZEN_P4_SHA
        || trunk_sha != FROZEN_TRUNK_SHA
        || standard_sha != PRIOR_STANDARD_SHA
        || standard.parent_trunk_sha256 != base.parent_trunk_sha256
        || penalty.parent_trunk_sha256 != base.parent_trunk_sha256
    {
        return Err(fail("alias evaluation must reproduce the frozen P4, parent trunk and prior standard checkpoint"));
    }
    let prior = dev.dev_sample(PRIOR_COUNT, PRIOR_SEED)?;
    let fresh = fresh_positions(dev, &prior)?;
    let regression = REGRESSION_INDICES.map(|i| prior[i]);
    let positions: Vec<_> = fresh.iter().chain(&regression).copied().collect();
    let samples: Vec<_> = positions
        .iter()
        .map(|p| Sample {
            record: p.record,
            target_at: p.target_at,
        })
        .collect();
    let start = Instant::now();
    let prepared = reader.prepare_for_training(&dev.ids, &samples)?;
    let prepare_us = start.elapsed().as_micros();
    let standard_codes = standard.derive_pick_codes_4()?;
    let penalty_codes = penalty.derive_pick_codes_4()?;
    let (frozen_rows, frozen_time) =
        evaluate(gpu, reader, base, &prepared.batch, dev, &samples, None)?;
    let (standard_rows, standard_time) = evaluate(
        gpu,
        reader,
        standard,
        &prepared.batch,
        dev,
        &samples,
        Some(&standard_codes),
    )?;
    let (penalty_rows, penalty_time) = evaluate(
        gpu,
        reader,
        penalty,
        &prepared.batch,
        dev,
        &samples,
        Some(&penalty_codes),
    )?;
    drop(prepared);
    let arms = [frozen_rows, standard_rows, penalty_rows];
    let [frozen_dev, standard_dev, penalty_dev] = arms.each_ref().map(|arm| &arm[..FRESH_COUNT]);
    let rows: Vec<_> = fresh
        .iter()
        .enumerate()
        .map(|(i, &p)| row_evidence(dev, p, i, &arms, i))
        .collect();
    let regressions: Vec<_> = REGRESSION_INDICES
        .iter()
        .enumerate()
        .map(|(i, &old_i)| row_evidence(dev, regression[i], old_i, &arms, FRESH_COUNT + i))
        .collect();
    prior_regressions(&regressions)?;

    let useful: Vec<_> = fresh
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            let row = &frozen_dev[i];
            (pair_kind(active_window(dev, p), row) == "nonliteral"
                && row.pick_successor == Some(row.target))
            .then_some(i)
        })
        .collect();
    let fixed_indices = [0, 1, FRESH_COUNT + 1, FRESH_COUNT + 2];
    let fixed: Vec<_> = fixed_indices
        .iter()
        .map(|&i| active_window(dev, positions[i]).to_vec())
        .collect();
    let ordinary_difference = |i: usize| {
        let a = &standard_dev[i];
        let b = &penalty_dev[i];
        a.brier_q32 != b.brier_q32
            || a.target_mass_q48 != b.target_mass_q48
            || a.top1_mass_q48 != b.top1_mass_q48
            || a.top1 != b.top1
            || !measure::same_pick(a, b)
    };
    let diagnostic_index = (2..FRESH_COUNT).find(|&i| {
        ordinary_difference(i)
            && !fixed
                .iter()
                .any(|prefix| prefix == active_window(dev, positions[i]))
    });
    let diagnostic = diagnostic_index.map(|i| active_window(dev, positions[i]).to_vec());
    let witness_indices: Vec<_> = fixed_indices.into_iter().chain(diagnostic_index).collect();
    let witness_samples: Vec<_> = witness_indices.iter().map(|&i| samples[i]).collect();
    let hashes = [
        reader.mass_hashes(&dev.ids, &witness_samples, None)?,
        reader.mass_hashes(&dev.ids, &witness_samples, Some(&standard_codes))?,
        reader.mass_hashes(&dev.ids, &witness_samples, Some(&penalty_codes))?,
    ];
    if hashes.iter().any(|arm| arm.len() != witness_indices.len()) {
        return Err(fail(
            "GPU score vector digests do not cover every replay witness",
        ));
    }
    let expected: Vec<_> = witness_indices
        .iter()
        .enumerate()
        .map(|(j, &i)| {
            let (original, selection) = if i >= FRESH_COUNT {
                (&regressions[i - FRESH_COUNT], "fixed_historical")
            } else if j >= fixed_indices.len() {
                (
                    &rows[i],
                    "first_additional_fresh_standard_penalty_disagreement",
                )
            } else {
                (&rows[i], "fixed_fresh")
            };
            let mut evidence = original.clone();
            evidence["selection"] = json!(selection);
            for (arm, scores) in ARM_NAMES.into_iter().zip(&hashes) {
                evidence[arm]["mass_sha256"] = json!(scores[j]);
            }
            evidence
        })
        .collect();

    measure::json_file(
        &out.join("alias-dev-rows.json"),
        &json!({"fresh":rows,"historical_regressions":regressions}),
    )?;
    measure::json_file(
        &out.join("alias-dev.json"),
        &json!({
            "schema":"uor-r4.p5-alias-dev/1","p4_sha256":reader.artifact_sha256,"parent_trunk_sha256":trunk_sha,
            "dev_tune_sha256":sample::DEV_SHA,"rows_cubin_sha256":reader.cubin_sha256,
            "checkpoint_sha256":{"frozen":sha_state(base)?,"standard":standard_sha,"penalty":penalty_sha},
            "epoch":{"frozen":base.epoch,"standard":standard.epoch,"penalty":penalty.epoch},
            "sampling":{"fresh_seed":FRESH_SEED,"fresh_records":fresh.len(),"fresh_unique_records":fresh.len(),
                "corpus_records":dev.record_count(),"eligible_after_exclusion":ELIGIBLE_COUNT,
                "prior_seed":PRIOR_SEED,"prior_excluded_records":prior.len(),"overlap_records":0,
                "prior_panel_sha256":sample::sample_digest(dev, &prior),"fresh_panel_sha256":sample::sample_digest(dev, &fresh),
                "regression_indices":REGRESSION_INDICES,"regression_panel_sha256":sample::sample_digest(dev, &regression),
                "slid_fresh_records":fresh.iter().filter(|p| p.target_at > P4_MAX_CONTEXT).count(),
                "method":"Exclude prior BOS offsets. Split the remaining ordered BOS offsets into 1024 strata with floor-scaled endpoints. Draw one record and one non-BOS target from each stratum with SplitMix64."},
            "bootstrap":{"unit":"paired fresh record","seed":4312,"replicates":1000,"percentile_indices_zero_based":[24,974]},
            "fresh":{"frozen":measure::compare(frozen_dev, frozen_dev)?,
                "standard_vs_frozen":comparison(frozen_dev, standard_dev)?,"penalty_vs_frozen":comparison(frozen_dev, penalty_dev)?,
                "penalty_vs_standard":comparison(standard_dev, penalty_dev)?},
            "useful_nonliteral":{"definition":"Frozen selected source has different original token IDs from the query pair and its successor equals the held target.",
                "fresh_indices":useful,
                "frozen":useful_nonliteral(&useful, dev, &fresh, frozen_dev, frozen_dev),
                "standard":useful_nonliteral(&useful, dev, &fresh, frozen_dev, standard_dev),
                "penalty":useful_nonliteral(&useful, dev, &fresh, frozen_dev, penalty_dev)},
            "literal_alias_displacement":{"definition":"Changed from frozen zero-distance literal pair to a zero-distance nonliteral pair. Precision is the fraction of correct-literal displacements whose new successor is wrong.",
                "standard":alias_precision(dev, &fresh, frozen_dev, standard_dev),
                "penalty":alias_precision(dev, &fresh, frozen_dev, penalty_dev)},
            "regression_matched_pilot":true,"fixed_fresh_replay_indices":[0,1],"fixed_regression_replay_indices":[36,413],
            "first_fresh_standard_penalty_disagreement":(0..FRESH_COUNT).find(|&i| ordinary_difference(i)),
            "optional_fresh_replay_index":diagnostic_index,"native_lookup_matching_rows_per_arm":positions.len(),
            "timing":{"prepare_us":prepare_us,"frozen":frozen_time,"standard":standard_time,"penalty":penalty_time},
        }),
    )?;
    measure::json_file(&out.join("alias-replay-gpu.json"), &json!(expected))?;
    Ok(Witnesses {
        fixed,
        diagnostic,
        expected,
        p4_sha: reader.artifact_sha256.clone(),
        trunk_sha,
        penalty_sha,
        frozen_epoch: base.epoch,
        penalty_epoch: penalty.epoch,
    })
}

fn regression_cli(
    chat: &Path,
    artifact: &Path,
    identity: &Value,
    prefix: &[u16],
    observation: &Value,
) -> Result<Value> {
    let csv = prefix
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let start = Instant::now();
    let output = Command::new(chat)
        .env("UOR_P1_TIMING", "1")
        .arg("--discrete")
        .arg(artifact)
        .args(["--prompt-ids", &csv, "--max-tokens", "1", "--p1-trace"])
        .output()?;
    let wall_ns = start.elapsed().as_nanos();
    if !output.status.success() {
        return Err(fail(format!(
            "cold regression uor-chat exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let trace: Value = serde_json::from_slice(&output.stdout)?;
    if trace["schema"] != "uor-r4.uor-chat-discrete-parity/1"
        || trace["model_schema"] != SCHEMA_P4
        || trace["read_enabled"] != true
        || trace["artifact_sha256"] != identity["artifact_sha256"]
        || trace["p4_trunk_sha256"] != identity["trunk_sha256"]
        || trace["p4_state_sha256"] != identity["state_sha256"]
        || trace["p4_components"] != identity["components"]
        || trace["prompt_ids"] != json!(prefix)
        || trace["generated_ids"] != json!([observation["token"]])
        || trace["p4_steps"] != json!([observation])
    {
        return Err(fail(
            "cold regression uor-chat disagrees with the GPU-linked P4 trace",
        ));
    }
    let timing = std::str::from_utf8(&output.stderr)?
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|line| line["schema"] == "uor-r4.p1-client-timing/1")
        .ok_or_else(|| fail("cold regression uor-chat did not return load and decode timing"))?;
    Ok(json!({"wall_ns":wall_ns,"timing":timing,
        "stdout_sha256":hex::encode(Sha256::digest(&output.stdout)),
        "artifact_sha256":trace["artifact_sha256"],"trunk_sha256":trace["p4_trunk_sha256"],
        "state_sha256":trace["p4_state_sha256"],"generated_ids":trace["generated_ids"],"step":observation}))
}

pub fn reload(
    witnesses: &Witnesses,
    p4: &Path,
    trunk: &Path,
    penalty_checkpoint: &Path,
    chat: &Path,
    out: &Path,
) -> Result<()> {
    measure::verify_file(penalty_checkpoint, &witnesses.penalty_sha)?;
    let serving = replay::run(
        p4,
        trunk,
        penalty_checkpoint,
        chat,
        &witnesses.fixed,
        witnesses.diagnostic.as_deref(),
        &out.join("alias-serving"),
    )?;
    if serving["baseline"]["artifact_sha256"] != witnesses.p4_sha
        || serving["baseline"]["trunk_sha256"] != witnesses.trunk_sha
        || serving["checkpoint_sha256"] != witnesses.penalty_sha
        || serving["baseline_epoch"] != witnesses.frozen_epoch
        || serving["trained_epoch"] != witnesses.penalty_epoch
    {
        return Err(fail(
            "ordinary packed reload identity differs from GPU evaluation",
        ));
    }
    let replay_rows = serving["rows"]
        .as_array()
        .ok_or_else(|| fail("missing packed reload rows"))?;
    if replay_rows.len() != witnesses.expected.len() {
        return Err(fail("packed reload did not return every GPU witness"));
    }
    let mut historical = Vec::new();
    for (i, (actual, expected)) in replay_rows.iter().zip(&witnesses.expected).enumerate() {
        for (loaded, gpu_arm) in [("baseline", "frozen"), ("trained", "penalty")] {
            let trace = &actual[loaded];
            let gpu = &expected[gpu_arm];
            for (a, b) in [
                ("token", "top1"),
                ("mass_q48", "top1_mass_q48"),
                ("pick_pos", "pick_pos"),
                ("pick_distance", "distance"),
                ("pick_successor", "successor"),
                ("pick_class", "class"),
                ("pick_ties", "ties"),
                ("bucket", "bucket"),
                ("mass_sha256", "mass_sha256"),
            ] {
                if trace.get(a).is_none() || trace[a] != gpu[b] {
                    return Err(fail(format!(
                        "packed reload {loaded} {a} differs from GPU witness {} ({})",
                        expected["index"], expected["selection"]
                    )));
                }
            }
            if trace["prefix_sha256"] != expected["active_prefix_sha256"]
                || trace["context_tokens"] != expected["active_window_tokens"]
            {
                return Err(fail(
                    "packed reload did not receive the exact GPU causal prefix",
                ));
            }
        }
        if expected["selection"] == "fixed_historical" {
            let index = expected["index"]
                .as_u64()
                .ok_or_else(|| fail("historical witness is missing its index"))?;
            historical.push((i, index));
        }
    }
    if historical
        .iter()
        .map(|&(_, index)| index)
        .ne(REGRESSION_INDICES[1..].iter().map(|&index| index as u64))
    {
        return Err(fail(
            "cold regression CLI witnesses must be historical rows 36 and 413",
        ));
    }
    let trained = serving["trained_path"]
        .as_str()
        .ok_or_else(|| fail("missing packed penalty artifact path"))?;
    let mut cold_regressions = Vec::with_capacity(historical.len());
    for (i, index) in historical {
        let actual = &replay_rows[i];
        let prefix = witnesses
            .fixed
            .get(i)
            .ok_or_else(|| fail("historical witness is missing its fixed prefix"))?;
        if actual["prefix_ids"] != json!(prefix) {
            return Err(fail(format!(
                "historical witness {index} changed its prompt IDs"
            )));
        }
        let mut row = json!({"historical_dev_index":index});
        for (arm, loaded, artifact) in [
            ("frozen", "baseline", p4),
            ("penalty", "trained", Path::new(trained)),
        ] {
            let receipt = regression_cli(chat, artifact, &serving[loaded], prefix, &actual[loaded])
                .map_err(|error| fail(format!("historical witness {index}, {arm}: {error}")))?;
            measure::json_file(
                &out.join("alias-serving")
                    .join(format!("cold-{index}-{arm}.json")),
                &receipt,
            )?;
            row[arm] = receipt;
        }
        cold_regressions.push(row);
    }
    measure::json_file(
        &out.join("alias-replay.json"),
        &json!({
            "schema":"uor-r4.p5-alias-replay/1","matched":true,"gpu_witnesses":"alias-replay-gpu.json",
            "ordinary_receipt":"alias-serving/serving.json","frozen":serving["baseline"],"penalty":serving["trained"],
            "penalty_checkpoint_sha256":witnesses.penalty_sha,"ordinary_cli_selected_index":serving["selected_cli_row"],
            "ordinary_cli_regressions":cold_regressions,
            "rows":witnesses.expected.iter().map(|r| json!({"index":r["index"],"selection":r["selection"],
                "prefix_sha256":r["active_prefix_sha256"],"frozen_mass_sha256":r["frozen"]["mass_sha256"],
                "penalty_mass_sha256":r["penalty"]["mass_sha256"]})).collect::<Vec<_>>(),
        }),
    )
}
