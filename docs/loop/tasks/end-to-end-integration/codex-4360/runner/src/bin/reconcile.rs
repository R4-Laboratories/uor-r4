#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::{error::Error, fs, io};
use uor_r4_native_vector::butterfly_rotor::{HIER_COARSE, HIER_FINE, PKM_TOP_K};
use uor_r4_native_vector::butterfly_rotor_v3::ButterflyV3Model;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const ARTIFACT_SHA: &str = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";
const PAIRS_SHA: &str = "8640f5f72de45f2c8a0c6fb8990830691da3e5602cf87ac506ff584b3f4182b4";
const PARITY_IDS: [usize; 6] = [1, 2, 17, 32, 48, 64];
const PAIRS: usize = 64;
const CAP: usize = 32;
const BOS: u16 = 0;
const EOS: u16 = 1;
const ARMS: [&str; 3] = ["native", "swap", "zero"];

#[derive(Debug, Deserialize)]
struct Pair {
    pair_id: usize,
    a: Row,
    b: Row,
    common_suffix_ids: usize,
    stratum: String,
    length_stratum: String,
}

#[derive(Debug, Deserialize)]
struct Row {
    prefix: Vec<u16>,
    expected: Vec<u16>,
    group: [String; 3],
    target_turn: usize,
    first_content: u16,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct State {
    carry: Vec<Vec<i32>>,
    shift_hist: Vec<Vec<i32>>,
    tokens_seen: usize,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Outcome {
    verdict: &'static str,
    first_mismatch_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Target {
    token: u16,
    cell: usize,
    within_cell_slot: usize,
    admitted: bool,
    coarse_score_q15: i32,
    fine_score_q15: i32,
    coarse_score_minus_winner_q15: i32,
    fine_score_minus_best_other_q15: Option<i32>,
    diagnostic_within_cell_rank: usize,
    admitted_within_cell_rank: Option<usize>,
    target_mass_q48: u64,
}

#[derive(Default, Serialize)]
struct Preferences {
    own: usize,
    other: usize,
    tie: usize,
}

impl Preferences {
    fn record(&mut self, order: Ordering) {
        match order {
            Ordering::Greater => self.own += 1,
            Ordering::Less => self.other += 1,
            Ordering::Equal => self.tie += 1,
        }
    }
}

#[derive(Default, Serialize)]
struct Rollouts {
    exact_through_eos: usize,
    first_mismatch_at_zero_based_index: BTreeMap<usize, usize>,
    censored: usize,
    premature_eos: usize,
}

#[derive(Default, Serialize)]
struct RankSummary {
    coarse_admitted_rows: usize,
    forced_cell_rank: BTreeMap<usize, usize>,
    forced_cell_top_one: usize,
    forced_cell_top_eight: usize,
    forced_cell_median: [usize; 2],
    genuine_top_one: usize,
}

impl RankSummary {
    fn record(&mut self, target: &Target, first: u16) {
        self.coarse_admitted_rows += usize::from(target.admitted);
        let rank = target.diagnostic_within_cell_rank;
        increment(&mut self.forced_cell_rank, rank);
        self.forced_cell_top_one += usize::from(rank == 1);
        self.forced_cell_top_eight += usize::from(rank <= 8);
        self.genuine_top_one += usize::from(target.token == first);
    }

    fn finish(&mut self) {
        let count: usize = self.forced_cell_rank.values().sum();
        if count != 0 {
            let rank_at = |index| {
                self.forced_cell_rank
                    .iter()
                    .flat_map(|(&rank, &n)| std::iter::repeat_n(rank, n))
                    .nth(index)
                    .unwrap_or_default()
            };
            self.forced_cell_median = [rank_at((count - 1) / 2), rank_at(count / 2)];
        }
    }
}

#[derive(Default, Serialize)]
struct ArmSummary {
    #[serde(flatten)]
    own_ranks: RankSummary,
    other_ranks: RankSummary,
    both_own_coarse_admitted_pairs: usize,
    both_other_coarse_admitted_pairs: usize,
    raw_mass_row_preferences: Preferences,
    raw_mass_sum_pair_assignments: Preferences,
    both_rows_strictly_prefer_own: usize,
    both_rows_strictly_prefer_other: usize,
    pairs_with_either_row_tied: usize,
    own_rollouts: Rollouts,
    other_rollouts: Rollouts,
    natural_eos: usize,
    cap: usize,
    invalid_bos: usize,
    first_original_id_counts: BTreeMap<u16, usize>,
    distinct_first_selected_cells: usize,
    distinct_own_target_cells: usize,
    distinct_other_target_cells: usize,
    pair_selected_cells_differ: usize,
    pair_target_cells_differ: usize,
    #[serde(skip)]
    selected: HashSet<usize>,
    #[serde(skip)]
    own_cells: HashSet<usize>,
    #[serde(skip)]
    other_cells: HashSet<usize>,
}

#[derive(Default, Serialize)]
struct Changes {
    first_hidden: usize,
    first_pkm_slots: usize,
    first_coarse_scores: usize,
    first_selected_cell: usize,
    first_emitted_id: usize,
    emitted_sequence: usize,
    first_carry_words: usize,
    own_forced_cell_rank_improved: usize,
    own_forced_cell_rank_worsened: usize,
    own_cell_gained_admission: usize,
    own_cell_lost_admission: usize,
    other_forced_cell_rank_improved: usize,
    other_forced_cell_rank_worsened: usize,
    other_cell_gained_admission: usize,
    other_cell_lost_admission: usize,
}

struct RecountedRow {
    first: u16,
    selected: usize,
    own: Target,
    other: Target,
    emitted: Vec<u16>,
    stop: &'static str,
    own_outcome: Outcome,
    other_outcome: Outcome,
}

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.to_owned().into());
    }
    Ok(())
}

fn increment<K: Ord>(map: &mut BTreeMap<K, usize>, key: K) {
    *map.entry(key).or_default() += 1;
}

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn parse_lines<T: DeserializeOwned>(bytes: &[u8]) -> Result<Vec<T>> {
    std::str::from_utf8(bytes)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|err| format!("JSONL line {}: {err}", i + 1).into())
        })
        .collect()
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| format!("missing field {name}").into())
}

fn read<T: DeserializeOwned>(value: &Value, name: &str) -> Result<T> {
    serde_json::from_value(field(value, name)?.clone())
        .map_err(|err| format!("invalid {name}: {err}").into())
}

fn matches(value: &Value, name: &str, expected: &Value) -> Result<()> {
    require(
        field(value, name)? == expected,
        &format!("mismatch in {name}"),
    )
}

fn validate_pair(
    pair: &Pair,
    vocab: usize,
    groups: &mut [HashSet<String>; 3],
    prefixes: &mut HashSet<Vec<u16>>,
) -> Result<()> {
    for row in [&pair.a, &pair.b] {
        require(
            row.prefix.len() >= 2
                && row.prefix.len() <= 2048
                && row.prefix[0] == BOS
                && row.prefix.iter().all(|&id| usize::from(id) < vocab),
            "invalid opening tape",
        )?;
        require(
            row.expected.len() >= 2
                && row.expected[0] == row.first_content
                && row.expected.last() == Some(&EOS)
                && row.expected[..row.expected.len() - 1]
                    .iter()
                    .all(|&id| id != BOS && id != EOS && usize::from(id) < vocab),
            "invalid gold tape",
        )?;
        require(row.target_turn > 0, "invalid target turn")?;
        for (seen, key) in groups.iter_mut().zip(&row.group) {
            require(
                !key.is_empty() && seen.insert(key.clone()),
                "reused or empty passage/group identity",
            )?;
        }
        require(prefixes.insert(row.prefix.clone()), "reused full opening")?;
    }
    let largest_opening = pair.a.prefix.len().max(pair.b.prefix.len());
    let stratum = if largest_opening <= 512 {
        "both_at_most_512"
    } else {
        "maximum_513_to_2048"
    };
    require(
        pair.length_stratum == stratum,
        "incorrect opening-length stratum",
    )?;
    let suffix = pair
        .a
        .prefix
        .iter()
        .rev()
        .zip(pair.b.prefix.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    require(
        suffix > 0
            && suffix == pair.common_suffix_ids
            && pair.a.first_content != pair.b.first_content
            && pair.stratum == "identical_literal_final_question",
        "invalid paired question, suffix or targets",
    )
}

fn outcome(emitted: &[u16], expected: &[u16], stop: &str) -> Outcome {
    let mismatch = emitted
        .iter()
        .enumerate()
        .find(|&(index, token)| expected.get(index) != Some(token))
        .map(|(index, _)| index);
    let verdict = match (mismatch, stop == "eos" && expected == emitted) {
        (Some(_), _) => "first_wrong",
        (None, true) => "exact_through_eos",
        _ => "censored",
    };
    Outcome {
        verdict,
        first_mismatch_index: mismatch,
    }
}

fn stop(emitted: &[u16], vocab: usize) -> Result<&'static str> {
    require((1..=CAP).contains(&emitted.len()), "invalid emitted length")?;
    let (last, prefix) = emitted.split_last().ok_or("empty emitted tape")?;
    require(
        prefix
            .iter()
            .all(|&token| token != BOS && token != EOS && usize::from(token) < vocab),
        "emitted after a terminating or invalid ID",
    )?;
    match *last {
        BOS => Ok("invalid_bos"),
        EOS => Ok("eos"),
        id if usize::from(id) < vocab && emitted.len() == CAP => Ok("cap"),
        _ => Err("invalid emitted ID or cap stop before cap".into()),
    }
}

fn validate_state(state: &State, tokens: usize, model: &ButterflyV3Model) -> Result<()> {
    let cfg = model.cfg;
    require(
        state.tokens_seen == tokens
            && state.carry.len() == cfg.layers
            && state.shift_hist.len() == cfg.layers
            && state.carry.iter().all(|x| x.len() == cfg.dim)
            && state.shift_hist.iter().all(|x| x.len() == cfg.dim * 3),
        "invalid per-layer state shape or token counter",
    )
}

fn validate_transplant(before: &State, recipient: &State, donor: &State, arm: &str) -> Result<()> {
    require(
        before.shift_hist == recipient.shift_hist && before.tokens_seen == recipient.tokens_seen,
        "recipient shift history or token count changed",
    )?;
    match arm {
        "native" => require(
            before.carry == recipient.carry,
            "native carry differs from recipient",
        ),
        "swap" => require(
            before.carry == donor.carry,
            "swapped carry differs from donor",
        ),
        "zero" => require(
            before.carry.iter().flatten().all(|&word| word == 0),
            "zero carry contains a nonzero word",
        ),
        _ => Err("unknown arm".into()),
    }
}

fn route(model: &ButterflyV3Model, token: u16) -> (usize, usize) {
    let slot = usize::from(model.vocab_perm[usize::from(token)]);
    (slot / HIER_FINE, slot % HIER_FINE)
}

fn validate_target(
    target: &Target,
    expected: u16,
    first_token: u16,
    scores: &[i32],
    model: &ButterflyV3Model,
) -> Result<()> {
    let (selected, _) = route(model, first_token);
    let (cell, fine) = route(model, expected);
    let rank = target.diagnostic_within_cell_rank;
    let admitted = cell == selected;
    require(
        target.token == expected
            && target.cell == cell
            && target.within_cell_slot == fine
            && target.admitted == admitted
            && target.coarse_score_q15 == scores[cell]
            && i64::from(target.coarse_score_minus_winner_q15)
                == i64::from(scores[cell]) - i64::from(scores[selected])
            && (1..=HIER_FINE).contains(&rank)
            && target.admitted_within_cell_rank == admitted.then_some(rank)
            && (expected == first_token) == (admitted && rank == 1)
            && target.target_mass_q48 <= (1_u64 << 48),
        "incorrect first-target word mapping, admission, rank, or mass bounds",
    )?;
    if let Some(margin) = target.fine_score_minus_best_other_q15 {
        require(
            !(margin > 0 && rank > 1 || margin < 0 && rank == 1),
            "fine margin contradicts rank",
        )?;
    }
    Ok(())
}

fn verify_ordinary(
    gpu: &Value,
    ordinary: &Value,
    row: &RecountedRow,
    model: &ButterflyV3Model,
) -> Result<()> {
    for name in ["first", "after", "emitted", "own_outcome"] {
        matches(ordinary, name, field(gpu, name)?)?;
    }
    matches(ordinary, "rival_outcome", field(gpu, "donor_outcome")?)?;
    matches(ordinary, "stop", &json!(row.stop))?;
    let trace = field(ordinary, "trace")?
        .as_array()
        .ok_or("ordinary trace is not an array")?;
    require(
        trace.len() == row.emitted.len(),
        "ordinary trace length differs from emissions",
    )?;
    for (index, (&token, record)) in row.emitted.iter().zip(trace).enumerate() {
        let (coarse, fine) = route(model, token);
        matches(record, "index", &json!(index))?;
        matches(record, "token", &json!(token))?;
        matches(record, "greedy_coarse", &json!(coarse))?;
        matches(record, "greedy_fine", &json!(fine))?;
    }
    let first = trace.first().ok_or("ordinary trace is empty")?;
    for (name, target) in [("own", &row.own), ("rival", &row.other)] {
        let diag = field(first, name)?;
        for (key, word) in [
            ("target", json!(target.token)),
            ("cell", json!(target.cell)),
            ("coarse_score", json!(target.coarse_score_q15)),
            (
                "coarse_score_minus_winner",
                json!(target.coarse_score_minus_winner_q15),
            ),
            ("entered_target_cell", json!(target.admitted)),
            ("target_fine", json!(target.within_cell_slot)),
            ("target_score", json!(target.fine_score_q15)),
            (
                "target_minus_best_other",
                json!(target.fine_score_minus_best_other_q15),
            ),
            (
                "target_fine_rank",
                json!(target.diagnostic_within_cell_rank),
            ),
        ] {
            matches(diag, key, &word)?;
        }
    }
    Ok(())
}

fn recount_arm(
    arm: &Value,
    label: &str,
    row: &Row,
    other_row: &Row,
    recipient: &State,
    donor: &State,
    model: &ButterflyV3Model,
) -> Result<RecountedRow> {
    let before: State = read(arm, "before")?;
    validate_state(&before, row.prefix.len() - 1, model)?;
    validate_transplant(&before, recipient, donor, label)?;
    validate_state(&read(arm, "after")?, row.prefix.len(), model)?;
    let emitted: Vec<u16> = read(arm, "emitted")?;
    let stop = stop(&emitted, model.cfg.vocab)?;
    let first_token = emitted[0];
    let first = field(arm, "first")?;
    let (selected, fine) = route(model, first_token);
    matches(first, "greedy_token", &json!(first_token))?;
    matches(first, "greedy_coarse", &json!(selected))?;
    matches(first, "greedy_fine", &json!(fine))?;
    let scores: Vec<i32> = read(first, "coarse_scores")?;
    let hidden: Vec<i32> = read(first, "hidden_q15")?;
    let slots: Vec<Vec<usize>> = read(first, "pkm_slots")?;
    require(
        scores.len() == HIER_COARSE
            && hidden.len() == model.cfg.dim
            && slots.len() == model.cfg.layers
            && slots.iter().all(|x| {
                x.len() == PKM_TOP_K
                    && x.iter()
                        .all(|&slot| slot < model.cfg.pkm_sub_keys * model.cfg.pkm_sub_keys)
            }),
        "first hidden, coarse or PKM words have invalid shape",
    )?;
    let elected = scores
        .iter()
        .enumerate()
        .max_by_key(|&(cell, score)| (*score, std::cmp::Reverse(cell)))
        .map(|(cell, _)| cell);
    require(
        elected == Some(selected),
        "first selected cell contradicts the full coarse scores and tie rule",
    )?;
    let own: Target = read(arm, "own")?;
    let other: Target = read(arm, "other")?;
    validate_target(&own, row.expected[0], first_token, &scores, model)?;
    validate_target(&other, other_row.expected[0], first_token, &scores, model)?;

    let own_outcome = outcome(&emitted, &row.expected, stop);
    let other_outcome = outcome(&emitted, &other_row.expected, stop);
    let reached_eos = stop == "eos";
    for (name, value) in [
        ("first_error", json!(own_outcome.first_mismatch_index)),
        (
            "donor_first_error",
            json!(other_outcome.first_mismatch_index),
        ),
        ("own_outcome", json!(own_outcome)),
        ("donor_outcome", json!(other_outcome)),
        ("exact", json!(own_outcome.verdict == "exact_through_eos")),
        (
            "donor_exact",
            json!(other_outcome.verdict == "exact_through_eos"),
        ),
        ("hit_eos", json!(reached_eos)),
        ("eos_token", json!(reached_eos.then_some(EOS))),
        ("stop_reason", json!(stop)),
    ] {
        matches(arm, name, &value)?;
    }
    Ok(RecountedRow {
        first: first_token,
        selected,
        own,
        other,
        emitted,
        stop,
        own_outcome,
        other_outcome,
    })
}

impl Rollouts {
    fn record(&mut self, outcome: &Outcome, stop: &str, emitted: usize, expected: usize) {
        if let Some(index) = outcome.first_mismatch_index {
            increment(&mut self.first_mismatch_at_zero_based_index, index);
        }
        self.exact_through_eos += usize::from(outcome.verdict == "exact_through_eos");
        self.censored += usize::from(outcome.verdict == "censored");
        self.premature_eos += usize::from(stop == "eos" && emitted < expected);
    }
}

impl ArmSummary {
    fn record(
        &mut self,
        a: &RecountedRow,
        b: &RecountedRow,
        pair: &Pair,
        assignment: &Value,
    ) -> Result<(Ordering, Ordering)> {
        let orders = [
            a.own.target_mass_q48.cmp(&a.other.target_mass_q48),
            b.own.target_mass_q48.cmp(&b.other.target_mass_q48),
        ];
        self.both_own_coarse_admitted_pairs += usize::from(a.own.admitted && b.own.admitted);
        self.both_other_coarse_admitted_pairs += usize::from(a.other.admitted && b.other.admitted);
        self.both_rows_strictly_prefer_own += usize::from(orders == [Ordering::Greater; 2]);
        self.both_rows_strictly_prefer_other += usize::from(orders == [Ordering::Less; 2]);
        self.pairs_with_either_row_tied += usize::from(orders.contains(&Ordering::Equal));
        self.pair_selected_cells_differ += usize::from(a.selected != b.selected);
        self.pair_target_cells_differ += usize::from(a.own.cell != b.own.cell);

        let own_mass = a.own.target_mass_q48 + b.own.target_mass_q48;
        let other_mass = a.other.target_mass_q48 + b.other.target_mass_q48;
        self.raw_mass_sum_pair_assignments
            .record(own_mass.cmp(&other_mass));
        let mut own_greedy = 0;
        let mut other_greedy = 0;
        for (row, on, own_len, other_len) in [
            (a, orders[0], pair.a.expected.len(), pair.b.expected.len()),
            (b, orders[1], pair.b.expected.len(), pair.a.expected.len()),
        ] {
            self.raw_mass_row_preferences.record(on);
            own_greedy += usize::from(row.first == row.own.token);
            other_greedy += usize::from(row.first == row.other.token);
            self.own_ranks.record(&row.own, row.first);
            self.other_ranks.record(&row.other, row.first);
            self.own_rollouts
                .record(&row.own_outcome, row.stop, row.emitted.len(), own_len);
            self.other_rollouts
                .record(&row.other_outcome, row.stop, row.emitted.len(), other_len);
            self.natural_eos += usize::from(row.stop == "eos");
            self.cap += usize::from(row.stop == "cap");
            self.invalid_bos += usize::from(row.stop == "invalid_bos");
            increment(&mut self.first_original_id_counts, row.first);
            self.selected.insert(row.selected);
            self.own_cells.insert(row.own.cell);
            self.other_cells.insert(row.other.cell);
        }
        let computed = json!({
            "native_mass_q48_sum": own_mass,
            "swapped_mass_q48_sum": other_mass,
            "native_minus_swapped_q48": own_mass as i64 - other_mass as i64,
            "native_greedy_matches": own_greedy,
            "swapped_greedy_matches": other_greedy,
        });
        require(
            assignment == &computed,
            "raw assignment disagrees with member masses or IDs",
        )?;
        Ok((orders[0], orders[1]))
    }

    fn finish(&mut self) {
        self.own_ranks.finish();
        self.other_ranks.finish();
        self.distinct_first_selected_cells = self.selected.len();
        self.distinct_own_target_cells = self.own_cells.len();
        self.distinct_other_target_cells = self.other_cells.len();
    }
}

impl Changes {
    fn record(
        &mut self,
        native: &Value,
        changed: &Value,
        clean: &RecountedRow,
        transplant: &RecountedRow,
    ) -> Result<()> {
        let clean_first = field(native, "first")?;
        let changed_first = field(changed, "first")?;
        self.first_hidden +=
            usize::from(field(clean_first, "hidden_q15")? != field(changed_first, "hidden_q15")?);
        self.first_pkm_slots +=
            usize::from(field(clean_first, "pkm_slots")? != field(changed_first, "pkm_slots")?);
        self.first_coarse_scores += usize::from(
            field(clean_first, "coarse_scores")? != field(changed_first, "coarse_scores")?,
        );
        self.first_carry_words += usize::from(
            field(field(native, "after")?, "carry")? != field(field(changed, "after")?, "carry")?,
        );
        self.first_selected_cell += usize::from(clean.selected != transplant.selected);
        self.first_emitted_id += usize::from(clean.first != transplant.first);
        self.emitted_sequence += usize::from(clean.emitted != transplant.emitted);
        for (old, new, improved, worsened, gained, lost) in [
            (
                &clean.own,
                &transplant.own,
                &mut self.own_forced_cell_rank_improved,
                &mut self.own_forced_cell_rank_worsened,
                &mut self.own_cell_gained_admission,
                &mut self.own_cell_lost_admission,
            ),
            (
                &clean.other,
                &transplant.other,
                &mut self.other_forced_cell_rank_improved,
                &mut self.other_forced_cell_rank_worsened,
                &mut self.other_cell_gained_admission,
                &mut self.other_cell_lost_admission,
            ),
        ] {
            *improved +=
                usize::from(new.diagnostic_within_cell_rank < old.diagnostic_within_cell_rank);
            *worsened +=
                usize::from(new.diagnostic_within_cell_rank > old.diagnostic_within_cell_rank);
            *gained += usize::from(new.admitted && !old.admitted);
            *lost += usize::from(!new.admitted && old.admitted);
        }
        Ok(())
    }
}

fn by_id(
    rows: Vec<Value>,
    expected: impl Iterator<Item = usize>,
) -> Result<BTreeMap<usize, Value>> {
    let mut result = BTreeMap::new();
    for row in rows {
        let id: usize = read(&row, "pair_id")?;
        require(result.insert(id, row).is_none(), "duplicate runner pair ID")?;
    }
    require(
        result.keys().copied().eq(expected),
        "unexpected or missing runner pair IDs",
    )?;
    Ok(result)
}

#[derive(Default, Serialize)]
struct Summary {
    #[serde(skip)]
    arms: [ArmSummary; 3],
    swap_vs_native: Changes,
    zero_vs_native: Changes,
    both_members_native_own_and_swap_other: usize,
    native_sum_own_and_swap_sum_other: usize,
    recipient_prefill_carry_differs_across_pair: usize,
    recipient_shift_histories_equal_across_pair: usize,
    recipient_tokens_seen_equal_across_pair: usize,
    parity_rollouts: usize,
    parity_emitted_ids: usize,
    length_strata: BTreeMap<String, usize>,
}

impl Summary {
    fn add(
        &mut self,
        pair: &Pair,
        gpu: &Value,
        ordinary: Option<&Value>,
        model: &ButterflyV3Model,
    ) -> Result<()> {
        increment(&mut self.length_strata, pair.length_stratum.clone());
        let a = field(gpu, "a")?;
        let b = field(gpu, "b")?;
        let sa: State = read(a, "before")?;
        let sb: State = read(b, "before")?;
        self.recipient_prefill_carry_differs_across_pair += usize::from(sa.carry != sb.carry);
        self.recipient_shift_histories_equal_across_pair +=
            usize::from(sa.shift_hist == sb.shift_hist);
        self.recipient_tokens_seen_equal_across_pair +=
            usize::from(sa.tokens_seen == sb.tokens_seen);

        let recount_side = |side: &str,
                            gpu_side: &Value,
                            row: &Row,
                            rival: &Row,
                            own_state: &State,
                            donor: &State| {
            let run = || -> Result<Vec<RecountedRow>> {
                matches(
                    gpu_side,
                    "row",
                    &json!({"prefix": row.prefix, "expected": row.expected}),
                )?;
                validate_state(own_state, row.prefix.len() - 1, model)?;
                let ordinary_side = ordinary.map(|record| field(record, side)).transpose()?;
                if let Some(record) = ordinary_side {
                    matches(record, "before", field(gpu_side, "before")?)?;
                }
                ARMS.iter()
                    .map(|label| {
                        let raw = field(gpu_side, label)?;
                        let replay = recount_arm(raw, label, row, rival, own_state, donor, model)
                            .map_err(|error| format!("{label}: {error}"))?;
                        if let Some(record) = ordinary_side {
                            verify_ordinary(raw, field(record, label)?, &replay, model)
                                .map_err(|error| format!("{label} ordinary parity: {error}"))?;
                        }
                        Ok(replay)
                    })
                    .collect()
            };
            run().map_err(|error| format!("side {side}: {error}"))
        };
        let a_rows = recount_side("a", a, &pair.a, &pair.b, &sa, &sb)?;
        let b_rows = recount_side("b", b, &pair.b, &pair.a, &sb, &sa)?;
        let mut both = [false; 2];
        let mut sum_order = [Ordering::Equal; 2];
        for (index, label) in ARMS.into_iter().enumerate() {
            let (a_arm, b_arm) = (&a_rows[index], &b_rows[index]);
            let assignment = field(field(gpu, "assignment")?, label)?;
            let (a_order, b_order) = self.arms[index].record(a_arm, b_arm, pair, assignment)?;
            if index < both.len() {
                let desired = [Ordering::Greater, Ordering::Less][index];
                both[index] = a_order == desired && b_order == desired;
                sum_order[index] = (a_arm.own.target_mass_q48 + b_arm.own.target_mass_q48)
                    .cmp(&(a_arm.other.target_mass_q48 + b_arm.other.target_mass_q48));
            }
            if index != 0 {
                let summary = [&mut self.swap_vs_native, &mut self.zero_vs_native][index - 1]
                    .record(field(a, "native")?, field(a, label)?, &a_rows[0], a_arm);
                summary?;
                [&mut self.swap_vs_native, &mut self.zero_vs_native][index - 1].record(
                    field(b, "native")?,
                    field(b, label)?,
                    &b_rows[0],
                    b_arm,
                )?;
            }
            if ordinary.is_some() {
                self.parity_rollouts += 2;
                self.parity_emitted_ids += a_arm.emitted.len() + b_arm.emitted.len();
            }
        }
        self.both_members_native_own_and_swap_other += usize::from(both == [true; 2]);
        self.native_sum_own_and_swap_sum_other +=
            usize::from(sum_order == [Ordering::Greater, Ordering::Less]);
        Ok(())
    }
}

fn require_sha(label: &str, bytes: &[u8], pin: &str) -> Result<()> {
    let observed = sha(bytes);
    require(
        observed == pin,
        &format!("{label} SHA-256 mismatch; observed {observed}"),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [artifact, pairs, raw, ordinary] = args.as_slice() else {
        return Err(
            "usage: reconcile ARTIFACT PAIRS.jsonl GPU_RAW.jsonl ORDINARY_RAW.jsonl".into(),
        );
    };
    let artifact = fs::read(artifact)?;
    let pairs = fs::read(pairs)?;
    let raw = fs::read(raw)?;
    let ordinary = fs::read(ordinary)?;
    require_sha("artifact", &artifact, ARTIFACT_SHA)?;
    require_sha("pair tape", &pairs, PAIRS_SHA)?;
    let model = ButterflyV3Model::from_bytes(&artifact)?;
    require(model.cfg.vocab == 4096, "unexpected artifact vocabulary")?;
    let pairs = parse_lines::<Pair>(&pairs)?;
    require(
        pairs.iter().map(|pair| pair.pair_id).eq(1..=PAIRS),
        "misordered, duplicate or missing frozen pair IDs",
    )?;
    let gpu = by_id(parse_lines(&raw)?, 1..=PAIRS)?;
    let parity = by_id(parse_lines(&ordinary)?, PARITY_IDS.into_iter())?;
    let mut groups = std::array::from_fn(|_| HashSet::new());
    let mut prefixes = HashSet::new();
    let mut summary = Summary::default();
    for pair in &pairs {
        let run = || -> Result<()> {
            validate_pair(pair, model.cfg.vocab, &mut groups, &mut prefixes)?;
            summary.add(
                pair,
                gpu.get(&pair.pair_id).ok_or("missing GPU pair")?,
                parity.get(&pair.pair_id),
                &model,
            )
        };
        let mut run = run;
        run().map_err(|error| format!("pair {}: {error}", pair.pair_id))?;
    }
    let mut arms = BTreeMap::new();
    for (label, mut arm) in ARMS.into_iter().zip(std::mem::take(&mut summary.arms)) {
        arm.finish();
        arms.insert(label, arm);
    }
    let receipt = json!({
        "status": "PASS",
        "pairs": pairs.len(),
        "replies_per_arm": prefixes.len(),
        "artifact_sha256": ARTIFACT_SHA,
        "pairs_sha256": PAIRS_SHA,
        "gpu_raw_sha256": sha(&raw),
        "ordinary_raw_sha256": sha(&ordinary),
        "eos_original_id": EOS,
        "bos_original_id": BOS,
        "rollout_cap": CAP,
        "arms": arms,
        "summary": summary,
        "ordinary_parity_ids": PARITY_IDS,
        "ordinary_parity_scope": "Prefill and first-transition carry/shift/tokens, hidden, PKM, coarse scores, first target diagnostics, and every self-fed original ID with its slot-derived route. Q48 has no ordinary-JSONL comparator.",
        "rank_median_representation": "[lower_middle, upper_middle]",
    });
    serde_json::to_writer_pretty(io::stdout().lock(), &receipt)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_and_disagreement_accounting_catches_bos_and_premature_eos() {
        let expected = [5, 7, EOS];
        assert_eq!(
            outcome(&expected, &expected, stop(&expected, 4096).unwrap()).verdict,
            "exact_through_eos"
        );
        for emitted in [[5, BOS], [5, EOS]] {
            let result = outcome(&emitted, &expected, stop(&emitted, 4096).unwrap());
            assert_eq!(result.first_mismatch_index, Some(1));
            assert_eq!(result.verdict, "first_wrong");
        }
        assert!(stop(&[5, EOS, 7], 4096).is_err());
        assert!(stop(&[5, BOS, 7], 4096).is_err());
        assert!(stop(&[5, 7], 4096).is_err());
        assert!(stop(&[5, 4096], 4096).is_err());
        let capped = vec![5; CAP];
        let mut long = vec![5; CAP + 1];
        long.push(EOS);
        assert_eq!(
            outcome(&capped, &long, stop(&capped, 4096).unwrap()).verdict,
            "censored"
        );
    }

    #[test]
    fn rejects_wrong_donor_and_recipient_state_mutations() {
        let state = |carry, shift, tokens_seen| State {
            carry: vec![vec![carry]],
            shift_hist: vec![vec![shift]],
            tokens_seen,
        };
        let recipient = state(7, 3, 50);
        let donor = state(9, 8, 80);
        assert!(validate_transplant(&state(9, 3, 50), &recipient, &donor, "swap").is_ok());
        assert!(validate_transplant(&state(7, 3, 50), &recipient, &donor, "swap").is_err());
        assert!(validate_transplant(&donor, &recipient, &donor, "swap").is_err());
        assert!(validate_transplant(&state(9, 3, 51), &recipient, &donor, "swap").is_err());
        assert!(validate_transplant(&state(1, 3, 50), &recipient, &donor, "zero").is_err());
    }

    #[test]
    fn rejects_duplicate_runner_identity_and_missing_null() {
        let one = json!({"pair_id": 1});
        assert!(by_id(vec![one.clone(), one.clone()], 1..=2).is_err());
        assert!(by_id(vec![one.clone()], 1..=2).is_err());
        assert!(by_id(vec![one.clone()], 1..=1).is_ok());
        assert!(matches(&one, "first_error", &Value::Null).is_err());
    }

    #[test]
    fn planted_first_transition_and_later_route_corruption_fail_parity() {
        use uor_r4_native_vector::butterfly_rotor::LossKind;
        use uor_r4_native_vector::butterfly_rotor_v3::ButterflyV3Config;

        // Build only a slot map. This fixture never executes the model.
        let model = ButterflyV3Model::new(ButterflyV3Config {
            vocab: 64,
            dim: 16,
            layers: 2,
            ffn_branches: 2,
            pkm_sub_keys: 4,
            pkm_key_dim: 4,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 2 },
            seed: 734,
        })
        .unwrap();
        let (selected, fine) = route(&model, 5);
        let (terminal_cell, terminal_fine) = route(&model, EOS);
        let target = |token| {
            let (cell, within_cell_slot) = route(&model, token);
            Target {
                token,
                cell,
                within_cell_slot,
                admitted: cell == selected,
                coarse_score_q15: 123,
                fine_score_q15: 47,
                coarse_score_minus_winner_q15: 0,
                fine_score_minus_best_other_q15: Some(4),
                diagnostic_within_cell_rank: 1,
                admitted_within_cell_rank: (cell == selected).then_some(1),
                target_mass_q48: 99,
            }
        };
        let row = RecountedRow {
            first: 5,
            selected,
            own: target(5),
            other: target(7),
            emitted: vec![5, EOS],
            stop: "eos",
            own_outcome: outcome(&[5, EOS], &[5, EOS], "eos"),
            other_outcome: outcome(&[5, EOS], &[7, EOS], "eos"),
        };
        let mut ordinary = json!({
            "first": {"hidden_q15": [31], "pkm_slots": [[3, 6]], "coarse_scores": [3, 4]},
            "after": {"carry": [[11], [19]], "shift_hist": [[12, 13]], "tokens_seen": 4},
            "emitted": row.emitted,
            "stop": row.stop,
            "own_outcome": row.own_outcome,
            "rival_outcome": row.other_outcome,
            "trace": [
                {"index": 0, "token": 5, "greedy_coarse": selected, "greedy_fine": fine},
                {"index": 1, "token": EOS, "greedy_coarse": terminal_cell, "greedy_fine": terminal_fine},
            ],
        });
        for (key, t) in [("own", &row.own), ("rival", &row.other)] {
            ordinary["trace"][0][key] = json!({
                "target": t.token, "cell": t.cell, "coarse_score": t.coarse_score_q15,
                "coarse_score_minus_winner": t.coarse_score_minus_winner_q15,
                "entered_target_cell": t.admitted, "target_fine": t.within_cell_slot,
                "target_score": t.fine_score_q15, "target_minus_best_other": t.fine_score_minus_best_other_q15,
                "target_fine_rank": t.diagnostic_within_cell_rank,
            });
        }
        let mut gpu = ordinary.clone();
        gpu["donor_outcome"] = json!(row.other_outcome);
        assert!(verify_ordinary(&gpu, &ordinary, &row, &model).is_ok());
        for pointer in [
            "/first/hidden_q15/0",
            "/first/pkm_slots/0/0",
            "/first/coarse_scores/1",
            "/after/carry/1/0",
            "/after/shift_hist/0/0",
            "/emitted/1",
            "/trace/0/rival/target_score",
            "/trace/1/greedy_fine",
        ] {
            let mut corrupted = ordinary.clone();
            *corrupted.pointer_mut(pointer).unwrap() = json!(4321);
            assert!(
                verify_ordinary(&gpu, &corrupted, &row, &model).is_err(),
                "{pointer}"
            );
        }
    }
}
