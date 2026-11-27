use super::prepare::Key;
use super::*;
use serde_json::{json, Value};

const ARMS: [&str; 4] = ["native", "correct", "swapped", "supplied_next"];
const CAP: usize = 4;
const PANEL_SHA256: &str = "56091ad40d7e5da6f3d6742e02f04e37cdb01171fc3fddb4acba9bcf4f11b876";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Source {
    Copy,
    SuppliedPeriod,
    Native,
    Correct,
    Swapped,
    SuppliedNext,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
struct Step {
    committed_id: u32,
    native_proposal: u32,
    selected_source: Source,
    copied: bool,
    claim_pos: Option<u32>,
    read_pos: Option<u32>,
    #[serde(flatten)]
    trace: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct Trajectory {
    steps: Vec<Step>,
    terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ThirdTrace {
    writer_id: u32,
    gate_id: u32,
    pointer_predecessor: Option<(usize, u32)>,
    pointer_successor: Option<(usize, u32)>,
}

#[derive(Deserialize)]
struct Observation {
    #[serde(flatten)]
    row: Row,
    arms: BTreeMap<String, Trajectory>,
    third_trace: ThirdTrace,
}

#[derive(Clone, Serialize)]
struct Wrong {
    position: usize,
    expected: Option<u32>,
    actual: Option<u32>,
}

fn first_wrong(actual: &[u32], expected: &[u32]) -> Option<Wrong> {
    (0..actual.len().max(expected.len())).find_map(|pos| {
        let expected = expected.get(pos).copied();
        let actual = actual.get(pos).copied();
        (actual != expected).then_some(Wrong {
            position: pos + 1,
            expected,
            actual,
        })
    })
}

#[derive(Clone, Serialize)]
struct Outcome {
    tokens: Vec<u32>,
    decoded: String,
    pieces: Vec<String>,
    selection: Vec<Source>,
    exact: bool,
    terminal: bool,
    eos_source: Option<Source>,
    first_wrong: Option<Wrong>,
}

fn tally(outcome: &Trajectory, reference: &[u32], tokenizer: &ByteBpeTokenizer) -> Result<Outcome> {
    let tokens = outcome
        .steps
        .iter()
        .map(|step| step.committed_id)
        .collect::<Vec<_>>();
    let terminal = tokens.last() == Some(&EOS);
    let first_wrong = first_wrong(&tokens, reference);
    require(
        (3..=CAP).contains(&tokens.len())
            && (tokens.len() == CAP || terminal)
            && !tokens[..tokens.len() - 1].contains(&EOS)
            && tokens
                .iter()
                .all(|&id| (id as usize) < tokenizer.vocab_size())
            && outcome.terminal == terminal
            && outcome.first_mismatch == first_wrong.as_ref().map(|wrong| wrong.position),
        "malformed, post-EOS, truncated or misgraded trajectory",
    )?;
    let eos_source = outcome
        .steps
        .last()
        .filter(|step| step.committed_id == EOS)
        .map(|step| step.selected_source);
    Ok(Outcome {
        decoded: tokenizer.decode(&tokens[..tokens.len() - usize::from(terminal)]),
        pieces: tokens.iter().map(|&id| decode_id(tokenizer, id)).collect(),
        selection: outcome
            .steps
            .iter()
            .map(|step| step.selected_source)
            .collect(),
        exact: terminal && first_wrong.is_none(),
        terminal,
        eos_source,
        first_wrong,
        tokens,
    })
}

fn decode_id(tokenizer: &ByteBpeTokenizer, id: u32) -> String {
    if id == EOS {
        "<EOS>".to_owned()
    } else {
        tokenizer.decode(&[id])
    }
}

#[derive(Serialize)]
struct Graded<'a> {
    key: &'a Key,
    family: &'a str,
    prefix: &'a [u32],
    gold_pos: u32,
    target: u32,
    next_pos: u32,
    next_id: u32,
    expected: &'a [u32],
    native_proposal_after_word: u32,
    native_proposal_after_period: u32,
    gate_after_period: u32,
    pointer_predecessor: Option<(usize, u32)>,
    pointer_successor: Option<(usize, u32)>,
    arms: BTreeMap<&'static str, Outcome>,
}

fn grade<'a>(
    row: &'a Row,
    peer: &Row,
    key: &'a Key,
    observed: &Observation,
    tokenizer: &ByteBpeTokenizer,
) -> Result<Graded<'a>> {
    require(
        observed.row == *row
            && observed.arms.len() == ARMS.len()
            && ARMS.iter().all(|name| observed.arms.contains_key(*name)),
        format!("observation differs from sealed row {}", row.pair_id),
    )?;
    let exact_binding = row.prefix.get(row.gold_pos as usize) == Some(&row.target)
        && row.prefix.get(row.next_pos as usize) == Some(&row.next_id)
        && row.target != row.next_id
        && row.gold_pos != row.next_pos;
    let question = row
        .prefix
        .iter()
        .position(|id| *id == QUESTION)
        .ok_or("missing question")?;
    let raw_user = format!("{}\n\n{} {}", key.context, key.question, key.directive);
    let expected = if row.side == 0 {
        vec![row.target, PERIOD, EOS]
    } else {
        vec![row.target, PERIOD, row.next_id, EOS]
    };
    require(
        key.user == raw_user
            && key.instruction_final_period == key.directive.ends_with('.')
            && key.instruction_final_period
                == (row.prefix.iter().rev().nth(MARKER.len()) == Some(&PERIOD))
            && tokenizer.decode(&row.expected[..row.expected.len().saturating_sub(1)])
                == format!(" {}", key.reply)
            && exact_binding
            && row.expected == expected
            && row.prefix[..=question]
                == *peer
                    .prefix
                    .get(..=question)
                    .ok_or("missing peer question")?
            && row.prefix != peer.prefix
            && row.family == peer.family
            && row.target == peer.target
            && row.next_id == peer.next_id
            && row.gold_pos == peer.gold_pos
            && row.next_pos == peer.next_pos,
        format!(
            "malformed paired source, key or reference for {}",
            row.pair_id
        ),
    )?;
    let origin = &observed.arms["native"];
    require(origin.steps.len() >= 3, "missing native third event")?;
    let opening = &origin.steps[0];
    let period = &origin.steps[1];
    let next_native = &origin.steps[2];
    require(
        opening.copied
            && opening.read_pos == Some(row.gold_pos)
            && opening.claim_pos == Some(row.gold_pos)
            && opening.committed_id == row.target
            && opening.selected_source == Source::Copy
            && period.committed_id == PERIOD
            && period.selected_source == Source::SuppliedPeriod
            && observed.third_trace.writer_id == next_native.native_proposal,
        format!(
            "copy, fixed period or native proposal drift in {}",
            row.pair_id
        ),
    )?;
    let trace = &observed.third_trace;
    let mut comparison = BTreeMap::new();
    let mut graded = BTreeMap::new();
    for (name, selected_id, source) in [
        ("native", next_native.native_proposal, Source::Native),
        (
            "correct",
            if row.side == 0 {
                EOS
            } else {
                next_native.native_proposal
            },
            if row.side == 0 {
                Source::Correct
            } else {
                Source::Native
            },
        ),
        (
            "swapped",
            if peer.side == 0 {
                EOS
            } else {
                next_native.native_proposal
            },
            if peer.side == 0 {
                Source::Swapped
            } else {
                Source::Native
            },
        ),
        ("supplied_next", row.expected[2], Source::SuppliedNext),
    ] {
        let arm = &observed.arms[name];
        let result = tally(arm, &row.expected, tokenizer)?;
        let [first, second, third, rest @ ..] = arm.steps.as_slice() else {
            return Err("missing shared first events".into());
        };
        require(
            [first, second] == [opening, period]
                && third.native_proposal == next_native.native_proposal
                && third.committed_id == selected_id
                && third.selected_source == source
                && third.trace == next_native.trace
                && third.read_pos == next_native.read_pos
                && third.claim_pos == next_native.claim_pos
                && third.copied == next_native.copied
                && rest.iter().all(|step| {
                    step.selected_source == Source::Native
                        && step.committed_id == step.native_proposal
                }),
            format!(
                "invalid shared prefix, intervention or self-fed successor in {} {name}",
                row.pair_id
            ),
        )?;
        if let Some(previous) = comparison.insert(third.committed_id, rest) {
            require(
                previous == rest,
                "identical actual third ID produced different subsequent event",
            )?;
        }
        graded.insert(name, result);
    }
    Ok(Graded {
        key,
        family: &row.family,
        prefix: &row.prefix,
        gold_pos: row.gold_pos,
        target: row.target,
        next_pos: row.next_pos,
        next_id: row.next_id,
        expected: &row.expected,
        native_proposal_after_word: period.native_proposal,
        native_proposal_after_period: next_native.native_proposal,
        gate_after_period: trace.gate_id,
        pointer_predecessor: trace.pointer_predecessor,
        pointer_successor: trace.pointer_successor,
        arms: graded,
    })
}

#[derive(Default, Serialize)]
struct Totals {
    rows: usize,
    pairs: usize,
    third_correct_by_side: [usize; 2],
    third_correct_pairs: usize,
    exact_by_side: [usize; 2],
    exact_pairs: usize,
    natural_eos: usize,
    supplied_eos: usize,
    exact_natural_eos: usize,
    exact_supplied_eos: usize,
    reached_cap_without_eos: usize,
    retained_native_exact_by_side: [usize; 2],
    gained_vs_native_exact_by_side: [usize; 2],
    lost_vs_native_exact_by_side: [usize; 2],
    retained_native_continuation_word: usize,
    first_wrong_by_position: BTreeMap<usize, usize>,
    first_wrong_id_transitions: BTreeMap<String, usize>,
    after_word_observed: BTreeMap<u32, usize>,
    after_word_native_proposals: BTreeMap<u32, usize>,
    after_period_observed: BTreeMap<u32, usize>,
    after_period_native_proposals: BTreeMap<u32, usize>,
    fourth_after_correct_continuation: BTreeMap<u32, usize>,
    pair_third_direction: BTreeMap<String, usize>,
    pairs_same_third_id: usize,
}

impl Totals {
    fn add(&mut self, arm: &str, row: &Graded<'_>) {
        let this = &row.arms[arm];
        let native = &row.arms["native"];
        let side = usize::from(row.key.side);
        self.rows += 1;
        self.third_correct_by_side[side] += usize::from(this.tokens[2] == row.expected[2]);
        self.exact_by_side[side] += usize::from(this.exact);
        match this.eos_source {
            Some(Source::Native) => {
                self.natural_eos += 1;
                self.exact_natural_eos += usize::from(this.exact);
            }
            Some(_) => {
                self.supplied_eos += 1;
                self.exact_supplied_eos += usize::from(this.exact);
            }
            None => self.reached_cap_without_eos += 1,
        }
        self.retained_native_exact_by_side[side] += usize::from(this.exact && native.exact);
        self.gained_vs_native_exact_by_side[side] += usize::from(this.exact && !native.exact);
        self.lost_vs_native_exact_by_side[side] += usize::from(!this.exact && native.exact);
        if side == 1 && this.tokens[2] == row.next_id {
            self.retained_native_continuation_word += usize::from(native.tokens[2] == row.next_id);
            *self
                .fourth_after_correct_continuation
                .entry(this.tokens[3])
                .or_default() += 1;
        }
        if let Some(wrong) = &this.first_wrong {
            *self
                .first_wrong_by_position
                .entry(wrong.position)
                .or_default() += 1;
            let id = |value: Option<u32>| value.map_or("missing".to_owned(), |id| id.to_string());
            let index = format!(
                "{}:{}->{}",
                wrong.position,
                id(wrong.expected),
                id(wrong.actual)
            );
            *self.first_wrong_id_transitions.entry(index).or_default() += 1;
        }
        *self.after_word_observed.entry(this.tokens[1]).or_default() += 1;
        *self
            .after_word_native_proposals
            .entry(row.native_proposal_after_word)
            .or_default() += 1;
        *self
            .after_period_observed
            .entry(this.tokens[2])
            .or_default() += 1;
        *self
            .after_period_native_proposals
            .entry(row.native_proposal_after_period)
            .or_default() += 1;
    }

    fn pair(&mut self, arm: &str, stop: &Graded<'_>, cont: &Graded<'_>) {
        let stop_arm = &stop.arms[arm];
        let cont_arm = &cont.arms[arm];
        self.pairs += 1;
        self.exact_pairs += usize::from(stop_arm.exact && cont_arm.exact);
        self.add(arm, stop);
        self.add(arm, cont);

        let stop_id = stop_arm.tokens[2];
        let cont_id = cont_arm.tokens[2];
        self.third_correct_pairs += usize::from(stop_id == EOS && cont_id == cont.next_id);
        self.pairs_same_third_id += usize::from(stop_id == cont_id);
        let direction = |id| {
            if id == EOS {
                "eos"
            } else if id == cont.next_id {
                "factual_next"
            } else {
                "other"
            }
        };
        let key = format!("{}->{}", direction(stop_id), direction(cont_id));
        *self.pair_third_direction.entry(key).or_default() += 1;
    }
}

fn group<'a>(
    rows: impl Iterator<Item = (&'a Graded<'a>, &'a Graded<'a>)>,
) -> BTreeMap<&'static str, Totals> {
    let mut result = ARMS
        .into_iter()
        .map(|arm| (arm, Totals::default()))
        .collect::<BTreeMap<_, _>>();
    for (stop, cont) in rows {
        for (arm, tally) in &mut result {
            tally.pair(arm, stop, cont);
        }
    }
    result
}

fn pair_values<'a>(
    graded: &'a BTreeMap<(String, u8), Graded<'a>>,
) -> impl Iterator<Item = (&'a Graded<'a>, &'a Graded<'a>)> {
    graded.values().filter(|row| row.key.side == 0).map(|stop| {
        let peer_key = (stop.key.pair_id.clone(), 1);
        (stop, &graded[&peer_key])
    })
}

fn example<'a>(stop: &'a Graded<'a>, cont: &'a Graded<'a>) -> Value {
    json!({"pair_id": stop.key.pair_id, "stop": stop, "continue": cont})
}

fn examples(graded: &BTreeMap<(String, u8), Graded<'_>>) -> Value {
    let mut clean = BTreeMap::new();
    let mut conditioned = BTreeMap::new();
    for (stop, cont) in pair_values(graded) {
        let stratum = if stop.key.instruction_final_period {
            "final_period"
        } else {
            "unpunctuated"
        };
        // These IDs are selected by frozen input properties and ordering only.
        if !stop.key.opening_collision && !stop.key.next_collision {
            clean
                .entry(format!("{}:{stratum}", stop.family))
                .or_insert_with(|| example(stop, cont));
        }
        for arm in ARMS {
            let a = &stop.arms[arm];
            let b = &cont.arms[arm];
            let condition = if a.exact && b.exact {
                "both_exact"
            } else if b.exact {
                "continue_exact"
            } else if a.exact {
                "stop_exact"
            } else if b.tokens[2] == cont.next_id {
                "next_word_exact_but_unfinished"
            } else {
                "neither_exact"
            };
            conditioned
                .entry(format!("{arm}:{stratum}:{condition}"))
                .or_insert_with(|| example(stop, cont));
        }
    }
    json!({
        "input_selected_noncollision_by_family_and_directive_punctuation": clean,
        "outcome_selected_by_arm_and_directive_punctuation": conditioned,
    })
}

pub(super) fn run(
    tokenizer: &ByteBpeTokenizer,
    panel: &Path,
    split: &str,
    observations: &Path,
    output: &Path,
) -> Result<()> {
    let expected_rows = match split {
        "held" => 400,
        "finite" => 32,
        _ => return Err("expected held or finite".into()),
    };
    require(
        hash(&panel.join("SHA256SUMS"))? == PANEL_SHA256,
        "sealed panel manifest changed",
    )?;
    verify_manifest(panel, &panel.join("SHA256SUMS"))?;
    let directory = panel.join(split);
    verify_manifest(&directory, &directory.join("SHA256SUMS"))?;
    let rows = read_jsonl::<Row>(&directory.join("rows.jsonl"))?;
    let keys = read_jsonl::<Key>(&directory.join("keys.jsonl"))?;
    let events = read_jsonl::<Observation>(observations)?;
    require(
        rows.len() == expected_rows && keys.len() == rows.len() && events.len() == rows.len(),
        "row/key/observation count differs",
    )?;
    let index = rows
        .iter()
        .map(|row| ((row.pair_id.clone(), row.side), row))
        .collect::<BTreeMap<_, _>>();
    let keys = keys
        .iter()
        .map(|key| ((key.pair_id.clone(), key.side), key))
        .collect::<BTreeMap<_, _>>();
    require(
        index.len() == rows.len() && index.keys().eq(keys.keys()),
        "duplicate or different keys in tape",
    )?;

    let mut graded = BTreeMap::new();
    for observed in &events {
        let id = (observed.row.pair_id.clone(), observed.row.side);
        require(id.1 <= 1, "invalid panel side")?;
        let other = (id.0.clone(), 1 - id.1);
        let row = index
            .get(&id)
            .ok_or("observation row missing from frozen panel")?;
        let peer = index.get(&other).ok_or("row missing matched side")?;
        let key = keys.get(&id).ok_or("row missing private key")?;
        let peer_key = keys.get(&other).ok_or("peer missing private key")?;
        require(
            key.instruction_final_period == peer_key.instruction_final_period
                && key.context == peer_key.context
                && key.question == peer_key.question
                && key.queried_person == peer_key.queried_person
                && key.rival_person == peer_key.rival_person
                && key.opening_word == peer_key.opening_word
                && key.next_word == peer_key.next_word
                && key.authorized_fact == peer_key.authorized_fact
                && key.rival_fact == peer_key.rival_fact
                && key.directive != peer_key.directive,
            "matched keys differ outside the final directive and answer",
        )?;
        let result = grade(row, peer, key, observed, tokenizer)?;
        require(graded.insert(id, result).is_none(), "duplicate observation")?;
    }
    require(
        graded.keys().eq(index.keys()),
        "incomplete observation coverage",
    )?;

    let families = rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    let mut by_family = BTreeMap::new();
    for family in families {
        by_family.insert(
            family,
            group(pair_values(&graded).filter(|(stop, _)| stop.family == family)),
        );
    }
    let (mut punctuation, mut collision) = (BTreeMap::new(), BTreeMap::new());
    for marked in [false, true] {
        punctuation.insert(
            if marked {
                "final_period"
            } else {
                "unpunctuated"
            },
            group(
                pair_values(&graded)
                    .filter(|(stop, _)| stop.key.instruction_final_period == marked),
            ),
        );
        collision.insert(
            if marked {
                "opening_or_next_collision"
            } else {
                "ordinary_values"
            },
            group(pair_values(&graded).filter(|(stop, _)| {
                (stop.key.opening_collision || stop.key.next_collision) == marked
            })),
        );
    }
    let emitted_ids = graded
        .values()
        .flat_map(|row| {
            row.arms
                .values()
                .flat_map(|outcome| outcome.tokens.iter().copied())
                .chain([
                    row.native_proposal_after_word,
                    row.native_proposal_after_period,
                ])
        })
        .collect::<BTreeSet<_>>();
    let summary = json!({
        "schema": "codex-4320.postperiod-score/1",
        "split": split,
        "tape_sha256": hash(&directory.join("rows.jsonl"))?,
        "keys_sha256": hash(&directory.join("keys.jsonl"))?,
        "observations_sha256": hash(observations)?,
        "side_order": ["stop", "continue"], "response_cap": CAP,
        "vocab_pieces": emitted_ids.into_iter().map(|id| (id, decode_id(tokenizer, id))).collect::<BTreeMap<_, _>>(),
        "arms": group(pair_values(&graded)), "families": by_family,
        "by_instruction_final_punctuation": punctuation, "by_collision": collision,
    });
    fs::create_dir(output)?;
    let mut aggregate = fresh_writer(&output.join("aggregate.json"))?;
    serde_json::to_writer_pretty(&mut aggregate, &summary)?;
    writeln!(aggregate)?;
    aggregate.flush()?;
    let mut decoded = fresh_writer(&output.join("decoded.jsonl"))?;
    for row in graded.values() {
        serde_json::to_writer(&mut decoded, row)?;
        writeln!(decoded)?;
    }
    decoded.flush()?;
    let mut samples = fresh_writer(&output.join("examples.json"))?;
    serde_json::to_writer_pretty(&mut samples, &examples(&graded))?;
    writeln!(samples)?;
    samples.flush()?;
    write_manifest(
        output,
        &["aggregate.json", "decoded.jsonl", "examples.json"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planted_step(id: u32, proposal: u32, source: Source, position: Option<u32>) -> Value {
        json!({
            "committed_id": id, "native_proposal": proposal, "selected_source": source,
            "copied": position.is_some(), "claim_pos": position, "read_pos": position,
        })
    }

    #[test]
    #[ignore = "requires the pinned tokenizer and sealed finite tape on Wick"]
    fn frozen_finite_planted_score() -> Result<()> {
        let runs = std::env::var("POSTPERIOD_RUNS")?;
        let runs = Path::new(&runs);
        let vocab = tokenizer(&runs.join(history::TOKENIZER))?;
        let experiment = runs.join("20261127-codex-postperiod");
        let panel = experiment.join("panel-v1");
        let out = experiment
            .join("staging")
            .join(format!("planted-score-{}", std::process::id()));
        fs::create_dir(&out)?;
        let inputs = read_jsonl::<Row>(&panel.join("finite/rows.jsonl"))?;
        let raw = out.join("planted.jsonl");
        let mut raw_file = fresh_writer(&raw)?;
        let mut observations = Vec::new();
        for (i, row) in inputs.iter().enumerate() {
            let pair = i / 2;
            let native = if pair % 4 == 0 { row.next_id } else { 73 };
            let mut value = serde_json::to_value(row)?;
            let mut arms = BTreeMap::new();
            for (arm, id, source) in [
                ("native", native, Source::Native),
                (
                    "correct",
                    if row.side == 0 { EOS } else { native },
                    if row.side == 0 {
                        Source::Correct
                    } else {
                        Source::Native
                    },
                ),
                (
                    "swapped",
                    if row.side == 1 { EOS } else { native },
                    if row.side == 1 {
                        Source::Swapped
                    } else {
                        Source::Native
                    },
                ),
                ("supplied_next", row.expected[2], Source::SuppliedNext),
            ] {
                let mut steps = vec![
                    planted_step(row.target, 73, Source::Copy, Some(row.gold_pos)),
                    planted_step(PERIOD, 73, Source::SuppliedPeriod, None),
                    planted_step(id, native, source, None),
                ];
                let mut ids = vec![row.target, PERIOD, id];
                if id != EOS {
                    let fourth = if id == row.next_id && pair % 2 == 0 {
                        EOS
                    } else {
                        73
                    };
                    steps.push(planted_step(fourth, fourth, Source::Native, None));
                    ids.push(fourth);
                }
                arms.insert(arm, json!({
                    "steps": steps, "terminal": ids.last() == Some(&EOS),
                    "first_mismatch": first_wrong(&ids, &row.expected).map(|wrong| wrong.position),
                }));
            }
            value["arms"] = serde_json::to_value(arms)?;
            value["third_trace"] = json!({"writer_id": native, "gate_id": native});
            serde_json::to_writer(&mut raw_file, &value)?;
            writeln!(raw_file)?;
            observations.push(value);
        }
        raw_file.flush()?;
        let analysis = out.join("analysis");
        run(&vocab, &panel, "finite", &raw, &analysis)?;
        let summary: Value = serde_json::from_reader(File::open(analysis.join("aggregate.json"))?)?;
        assert_eq!(summary["arms"]["native"]["exact_by_side"], json!([0, 4]));
        assert_eq!(summary["arms"]["correct"]["exact_pairs"], json!(4));
        assert_eq!(summary["arms"]["swapped"]["exact_pairs"], json!(0));
        assert_eq!(summary["arms"]["supplied_next"]["exact_pairs"], json!(8));
        assert_eq!(
            summary["arms"]["supplied_next"]["exact_natural_eos"],
            json!(8)
        );
        assert_eq!(
            summary["arms"]["supplied_next"]["exact_supplied_eos"],
            json!(16)
        );
        assert_eq!(
            summary["arms"]["supplied_next"]["reached_cap_without_eos"],
            json!(8)
        );
        assert_eq!(
            summary["arms"]["supplied_next"]["retained_native_continuation_word"],
            json!(4)
        );
        assert_eq!(
            summary["arms"]["supplied_next"]["pair_third_direction"],
            json!({"eos->factual_next": 16})
        );

        let bad = out.join("false-terminal.jsonl");
        let mut bad_file = fresh_writer(&bad)?;
        observations[2]["arms"]["native"]["terminal"] = json!(true);
        for value in &observations {
            serde_json::to_writer(&mut bad_file, value)?;
            writeln!(bad_file)?;
        }
        bad_file.flush()?;
        let refused = out.join("refused");
        assert!(run(&vocab, &panel, "finite", &bad, &refused).is_err());
        assert!(!refused.exists());
        Ok(())
    }

    #[test]
    fn first_mismatch_preserves_missing_or_actual_original_id() {
        assert!(first_wrong(&[1808, 16, 1], &[1808, 16, 1]).is_none());
        let extra = first_wrong(&[1808, 16, 73, 1], &[1808, 16, 1]).unwrap();
        assert_eq!(
            (extra.position, extra.expected, extra.actual),
            (3, Some(1), Some(73))
        );
        let at_cap = first_wrong(&[1808, 16, 3984, 73], &[1808, 16, 3984, 1]).unwrap();
        assert_eq!(
            (at_cap.position, at_cap.expected, at_cap.actual),
            (4, Some(1), Some(73))
        );
        let missing = first_wrong(&[1808, 16, 3984], &[1808, 16, 3984, 1]).unwrap();
        assert_eq!(
            (missing.position, missing.expected, missing.actual),
            (4, Some(1), None)
        );
    }
}
