use super::prepare::Key;
use super::*;
use serde_json::{json, Value};

const ARMS: [&str; 5] = ["native", "learned", "swapped", "oracle", "wrong_oracle"];
const CAP: usize = 3;

#[derive(Deserialize)]
struct Observation {
    #[serde(flatten)]
    row: Row,
    score: i32,
    peer_score: i32,
    arms: BTreeMap<String, Arm>,
}

#[derive(Deserialize)]
struct Arm {
    steps: Vec<Step>,
    terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(PartialEq, Deserialize)]
struct Step {
    committed_id: u32,
    native_proposal: u32,
    selected_source: String,
    copied: bool,
    read_pos: Option<u32>,
    claim_pos: Option<u32>,
    source_id: Option<u32>,
    source_admitted: bool,
    #[serde(flatten)]
    other_event: BTreeMap<String, Value>,
}

type PairSide = (String, u8);

struct ResultRow<'a> {
    row: &'a Row,
    key: &'a Key,
    arm: &'a str,
    score: i32,
    peer_score: i32,
    emitted: Vec<u32>,
    terminal: bool,
    first_wrong: Option<usize>,
    exact: bool,
    native_exact: bool,
    second_source: &'a str,
}

fn bind_rows<'a, T>(
    rows: &'a [T],
    identify: impl Fn(&T) -> (&str, u8),
) -> Result<BTreeMap<PairSide, &'a T>> {
    let mut result = BTreeMap::new();
    for row in rows {
        let (pair, side) = identify(row);
        require(
            !pair.is_empty() && side < 2 && result.insert((pair.to_owned(), side), row).is_none(),
            format!("duplicate or malformed previously sealed row {pair}:{side}"),
        )?;
    }
    Ok(result)
}

fn verify_manifest(panel: &Path) -> Result<()> {
    let mut names = BTreeSet::new();
    for entry in fs::read_to_string(panel.join("SHA256SUMS"))?.lines() {
        let (digest, relative) = entry.split_once("  ").ok_or("invalid input manifest")?;
        require(
            hash(&panel.join(relative))? == digest && names.insert(relative.to_owned()),
            format!("sealed input digest differs for {relative}"),
        )?;
    }
    let provided = [
        "receipt.json",
        "train/rows.jsonl",
        "train/keys.jsonl",
        "held/rows.jsonl",
        "held/keys.jsonl",
        "finite/rows.jsonl",
        "finite/keys.jsonl",
    ];
    require(
        names == provided.iter().map(|path| (*path).to_owned()).collect(),
        "sealed input manifest has incomplete inventory",
    )
}

fn choice(score: i32, fallback: u32, arm: &'static str) -> (u32, &'static str) {
    match score.cmp(&0) {
        std::cmp::Ordering::Less => (EOS, arm),
        std::cmp::Ordering::Equal => (fallback, "native"),
        std::cmp::Ordering::Greater => (PERIOD, arm),
    }
}

fn first_wrong(expected: &[u32], actual: &[u32]) -> Option<usize> {
    (0..expected.len().max(actual.len()))
        .find(|&position| actual.get(position) != expected.get(position))
        .map(|index| index + 1)
}

fn measure<'a>(
    row: &'a Row,
    key: &'a Key,
    event: &'a Observation,
    vocab_size: usize,
) -> Result<Vec<ResultRow<'a>>> {
    let expected_second = if row.side == 0 { EOS } else { PERIOD };
    let expected = if row.side == 0 {
        vec![row.target, EOS]
    } else {
        vec![row.target, PERIOD, EOS]
    };
    require(
        row == &event.row
            && key.pair_id == row.pair_id
            && key.side == row.side
            && row.expected == expected
            && row.prefix.get(row.gold_pos as usize) == Some(&row.target)
            && event.arms.len() == ARMS.len(),
        format!(
            "raw event does not bind to the sealed row {}:{}",
            row.pair_id, row.side
        ),
    )?;
    let native = event.arms.get("native").ok_or("native arm missing")?;
    let first = native.steps.first().ok_or("missing native token one")?;
    let second = native.steps.get(1).ok_or("missing native token two")?;
    require(
        first.copied
            && first.source_admitted
            && first.selected_source == "copy"
            && first.read_pos == Some(row.gold_pos)
            && first.claim_pos == Some(row.gold_pos)
            && first.source_id == Some(row.target)
            && first.committed_id == row.target,
        format!("native opening changed on {}:{}", row.pair_id, row.side),
    )?;
    let native_exact = native.terminal
        && native
            .steps
            .iter()
            .map(|step| step.committed_id)
            .eq(expected.iter().copied());
    let mut after_second = BTreeMap::new();
    let mut result = Vec::with_capacity(ARMS.len());
    for name in ARMS {
        let arm = event
            .arms
            .get(name)
            .ok_or_else(|| format!("missing arm {name}"))?;
        let step_two = arm.steps.get(1).ok_or("missing token two")?;
        let emitted = arm
            .steps
            .iter()
            .map(|step| step.committed_id)
            .collect::<Vec<_>>();
        let actual_second = step_two.committed_id;
        let desired = match name {
            "native" => (second.native_proposal, "native"),
            "learned" => choice(event.score, second.native_proposal, name),
            "swapped" => choice(event.peer_score, second.native_proposal, name),
            "oracle" => (expected_second, "oracle"),
            "wrong_oracle" => (
                if expected_second == EOS { PERIOD } else { EOS },
                "wrong_oracle",
            ),
            _ => return Err(format!("unsupported arm {name}").into()),
        };
        let actual_wrong = first_wrong(&expected, &emitted);
        require(
            arm.steps.first() == Some(first)
                && (actual_second, step_two.selected_source.as_str()) == desired
                && !step_two.copied
                && step_two.native_proposal == second.native_proposal
                && emitted.iter().all(|&id| (id as usize) < vocab_size)
                && emitted.len() == if actual_second == EOS { 2 } else { CAP }
                && arm.terminal == (emitted.last() == Some(&EOS))
                && !emitted[..emitted.len() - 1].contains(&EOS)
                && actual_wrong == arm.first_mismatch,
            format!("invalid {name} trajectory on {}:{}", row.pair_id, row.side),
        )?;
        if let Some(third) = arm.steps.get(2) {
            let same_transition = after_second
                .insert(actual_second, third)
                .is_none_or(|previous| previous == third);
            require(
                !third.copied
                    && third.selected_source == "native"
                    && third.native_proposal == third.committed_id
                    && same_transition,
                format!(
                    "native self-fed third transition changed in {name} on {}:{}",
                    row.pair_id, row.side
                ),
            )?;
        }
        result.push(ResultRow {
            row,
            key,
            arm: name,
            score: event.score,
            peer_score: event.peer_score,
            exact: actual_wrong.is_none() && arm.terminal,
            emitted,
            terminal: arm.terminal,
            first_wrong: actual_wrong,
            native_exact,
            second_source: &step_two.selected_source,
        });
    }
    Ok(result)
}

fn piece(tokenizer: &ByteBpeTokenizer, id: Option<u32>) -> String {
    match id {
        Some(EOS) => "<EOS>".to_owned(),
        Some(id) => tokenizer.decode(&[id]),
        None => "<none>".to_owned(),
    }
}

#[derive(Default, Serialize)]
struct Counts {
    rows: usize,
    exact: usize,
    copied: usize,
    second_correct: usize,
    retained_native_exact: usize,
    gained_vs_native: usize,
    lost_vs_native: usize,
    first_wrong_position: BTreeMap<String, usize>,
    token_two_choices: BTreeMap<u32, usize>,
    terminal: usize,
}

impl Counts {
    fn add(&mut self, row: &ResultRow<'_>) {
        self.rows += 1;
        self.exact += usize::from(row.exact);
        self.copied += usize::from(row.emitted[0] == row.row.target);
        self.second_correct += usize::from(row.emitted[1] == row.row.expected[1]);
        self.retained_native_exact += usize::from(row.native_exact && row.exact);
        self.gained_vs_native += usize::from(!row.native_exact && row.exact);
        self.lost_vs_native += usize::from(row.native_exact && !row.exact);
        let position = row
            .first_wrong
            .map_or_else(|| "none".to_owned(), |at| at.to_string());
        *self.first_wrong_position.entry(position).or_default() += 1;
        *self.token_two_choices.entry(row.emitted[1]).or_default() += 1;
        self.terminal += usize::from(row.terminal);
    }
}

#[derive(Default)]
struct PairCounts {
    all: Counts,
    stop: Counts,
    period: Counts,
    pairs: BTreeMap<String, [Option<(bool, bool, bool)>; 2]>,
    faults: BTreeMap<(usize, Option<u32>, Option<u32>), usize>,
}

impl PairCounts {
    fn add(&mut self, row: &ResultRow<'_>) {
        self.all.add(row);
        if row.row.side == 0 {
            self.stop.add(row);
        } else {
            self.period.add(row);
        }
        self.pairs.entry(row.row.pair_id.clone()).or_default()[row.row.side as usize] = Some((
            row.exact,
            row.native_exact,
            row.emitted[1] == row.row.expected[1],
        ));
        if let Some(at) = row.first_wrong {
            let expected = row.row.expected.get(at - 1).copied();
            let actual = row.emitted.get(at - 1).copied();
            *self.faults.entry((at, expected, actual)).or_default() += 1;
        }
    }

    fn render(&self, tokenizer: &ByteBpeTokenizer) -> Result<Value> {
        let mut exact = 0;
        let mut retained = 0;
        let mut gained = 0;
        let mut lost = 0;
        let mut token_two = 0;
        for (id, pair) in &self.pairs {
            let [Some(stop), Some(period)] = pair else {
                return Err(format!("partial pair in summary: {id}").into());
            };
            let paired_exact = stop.0 && period.0;
            let native_paired_exact = stop.1 && period.1;
            exact += usize::from(paired_exact);
            retained += usize::from(paired_exact && native_paired_exact);
            gained += usize::from(paired_exact && !native_paired_exact);
            lost += usize::from(!paired_exact && native_paired_exact);
            token_two += usize::from(stop.2 && period.2);
        }
        let mut faults: Vec<_> = self.faults.iter().collect();
        faults.sort_by_key(|(&(at, expected, actual), &count)| {
            (std::cmp::Reverse(count), at, expected, actual)
        });
        let hist = faults.into_iter().map(|(&(at, expected, actual), &count)| json!({
            "position": at, "expected": expected, "actual": actual, "n": count,
            "expected_piece": piece(tokenizer, expected), "actual_piece": piece(tokenizer, actual),
        })).collect::<Vec<_>>();
        Ok(json!({
            "all": self.all, "stop": self.stop, "period": self.period,
            "pairs": self.pairs.len(), "exact_pairs": exact, "token_two_exact_pairs": token_two,
            "retained_native_exact_pairs": retained, "gained_vs_native_pairs": gained,
            "lost_vs_native_pairs": lost, "first_wrong_choice_counts": hist,
        }))
    }
}

fn csv_row(writer: &mut impl Write, fields: &[String]) -> Result<()> {
    for (index, field) in fields.iter().enumerate() {
        if index != 0 {
            writer.write_all(b",")?;
        }
        write!(writer, "\"{}\"", field.replace('"', "\"\""))?;
    }
    writeln!(writer)?;
    Ok(())
}

fn export_row(
    writer: &mut impl Write,
    tokenizer: &ByteBpeTokenizer,
    result: &ResultRow<'_>,
) -> Result<()> {
    let actual_next = result.emitted[1];
    let expected_next = result.row.expected[1];
    let first_wrong_piece = result
        .first_wrong
        .map(|pos| piece(tokenizer, result.emitted.get(pos - 1).copied()))
        .unwrap_or_default();
    let mut reply_ids = result.emitted.as_slice();
    if result.terminal {
        reply_ids = &reply_ids[..reply_ids.len() - 1];
    }
    csv_row(
        writer,
        &[
            result.row.pair_id.clone(),
            result.row.side.to_string(),
            result.row.family.clone(),
            result.key.stratum.clone(),
            result.arm.to_owned(),
            result.key.question.clone(),
            result.key.instruction.clone(),
            result.key.target_word.clone(),
            result.row.gold_pos.to_string(),
            serde_json::to_string(&result.row.expected)?,
            serde_json::to_string(&result.emitted)?,
            result.key.reply.clone(),
            tokenizer.decode(reply_ids),
            expected_next.to_string(),
            piece(tokenizer, Some(expected_next)),
            actual_next.to_string(),
            piece(tokenizer, Some(actual_next)),
            result.second_source.to_owned(),
            result.score.to_string(),
            result.peer_score.to_string(),
            result.terminal.to_string(),
            result.exact.to_string(),
            result.native_exact.to_string(),
            result
                .first_wrong
                .map(|position| position.to_string())
                .unwrap_or_default(),
            first_wrong_piece,
        ],
    )
}

pub(super) fn run(
    tokenizer: &ByteBpeTokenizer,
    panel: &Path,
    split: &str,
    observations: &Path,
    output: &Path,
) -> Result<()> {
    require(
        ["train", "held", "finite"].contains(&split),
        "undefined split",
    )?;
    verify_manifest(panel)?;
    let row_path = panel.join(split).join("rows.jsonl");
    let rows = read_jsonl::<Row>(&row_path)?;
    let keys = read_jsonl::<Key>(&panel.join(split).join("keys.jsonl"))?;
    let events = read_jsonl::<Observation>(observations)?;
    let rows = bind_rows(&rows, |row| (&row.pair_id, row.side))?;
    let keys = bind_rows(&keys, |row| (&row.pair_id, row.side))?;
    let events = bind_rows(&events, |event| (&event.row.pair_id, event.row.side))?;
    require(
        !rows.is_empty() && rows.keys().eq(keys.keys()) && rows.keys().eq(events.keys()),
        "sealed rows, raw observations and transcript keys differ",
    )?;
    fs::create_dir(output)?;
    let mut file = fresh_writer(&output.join("rows.csv"))?;
    writeln!(file, "pair_id,side,family,stratum,arm,question,instruction,target_word,gold_pos,expected_ids,committed_ids,expected_reply,decoded_reply,expected_token_two_id,expected_token_two_piece,actual_token_two_id,actual_token_two_piece,actual_token_two_source,score,peer_score,terminal,exact,native_exact,first_wrong,first_wrong_piece")?;
    let mut summaries = BTreeMap::<&str, BTreeMap<&str, PairCounts>>::new();
    let mut faults = BTreeMap::<(&str, u8, &str), Vec<Value>>::new();
    let mut proper_sign_pair = 0;
    let mut any_opposed_sign_pair = 0;
    let mut zero_scores = 0;

    for (index, row) in &rows {
        let peer = (row.pair_id.clone(), 1 - row.side);
        let event = events[index];
        let peer_event = events
            .get(&peer)
            .ok_or("observation without matched peer")?;
        require(
            event.score == peer_event.peer_score && event.peer_score == peer_event.score,
            "raw swapped score does not match opposite wording",
        )?;
        zero_scores += usize::from(event.score == 0);
        if row.side == 0 {
            proper_sign_pair += usize::from(event.score < 0 && event.peer_score > 0);
            any_opposed_sign_pair +=
                usize::from(event.score.signum() * event.peer_score.signum() == -1);
        }
        for result in measure(row, keys[index], event, tokenizer.vocab_size())? {
            for group in ["all", &row.family, &result.key.stratum] {
                summaries
                    .entry(group)
                    .or_default()
                    .entry(result.arm)
                    .or_default()
                    .add(&result);
            }
            if let Some(at) = result.first_wrong {
                let examples = faults
                    .entry((&row.family, row.side, result.arm))
                    .or_default();
                if examples.len() < 2 {
                    examples.push(json!({
                        "pair_id": row.pair_id, "side": row.side, "family": row.family, "arm": result.arm,
                        "context": result.key.context, "question": result.key.question,
                        "instruction": result.key.instruction, "guide_word": result.key.target_word,
                        "expected": row.expected, "committed": result.emitted, "terminal": result.terminal,
                        "score": result.score, "peer_score": result.peer_score,
                        "first_wrong": at, "expected_piece": piece(tokenizer, row.expected.get(at - 1).copied()),
                        "actual_piece": piece(tokenizer, result.emitted.get(at - 1).copied()),
                    }));
                }
            }
            export_row(&mut file, tokenizer, &result)?;
        }
    }
    file.flush()?;
    let mut examples = fresh_writer(&output.join("fault-examples.jsonl"))?;
    for example in faults.into_values().flatten() {
        serde_json::to_writer(&mut examples, &example)?;
        writeln!(examples)?;
    }
    examples.flush()?;

    let summary = summaries
        .iter()
        .map(|(&name, arms)| {
            let arms = arms
                .iter()
                .map(|(&arm, summary)| Ok((arm, summary.render(tokenizer)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            Ok((name, arms))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let receipt = json!({
        "schema": "codex-4319.word-ending-panel-score/1",
        "panel_manifest_sha256": hash(&panel.join("SHA256SUMS"))?,
        "sealed_rows_sha256": hash(&row_path)?,
        "observations_sha256": hash(observations)?,
        "split": split, "rows": rows.len(), "arms": ARMS, "by_group": summary,
        "state_signs": {"stop_negative_period_positive_pairs": proper_sign_pair, "opposite_nonzero_pairs": any_opposed_sign_pair, "zero_rows": zero_scores},
    });
    let mut report = fresh_writer(&output.join("summary.json"))?;
    serde_json::to_writer_pretty(&mut report, &receipt)?;
    writeln!(report)?;
    report.flush()?;
    let mut manifest = fresh_writer(&output.join("SHA256SUMS"))?;
    for relative in ["summary.json", "fault-examples.jsonl", "rows.csv"] {
        writeln!(manifest, "{}  {relative}", hash(&output.join(relative))?)?;
    }
    manifest.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Planted event tape for the grading instrument. No model is called.
    fn fixture(terminal_token: u32) -> Result<(Row, Key, Observation)> {
        let row: Row = serde_json::from_value(json!({
            "pair_id":"planted-fixture", "side":1, "prefix":[1808], "gold_pos":0,
            "target":1808, "expected":[1808,16,1], "family":"instrument",
        }))?;
        let key: Key = serde_json::from_value(json!({
            "pair_id":"planted-fixture", "side":1, "user":"", "context":"", "question":"",
            "instruction":"", "reply":"lake.", "target_word":"lake", "distractor_word":"",
            "target_person":"", "distractor_person":"", "target_fact":"", "fact_family":"",
            "fact_first":true, "target_after_subject":true, "instruction_variant":0, "stratum":"instrument",
        }))?;
        let first = json!({
            "committed_id":1808, "native_proposal":1, "selected_source":"copy",
            "copied":true, "read_pos":0, "claim_pos":0, "source_id":1808, "source_admitted":true,
        });
        let fallback = |committed, proposal, source| {
            json!({
                "committed_id":committed, "native_proposal":proposal, "selected_source":source,
                "copied":false, "read_pos":null, "claim_pos":null, "source_id":null, "source_admitted":false,
            })
        };
        let mut arms = BTreeMap::new();
        for (name, committed) in [
            ("native", EOS),
            ("learned", PERIOD),
            ("swapped", EOS),
            ("oracle", PERIOD),
            ("wrong_oracle", EOS),
        ] {
            let mut steps = vec![first.clone(), fallback(committed, EOS, name)];
            if committed == PERIOD {
                steps.push(fallback(terminal_token, terminal_token, "native"));
            }
            let terminal = committed == EOS || terminal_token == EOS;
            let first_mismatch = if committed == EOS {
                Some(2)
            } else if terminal {
                None
            } else {
                Some(3)
            };
            arms.insert(
                name,
                json!({"steps":steps, "terminal":terminal, "first_mismatch":first_mismatch}),
            );
        }
        let mut event = serde_json::to_value(&row)?;
        event["score"] = json!(1);
        event["peer_score"] = json!(-1);
        event["arms"] = serde_json::to_value(arms)?;
        Ok((row, key, serde_json::from_value(event)?))
    }

    #[test]
    fn period_requires_an_observed_native_eos() -> Result<()> {
        let (row, key, exact) = fixture(EOS)?;
        let full = measure(&row, &key, &exact, 4096)?;
        let oracle = full
            .iter()
            .find(|result| result.arm == "oracle")
            .ok_or("oracle missing")?;
        assert_eq!(oracle.emitted, [1808, PERIOD, EOS]);
        assert!(oracle.exact && oracle.terminal && oracle.first_wrong.is_none());

        let (_, _, wrong) = fixture(73)?;
        let capped = measure(&row, &key, &wrong, 4096)?;
        let oracle = capped
            .iter()
            .find(|result| result.arm == "oracle")
            .ok_or("oracle missing")?;
        assert_eq!(oracle.emitted, [1808, PERIOD, 73]);
        assert!(!oracle.exact && !oracle.terminal);
        assert_eq!(oracle.first_wrong, Some(3));

        let (_, _, mut mislabeled) = fixture(73)?;
        mislabeled
            .arms
            .get_mut("oracle")
            .ok_or("oracle missing")?
            .terminal = true;
        assert!(measure(&row, &key, &mislabeled, 4096).is_err());
        Ok(())
    }
}
