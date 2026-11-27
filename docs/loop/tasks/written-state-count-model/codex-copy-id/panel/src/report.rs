use super::*;
use serde_json::{json, Value};

const ARMS: [&str; 8] = [
    "generate",
    "learned",
    "correct",
    "swapped",
    "same_id_wrong_occurrence",
    "invalid",
    "missing",
    "marker",
];
const COMPARISONS: [(&str, &str); 3] = [
    ("learned", "generate"),
    ("correct", "missing"),
    ("correct", "swapped"),
];

#[derive(Deserialize)]
struct RegisteredRow {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    cohort: String,
    stratum: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    gold_pos: Option<usize>,
    rival_pos: Option<usize>,
    wrong_same_id_pos: Option<usize>,
    expected_second: Option<u32>,
    expected_terminal_after_first: Option<bool>,
    expected_terminal_after_second: Option<bool>,
    variant: Option<String>,
}

#[derive(Deserialize)]
struct DecodeKey {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    context: String,
    question: String,
    reply: Option<String>,
    gold: Vec<u32>,
    pieces: Vec<String>,
    topic: String,
    first_fact_side: usize,
    value_before_subject: bool,
    punctuation: bool,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Event {
    read_pos: Option<usize>,
    claim_pos: Option<usize>,
    gate_id: u32,
    source_id: Option<u32>,
    source_distance: Option<u32>,
    source_admitted: bool,
    minimum_distance: u32,
    first_admitted_cell: Option<u32>,
    native_proposal: u32,
    committed_id: u32,
    copied: bool,
    read_code: u32,
    recurrence_code: u32,
    causal_successor: Option<(usize, u32)>,
    read_vector: [i32; 32],
}

#[derive(Deserialize)]
struct Arm {
    first: Event,
    successor: Option<Event>,
    terminal: bool,
}

impl Arm {
    fn second(&self) -> Option<u32> {
        self.successor.as_ref().map(|event| event.committed_id)
    }

    fn first_correct(&self, row: &RegisteredRow) -> bool {
        row.expected_second.is_some() && self.first.committed_id == row.target
    }

    fn second_correct(&self, row: &RegisteredRow) -> bool {
        row.expected_second
            .is_some_and(|id| self.second() == Some(id))
    }

    fn two_correct(&self, row: &RegisteredRow) -> bool {
        self.first_correct(row) && self.second_correct(row)
    }

    fn read_eligible(&self, expected: Option<usize>) -> bool {
        expected.is_some()
            && self.first.read_pos == expected
            && self.first.source_admitted
            && self.first.source_distance == Some(self.first.minimum_distance)
    }
}

#[derive(Deserialize)]
struct Observation {
    pair_id: String,
    prefix: Vec<u32>,
    arms: BTreeMap<String, Arm>,
}

struct Bound {
    row: RegisteredRow,
    key: DecodeKey,
    observation: Observation,
}

#[derive(Default, Serialize)]
struct Counts {
    rows: usize,
    first_correct: usize,
    first_and_second_correct: usize,
    native_successor_correct_regardless_of_first: usize,
    first_eos: usize,
    second_eos: usize,
    terminal_after_either: usize,
    has_successor: usize,
    immediate_space: usize,
    correct_first_with_successor: usize,
    reads_any_position: usize,
    reads_exact_fact: usize,
    reads_rival: usize,
    reads_same_id_other_occurrence: usize,
    authorized_copies: usize,
    copy_changed_id: usize,
    copy_was_eligible_at_fact: usize,
    copy_was_eligible_at_rival: usize,
    supplied_same_id_other_claim_reached: usize,
    writer_alias_to_eos_after_eligible_read: usize,
    wrong_first_that_equals_rival: usize,
    first_ids: BTreeMap<u32, usize>,
    successor_ids: BTreeMap<u32, usize>,
}

impl Counts {
    fn add(&mut self, row: &RegisteredRow, arm: &Arm) {
        let [read, claim] = [arm.first.read_pos, arm.first.claim_pos];
        let first_correct = arm.first_correct(row);
        let first = arm.first.committed_id;
        let second = arm.second();
        self.rows += 1;
        self.first_correct += usize::from(first_correct);
        self.first_and_second_correct += usize::from(arm.two_correct(row));
        self.native_successor_correct_regardless_of_first += usize::from(arm.second_correct(row));
        self.first_eos += usize::from(first == EOS);
        self.second_eos += usize::from(second == Some(EOS));
        self.terminal_after_either += usize::from(arm.terminal);
        self.has_successor += usize::from(second.is_some());
        self.immediate_space += usize::from(first == SPACE);
        self.correct_first_with_successor += usize::from(first_correct && second.is_some());
        self.reads_any_position += usize::from(read.is_some());
        self.reads_exact_fact += usize::from(read.is_some() && read == row.gold_pos);
        self.reads_rival += usize::from(read.is_some() && read == row.rival_pos);
        self.reads_same_id_other_occurrence +=
            usize::from(read.is_some() && read == row.wrong_same_id_pos);
        self.authorized_copies += usize::from(arm.first.copied);
        self.copy_changed_id += usize::from(first != arm.first.native_proposal);
        self.copy_was_eligible_at_fact += usize::from(arm.read_eligible(row.gold_pos));
        self.copy_was_eligible_at_rival += usize::from(arm.read_eligible(row.rival_pos));
        self.supplied_same_id_other_claim_reached += usize::from(
            row.wrong_same_id_pos.is_some()
                && claim == row.wrong_same_id_pos
                && read == row.gold_pos,
        );
        self.writer_alias_to_eos_after_eligible_read +=
            usize::from(arm.read_eligible(row.gold_pos) && arm.first.native_proposal == EOS);
        self.wrong_first_that_equals_rival +=
            usize::from(row.expected_second.is_some() && first == row.rival);
        *self.first_ids.entry(first).or_insert(0) += 1;
        if let Some(second) = second {
            *self.successor_ids.entry(second).or_insert(0) += 1;
        }
    }
}

#[derive(Default, Serialize)]
struct Paired {
    both: usize,
    intervention_only: usize,
    control_only: usize,
    neither: usize,
}

impl Paired {
    fn add(&mut self, intervention: bool, control: bool) {
        match (intervention, control) {
            (true, true) => self.both += 1,
            (true, false) => self.intervention_only += 1,
            (false, true) => self.control_only += 1,
            (false, false) => self.neither += 1,
        }
    }
}

fn piece(tokenizer: &ByteBpeTokenizer, id: u32) -> String {
    if id == EOS {
        "<EOS>".into()
    } else {
        tokenizer.decode(&[id])
    }
}

fn optional_text<T: ToString>(value: Option<T>) -> String {
    value.map(|x| x.to_string()).unwrap_or_default()
}

fn csv(file: &mut impl Write, fields: impl IntoIterator<Item = String>) -> Result<()> {
    for (index, field) in fields.into_iter().enumerate() {
        if index != 0 {
            file.write_all(b",")?;
        }
        let escaped = field.replace('"', "\"\"");
        write!(file, "\"{escaped}\"")?;
    }
    writeln!(file)?;
    Ok(())
}

fn validate_event(event: &Event, prefix: &[u32]) -> Result<()> {
    require(
        [event.gate_id, event.native_proposal, event.committed_id]
            .into_iter()
            .all(|id| id < 4096)
            && event.source_id.is_none_or(|id| id < 4096),
        "writer token outside vocabulary",
    )?;
    require(
        event.read_pos.is_none_or(|pos| pos < prefix.len())
            && event
                .source_id
                .is_none_or(|id| event.claim_pos.and_then(|pos| prefix.get(pos)) == Some(&id)),
        "recorded reader or claimed token differs from the causal prefix",
    )?;
    if let Some((pos, id)) = event.causal_successor {
        require(
            prefix.get(pos) == Some(&id),
            "causal successor pointer outside given prefix",
        )?;
    }
    if event.copied {
        require(
            event.read_pos.is_some()
                && event.read_pos == event.claim_pos
                && event.source_id == Some(event.committed_id)
                && event.read_pos.and_then(|pos| prefix.get(pos)) == event.source_id.as_ref()
                && event.source_admitted
                && event.source_distance == Some(event.minimum_distance),
            "copy lacks original-ID address and minimum-distance witness",
        )?;
    } else {
        require(
            event.committed_id == event.native_proposal,
            "Generate changed the writer proposal",
        )?;
    }
    Ok(())
}

fn bind(
    row: RegisteredRow,
    key: DecodeKey,
    observation: Observation,
    tokenizer: &ByteBpeTokenizer,
) -> Result<Bound> {
    require(
        row.pair_id == key.pair_id
            && row.group_id == key.group_id
            && row.assignment == key.assignment
            && row.side == key.side
            && row.pair_id == observation.pair_id
            && row.prefix == observation.prefix,
        "panel key, causal prefix and raw observation disagree",
    )?;
    require(
        key.gold
            .iter()
            .map(|&id| piece(tokenizer, id))
            .eq(key.pieces.iter().cloned()),
        "registered token-piece decoding changed",
    )?;
    if row.expected_second.is_some() {
        require(
            row.expected_second == key.gold.get(1).copied()
                && key.gold.first() == Some(&row.target)
                && row.expected_terminal_after_first == Some(false)
                && row.expected_terminal_after_second == Some(row.expected_second == Some(EOS))
                && row.gold_pos.is_some(),
            "registered terminal or successor target changed",
        )?;
    } else {
        require(
            key.gold.is_empty()
                && row.gold_pos.is_none()
                && row.expected_terminal_after_first.is_none()
                && row.expected_terminal_after_second.is_none(),
            "unsupported prompt has a supervised answer",
        )?;
    }
    require(
        observation.arms.len() == ARMS.len()
            && ARMS.iter().all(|name| observation.arms.contains_key(*name)),
        "observation has the wrong arms",
    )?;
    let mut replay = BTreeMap::new();
    for (name, arm) in &observation.arms {
        validate_event(&arm.first, &row.prefix)?;
        let first = arm.first.committed_id;
        match &arm.successor {
            None => require(
                first == EOS && arm.terminal,
                "EOS must immediately terminate",
            )?,
            Some(successor) => {
                require(
                    first != EOS
                        && !successor.copied
                        && successor.claim_pos.is_none()
                        && arm.terminal == (successor.committed_id == EOS),
                    "invalid successor or terminal state",
                )?;
                let mut prefix = row.prefix[usize::from(row.prefix.len() == LIMIT)..].to_vec();
                prefix.push(first);
                validate_event(successor, &prefix)?;
            }
        }
        if let Some(previous) = replay.insert(first, arm.successor.as_ref()) {
            require(
                previous == arm.successor.as_ref(),
                "same actual self-feed ID yielded a different native successor",
            )?;
        }
        if matches!(
            name.as_str(),
            "generate" | "same_id_wrong_occurrence" | "invalid" | "missing" | "marker"
        ) {
            require(!arm.first.copied, format!("prohibited Copy in {name}"))?;
        }
        if name == "correct" && arm.read_eligible(row.gold_pos) {
            require(
                arm.first.copied && arm.first.committed_id == row.target,
                "eligible correct read did not commit original ID",
            )?;
        }
        if name == "swapped" && arm.read_eligible(row.rival_pos) {
            require(
                arm.first.copied && arm.first.committed_id == row.rival,
                "consistently swapped read did not commit rival ID",
            )?;
        }
    }
    let arms = &observation.arms;
    let generate = &arms["generate"].first;
    let learned = &arms["learned"].first;
    require(
        generate.native_proposal == learned.native_proposal
            && generate.read_pos == learned.read_pos
            && generate.gate_id == learned.gate_id
            && generate.read_code == learned.read_code
            && generate.read_vector == learned.read_vector
            && generate.recurrence_code == learned.recurrence_code,
        "learned Copy and Generate disagree on the native reader or proposal",
    )?;
    let correct = &arms["correct"].first;
    let missing = &arms["missing"].first;
    require(
        correct.native_proposal == missing.committed_id
            && correct.read_pos == missing.read_pos
            && correct.gate_id == missing.gate_id,
        "missing-claim control changed the supplied reader or native writer",
    )?;
    if let Some(other) = row.wrong_same_id_pos {
        let bad = &arms["same_id_wrong_occurrence"].first;
        require(
            row.prefix.get(other) == Some(&row.target)
                && Some(other) != row.gold_pos
                && bad.read_pos == row.gold_pos
                && bad.claim_pos == Some(other)
                && bad.source_id == Some(row.target)
                && bad.gate_id == correct.gate_id
                && bad.native_proposal == correct.native_proposal,
            "same-ID wrong-occurrence control did not reach its independent witness",
        )?;
    }
    Ok(Bound {
        row,
        key,
        observation,
    })
}

fn arm_csv(
    file: &mut impl Write,
    bound: &Bound,
    name: &str,
    tokenizer: &ByteBpeTokenizer,
) -> Result<()> {
    let Bound {
        row,
        key,
        observation,
    } = bound;
    let arm = &observation.arms[name];
    let first = &arm.first;
    let successor = arm.successor.as_ref();
    let first_text = piece(tokenizer, first.committed_id);
    let second_text = arm.second().map(|id| piece(tokenizer, id));
    let actual_text = tokenizer.decode(
        &[Some(first.committed_id), arm.second()]
            .into_iter()
            .flatten()
            .filter(|id| *id != EOS)
            .collect::<Vec<_>>(),
    );
    csv(
        file,
        [
            row.cohort.clone(),
            row.stratum.clone(),
            key.topic.clone(),
            row.group_id.clone(),
            row.pair_id.clone(),
            row.assignment.to_string(),
            row.side.to_string(),
            name.into(),
            row.variant.clone().unwrap_or_default(),
            key.first_fact_side.to_string(),
            key.value_before_subject.to_string(),
            key.punctuation.to_string(),
            key.context.clone(),
            key.question.clone(),
            key.reply.clone().unwrap_or_default(),
            optional_text(row.expected_second.map(|_| row.target)),
            optional_text(row.expected_second),
            key.pieces.first().cloned().unwrap_or_default(),
            key.pieces.get(1).cloned().unwrap_or_default(),
            row.target.to_string(),
            row.rival.to_string(),
            optional_text(row.gold_pos),
            optional_text(row.rival_pos),
            optional_text(row.wrong_same_id_pos),
            optional_text(first.read_pos),
            optional_text(first.claim_pos),
            optional_text(first.source_id),
            first.gate_id.to_string(),
            first.native_proposal.to_string(),
            piece(tokenizer, first.native_proposal),
            first.source_admitted.to_string(),
            optional_text(first.source_distance),
            first.minimum_distance.to_string(),
            optional_text(first.first_admitted_cell),
            first.copied.to_string(),
            first.committed_id.to_string(),
            first_text,
            optional_text((first.committed_id != EOS).then_some(first.committed_id)),
            optional_text(arm.second()),
            second_text.unwrap_or_default(),
            actual_text,
            (first.committed_id == EOS).to_string(),
            (arm.second() == Some(EOS)).to_string(),
            arm.terminal.to_string(),
            optional_text(row.expected_terminal_after_first),
            optional_text(row.expected_terminal_after_second),
            optional_text(row.expected_second.map(|_| arm.first_correct(row))),
            optional_text(row.expected_second.map(|_| arm.second_correct(row))),
            optional_text(row.expected_second.map(|_| arm.two_correct(row))),
            optional_text(successor.map(|event| event.gate_id)),
            optional_text(successor.map(|event| event.native_proposal)),
            optional_text(successor.and_then(|event| event.read_pos)),
            optional_text(successor.and_then(|event| event.causal_successor.map(|entry| entry.0))),
            optional_text(successor.and_then(|event| event.causal_successor.map(|entry| entry.1))),
        ],
    )
}

fn scoring_groups<'a>(bound: &[&'a Bound], level: &str) -> Result<Vec<Vec<&'a Bound>>> {
    if level == "row" {
        return Ok(bound.iter().map(|&row| vec![row]).collect());
    }
    let pair = level == "assignment";
    let mut groups = BTreeMap::<_, Vec<_>>::new();
    for &record in bound {
        let row = &record.row;
        groups
            .entry((&row.group_id, pair.then_some(row.assignment)))
            .or_default()
            .push(record);
    }
    require(
        groups
            .values()
            .all(|rows| rows.len() == if pair { 2 } else { 4 }),
        "incomplete matched assignment or group",
    )?;
    Ok(groups.into_values().collect())
}

fn hits(records: &[&Bound], arm: &str, second: bool) -> bool {
    records.iter().all(|bound| {
        let event = &bound.observation.arms[arm];
        if second {
            event.two_correct(&bound.row)
        } else {
            event.first_correct(&bound.row)
        }
    })
}

fn write_metrics(file: &mut impl Write, bound: &[Bound]) -> Result<Value> {
    csv(
        file,
        [
            "stratum",
            "unit",
            "group_id",
            "assignment",
            "side",
            "criterion",
            "intervention",
            "control",
            "intervention_correct",
            "control_correct",
        ]
        .map(String::from),
    )?;
    let mut content = BTreeMap::new();
    for stratum in ["all", "distinct_held_words", "same_code_1808_3984"] {
        let subset = bound
            .iter()
            .filter(|record| {
                record.row.cohort == "controlled"
                    && (stratum == "all" || stratum == record.row.stratum)
            })
            .collect::<Vec<_>>();
        let mut summary = BTreeMap::new();
        for level in ["row", "assignment", "quartet"] {
            let groups = scoring_groups(&subset, level)?;
            let mut metrics = BTreeMap::new();
            let outcomes = |arm, second| {
                groups
                    .iter()
                    .filter(|records| hits(records, arm, second))
                    .count()
            };
            let accuracy: BTreeMap<_, _> = ARMS.into_iter().map(|arm| (arm,
                json!({ "correct_first": outcomes(arm, false), "correct_first_and_second": outcomes(arm, true) }))).collect();
            metrics.insert("exact_matches", json!(accuracy));
            metrics.insert("denominator", json!(groups.len()));
            let mut paired = BTreeMap::new();
            for (intervention, control) in COMPARISONS {
                for (criterion, second) in [("first", false), ("first_and_second", true)] {
                    let mut tallies = Paired::default();
                    for records in &groups {
                        let hit = hits(records, intervention, second);
                        let baseline = hits(records, control, second);
                        tallies.add(hit, baseline);
                        let first = &records[0].row;
                        csv(
                            file,
                            [
                                stratum.into(),
                                level.into(),
                                first.group_id.clone(),
                                optional_text((level != "quartet").then_some(first.assignment)),
                                optional_text((level == "row").then_some(first.side)),
                                criterion.into(),
                                intervention.into(),
                                control.into(),
                                hit.to_string(),
                                baseline.to_string(),
                            ],
                        )?;
                    }
                    paired.insert(format!("{intervention}_vs_{control}_{criterion}"), tallies);
                }
            }
            metrics.insert("paired_discordance", json!(paired));
            summary.insert(level, metrics);
        }
        content.insert(stratum, summary);
    }
    Ok(json!(content))
}

fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = fresh_writer(path)?;
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn verify_manifest(panel: &Path) -> Result<()> {
    let manifest = fs::read_to_string(panel.join("SHA256SUMS"))?;
    let mut expected = BTreeSet::from([
        "held/rows.jsonl",
        "held/keys.jsonl",
        "absent/rows.jsonl",
        "absent/keys.jsonl",
        "finite/rows.jsonl",
        "finite/keys.jsonl",
        "receipt.json",
    ]);
    for line in manifest.lines() {
        let (digest, file) = line.split_once("  ").ok_or("malformed panel manifest")?;
        require(
            expected.remove(file) && hash(&panel.join(file))? == digest,
            format!("panel checksum mismatch: {file}"),
        )?;
    }
    require(expected.is_empty(), "panel manifest omitted required files")
}

pub(crate) fn run(
    tokenizer: &Path,
    panel: &Path,
    observations: &Path,
    output: &Path,
) -> Result<()> {
    require(
        hash(tokenizer)? == history::TOKENIZER_SHA256,
        "report tokenizer digest changed",
    )?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or("invalid sealed tokenizer")?;
    verify_manifest(panel)?;
    let mut observations_index = BTreeMap::new();
    for observation in read_jsonl::<Observation>(observations)? {
        let key = (observation.pair_id.clone(), observation.prefix.clone());
        require(
            observations_index.insert(key, observation).is_none(),
            "duplicate observed causal row",
        )?;
    }
    let mut bound = Vec::new();
    for split in ["held", "absent", "finite"] {
        let dir = panel.join(split);
        let rows = read_jsonl::<RegisteredRow>(&dir.join("rows.jsonl"))?;
        let keys = read_jsonl::<DecodeKey>(&dir.join("keys.jsonl"))?;
        require(
            rows.len() == keys.len(),
            "panel rows and decoded keys are different sizes",
        )?;
        for (row, key) in rows.into_iter().zip(keys) {
            let observation = observations_index
                .remove(&(row.pair_id.clone(), row.prefix.clone()))
                .ok_or_else(|| {
                    format!("missing observation for {} side {}", row.pair_id, row.side)
                })?;
            let label = format!(
                "{} assignment {} question {}",
                row.pair_id, row.assignment, row.side
            );
            bound.push(
                bind(row, key, observation, &tokenizer)
                    .map_err(|error| format!("{label}: {error}"))?,
            );
        }
    }
    require(
        observations_index.is_empty(),
        "observation file includes unregistered rows",
    )?;
    fs::create_dir(output)?;
    let mut arms_csv = fresh_writer(&output.join("arms.csv"))?;
    let mut paired_csv = fresh_writer(&output.join("paired.csv"))?;
    csv(
        &mut arms_csv,
        [
            "cohort",
            "stratum",
            "family",
            "group_id",
            "pair_id",
            "assignment",
            "side",
            "arm",
            "variant",
            "first_fact_side",
            "value_before_subject",
            "punctuation_requested",
            "context",
            "question",
            "registered_reply",
            "expected_first_id",
            "expected_second_id",
            "expected_first_piece",
            "expected_second_piece",
            "diagnostic_target_id",
            "rival_id",
            "authorizing_fact_pos",
            "rival_pos",
            "wrong_same_id_pos",
            "read_pos",
            "claim_pos",
            "source_id",
            "first_gate_id",
            "first_native_proposal_id",
            "first_native_proposal_piece",
            "source_admitted",
            "source_distance",
            "minimum_distance",
            "first_admitted_cell",
            "copied",
            "first_committed_id",
            "first_committed_piece",
            "self_fed_id",
            "successor_id",
            "successor_piece",
            "visible_decoded_reply",
            "terminal_at_first",
            "terminal_at_second",
            "terminal_after_either",
            "expected_terminal_after_first",
            "expected_terminal_after_second",
            "first_correct",
            "second_correct",
            "first_and_second_correct",
            "successor_gate_id",
            "successor_native_proposal_id",
            "successor_read_pos",
            "successor_causal_pointer_pos",
            "successor_causal_pointer_id",
        ]
        .map(String::from),
    )?;
    let mut scoped = BTreeMap::<_, BTreeMap<_, Counts>>::new();
    for record in &bound {
        for name in ARMS {
            arm_csv(&mut arms_csv, record, name, &tokenizer)?;
            for scope in [
                record.row.cohort.clone(),
                format!("{}/{}", record.row.cohort, record.row.stratum),
                format!("{}/{}", record.row.cohort, record.key.topic),
                format!(
                    "{}/{}",
                    record.row.cohort,
                    if record.key.punctuation {
                        "next_period"
                    } else {
                        "next_eos"
                    }
                ),
            ] {
                scoped
                    .entry(scope)
                    .or_default()
                    .entry(name)
                    .or_default()
                    .add(&record.row, &record.observation.arms[name]);
            }
        }
    }
    let crossed = write_metrics(&mut paired_csv, &bound)?;
    arms_csv.flush()?;
    paired_csv.flush()?;
    let collision = bound
        .iter()
        .filter(|record| {
            record.row.cohort == "controlled" && record.row.stratum == "same_code_1808_3984"
        })
        .map(|record| &record.row.group_id)
        .min()
        .ok_or("missing registered collision stratum")?;
    let examples = bound.iter().filter(|record| &record.row.group_id == collision && record.row.cohort == "controlled")
        .map(|record| {
            let details = |name| {
                let arm = &record.observation.arms[name];
                json!({
                    "read_pos": arm.first.read_pos, "claim_pos": arm.first.claim_pos,
                    "source_id": arm.first.source_id, "native_proposal_id": arm.first.native_proposal,
                    "native_proposal_piece": piece(&tokenizer, arm.first.native_proposal),
                    "committed_id": arm.first.committed_id, "committed_piece": piece(&tokenizer, arm.first.committed_id),
                    "second_id": arm.second(), "second_piece": arm.second().map(|id| piece(&tokenizer, id)),
                    "copied": arm.first.copied, "terminal": arm.terminal,
                })
            };
            json!({
                "assignment": record.row.assignment, "side": record.row.side,
                "context": record.key.context, "question": record.key.question,
                "target": record.row.target, "reply": record.key.reply, "gold": record.key.gold,
                "gold_pos": record.row.gold_pos, "rival": record.row.rival, "rival_pos": record.row.rival_pos,
                "generate": details("generate"), "learned": details("learned"),
                "correct": details("correct"), "swapped": details("swapped"),
            })
        }).collect::<Vec<_>>();
    write(
        &output.join("collision-examples.json"),
        &json!({"group_id": collision, "rows": examples}),
    )?;
    write(
        &output.join("summary.json"),
        &json!({
            "schema": "uor-r4.authenticated-copy-opening-analysis/1",
            "panel_manifest_sha256": hash(&panel.join("SHA256SUMS"))?,
            "raw_observations_sha256": hash(observations)?,
            "raw_rows": bound.len(), "arms_per_row": ARMS.len(),
            "paired": crossed, "row_counts": scoped,
        }),
    )?;
    let mut manifest = fresh_writer(&output.join("SHA256SUMS"))?;
    for name in [
        "arms.csv",
        "paired.csv",
        "collision-examples.json",
        "summary.json",
    ] {
        writeln!(manifest, "{}  {name}", hash(&output.join(name))?)?;
    }
    manifest.flush()?;
    println!("{}  SHA256SUMS", hash(&output.join("SHA256SUMS"))?);
    Ok(())
}
