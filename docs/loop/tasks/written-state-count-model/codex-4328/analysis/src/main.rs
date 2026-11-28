use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CAP: usize = 12;
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const ARTIFACTS: [(&str, &str); 3] = [
    (
        "base",
        "93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a",
    ),
    (
        "balanced",
        "5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa",
    ),
    (
        "paired",
        "4a13e77f0320aca1fab33bf92a668e28cef14189afea6ae1f2bc387b968a13e6",
    ),
];
const DECODE_ARMS: [&str; 3] = ["default", "authenticated_copy", "no_read"];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Source {
    pos: usize,
    id: u32,
    byte_start: usize,
    byte_end: usize,
}

#[derive(Deserialize)]
struct Truth {
    case_id: String,
    group: String,
    family: String,
    fact_order: usize,
    question_target: String,
    fenced: bool,
    supported: bool,
    reply: String,
    prompt_ids: Vec<u32>,
    sources: Vec<Source>,
    same_id_rival: Option<Source>,
    expected_ids: Vec<u32>,
    period_step: usize,
    eos_id: u32,
}

#[derive(Deserialize)]
struct Record {
    case_id: String,
    artifact_arm: String,
    decode_arm: String,
    execution: String,
    client: Client,
}

#[derive(Deserialize)]
struct Client {
    context: usize,
    artifact_sha256: String,
    tokenizer_sha256: String,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    text: String,
    stop: String,
    native_occurrence_steps: Option<Vec<Step>>,
}

#[derive(Deserialize, PartialEq, Eq)]
struct Step {
    prefix_start: usize,
    prefix_len: usize,
    previous_id: Option<u32>,
    framing: String,
    parsed_start: Option<usize>,
    parsed_end: Option<usize>,
    decision: String,
    source_pos: Option<usize>,
    source_id: Option<u32>,
    claim_pos: Option<usize>,
    claimed_id: Option<u32>,
    gate_id: u32,
    native_proposal: u32,
    committed_id: u32,
    copied: bool,
    source_admitted: bool,
    #[serde(flatten)]
    other_words: BTreeMap<String, serde_json::Value>,
}

impl Client {
    fn same_transcript(&self, other: &Self) -> bool {
        self.prompt_ids == other.prompt_ids
            && self.generated_ids == other.generated_ids
            && self.stop == other.stop
            && self.text == other.text
            && self.context == other.context
            && self.tokenizer_sha256 == other.tokenizer_sha256
    }
}

#[derive(Serialize)]
struct Wrong {
    step: usize,
    expected_id: Option<u32>,
    emitted_id: Option<u32>,
}

#[derive(Serialize)]
struct Grade<'a> {
    case_id: &'a str,
    artifact_arm: &'a str,
    decode_arm: &'a str,
    group: &'a str,
    family: &'a str,
    fact_order: usize,
    question_target: &'a str,
    fenced: bool,
    supported: bool,
    framing: &'a str,
    decision: &'a str,
    expected_source: Option<&'a Source>,
    same_id_rival: Option<&'a Source>,
    observed_pos: Option<usize>,
    observed_id: Option<u32>,
    claimed_pos: Option<usize>,
    claimed_id: Option<u32>,
    copied: bool,
    source_correct: bool,
    same_id_wrong_address: bool,
    authenticated_opening_correct: bool,
    opening_proposal: u32,
    opening_id: u32,
    expected_opening_id: u32,
    opening_id_correct: bool,
    observed_next_id: Option<u32>,
    expected_next_id: Option<u32>,
    next_id_correct_after_opening: bool,
    period_without_forcing: bool,
    release_boundary: bool,
    next_after_period: Option<u32>,
    expected_next_after_period: Option<u32>,
    exact: bool,
    first_wrong: Option<Wrong>,
    stop: &'a str,
    text: &'a str,
    generated_ids: &'a [u32],
    expected_reply: &'a str,
    expected_ids: &'a [u32],
}

#[derive(Default, Serialize)]
struct Counts {
    rows: usize,
    supported: usize,
    unsupported: usize,
    parsed: usize,
    framing_rejected: usize,
    no_read: usize,
    supported_no_read: usize,
    unsupported_no_read: usize,
    unsupported_parsed_abstention: usize,
    disabled: usize,
    score_tie: usize,
    no_eligible_candidate: usize,
    no_read_threshold: usize,
    selected: usize,
    correct_source: usize,
    supported_wrong_source: usize,
    same_id_wrong_address: usize,
    same_id_wrong_address_copied: usize,
    unsupported_false_read: usize,
    copied: usize,
    supported_wrong_copy: usize,
    unsupported_false_copy: usize,
    correct_authenticated_opening: usize,
    correct_supported_opening_id: usize,
    correct_unsupported_opening_id: usize,
    correct_next_id_after_opening: usize,
    period_after_correct_opening: usize,
    supported_period_after_correct_opening: usize,
    release_boundary: usize,
    correct_postperiod_id_at_release_boundary: usize,
    exact: usize,
    eos: usize,
    cap: usize,
    first_wrong_at: BTreeMap<usize, usize>,
    next_after_correct_opening: BTreeMap<u32, usize>,
    next_after_release_boundary: BTreeMap<u32, usize>,
}

impl Counts {
    fn add(&mut self, row: &Grade<'_>) {
        let selected = row.observed_pos.is_some();
        let parsed = row.framing == "parsed";
        self.rows += 1;
        self.supported += n(row.supported);
        self.unsupported += n(!row.supported);
        self.parsed += n(parsed);
        self.framing_rejected += n(!parsed);
        self.no_read += n(!selected);
        self.supported_no_read += n(row.supported && !selected);
        self.unsupported_no_read += n(!row.supported && !selected);
        self.unsupported_parsed_abstention +=
            n(!row.supported && parsed && !selected && row.decision != "disabled");
        self.disabled += n(row.decision == "disabled");
        self.score_tie += n(row.decision == "score_tie");
        self.no_eligible_candidate += n(row.decision == "no_eligible_candidate");
        self.no_read_threshold += n(row.decision == "no_read_threshold");
        self.selected += n(selected);
        self.correct_source += n(row.source_correct);
        self.supported_wrong_source += n(row.supported && selected && !row.source_correct);
        self.same_id_wrong_address += n(row.same_id_wrong_address);
        self.same_id_wrong_address_copied += n(row.same_id_wrong_address && row.copied);
        self.unsupported_false_read += n(!row.supported && selected);
        self.copied += n(row.copied);
        self.supported_wrong_copy += n(row.supported && row.copied && !row.source_correct);
        self.unsupported_false_copy += n(!row.supported && row.copied);
        self.correct_authenticated_opening += n(row.authenticated_opening_correct);
        self.correct_supported_opening_id += n(row.supported && row.opening_id_correct);
        self.correct_unsupported_opening_id += n(!row.supported && row.opening_id_correct);
        self.correct_next_id_after_opening += n(row.next_id_correct_after_opening);
        self.period_after_correct_opening += n(row.period_without_forcing);
        self.supported_period_after_correct_opening +=
            n(row.supported && row.period_without_forcing);
        self.release_boundary += n(row.release_boundary);
        self.correct_postperiod_id_at_release_boundary += n(row.release_boundary
            && row.next_after_period.is_some()
            && row.next_after_period == row.expected_next_after_period);
        self.exact += n(row.exact);
        self.eos += n(row.stop == "eos");
        self.cap += n(row.stop == "max_tokens");
        if let Some(wrong) = &row.first_wrong {
            *self.first_wrong_at.entry(wrong.step).or_default() += 1;
        }
        if row.opening_id_correct {
            if let Some(id) = row.observed_next_id {
                *self.next_after_correct_opening.entry(id).or_default() += 1;
            }
        }
        if row.release_boundary {
            if let Some(id) = row.next_after_period {
                *self.next_after_release_boundary.entry(id).or_default() += 1;
            }
        }
    }
}

fn n(value: bool) -> usize {
    usize::from(value)
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(line_no, line)| {
            serde_json::from_str(&line?)
                .map_err(|err| format!("{}:{}: {err}", path.display(), line_no + 1).into())
        })
        .collect()
}

fn need(ok: bool, scope: &str, message: &str) -> Result<()> {
    if !ok {
        return Err(format!("{scope}: {message}").into());
    }
    Ok(())
}

fn first_wrong(expected: &[u32], actual: &[u32]) -> Option<Wrong> {
    (0..expected.len().max(actual.len())).find_map(|step| {
        let expected_id = expected.get(step).copied();
        let emitted_id = actual.get(step).copied();
        (expected_id != emitted_id).then_some(Wrong {
            step,
            expected_id,
            emitted_id,
        })
    })
}

fn count_cross_artifact_agreement(records: &[Record]) -> BTreeMap<&str, usize> {
    let by_case: BTreeMap<_, _> = records
        .iter()
        .map(|row| {
            (
                (
                    row.case_id.as_str(),
                    row.decode_arm.as_str(),
                    row.artifact_arm.as_str(),
                ),
                &row.client,
            )
        })
        .collect();
    let mut exact = BTreeMap::from(DECODE_ARMS.map(|decode| (decode, 0)));
    for base in records
        .iter()
        .filter(|row| row.artifact_arm == ARTIFACTS[0].0)
    {
        let same = ARTIFACTS[1..].iter().all(|&(name, _)| {
            by_case
                .get(&(base.case_id.as_str(), base.decode_arm.as_str(), name))
                .is_some_and(|client| {
                    client.same_transcript(&base.client)
                        && client.native_occurrence_steps == base.client.native_occurrence_steps
                })
        });
        *exact.entry(&base.decode_arm).or_default() += n(same);
    }
    exact
}

fn validate_step(step: &Step, causal: &[u32], context: usize, id: u32, scope: &str) -> Result<()> {
    let expected_start = causal.len().saturating_sub(context);
    let window = &causal[expected_start..];
    let parsed = match (step.parsed_start, step.parsed_end) {
        (Some(start), Some(end)) if start < end && end <= window.len() => true,
        (None, None) => false,
        _ => return Err(format!("{scope}: invalid parsed span").into()),
    };
    need(
        step.prefix_start == expected_start
            && step.prefix_len == window.len()
            && step.previous_id == window.last().copied()
            && step.committed_id == id
            && step.framing == if parsed { "parsed" } else { "rejected" },
        scope,
        "trace does not follow the committed causal window",
    )?;
    let value = |pos: Option<usize>| pos.and_then(|pos| window.get(pos)).copied();
    need(
        step.source_id == value(step.source_pos)
            && step.claimed_id == value(step.claim_pos)
            && (step.source_pos.is_some() == step.source_id.is_some())
            && step.source_id.is_none_or(|id| id == step.gate_id)
            && (step.claim_pos.is_some() == step.claimed_id.is_some()),
        scope,
        "trace source or claim differs from original token bytes",
    )?;
    if let Some(source_pos) = step.source_pos {
        need(
            step.parsed_start
                .zip(step.parsed_end)
                .is_some_and(|(start, end)| (start..end).contains(&source_pos)),
            scope,
            "read position outside parsed facts",
        )?;
    }
    need(
        (step.decision == "selected") == step.source_pos.is_some()
            && (parsed || matches!(step.decision.as_str(), "framing_rejected" | "disabled"))
            && (!step.copied
                || (step.claim_pos.is_some()
                    && step.claim_pos == step.source_pos
                    && step.source_admitted
                    && step.claimed_id == Some(id))),
        scope,
        "inconsistent framing, selection or authenticated copy",
    )
}

fn grade<'a>(record: &'a Record, truth: &'a Truth) -> Result<Grade<'a>> {
    let client = &record.client;
    let scope = format!(
        "{}/{}/{}",
        record.case_id, record.artifact_arm, record.decode_arm
    );
    let steps = client
        .native_occurrence_steps
        .as_ref()
        .ok_or_else(|| format!("{scope}: no traced steps"))?;
    let actual = &client.generated_ids;
    let expected = &truth.expected_ids;
    need(
        ARTIFACTS.contains(&(
            record.artifact_arm.as_str(),
            client.artifact_sha256.as_str(),
        )) && DECODE_ARMS.contains(&record.decode_arm.as_str())
            && client.tokenizer_sha256 == TOKENIZER
            && client.prompt_ids == truth.prompt_ids
            && client.context >= truth.prompt_ids.len()
            && actual.len() == steps.len()
            && (1..=CAP).contains(&actual.len())
            && expected.last() == Some(&truth.eos_id)
            && expected.get(truth.period_step) == Some(&16)
            && truth.supported != truth.sources.is_empty(),
        &scope,
        "input, output or sealed truth mismatch",
    )?;
    let eos = actual.last() == Some(&truth.eos_id);
    need(
        !actual[..actual.len() - 1].contains(&truth.eos_id)
            && ((eos && client.stop == "eos")
                || (!eos && actual.len() == CAP && client.stop == "max_tokens")),
        &scope,
        "stop reason differs from committed EOS/cap",
    )?;
    let mut causal = client.prompt_ids.clone();
    for (index, (step, &id)) in steps.iter().zip(actual).enumerate() {
        validate_step(step, &causal, client.context, id, &scope)?;
        need(
            step.source_pos
                .is_none_or(|pos| pos + step.prefix_start < truth.prompt_ids.len())
                && (!step.copied || (index == 0 && record.decode_arm == "authenticated_copy")),
            &scope,
            "selected a generated token or copied outside the one-shot opening arm",
        )?;
        if record.decode_arm == "no_read" {
            need(
                step.decision == "disabled" && step.source_pos.is_none(),
                &scope,
                "NoRead selected an occurrence",
            )?;
        } else {
            need(
                step.decision != "disabled",
                &scope,
                "ordinary read arm was disabled",
            )?;
        }
        causal.push(id);
    }

    let opening = &steps[0];
    let expected_source = truth.sources.first();
    let observed_pos = opening.source_pos.map(|pos| opening.prefix_start + pos);
    let claimed_pos = opening.claim_pos.map(|pos| opening.prefix_start + pos);
    let source_correct = expected_source.is_some_and(|source| {
        observed_pos == Some(source.pos) && opening.source_id == Some(source.id)
    });
    let same_id_wrong_address = expected_source.is_some_and(|source| {
        observed_pos.is_some()
            && observed_pos != Some(source.pos)
            && opening.source_id == Some(source.id)
    });
    let opening_id_correct = actual.first() == expected.first();
    let authenticated_opening_correct = source_correct && opening.copied && opening_id_correct;
    let observed_next_id = actual.get(1).copied();
    let expected_next_id = expected.get(1).copied();
    let next_id_correct_after_opening =
        opening_id_correct && observed_next_id.is_some() && observed_next_id == expected_next_id;
    let period_without_forcing = actual.starts_with(&expected[..=truth.period_step]);
    let release_boundary = authenticated_opening_correct && period_without_forcing;
    let first_wrong = first_wrong(expected, actual);
    Ok(Grade {
        case_id: &record.case_id,
        artifact_arm: &record.artifact_arm,
        decode_arm: &record.decode_arm,
        group: &truth.group,
        family: &truth.family,
        fact_order: truth.fact_order,
        question_target: &truth.question_target,
        fenced: truth.fenced,
        supported: truth.supported,
        framing: &opening.framing,
        decision: &opening.decision,
        expected_source,
        same_id_rival: truth.same_id_rival.as_ref(),
        observed_pos,
        observed_id: opening.source_id,
        claimed_pos,
        claimed_id: opening.claimed_id,
        copied: opening.copied,
        source_correct,
        same_id_wrong_address,
        authenticated_opening_correct,
        opening_proposal: opening.native_proposal,
        opening_id: actual[0],
        expected_opening_id: expected[0],
        opening_id_correct,
        observed_next_id,
        expected_next_id,
        next_id_correct_after_opening,
        period_without_forcing,
        release_boundary,
        next_after_period: actual.get(truth.period_step + 1).copied(),
        expected_next_after_period: expected.get(truth.period_step + 1).copied(),
        exact: first_wrong.is_none(),
        first_wrong,
        stop: &client.stop,
        text: &client.text,
        generated_ids: actual,
        expected_reply: &truth.reply,
        expected_ids: expected,
    })
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [truth_path, traces_path, plain_path, output_path] = args.as_slice() else {
        return Err("usage: ordinary-source-grade TRUTH.jsonl TRACED.jsonl UNINSTRUMENTED.jsonl NEW_OUTPUT_DIR".into());
    };
    let truth = read_jsonl::<Truth>(Path::new(truth_path))?;
    let indexed: BTreeMap<_, _> = truth
        .iter()
        .map(|row| (row.case_id.as_str(), row))
        .collect();
    need(
        indexed.len() == 48 && indexed.len() == truth.len(),
        "truth",
        "unexpected or repeated cases",
    )?;
    let records = read_jsonl::<Record>(Path::new(traces_path))?;
    let plain_records = read_jsonl::<Record>(Path::new(plain_path))?;
    let mut plain = BTreeMap::new();
    let key = |row: &Record| {
        (
            row.artifact_arm.clone(),
            row.decode_arm.clone(),
            row.case_id.clone(),
        )
    };
    for row in &plain_records {
        need(
            row.execution == "uninstrumented",
            &row.case_id,
            "wrong execution tape",
        )?;
        need(
            plain.insert(key(row), &row.client).is_none(),
            &row.case_id,
            "duplicate uninstrumented record",
        )?;
    }

    let mut counts: BTreeMap<_, Counts> = BTreeMap::new();
    let mut cohorts: BTreeMap<_, BTreeMap<_, Counts>> = BTreeMap::new();
    let mut families: BTreeMap<_, BTreeMap<_, Counts>> = BTreeMap::new();
    let mut cases_per_arm = BTreeMap::<_, BTreeSet<_>>::new();
    let mut parity_per_arm = BTreeMap::<_, usize>::new();
    let mut grades = Vec::new();
    for row in &records {
        need(
            row.execution == "traced",
            &row.case_id,
            "wrong execution tape",
        )?;
        let truth = indexed
            .get(row.case_id.as_str())
            .ok_or_else(|| format!("unknown case {}", row.case_id))?;
        let arm = format!("{}/{}", row.artifact_arm, row.decode_arm);
        need(
            cases_per_arm
                .entry(arm.clone())
                .or_default()
                .insert(&row.case_id),
            &row.case_id,
            "duplicate traced record",
        )?;
        if let Some(plain) = plain.remove(&key(row)) {
            let traced = &row.client;
            need(
                plain.native_occurrence_steps.is_none()
                    && plain.same_transcript(traced)
                    && plain.artifact_sha256 == traced.artifact_sha256,
                &row.case_id,
                "trace changed ordinary transcript",
            )?;
            *parity_per_arm.entry(arm.clone()).or_default() += 1;
        } else {
            return Err(format!("{arm}/{}: missing uninstrumented record", row.case_id).into());
        }
        let grade = grade(row, truth)?;
        counts.entry(arm.clone()).or_default().add(&grade);
        cohorts
            .entry(arm.clone())
            .or_default()
            .entry(grade.group)
            .or_default()
            .add(&grade);
        families
            .entry(arm)
            .or_default()
            .entry(grade.family)
            .or_default()
            .add(&grade);
        grades.push(grade);
    }
    need(
        cases_per_arm.len() == ARTIFACTS.len() * DECODE_ARMS.len()
            && plain.is_empty()
            && cases_per_arm
                .values()
                .all(|cases| cases.len() == truth.len()),
        "coverage",
        "incomplete arm or unmatched uninstrumented record",
    )?;

    let directory = Path::new(output_path);
    fs::create_dir(directory)?;
    let mut rows = BufWriter::new(File::create(directory.join("rows.jsonl"))?);
    for row in &grades {
        serde_json::to_writer(&mut rows, row)?;
        writeln!(rows)?;
    }
    rows.flush()?;
    let release = grades
        .iter()
        .filter(|row| row.release_boundary)
        .collect::<Vec<_>>();
    let summary = serde_json::json!({
        "panel_rows": truth.len(),
        "generation_cap": CAP,
        "arms": counts,
        "cohorts": cohorts,
        "families": families,
        "uninstrumented_parity_rows": parity_per_arm,
        "all_three_artifacts_same_transcript_and_trace": count_cross_artifact_agreement(&records),
        "release_boundary_rows": release,
    });
    let mut output = BufWriter::new(File::create(directory.join("summary.json"))?);
    serde_json::to_writer_pretty(&mut output, &summary)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_answer_ids_from_wrong_occurrence_never_reach_release_gate() -> Result<()> {
        let test = Truth {
            case_id: "duplicate".to_owned(),
            group: "duplicate_fenced".to_owned(),
            family: "test".to_owned(),
            fact_order: 0,
            question_target: "0".to_owned(),
            fenced: true,
            supported: true,
            reply: " green.".to_owned(),
            prompt_ids: vec![0, 55, 2728, 28, 1074, 1074, 201, 201, 55, 28],
            sources: vec![Source {
                pos: 4,
                id: 1074,
                byte_start: 1,
                byte_end: 7,
            }],
            same_id_rival: Some(Source {
                pos: 5,
                id: 1074,
                byte_start: 8,
                byte_end: 14,
            }),
            expected_ids: vec![1074, 16, 1],
            period_step: 1,
            eos_id: 1,
        };
        let events = [(28, 1074), (1074, 16), (16, 1)]
            .into_iter()
            .enumerate()
            .map(|(i, (previous_id, committed_id))| {
                let opening = i == 0;
                Step {
                    prefix_start: 0,
                    prefix_len: test.prompt_ids.len() + i,
                    previous_id: Some(previous_id),
                    framing: if opening { "parsed" } else { "rejected" }.to_owned(),
                    parsed_start: opening.then_some(4),
                    parsed_end: opening.then_some(6),
                    decision: if opening {
                        "selected"
                    } else {
                        "framing_rejected"
                    }
                    .to_owned(),
                    source_pos: opening.then_some(5),
                    source_id: opening.then_some(1074),
                    claim_pos: opening.then_some(5),
                    claimed_id: opening.then_some(1074),
                    gate_id: 1074,
                    native_proposal: committed_id,
                    committed_id,
                    copied: opening,
                    source_admitted: opening,
                    other_words: BTreeMap::new(),
                }
            })
            .collect();
        let mut record = Record {
            case_id: test.case_id.clone(),
            artifact_arm: ARTIFACTS[0].0.to_owned(),
            decode_arm: "authenticated_copy".to_owned(),
            execution: "traced".to_owned(),
            client: Client {
                context: 192,
                artifact_sha256: ARTIFACTS[0].1.to_owned(),
                tokenizer_sha256: TOKENIZER.to_owned(),
                prompt_ids: test.prompt_ids.clone(),
                generated_ids: test.expected_ids.clone(),
                text: test.reply.clone(),
                stop: "eos".to_owned(),
                native_occurrence_steps: Some(events),
            },
        };

        let wrong = grade(&record, &test)?;
        assert!(wrong.exact && wrong.copied && wrong.same_id_wrong_address);
        assert!(wrong.period_without_forcing);
        assert!(!wrong.source_correct && !wrong.release_boundary);

        let events = record
            .client
            .native_occurrence_steps
            .as_mut()
            .ok_or("missing test events")?;
        events[0].source_pos = Some(4);
        events[0].claim_pos = Some(4);
        let correct = grade(&record, &test)?;
        assert!(correct.release_boundary);
        assert_eq!(correct.next_after_period, Some(1));

        let events = record
            .client
            .native_occurrence_steps
            .as_mut()
            .ok_or("missing test events")?;
        events[1].previous_id = Some(0);
        assert!(grade(&record, &test).is_err());
        Ok(())
    }
}
