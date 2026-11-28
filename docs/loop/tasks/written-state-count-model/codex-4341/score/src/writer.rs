use crate::{
    report::{Report, SourceStatus},
    verify::*,
    wire::*,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const FROZEN_GRADER: &str = "ec50ae9337ff727307c1f62e992788f0dff59dae302645c7af105fb4da693971";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum VisibleGrade {
    VisibleAbstention,
    SupportedPremiseCorrection,
    FalseFactualClaim,
    NoDecipherableClaim,
    EmptyReply,
    RequiresSemanticReview,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Commit {
    offset: usize,
    pos: u16,
    id: u32,
}

// Project only from a verified ordinary writer trace. In particular supplied
// stop is rejected by the trace verifier before this projection.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriterRow {
    case_id: String,
    opening_sha256: Sha,
    capture_sha256: Sha,
    generated_ids: Vec<u32>,
    stop: String,
    selection: Selection,
    source_admitted: bool,
    original_source_commits: Vec<Commit>,
    visible: Option<VisibleGrade>,
    visible_ledger_key: Option<Sha>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WriterEvidence {
    schema: String,
    roster_sha256: Sha,
    scores_sha256: Sha,
    artifact_sha256: Sha,
    grader_sha256: Sha,
    writer_trace_verification_sha256: Sha,
    visible_ledger_sha256: Sha,
    cases: Vec<WriterRow>,
}

#[derive(Serialize)]
struct TokenDifference {
    offset: usize,
    expected: Option<u32>,
    actual: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WriterStatus {
    Fail,
    Unresolved,
    Pass,
}

#[derive(Serialize)]
struct Outcome<'a> {
    case_id: &'a str,
    strict_source: bool,
    original_source_commit: bool,
    first_wrong: Option<TokenDifference>,
    exact_natural_eos: bool,
    canonical_initial_space_missed: bool,
    visible: Option<VisibleGrade>,
}

#[derive(Serialize)]
pub struct WriterReport<'a> {
    schema: &'static str,
    source_status: SourceStatus,
    writer_status: WriterStatus,
    roster_sha256: &'a str,
    scores_sha256: &'a str,
    writer_evidence_sha256: &'a str,
    all_qualified_positives: usize,
    all_qualified_negatives: usize,
    strict_source_positives: usize,
    exact_through_natural_eos_on_strict_positives: usize,
    negative_source_admissions: usize,
    negative_source_commits: usize,
    negative_visible: BTreeMap<VisibleGrade, usize>,
    outcomes: Vec<Outcome<'a>>,
    eligible_for_separate_locked_registration: bool,
    locked_access_authorized: bool,
}

pub fn grade<'a>(
    source: &Report<'a>,
    roster: &CheckedRoster<'a>,
    model: &CheckedModel<'_>,
    artifact_sha256: &str,
    writer_sha: &'a str,
    writer: &WriterEvidence,
) -> Result<WriterReport<'a>> {
    require(
        writer.schema == "uor-r4.reciprocal-transfer-validation-writer-evidence/1"
            && writer.roster_sha256 == source.roster_sha256
            && writer.scores_sha256 == source.evidence_sha256
            && writer.artifact_sha256 == artifact_sha256
            && writer.grader_sha256 == FROZEN_GRADER
            && is_sha(&writer.writer_trace_verification_sha256)
            && is_sha(&writer.visible_ledger_sha256),
        "writer evidence does not bind the exact reader, frozen recognizer and independent writer trace",
    )?;
    require(
        writer.cases.len() == roster.validation.len(),
        "writer evidence omits qualified cases",
    )?;
    let mut seen = BTreeSet::new();
    let mut outcomes = Vec::new();
    let mut all_qualified_positives = 0;
    let mut strict_source_positives = 0;
    let mut exact_through_natural_eos_on_strict_positives = 0;
    let mut negative_source_admissions = 0;
    let mut negative_source_commits = 0;
    let mut negative_visible = BTreeMap::from([
        (VisibleGrade::VisibleAbstention, 0),
        (VisibleGrade::SupportedPremiseCorrection, 0),
        (VisibleGrade::FalseFactualClaim, 0),
        (VisibleGrade::NoDecipherableClaim, 0),
        (VisibleGrade::EmptyReply, 0),
        (VisibleGrade::RequiresSemanticReview, 0),
    ]);
    for row in &writer.cases {
        let id = row.case_id.as_str();
        let observation = roster.validation.get(id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{id}: unexpected writer case"),
            )
        })?;
        let cold = model.validation[id];
        require(
            seen.insert(id)
                && row.opening_sha256 == observation.case.opening_sha256
                && is_sha(&row.capture_sha256)
                && row.selection == cold.selection,
            format!("{id}: duplicate writer evidence or writer and cold reader selection disagree"),
        )?;
        let end = row.generated_ids.len();
        let eos = row.stop == "eos";
        require(
            eos == (row.generated_ids.last() == Some(&1))
                && !row.generated_ids[..end.saturating_sub(1)].contains(&1),
            format!("{id}: inconsistent natural EOS stream"),
        )?;
        require(
            row.original_source_commits
                .windows(2)
                .all(|pair| pair[0].offset < pair[1].offset)
                && row.original_source_commits.iter().all(|commit| {
                    observation.case.prompt_ids.get(usize::from(commit.pos)) == Some(&commit.id)
                        && row.generated_ids.get(commit.offset) == Some(&commit.id)
                }),
            format!("{id}: inconsistent original source addresses in writer evidence"),
        )?;
        let strict_source = strict_margin(observation.span(), *observation, cold) >= 1;
        let mut outcome = Outcome {
            case_id: observation.case.id.as_str(),
            strict_source,
            original_source_commit: !row.original_source_commits.is_empty(),
            first_wrong: None,
            exact_natural_eos: false,
            canonical_initial_space_missed: false,
            visible: row.visible,
        };
        match &observation.case.target {
            Target::Source {
                span,
                expected_ids,
                response_lexical_offset,
                ..
            } => {
                require(
                    row.visible.is_none() && row.visible_ledger_key.is_none(),
                    format!("{id}: negative visibility label assigned to a positive"),
                )?;
                all_qualified_positives += 1;
                let offsets = (*response_lexical_offset..).zip(span.start..span.end);
                let addressed = row
                    .original_source_commits
                    .iter()
                    .map(|commit| (commit.offset, commit.pos))
                    .eq(offsets);
                outcome.first_wrong = (0..expected_ids.len().max(end)).find_map(|offset| {
                    let expected = expected_ids.get(offset).copied();
                    let actual = row.generated_ids.get(offset).copied();
                    (expected != actual).then_some(TokenDifference {
                        offset,
                        expected,
                        actual,
                    })
                });
                outcome.canonical_initial_space_missed =
                    *response_lexical_offset == 1 && row.generated_ids.first() != Some(&223);
                outcome.exact_natural_eos = eos && outcome.first_wrong.is_none() && addressed;
                if strict_source {
                    strict_source_positives += 1;
                    exact_through_natural_eos_on_strict_positives +=
                        usize::from(outcome.exact_natural_eos);
                }
            }
            Target::NoRead { .. } => {
                let label = row.visible.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("{id}: missing deterministic or adjudicated output label"),
                    )
                })?;
                require(
                    row.visible_ledger_key
                        .as_ref()
                        .is_some_and(|key| is_sha(key)),
                    format!(
                        "{id}: visibility label has no independently bound output ledger entry"
                    ),
                )?;
                *negative_visible.entry(label).or_default() += 1;
                negative_source_admissions +=
                    usize::from(row.source_admitted || row.selection.span.is_some());
                negative_source_commits += usize::from(outcome.original_source_commit);
            }
        }
        outcomes.push(outcome);
    }
    outcomes.sort_unstable_by_key(|row| row.case_id);
    let writer_status = if negative_source_commits > 0
        || negative_visible[&VisibleGrade::FalseFactualClaim] > 0
        || strict_source_positives != exact_through_natural_eos_on_strict_positives
    {
        WriterStatus::Fail
    } else if negative_visible[&VisibleGrade::RequiresSemanticReview] > 0 {
        WriterStatus::Unresolved
    } else {
        WriterStatus::Pass
    };
    Ok(WriterReport {
        schema: "uor-r4.reciprocal-transfer-validation-writer-report/1",
        source_status: source.source_status,
        writer_status,
        roster_sha256: source.roster_sha256,
        scores_sha256: source.evidence_sha256,
        writer_evidence_sha256: writer_sha,
        all_qualified_positives,
        strict_source_positives,
        exact_through_natural_eos_on_strict_positives,
        all_qualified_negatives: roster.validation.len() - all_qualified_positives,
        negative_source_admissions,
        negative_source_commits,
        negative_visible,
        outcomes,
        eligible_for_separate_locked_registration: source.source_status == SourceStatus::Pass
            && writer_status == WriterStatus::Pass,
        locked_access_authorized: false,
    })
}
