use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{self, Write},
    path::Path,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::{
    ByteBpeTokenizer,
    dialogue::{DialogueEncoder, DialogueProtocol, EncodedUserOpening},
};

mod contrast;
mod oracle;
mod tape;

use contrast::{BANKS, Contrast, Keys, RELATION_BANK, cancel, difference, obstruction};
use oracle::{RelationOracle, Roots};
use tape::{ARMS, Endpoints, Tape};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "uor-r4/directed-relation-structural/1";

fn check(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pinned(path: impl AsRef<Path>, expected: &str) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if digest(&bytes) != expected {
        return Err(format!("SHA-256 mismatch for {}", path.display()).into());
    }
    Ok(bytes)
}

fn jsonl<T: DeserializeOwned>(bytes: &[u8]) -> Result<Vec<T>> {
    std::str::from_utf8(bytes)?
        .lines()
        .enumerate()
        .map(|(line, raw)| {
            serde_json::from_str(raw)
                .map_err(|error| format!("JSONL line {}: {error}", line + 1).into())
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Blind {
    case_id: String,
    text: String,
}

#[derive(Serialize)]
struct Admission<'a> {
    case_id: &'a str,
    tokens: Option<usize>,
    refusal: Option<&'static str>,
}

fn admit_opening<'a>(
    prompt: &'a Blind,
    encoder: &DialogueEncoder<'_>,
    roots: &Roots,
) -> (Admission<'a>, Option<EncodedUserOpening>) {
    let opening = encoder.encode_single_user_opening(&prompt.text);
    let tokens = opening.as_ref().map(|opening| opening.tokens().len());
    // As in the ordinary client's validate_prefix, the capacity applies
    // to the complete protocol-2 token sequence, including role markers.
    let refusal = match &opening {
        None => Some("invalid-protocol-2-opening"),
        Some(opening) if opening.tokens().is_empty() => Some("empty-prefix"),
        Some(opening) if opening.tokens().len() > roots.context => Some("prefix-exceeds-context"),
        Some(opening)
            if opening
                .tokens()
                .iter()
                .any(|&id| roots.block_2(id).is_none()) =>
        {
            Some("prefix-token-outside-vocabulary")
        }
        Some(_) => None,
    };
    (
        Admission {
            case_id: &prompt.case_id,
            tokens,
            refusal,
        },
        if refusal.is_none() { opening } else { None },
    )
}

#[derive(Debug, Deserialize)]
struct Truth {
    case_id: String,
    split: String,
    article: String,
    paragraph: usize,
    passage_sha256: String,
    passage_bytes: usize,
    question_source_id: String,
    family: String,
    question: String,
    original_bytes: Option<[usize; 2]>,
    original_char: Option<[usize; 2]>,
    span_text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    group_id: String,
    split: String,
    stratum: String,
    article: String,
    paragraph: usize,
    passage_sha256: String,
    families: Vec<Vec<String>>,
    contrast: String,
}

fn original_passage<'a>(blind: &'a Blind, truth: &Truth) -> Result<&'a str> {
    check(
        blind.case_id == truth.case_id
            && !truth.question_source_id.is_empty()
            && !truth.question.is_empty()
            && !truth.family.is_empty(),
        "invalid case or official QA identity",
    )?;
    let passage = blind
        .text
        .get(..truth.passage_bytes)
        .ok_or("invalid original passage boundary")?;
    check(
        digest(passage.as_bytes()) == truth.passage_sha256,
        "original passage digest differs",
    )?;
    let tail = blind
        .text
        .get(truth.passage_bytes..)
        .ok_or("invalid original question boundary")?;
    check(
        tail.strip_prefix(' ') == Some(truth.question.as_str()),
        "original question bytes differ",
    )?;
    match (
        truth.original_bytes,
        truth.original_char,
        truth.span_text.as_deref(),
    ) {
        (Some([start, end]), Some([char_start, char_end]), Some(surface)) => {
            check(
                end > start && passage.get(start..end) == Some(surface),
                "original answer bytes differ",
            )?;
            let start_char = passage[..start].chars().count();
            check(
                char_start == start_char && char_end == start_char + surface.chars().count(),
                "original answer code-point positions differ",
            )?;
        }
        (None, None, None) => {}
        _ => return Err("partial exact source interval or NoRead label".into()),
    }
    Ok(passage)
}

fn grouping(groups: &[Group], truth: &BTreeMap<&str, &Truth>) -> Result<()> {
    let mut seen_cases = BTreeSet::new();
    let mut seen_qa = BTreeSet::new();
    let mut seen_groups = BTreeSet::new();
    let mut seen_passages = BTreeSet::new();
    let mut article_split = BTreeMap::new();
    for group in groups {
        check(seen_groups.insert(&group.group_id), "repeated group ID")?;
        check(
            seen_passages.insert(&group.passage_sha256),
            "repeated passage in sealed selection",
        )?;
        check(
            matches!(group.split.as_str(), "screen" | "construction"),
            "unregistered split",
        )?;
        let previous_split = article_split.insert(&group.article, &group.split);
        check(
            previous_split.is_none_or(|previous| previous == &group.split),
            "article leaks across splits",
        )?;
        let sizes: Vec<_> = group.families.iter().map(Vec::len).collect();
        let valid_stratum = match group.stratum.as_str() {
            "quad" => sizes == [2, 2],
            "triple" => sizes == [2, 1],
            "switch" => sizes == [1, 1],
            "paraphrase" => sizes == [2],
            _ => false,
        };
        let valid_contrast = if sizes.len() == 1 {
            group.contrast == "paraphrase"
        } else {
            matches!(
                group.contrast.as_str(),
                "target-switch" | "same-fact-inverse" | "answerability"
            )
        };
        check(
            valid_stratum && valid_contrast,
            "unregistered group shape and contrast",
        )?;
        let mut distinct_questions = BTreeSet::new();
        let mut targets = Vec::new();
        for family in &group.families {
            let mut family_target = None;
            for id in family {
                let case = truth
                    .get(id.as_str())
                    .ok_or("group case missing from exact source truth")?;
                check(
                    seen_cases.insert(id.as_str()) && seen_qa.insert(&case.question_source_id),
                    "case or official QA identity reused",
                )?;
                check(
                    case.split == group.split
                        && case.article == group.article
                        && case.paragraph == group.paragraph
                        && case.passage_sha256 == group.passage_sha256,
                    "group crosses original article or passage",
                )?;
                check(
                    distinct_questions.insert(&case.question),
                    "identical question reused in group",
                )?;
                let expected = family_target.get_or_insert(case.original_bytes);
                check(
                    *expected == case.original_bytes,
                    "same target family points to different source bytes",
                )?;
            }
            targets.push(family_target.flatten());
        }
        if let [left, right] = targets.as_slice() {
            let valid_targets = match (*left, *right) {
                (Some(a), Some(b)) => {
                    group.contrast != "answerability" && (a[1] <= b[0] || b[1] <= a[0])
                }
                (Some(_), None) | (None, Some(_)) => group.contrast == "answerability",
                _ => false,
            };
            check(
                valid_targets,
                "cross-family exact source intervals are not disjoint",
            )?;
        }
    }
    check(
        seen_cases.len() == truth.len(),
        "truth includes unassigned cases",
    )
}

fn indexed<T>(rows: &[T], id: impl Fn(&T) -> &str) -> Result<BTreeMap<&str, &T>> {
    let mut result = BTreeMap::new();
    for row in rows {
        check(
            result.insert(id(row), row).is_none(),
            "duplicate input case",
        )?;
    }
    Ok(result)
}

type OriginalAddress = std::result::Result<Option<Span>, &'static str>;

fn address(
    source_label: Option<[usize; 2]>,
    passage: &str,
    passing: &Tape,
    token_bytes: &[Vec<u8>],
    ids: &[u32],
) -> OriginalAddress {
    let Some([start, end]) = source_label else {
        return Ok(None);
    };
    // Independently encoded protocol-2 content starts with one ASCII space.
    // Allow precisely the original adjacent BPE space, never a trimmed or
    // shifted answer. A source span inside a token remains unreachable.
    let offsets: Vec<_> = std::iter::once(0)
        .chain(
            token_bytes[passing.user[0]..passing.user[1]]
                .iter()
                .scan(0, |offset, token| {
                    *offset += token.len();
                    Some(*offset)
                }),
        )
        .collect();
    let fits_start = |first| {
        first == start + 1
            || (first == start && (start == 0 || passage.as_bytes()[start - 1] == b' '))
    };
    for candidate in &passing.candidates {
        let first = offsets[candidate.start - passing.user[0]];
        let last = offsets[candidate.end - passing.user[0]];
        if last == end + 1 && fits_start(first) {
            return Ok(Some(candidate.span()));
        }
    }
    let left = offsets.iter().rposition(|&first| fits_start(first));
    let right = offsets.iter().position(|&last| last == end + 1);
    let (first, last) = match (left, right) {
        (None, None) => return Err("both-exact-byte-edges-inside-tokens"),
        (None, Some(_)) => return Err("exact-left-byte-edge-inside-token"),
        (Some(_), None) => return Err("exact-right-byte-edge-inside-token"),
        (Some(first), Some(last)) => (passing.user[0] + first, passing.user[0] + last),
    };
    if first < passing.fact[0] || last > passing.fact[1] {
        return Err("outside-original-fact-token-range");
    }
    if first >= last {
        return Err("empty-original-token-span");
    }
    if last - first > tape::MAX_SPAN {
        return Err("original-token-span-exceeds-eight");
    }
    if token_bytes[first..last]
        .iter()
        .flatten()
        .copied()
        .any(tape::terminal)
    {
        return Err("original-token-span-intersects-clause-delimiter");
    }
    if ids[first..last].iter().any(|&id| !tape::copyable(id)) {
        return Err("original-token-span-has-uncopyable-id");
    }
    Err("exact-token-edges-are-not-native-lexical-edges")
}

struct Case<'a> {
    truth: &'a Truth,
    passage: &'a str,
    rows: Vec<&'a Tape>,
    bytes: Vec<Vec<u8>>,
    ids: Vec<u32>,
    original_text_preserved: bool,
}

impl Case<'_> {
    fn fact_ids(&self) -> &[u32] {
        let bounds = self.rows[0].fact;
        &self.ids[bounds[0]..bounds[1]]
    }

    fn original_address(&self, label: Option<[usize; 2]>) -> OriginalAddress {
        if !self.original_text_preserved {
            return Err("protocol-changed-original-bytes");
        }
        address(label, self.passage, self.rows[0], &self.bytes, &self.ids)
    }

    fn candidate_list(&self) -> impl Iterator<Item = Span> + '_ {
        self.rows[0].candidates.iter().map(tape::Candidate::span)
    }
}

fn key_and_endpoints(tape: &Tape, address: Option<Span>) -> Result<(&Keys, Option<&Endpoints>)> {
    let Some(address) = address else {
        return Ok((&tape.no_read_keys, None));
    };
    let index = tape
        .candidates
        .binary_search_by_key(&(address.start, address.end), |candidate| {
            (candidate.start, candidate.end)
        })
        .map_err(|_| "verified target disappeared")?;
    Ok((
        &tape.candidates[index].keys,
        tape.relation_endpoints.as_ref().map(|all| &all[index]),
    ))
}

fn rivals(
    tape: &Tape,
    span: Option<Span>,
    mate: Option<Option<Span>>,
    banks: &[usize],
) -> Result<(usize, bool)> {
    let (keys, _) = key_and_endpoints(tape, span)?;
    let same = |other: &Keys| banks.iter().all(|&bank| keys[bank] == other[bank]);
    let source_aliases = tape
        .candidates
        .iter()
        .filter(|candidate| {
            Some(candidate.span()) != span
                && Some(Some(candidate.span())) != mate
                && same(&candidate.keys)
        })
        .count();
    let no_read_alias = span.is_some() && mate != Some(None) && same(&tape.no_read_keys);
    Ok((source_aliases, no_read_alias))
}

type Side<'a> = (&'a Case<'a>, Option<Span>, Option<Option<Span>>);

#[derive(Serialize)]
struct BankSummary {
    equal_openings: usize,
    reverse_comparisons: usize,
    reverse_pairs_canceling: usize,
    reverse_pairs_distinct_nonzero: usize,
    reverse_pairs_compatible: usize,
    repeated_target_pairs: usize,
    paraphrase_target_key_equal: usize,
    paraphrase_both_action_keys_equal: usize,
    paraphrase_contrasts_equal: usize,
    all_wording_strict_compatible: Option<bool>,
    minimal_obstruction_case_ids: Vec<String>,
    source_swap_changed_contrasts: usize,
    target_equal_third_sources: usize,
    target_equal_no_read_sides: usize,
}

fn bank_summary(
    sides: &[Side<'_>],
    families: &[Vec<String>],
    arm: usize,
    banks: &[usize],
) -> Result<BankSummary> {
    let mut target_keys = Vec::new();
    let mut mate_keys = Vec::new();
    let mut contrasts = Vec::new();
    let mut source_swap_changed_contrasts = 0;
    let mut target_equal_third_sources = 0;
    let mut target_equal_no_read_sides = 0;
    for &(case, target, mate) in sides {
        let tape = case.rows[arm];
        let (own, _) = key_and_endpoints(tape, target)?;
        let other = mate
            .map(|span| key_and_endpoints(tape, span))
            .transpose()?
            .map(|(keys, _)| keys);
        target_keys.push(own);
        mate_keys.push(other);
        if let (Some(mate), Some(other)) = (mate, other) {
            let terms = difference(own, other, banks);
            if arm == 2 {
                let (swap_own, _) = key_and_endpoints(case.rows[4], target)?;
                let (swap_mate, _) = key_and_endpoints(case.rows[4], mate)?;
                source_swap_changed_contrasts +=
                    usize::from(terms != difference(swap_own, swap_mate, banks));
            }
            contrasts.push(terms);
        }
        let (source_aliases, no_read_alias) = rivals(tape, target, mate, banks)?;
        target_equal_third_sources += source_aliases;
        target_equal_no_read_sides += usize::from(no_read_alias);
    }
    let equivalent = |a: &Keys, b: &Keys| banks.iter().all(|&bank| a[bank] == b[bank]);
    let mut offset = 0;
    let mut repeated_target_pairs = 0;
    let mut paraphrase_target_key_equal = 0;
    let mut paraphrase_both_action_keys_equal = 0;
    let mut paraphrase_contrasts_equal = 0;
    for family in families {
        if family.len() == 2 {
            repeated_target_pairs += 1;
            let own_stable = equivalent(target_keys[offset], target_keys[offset + 1]);
            paraphrase_target_key_equal += usize::from(own_stable);
            if let (Some(a), Some(b)) = (mate_keys[offset], mate_keys[offset + 1]) {
                paraphrase_both_action_keys_equal += usize::from(own_stable && equivalent(a, b));
                paraphrase_contrasts_equal +=
                    usize::from(contrasts[offset] == contrasts[offset + 1]);
            }
        }
        offset += family.len();
    }
    let mut reverse_comparisons = 0;
    let mut reverse_pairs_canceling = 0;
    let mut reverse_pairs_distinct_nonzero = 0;
    let mut reverse_pairs_compatible = 0;
    let mut all_wording_strict_compatible = None;
    let mut minimal_obstruction_case_ids = Vec::new();
    if !contrasts.is_empty() {
        for left in &contrasts[..families[0].len()] {
            for right in &contrasts[families[0].len()..] {
                reverse_comparisons += 1;
                reverse_pairs_canceling += usize::from(cancel(left, right));
                reverse_pairs_distinct_nonzero +=
                    usize::from(left != right && !left.is_empty() && !right.is_empty());
                reverse_pairs_compatible += usize::from(obstruction(&[left, right]).is_none());
            }
        }
        let rows: Vec<&Contrast> = contrasts.iter().collect();
        let contradiction = obstruction(&rows);
        all_wording_strict_compatible = Some(contradiction.is_none());
        minimal_obstruction_case_ids = contradiction
            .into_iter()
            .flatten()
            .map(|index| sides[index].0.truth.case_id.clone())
            .collect();
    }
    Ok(BankSummary {
        equal_openings: contrasts.iter().filter(|row| row.is_empty()).count(),
        reverse_comparisons,
        reverse_pairs_canceling,
        reverse_pairs_distinct_nonzero,
        reverse_pairs_compatible,
        repeated_target_pairs,
        paraphrase_target_key_equal,
        paraphrase_both_action_keys_equal,
        paraphrase_contrasts_equal,
        all_wording_strict_compatible,
        minimal_obstruction_case_ids,
        source_swap_changed_contrasts,
        target_equal_third_sources,
        target_equal_no_read_sides,
    })
}

#[derive(Serialize)]
struct ArmGrade {
    all_banks: BankSummary,
    bank_10: BankSummary,
}

#[derive(Default, Serialize)]
struct Missing {
    target_queries: usize,
    target_sources: usize,
    target_either: usize,
    mate_queries: usize,
    mate_sources: usize,
    mate_either: usize,
}

#[derive(Serialize)]
struct GroupGrade<'a> {
    group_id: &'a str,
    split: &'a str,
    stratum: &'a str,
    article: &'a str,
    paragraph: usize,
    contrast: &'a str,
    truth_cases: usize,
    labeled_no_read: usize,
    original_target_unreachable: usize,
    original_mate_unreachable: usize,
    unreachable_address_reasons: Vec<Unreachable<'a>>,
    different_fact_token_prefix: bool,
    different_candidate_spans: bool,
    original_text_changed_by_protocol: usize,
    opening_admissions: Vec<&'a Admission<'a>>,
    missing: Missing,
    no_read_choices_seen: usize,
    source_choices_seen: usize,
    all_source_choices_missing_query: usize,
    all_source_choices_missing_source: usize,
    all_source_choices_missing_either: usize,
    arms: BTreeMap<&'static str, ArmGrade>,
    registered_group_pass: bool,
    case_source_ids: Vec<&'a str>,
}

#[derive(Serialize)]
struct Unreachable<'a> {
    case_id: &'a str,
    action: &'static str,
    reason: &'static str,
}

fn grade_group<'a>(
    group: &'a Group,
    cases: &BTreeMap<&str, Case<'a>>,
    truth: &BTreeMap<&str, &'a Truth>,
    admissions: &'a BTreeMap<&str, Admission<'a>>,
) -> Result<GroupGrade<'a>> {
    let mut grade = GroupGrade {
        group_id: &group.group_id,
        split: &group.split,
        stratum: &group.stratum,
        article: &group.article,
        paragraph: group.paragraph,
        contrast: &group.contrast,
        truth_cases: 0,
        labeled_no_read: 0,
        original_target_unreachable: 0,
        original_mate_unreachable: 0,
        unreachable_address_reasons: Vec::new(),
        different_fact_token_prefix: false,
        different_candidate_spans: false,
        original_text_changed_by_protocol: 0,
        opening_admissions: Vec::new(),
        missing: Missing::default(),
        no_read_choices_seen: 0,
        source_choices_seen: 0,
        all_source_choices_missing_query: 0,
        all_source_choices_missing_source: 0,
        all_source_choices_missing_either: 0,
        arms: BTreeMap::new(),
        registered_group_pass: false,
        case_source_ids: Vec::new(),
    };
    let baseline = group
        .families
        .iter()
        .flatten()
        .find_map(|id| cases.get(id.as_str()));
    let mut sides = Vec::new();
    for (family_index, family) in group.families.iter().enumerate() {
        let mate_family_target = if group.families.len() == 2 {
            Some(truth[group.families[1 - family_index][0].as_str()].original_bytes)
        } else {
            None
        };
        for id in family {
            let original = truth[id.as_str()];
            grade.truth_cases += 1;
            grade.case_source_ids.push(&original.question_source_id);
            grade.labeled_no_read += usize::from(original.original_bytes.is_none());
            grade.opening_admissions.push(&admissions[id.as_str()]);
            let Some(case) = cases.get(id.as_str()) else {
                continue;
            };
            grade.original_text_changed_by_protocol += usize::from(!case.original_text_preserved);
            grade.no_read_choices_seen += 1;
            grade.source_choices_seen += case.rows[0].candidates.len();
            if let Some(endpoints) = &case.rows[2].relation_endpoints {
                grade.all_source_choices_missing_query +=
                    endpoints.iter().filter(|e| e.missing_query()).count();
                grade.all_source_choices_missing_source +=
                    endpoints.iter().filter(|e| e.missing_source()).count();
                grade.all_source_choices_missing_either +=
                    endpoints.iter().filter(|e| e.missing()).count();
            }
            if let Some(baseline) = baseline {
                grade.different_fact_token_prefix |= case.fact_ids() != baseline.fact_ids();
                grade.different_candidate_spans |=
                    case.candidate_list().ne(baseline.candidate_list());
            }
            let target = case.original_address(case.truth.original_bytes);
            let mate = mate_family_target
                .map(|label| case.original_address(label))
                .transpose();
            if let Err(reason) = target {
                grade.original_target_unreachable += 1;
                grade.unreachable_address_reasons.push(Unreachable {
                    case_id: id,
                    action: "target",
                    reason,
                });
            }
            if let Err(reason) = mate {
                grade.original_mate_unreachable += 1;
                grade.unreachable_address_reasons.push(Unreachable {
                    case_id: id,
                    action: "mate",
                    reason,
                });
            }
            if let Ok(target) = target
                && let (_, Some(endpoints)) = key_and_endpoints(case.rows[2], target)?
            {
                grade.missing.target_queries += usize::from(endpoints.missing_query());
                grade.missing.target_sources += usize::from(endpoints.missing_source());
                grade.missing.target_either += usize::from(endpoints.missing());
            }
            if let Ok(Some(mate)) = mate
                && let (_, Some(endpoints)) = key_and_endpoints(case.rows[2], mate)?
            {
                grade.missing.mate_queries += usize::from(endpoints.missing_query());
                grade.missing.mate_sources += usize::from(endpoints.missing_source());
                grade.missing.mate_either += usize::from(endpoints.missing());
            }
            if let (Ok(target), Ok(mate)) = (target, mate) {
                sides.push((case, target, mate));
            }
        }
    }
    if sides.len() != grade.truth_cases
        || grade.different_fact_token_prefix
        || grade.different_candidate_spans
    {
        return Ok(grade);
    }
    let banks: Vec<_> = (0..BANKS).collect();
    for (arm, &name) in ARMS.iter().enumerate() {
        grade.arms.insert(
            name,
            ArmGrade {
                all_banks: bank_summary(&sides, &group.families, arm, &banks)?,
                bank_10: bank_summary(&sides, &group.families, arm, &[RELATION_BANK])?,
            },
        );
    }
    let proposed = &grade.arms[ARMS[2]].bank_10;
    grade.registered_group_pass = group.split == "screen"
        && group.stratum == "quad"
        && group.contrast == "same-fact-inverse"
        && grade.missing.target_either == 0
        && grade.missing.mate_either == 0
        && proposed.all_wording_strict_compatible == Some(true)
        && proposed.source_swap_changed_contrasts > 0;
    Ok(grade)
}

fn print_json(output: &serde_json::Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, output)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn model_inputs(
    base: &str,
    tokenizer_path: &str,
    tokenizer_sha: &str,
) -> Result<(Roots, ByteBpeTokenizer)> {
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&pinned(tokenizer_path, tokenizer_sha)?)
            .ok_or("invalid original tokenizer")?;
    let roots = Roots::from_base(Path::new(base), tokenizer_sha)?;
    Ok((roots, tokenizer))
}

fn preflight(
    blind_path: &str,
    blind_sha: &str,
    base: &str,
    tokenizer_path: &str,
    tokenizer_sha: &str,
) -> Result<()> {
    let blind: Vec<Blind> = jsonl(&pinned(blind_path, blind_sha)?)?;
    indexed(&blind, |row| &row.case_id)?;
    let (roots, tokenizer) = model_inputs(base, tokenizer_path, tokenizer_sha)?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let mut admitted = Vec::new();
    let mut openings = Vec::new();
    for prompt in &blind {
        let (admission, opening) = admit_opening(prompt, &encoder, &roots);
        openings.push(admission);
        if opening.is_some() {
            admitted.push(prompt);
        }
    }
    print_json(&serde_json::json!({
        "schema": "uor-r4/directed-relation-admission/1",
        "artifact_context": roots.context,
        "cases": openings,
        "admitted": admitted,
    }))
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if let [command, blind, blind_sha, base, tokenizer, tokenizer_sha] = args.as_slice()
        && command == "preflight"
    {
        return preflight(blind, blind_sha, base, tokenizer, tokenizer_sha);
    }
    let [
        stage,
        blind_sha,
        truth_sha,
        groups_sha,
        base_path,
        tokenizer_path,
        tokenizer_sha,
        old_path,
        old_sha,
        new_path,
        new_sha,
    ] = args.as_slice()
    else {
        return Err("usage: directed-relation-probe STAGE BLIND_SHA TRUTH_SHA GROUPS_SHA BASE.d11 TOKENIZER TOKENIZER_SHA OLD.jsonl OLD_SHA NEW.jsonl NEW_SHA".into());
    };
    let stage = Path::new(stage);
    let blind: Vec<Blind> = jsonl(&pinned(stage.join("blind.jsonl"), blind_sha)?)?;
    let truth: Vec<Truth> = jsonl(&pinned(stage.join("truth.jsonl"), truth_sha)?)?;
    let groups: Vec<Group> = jsonl(&pinned(stage.join("groups.jsonl"), groups_sha)?)?;
    let (roots, tokenizer) = model_inputs(base_path, tokenizer_path, tokenizer_sha)?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let old_rows: Vec<Tape> = jsonl(&pinned(old_path, old_sha)?)?;
    let new_rows: Vec<Tape> = jsonl(&pinned(new_path, new_sha)?)?;
    let mut old_capture = old_rows.chunks_exact(2);
    let mut new_capture = new_rows.chunks_exact(ARMS.len());
    check(
        old_capture.remainder().is_empty() && new_capture.remainder().is_empty(),
        "missing or trailing capture cases",
    )?;
    let truth = indexed(&truth, |row| &row.case_id)?;
    indexed(&blind, |row| &row.case_id)?;
    check(
        truth.len() == blind.len(),
        "truth and blind cardinalities differ",
    )?;
    grouping(&groups, &truth)?;
    let oracle = RelationOracle::new()?;
    let mut cases = BTreeMap::new();
    let mut admissions = BTreeMap::new();
    for prompt in &blind {
        let truth = truth
            .get(prompt.case_id.as_str())
            .ok_or("blind case missing exact truth")?;
        let passage = original_passage(prompt, truth)?;
        let (admission, opening) = admit_opening(prompt, &encoder, &roots);
        admissions.insert(prompt.case_id.as_str(), admission);
        let Some(opening) = opening else {
            continue;
        };
        let old: Vec<_> = old_capture
            .next()
            .ok_or("missing client capture")?
            .iter()
            .collect();
        let rows: Vec<_> = new_capture
            .next()
            .ok_or("missing client capture")?
            .iter()
            .collect();
        check(
            old.iter()
                .chain(&rows)
                .all(|row| row.case_id == prompt.case_id),
            "captured case order or prompt ID differs",
        )?;
        tape::verify(&opening, &tokenizer, &old, &rows, &roots, &oracle)
            .map_err(|error| format!("{}: {error}", prompt.case_id))?;
        let ids = opening.tokens().to_vec();
        let bytes: Vec<_> = ids
            .iter()
            .map(|&id| tokenizer.decode_bytes(&[id]))
            .collect();
        let encoded_content = bytes[opening.user_range()].concat();
        let original_text_preserved =
            encoded_content.strip_prefix(b" ") == Some(prompt.text.as_bytes());
        cases.insert(
            prompt.case_id.as_str(),
            Case {
                truth,
                passage,
                rows,
                bytes,
                ids,
                original_text_preserved,
            },
        );
    }
    check(
        old_capture.next().is_none() && new_capture.next().is_none(),
        "client capture includes refused or unknown cases",
    )?;
    let grades: Vec<_> = groups
        .iter()
        .map(|group| grade_group(group, &cases, &truth, &admissions))
        .collect::<Result<_>>()?;
    let passes: Vec<_> = grades
        .iter()
        .filter(|row| row.registered_group_pass)
        .collect();
    let passed_articles: BTreeSet<_> = passes.iter().map(|row| row.article).collect();
    let output = serde_json::json!({
        "schema": SCHEMA,
        "source_sha256": {"blind": blind_sha, "truth": truth_sha, "groups": groups_sha,
                          "base": uor_r4_integer::discrete_d11::native_whole_turn::BASE_SHA256,
                          "tokenizer": tokenizer_sha, "legacy": old_sha, "relation": new_sha},
        "artifact_context": roots.context,
        "source_qualified_cases": blind.len(),
        "verified_client_cases": cases.len(),
        "unadmitted_client_cases": blind.len() - cases.len(),
        "verified_legacy_openings": old_rows.len(),
        "verified_relation_openings": new_rows.len(),
        "qualified_group_rows": grades,
        "registered_group_pass_count": passes.len(),
        "registered_article_pass_count": passed_articles.len(),
        "screen_licenses_later_fit_design": passes.len() >= 2 && passed_articles.len() >= 2,
    });
    print_json(&output)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
