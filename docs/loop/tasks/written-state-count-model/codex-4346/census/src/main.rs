//! Data-only replay of the frozen #4342 TRAIN source selection.
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{self, Write},
    path::Path,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::native_interval::{
    Arm, BANKS, MAX_SPAN, SOURCE_CENSUS_CONTEXT, SourceCandidates, Span, census_source_candidates,
};
use uor_r4_tokenizer::{
    ByteBpeTokenizer,
    dialogue::{DialogueProtocol, EncodedUserOpening},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const BASE_CONTEXT: usize = 192;
const TOKENIZER_SHA256: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const LEGACY_ARMS: [Arm; 2] = [Arm::Geometry, Arm::Count];
const ARMS: [Arm; 5] = [
    Arm::Geometry,
    Arm::Count,
    Arm::DirectedGeometry,
    Arm::DirectedCount,
    Arm::SwappedGeometry,
];

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        digest(&bytes) == sha,
        &format!("SHA-256 differs at {}", path.display()),
    )?;
    Ok(bytes)
}

fn jsonl<T: DeserializeOwned>(bytes: &[u8]) -> Result<Vec<T>> {
    std::str::from_utf8(bytes)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

#[derive(Debug, Deserialize, PartialEq)]
struct Blind {
    case_id: String,
    text: String,
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
    original_bytes: [usize; 2],
    original_char: [usize; 2],
    span_text: String,
}

impl Truth {
    fn verify_source(&self, prompt: &Blind) -> Result<()> {
        let passage = prompt
            .text
            .get(..self.passage_bytes)
            .ok_or("invalid passage edge")?;
        let question = prompt
            .text
            .get(self.passage_bytes..)
            .and_then(|tail| tail.strip_prefix(' '));
        require(
            self.case_id == prompt.case_id
                && question == Some(&self.question)
                && digest(passage.as_bytes()) == self.passage_sha256,
            "prompt is not the original passage and question",
        )?;
        let [left, right] = self.original_bytes;
        require(
            left < right && passage.get(left..right) == Some(&self.span_text),
            "original answer byte interval differs",
        )?;
        require(
            self.original_char
                == [
                    passage[..left].chars().count(),
                    passage[..right].chars().count(),
                ],
            "original answer character interval differs",
        )
    }
}

#[derive(Deserialize)]
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

#[derive(Debug, Deserialize)]
struct Admission {
    case_id: String,
    tokens: usize,
    refusal: Option<String>,
}

#[derive(Deserialize)]
struct AdmissionArchive {
    artifact_context: usize,
    cases: Vec<Admission>,
    admitted: Vec<Blind>,
}

#[derive(Deserialize)]
struct Tape {
    case_id: String,
    arm: Arm,
    #[serde(flatten)]
    source: SourceCandidates,
    no_read_keys: [u16; BANKS],
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case", tag = "status", content = "span")]
enum Address {
    Available(Span),
    NormalizedSource,
    MissingBothTokenEdges,
    MissingLeftTokenEdge,
    MissingRightTokenEdge,
    OutsideFactRole,
    OverEightTokens(Span),
    ExcludedByCopyOrLexicalEdge,
}

impl Address {
    fn available(self) -> bool {
        matches!(self, Self::Available(_))
    }
}

/// Map original byte boundaries to protocol-normalized user bytes. The LF in
/// a CRLF pair has no independent original boundary after normalization.
struct ByteMap {
    offsets: Vec<Option<usize>>,
    decoded: Vec<u8>,
    token_edges: Vec<usize>,
}

impl ByteMap {
    fn new(
        original: &str,
        opening: &EncodedUserOpening,
        tokenizer: &ByteBpeTokenizer,
    ) -> Result<Self> {
        let mut offsets = vec![None; original.len() + 1];
        let mut normalized = Vec::new();
        let mut input = original.bytes().enumerate().peekable();
        while let Some((index, byte)) = input.next() {
            offsets[index] = Some(normalized.len());
            if byte == b'\r' {
                if input.peek().is_some_and(|(_, next)| *next == b'\n') {
                    input.next();
                }
                normalized.push(b'\n');
            } else {
                normalized.push(byte);
            }
        }
        offsets[original.len()] = Some(normalized.len());
        let normalized = std::str::from_utf8(&normalized)?;
        let left = normalized.len() - normalized.trim_start().len();
        let right = normalized.trim_end().len();
        for offset in &mut offsets {
            *offset = offset
                .filter(|n| (left..=right).contains(n))
                .map(|n| n - left + 1);
        }
        let mut decoded = Vec::new();
        let mut token_edges = vec![0];
        for &id in &opening.tokens()[opening.user_range()] {
            let bytes = tokenizer.decode_bytes(&[id]);
            require(!bytes.is_empty(), "user token decodes to zero bytes")?;
            decoded.extend(bytes);
            token_edges.push(decoded.len());
        }
        require(
            decoded == format!(" {}", normalized.trim()).as_bytes(),
            "tokens do not reconstruct the exact protocol-normalized user bytes",
        )?;
        Ok(Self {
            offsets,
            decoded,
            token_edges,
        })
    }

    fn grade(
        &self,
        original: &str,
        [start, end]: [usize; 2],
        sources: &SourceCandidates,
    ) -> Address {
        let (Some(start_byte), Some(end_byte)) = (self.offsets[start], self.offsets[end]) else {
            return Address::NormalizedSource;
        };
        if self.decoded.get(start_byte..end_byte) != original.as_bytes().get(start..end) {
            return Address::NormalizedSource;
        }
        // Allow the adjacent ASCII space joined by BPE, including protocol 2's
        // first content space. Never strip an arbitrary neighboring token.
        let with_space = start_byte
            .checked_sub(1)
            .filter(|&i| self.decoded[i] == b' ');
        let user_start = usize::from(sources.user[0]);
        let fits_start = |byte| byte == start_byte || Some(byte) == with_space;
        for &span in &sources.candidates {
            let first = self.token_edges[usize::from(span.start) - user_start];
            let last = self.token_edges[usize::from(span.end) - user_start];
            if last == end_byte && fits_start(first) {
                return Address::Available(span);
            }
        }
        let left = self.token_edges.iter().rposition(|&byte| fits_start(byte));
        let right = self.token_edges.binary_search(&end_byte).ok();
        match (left, right) {
            (None, None) => Address::MissingBothTokenEdges,
            (None, Some(_)) => Address::MissingLeftTokenEdge,
            (Some(_), None) => Address::MissingRightTokenEdge,
            (Some(first), Some(last)) => {
                if first + user_start < usize::from(sources.fact[0])
                    || last + user_start > usize::from(sources.fact[1])
                {
                    Address::OutsideFactRole
                } else if last.saturating_sub(first) > MAX_SPAN {
                    Address::OverEightTokens(Span {
                        start: (first + user_start) as u16,
                        end: (last + user_start) as u16,
                    })
                } else {
                    Address::ExcludedByCopyOrLexicalEdge
                }
            }
        }
    }
}

#[derive(Serialize)]
struct Row<'a> {
    case_id: &'a str,
    tokens: usize,
    admitted_192: bool,
    admitted_512: bool,
    source_candidates: usize,
    no_read_choices: usize,
    own: Option<Address>,
    own_original_bytes: [usize; 2],
    named_opponent: Option<Address>,
    opponent_original_bytes: Option<[usize; 2]>,
}

#[derive(Serialize)]
struct GroupResult<'a> {
    group_id: &'a str,
    stratum: &'a str,
    contrast: &'a str,
    admitted_192: bool,
    own_192: bool,
    both_targets_192: Option<bool>,
    admitted_512: bool,
    own_512: bool,
    both_targets_512: Option<bool>,
}

fn verify_short(blind: &Blind, sources: &SourceCandidates, tapes: &[Tape]) -> Result<()> {
    require(
        tapes.len() == LEGACY_ARMS.len() + ARMS.len(),
        "ordinary arm set differs",
    )?;
    for (tape, arm) in tapes.iter().zip(LEGACY_ARMS.into_iter().chain(ARMS)) {
        require(
            tape.case_id == blind.case_id && tape.arm == arm && tape.source == *sources,
            "archived ordinary source ranges or candidate order differ",
        )?;
        require(
            tape.no_read_keys.iter().enumerate().all(|(bank, &key)| {
                (bank * 256 + 224..(bank + 1) * 256).contains(&usize::from(key))
            }),
            "archived no-read keys leave their reserved slots",
        )?;
    }
    Ok(())
}

fn verify_group<'a>(
    group: &Group,
    truth: &BTreeMap<&str, &'a Truth>,
    seen: &mut BTreeSet<String>,
) -> Result<Vec<&'a Truth>> {
    let lengths: Vec<_> = group.families.iter().map(Vec::len).collect();
    let registered = match group.stratum.as_str() {
        "quad" => lengths == [2, 2] && group.contrast == "target-switch",
        "triple" => lengths == [2, 1] && group.contrast == "target-switch",
        "switch" => {
            lengths == [1, 1]
                && matches!(
                    group.contrast.as_str(),
                    "target-switch" | "same-fact-inverse"
                )
        }
        "paraphrase" => lengths == [2] && group.contrast == "paraphrase",
        _ => false,
    };
    require(registered, "unregistered group structure")?;
    let mut targets = Vec::new();
    for family in &group.families {
        let first = truth
            .get(family[0].as_str())
            .ok_or("missing family representative")?;
        for id in family {
            let row = truth.get(id.as_str()).ok_or("missing group truth")?;
            require(
                seen.insert(id.clone())
                    && row.passage_sha256 == group.passage_sha256
                    && row.paragraph == group.paragraph
                    && row.split == group.split
                    && row.article == group.article
                    && !row.family.is_empty()
                    && row.original_bytes == first.original_bytes,
                "group changes original identity",
            )?;
        }
        targets.push(*first);
    }
    if let [a, b] = targets.as_slice() {
        require(
            a.original_bytes[1] <= b.original_bytes[0]
                || b.original_bytes[1] <= a.original_bytes[0],
            "qualified group targets overlap",
        )?;
    }
    Ok(targets)
}

fn same_suffix(tokenizer: &ByteBpeTokenizer) -> Result<serde_json::Value> {
    let encoder = DialogueProtocol::literal_roles_v2(tokenizer)?.bind(tokenizer)?;
    let filler = "The river flows east. ".repeat(54);
    let mut witnesses = Vec::new();
    let mut openings = Vec::new();
    for name in ["A", "B"] {
        let text = format!("{name} is the marker. {filler}What is the marker?");
        let opening = encoder
            .encode_single_user_opening(&text)
            .ok_or("fixture refused protocol 2")?;
        require(
            opening.tokens().len() > BASE_CONTEXT,
            "fixture fits the short opening",
        )?;
        let fields = census_source_candidates(&opening, tokenizer)?;
        let Address::Available(span) =
            ByteMap::new(&text, &opening, tokenizer)?.grade(&text, [0, name.len()], &fields)
        else {
            return Err("fixture source is unreachable".into());
        };
        let ids = &opening.tokens()[usize::from(span.start)..usize::from(span.end)];
        require(
            usize::from(span.end) <= opening.tokens().len() - BASE_CONTEXT,
            "fixture source lies in the shared suffix",
        )?;
        witnesses.push((span, ids.to_vec()));
        openings.push(opening);
    }
    let [a, b] = openings.as_slice() else {
        return Err("fixture must have two openings".into());
    };
    let [left, right] = witnesses.as_slice() else {
        return Err("fixture must have two spans".into());
    };
    require(
        a.tokens().len() == b.tokens().len()
            && a.tokens()[a.tokens().len() - BASE_CONTEXT..]
                == b.tokens()[b.tokens().len() - BASE_CONTEXT..]
            && left.0 == right.0
            && left.1 != right.1,
        "fixture does not preserve a shared suffix and different earlier sources",
    )?;
    Ok(serde_json::json!({
        "tokens": a.tokens().len(),
        "identical_final_ids": BASE_CONTEXT,
        "original_span": left.0,
        "source_ids": [ &left.1, &right.1 ],
    }))
}

fn cohort<'a>(rows: impl Iterator<Item = &'a Row<'a>>) -> serde_json::Value {
    let mut total = 0;
    let mut openings_with_rival = 0;
    let mut source_candidates = 0;
    let mut no_read_choices = 0;
    let mut maximum_candidates = 0;
    let mut minimum_candidates = None;
    let mut own_available = 0;
    let mut own_unavailable = BTreeMap::new();
    let mut over_eight_widths = BTreeMap::new();
    let mut with_named_opponent = 0;
    let mut opponent_available = 0;
    let mut own_and_opponent = 0;
    let mut gold_with_distinct_rival = 0;
    for row in rows {
        total += 1;
        let count = row.source_candidates;
        source_candidates += count;
        no_read_choices += row.no_read_choices;
        maximum_candidates = maximum_candidates.max(count);
        minimum_candidates =
            Some(minimum_candidates.map_or(count, |prior: usize| prior.min(count)));
        let own = row.own.is_some_and(Address::available);
        openings_with_rival += usize::from(count > usize::from(own));
        own_available += usize::from(own);
        gold_with_distinct_rival += usize::from(own && count > 1);
        if let Some(address) = row.own.filter(|address| !address.available()) {
            let status = serde_json::json!(address)["status"]
                .as_str()
                .unwrap_or("unknown")
                .to_owned();
            *own_unavailable.entry(status).or_insert(0) += 1;
        }
        if let Some(Address::OverEightTokens(span)) = row.own {
            *over_eight_widths.entry(span.end - span.start).or_insert(0) += 1;
        }
        with_named_opponent += usize::from(row.opponent_original_bytes.is_some());
        let opponent = row.named_opponent.is_some_and(Address::available);
        opponent_available += usize::from(opponent);
        own_and_opponent += usize::from(own && opponent);
    }
    serde_json::json!({
        "openings": total,
        "source_candidates": source_candidates,
        "no_read_choices": no_read_choices,
        "max_candidate_count": maximum_candidates,
        "min_candidate_count": minimum_candidates,
        "openings_with_distinct_rival": openings_with_rival,
        "gold_with_distinct_rival": gold_with_distinct_rival,
        "own_available": own_available,
        "own_unavailable": own_unavailable,
        "over_eight_token_widths": over_eight_widths,
        "with_named_opponent": with_named_opponent,
        "opponent_available": opponent_available,
        "own_and_opponent_available": own_and_opponent,
    })
}

fn run(root: &Path, tokenizer_path: &Path) -> Result<serde_json::Value> {
    let load = |name: &str, sha| pinned(&root.join(name), sha);
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&pinned(tokenizer_path, TOKENIZER_SHA256)?)
            .ok_or("invalid pinned tokenizer")?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let fixture = same_suffix(&tokenizer)?;
    let blind: Vec<Blind> = jsonl(&load(
        "supply/blind.jsonl",
        "54219a05ae0a1402ad09f92e5133e376adb73df577e5f7075c7db576f5d3a9ca",
    )?)?;
    let targets: Vec<Truth> = jsonl(&load(
        "supply/truth.jsonl",
        "79a201c74dafc024dda5700b98d260e70df48b4e202700884e627592505b8d77",
    )?)?;
    let truth: BTreeMap<_, _> = targets
        .iter()
        .map(|row| (row.case_id.as_str(), row))
        .collect();
    let prompts: BTreeMap<_, _> = blind
        .iter()
        .map(|row| (row.case_id.as_str(), row))
        .collect();
    let groups: Vec<Group> = jsonl(&load(
        "supply/groups.jsonl",
        "4576e3de5fd29611c82ea3bb2f2d6ec4d58918629ce9f4117b57702aa4e9806d",
    )?)?;
    let admission: AdmissionArchive = serde_json::from_slice(&load(
        "probe/admission.json",
        "0e7ca04a140e65e22888d8ee5d845eaa57b310a06113366abe9b0f9ead3f9d5f",
    )?)?;
    require(
        blind.len() == 73
            && prompts.len() == 73
            && targets.len() == 73
            && truth.len() == 73
            && groups.len() == 29
            && admission.cases.len() == 73
            && admission.admitted.len() == 20
            && admission.artifact_context == BASE_CONTEXT
            && targets
                .iter()
                .map(|row| &row.question_source_id)
                .collect::<BTreeSet<_>>()
                .len()
                == 73,
        "frozen roster size or ordinary context differs",
    )?;
    let mut tapes = BTreeMap::<String, Vec<Tape>>::new();
    let legacy: Vec<Tape> = jsonl(&load(
        "probe/old.jsonl",
        "815b788890b03c13d8af75f362f1d44fe692884aba2f864df5f989a28ae26ca6",
    )?)?;
    let directed: Vec<Tape> = jsonl(&load(
        "probe/new.jsonl",
        "1e09407abe753df6bf7af898ce0f45c9fe309efa1a758e473c109dc82de51733",
    )?)?;
    for tape in legacy.into_iter().chain(directed) {
        tapes.entry(tape.case_id.clone()).or_default().push(tape);
    }
    let mut openings = BTreeMap::new();
    let mut short = Vec::new();
    for (prompt, archived) in blind.iter().zip(&admission.cases) {
        let opening = encoder
            .encode_single_user_opening(&prompt.text)
            .ok_or("frozen protocol opening refused")?;
        let length = opening.tokens().len();
        let refusal = (length > BASE_CONTEXT).then_some("prefix-exceeds-context");
        require(
            archived.case_id == prompt.case_id
                && archived.tokens == length
                && archived.refusal.as_deref() == refusal,
            "archived ordinary admission differs",
        )?;
        if refusal.is_none() {
            short.push(prompt);
        }
        openings.insert(prompt.case_id.as_str(), opening);
    }
    require(
        short.into_iter().eq(&admission.admitted),
        "ordinary admitted prompts differ",
    )?;
    let mut rows = Vec::new();
    let mut group_results = Vec::new();
    let mut seen = BTreeSet::new();
    let mut reversal = None;
    for group in &groups {
        let families = verify_group(group, &truth, &mut seen)?;
        let start = rows.len();
        for (index, family) in group.families.iter().enumerate() {
            let opponent = (families.len() == 2).then(|| families[1 - index].original_bytes);
            for id in family {
                let case = truth[id.as_str()];
                let prompt = prompts.get(id.as_str()).ok_or("missing original prompt")?;
                case.verify_source(prompt)?;
                let opening = &openings[id.as_str()];
                let tokens = opening.tokens().len();
                let admitted_192 = tokens <= BASE_CONTEXT;
                let admitted_512 = tokens <= SOURCE_CENSUS_CONTEXT;
                let mut row = Row {
                    case_id: id,
                    tokens,
                    admitted_192,
                    admitted_512,
                    source_candidates: 0,
                    no_read_choices: 0,
                    own: None,
                    own_original_bytes: case.original_bytes,
                    named_opponent: None,
                    opponent_original_bytes: opponent,
                };
                if admitted_512 {
                    let fields = census_source_candidates(opening, &tokenizer)?;
                    if admitted_192 {
                        let tape = tapes.remove(id).ok_or("missing ordinary capture")?;
                        verify_short(prompt, &fields, &tape)?;
                    }
                    let offsets = ByteMap::new(&prompt.text, opening, &tokenizer)?;
                    row.source_candidates = fields.candidates.len();
                    row.no_read_choices = 1;
                    row.own = Some(offsets.grade(&prompt.text, case.original_bytes, &fields));
                    row.named_opponent =
                        opponent.map(|span| offsets.grade(&prompt.text, span, &fields));
                }
                rows.push(row);
            }
        }
        let cases = &rows[start..];
        let admitted_192 = cases.iter().all(|row| row.admitted_192);
        let admitted_512 = cases.iter().all(|row| row.admitted_512);
        let own_512 = admitted_512
            && cases
                .iter()
                .all(|row| row.own.is_some_and(Address::available));
        let own_192 = admitted_192 && own_512;
        let opponents = (families.len() == 2).then(|| {
            cases
                .iter()
                .all(|row| row.named_opponent.is_some_and(Address::available))
        });
        group_results.push(GroupResult {
            group_id: &group.group_id,
            stratum: &group.stratum,
            contrast: &group.contrast,
            admitted_192,
            own_192,
            both_targets_192: opponents.map(|available| own_192 && available),
            admitted_512,
            own_512,
            both_targets_512: opponents.map(|available| own_512 && available),
        });
        if group.contrast == "same-fact-inverse" {
            require(reversal.is_none(), "reversal denominator exceeds one")?;
            require(
                cases.iter().map(|row| row.tokens).collect::<BTreeSet<_>>() == [345, 349].into(),
                "reversal opening lengths differ",
            )?;
            reversal = Some(serde_json::json!(cases));
        }
    }
    require(
        seen.len() == 73 && tapes.is_empty(),
        "unassigned original or ordinary rows",
    )?;
    let mut group_totals = BTreeMap::new();
    for group in &group_results {
        let counts = group_totals.entry(group.stratum).or_insert([0; 7]);
        counts[0] += 1;
        counts[1] += usize::from(group.admitted_192);
        counts[2] += usize::from(group.admitted_512);
        counts[3] += usize::from(group.own_192);
        counts[4] += usize::from(group.own_512);
        counts[5] += usize::from(group.both_targets_192 == Some(true));
        counts[6] += usize::from(group.both_targets_512 == Some(true));
    }
    Ok(serde_json::json!({
        "schema": "uor-r4/source-opening-census/1",
        "original_openings": rows.len(),
        "original_groups": groups.len(),
        "opening_caps": [BASE_CONTEXT, SOURCE_CENSUS_CONTEXT],
        "opening_token_range": [
            rows.iter().map(|row| row.tokens).min(),
            rows.iter().map(|row| row.tokens).max(),
        ],
        "ordinary_distinct_arm_tapes_matching_order": admission.admitted.len() * ARMS.len(),
        "ordinary_archived_tapes_matching_order": admission.admitted.len() * (LEGACY_ARMS.len() + ARMS.len()),
        "baseline_192": cohort(rows.iter().filter(|row| row.admitted_192)),
        "recovered_at_512": cohort(rows.iter().filter(|row| !row.admitted_192 && row.admitted_512)),
        "all_admitted_at_512": cohort(rows.iter().filter(|row| row.admitted_512)),
        "refused_at_512": rows.iter().filter(|row| !row.admitted_512).count(),
        "group_counts_columns": ["groups", "all_admitted_192", "all_admitted_512", "all_own_192", "all_own_512", "both_targets_192", "both_targets_512"],
        "group_counts": group_totals,
        "same_fact_inverse": reversal.ok_or("missing authenticated reversal")?,
        "identical_suffix_fixture": fixture,
        "cases": rows,
        "groups": group_results,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [root, tokenizer] = args.as_slice() else {
        return Err("usage: source-opening-census VERIFIED_RESTORE TOKENIZER".into());
    };
    let mut output = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer_pretty(&mut output, &run(Path::new(root), Path::new(tokenizer))?)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}
