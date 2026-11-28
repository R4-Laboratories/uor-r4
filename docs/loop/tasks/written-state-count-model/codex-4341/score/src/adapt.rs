use crate::wire::*;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Blind {
    case_id: String,
    text: String,
}

#[derive(Clone, Deserialize)]
struct Gold {
    case_id: String,
    #[serde(deserialize_with = "Option::deserialize")]
    start: Option<u16>,
    #[serde(deserialize_with = "Option::deserialize")]
    end: Option<u16>,
    swapped_case_id: Option<String>,
}

#[derive(Clone, Deserialize)]
struct Truth {
    case_id: String,
    canonical_prefix: Option<Vec<u32>>,
    user_bounds: Option<[u16; 2]>,
    source_ids: Option<Vec<u32>>,
    expected_ids: Option<Vec<u32>>,
    original_bytes: Option<[usize; 2]>,
    response_lexical_offset: Option<usize>,
    #[serde(alias = "negative_group")]
    negative_category: Option<String>,
    article: Option<String>,
    tape: Option<String>,
    passage_sha256: Option<Sha>,
}

#[derive(Clone, Deserialize)]
struct Retokenized {
    case_id: String,
    user: [u16; 2],
    ids: Vec<u32>,
    byte_bounds: Vec<[usize; 2]>,
}

fn jsonl<T: DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn invalid(reason: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, reason)
}

fn features(blind: &[Blind], tape: Vec<NativeFeature>) -> Result<Vec<Capture>> {
    require(
        tape.len() == blind.len() * 2,
        "ordinary feature rows do not cover exact blind roster",
    )?;
    let mut tape = tape.into_iter();
    let mut captures = Vec::new();
    for row in blind {
        let geometry = tape
            .next()
            .ok_or_else(|| invalid("missing geometry feature row"))?;
        let count = tape
            .next()
            .ok_or_else(|| invalid("missing count feature row"))?;
        require(
            geometry.case_id == row.case_id
                && count.case_id == row.case_id
                && geometry.arm == Key::Geometry
                && count.arm == Key::Count,
            format!(
                "{}: ordinary feature case or arm order changed",
                row.case_id
            ),
        )?;
        captures.push(Capture {
            request_id: row.case_id.clone(),
            opening_sha256: sha(row.text.as_bytes()),
            features: vec![geometry, count],
        });
    }
    Ok(captures)
}

pub fn opening(
    mode: &str,
    blind_path: &str,
    gold_path: &str,
    truth_path: &str,
    tokenized_path: &str,
    features_path: &str,
) -> Result<Opening> {
    require(
        matches!(mode, "validation" | "legacy"),
        "opening mode must be validation or legacy",
    )?;
    let tables = Tapes::read([
        blind_path,
        gold_path,
        truth_path,
        tokenized_path,
        features_path,
    ])?;
    let order: Vec<_> = tables.ids.iter().map(String::as_str).collect();
    tables.opening(&order, mode == "validation")
}

struct Tapes {
    ids: Vec<String>,
    rows: BTreeMap<String, TapeRow>,
}

struct TapeRow {
    blind: Blind,
    gold: Gold,
    truth: Truth,
    tokenized: Retokenized,
    capture: Capture,
}

impl Tapes {
    fn read([blind, gold, truth, tokenized, features_path]: [&str; 5]) -> Result<Self> {
        let blind: Vec<Blind> = jsonl(blind)?;
        let gold: Vec<Gold> = jsonl(gold)?;
        let truth: Vec<Truth> = jsonl(truth)?;
        let tokenized: Vec<Retokenized> = jsonl(tokenized)?;
        let captures = features(&blind, jsonl(features_path)?)?;
        require(
            [gold.len(), truth.len(), tokenized.len()]
                .into_iter()
                .all(|n| n == blind.len()),
            "input/gold/truth/tokenized rosters differ",
        )?;
        let mut ids = Vec::new();
        let mut rows = BTreeMap::new();
        for ((((blind, gold), truth), tokenized), capture) in blind
            .into_iter()
            .zip(gold)
            .zip(truth)
            .zip(tokenized)
            .zip(captures)
        {
            let id = &blind.case_id;
            require(
                *id == gold.case_id && *id == truth.case_id && *id == tokenized.case_id,
                "input/gold/truth/tokenized case order differs",
            )?;
            ids.push(id.clone());
            require(
                rows.insert(
                    id.clone(),
                    TapeRow {
                        blind,
                        gold,
                        truth,
                        tokenized,
                        capture,
                    },
                )
                .is_none(),
                "duplicate input/gold/truth case",
            )?;
        }
        Ok(Self { ids, rows })
    }

    fn opening(&self, ids: &[&str], validation: bool) -> Result<Opening> {
        require(
            (1..=2).contains(&ids.len()) && (!validation || ids.len() == 2),
            "opening adapter requires one complete registered pair or legacy singleton",
        )?;
        let rows: Vec<_> = ids
            .iter()
            .map(|id| {
                self.rows
                    .get(*id)
                    .ok_or_else(|| invalid("qualified opening missing from sealed input tape"))
            })
            .collect::<std::io::Result<_>>()?;
        if validation {
            require(
                rows[0].gold.swapped_case_id.as_deref() == Some(ids[1])
                    && rows[1].gold.swapped_case_id.as_deref() == Some(ids[0]),
                "frozen gold lacks exact reciprocal membership",
            )?;
        }
        make_opening(&rows, validation)
    }
}

fn make_opening(rows: &[&TapeRow], validation: bool) -> Result<Opening> {
    let first = &rows[0].capture.features[0];
    let fact = first.fact;
    let intervals = |row: &NativeFeature| -> Vec<Span> {
        row.candidates
            .iter()
            .map(|span| Span {
                start: span.start,
                end: span.end,
            })
            .collect()
    };
    let candidates = intervals(first);
    let slice = |row: &Retokenized| -> Result<Vec<u32>> {
        row.ids
            .get(usize::from(fact[0])..usize::from(fact[1]))
            .map(<[u32]>::to_vec)
            .ok_or_else(|| invalid("fact does not fit retokenized opening").into())
    };
    let fact_ids = slice(&rows[0].tokenized)?;
    let mut cases = Vec::new();
    for tape in rows {
        let (row, g, t, ids) = (&tape.capture, &tape.gold, &tape.truth, &tape.tokenized);
        let id = &row.request_id;
        require(
            g.case_id == *id
                && t.case_id == *id
                && ids.case_id == *id
                && t.canonical_prefix
                    .as_ref()
                    .is_none_or(|prefix| prefix == &ids.ids)
                && t.user_bounds.is_none_or(|bounds| bounds == ids.user)
                && row.features.iter().all(|feature| {
                    feature.user == ids.user
                        && feature.fact == fact
                        && feature.question == row.features[0].question
                        && intervals(feature) == candidates
                })
                && slice(ids)? == fact_ids,
            format!("{id}: canonical opening, fact or candidate identity changed"),
        )?;
        let target = match (g.start, g.end) {
            (Some(start), Some(end)) => {
                let span = Span { start, end };
                if validation {
                    let byte_bounds = span
                        .start
                        .checked_sub(ids.user[0])
                        .zip(
                            span.end
                                .checked_sub(ids.user[0])
                                .and_then(|end| end.checked_sub(1)),
                        )
                        .and_then(|(first, last)| {
                            Some([
                                ids.byte_bounds.get(usize::from(first))?[0].checked_sub(1)?,
                                ids.byte_bounds.get(usize::from(last))?[1].checked_sub(1)?,
                            ])
                        });
                    require(
                        byte_bounds.is_some() && byte_bounds == t.original_bytes,
                        format!(
                            "{id}: registered answer bytes changed under ordinary tokenization"
                        ),
                    )?;
                }
                let source_ids = t
                    .source_ids
                    .clone()
                    .ok_or_else(|| invalid("missing independent source token IDs"))?;
                let expected_ids = t
                    .expected_ids
                    .clone()
                    .ok_or_else(|| invalid("missing expected reply IDs"))?;
                let response_lexical_offset = match t.response_lexical_offset {
                    Some(offset) => offset,
                    None if !validation => usize::from(!expected_ids.starts_with(&source_ids)),
                    None => return Err(invalid("missing canonical lexical offset").into()),
                };
                Target::Source {
                    span,
                    source_ids,
                    expected_ids,
                    response_lexical_offset,
                }
            }
            (None, None) => {
                require(
                    (t.source_ids.is_none()
                        || (!validation && t.source_ids.as_ref().is_some_and(Vec::is_empty)))
                        && t.expected_ids.is_none()
                        && t.response_lexical_offset.is_none(),
                    "NoRead gold carries an unexpected positive target",
                )?;
                let group = match t.negative_category.as_deref() {
                    Some("unanswerable") | Some("unanswered") | Some("genuine_unanswered") => {
                        Negative::Unanswered
                    }
                    Some("false_premise")
                    | Some("contradicted_premise")
                    | Some("contradicted_false_premise") => Negative::ContradictedPremise,
                    None if !validation => Negative::Unanswered,
                    _ => {
                        return Err(
                            invalid("unknown or absent registered negative category").into()
                        );
                    }
                };
                Target::NoRead { group }
            }
            _ => return Err(invalid("only one endpoint present in gold").into()),
        };
        cases.push(Case {
            id: id.clone(),
            opening_sha256: row.opening_sha256.clone(),
            prompt_ids: ids.ids.clone(),
            user: ids.user,
            question: row.features[0].question,
            target,
        });
    }
    let opening = Opening {
        fact,
        fact_ids,
        candidates,
        cases,
    };
    crate::verify::valid_opening(&opening, validation)?;
    Ok(opening)
}

// #4337's device replay format is, for each case, one winning index, one
// NoRead score and one signed i32 score for every interval in enumeration order.
pub fn panel(key: &str, blind: &str, cold: &str, gpu_scores: &str) -> Result<Panel> {
    let index = match key {
        "geometry" => 0,
        "count" => 1,
        _ => return Err(invalid("unknown native key arm").into()),
    };
    let capture = features(&jsonl::<Blind>(blind)?, jsonl(cold)?)?;
    let bytes = std::fs::read(Path::new(gpu_scores))?;
    require(bytes.len().is_multiple_of(4), "truncated CUDA score words")?;
    let mut words = bytes
        .chunks_exact(4)
        .map(|raw| i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
    let mut reference = Vec::new();
    for row in &capture {
        let feature = &row.features[index];
        let winner = words
            .next()
            .ok_or_else(|| invalid("missing CUDA winning index"))?;
        let no_read_score = words
            .next()
            .ok_or_else(|| invalid("missing CUDA NoRead score"))?;
        let candidates: Vec<_> = words.by_ref().take(feature.candidates.len()).collect();
        require(
            candidates.len() == feature.candidates.len(),
            "truncated CUDA candidate scores",
        )?;
        let span = if winner == 0 {
            None
        } else {
            let interval = feature
                .candidates
                .get(usize::try_from(winner)? - 1)
                .ok_or_else(|| invalid("CUDA winning index out of range"))?;
            Some(Span {
                start: interval.start,
                end: interval.end,
            })
        };
        let selection = Selection {
            span,
            no_read_score,
            best_candidate_score: candidates.iter().copied().max(),
        };
        let scores = Scores {
            candidates,
            selection,
        };
        require(
            feature.scores.as_ref() == Some(&scores),
            format!(
                "{}: CUDA scores or winner differ from cold client",
                row.request_id
            ),
        )?;
        reference.push(ReferenceRow {
            case_id: row.request_id.clone(),
            scores,
        });
    }
    require(words.next().is_none(), "trailing CUDA score words")?;
    Ok(Panel { capture, reference })
}

#[derive(Clone, Deserialize, Eq, PartialEq)]
struct Identity {
    family: Family,
    article: String,
    passage_sha256: Sha,
    left: String,
    right: String,
    rank: usize,
    pair_id: String,
}

#[derive(Deserialize)]
struct SelectionPair {
    article: String,
    passage_sha256: Sha,
    left: String,
    right: String,
}

#[derive(Deserialize)]
struct SelectionCandidate {
    pair: SelectionPair,
    rank: usize,
    disposition: String,
}

#[derive(Deserialize)]
struct SelectionTitle {
    title: String,
    passages: usize,
    pairs: usize,
    candidates: Vec<SelectionCandidate>,
}

#[derive(Deserialize)]
struct SelectionManifest {
    schema: String,
    two_answer: Vec<SelectionTitle>,
    answerability: Vec<SelectionTitle>,
}

#[derive(Deserialize)]
struct QualifiedPair {
    pair_id: String,
    family: Family,
    article: String,
    passage_sha256: Sha,
    left_case_id: String,
    right_case_id: String,
    accepted: bool,
    reasons: Vec<String>,
    native_reasons: Vec<String>,
    qualified: bool,
}

pub fn validation(
    study: Study,
    selection: &str,
    identities: &str,
    directory: &str,
) -> Result<Vec<Title>> {
    let selection: SelectionManifest = serde_json::from_reader(File::open(selection)?)?;
    require(
        selection.schema == "natural-reciprocal-validation-selection/1",
        "unregistered selector schema",
    )?;
    let identities: Vec<Identity> = jsonl(identities)?;
    let path = |name: &str| format!("{directory}/{name}.jsonl");
    let pair_dispositions: Vec<QualifiedPair> = jsonl(&path("pairs"))?;
    require(
        identities.len() == pair_dispositions.len()
            && identities.len()
                == if study == Study::Expanded4341 {
                    113
                } else {
                    67
                },
        "incomplete registered identity/disposition count",
    )?;
    require(
        pair_dispositions
            .iter()
            .skip(67)
            .map(|pair| pair.pair_id.as_str())
            .eq(identities
                .iter()
                .skip(67)
                .map(|identity| identity.pair_id.as_str())),
        "additional qualification order differs from registered complement",
    )?;
    let mut by_id = BTreeMap::new();
    for pair in pair_dispositions {
        require(
            by_id.insert(pair.pair_id.clone(), pair).is_none(),
            "repeated qualification pair",
        )?;
    }
    for identity in &identities {
        let pair = by_id
            .get(&identity.pair_id)
            .ok_or_else(|| invalid("missing registered qualification pair"))?;
        require(
            identity.family == pair.family
                && identity.article == pair.article
                && identity.passage_sha256 == pair.passage_sha256
                && identity.left == pair.left_case_id
                && identity.right == pair.right_case_id,
            "qualification record differs from registered original identity",
        )?;
    }
    let paths = ["input", "gold", "truth", "openings", "features"].map(path);
    let accepted_count = by_id.values().filter(|row| row.accepted).count();
    let tapes = if accepted_count > 0 {
        let tapes = Tapes::read(paths.each_ref().map(String::as_str))?;
        require(
            tapes.ids.len() == 2 * accepted_count,
            "wrong accepted tape count",
        )?;
        Some(tapes)
    } else {
        None
    };
    let mut outputs = Vec::new();
    let mut original = Vec::new();
    let mut additional = Vec::new();
    let mut used = BTreeSet::new();
    for (family, titles) in [
        (Family::TwoAnswer, selection.two_answer),
        (Family::Answerability, selection.answerability),
    ] {
        let family_str = if family == Family::TwoAnswer {
            "two_answer"
        } else {
            "answerability"
        };
        for title in titles {
            let complement = study.additional(family, &title.title) > 0;
            require(
                title.candidates.len() == title.pairs,
                "mechanical candidate census changed",
            )?;
            let mut pairs = Vec::new();
            for row in title.candidates {
                let is_original = row.disposition == "review";
                if !is_original && !(complement && row.disposition == "outside_article_quota") {
                    continue;
                }
                let identity = Identity {
                    pair_id: format!("{family_str}-{}", row.pair.passage_sha256),
                    family,
                    rank: row.rank,
                    article: row.pair.article,
                    passage_sha256: row.pair.passage_sha256,
                    left: row.pair.left,
                    right: row.pair.right,
                };
                require(identity.article == title.title, "selected article changed")?;
                if is_original {
                    original.push(identity.clone());
                } else {
                    additional.push(identity.clone());
                }
                let disposition = by_id
                    .get(identity.pair_id.as_str())
                    .ok_or_else(|| invalid("missing registered disposition"))?;
                let qualification = match (disposition.accepted, disposition.qualified) {
                    (true, true) => {
                        require(
                            disposition.reasons.is_empty() && disposition.native_reasons.is_empty(),
                            "qualified pair has a contradictory exclusion",
                        )?;
                        let tapes = tapes
                            .as_ref()
                            .ok_or_else(|| invalid("missing accepted tapes"))?;
                        Qualification::Qualified {
                            opening: tapes.opening(&[&identity.left, &identity.right], true)?,
                        }
                    }
                    (true, false) => {
                        require(
                            disposition.reasons.is_empty()
                                && !disposition.native_reasons.is_empty(),
                            "native exclusion has no reason",
                        )?;
                        let reason = disposition.native_reasons.join(", ");
                        if disposition.native_reasons.iter().any(|reason| {
                            reason.contains("rebind") || reason.contains("source_ids")
                        }) {
                            Qualification::RebindExcluded { reason }
                        } else {
                            Qualification::CaptureExcluded { reason }
                        }
                    }
                    (false, false) => {
                        require(
                            !disposition.reasons.is_empty()
                                && disposition.native_reasons.is_empty(),
                            "source exclusion has missing reason or unexpected capture",
                        )?;
                        Qualification::SourceExcluded {
                            reason: disposition.reasons.join(", "),
                        }
                    }
                    (false, true) => {
                        return Err(invalid("source-excluded pair cannot qualify").into());
                    }
                };
                if disposition.accepted {
                    let tapes = tapes
                        .as_ref()
                        .ok_or_else(|| invalid("missing accepted tapes"))?;
                    for (id, mate) in [
                        (&identity.left, &identity.right),
                        (&identity.right, &identity.left),
                    ] {
                        require(
                            used.insert(id.clone())
                                && tapes.rows.get(id).is_some_and(|row| {
                                    row.gold.swapped_case_id.as_ref() == Some(mate)
                                        && row.truth.passage_sha256.as_ref()
                                            == Some(&identity.passage_sha256)
                                        && row.truth.article.as_ref() == Some(&identity.article)
                                }),
                            "accepted tape belongs to a different original source or question",
                        )?;
                    }
                }
                pairs.push(Pair {
                    passage_sha256: identity.passage_sha256,
                    left_id: identity.left,
                    right_id: identity.right,
                    qualification,
                });
            }
            outputs.push(Title {
                name: title.title,
                family,
                mechanical_candidates: title.pairs,
                passages: title.passages,
                selected: pairs,
            });
        }
    }
    let by_identity = |rows: &[Identity]| {
        rows.iter()
            .map(|row| (row.pair_id.clone(), row.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    require(
        original.len() == 67
            && by_identity(&original) == by_identity(&identities[..67])
            && additional == identities[67..],
        "review records differ from #4340 roster or ordered #4341 complement",
    )?;
    Ok(outputs)
}

pub fn controls(mode: &str, paths: [&str; 5]) -> Result<Vec<ControlSource>> {
    require(
        matches!(mode, "original" | "natural" | "protected"),
        "unregistered control input",
    )?;
    let tapes = Tapes::read(paths)?;
    let mut controls = Vec::new();
    let mut used = BTreeSet::new();
    for id in &tapes.ids {
        if !used.insert(id.as_str()) {
            continue;
        }
        let row = &tapes.rows[id];
        let mut pair = vec![id.as_str()];
        if mode != "protected"
            && let Some(mate) = row.gold.swapped_case_id.as_deref()
        {
            require(
                used.insert(mate)
                    && tapes
                        .rows
                        .get(mate)
                        .is_some_and(|peer| peer.gold.swapped_case_id.as_deref() == Some(id)),
                "TRAIN gold does not contain reciprocal mate",
            )?;
            pair.push(mate);
        }
        pair.sort_by_key(|id| (tapes.rows[*id].gold.start.is_none(), *id));
        let mut openings = Vec::new();
        for id in pair {
            openings.push(tapes.opening(&[id], false)?);
        }
        let passage_sha256 = row.truth.passage_sha256.clone().map_or_else(
            || {
                let [fact_start, fact_end] = openings[0].fact;
                let [user_start, _] = openings[0].cases[0].user;
                let first = usize::from(fact_start - user_start);
                let last = usize::from(fact_end - user_start - 1);
                let bounds = &row.tokenized.byte_bounds;
                let exact = bounds.get(first).zip(bounds.get(last)).and_then(|(a, b)| {
                    row.blind
                        .text
                        .as_bytes()
                        .get(a[0].checked_sub(1)?..b[1].checked_sub(1)?)
                });
                exact
                    .map(sha)
                    .ok_or_else(|| invalid("cannot recover exact original authored fact bytes"))
            },
            Ok,
        )?;
        let title = row
            .truth
            .article
            .clone()
            .or_else(|| row.truth.tape.clone())
            .ok_or_else(|| invalid("missing frozen TRAIN or protected provenance"))?;
        controls.push(ControlSource {
            title,
            passage_sha256,
            openings,
        });
    }
    Ok(controls)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SwapRequest {
    request_id: String,
    case_id: String,
    text: String,
}

pub fn swaps(request_path: &str, cold: &str) -> Result<Vec<Capture>> {
    let requests: Vec<SwapRequest> = jsonl(request_path)?;
    let blind: Vec<_> = requests
        .iter()
        .map(|row| Blind {
            case_id: row.case_id.clone(),
            text: row.text.clone(),
        })
        .collect();
    let mut captures = features(&blind, jsonl(cold)?)?;
    for (capture, request) in captures.iter_mut().zip(requests) {
        capture.request_id = request.request_id;
    }
    Ok(captures)
}
