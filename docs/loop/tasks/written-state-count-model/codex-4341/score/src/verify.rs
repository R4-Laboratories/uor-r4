use crate::wire::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct Observation<'a> {
    pub opening: &'a Opening,
    pub case: &'a Case,
}

impl Observation<'_> {
    pub fn span(self) -> Option<Span> {
        self.case.target.span()
    }
}

pub type Cases<'a> = BTreeMap<&'a str, Observation<'a>>;
pub type ScoresByCase<'a> = BTreeMap<&'a str, &'a Scores>;

pub struct CheckedRoster<'a> {
    pub validation: Cases<'a>,
    pub mates: BTreeMap<&'a str, &'a str>,
    pub train: BTreeMap<Mixture, Cases<'a>>,
    pub protected: Cases<'a>,
    pub coverage: bool,
}

pub(crate) fn valid_opening(opening: &Opening, canonical_reply: bool) -> Result<()> {
    require(
        opening.fact[0] < opening.fact[1]
            && !opening.candidates.is_empty()
            && opening.candidates.windows(2).all(|pair| pair[0] < pair[1])
            && opening
                .candidates
                .iter()
                .all(|span| span.fits(opening.fact)),
        "invalid or missing ordered original interval candidates",
    )?;
    for case in &opening.cases {
        let fact = Span {
            start: opening.fact[0],
            end: opening.fact[1],
        };
        let [from, to] = case.user;
        let [q_from, q_to] = case.question;
        require(
            !case.id.is_empty()
                && is_sha(&case.opening_sha256)
                && from <= fact.start
                && fact.end <= to
                && from <= q_from
                && q_from < q_to
                && q_to <= to
                && usize::from(to) <= case.prompt_ids.len()
                && fact.tokens(&case.prompt_ids) == Some(&opening.fact_ids),
            format!("{}: inconsistent exact prompt, fact, or question", case.id),
        )?;
        if let Target::Source {
            span,
            source_ids,
            expected_ids,
            response_lexical_offset,
        } = &case.target
        {
            require(
                span.fits(opening.fact)
                    && opening.candidates.binary_search(span).is_ok()
                    && span.tokens(&case.prompt_ids) == Some(source_ids)
                    && (*response_lexical_offset == 0
                        || (*response_lexical_offset == 1 && expected_ids.first() == Some(&223)))
                    && expected_ids.last() == Some(&1)
                    && (!canonical_reply
                        || expected_ids.len() == source_ids.len() + response_lexical_offset + 1)
                    && expected_ids.get(
                        *response_lexical_offset..(*response_lexical_offset + source_ids.len()),
                    ) == Some(source_ids)
                    && !source_ids.iter().any(|id| [0, 1, 16, 201].contains(id)),
                format!("{}: authentic target unavailable or unrebound", case.id),
            )?;
        }
        require(
            opening.candidates.iter().all(|span| {
                span.tokens(&case.prompt_ids)
                    .is_some_and(|ids| !ids.iter().any(|id| [0, 1, 16, 201].contains(id)))
            }),
            format!("{}: invalid source-copy candidate", case.id),
        )?;
    }
    Ok(())
}

fn add_cases<'a>(
    destination: &mut Cases<'a>,
    opening: &'a Opening,
    canonical_reply: bool,
) -> Result<()> {
    valid_opening(opening, canonical_reply)?;
    for case in &opening.cases {
        require(
            destination
                .insert(&case.id, Observation { opening, case })
                .is_none(),
            format!("duplicate case identity: {}", case.id),
        )?;
    }
    Ok(())
}

fn valid_pair(opening: &Opening, family: Family) -> Result<()> {
    let [left, right] = opening.cases.as_slice() else {
        return require(false, "qualified pair must contain exactly two openings");
    };
    let targets = (left.target.span(), right.target.span());
    require(
        left.opening_sha256 != right.opening_sha256,
        "reciprocal openings have identical prompt identity",
    )?;
    match (family, targets) {
        (Family::TwoAnswer, (Some(a), Some(b))) => {
            require(a != b, "two-answer targets use the same address")
        }
        (Family::Answerability, (Some(_), None)) => Ok(()),
        _ => require(false, "pair targets disagree with registered family"),
    }
}

pub fn roster(roster: &Roster) -> Result<CheckedRoster<'_>> {
    require(
        roster.schema == "uor-r4.reciprocal-transfer-validation-roster/1",
        "unknown roster schema",
    )?;
    require(
        [
            &roster.registration_sha256,
            &roster.selection_sha256,
            &roster.judgments_sha256,
            &roster.source_auth_sha256,
            &roster.tokenizer_sha256,
            &roster.protected_reference_sha256,
        ]
        .iter()
        .all(|digest| is_sha(digest)),
        "unbound registration or qualification provenance",
    )?;
    let mut validation = BTreeMap::new();
    let mut mates = BTreeMap::new();
    let mut selected_ids = BTreeSet::new();
    let mut all_passages = BTreeSet::new();
    let mut titles = BTreeSet::new();
    let mut coverage = true;
    let mut two_titles_with_two = 0;
    let mut two_passages = Vec::new();
    let mut answer_titles = 0;
    let mut mechanical_two = 0;
    let mut mechanical_answer = 0;
    let mut additional = 0;
    for title in &roster.titles {
        require(
            !title.name.is_empty() && titles.insert(title.name.as_str()),
            "duplicate or missing registered validation title",
        )?;
        require(
            {
                let extra = roster.study.additional(title.family, &title.name);
                additional += extra;
                title.selected.len() == title.passages.min(4) + extra
                    && (extra == 0 || title.selected.len() == title.passages)
            } && title.mechanical_candidates >= title.passages,
            "wrong fixed title quota or mechanical count",
        )?;
        let mut qualified = 0;
        for pair in &title.selected {
            require(
                is_sha(&pair.passage_sha256)
                    && all_passages.insert(pair.passage_sha256.as_str())
                    && !pair.left_id.is_empty()
                    && !pair.right_id.is_empty()
                    && selected_ids.insert(pair.left_id.as_str())
                    && selected_ids.insert(pair.right_id.as_str())
                    && (title.family != Family::TwoAnswer || pair.left_id < pair.right_id),
                "forged, duplicate, or noncanonical selected pair",
            )?;
            match &pair.qualification {
                Qualification::Qualified { opening } => {
                    valid_pair(opening, title.family)?;
                    require(
                        opening.cases[0].id == pair.left_id && opening.cases[1].id == pair.right_id,
                        "opening substituted for selected pair",
                    )?;
                    add_cases(&mut validation, opening, true)?;
                    mates.insert(pair.left_id.as_str(), pair.right_id.as_str());
                    mates.insert(pair.right_id.as_str(), pair.left_id.as_str());
                    qualified += 1;
                }
                Qualification::SourceExcluded { reason }
                | Qualification::CaptureExcluded { reason }
                | Qualification::RebindExcluded { reason } => require(
                    !reason.trim().is_empty(),
                    "excluded pair lacks disposition reason",
                )?,
            }
        }
        match title.family {
            Family::TwoAnswer => {
                two_passages.push(title.passages);
                mechanical_two += title.mechanical_candidates;
                coverage &= qualified >= 1;
                two_titles_with_two += usize::from(qualified >= 2);
            }
            Family::Answerability => {
                answer_titles += 1;
                mechanical_answer += title.mechanical_candidates;
                require(
                    title.passages >= 4 && title.mechanical_candidates == title.passages,
                    "wrong answerability candidate census",
                )?;
                coverage &= qualified >= 2;
            }
        }
    }
    two_passages.sort_unstable();
    require(
        two_passages == [1, 2, 7, 9, 10, 18, 30]
            && answer_titles == 11
            && mechanical_two == 187
            && mechanical_answer == 256
            && additional
                == if roster.study == Study::Expanded4341 {
                    46
                } else {
                    0
                },
        "validation title strata or mechanical candidate census differ from registration",
    )?;
    coverage &= two_titles_with_two >= 5;

    let mut train = BTreeMap::new();
    let mut train_passages = BTreeSet::new();
    let mut train_ids = BTreeSet::new();
    for training in &roster.training {
        let mut cases = BTreeMap::new();
        let mut paired = 0;
        for source in &training.sources {
            require(
                !source.title.is_empty()
                    && !titles.contains(source.title.as_str())
                    && is_sha(&source.passage_sha256)
                    && !all_passages.contains(source.passage_sha256.as_str())
                    && (1..=2).contains(&source.openings.len()),
                "TRAIN overlaps validation or has malformed source",
            )?;
            train_passages.insert(source.passage_sha256.as_str());
            if source.openings.len() == 2 {
                let left = &source.openings[0];
                let right = &source.openings[1];
                let [a] = left.cases.as_slice() else {
                    return Err(std::io::Error::other("legacy opening must be a singleton").into());
                };
                let [b] = right.cases.as_slice() else {
                    return Err(std::io::Error::other("legacy opening must be a singleton").into());
                };
                require(
                    a.target.span().is_some()
                        && a.target.span() != b.target.span()
                        && a.target
                            .span()
                            .is_none_or(|span| right.candidates.binary_search(&span).is_ok())
                        && b.target
                            .span()
                            .is_none_or(|span| left.candidates.binary_search(&span).is_ok()),
                    "TRAIN reciprocal target unavailable",
                )?;
                paired += 2;
            }
            require(
                source
                    .openings
                    .iter()
                    .flat_map(|opening| &opening.cases)
                    .all(|case| !selected_ids.contains(case.id.as_str())),
                "TRAIN case ID overlaps validation",
            )?;
            for opening in &source.openings {
                require(
                    opening.cases.len() == 1,
                    "legacy TRAIN opening must be a singleton",
                )?;
                add_cases(&mut cases, opening, false)?;
            }
        }
        let positives = cases.values().filter(|case| case.span().is_some()).count();
        require(
            cases.len() == 264
                && positives == 204
                && paired
                    == match training.mixture {
                        Mixture::Original => 168,
                        Mixture::Natural => 264,
                    },
            "TRAIN polarity or paired/unpaired count differs from registration",
        )?;
        train_ids.extend(cases.keys().copied());
        require(
            train.insert(training.mixture, cases).is_none(),
            "duplicate TRAIN mixture",
        )?;
    }
    require(train.len() == 2, "missing registered TRAIN mixture")?;
    let mut protected = BTreeMap::new();
    for source in &roster.protected {
        require(
            !source.title.is_empty()
                && is_sha(&source.passage_sha256)
                && !all_passages.contains(source.passage_sha256.as_str())
                && !train_passages.contains(source.passage_sha256.as_str()),
            "protected original passage overlaps evaluation or TRAIN",
        )?;
        let [opening] = source.openings.as_slice() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "protected reference must be a singleton",
            )
            .into());
        };
        let [case] = opening.cases.as_slice() else {
            return Err(std::io::Error::other("malformed protected opening").into());
        };
        require(
            case.target.span().is_some()
                && !selected_ids.contains(case.id.as_str())
                && !train_ids.contains(case.id.as_str()),
            "protected case overlaps evaluation or TRAIN, or has no original target",
        )?;
        add_cases(&mut protected, opening, false)?;
    }
    require(
        protected.len() == 4,
        "protected anchor roster must contain exactly four cases",
    )?;
    Ok(CheckedRoster {
        validation,
        mates,
        train,
        protected,
        coverage,
    })
}

fn feature_scores<'a>(
    capture: &'a Capture,
    expected: Observation<'_>,
    key: Key,
) -> Result<&'a Scores> {
    let id = &expected.case.id;
    require(
        capture.opening_sha256 == expected.case.opening_sha256 && capture.features.len() == 2,
        format!("{id}: wrong opening identity or missing feature arm"),
    )?;
    let mut selected = None;
    let mut seen = BTreeSet::new();
    for row in &capture.features {
        require(
            seen.insert(row.arm)
                && row.case_id == *id
                && row.user == expected.case.user
                && row.fact == expected.opening.fact
                && row.question == expected.case.question
                && row
                    .candidates
                    .iter()
                    .map(|span| Span {
                        start: span.start,
                        end: span.end,
                    })
                    .eq(expected.opening.candidates.iter().copied()),
            format!("{id}: changed case, original bounds, or ordered candidate addresses"),
        )?;
        if row.arm == key {
            selected = row.scores.as_ref();
        } else {
            require(
                row.scores.is_none(),
                format!("{id}: unexpected scores from unselected key arm"),
            )?;
        }
    }
    let scores = selected.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{id}: missing explicit interval/NoRead scores"),
        )
    })?;
    let first_best = scores
        .candidates
        .iter()
        .enumerate()
        .max_by_key(|&(i, score)| (score, std::cmp::Reverse(i)));
    let no_read = scores.selection.no_read_score;
    require(
        scores.candidates.len() == expected.opening.candidates.len()
            && scores
                .candidates
                .iter()
                .chain([&no_read])
                .all(|score| (-524288..=524272).contains(score))
            && first_best.is_some_and(|(i, score)| {
                scores.selection.best_candidate_score == Some(*score)
                    && scores.selection.span
                        == (*score > no_read).then(|| expected.opening.candidates[i])
            }),
        format!("{id}: inconsistent signed scores or ordinary selection"),
    )?;
    Ok(scores)
}

fn panel<'a>(panel: &'a Panel, cases: &Cases<'a>, model: &Model) -> Result<ScoresByCase<'a>> {
    let mut scores = BTreeMap::new();
    require(
        panel.capture.len() == cases.len() && panel.reference.len() == cases.len(),
        format!("{:?}: missing or extra cold/reference rows", model.arm),
    )?;
    let reference: BTreeMap<_, _> = panel
        .reference
        .iter()
        .map(|row| (&*row.case_id, &row.scores))
        .collect();
    require(
        reference.len() == panel.reference.len(),
        "duplicate independent reference case",
    )?;
    for capture in &panel.capture {
        let id = capture.request_id.as_str();
        let expected = case_by_id(cases, id)?;
        let value = feature_scores(capture, expected, model.arm.key)?;
        require(
            reference.get(id) == Some(&value),
            format!("{id}: cold client scores differ from independent reference"),
        )?;
        require(
            model.arm.fit != Fit::Zero
                || value
                    .candidates
                    .iter()
                    .chain([&value.selection.no_read_score])
                    .all(|score| *score == 0),
            format!("{id}: nonzero score from zero table"),
        )?;
        require(
            scores.insert(expected.case.id.as_str(), value).is_none(),
            format!("{id}: duplicate cold row"),
        )?;
    }
    Ok(scores)
}

fn case_by_id<'a>(cases: &Cases<'a>, id: &str) -> Result<Observation<'a>> {
    cases.get(id).copied().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("foreign or unregistered case: {id}"),
        )
        .into()
    })
}

pub fn strict_margin(target: Option<Span>, observation: Observation<'_>, scores: &Scores) -> i32 {
    let choices =
        std::iter::once(None).chain(observation.opening.candidates.iter().copied().map(Some));
    let mut target_score = i32::MIN;
    let mut best_rival = i32::MIN;
    for (choice, score) in choices.zip(
        std::iter::once(scores.selection.no_read_score).chain(scores.candidates.iter().copied()),
    ) {
        if choice == target {
            target_score = score;
        } else {
            best_rival = best_rival.max(score);
        }
    }
    if target_score == i32::MIN || best_rival == i32::MIN {
        return i32::MIN;
    }
    target_score - best_rival
}

pub struct CheckedModel<'a> {
    pub validation: ScoresByCase<'a>,
    pub train: ScoresByCase<'a>,
    pub protected: ScoresByCase<'a>,
}

pub fn evidence<'a>(
    roster: &Roster,
    roster_sha: &str,
    checked: &CheckedRoster<'a>,
    evidence: &'a Evidence,
) -> Result<BTreeMap<Arm, CheckedModel<'a>>> {
    require(
        evidence.schema == "uor-r4.reciprocal-transfer-validation-evidence/1"
            && evidence.roster_sha256 == roster_sha
            && evidence.tokenizer_sha256 == roster.tokenizer_sha256
            && evidence.protected_reference_sha256 == roster.protected_reference_sha256
            && is_sha(&evidence.ordinary_client_sha256)
            && is_sha(&evidence.base_writer_sha256),
        "wrong evidence schema, roster, tokenizer, writer or protected binding",
    )?;
    let mut result = BTreeMap::new();
    for model in &evidence.models {
        require(
            [
                &model.artifact_sha256,
                &model.state_sha256,
                &model.fit_input_sha256,
                &model.verified_device_sha256,
                &model.fit_output_sha256,
            ]
            .iter()
            .all(|digest| is_sha(digest))
                && model.fit_replay_state_sha256 == model.state_sha256
                && model.fit_replay_output_sha256 == model.fit_output_sha256,
            format!(
                "{:?}: missing source identity or nondeterministic fit replay",
                model.arm
            ),
        )?;
        let validation = panel(&model.validation, &checked.validation, model)?;
        let train_roster = checked.train.get(&model.arm.mixture).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing TRAIN mixture")
        })?;
        let train = panel(&model.train, train_roster, model)?;
        let protected = panel(&model.protected, &checked.protected, model)?;
        let mut seen = BTreeSet::new();
        require(
            model.swaps.len() == checked.mates.len(),
            "missing or extra exact mate replays",
        )?;
        for swap in &model.swaps {
            let request = swap.request_id.as_str();
            let own = case_by_id(&checked.validation, request)?;
            let peer_id = checked.mates.get(request).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unregistered mate replay request",
                )
            })?;
            let peer = case_by_id(&checked.validation, peer_id)?;
            let value = feature_scores(swap, peer, model.arm.key)?;
            require(
                seen.insert(request) && validation.get(peer_id) == Some(&value),
                format!("{request}: mate replay diverges from its own ordinary row"),
            )?;
            let own_value = validation[request];
            if strict_margin(own.span(), own, own_value) >= 1
                && strict_margin(peer.span(), peer, value) >= 1
            {
                require(
                    strict_margin(own.span(), peer, value) < 1,
                    format!("{request}: retained own credit on mate swap"),
                )?;
            }
        }
        require(
            result
                .insert(
                    model.arm,
                    CheckedModel {
                        validation,
                        train,
                        protected,
                    },
                )
                .is_none(),
            "duplicate model arm",
        )?;
    }
    require(
        result.keys().copied().eq(arms()),
        "omitted or unregistered model controls",
    )?;
    Ok(result)
}
