use super::{report, verify, wire::*, writer};
use serde_json::{Value, json};

const LEFT: Span = Span { start: 1, end: 2 };
const RIGHT: Span = Span { start: 2, end: 3 };
const SAME_VALUE: Span = Span { start: 3, end: 4 };
const WRONG_END: Span = Span { start: 1, end: 3 };

fn opening(name: &str, left: Option<Span>, right: Option<Option<Span>>) -> Opening {
    let fact_ids = vec![71, 72, 71];
    let cases = std::iter::once(left)
        .chain(right)
        .enumerate()
        .map(|(i, span)| {
            let id = format!("{name}-{i}");
            let mut prompt_ids = vec![16, 71, 72, 71, (90 + i) as u32];
            prompt_ids.push(201);
            let target = match span {
                Some(span) => {
                    let source_ids = span.tokens(&prompt_ids).unwrap().to_vec();
                    let mut expected_ids = source_ids.clone();
                    expected_ids.push(1);
                    Target::Source {
                        span,
                        source_ids,
                        expected_ids,
                        response_lexical_offset: 0,
                    }
                }
                None => Target::NoRead {
                    group: Negative::Unanswered,
                },
            };
            Case {
                opening_sha256: sha(id.as_bytes()),
                id,
                prompt_ids,
                user: [1, 5],
                question: [4, 5],
                target,
            }
        })
        .collect();
    Opening {
        fact: [1, 4],
        fact_ids,
        candidates: vec![LEFT, WRONG_END, RIGHT, SAME_VALUE],
        cases,
    }
}

fn fixture() -> (Roster, Evidence) {
    fixture_with(Study::Original4340)
}

fn fixture_with(study: Study) -> (Roster, Evidence) {
    let digest = sha(b"synthetic receipt");
    let reference = sha(b"synthetic anchors");
    let mut titles: Vec<_> = [1usize, 2, 7, 9, 10, 18, 30]
        .into_iter()
        .enumerate()
        .map(|(i, passages)| Title {
            name: format!("two-{i}"),
            family: Family::TwoAnswer,
            mechanical_candidates: passages + usize::from(i == 6) * 110,
            passages,
            selected: Vec::new(),
        })
        .collect();
    titles.extend((0..11).map(|i| {
        let (name, passages) = if study == Study::Expanded4341 {
            match i {
                0 => ("1973_oil_crisis".into(), 14),
                1 => ("Harvard_University".into(), 17),
                2 => ("Islamism".into(), 27),
                3 => (format!("answer-{i}"), 37),
                _ => (format!("answer-{i}"), 23),
            }
        } else {
            (format!("answer-{i}"), 23 + usize::from(i == 10) * 3)
        };
        Title {
            name,
            family: Family::Answerability,
            mechanical_candidates: passages,
            passages,
            selected: Vec::new(),
        }
    }));
    for title in &mut titles {
        for i in 0..title.passages.min(4) + study.additional(title.family, &title.name) {
            let name = format!("{}-{}", title.name, i);
            let right = (title.family == Family::TwoAnswer).then_some(RIGHT);
            let opening = opening(&name, Some(LEFT), Some(right));
            title.selected.push(Pair {
                passage_sha256: sha(name.as_bytes()),
                left_id: opening.cases[0].id.clone(),
                right_id: opening.cases[1].id.clone(),
                qualification: Qualification::Qualified { opening },
            });
        }
    }
    let training = [Mixture::Original, Mixture::Natural]
        .into_iter()
        .map(|mixture| {
            let (positive_pairs, answer_pairs, singleton_positive, singleton_negative) =
                match mixture {
                    Mixture::Original => (60, 24, 60, 36),
                    Mixture::Natural => (72, 60, 0, 0),
                };
            let groups = (0..positive_pairs)
                .map(|_| (Some(LEFT), Some(Some(RIGHT))))
                .chain((0..answer_pairs).map(|_| (Some(LEFT), Some(None))))
                .chain((0..singleton_positive).map(|_| (Some(LEFT), None)))
                .chain((0..singleton_negative).map(|_| (None, None)));
            let sources = groups
                .enumerate()
                .map(|(i, (left, right))| {
                    let name = format!("train-{mixture:?}-{i}");
                    ControlSource {
                        title: format!("train-{mixture:?}"),
                        passage_sha256: sha(name.as_bytes()),
                        openings: {
                            let opening = opening(&name, left, right);
                            opening
                                .cases
                                .iter()
                                .cloned()
                                .map(|case| Opening {
                                    cases: vec![case],
                                    ..opening.clone()
                                })
                                .collect()
                        },
                    }
                })
                .collect();
            Training { mixture, sources }
        })
        .collect();
    let protected = (0..4)
        .map(|i| {
            let name = format!("protected-{i}");
            ControlSource {
                title: "protected".into(),
                passage_sha256: sha(name.as_bytes()),
                openings: vec![opening(&name, Some(LEFT), None)],
            }
        })
        .collect();
    let roster = Roster {
        schema: "uor-r4.reciprocal-transfer-validation-roster/1".into(),
        study,
        registration_sha256: digest.clone(),
        selection_sha256: digest.clone(),
        judgments_sha256: digest.clone(),
        source_auth_sha256: digest.clone(),
        tokenizer_sha256: digest.clone(),
        protected_reference_sha256: reference.clone(),
        titles,
        training,
        protected,
    };
    let checked = verify::roster(&roster).unwrap();
    let models = arms()
        .map(|arm| {
            let state = sha(format!("synthetic {arm:?}").as_bytes());
            let validation = panel(&checked.validation, arm, |row| {
                (arm == PROPOSED).then_some(row.span())
            });
            let mates: std::collections::BTreeMap<_, _> = roster
                .training
                .iter()
                .find(|training| training.mixture == arm.mixture)
                .unwrap()
                .sources
                .iter()
                .filter_map(|source| {
                    let [left, right] = source.openings.as_slice() else {
                        return None;
                    };
                    let (left, right) = (&left.cases[0], &right.cases[0]);
                    Some([
                        (left.id.as_str(), right.target.span()),
                        (right.id.as_str(), left.target.span()),
                    ])
                })
                .flatten()
                .collect();
            let train = panel(&checked.train[&arm.mixture], arm, |row| match arm.fit {
                Fit::Actual => Some(row.span()),
                Fit::Permuted => Some(
                    mates
                        .get(row.case.id.as_str())
                        .copied()
                        .unwrap_or(row.span()),
                ),
                Fit::Zero => None,
            });
            let protected = panel(&checked.protected, arm, |row| {
                (arm == PROPOSED).then_some(row.span())
            });
            let swaps = checked
                .mates
                .iter()
                .map(|(request, mate)| {
                    let peer = checked.validation[mate];
                    capture(
                        request,
                        peer,
                        arm,
                        scores(peer, (arm == PROPOSED).then_some(peer.span())),
                    )
                })
                .collect();
            Model {
                arm,
                artifact_sha256: state.clone(),
                state_sha256: state.clone(),
                verified_device_sha256: digest.clone(),
                fit_input_sha256: digest.clone(),
                fit_replay_state_sha256: state,
                fit_output_sha256: digest.clone(),
                fit_replay_output_sha256: digest.clone(),
                validation,
                swaps,
                train,
                protected,
            }
        })
        .collect();
    let evidence = Evidence {
        schema: "uor-r4.reciprocal-transfer-validation-evidence/1".into(),
        roster_sha256: digest.clone(),
        ordinary_client_sha256: digest.clone(),
        base_writer_sha256: digest.clone(),
        tokenizer_sha256: digest,
        protected_reference_sha256: reference,
        models,
    };
    (roster, evidence)
}

fn scores(row: verify::Observation<'_>, winner: Option<Option<Span>>) -> Scores {
    let candidates = row
        .opening
        .candidates
        .iter()
        .map(|span| i32::from(winner == Some(Some(*span))))
        .collect();
    Scores {
        selection: Selection {
            span: winner.flatten(),
            best_candidate_score: Some(i32::from(winner.flatten().is_some())),
            no_read_score: i32::from(winner == Some(None)),
        },
        candidates,
    }
}

fn capture(request: &str, row: verify::Observation<'_>, arm: Arm, scores: Scores) -> Capture {
    Capture {
        request_id: request.into(),
        opening_sha256: row.case.opening_sha256.clone(),
        features: [Key::Geometry, Key::Count]
            .into_iter()
            .map(|key| NativeFeature {
                case_id: row.case.id.clone(),
                arm: key,
                user: row.case.user,
                fact: row.opening.fact,
                question: row.case.question,
                candidates: row
                    .opening
                    .candidates
                    .iter()
                    .map(|span| NativeCandidate {
                        start: span.start,
                        end: span.end,
                    })
                    .collect(),
                scores: (key == arm.key).then_some(scores.clone()),
            })
            .collect(),
    }
}

fn panel(
    cases: &verify::Cases<'_>,
    arm: Arm,
    winner: impl Fn(verify::Observation<'_>) -> Option<Option<Span>>,
) -> Panel {
    let mut panel = Panel {
        capture: Vec::new(),
        reference: Vec::new(),
    };
    for (id, row) in cases {
        let scores = scores(*row, winner(*row));
        panel.capture.push(capture(id, *row, arm, scores.clone()));
        panel.reference.push(ReferenceRow {
            case_id: (*id).into(),
            scores,
        });
    }
    panel
}

fn score(roster: &Roster, evidence: &Evidence) -> Result<Value> {
    let frozen = verify::roster(roster)?;
    let models = verify::evidence(roster, &evidence.roster_sha256, &frozen, evidence)?;
    Ok(serde_json::to_value(report::score(
        roster,
        &frozen,
        &models,
        &evidence.roster_sha256,
        &sha(b"synthetic score evidence"),
    ))?)
}

#[test]
fn full_synthetic_screen_passes_with_writer_gate_separate() {
    let (roster, evidence) = fixture();
    let result = score(&roster, &evidence).unwrap();
    assert_eq!(result["source_status"], "pass");
    assert_eq!(result["two_answer_titles_strictly_above_both"], 7);
    assert_eq!(
        result["two_answer_contrasts"]["natural_geometry_vs_natural_count"],
        json!({"numerator":1,"denominator":1})
    );
    assert_eq!(result["writer_grade_required"], true);
    assert_eq!(result["locked_access_authorized"], false);
}

#[test]
fn missing_controls_rows_or_references_never_drop_denominators() {
    let (roster, mut evidence) = fixture();
    let missing = evidence.models.pop().unwrap();
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("omitted")
    );
    evidence.models.push(missing);
    let arm = evidence.models.last_mut().unwrap();
    let row = arm.validation.capture.pop().unwrap();
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("missing or extra cold/reference")
    );
    let arm = evidence.models.last_mut().unwrap();
    arm.validation.capture.push(row);
    arm.validation.reference[0].scores.selection.no_read_score += 1;
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("independent reference")
    );
}

#[test]
fn forgery_and_noncanonical_pairs_refuse() {
    let (mut roster, evidence) = fixture();
    let (_, roster_bytes) = ("roster", serde_json::to_vec(&roster).unwrap());
    let roster_digest = sha(&roster_bytes);
    let current_exe = std::env::current_exe().unwrap();
    let fixture_file = current_exe
        .parent()
        .unwrap()
        .join("forged-synthetic-roster.json");
    std::fs::write(&fixture_file, &roster_bytes).unwrap();
    assert!(pinned::<Roster>(&fixture_file, &roster_digest).is_ok());
    roster.titles[0].selected[0].left_id = "invented-foreign-id".into();
    std::fs::write(&fixture_file, serde_json::to_vec(&roster).unwrap()).unwrap();
    assert!(pinned::<Roster>(&fixture_file, &roster_digest).is_err());
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("substituted")
    );
    let pair = &mut roster.titles[0].selected[0];
    pair.left_id = pair.right_id.clone();
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("noncanonical")
    );
    std::fs::remove_file(fixture_file).unwrap();
}

#[test]
fn overlap_between_train_and_validation_refuses() {
    let (mut roster, evidence) = fixture();
    roster.training[0].sources[0]
        .passage_sha256
        .clone_from(&roster.titles[0].selected[0].passage_sha256);
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("overlaps validation")
    );
}

#[test]
fn insufficient_coverage_preserves_selected_denominator() {
    let (mut roster, mut evidence) = fixture();
    let pair = &mut roster.titles[0].selected[0];
    let ids = [pair.left_id.clone(), pair.right_id.clone()];
    pair.qualification = Qualification::CaptureExcluded {
        reason: "synthetic missing authentic span".into(),
    };
    for model in &mut evidence.models {
        model
            .validation
            .capture
            .retain(|row| !ids.contains(&row.request_id));
        model
            .validation
            .reference
            .retain(|row| !ids.contains(&row.case_id));
        model.swaps.retain(|row| !ids.contains(&row.request_id));
    }
    let result = score(&roster, &evidence).unwrap();
    assert_eq!(result["source_status"], "instrument_inconclusive");
    let arm = result["arms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["arm"] == serde_json::to_value(PROPOSED).unwrap())
        .unwrap();
    assert_eq!(arm["two_answer"]["selected_pairs"]["pooled"]["total"], 23);
    assert_eq!(arm["two_answer"]["qualified_pairs"]["pooled"]["total"], 22);
    assert_eq!(
        arm["two_answer"]["qualified_pairs"]["macro_rate"],
        Value::Null
    );
    assert_eq!(arm["titles"][0]["rows"][0]["stage"], "capture_excluded");
}

#[test]
fn mate_and_prediction_forgeries_have_distinct_failures() {
    let (roster, mut evidence) = fixture();
    let proposed = evidence
        .models
        .iter_mut()
        .find(|model| model.arm == PROPOSED)
        .unwrap();
    let saved = proposed.swaps[0].clone();
    proposed.swaps[0] = proposed.validation.capture[0].clone();
    assert!(
        score(&roster, &evidence)
            .unwrap_err()
            .to_string()
            .contains("wrong opening identity")
    );
    let proposed = evidence
        .models
        .iter_mut()
        .find(|model| model.arm == PROPOSED)
        .unwrap();
    proposed.swaps[0] = saved;
    let checked = verify::roster(&roster).unwrap();
    let model = PROPOSED;
    proposed.validation = panel(&checked.validation, model, |row| {
        let mate = checked.mates[row.case.id.as_str()];
        Some(checked.validation[mate].span())
    });
    proposed.swaps = checked
        .mates
        .iter()
        .map(|(request, mate)| {
            let peer = checked.validation[mate];
            capture(
                request,
                peer,
                model,
                scores(peer, Some(checked.validation[request].span())),
            )
        })
        .collect();
    let result = score(&roster, &evidence).unwrap();
    assert_eq!(result["source_status"], "negative");
}

#[test]
fn explicit_noread_tie_wrong_end_and_duplicate_surface_are_all_unsolved() {
    let (roster, _) = fixture();
    let checked = verify::roster(&roster).unwrap();
    let positive = checked.validation["answer-0-0-0"];
    let negative = checked.validation["answer-0-0-1"];
    assert_eq!(
        verify::strict_margin(
            positive.span(),
            positive,
            &scores(positive, Some(Some(WRONG_END)))
        ),
        -1
    );
    assert_eq!(
        verify::strict_margin(
            positive.span(),
            positive,
            &scores(positive, Some(Some(SAME_VALUE)))
        ),
        -1
    );
    assert_eq!(
        verify::strict_margin(None, negative, &scores(negative, None)),
        0
    );
    assert_eq!(
        verify::strict_margin(None, negative, &scores(negative, Some(None))),
        1
    );
}

#[test]
fn writer_false_claim_and_unclassified_outputs_block_locked_registration() {
    let (roster, evidence) = fixture();
    let checked = verify::roster(&roster).unwrap();
    let models = verify::evidence(&roster, &evidence.roster_sha256, &checked, &evidence).unwrap();
    let evidence_sha = sha(b"synthetic score evidence");
    let result = report::score(
        &roster,
        &checked,
        &models,
        &evidence.roster_sha256,
        &evidence_sha,
    );
    let artifact = &evidence
        .models
        .iter()
        .find(|row| row.arm == PROPOSED)
        .unwrap()
        .artifact_sha256;
    let model = &models[&PROPOSED];
    let grade = |negative_status: &str| {
        let mut rows = Vec::new();
        for (id, row) in &checked.validation {
            let (generated, stop, commits, visible, ledger) = match &row.case.target {
                Target::Source {
                    span, expected_ids, ..
                } => (
                    expected_ids.clone(),
                    "eos",
                    vec![json!({"offset":0, "pos":span.start, "id": expected_ids[0]})],
                    Value::Null,
                    Value::Null,
                ),
                Target::NoRead { .. } => (
                    vec![1],
                    "eos",
                    Vec::new(),
                    json!(negative_status),
                    json!(sha(id.as_bytes())),
                ),
            };
            rows.push(json!({"case_id":id,"opening_sha256":row.case.opening_sha256,"capture_sha256":sha(id.as_bytes()),
                "generated_ids":generated,"stop":stop,"selection":model.validation[id].selection,
                "source_admitted":row.span().is_some(),"original_source_commits":commits,
                "visible":visible,"visible_ledger_key":ledger}));
        }
        let receipt = serde_json::from_value(json!({
            "schema":"uor-r4.reciprocal-transfer-validation-writer-evidence/1",
            "roster_sha256":evidence.roster_sha256,"scores_sha256":evidence_sha,"artifact_sha256":artifact,
            "grader_sha256":"ec50ae9337ff727307c1f62e992788f0dff59dae302645c7af105fb4da693971",
            "writer_trace_verification_sha256":sha(b"synthetic writer verifier"),
            "visible_ledger_sha256":sha(b"synthetic visibility ledger"),"cases":rows
        })).unwrap();
        let grade =
            writer::grade(&result, &checked, model, artifact, &evidence_sha, &receipt).unwrap();
        serde_json::to_value(grade).unwrap()
    };
    let fail = grade("false_factual_claim");
    assert_eq!(fail["writer_status"], "fail");
    assert_eq!(fail["eligible_for_separate_locked_registration"], false);
    assert_eq!(fail["negative_source_commits"], 0);
    let unresolved = grade("requires_semantic_review");
    assert_eq!(unresolved["writer_status"], "unresolved");
    assert_eq!(
        unresolved["eligible_for_separate_locked_registration"],
        false
    );
    let pass = grade("empty_reply");
    assert_eq!(pass["writer_status"], "pass");
    assert_eq!(pass["negative_visible"]["visible_abstention"], 0);
    assert_eq!(pass["eligible_for_separate_locked_registration"], true);
    assert_eq!(pass["locked_access_authorized"], false);
}

#[test]
fn ordinary_tape_adapters_enforce_byte_identity_and_exact_device_scores() {
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("synthetic-adapter");
    std::fs::create_dir_all(&dir).unwrap();
    let blind = dir.join("blind.jsonl");
    let gold = dir.join("gold.jsonl");
    let truth = dir.join("truth.jsonl");
    let ids = dir.join("ids.jsonl");
    let native = dir.join("native.jsonl");
    let gpu = dir.join("cuda.scores");
    let opening = opening("adapter", Some(LEFT), Some(Some(RIGHT)));
    let write_tape = |path: &std::path::Path, rows: &[Value]| {
        let text = rows
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, text).unwrap();
    };
    let mut prompts = Vec::new();
    let mut labels = Vec::new();
    let mut independent = Vec::new();
    let mut reencoded = Vec::new();
    let mut features = Vec::new();
    let mut device = Vec::new();
    for (i, case) in opening.cases.iter().enumerate() {
        let input = format!("a b c q{i}");
        prompts.push(json!({"case_id":case.id,"text":input}));
        let span = case.target.span().unwrap();
        labels.push(json!({"case_id":case.id,"start":span.start,"end":span.end,
            "swapped_case_id":opening.cases[1-i].id}));
        independent.push(json!({"case_id":case.id,"canonical_prefix":case.prompt_ids,
            "user_bounds":case.user,"source_ids":span.tokens(&case.prompt_ids).unwrap(),
            "original_bytes":[i*2,i*2+1],"expected_ids":[case.prompt_ids[usize::from(span.start)],1],
            "response_lexical_offset":0}));
        reencoded.push(
            json!({"case_id":case.id,"user":case.user,"ids":case.prompt_ids,
            "byte_bounds":[[1,2],[3,4],[5,6],[7,9]]}),
        );
        let observation = verify::Observation {
            opening: &opening,
            case,
        };
        let score = scores(observation, Some(Some(span)));
        let mut row = capture(&case.id, observation, PROPOSED, score.clone());
        features.extend(
            row.features
                .drain(..)
                .map(|feature| serde_json::to_value(feature).unwrap()),
        );
        let winner = 1 + opening
            .candidates
            .iter()
            .position(|candidate| *candidate == span)
            .unwrap() as i32;
        for word in std::iter::once(winner)
            .chain([score.selection.no_read_score])
            .chain(score.candidates)
        {
            device.extend(word.to_le_bytes());
        }
    }
    write_tape(&blind, &prompts);
    write_tape(&gold, &labels);
    write_tape(&truth, &independent);
    write_tape(&ids, &reencoded);
    write_tape(&native, &features);
    std::fs::write(&gpu, &device).unwrap();
    let arguments = [&blind, &gold, &truth, &ids, &native].map(|path| path.to_str().unwrap());
    let use_opening = || {
        crate::adapt::opening(
            "validation",
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
        )
    };
    assert_eq!(use_opening().unwrap().cases.len(), 2);
    let use_panel = || {
        crate::adapt::panel(
            "geometry",
            arguments[0],
            arguments[4],
            gpu.to_str().unwrap(),
        )
    };
    assert_eq!(use_panel().unwrap().reference.len(), 2);
    independent[0]["original_bytes"] = json!([1, 2]);
    write_tape(&truth, &independent);
    assert!(
        use_opening()
            .unwrap_err()
            .to_string()
            .contains("answer bytes")
    );
    device[4] ^= 1;
    std::fs::write(&gpu, &device).unwrap();
    assert!(
        use_panel()
            .unwrap_err()
            .to_string()
            .contains("differ from cold client")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn expanded_report_preserves_original_and_complete_development_denominators() {
    let (roster, evidence) = fixture_with(Study::Expanded4341);
    let result = score(&roster, &evidence).unwrap();
    let proposed = result["arms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["arm"] == serde_json::to_value(PROPOSED).unwrap())
        .unwrap();
    assert_eq!(
        proposed["two_answer"]["selected_pairs"]["pooled"]["total"],
        23
    );
    assert_eq!(
        proposed["answerability"]["selected_pairs"]["pooled"]["total"],
        90
    );
    assert_eq!(
        proposed["answerability"]["original_selected_pairs"]["pooled"]["total"],
        44
    );
    assert_eq!(
        proposed["answerability"]["original_qualified_pairs"]["pooled"]["total"],
        44
    );
    let mut wrong = roster;
    wrong
        .titles
        .iter_mut()
        .find(|title| title.name == "Islamism")
        .unwrap()
        .selected
        .pop();
    assert!(
        score(&wrong, &evidence)
            .unwrap_err()
            .to_string()
            .contains("quota")
    );
}

#[test]
fn legacy_family_order_can_differ_but_the_extension_order_is_fixed() {
    let (roster, _) = fixture_with(Study::Expanded4341);
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("synthetic-4341-roster");
    std::fs::create_dir_all(&dir).unwrap();
    let mut original = Vec::new();
    let mut additional = Vec::new();
    let mut mechanical = [Vec::new(), Vec::new()];
    for title in roster.titles {
        let family = if title.family == Family::TwoAnswer {
            "two_answer"
        } else {
            "answerability"
        };
        let mut candidates = Vec::new();
        for (i, pair) in title.selected.iter().enumerate() {
            let identity = json!({"family":family,"article":title.name,"passage_sha256":pair.passage_sha256,
                "left":pair.left_id,"right":pair.right_id,"rank":i+1,
                "pair_id":format!("{family}-{}",pair.passage_sha256)});
            let disposition = if i < title.passages.min(4) {
                original.push(identity.clone());
                "review"
            } else {
                additional.push(identity);
                "outside_article_quota"
            };
            candidates.push(json!({"rank":i+1,"disposition":disposition,
                "pair":{"article":title.name,"passage_sha256":pair.passage_sha256,"left":pair.left_id,"right":pair.right_id}}));
        }
        for i in candidates.len()..title.mechanical_candidates {
            let pair_id = format!("{}-{i}", title.name);
            candidates.push(json!({"rank":i+1,"disposition":"unused_synthetic_candidate",
                "pair":{"article":title.name,"passage_sha256":sha(pair_id.as_bytes()),"left":format!("{pair_id}-a"),"right":format!("{pair_id}-b")}}));
        }
        mechanical[usize::from(title.family == Family::Answerability)].push(json!({
            "title":title.name,"passages":title.passages,"pairs":title.mechanical_candidates,"candidates":candidates}));
    }
    let selection = dir.join("selection.json");
    let identities = dir.join("identities.jsonl");
    let disposition_file = dir.join("pairs.jsonl");
    std::fs::write(
        &selection,
        json!({"schema":"natural-reciprocal-validation-selection/1",
        "two_answer":mechanical[0],"answerability":mechanical[1]})
        .to_string(),
    )
    .unwrap();
    let as_jsonl = |rows: &[Value]| {
        rows.iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut selector_order = original.clone();
    selector_order.sort_by_key(|row| row["family"].as_str().unwrap().to_owned());
    selector_order.extend(additional.clone());
    std::fs::write(&identities, as_jsonl(&selector_order)).unwrap();
    let mut dispositions: Vec<_> = original.iter().chain(&additional).map(|row| json!({
        "pair_id":row["pair_id"],"family":row["family"],"article":row["article"],
        "passage_sha256":row["passage_sha256"],"left_case_id":row["left"],"right_case_id":row["right"],
        "accepted":false,"reasons":["synthetic source exclusion"],"native_reasons":[],"qualified":false
    })).collect();
    std::fs::write(&disposition_file, as_jsonl(&dispositions)).unwrap();
    let validate = || {
        crate::adapt::validation(
            Study::Expanded4341,
            selection.to_str().unwrap(),
            identities.to_str().unwrap(),
            dir.to_str().unwrap(),
        )
    };
    assert_eq!(
        validate()
            .unwrap()
            .iter()
            .map(|title| title.selected.len())
            .sum::<usize>(),
        113
    );
    dispositions[0]["left_case_id"] = json!("wrong-official-id");
    std::fs::write(&disposition_file, as_jsonl(&dispositions)).unwrap();
    assert!(
        validate()
            .unwrap_err()
            .to_string()
            .contains("original identity")
    );
    dispositions[0]["left_case_id"] = original[0]["left"].clone();
    dispositions.swap(67, 68);
    std::fs::write(&disposition_file, as_jsonl(&dispositions)).unwrap();
    assert!(
        validate()
            .unwrap_err()
            .to_string()
            .contains("additional qualification order")
    );
    std::fs::remove_dir_all(dir).unwrap();
}
