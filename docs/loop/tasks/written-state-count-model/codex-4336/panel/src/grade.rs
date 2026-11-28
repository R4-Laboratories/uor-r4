use crate::{
    capture::{baseline, Record},
    open_panel, read_lines, require, write_json, write_lines, Key, Result, ARTIFACT_SHA,
    DECODE_LIMIT, TOKENIZER_SHA,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use uor_r4_tokenizer::{dialogue::DialogueProtocol, ByteBpeTokenizer};

#[derive(Deserialize)]
struct Client {
    schema: String,
    protocol: String,
    artifact_sha256: String,
    tokenizer_sha256: String,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    stop: String,
    text: String,
    #[serde(default)]
    authenticated_copy_steps: Vec<OrdinaryStep>,
    #[serde(default)]
    native_occurrence_steps: Vec<Occurrence>,
    #[serde(default)]
    verified_span_release: Option<Release>,
    #[serde(default)]
    native_period_release: Value,
}

#[derive(Deserialize)]
struct OrdinaryStep {
    committed_id: u32,
    claim_pos: Option<usize>,
    read_pos: Option<usize>,
    source_id: Option<u32>,
    copied: bool,
}

#[derive(Deserialize)]
struct Occurrence {
    #[serde(flatten)]
    step: OrdinaryStep,
    prefix_start: usize,
    prefix_len: usize,
    previous_id: Option<u32>,
    source_pos: Option<usize>,
}

#[derive(Deserialize)]
struct Release {
    schema: String,
    trace: bool,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    phase: Option<String>,
    event: String,
    second_event: Option<String>,
    source_pos: Option<usize>,
    source_id: Option<u32>,
    ordinary_id: u32,
    committed_id: u32,
    commit: String,
    action: Option<String>,
    start: Option<usize>,
    cursor: Option<usize>,
    end: Option<usize>,
    prompt_end: usize,
}

#[derive(Serialize)]
struct Difference {
    offset: usize,
    expected_id: Option<u32>,
    actual_id: Option<u32>,
}

#[derive(Serialize)]
struct Boundary {
    offset: usize,
    event: String,
    second_event: Option<String>,
    action: Option<String>,
    commit: String,
    ordinary_id: u32,
    committed_id: u32,
    cursor: Option<usize>,
    end: Option<usize>,
}

#[derive(Serialize)]
struct Score {
    id: String,
    arm: String,
    length: usize,
    order: usize,
    wording: usize,
    pair: Option<String>,
    request: String,
    first_id_correct: Option<bool>,
    first_span_ids_exact: Option<bool>,
    first_copy_address_correct: Option<bool>,
    first_span_address_exact: Option<bool>,
    first_wrong_committed: Option<Difference>,
    exact_reply_through_eos: Option<bool>,
    exact_reply_with_addresses: Option<bool>,
    original_addresses: Vec<Option<usize>>,
    first_original_claim: Option<usize>,
    first_original_read: Option<usize>,
    unsupported_original_claim: Option<bool>,
    unsupported_original_copy: Option<bool>,
    reached_eos: bool,
    period_release_observed: bool,
    first_boundary: Option<Boundary>,
    second_boundary: Option<Boundary>,
    supplied_stop: bool,
    control_passed: Option<bool>,
}

fn fail(record: &Record, message: &str) -> String {
    format!("{}/{}: {message}", record.id, record.arm)
}

fn first_difference(actual: &[u32], expected: &[u32]) -> Option<Difference> {
    (0..actual.len().max(expected.len())).find_map(|offset| {
        let actual_id = actual.get(offset).copied();
        let expected_id = expected.get(offset).copied();
        (actual_id != expected_id).then_some(Difference {
            offset,
            expected_id,
            actual_id,
        })
    })
}

fn boundary(offset: usize, step: &Step) -> Boundary {
    Boundary {
        offset,
        event: step.event.clone(),
        second_event: step.second_event.clone(),
        action: step.action.clone(),
        commit: step.commit.clone(),
        ordinary_id: step.ordinary_id,
        committed_id: step.committed_id,
        cursor: step.cursor,
        end: step.end,
    }
}

fn emission_eq(a: &Value, b: &Value) -> bool {
    ["prompt_ids", "generated_ids", "stop", "text"]
        .iter()
        .all(|key| a[key] == b[key])
}

fn original_pos(key: &Key, relative: Option<usize>, start: usize) -> Option<usize> {
    relative
        .and_then(|pos| start.checked_add(pos))
        .filter(|pos| (key.user_bounds[0]..key.user_bounds[1]).contains(pos))
}

fn turns_and_sources(
    client: &Client,
    key: &Key,
    record: &Record,
    eos: u32,
) -> Result<(Vec<Option<usize>>, Option<usize>, Option<usize>)> {
    let ids = &client.generated_ids;
    let traced = !record.arm.ends_with("_stock") && !record.arm.ends_with("_quiet");
    require(
        client.native_occurrence_steps.len() == if traced { ids.len() } else { 0 },
        || fail(record, "native trace switch or length differs"),
    )?;
    let mut sources = vec![None; ids.len()];
    let mut claim = None;
    let mut read = None;
    let mut causal = key.prompt_ids.clone();
    for (offset, (&id, event)) in ids.iter().zip(&client.native_occurrence_steps).enumerate() {
        let auth = client
            .authenticated_copy_steps
            .get(offset)
            .ok_or_else(|| fail(record, "missing authenticated writer step"))?;
        require(
            event.prefix_start + event.prefix_len == causal.len()
                && event.previous_id == causal.last().copied()
                && event.step.committed_id == id
                && auth.committed_id == id
                && auth.claim_pos == event.step.claim_pos
                && auth.read_pos == event.source_pos
                && auth.read_pos == event.step.read_pos
                && auth.copied == event.step.copied,
            || {
                fail(
                    record,
                    "native trace diverged from actual self-fed causal prefix",
                )
            },
        )?;
        if auth.copied {
            let copied = auth
                .read_pos
                .and_then(|pos| causal.get(pos + event.prefix_start));
            require(copied == Some(&id) && auth.source_id == Some(id), || {
                fail(
                    record,
                    "ordinary committed copy fails original/causal token identity",
                )
            })?;
            sources[offset] = original_pos(key, auth.read_pos, event.prefix_start);
        }
        claim = claim.or_else(|| original_pos(key, auth.claim_pos, event.prefix_start));
        read = read.or_else(|| original_pos(key, auth.read_pos, event.prefix_start));
        if id != eos {
            causal.push(id);
        }
    }
    if !record.arm.ends_with("_stock") {
        require(client.authenticated_copy_steps.len() == ids.len(), || {
            fail(
                record,
                "authenticated writer length does not match generated IDs",
            )
        })?;
    }
    Ok((sources, claim, read))
}

fn grade_record(
    tokenizer: &ByteBpeTokenizer,
    eos: u32,
    key: &Key,
    record: &Record,
    other: &BTreeMap<(String, String), &Record>,
) -> Result<Score> {
    let client: Client = serde_json::from_value(record.client.clone())?;
    let ids = &client.generated_ids;
    let decoded = ids.strip_suffix(&[eos]);
    require(
        client.schema == "uor-r4.uor-chat-discrete-text/1"
            && client.protocol == "uor-r4.literal-role-dialogue/2"
            && client.artifact_sha256 == ARTIFACT_SHA
            && client.tokenizer_sha256 == TOKENIZER_SHA
            && client.prompt_ids == key.prompt_ids
            && !ids.is_empty()
            && ids.len() <= DECODE_LIMIT
            && !ids[..ids.len() - 1].contains(&eos)
            && if let Some(visible) = decoded {
                client.stop == "eos" && client.text == tokenizer.decode(visible)
            } else {
                client.stop == "max_tokens"
                    && ids.len() == DECODE_LIMIT
                    && client.text == tokenizer.decode(ids)
            },
        || {
            fail(
                record,
                "invalid trace identity, serialization, stop, or text",
            )
        },
    )?;
    let (mut addresses, claimed, read) = turns_and_sources(&client, key, record, eos)?;
    let mut first_boundary = None;
    let mut second_boundary = None;
    let mut supplied_stop = false;
    let mut verified_copies = Vec::new();
    let mut all_trace_copies = 0;
    if baseline(&record.arm) {
        require(client.verified_span_release.is_none(), || {
            fail(record, "baseline used a span")
        })?;
    } else {
        let release = client
            .verified_span_release
            .ok_or_else(|| fail(record, "diagnostic trace missing"))?;
        let traced = !record.arm.ends_with("_quiet");
        require(
            release.schema == "uor-r4.verified-span-release/1"
                && release.trace == traced
                && release.steps.len() == if traced { ids.len() } else { 0 },
            || fail(record, "bad span trace identity, switch, or length"),
        )?;
        for (offset, (&id, step)) in ids.iter().zip(&release.steps).enumerate() {
            require(
                step.prompt_end == key.prompt_ids.len()
                    && step.committed_id == id
                    && matches!(
                        step.event.as_str(),
                        "copy" | "verified_end" | "cap" | "delimiter" | "rejected" | "ordinary"
                    )
                    && step.start.zip(step.end).is_none_or(|(s, e)| {
                        key.user_bounds[0] <= s
                            && s < e
                            && e <= key.user_bounds[1]
                            && step.cursor.is_some_and(|c| s <= c && c <= e)
                    })
                    && step
                        .source_pos
                        .zip(step.source_id)
                        .is_none_or(|(pos, tok)| {
                            original_pos(key, Some(pos), 0) == Some(pos)
                                && key.prompt_ids.get(pos) == Some(&tok)
                        })
                    && step.source_pos.is_some() == step.source_id.is_some(),
                || {
                    fail(
                        record,
                        "span trace fails address, token, event, fence, or commit integrity",
                    )
                },
            )?;
            let boundary_holds = match step.event.as_str() {
                "verified_end" => {
                    step.cursor.is_some()
                        && step.cursor == step.end
                        && (step.source_pos.is_none()
                            || matches!(step.second_event.as_deref(), Some("copy" | "delimiter")))
                }
                "cap" => {
                    step.start.zip(step.cursor).is_some_and(|(start, cursor)| {
                        cursor == start + 8 && step.end.is_some_and(|end| cursor < end)
                    }) && step.source_pos.is_none()
                        && step.action.is_none()
                }
                "delimiter" => {
                    step.cursor.is_some()
                        && step.cursor == step.source_pos
                        && step
                            .source_id
                            .is_some_and(|id| [16, 1, 0, 201].contains(&id))
                }
                _ => true,
            };
            require(boundary_holds, || {
                fail(
                    record,
                    "boundary metadata does not certify its declared release",
                )
            })?;
            match step.commit.as_str() {
                "copy" => {
                    let pos = step
                        .source_pos
                        .ok_or_else(|| fail(record, "copy has no source position"))?;
                    require(
                        step.source_id == Some(id)
                            && (step.event == "copy"
                                || step.second_event.as_deref() == Some("copy")),
                        || {
                            fail(
                                record,
                                "copy differs from the traced original token and event",
                            )
                        },
                    )?;
                    addresses[offset] = Some(pos);
                    verified_copies.push(pos);
                    all_trace_copies += 1;
                }
                "ordinary" => require(id == step.ordinary_id, || {
                    fail(record, "fallback committed a different ID")
                })?,
                "supplied_stop" => {
                    require(
                        step.event == "verified_end"
                            && step.action.as_deref() == Some("stop")
                            && id == eos,
                        || fail(record, "EOS intervention outside authenticated end"),
                    )?;
                    supplied_stop = true;
                }
                _ => return Err(fail(record, "unknown commit kind").into()),
            }
            if first_boundary.is_none()
                && step.phase.as_deref() == Some("first")
                && step.event != "copy"
            {
                first_boundary = Some(boundary(offset, step));
            }
            if second_boundary.is_none()
                && (step.phase.as_deref() == Some("second") && step.event != "copy"
                    || step
                        .second_event
                        .as_deref()
                        .is_some_and(|event| event != "copy"))
            {
                second_boundary = Some(boundary(offset, step));
            }
        }
    }
    let expected = key.expected_ids.as_deref();
    let has_address_trace = !client.native_occurrence_steps.is_empty();
    let exact = expected.map(|expected| {
        ids == expected
            && key.expected_text.as_deref() == Some(client.text.as_str())
            && decoded.is_some()
    });
    let expect_address_at = |offset: usize, pos: usize| addresses.get(offset) == Some(&Some(pos));
    let first_copy_address_correct = expected
        .filter(|_| has_address_trace)
        .map(|_| expect_address_at(0, key.first.start));
    let unsupported_original_copy = (key.pair.is_none()
        && !client.native_occurrence_steps.is_empty())
    .then(|| addresses.iter().any(Option::is_some));
    let first_span_address_exact = expected.filter(|_| has_address_trace).map(|_| {
        (key.first.start..key.first.end)
            .enumerate()
            .all(|(offset, pos)| expect_address_at(offset, pos))
    });
    let exact_address = exact.filter(|_| has_address_trace).map(|exact| {
        exact
            && key
                .expected_addresses
                .iter()
                .copied()
                .enumerate()
                .all(|(offset, pos)| expect_address_at(offset, pos))
    });
    let first_event = first_boundary.as_ref().map(|step| step.event.as_str());
    let first_boundary_at = |offset: usize, event: &str| {
        first_boundary
            .as_ref()
            .is_some_and(|step| step.offset == offset && step.event == event)
    };
    let fallback = || {
        other
            .get(&(key.id.clone(), "new_auth".into()))
            .is_some_and(|baseline| emission_eq(&record.client, &baseline.client))
    };
    let copied_prefix = || {
        verified_copies
            .iter()
            .copied()
            .take(key.length.min(8))
            .eq(key.first.start..key.first.start + key.length.min(8))
    };
    let control_passed = match record.arm.as_str() {
        "early_end" | "late_end" | "rival_address" | "missing_claim" | "missing_proof"
        | "wrong_prompt" | "unsupported_claim" => Some(
            first_boundary_at(0, "rejected")
                && all_trace_copies == 0
                && !supplied_stop
                && fallback(),
        ),
        "wrong_second" => Some(
            first_boundary_at(key.length, "verified_end")
                && first_boundary
                    .as_ref()
                    .and_then(|step| step.second_event.as_deref())
                    == Some("rejected")
                && copied_prefix()
                && all_trace_copies == key.length
                && !supplied_stop,
        ),
        "delimiter" => Some(
            first_boundary_at(key.length, "delimiter")
                && copied_prefix()
                && all_trace_copies == key.length
                && !supplied_stop,
        ),
        "oracle" | "native" | "swapped" if key.length == 9 => Some(
            first_boundary_at(8, "cap")
                && copied_prefix()
                && all_trace_copies == 8
                && !supplied_stop
                && first_boundary
                    .as_ref()
                    .is_some_and(|step| step.action.is_none()),
        ),
        "oracle" => Some(
            exact_address == Some(true) && first_event == Some("verified_end") && supplied_stop,
        ),
        "native" => Some(
            first_boundary_at(key.length, "verified_end")
                && copied_prefix()
                && !supplied_stop
                && first_boundary
                    .as_ref()
                    .and_then(|step| step.action.as_deref())
                    == Some("native"),
        ),
        "swapped" => {
            let mut opposite_ids = key.first.ids.clone();
            let mut opposite_addresses = (key.first.start..key.first.end).collect::<Vec<_>>();
            if key.request == "stop" {
                opposite_ids.extend(&key.second.ids);
                opposite_addresses.extend(key.second.start..key.second.end);
            }
            opposite_ids.push(eos);
            Some(
                first_boundary_at(key.length, "verified_end")
                    && supplied_stop
                    && ids == &opposite_ids
                    && verified_copies == opposite_addresses,
            )
        }
        _ => None,
    };
    Ok(Score {
        id: key.id.clone(),
        arm: record.arm.clone(),
        length: key.length,
        order: key.order,
        wording: key.wording,
        pair: key.pair.clone(),
        request: key.request.clone(),
        first_id_correct: expected.map(|expected| ids.first() == expected.first()),
        first_span_ids_exact: expected.map(|_| ids.starts_with(&key.first.ids)),
        first_copy_address_correct,
        first_span_address_exact,
        first_wrong_committed: expected.and_then(|expected| first_difference(ids, expected)),
        exact_reply_through_eos: exact,
        exact_reply_with_addresses: exact_address,
        original_addresses: addresses,
        first_original_claim: claimed,
        first_original_read: read,
        unsupported_original_claim: (key.pair.is_none()
            && !client.native_occurrence_steps.is_empty())
        .then_some(claimed.is_some()),
        unsupported_original_copy,
        reached_eos: decoded.is_some(),
        period_release_observed: !client.native_period_release.is_null(),
        first_boundary,
        second_boundary,
        supplied_stop,
        control_passed,
    })
}

fn counts(scores: &[&Score]) -> Value {
    let count =
        |predicate: fn(&Score) -> bool| scores.iter().copied().filter(|row| predicate(row)).count();
    json!({
        "rows":scores.len(),
        "supported":count(|r| r.pair.is_some()),
        "first_id_correct":count(|r| r.first_id_correct == Some(true)),
        "first_span_ids_exact":count(|r| r.first_span_ids_exact == Some(true)),
        "first_copy_address_correct":count(|r| r.first_copy_address_correct == Some(true)),
        "first_span_address_exact":count(|r| r.first_span_address_exact == Some(true)),
        "exact_reply_through_eos":count(|r| r.exact_reply_through_eos == Some(true)),
        "exact_reply_with_addresses":count(|r| r.exact_reply_with_addresses == Some(true)),
        "eos":count(|r| r.reached_eos),
        "period_release_observed":count(|r| r.period_release_observed),
        "unsupported_claim_observed":count(|r| r.unsupported_original_claim == Some(true)),
        "unsupported_copy_observed":count(|r| r.unsupported_original_copy == Some(true)),
        "unsupported_scored":count(|r| r.unsupported_original_claim.is_some()),
        "supplied_stop":count(|r| r.supplied_stop),
        "verified_first_end":count(|r| r.first_boundary.as_ref().is_some_and(|s| s.event == "verified_end")),
        "first_cap":count(|r| r.first_boundary.as_ref().is_some_and(|s| s.event == "cap")),
        "first_delimiter":count(|r| r.first_boundary.as_ref().is_some_and(|s| s.event == "delimiter")),
        "first_rejected":count(|r| r.first_boundary.as_ref().is_some_and(|s| s.event == "rejected")),
        "second_rejected":count(|r| r.first_boundary.as_ref().is_some_and(|s| s.second_event.as_deref() == Some("rejected"))),
        "controls_scored":count(|r| r.control_passed.is_some()),
        "controls_passed":count(|r| r.control_passed == Some(true)),
    })
}

fn summary(scores: &[Score], lookup: &BTreeMap<(String, String), &Record>) -> Value {
    let mut by_arm: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for score in scores {
        by_arm.entry(&score.arm).or_default().push(score);
    }
    let baseline = scores
        .iter()
        .filter(|row| row.arm == "new_auth")
        .map(|row| (row.id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let arms = by_arm.into_iter().map(|(arm, rows)| {
        let mut pairs: BTreeMap<&str, Vec<&Score>> = BTreeMap::new();
        for row in &rows {
            if let Some(pair) = row.pair.as_deref() {
                pairs.entry(pair).or_default().push(row);
            }
        }
        let complete = pairs.values().filter(|rows| rows.len() == 2).collect::<Vec<_>>();
        let pair_wins = complete.iter().filter(|rows| {
            rows.iter().all(|r| r.exact_reply_with_addresses == Some(true))
        }).count();
        let surface_pair_wins = complete.iter().filter(|rows| {
            rows.iter().all(|r| r.exact_reply_through_eos == Some(true))
        }).count();
        let paired_improvement = complete.iter().filter(|rows| {
            rows.iter().all(|r| r.exact_reply_with_addresses == Some(true))
                && rows.iter().any(|r| baseline[r.id.as_str()].exact_reply_with_addresses != Some(true))
        }).count();
        let pair_prefix_match = complete.iter().filter(|rows| {
            let length = rows[0].length.min(8);
            let left = lookup[&(rows[0].id.clone(), arm.to_owned())];
            let right = lookup[&(rows[1].id.clone(), arm.to_owned())];
            let prefix = |r: &Record| r.client["generated_ids"].as_array()
                .map(|ids| ids.iter().take(length).cloned().collect::<Vec<_>>());
            prefix(left) == prefix(right)
                && rows[0].original_addresses.get(..length).is_some_and(|left| {
                    left.iter().all(Option::is_some)
                        && Some(left) == rows[1].original_addresses.get(..length)
                })
                && prefix(left).is_some_and(|ids| ids.len() == length)
        }).count();
        let by_length = [7usize, 8, 9].into_iter().map(|length| {
            let selected = rows.iter().copied().filter(|row| row.length == length).collect::<Vec<_>>();
            (length.to_string(), counts(&selected))
        }).collect::<BTreeMap<_, _>>();
        let by_request = ["stop", "continue", "unspecified", "denied", "excluded"].into_iter().map(|request| {
            let selected = rows.iter().copied().filter(|row| row.request == request).collect::<Vec<_>>();
            (request, counts(&selected))
        }).collect::<BTreeMap<_, _>>();
        (
            arm, json!({"totals":counts(&rows),"by_length":by_length,"by_request":by_request,
                "complete_pairs":complete.len(),"same_committed_first_prefix_across_pair":pair_prefix_match,
                "both_exact_replies":surface_pair_wins,"both_exact_with_original_addresses":pair_wins,
                "both_exact_with_original_addresses_gained_over_baseline":paired_improvement}),
        )
    }).collect::<BTreeMap<_, _>>();
    let matches = [
        ("historical_auth", "base_auth", true),
        ("base_auth", "new_auth", true),
        ("historical_stock", "base_stock", true),
        ("base_stock", "new_stock", true),
        ("new_auth", "new_auth_quiet", false),
        ("oracle", "oracle_quiet", false),
    ].into_iter().map(|(a, b, full)| {
        let checks = lookup.iter().filter(|((_, arm), _)| arm == b).map(|((id, _), row)| {
            let base = lookup[&(id.clone(), a.into())];
            let equal = if full { base.client == row.client } else { emission_eq(&base.client, &row.client) };
            (id, equal)
        }).collect::<Vec<_>>();
        let different = checks.iter().filter(|(_, same)| !same).map(|(id, _)| id).collect::<Vec<_>>();
        (format!("{a}_vs_{b}"), json!({
            "matching":checks.len() - different.len(), "compared":checks.len(), "different":different,
            "scope":if full { "normalized_whole_client_record" } else { "prompt/generated IDs, text and stopping" }
        }))
    }).collect::<BTreeMap<_, _>>();
    json!({
        "schema":"uor-r4.verified-span-release-panel/1",
        "records":scores.len(),
        "decoded_arms":arms,
        "equivalence":matches,
    })
}

pub fn grade(
    tokenizer: &ByteBpeTokenizer,
    panel: &Path,
    out: &Path,
    records: &[&Path],
) -> Result<()> {
    let (_, keys, jobs) = open_panel(tokenizer, panel)?;
    let protocol = DialogueProtocol::literal_roles_version(tokenizer, 2)?;
    let keys = keys
        .iter()
        .map(|key| (&key.id, key))
        .collect::<BTreeMap<_, _>>();
    let records_in = records
        .iter()
        .map(|path| read_lines::<Record>(path))
        .collect::<Result<Vec<_>>>()?;
    let mut records = records_in
        .into_iter()
        .flatten()
        .map(|r| ((r.id.clone(), r.arm.clone()), r))
        .collect::<Vec<_>>();
    let unique = records
        .iter()
        .map(|(id, _)| id.clone())
        .collect::<BTreeSet<_>>();
    require(
        records.len() == jobs.len() && records.len() == unique.len(),
        || "full capture must cover every frozen job exactly once".into(),
    )?;
    let mut records_by_job = records.drain(..).collect::<BTreeMap<_, _>>();
    let mut ordered = Vec::new();
    for job in jobs {
        ordered.push(
            records_by_job
                .remove(&(job.id, job.arm))
                .ok_or("missing frozen job")?,
        );
    }
    let lookup = ordered
        .iter()
        .map(|r| ((r.id.clone(), r.arm.clone()), r))
        .collect::<BTreeMap<_, _>>();
    let mut scores = Vec::new();
    for record in &ordered {
        let key = keys
            .get(&record.id)
            .ok_or("missing independently derived key")?;
        scores.push(grade_record(
            tokenizer,
            protocol.eos_id,
            key,
            record,
            &lookup,
        )?);
    }
    if !out.exists() {
        fs::create_dir(out)?;
        write_lines(&out.join("records.jsonl"), &ordered)?;
    }
    write_lines(&out.join("scores.jsonl"), &scores)?;
    write_json(&out.join("summary.json"), &summary(&scores, &lookup))
}
