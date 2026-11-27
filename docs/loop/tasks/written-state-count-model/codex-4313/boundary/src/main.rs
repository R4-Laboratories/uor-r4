#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, error::Error, fs, path::Path};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MARKER: [u16; 6] = [201, 35, 560, 652, 714, 28];
const ID_NEWLINE: u16 = 201;
const ID_SPACE: u16 = 223;
const ID_WE: u16 = 799;
const ARMS: [&str; 2] = ["reasoning-first", "answer-first"];
const UORT_HEADER: usize = 64;
const FITTED_PLAN_SHA: &str = "3d5be4978d75afa25e781a50641702e6219f08fdc70424e65ca80b9ea4baa651";

fn check(ok: bool, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(why.into())
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_pinned(root: &Path, receipt: &Value, name: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(root.join(name))?;
    let pin = &receipt["files"][name];
    check(
        pin["sha256"] == digest(&bytes) && pin["bytes"] == bytes.len(),
        &format!("donor pin differs for {name}"),
    )?;
    Ok(bytes)
}

#[derive(Deserialize)]
struct Group {
    id: String,
    raw_record_sha256: String,
    payload_start: usize,
    payload_tokens_both_arms: usize,
    visible_tokens_reasoning_first: usize,
    visible_tokens_answer_first: usize,
    turns: usize,
}

#[derive(Deserialize)]
struct Record<'a> {
    id: &'a str,
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    reasoning: String,
    answer: String,
}

#[derive(Default)]
struct Histogram(BTreeMap<u16, usize>);

impl Histogram {
    fn add(&mut self, id: u16) {
        *self.0.entry(id).or_default() += 1;
    }

    fn summary(&self, tok: &ByteBpeTokenizer) -> Value {
        let mut counts: Vec<_> = self.0.iter().map(|(&id, &n)| (id, n)).collect();
        counts.sort_unstable_by_key(|&(id, n)| (std::cmp::Reverse(n), id));
        let top: Vec<_> = counts
            .iter()
            .take(12)
            .map(|&(id, n)| json!({"id": id, "surface": tok.decode(&[id.into()]), "events": n}))
            .collect();
        let count = |id| self.0.get(&id).copied().unwrap_or_default();
        json!({"events": self.0.values().sum::<usize>(), "distinct_ids": self.0.len(),
            "id_223_space": count(ID_SPACE), "id_201_newline": count(ID_NEWLINE),
            "id_799_space_We": count(ID_WE), "top": top})
    }
}

#[derive(Default)]
struct Opening {
    immediate: Histogram,
    first_nonwhitespace: Histogram,
    standalone_whitespace: BTreeMap<usize, usize>,
    immediate_begins_ascii_space: usize,
    first_nonwhitespace_alphanumeric: usize,
}

impl Opening {
    fn add(&mut self, ids: &[u16], tok: &ByteBpeTokenizer) -> Result<()> {
        let (&first, _) = ids.split_first().ok_or("empty opening")?;
        self.immediate.add(first);
        self.immediate_begins_ascii_space +=
            usize::from(tok.decode_bytes(&[first.into()]).starts_with(b" "));
        let (offset, id) = ids
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, id)| !tok.decode(&[id.into()]).chars().all(char::is_whitespace))
            .ok_or("opening has only whitespace")?;
        self.first_nonwhitespace.add(id);
        *self.standalone_whitespace.entry(offset).or_default() += 1;
        self.first_nonwhitespace_alphanumeric += usize::from(
            tok.decode(&[id.into()])
                .trim_start()
                .starts_with(char::is_alphanumeric),
        );
        Ok(())
    }

    fn summary(&self, tok: &ByteBpeTokenizer) -> Value {
        json!({"immediate": self.immediate.summary(tok),
            "first_nonwhitespace": self.first_nonwhitespace.summary(tok),
            "leading_standalone_whitespace_tokens": self.standalone_whitespace,
            "immediate_begins_ascii_space": self.immediate_begins_ascii_space,
            "first_nonwhitespace_starts_alphanumeric": self.first_nonwhitespace_alphanumeric})
    }
}

fn u32_positions(bytes: &[u8], mask: &[u8], expected: usize) -> Result<()> {
    check(bytes.len() == 4 * expected, "target-map byte count differs")?;
    let mut remaining = mask.to_vec();
    for bytes in bytes.chunks_exact(4) {
        let index = u32::from_le_bytes(bytes.try_into()?) as usize;
        let state = remaining.get_mut(index).ok_or("target outside tape")?;
        check(*state == 1, "target map violates eligibility or uniqueness")?;
        *state = 0;
    }
    check(
        remaining.iter().all(|&m| m == 0),
        "target map skipped eligible positions",
    )
}

fn audit_arm(
    root: &Path,
    receipt: &Value,
    plan: &Value,
    groups: &[Group],
    records: &[&str],
    arm: usize,
    tok: &ByteBpeTokenizer,
) -> Result<Value> {
    let name = ARMS[arm];
    let pins = &plan["arms"][arm];
    check(
        pins["name"] == name && receipt["arms"][arm]["name"] == name,
        "arm order differs",
    )?;
    let read = |suffix: &str| read_pinned(root, receipt, &format!("{name}.{suffix}"));
    let token_bytes = read("tokens.u16")?;
    let mask = read("mask.u8")?;
    let segments = read("segment.u8")?;
    let targets = read("targets.u32")?;
    let geometry_mask = read("geometry-mask.u8")?;
    let geometry_targets = read("geometry-targets.u32")?;
    for (pin, bytes) in [
        (&pins["full"]["tokens"], token_bytes.as_slice()),
        (&pins["full"]["mask"], &mask),
        (&pins["segments"], &segments),
        (&pins["targets_map"], &targets),
        (&pins["geometry"]["train"]["mask"], &geometry_mask),
        (&pins["geometry_order"], &geometry_targets),
    ] {
        check(pin["sha256"] == digest(bytes), "fitted plan pin differs")?;
    }
    let header = token_bytes
        .get(..UORT_HEADER)
        .ok_or("missing UORT header")?;
    check(
        &header[..8] == b"UORT\x01\0\0\0"
            && u64::from_le_bytes(header[8..16].try_into()?) == mask.len() as u64
            && u32::from_le_bytes(header[16..20].try_into()?) == tok.vocab_size() as u32
            && header[20..].iter().all(|&b| b == 0),
        "UORT header differs",
    )?;
    let payload = &token_bytes[UORT_HEADER..];
    let tokens: Vec<_> = payload
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    check(
        payload.len() == 2 * mask.len()
            && tokens.len() == segments.len()
            && mask.len() == geometry_mask.len()
            && tokens
                .iter()
                .all(|&id| u32::from(id) < tok.vocab_size() as u32)
            && mask
                .iter()
                .zip(&geometry_mask)
                .all(|(&m, &g)| g <= m && m <= 1)
            && pins["full"]["expected_payload_tokens"] == tokens.len(),
        "tape inventories differ",
    )?;
    let count = |m: &[u8]| m.iter().filter(|&&n| n == 1).count();
    let full_count = count(&mask);
    let geometry_count = count(&geometry_mask);
    check(
        plan["full_targets"] == full_count && plan["geometry_targets"] == geometry_count,
        "fitted doses differ",
    )?;
    u32_positions(&targets, &mask, full_count)?;
    u32_positions(&geometry_targets, &geometry_mask, geometry_count)?;

    // The original donor encodes the framing, prompt and supervised pieces separately.
    // Validate all visible source turn openings inside the group and the single visible EOS.
    let mut genuine = vec![false; tokens.len()];
    let mut raw = Opening::default();
    let mut served_format = Opening::default();
    let mut first_id_matches_served_format = 0;
    let mut next_group = 0;
    let mut turns = 0;
    let expected_segment = if arm == 0 { 2 } else { 3 };
    for (group, line) in groups.iter().zip(records) {
        let record: Record<'_> = serde_json::from_str(line)?;
        check(
            group.payload_start == next_group
                && group.id == record.id
                && group.raw_record_sha256 == digest(line.as_bytes())
                && group.turns == record.turns.len(),
            "group lineage differs",
        )?;
        let begin = group.payload_start;
        let end = begin
            + if arm == 0 {
                group.visible_tokens_reasoning_first
            } else {
                group.visible_tokens_answer_first
            };
        next_group = begin + group.payload_tokens_both_arms;
        let visible = tokens
            .get(begin..end)
            .ok_or("visible document outside tape")?;
        check(
            visible.first() == Some(&0)
                && visible.last() == Some(&1)
                && visible[1..visible.len() - 1].iter().all(|&id| id > 1)
                && mask[begin] == 0
                && mask[end - 1] == 1
                && tokens[end..next_group].iter().all(|&id| id != 1)
                && tokens[end..next_group]
                    .iter()
                    .filter(|&&id| id == 0)
                    .count()
                    == group.turns,
            "visible EOS or auxiliary BOS framing differs",
        )?;
        let mut starts = (begin + MARKER.len() + 1..end - 1).filter(|&at| {
            segments[at] == expected_segment
                && mask[at] == 1
                && segments[at - MARKER.len() - 1] == 1
                && segments[at - MARKER.len()..at].iter().all(|&tag| tag == 0)
                && mask[at - MARKER.len()..at].iter().all(|&m| m == 0)
                && tokens[at - MARKER.len()..at] == MARKER
        });
        for turn in record.turns {
            check(!turn.reasoning.is_empty(), "source turn lacks reasoning")?;
            let at = starts
                .next()
                .ok_or("source turn lacks a matching six-ID opening")?;
            genuine[at] = true;
            turns += 1;
            let piece_end = at
                + segments[at..end]
                    .iter()
                    .take_while(|&&s| s == expected_segment)
                    .count();
            let raw_ids = &tokens[at..piece_end];
            let rendered = if arm == 0 {
                format!(" {}", turn.reasoning)
            } else {
                turn.answer
            };
            check(
                rendered
                    .as_bytes()
                    .starts_with(&tok.decode_bytes(&[raw_ids[0].into()])),
                "source opening bytes disagree with first tape token",
            )?;
            raw.add(raw_ids, tok)?;
            let served_ids: Vec<_> = tok
                .encode(&format!(" {}", rendered.trim()))
                .into_iter()
                .map(u16::try_from)
                .collect::<std::result::Result<_, _>>()?;
            served_format.add(&served_ids, tok)?;
            first_id_matches_served_format += usize::from(raw_ids.first() == served_ids.first());
        }
        check(
            starts.next().is_none(),
            "unmatched visible role opening in source group",
        )?;
    }
    check(
        next_group == tokens.len() && receipt["turns"] == turns,
        "lineage did not span all groups and visible role openings",
    )?;

    let mut previous_colon: [[Histogram; 4]; 2] = std::array::from_fn(|_| Default::default());
    for at in 1..tokens.len() {
        if mask[at] == 0 || tokens[at - 1] != MARKER[MARKER.len() - 1] {
            continue;
        }
        let exact = at >= MARKER.len() && tokens[at - MARKER.len()..at] == MARKER;
        let category = if genuine[at] {
            1
        } else if exact {
            2
        } else {
            3
        };
        previous_colon[0][0].add(tokens[at]);
        previous_colon[0][category].add(tokens[at]);
        if geometry_mask[at] == 1 {
            previous_colon[1][0].add(tokens[at]);
            previous_colon[1][category].add(tokens[at]);
        }
    }
    let summary = |part: &[Histogram; 4]| {
        json!({
            "all_previous_id_28": part[0].summary(tok),
            "exact_suffix_at_genuine_assistant_opening": part[1].summary(tok),
            "exact_suffix_elsewhere": part[2].summary(tok),
            "other_previous_id_28": part[3].summary(tok),
        })
    };
    Ok(
        json!({"name":name, "payload_tokens":tokens.len(), "eligible_count_fit_events":full_count,
            "eligible_geometry_events":geometry_count, "visible_documents":groups.len(),
            "genuine_assistant_openings":turns, "count_fit": summary(&previous_colon[0]),
            "geometry_subset": summary(&previous_colon[1]), "raw_donor_opening": raw.summary(tok),
            "same_text_protocol_2_format_opening": served_format.summary(tok),
            "raw_immediate_id_equals_protocol_2_immediate_id": first_id_matches_served_format,
        }),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [root, plan_path, tok_path] = args.as_slice() else {
        return Err("usage: question-swap-boundary-audit DONOR_DIR FITTED_PLAN TOKENIZER".into());
    };
    let root = Path::new(root);
    let receipt_bytes = fs::read(root.join("receipt.json"))?;
    let plan_bytes = fs::read(plan_path)?;
    let tok_bytes = fs::read(tok_path)?;
    let receipt: Value = serde_json::from_slice(&receipt_bytes)?;
    let plan: Value = serde_json::from_slice(&plan_bytes)?;
    check(
        digest(&plan_bytes) == FITTED_PLAN_SHA,
        "registered fitted plan pin differs",
    )?;
    check(
        plan["donor_manifest"]["sha256"] == digest(&receipt_bytes),
        "fitted plan donor receipt pin differs",
    )?;
    check(
        receipt["input"]["p4_tokenizer_sha256"] == digest(&tok_bytes),
        "tokenizer pin differs",
    )?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes).ok_or("invalid tokenizer")?;
    check(
        tok.encode("\nAssistant:") == MARKER.map(u32::from)
            && [tok.encode("\n"), tok.encode("Assistant:")].concat() == MARKER.map(u32::from),
        "original donor and served suffix IDs differ",
    )?;
    let group_bytes = read_pinned(root, &receipt, "groups.jsonl")?;
    let record_bytes = read_pinned(root, &receipt, "records.jsonl")?;
    let groups: Vec<Group> = std::str::from_utf8(&group_bytes)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let records: Vec<_> = std::str::from_utf8(&record_bytes)?.lines().collect();
    check(
        plan["groups"]["sha256"] == digest(&group_bytes)
            && plan["group_count"] == groups.len()
            && groups.len() == records.len(),
        "group inventories differ",
    )?;
    let arms = [
        audit_arm(root, &receipt, &plan, &groups, &records, 0, &tok)?,
        audit_arm(root, &receipt, &plan, &groups, &records, 1, &tok)?,
    ];
    let summary = json!({"schema": "uor-r4.question-swap-training-boundary/1",
        "scope": "Supplementary post-run static audit of fitted #4312 donor inputs only. No held rows, artifact execution, evaluation, or training.",
        "assistant_suffix": MARKER, "fitted_plan_sha256": digest(&plan_bytes),
        "donor_receipt_sha256": digest(&receipt_bytes), "tokenizer_sha256": digest(&tok_bytes),
        "donor_input_pins_verified": receipt["files"], "arms": arms});
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
