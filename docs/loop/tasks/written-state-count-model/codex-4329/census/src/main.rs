#![forbid(unsafe_code)]

mod classify;
mod inputs;

use classify::{Target, TurnContext};
use inputs::{Group, Opening, Record, SourceGroup};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
    ByteBpeTokenizer,
};

type Audit<T> = Result<T, Box<dyn Error>>;
type Counts = BTreeMap<&'static str, usize>;
const CONTEXT: usize = 192;
const VOCAB: usize = 4096;
const SPACE: u32 = 223;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];

const CLASSES_SHA: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";

const DONOR_RECEIPT: &str = "61ba19783fd29b39ee935e1dfa21ae14c5de1caddd3d380c7ed1bedb6b39f6e3";
const RAW_RECEIPT: &str = "4ff029e7962b6c58c88014f1baf6adeb4bd5fd1e3d0464cdbbf66d929d10bbd8";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

fn require(ok: bool, error: &str) -> Audit<()> {
    if !ok {
        return Err(error.into());
    }
    Ok(())
}

fn pin(receipt: &Value, name: &str) -> Audit<String> {
    let sha = receipt["files"][name]["sha256"]
        .as_str()
        .ok_or_else(|| format!("missing frozen hash for {name}"))?;
    Ok(sha.to_owned())
}

fn protocol_content(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    format!(" {}", normalized.trim())
}

#[derive(Serialize)]
struct Row<'a> {
    group: usize,
    turn: usize,
    train_ordinal: usize,
    source_id: &'a str,
    source_dataset: &'a str,
    source_kind: &'a str,
    source_record_sha256: &'a str,
    prompt: &'a str,
    answer: &'a str,
    prompt_offsets: [usize; 2],
    current_user_offsets: [usize; 2],
    answer_offsets: [usize; 2],
    historical_window_start: usize,
    initial_context_len: usize,
    fresh_current_user_context_len: usize,
    immediate: Target,
    first_nonwhitespace: Target,
}

#[derive(Default, Serialize)]
struct TargetCounts {
    full_current_user: Counts,
    ordinary_window: Counts,
    framed_window: Counts,
    prior_only_with_lexical_target: usize,
    prior_user_content_when_current_user_absent: usize,
    prior_assistant_content_when_current_user_absent: usize,
    window_unique_loses_full_turn_ambiguity: usize,
    full_unique_excluded_from_window: usize,
    full_unique_ids: BTreeSet<u32>,
    window_unique_ids: BTreeSet<u32>,
    framed_unique_ids: BTreeSet<u32>,
    full_unique_word_matches: usize,
    window_unique_word_matches: usize,
    framed_unique_word_matches: usize,
    full_unique_word_match_ids: BTreeSet<u32>,
    window_unique_word_match_ids: BTreeSet<u32>,
    framed_unique_word_match_ids: BTreeSet<u32>,
    unshifted_context_fits: usize,
    retained_after_decode_window_slid: usize,
}

impl TargetCounts {
    fn add(&mut self, target: &Target) {
        *self.full_current_user.entry(target.raw_kind).or_default() += 1;
        *self.ordinary_window.entry(target.window_kind).or_default() += 1;
        *self.framed_window.entry(target.framed_kind).or_default() += 1;
        self.prior_only_with_lexical_target += usize::from(
            target.raw_kind == "absent" && !target.prior_history_exact_offsets.is_empty(),
        );
        self.prior_user_content_when_current_user_absent += usize::from(
            target.raw_kind == "absent" && !target.prior_user_content_exact_offsets.is_empty(),
        );
        self.prior_assistant_content_when_current_user_absent += usize::from(
            target.raw_kind == "absent" && !target.prior_assistant_content_exact_offsets.is_empty(),
        );
        self.window_unique_loses_full_turn_ambiguity +=
            usize::from(target.window_kind == "unique" && target.raw_kind == "duplicate");
        self.full_unique_excluded_from_window +=
            usize::from(target.raw_kind == "unique" && target.window_kind != "unique");
        self.unshifted_context_fits += usize::from(target.unshifted_context_len <= CONTEXT);
        self.retained_after_decode_window_slid += usize::from(
            target.unshifted_context_len > CONTEXT && target.phase_window_start.is_some(),
        );
        if target.raw_kind == "unique" {
            self.full_unique_ids.insert(target.id);
            if target.target_standalone_word && target.raw_exact[0].standalone_word {
                self.full_unique_word_matches += 1;
                self.full_unique_word_match_ids.insert(target.id);
            }
        }
        if target.window_kind == "unique" {
            self.window_unique_ids.insert(target.id);
            if target.target_standalone_word && target.window_exact[0].standalone_word {
                self.window_unique_word_matches += 1;
                self.window_unique_word_match_ids.insert(target.id);
            }
        }
        if target.framed_kind == "unique" {
            self.framed_unique_ids.insert(target.id);
            if target.target_standalone_word && target.framed_exact[0].standalone_word {
                self.framed_unique_word_matches += 1;
                self.framed_unique_word_match_ids.insert(target.id);
            }
        }
    }
}

#[derive(Default, Serialize)]
struct Census {
    groups: usize,
    group_ids_verified: usize,
    exact_raw_prefixes: usize,
    exact_raw_user_segments: usize,
    exact_full_answer_spans: usize,
    exact_historical_windows: usize,
    exact_visible_tokens_and_masks: usize,
    context_fit: Counts,
    turns_by_source: BTreeMap<String, usize>,
    first_nonwhitespace_is_after_space: usize,
    current_user_contains_literal_role_marker_ids: usize,
    serveable_current_user_contains_literal_role_marker_ids: usize,
    current_user_contains_textual_role_marker_line: usize,
    later_turn_user_solo_fits_without_history: usize,
    immediate: TargetCounts,
    first_nonwhitespace: TargetCounts,
    first_nonwhitespace_after_separate_space: TargetCounts,
}

struct Frozen<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    user_marker: Vec<u32>,
    separator: Vec<u32>,
    token_bytes: Vec<Vec<u8>>,
    class_table: Vec<u8>,
    tape: Vec<u32>,
    mask: Vec<u8>,
}

struct GroupAudit<'a> {
    frozen: &'a Frozen<'a>,
    source: &'a SourceGroup,
    record: &'a Record,
    group: Group,
    history: Vec<Message<'a>>,
    prior_user: Vec<Range<usize>>,
    prior_assistant: Vec<Range<usize>>,
}

impl<'a> GroupAudit<'a> {
    fn turn(&mut self, turn: usize, opening: Opening, colon: &inputs::Colon) -> Audit<Row<'a>> {
        let frozen = self.frozen;
        let user = &self.record.messages[turn];
        let reply = &self.record.turns[turn];
        let raw_user = protocol_content(&user.content);
        let user_ids = frozen.tokenizer.encode(&raw_user);
        let preceding = frozen.encoder.encode_open_history(&self.history);
        let appended = frozen.encoder.encode_user_prefix(&user.content, turn != 0);
        let fresh_prefix = frozen.encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &user.content,
        }]);
        let prefix_start = self.group.payload_start;
        let offset = prefix_start + preceding.tokens.len();
        let user_start =
            offset + frozen.user_marker.len() + if turn == 0 { 0 } else { frozen.separator.len() };
        let user_end = user_start + user_ids.len();
        let user_range = user_start..user_end;
        let reply_offset = offset + appended.tokens.len();
        let fail = format!("at group {} turn {turn}", self.group.group);
        require(
            user.role == "user"
                && appended.emitted_turns == 1
                && appended.special_token_occurrences == 0
                && appended.tokens.ends_with(&MARKER)
                && user_end + MARKER.len() == reply_offset
                && fresh_prefix.tokens.len()
                    == 1 + frozen.user_marker.len() + user_ids.len() + MARKER.len()
                && fresh_prefix.tokens
                    [1 + frozen.user_marker.len()..fresh_prefix.tokens.len() - MARKER.len()]
                    == user_ids
                && frozen.tokenizer.decode_bytes(&user_ids) == raw_user.as_bytes()
                && frozen.tape.get(user_range.clone()) == Some(user_ids.as_slice()),
            &format!("raw current-user protocol span differs {fail}"),
        )?;
        let served = [preceding.tokens, appended.tokens].concat();
        require(
            frozen.tape.get(prefix_start..reply_offset) == Some(served.as_slice())
                && colon.prefix == served[served.len().saturating_sub(CONTEXT)..]
                && colon.target == opening.immediate_id,
            &format!("ordinary serving prefix or historical window differs {fail}"),
        )?;

        let raw_answer = protocol_content(&reply.answer);
        let answer_ids = frozen.tokenizer.encode(&raw_answer);
        let answer_end = reply_offset + answer_ids.len();
        let answer_bounds = classify::byte_bounds(&answer_ids, &frozen.token_bytes);
        let first_nonwhite_byte = raw_answer
            .bytes()
            .position(|byte| !byte.is_ascii_whitespace())
            .ok_or("empty post-normalization answer")?;
        let first_nonwhite_index = answer_bounds
            .windows(2)
            .position(|bounds| bounds[0] <= first_nonwhite_byte && first_nonwhite_byte < bounds[1])
            .ok_or("first non-whitespace byte has no answer ID")?;
        let first_nonwhite_offset = reply_offset + first_nonwhite_index;
        require(
            opening.group == self.group.group
                && opening.turn == turn
                && opening.prefix_start == prefix_start
                && opening.answer_start == reply_offset
                && opening.answer_end == answer_end
                && answer_ids.first() == Some(&opening.immediate_id)
                && first_nonwhite_offset == opening.lexical_offset
                && first_nonwhite_index == opening.lexical_answer_offset
                && answer_ids.get(first_nonwhite_index) == Some(&opening.lexical_id)
                && (first_nonwhite_index == 0
                    || (first_nonwhite_index == 1 && opening.immediate_id == SPACE))
                && frozen.tokenizer.decode_bytes(&answer_ids) == raw_answer.as_bytes()
                && frozen.tape.get(reply_offset..answer_end) == Some(answer_ids.as_slice())
                && frozen.tape.get(answer_end) == Some(&1)
                && frozen.mask[reply_offset..answer_end]
                    .iter()
                    .all(|&v| v == 1),
            &format!("raw answer IDs, offsets, terminal EOS or mask differ {fail}"),
        )?;

        self.history.extend([
            Message {
                role: "user",
                content: &user.content,
            },
            Message {
                role: "assistant",
                content: &reply.answer,
            },
        ]);
        let user_bounds = classify::byte_bounds(&user_ids, &frozen.token_bytes);
        let turn_context = TurnContext {
            tape: &frozen.tape,
            classes: &frozen.class_table,
            token_bytes: &frozen.token_bytes,
            prefix_start,
            answer_start: reply_offset,
            user: user_range.clone(),
            prior_end: offset,
            prior_user: &self.prior_user,
            prior_assistant: &self.prior_assistant,
            user_bytes: &user_bounds,
            answer_bytes: &answer_bounds,
            user_content: &raw_user,
            answer_content: &raw_answer,
        };
        let row = Row {
            group: self.group.group,
            turn,
            train_ordinal: self.group.canonical_train_group,
            source_id: &self.record.id,
            source_dataset: &self.record.source,
            source_kind: &self.record.kind,
            source_record_sha256: &self.source.raw_record_sha256,
            prompt: &user.content,
            answer: &reply.answer,
            prompt_offsets: [prefix_start, reply_offset],
            current_user_offsets: [user_start, user_end],
            answer_offsets: [reply_offset, answer_end],
            historical_window_start: reply_offset - colon.prefix.len(),
            initial_context_len: served.len(),
            fresh_current_user_context_len: fresh_prefix.tokens.len(),
            immediate: turn_context.target(reply_offset),
            first_nonwhitespace: turn_context.target(first_nonwhite_offset),
        };
        self.prior_user.push(user_range);
        self.prior_assistant.push(reply_offset..answer_end);
        Ok(row)
    }

    fn verify_visible(&self) -> Audit<()> {
        let encoded = self.frozen.encoder.encode_open_history(&self.history);
        let visible = self.group.payload_start..self.group.visible_end;
        let mut mask = encoded.response_mask;
        // The corrected tape leaves only the final assistant EOS supervised.
        let last = mask.len() - 1;
        for (index, (&id, value)) in encoded.tokens.iter().zip(&mut mask).enumerate() {
            if id == 1 && index != last {
                *value = 0;
            }
        }
        require(
            self.frozen.tape.get(visible.clone()) == Some(encoded.tokens.as_slice())
                && self.frozen.mask.get(visible) == Some(mask.as_slice()),
            &format!(
                "visible protocol IDs or mask differ at group {}",
                self.group.group
            ),
        )
    }
}

fn count(census: &mut Census, row: &Row<'_>, tape: &[u32]) {
    census.exact_raw_prefixes += 1;
    census.exact_raw_user_segments += 1;
    census.exact_full_answer_spans += 1;
    census.exact_historical_windows += 1;
    *census
        .turns_by_source
        .entry(row.source_dataset.to_owned())
        .or_default() += 1;
    let reach = match (row.turn == 0, row.initial_context_len <= CONTEXT) {
        (true, true) => "first_turn_within_192",
        (false, true) => "later_turn_within_192",
        (true, false) => "first_turn_over_192",
        (false, false) => "later_turn_over_192",
    };
    *census.context_fit.entry(reach).or_default() += 1;
    let user_ids = &tape[row.current_user_offsets[0]..row.current_user_offsets[1]];
    let contains_marker_ids = user_ids.windows(MARKER.len()).any(|ids| ids == MARKER)
        || user_ids.windows(4).any(|ids| ids == [201, 55, 2728, 28]);
    census.current_user_contains_literal_role_marker_ids += usize::from(contains_marker_ids);
    census.serveable_current_user_contains_literal_role_marker_ids +=
        usize::from(contains_marker_ids && row.initial_context_len <= CONTEXT);
    census.current_user_contains_textual_role_marker_line +=
        usize::from(row.prompt.lines().any(|line| {
            ["User:", "Assistant:", "System:"]
                .iter()
                .any(|marker| line.trim_start().starts_with(marker))
        }));
    census.later_turn_user_solo_fits_without_history +=
        usize::from(row.turn != 0 && row.fresh_current_user_context_len <= CONTEXT);
    census.immediate.add(&row.immediate);
    census.first_nonwhitespace.add(&row.first_nonwhitespace);
    if row.first_nonwhitespace.target_offset > row.immediate.target_offset {
        census.first_nonwhitespace_is_after_space += 1;
        census
            .first_nonwhitespace_after_separate_space
            .add(&row.first_nonwhitespace);
    }
}

fn main() -> Audit<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [donor, raw, tokenizer, classes, historical, output] = args.as_slice() else {
        return Err("usage: audit-whole-turn DONOR_DIR RAW_DIR TOKENIZER CLASSES HISTORICAL_COMMITTED_RECEIPT NEW_DIR".into());
    };
    let donor = Path::new(donor);
    let raw = Path::new(raw);
    let output = Path::new(output);
    let donor_receipt: Value = serde_json::from_slice(&inputs::checked_data(
        &donor.join("receipt.json"),
        DONOR_RECEIPT,
    )?)?;
    let raw_receipt: Value = serde_json::from_slice(&inputs::checked_data(
        &raw.join("receipt.json"),
        RAW_RECEIPT,
    )?)?;
    let mut input_hashes = BTreeMap::new();
    for name in [
        "answer-first.tokens.u16",
        "answer-first.mask.u8",
        "colon-targets.jsonl",
        "groups.jsonl",
        "openings.jsonl",
        "source.json",
    ] {
        let sha = pin(&donor_receipt, name)?;
        inputs::check(&donor.join(name), &sha)?;
        input_hashes.insert(format!("corrected/{name}"), sha);
    }
    for name in ["groups.jsonl", "records.jsonl"] {
        let sha = pin(&raw_receipt, name)?;
        inputs::check(&raw.join(name), &sha)?;
        input_hashes.insert(format!("raw/{name}"), sha);
    }
    let tokenizer_bytes = inputs::checked_data(Path::new(tokenizer), TOKENIZER)?;
    let classes = classify::canonical_classes(Path::new(classes))?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tokenizer_bytes)
        .ok_or("invalid original tokenizer")?;
    require(tok.vocab_size() == VOCAB, "tokenizer vocabulary differs")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tok)?;
    let source: Value = serde_json::from_slice(&fs::read(donor.join("source.json"))?)?;
    require(
        source["protocol"] == serde_json::to_value(&protocol)?
            && source["protocol_identity"] == protocol.identity()?
            && source["historical_records_sha256"] == input_hashes["raw/records.jsonl"]
            && source["historical_groups_sha256"] == input_hashes["raw/groups.jsonl"]
            && source["historical_receipt_sha256"] == RAW_RECEIPT,
        "corrected tape provenance or protocol mismatch",
    )?;
    let frozen = Frozen {
        tokenizer: &tok,
        encoder: protocol.bind(&tok)?,
        user_marker: tok.encode("User:"),
        separator: tok.encode("\n"),
        token_bytes: (0..VOCAB)
            .map(|id| tok.decode_bytes(&[id as u32]))
            .collect(),
        class_table: classes,
        tape: inputs::tape(&fs::read(donor.join("answer-first.tokens.u16"))?)?,
        mask: fs::read(donor.join("answer-first.mask.u8"))?,
    };
    require(
        frozen.mask.len() == frozen.tape.len() && frozen.mask.iter().all(|&v| v <= 1),
        "invalid corrected tape target mask",
    )?;
    require(
        protocol.bos_id == 0
            && protocol.eos_id == 1
            && frozen.user_marker == [55, 2728, 28]
            && frozen.separator == [201]
            && [frozen.separator.clone(), tok.encode("Assistant:")].concat() == MARKER
            && tok.encode(" ") == [SPACE],
        "marker or delimiter IDs differ",
    )?;
    let (legacy, mut colon) = classify::historical(
        &donor.join("colon-targets.jsonl"),
        &frozen.class_table,
        &frozen.tape,
        &frozen.mask,
    )?;
    let historical_bytes = fs::read(historical)?;
    let expected: Value = serde_json::from_slice(&historical_bytes)?;
    require(
        legacy == expected,
        "historical #4315 census differs from committed receipt",
    )?;

    let mut groups = inputs::rows::<Group>(&donor.join("groups.jsonl"))?;
    let mut source_groups = inputs::rows::<SourceGroup>(&raw.join("groups.jsonl"))?;
    let mut openings = inputs::rows::<Opening>(&donor.join("openings.jsonl"))?;
    fs::create_dir(output)?;
    let mut rows = BufWriter::new(File::create_new(output.join("rows.jsonl"))?);
    let mut census = Census::default();
    let mut next_start = 0;
    for line in inputs::lines(&raw.join("records.jsonl"))? {
        let line = line?;
        let record: Record = serde_json::from_str(&line)?;
        let source_group = inputs::next(&mut source_groups)?;
        let group = inputs::next(&mut groups)?;
        require(
            source_group.raw_record_sha256 == inputs::digest(line.as_bytes())
                && source_group.id == record.id
                && source_group.group == census.groups
                && group.group == census.groups
                && group.canonical_train_group == source_group.canonical_train_group
                && group.canonical_train_group == census.groups * 128
                && group.payload_start == next_start
                && group.payload_start < group.visible_end
                && group.visible_end <= group.payload_end
                && group.payload_end <= frozen.tape.len()
                && record.messages.len() == record.turns.len()
                && source_group.turns == record.turns.len()
                && (1..=2).contains(&source_group.turns),
            &format!(
                "frozen raw group identity or alignment differs at group {}",
                census.groups
            ),
        )?;
        next_start = group.payload_end;
        let mut audit = GroupAudit {
            frozen: &frozen,
            source: &source_group,
            record: &record,
            group,
            history: Vec::new(),
            prior_user: Vec::new(),
            prior_assistant: Vec::new(),
        };
        for turn in 0..record.turns.len() {
            let opening = inputs::next(&mut openings)?;
            let saved = colon
                .remove(&opening.answer_start)
                .ok_or("opening absent in colon tape")?;
            let row = audit.turn(turn, opening, &saved)?;
            count(&mut census, &row, &frozen.tape);
            serde_json::to_writer(&mut rows, &row)?;
            writeln!(rows)?;
        }
        audit.verify_visible()?;
        census.exact_visible_tokens_and_masks +=
            audit.group.visible_end - audit.group.payload_start;
        census.groups += 1;
        census.group_ids_verified += 1;
    }
    require(
        groups.next().is_none()
            && source_groups.next().is_none()
            && openings.next().is_none()
            && colon.is_empty()
            && next_start == frozen.tape.len()
            && census.groups == 4060
            && census.exact_raw_prefixes == 4345,
        "frozen group/opening coverage differs",
    )?;
    rows.flush()?;
    input_hashes.extend([
        (
            "corrected/receipt.json".to_owned(),
            DONOR_RECEIPT.to_owned(),
        ),
        ("raw/receipt.json".to_owned(), RAW_RECEIPT.to_owned()),
        ("tokenizer.json".to_owned(), TOKENIZER.to_owned()),
        ("classes.u8".to_owned(), CLASSES_SHA.to_owned()),
        (
            "historical-audit.json".to_owned(),
            inputs::digest(&historical_bytes),
        ),
    ]);
    let summary = json!({
        "schema": "uor-r4.whole-current-user-original-id-census/1",
        "context_limit": CONTEXT,
        "protocol": protocol,
        "protocol_identity": protocol.identity()?,
        "historical_4315": legacy,
        "census": census,
        "input_sha256": input_hashes,
        "private_rows_sha256": inputs::digest(&fs::read(output.join("rows.jsonl"))?),
    });
    let mut report = BufWriter::new(File::create_new(output.join("summary.json"))?);
    serde_json::to_writer_pretty(&mut report, &summary)?;
    writeln!(report)?;
    report.flush()?;
    Ok(())
}
