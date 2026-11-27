use super::*;
use serde_json::json;
use source::file_sha;
use std::collections::BTreeMap;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol};

#[derive(Default, Serialize)]
struct Census {
    groups: usize,
    groups_with_two_turns: usize,
    full_product_prefixes_verified: usize,
    genuine_device_prefixes_verified: usize,
    full_answer_spans_verified: usize,
    answer_ids_verified: usize,
    previous_closures_mask_zero: usize,
    terminal_eos_supervised: usize,
    auxiliary_documents_verified: usize,
    auxiliary_tokens_verified: usize,
    payload_tokens: usize,
    visible_tokens: usize,
    supervised_targets: usize,
    immediate_ids: BTreeMap<u32, usize>,
    first_nonwhitespace_ids: BTreeMap<u32, usize>,
    first_nonwhitespace_offsets: BTreeMap<usize, usize>,
    colon_rows: usize,
    colon_genuine_assistant: usize,
    colon_visible_other: usize,
    colon_auxiliary: usize,
    colon_full_marker_outside_genuine_opening: usize,
    colon_space_id: usize,
    colon_genuine_space_id: usize,
    colon_mask_zero_successors: usize,
    colon_rows_window_clipped: usize,
    colon_rows_with_bos: usize,
}

struct ExpectedOpening {
    opening: Opening,
    device_prefix: Vec<u32>,
}

fn visible_oracle(
    encoder: &DialogueEncoder<'_>,
    tok: &ByteBpeTokenizer,
    record: &Record,
    group: &Group,
    ids: &[u32],
    mask: &[u8],
    census: &mut Census,
) -> CorpusResult<Vec<ExpectedOpening>> {
    let visible = ids
        .get(group.payload_start..group.visible_end)
        .ok_or("bad visible tape range")?;
    let visible_mask = mask
        .get(group.payload_start..group.visible_end)
        .ok_or("bad visible mask range")?;
    let mut openings = Vec::new();
    let mut history = Vec::new();
    let mut expected_ids = vec![BOS];
    let mut expected_mask = vec![0];
    for (turn, (user, reply)) in record.messages.iter().zip(&record.turns).enumerate() {
        // Mirrors DialogueConversation's fresh history and appended request paths.
        let history_ids = encoder.encode_open_history(&history);
        let served_user = encoder.encode_user_prefix(&user.content, turn > 0);
        require(
            history_ids.tokens == expected_ids
                && served_user.emitted_turns == 1
                && served_user.special_token_occurrences == 0
                && served_user.tokens.ends_with(&MARKER)
                && served_user.response_mask.iter().all(|&entry| entry == 0),
            format!(
                "serving history or request differs in group {} turn {turn}",
                group.group
            ),
        )?;
        expected_mask.extend(served_user.response_mask);
        expected_ids.extend(served_user.tokens);
        require(
            visible.starts_with(&expected_ids) && visible_mask.starts_with(&expected_mask),
            format!(
                "full persisted serving prefix or mask differs in group {} turn {turn}",
                group.group
            ),
        )?;
        let start = expected_ids.len();

        // Encode the original answer directly under the protocol's documented
        // normalization, without slicing the builder's complete dialogue encoding.
        let normalized = reply.answer.replace("\r\n", "\n").replace('\r', "\n");
        let spaced = format!(" {}", normalized.trim());
        let answer = tok.encode(&spaced);
        let decoded = tok.decode_bytes(&answer);
        require(
            decoded == spaced.as_bytes(),
            format!(
                "answer text does not round-trip in group {} turn {turn}",
                group.group
            ),
        )?;
        let &immediate_id = answer.first().ok_or("missing immediate answer ID")?;
        let first_nonwhite_byte = decoded
            .iter()
            .position(|byte| !byte.is_ascii_whitespace())
            .ok_or("answer has no first non-whitespace byte")?;
        let mut byte_offset = 0;
        let lexical = answer
            .iter()
            .enumerate()
            .find(|(_, id)| {
                byte_offset += tok.decode_bytes(&[**id]).len();
                byte_offset > first_nonwhite_byte
            })
            .ok_or("lexical answer ID is missing")?;
        let end = start + answer.len();
        require(
            visible.get(start..end) == Some(answer.as_slice())
                && visible_mask
                    .get(start..end)
                    .is_some_and(|part| part.iter().all(|&entry| entry == 1))
                && visible.get(end) == Some(&EOS),
            format!(
                "persisted answer IDs or mask differ in group {} turn {turn}",
                group.group
            ),
        )?;
        let terminal = turn + 1 == record.turns.len();
        let expected_eos_mask = u8::from(terminal);
        require(
            visible_mask.get(end) == Some(&expected_eos_mask),
            "visible EOS target policy differs",
        )?;
        openings.push(ExpectedOpening {
            device_prefix: expected_ids[start.saturating_sub(WINDOW)..].to_vec(),
            opening: Opening {
                group: group.group,
                turn,
                prefix_start: group.payload_start,
                answer_start: group.payload_start + start,
                answer_end: group.payload_start + end,
                immediate_id,
                lexical_id: *lexical.1,
                lexical_offset: group.payload_start + start + lexical.0,
                lexical_answer_offset: lexical.0,
            },
        });
        census.full_product_prefixes_verified += 1;
        census.full_answer_spans_verified += 1;
        census.answer_ids_verified += answer.len();
        census.terminal_eos_supervised += usize::from(terminal);
        census.previous_closures_mask_zero += usize::from(!terminal);
        *census.immediate_ids.entry(immediate_id).or_default() += 1;
        *census
            .first_nonwhitespace_ids
            .entry(*lexical.1)
            .or_default() += 1;
        *census
            .first_nonwhitespace_offsets
            .entry(lexical.0)
            .or_default() += 1;

        expected_mask.resize(expected_mask.len() + answer.len(), 1);
        expected_mask.push(expected_eos_mask);
        expected_ids.extend(answer);
        expected_ids.push(EOS);
        history.extend([
            Message {
                role: "user",
                content: &user.content,
            },
            Message {
                role: "assistant",
                content: &reply.answer,
            },
        ]);
    }
    require(
        visible == expected_ids && visible_mask == expected_mask,
        format!(
            "visible document partition differs for group {}",
            group.group
        ),
    )?;
    census.visible_tokens += visible.len();
    Ok(openings)
}

pub(super) fn run(tok_path: &Path, prior: &Path, out: &Path) -> CorpusResult<()> {
    let tok = load_tokenizer(tok_path)?;
    let protocol = DialogueProtocol::literal_roles_v2(&tok)?;
    let encoder = protocol.bind(&tok)?;
    let source: Value = serde_json::from_slice(&fs::read(out.join("source.json"))?)?;
    let old_receipt = prior_receipt(prior)?;
    require(
        source["protocol"] == serde_json::to_value(&protocol)?
            && source["protocol_identity"] == protocol.identity()?
            && source["input"] == old_receipt["input"]
            && source["historical_receipt_sha256"] == OLD_RECEIPT
            && source["historical_records_sha256"]
                == old_receipt["files"]["records.jsonl"]["sha256"]
            && source["historical_groups_sha256"] == old_receipt["files"]["groups.jsonl"]["sha256"]
            && source["structured_records_match_historical_extraction_byte_identical"] == true
            && source["historical_raw_r_reconstruction_byte_identical"] == true,
        "source binding differs from protocol or historical donor inputs",
    )?;
    let records = old_file(prior, &old_receipt, "records.jsonl")?;
    let prior_groups = old_file(prior, &old_receipt, "groups.jsonl")?;
    let old_tape = read_tape(&old_file(prior, &old_receipt, TAPE_NAME)?)?;
    let old_mask = old_file(prior, &old_receipt, MASK_NAME)?;
    let tape = read_tape(&fs::read(out.join(TAPE_NAME))?)?;
    let mask = fs::read(out.join(MASK_NAME))?;
    require(
        tape.len() == mask.len()
            && old_tape.len() == old_mask.len()
            && mask.iter().all(|&entry| entry <= 1),
        "expected a binary mask for each persisted token",
    )?;
    let mut census = Census {
        payload_tokens: tape.len(),
        supervised_targets: mask.iter().map(|&entry| usize::from(entry)).sum(),
        ..Census::default()
    };
    let mut prior_groups = std::str::from_utf8(&prior_groups)?.lines();
    let mut groups = rows::<Group>(&out.join(GROUPS_NAME))?;
    let mut opening_rows = rows::<Opening>(&out.join(OPENINGS_NAME))?;
    let mut genuine = BTreeMap::new();
    let mut auxiliary_offsets = BTreeMap::new();
    let mut next_group_start = 0;
    let mut old_next_group_start = 0;
    for line in std::str::from_utf8(&records)?.lines() {
        let reference: PriorGroup =
            serde_json::from_str(prior_groups.next().ok_or("missing historical group")?)?;
        let group = groups.next().ok_or("missing rebuilt group")??;
        let record: Record = serde_json::from_str(line)?;
        require(
            reference.raw_record_sha256 == sha(line.as_bytes())
                && reference.id == record.id
                && reference.group == census.groups
                && reference.canonical_train_group == census.groups * 128
                && reference.turns == record.messages.len()
                && reference.turns == record.turns.len()
                && (1..=2).contains(&reference.turns)
                && record.messages.iter().all(|message| message.role == "user")
                && reference.payload_start == old_next_group_start
                && group.group == reference.group
                && group.canonical_train_group == reference.canonical_train_group
                && group.payload_start == next_group_start
                && group.payload_start < group.visible_end
                && group.visible_end <= group.payload_end
                && group.payload_end <= tape.len(),
            format!("bad persisted group inventory at group {}", census.groups),
        )?;
        for ExpectedOpening {
            opening,
            device_prefix,
        } in visible_oracle(&encoder, &tok, &record, &group, &tape, &mask, &mut census)?
        {
            let saved = opening_rows.next().ok_or("missing persisted opening")??;
            require(
                saved == opening
                    && genuine
                        .insert(opening.answer_start, device_prefix)
                        .is_none(),
                format!(
                    "wrong opening IDs or offsets in group {} turn {}",
                    group.group, opening.turn
                ),
            )?;
        }
        old_next_group_start += reference.payload_tokens_both_arms;
        let old_aux_range =
            reference.payload_start + reference.visible_tokens_answer_first..old_next_group_start;
        let old_aux = old_tape
            .get(old_aux_range.clone())
            .ok_or("bad historical auxiliary range")?;
        let new_aux = &tape[group.visible_end..group.payload_end];
        let aux_mask = &mask[group.visible_end..group.payload_end];
        require(
            old_aux == new_aux
                && old_mask.get(old_aux_range) == Some(aux_mask)
                && (new_aux.is_empty() || new_aux.first() == Some(&BOS))
                && new_aux.iter().filter(|&&id| id == BOS).count() == reference.reasoning_turns
                && new_aux
                    .iter()
                    .zip(aux_mask)
                    .all(|(&id, &target)| id != EOS && target == u8::from(id != BOS)),
            format!(
                "detached reasoning payload, BOS or masks differ for group {}",
                group.group
            ),
        )?;
        auxiliary_offsets.insert(group.visible_end, group.payload_end);
        census.auxiliary_documents_verified += reference.reasoning_turns;
        census.auxiliary_tokens_verified += new_aux.len();
        census.groups += 1;
        census.groups_with_two_turns += usize::from(reference.turns == 2);
        next_group_start = group.payload_end;
    }
    require(
        census.groups == GROUPS
            && census.full_product_prefixes_verified == TURNS
            && census.full_answer_spans_verified == TURNS
            && census.auxiliary_documents_verified == TURNS
            && census.previous_closures_mask_zero == TURNS - GROUPS
            && census.terminal_eos_supervised == GROUPS
            && genuine.len() == TURNS
            && next_group_start == tape.len()
            && old_next_group_start == old_tape.len()
            && prior_groups.next().is_none()
            && groups.next().is_none()
            && opening_rows.next().is_none(),
        "persisted or historical group/turn coverage differs",
    )?;

    let mut colon_rows = rows::<ColonRow>(&out.join(COLON_NAME))?;
    let mut document_bos = 0;
    for (offset, &id) in tape.iter().enumerate() {
        if id == BOS {
            document_bos = offset;
        }
        if offset <= document_bos || tape[offset - 1] != COLON {
            continue;
        }
        if mask[offset] == 0 {
            census.colon_mask_zero_successors += 1;
            continue;
        }
        let actual = colon_rows.next().ok_or("missing eligible colon row")??;
        let prefix_start = if offset - document_bos > WINDOW {
            census.colon_rows_window_clipped += 1;
            offset - WINDOW
        } else {
            document_bos
        };
        let served_prefix = genuine.get(&offset);
        let genuine_assistant = served_prefix.is_some();
        let prefix = &tape[prefix_start..offset];
        require(
            actual.offset == offset
                && actual.prefix == prefix
                && actual.target == id
                && actual.genuine_assistant == genuine_assistant,
            format!("persisted colon row differs from tape at offset {offset}"),
        )?;
        if let Some(served_prefix) = served_prefix {
            census.colon_genuine_assistant += 1;
            require(
                prefix.ends_with(&MARKER) && &actual.prefix == served_prefix,
                format!("genuine device prefix differs from ordinary serving at offset {offset}"),
            )?;
            census.genuine_device_prefixes_verified += 1;
        } else {
            if auxiliary_offsets
                .range(..=offset)
                .next_back()
                .is_some_and(|(_, &end)| offset < end)
            {
                census.colon_auxiliary += 1;
            } else {
                census.colon_visible_other += 1;
            }
            census.colon_full_marker_outside_genuine_opening +=
                usize::from(prefix.ends_with(&MARKER));
        }
        census.colon_rows += 1;
        census.colon_rows_with_bos += usize::from(prefix.first() == Some(&BOS));
        census.colon_space_id += usize::from(id == SPACE);
        census.colon_genuine_space_id += usize::from(genuine_assistant && id == SPACE);
    }
    require(
        colon_rows.next().is_none()
            && census.colon_genuine_assistant == TURNS
            && census.genuine_device_prefixes_verified == TURNS
            && census.colon_genuine_space_id
                == census.immediate_ids.get(&SPACE).copied().unwrap_or(0),
        "eligible colon coverage differs",
    )?;
    let mut files = BTreeMap::new();
    for name in DATA_FILES.into_iter().chain(["source.json"]) {
        let path = out.join(name);
        files.insert(
            name,
            json!({"bytes": path.metadata()?.len(), "sha256": file_sha(&path)?}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "uor-r4.canonical-d87-donor-readback/1",
            "source": source,
            "census": census,
            "files": files,
            "colon_window_includes_the_current_document_bos_if_it_fits": true,
            "genuine_assistant_marker_original_ids": MARKER,
            "standalone_space_original_id": SPACE,
        }))?
    );
    Ok(())
}
