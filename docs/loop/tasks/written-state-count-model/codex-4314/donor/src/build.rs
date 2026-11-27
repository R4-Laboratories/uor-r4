use super::*;
use m2_student_corpus_ingest::{parse_record, Coder, Format, Segment, SegmentCount};
use serde_json::json;
use std::io::{Seek, SeekFrom};
use std::os::unix::fs::DirBuilderExt;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol};

struct Tape {
    tokens: BufWriter<File>,
    mask: BufWriter<File>,
    len: usize,
}

impl Tape {
    fn new(out: &Path) -> CorpusResult<Self> {
        let mut tokens = create(out, TAPE_NAME)?;
        tokens.write_all(&[0; HEADER])?;
        Ok(Self {
            tokens,
            mask: create(out, MASK_NAME)?,
            len: 0,
        })
    }

    fn append(&mut self, tokens: &[u32], mask: &[u8]) -> CorpusResult<()> {
        require(tokens.len() == mask.len(), "tape mask length differs")?;
        for &token in tokens {
            require(token < VOCAB, "unrepresentable tape ID")?;
            self.tokens
                .write_all(&u16::try_from(token)?.to_le_bytes())?;
        }
        self.mask.write_all(mask)?;
        self.len += tokens.len();
        Ok(())
    }

    fn finish(mut self) -> CorpusResult<()> {
        let mut header = [0; HEADER];
        header[..8].copy_from_slice(b"UORT\x01\0\0\0");
        header[8..16].copy_from_slice(&u64::try_from(self.len)?.to_le_bytes());
        header[16..20].copy_from_slice(&VOCAB.to_le_bytes());
        self.tokens.seek(SeekFrom::Start(0))?;
        self.tokens.write_all(&header)?;
        self.tokens.flush()?;
        self.mask.flush()?;
        Ok(())
    }
}

fn historical_auxiliaries(
    coder: &Coder,
    line: &str,
    r: &[u8],
    r_position: &mut usize,
) -> CorpusResult<Vec<Vec<u32>>> {
    let record = parse_record(line)?;
    require(record.format == Format::D87, "expected D87 source")?;
    let mut auxiliaries = Vec::new();
    check_r_id(r, r_position, BOS)?;
    for piece in &record.pieces {
        let mut coverage = SegmentCount::default();
        let ids = coder.encode_piece(&piece.text, &mut coverage, &mut |_| {})?;
        require(
            coverage.dropped_bytes == 0 && coverage.special_in_text == 0,
            "historical text has dropped bytes or literal special IDs",
        )?;
        for &id in &ids {
            check_r_id(r, r_position, id)?;
        }
        match (piece.segment, piece.mask) {
            (Segment::Reasoning, 1) => {
                let mut doc = vec![BOS];
                doc.extend(ids);
                auxiliaries.push(doc);
            }
            (Segment::Frame, 1) => auxiliaries
                .last_mut()
                .ok_or("historical reasoning separator without initial reasoning")?
                .extend(ids),
            _ => {}
        }
    }
    check_r_id(r, r_position, EOS)?;
    Ok(auxiliaries)
}

fn check_r_id(r: &[u8], position: &mut usize, id: u32) -> CorpusResult<()> {
    require(
        r.get(*position..*position + 2) == Some(&u16::try_from(id)?.to_le_bytes()),
        format!(
            "historical raw source token differs from R at byte {}",
            *position
        ),
    )?;
    *position += 2;
    Ok(())
}

fn opening(
    encoder: &DialogueEncoder<'_>,
    tok: &ByteBpeTokenizer,
    messages: &[Message<'_>],
    document: &[u32],
    payload_start: usize,
    group: usize,
    turn: usize,
) -> CorpusResult<Opening> {
    let user_end = 2 * turn + 1;
    let prefix = encoder.encode_assistant_prefix(&messages[..user_end]);
    let through_reply = encoder.encode_open_history(&messages[..user_end + 1]);
    require(
        prefix.emitted_turns == user_end
            && through_reply.emitted_turns == user_end + 1
            && document.starts_with(&prefix.tokens)
            && document.starts_with(&through_reply.tokens)
            && prefix.tokens.ends_with(&MARKER)
            && through_reply.tokens.last() == Some(&EOS),
        format!("protocol-2 opening disagrees at group {group} turn {turn}"),
    )?;
    let start = prefix.tokens.len();
    let end = through_reply.tokens.len() - 1;
    let answer = document.get(start..end).ok_or("unbounded answer")?;
    let immediate_id = *answer.first().ok_or("empty answer")?;
    let (lexical_index, &lexical_id) = answer
        .iter()
        .enumerate()
        .find(|(_, id)| !tok.decode(&[**id]).chars().all(char::is_whitespace))
        .ok_or("answer has only whitespace IDs")?;
    Ok(Opening {
        group,
        turn,
        prefix_start: payload_start,
        answer_start: payload_start + start,
        answer_end: payload_start + end,
        immediate_id,
        lexical_id,
        lexical_offset: payload_start + start + lexical_index,
        lexical_answer_offset: lexical_index,
    })
}

fn write_colons(
    writer: &mut BufWriter<File>,
    tokens: &[u32],
    mask: &[u8],
    payload_start: usize,
    openings: &[Opening],
) -> CorpusResult<()> {
    let mut history_start = 0;
    for (index, pair) in tokens.windows(2).enumerate() {
        if pair[0] == BOS {
            history_start = index;
        }
        let target = index + 1;
        if pair[0] == COLON && mask[target] == 1 {
            let offset = payload_start + target;
            write_row(
                writer,
                &ColonRow {
                    prefix: tokens[history_start.max(target.saturating_sub(WINDOW))..target]
                        .to_vec(),
                    target: pair[1],
                    offset,
                    genuine_assistant: openings
                        .iter()
                        .any(|opening| opening.answer_start == offset),
                },
            )?;
        }
    }
    Ok(())
}

pub(super) fn run(inputs: &Inputs, prior: &Path, out: &Path, threads: usize) -> CorpusResult<()> {
    require(!out.exists(), "output directory already exists")?;
    let tok = load_tokenizer(&inputs.p4_bpe)?;
    let protocol = DialogueProtocol::literal_roles_v2(&tok)?;
    let encoder = protocol.bind(&tok)?;
    let (legacy_coder, r) = inputs.verify()?;
    let old_receipt = prior_receipt(prior)?;
    let refs = old_file(prior, &old_receipt, "groups.jsonl")?;
    let mut refs = std::str::from_utf8(&refs)?.lines();
    let records = old_file(prior, &old_receipt, "records.jsonl")?;
    let mut records = std::str::from_utf8(&records)?.lines();

    fs::DirBuilder::new().mode(0o700).create(out)?;
    let mut tape = Tape::new(out)?;
    let mut group_writer = create(out, GROUPS_NAME)?;
    let mut opening_writer = create(out, OPENINGS_NAME)?;
    let mut colon_writer = create(out, COLON_NAME)?;
    let mut r_position = HEADER;
    let mut group = 0;
    let mut turn_count = 0;
    inputs.each_record(threads, &mut |train, line| {
        let record: Record = serde_json::from_str(line)?;
        let reference: PriorGroup =
            serde_json::from_str(refs.next().ok_or("missing prior group")?)?;
        require(
            records.next() == Some(line)
                && reference.raw_record_sha256 == sha(line.as_bytes())
                && reference.id == record.id
                && reference.group == group
                && reference.canonical_train_group == train
                && reference.turns == record.turns.len()
                && reference.turns == record.messages.len()
                && record.messages.iter().all(|message| message.role == "user"),
            format!("raw source identity differs at canonical TRAIN ordinal {train}"),
        )?;
        let auxiliary = historical_auxiliaries(&legacy_coder, line, &r, &mut r_position)?;
        require(
            auxiliary.len() == reference.reasoning_turns,
            "auxiliary count differs",
        )?;

        let messages = record.messages();
        let mut document = encoder.encode_document(&messages);
        require(
            document.emitted_turns == messages.len()
                && document.special_token_occurrences == 0
                && document.tokens.first() == Some(&BOS)
                && document.tokens.last() == Some(&EOS),
            format!(
                "protocol 2 dropped a message or encoded a literal special ID in group {group}"
            ),
        )?;
        let mut openings = Vec::new();
        let payload_start = tape.len;
        for turn in 0..record.turns.len() {
            let row = opening(
                &encoder,
                &tok,
                &messages,
                &document.tokens,
                payload_start,
                group,
                turn,
            )?;
            if turn + 1 != record.turns.len() {
                // Carry completed assistant closures as serving context. The original
                // D87 policy supplies EOS supervision only at the end of a visible group.
                document.response_mask[row.answer_end - payload_start] = 0;
            }
            write_row(&mut opening_writer, &row)?;
            openings.push(row);
        }
        let visible_end = payload_start + document.tokens.len();
        for aux in auxiliary {
            document.response_mask.push(0);
            document
                .response_mask
                .resize(document.response_mask.len() + aux.len() - 1, 1);
            document.tokens.extend(aux);
        }
        tape.append(&document.tokens, &document.response_mask)?;
        write_colons(
            &mut colon_writer,
            &document.tokens,
            &document.response_mask,
            payload_start,
            &openings,
        )?;
        write_row(
            &mut group_writer,
            &Group {
                group,
                canonical_train_group: train,
                payload_start,
                visible_end,
                payload_end: tape.len,
            },
        )?;
        turn_count += openings.len();
        group += 1;
        Ok(())
    })?;
    require(
        group == GROUPS
            && turn_count == TURNS
            && refs.next().is_none()
            && records.next().is_none()
            && r_position == r.len(),
        "historical and rebuilt donor coverage differs",
    )?;
    tape.finish()?;
    group_writer.flush()?;
    opening_writer.flush()?;
    colon_writer.flush()?;
    let mut provenance = create(out, "source.json")?;
    write_row(
        &mut provenance,
        &json!({
            "input": inputs.binding(),
            "historical_receipt_sha256": OLD_RECEIPT,
            "historical_records_sha256": old_receipt["files"]["records.jsonl"]["sha256"],
            "historical_groups_sha256": old_receipt["files"]["groups.jsonl"]["sha256"],
            "historical_raw_r_reconstruction_byte_identical": true,
            "structured_records_match_historical_extraction_byte_identical": true,
            "protocol": protocol,
            "protocol_identity": protocol.identity()?,
        }),
    )?;
    provenance.flush()?;
    Ok(())
}
