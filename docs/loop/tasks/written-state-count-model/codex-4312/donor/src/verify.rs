use crate::{require, source::file_sha, CorpusResult, Token, GROUPS, TURNS};
use m2_student_corpus_ingest::{Segment, UORT_HEADER, VOCAB};
use serde_json::Value;
use std::fs;
use std::path::Path;

struct Readback {
    tokens: Vec<Token>,
    geometry: Vec<u8>,
    targets: Vec<usize>,
    geometry_targets: Vec<usize>,
}

impl Readback {
    fn open(dir: &Path, name: &str) -> CorpusResult<Self> {
        let read = |suffix| fs::read(dir.join(format!("{name}.{suffix}")));
        let tape = read("tokens.u16")?;
        let mask = read("mask.u8")?;
        let segment = read("segment.u8")?;
        let geometry = read("geometry-mask.u8")?;
        let (header, tape) = tape.split_at_checked(UORT_HEADER).ok_or("short UORT")?;
        let tokens = usize::try_from(u64::from_le_bytes(header[8..16].try_into()?))?;
        require(
            &header[..8] == b"UORT\x01\x00\x00\x00"
                && header[16..20] == VOCAB.to_le_bytes()
                && header[20..].iter().all(|&byte| byte == 0)
                && tape.len() == 2 * tokens
                && mask.len() == tokens
                && segment.len() == tokens
                && geometry.len() == tokens,
            "UORT, masks or segment lengths differ",
        )?;
        let tokens = tape
            .chunks_exact(2)
            .zip(segment)
            .zip(&mask)
            .map(|((id, segment), &mask)| Token {
                id: u16::from_le_bytes([id[0], id[1]]),
                segment,
                mask,
            })
            .collect::<Vec<_>>();
        require(
            tokens.iter().zip(&geometry).all(|(token, &geo)| {
                u32::from(token.id) < VOCAB
                    && token.segment <= Segment::Answer as u8
                    && geo <= token.mask
                    && token.mask <= 1
            }),
            "inadmissible id, segment, or mask value",
        )?;
        let positions = |suffix, mask: &[u8]| -> CorpusResult<_> {
            let bytes = read(suffix)?;
            let mut seen = vec![0u8; mask.len()];
            require(bytes.len() % 4 == 0, "partial event position")?;
            let positions = bytes
                .chunks_exact(4)
                .map(|bytes| {
                    let pos = usize::try_from(u32::from_le_bytes(bytes.try_into()?))?;
                    let old = seen.get_mut(pos).ok_or("event address outside payload")?;
                    require(*old == 0, "duplicate paired event address")?;
                    *old = 1;
                    Ok(pos)
                })
                .collect::<CorpusResult<Vec<_>>>()?;
            require(seen == mask, "event-map coverage differs from target mask")?;
            Ok(positions)
        };
        let targets = positions("targets.u32", &mask)?;
        let geometry_targets = positions("geometry-targets.u32", &geometry)?;
        Ok(Self {
            tokens,
            geometry,
            targets,
            geometry_targets,
        })
    }
}

fn number(record: &Value, key: &str) -> CorpusResult<usize> {
    Ok(usize::try_from(
        record[key]
            .as_u64()
            .ok_or_else(|| format!("missing numeric {key}"))?,
    )?)
}

fn verify_layout(path: &Path, receipt: &Value) -> CorpusResult<()> {
    let a = Readback::open(path, "reasoning-first")?;
    let b = Readback::open(path, "answer-first")?;
    let groups = fs::read_to_string(path.join("groups.jsonl"))?;
    let raw = fs::read_to_string(path.join("records.jsonl"))?;
    let mut raw = raw.lines();
    let mut start = 0;
    let mut target_start = 0;
    let mut geometry_start = 0;
    let mut groups_read = 0;
    let mut turns = 0;
    let mut reasoning_turns = 0;
    for line in groups.lines() {
        let meta: Value = serde_json::from_str(line)?;
        let raw_line = raw.next().ok_or("missing selected raw line")?;
        let raw_record: Value = serde_json::from_str(raw_line)?;
        require(
            meta["id"] == raw_record["id"]
                && meta["raw_record_sha256"].as_str()
                    == Some(&crate::source::sha(raw_line.as_bytes()))
                && number(&meta, "group")? == groups_read
                && number(&meta, "canonical_train_group")? == groups_read * 128
                && number(&meta, "payload_start")? == start
                && number(&meta, "target_start")? == target_start
                && number(&meta, "geometry_target_start")? == geometry_start,
            "group source order, identity, or cumulative offsets changed",
        )?;
        let raw_turns = raw_record["turns"]
            .as_array()
            .ok_or("missing selected raw turn array")?;
        let raw_reasoning_turns = raw_turns
            .iter()
            .filter(|turn| {
                turn["reasoning"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty())
            })
            .count();
        require(
            raw_turns.len() == number(&meta, "turns")?
                && raw_reasoning_turns == number(&meta, "reasoning_turns")?,
            "selected raw turn counts differ from the group manifest",
        )?;
        let length = number(&meta, "payload_tokens_both_arms")?;
        let a_group = a
            .tokens
            .get(start..start + length)
            .ok_or("arm A group outside payload")?;
        let b_group = b
            .tokens
            .get(start..start + length)
            .ok_or("arm B group outside payload")?;
        let (a_visible, a_aux) = a_group
            .split_at_checked(number(&meta, "visible_tokens_reasoning_first")?)
            .ok_or("invalid A visible length")?;
        let (b_visible, b_aux) = b_group
            .split_at_checked(number(&meta, "visible_tokens_answer_first")?)
            .ok_or("invalid B visible length")?;
        require(
            a_visible.first() == Some(&Token::BOS)
                && a_visible.last() == Some(&Token::EOS)
                && b_visible.first() == Some(&Token::BOS)
                && b_visible.last() == Some(&Token::EOS)
                && a_visible
                    .iter()
                    .filter(|&&token| token == Token::BOS || token == Token::EOS)
                    .count()
                    == 2
                && a_aux.len() == raw_reasoning_turns
                && a_aux.iter().all(|&token| token == Token::BOS),
            "bad visible frame or A auxiliary padding produced training events",
        )?;

        // In the canonical record only reasoning and its supervised separator precede an
        // answer as mask-1 non-EOS tokens with these tags. Derive the expected B span from the
        // read-back A tape, independent of the writer's input-piece offsets.
        let moved = |token: Token| {
            token != Token::EOS && token.mask == 1 && token.segment != Segment::Answer as u8
        };
        require(
            a_visible
                .iter()
                .copied()
                .filter(|&token| !moved(token))
                .eq(b_visible.iter().copied()),
            "B visible history contains detached material or differs in user/answer/framing tokens",
        )?;
        let mut aux = b_aux;
        for span in a_visible.chunk_by(|left, right| moved(*left) == moved(*right)) {
            if !moved(span[0]) {
                continue;
            }
            require(
                aux.first() == Some(&Token::BOS)
                    && aux.get(1..1 + span.len()) == Some(span)
                    && span[0].segment == Segment::Reasoning as u8
                    && span[span.len() - 1].segment == Segment::Frame as u8,
                "B auxiliary context lacks a reset, reasoning, or separator",
            )?;
            aux = &aux[1 + span.len()..];
        }
        require(aux.is_empty(), "unaccounted detached material")?;

        let targets = number(&meta, "eligible_targets_both_arms")?;
        let a_positions = a
            .targets
            .get(target_start..target_start + targets)
            .ok_or("short A target map")?;
        let b_positions = b
            .targets
            .get(target_start..target_start + targets)
            .ok_or("short B target map")?;
        require(
            a_positions
                .iter()
                .copied()
                .eq((start..start + a_visible.len()).filter(|&pos| a.tokens[pos].mask == 1)),
            "A target map is not in canonical order",
        )?;
        require(
            a_positions.iter().zip(b_positions).all(|(&a_pos, &b_pos)| {
                (start..start + length).contains(&b_pos) && a.tokens[a_pos] == b.tokens[b_pos]
            }),
            "paired supervision occurrence does not address the same token and segment",
        )?;

        let geometry_targets = number(&meta, "geometry_targets_both_arms")?;
        let geometry_a = a
            .geometry_targets
            .get(geometry_start..geometry_start + geometry_targets)
            .ok_or("short A geometry map")?;
        let geometry_b = b
            .geometry_targets
            .get(geometry_start..geometry_start + geometry_targets)
            .ok_or("short B geometry map")?;
        let expected_a = crate::geometry_selection(groups_read * 128, a_positions);
        require(
            geometry_a == expected_a
                && a_positions
                    .iter()
                    .zip(b_positions)
                    .filter(|&(&a_pos, _)| a.geometry[a_pos] == 1)
                    .map(|(_, &b_pos)| b_pos)
                    .eq(geometry_b.iter().copied()),
            "geometry order or occurrence selection differs between arms",
        )?;

        groups_read += 1;
        turns += raw_turns.len();
        reasoning_turns += raw_reasoning_turns;
        start += length;
        target_start += targets;
        geometry_start += geometry_targets;
    }
    require(
        groups_read == GROUPS
            && turns == TURNS
            && reasoning_turns == number(receipt, "reasoning_turns")?
            && raw.next().is_none()
            && start == a.tokens.len()
            && start == b.tokens.len()
            && target_start == a.targets.len()
            && target_start == b.targets.len()
            && geometry_start == a.geometry_targets.len()
            && geometry_start == b.geometry_targets.len(),
        "incomplete corpus readback",
    )?;
    Ok(())
}

pub fn run(path: &Path) -> CorpusResult<()> {
    let receipt: Value = serde_json::from_slice(&fs::read(path.join("receipt.json"))?)?;
    let files = receipt["files"]
        .as_object()
        .ok_or("missing paired-file digests")?;
    require(files.len() == 14, "incomplete paired-file digest set")?;
    for (name, record) in files {
        let file = path.join(name);
        require(
            Path::new(name).file_name().and_then(|path| path.to_str()) == Some(name)
                && file_sha(&file)? == record["sha256"]
                && fs::metadata(file)?.len() == number(record, "bytes")? as u64,
            "prepared corpus file differs from the receipt",
        )?;
    }
    verify_layout(path, &receipt)?;
    println!(
        "verified both tapes, all masks and maps, and isolated histories for {GROUPS} donor groups"
    );
    Ok(())
}
