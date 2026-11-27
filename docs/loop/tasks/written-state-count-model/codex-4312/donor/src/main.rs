#![forbid(unsafe_code)]

mod source;
mod verify;

use m2_student_corpus_ingest::{
    parse_record, Coder, Format, Piece, Segment, SegmentCount, ASSISTANT_MARKER, BOS, EOS,
    SEPARATOR, UORT_HEADER, USER_MARKER, VOCAB,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use source::{file_sha, sha, Inputs};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::ops::Range;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

type CorpusResult<T> = Result<T, Box<dyn Error>>;
const GROUPS: usize = 4060;
const TURNS: usize = 4345;
const GEOMETRY_SEED: u64 = 4312;
const GEOMETRY_PER_GROUP: usize = 64;
const GEOMETRY_DOMAIN: &[u8] = b"uor-r4/4312/matched-geometry/1\0";

fn require(condition: bool, message: impl Into<String>) -> CorpusResult<()> {
    if !condition {
        return Err(message.into().into());
    }
    Ok(())
}

fn create(out: &Path, name: &str) -> CorpusResult<BufWriter<File>> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join(name))?;
    Ok(BufWriter::new(file))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Token {
    id: u16,
    segment: u8,
    mask: u8,
}

impl Token {
    const BOS: Self = Self {
        id: BOS as u16,
        segment: 0,
        mask: 0,
    };
    const EOS: Self = Self {
        id: EOS as u16,
        segment: 0,
        mask: 1,
    };
}

struct Turn {
    detached: Range<usize>,
    answer: Range<usize>,
}

struct CanonicalGroup {
    id: String,
    tokens: Vec<Token>,
    turns: Vec<Turn>,
}

struct PairedOrders {
    reasoning_first: Vec<Option<usize>>,
    answer_first: Vec<Option<usize>>,
    answer_first_visible: usize,
}

impl CanonicalGroup {
    fn parse(coder: &Coder, line: &str) -> CorpusResult<Self> {
        let record = parse_record(line)?;
        require(
            record.format == Format::D87,
            "expected a D87 donor source record",
        )?;
        let mut group = Self {
            id: record.id,
            tokens: vec![Token::BOS],
            turns: Vec::new(),
        };
        let mut pieces = record.pieces.as_slice();
        while !pieces.is_empty() {
            let [user, prompt, assistant, rest @ ..] = pieces else {
                return Err("incomplete user and assistant markers".into());
            };
            require(
                user.segment == Segment::Frame
                    && user.mask == 0
                    && user.text == USER_MARKER
                    && prompt.segment == Segment::Prompt
                    && prompt.mask == 0
                    && assistant.segment == Segment::Frame
                    && assistant.mask == 0
                    && assistant.text == ASSISTANT_MARKER,
                "invalid role marker sequence from canonical ingest",
            )?;
            for piece in [user, prompt, assistant] {
                group.encode(coder, piece)?;
            }
            let boundary = group.tokens.len();
            pieces = rest;
            if let [reasoning, separator, tail @ ..] = pieces {
                if reasoning.segment == Segment::Reasoning {
                    require(
                        reasoning.mask == 1
                            && separator.segment == Segment::Frame
                            && separator.mask == 1
                            && separator.text == SEPARATOR,
                        "invalid reasoning separator from canonical ingest",
                    )?;
                    group.encode(coder, reasoning)?;
                    group.encode(coder, separator)?;
                    pieces = tail;
                }
            }
            let answer_start = group.tokens.len();
            let [answer, tail @ ..] = pieces else {
                return Err("missing final answer".into());
            };
            require(
                answer.segment == Segment::Answer && answer.mask == 1,
                "invalid final-answer field",
            )?;
            group.encode(coder, answer)?;
            let answer_end = group.tokens.len();
            require(
                answer_end > answer_start,
                "a final-answer field has no supervised tokens",
            )?;
            group.turns.push(Turn {
                detached: boundary..answer_start,
                answer: answer_start..answer_end,
            });
            pieces = tail;
        }
        group.tokens.push(Token::EOS);
        Ok(group)
    }

    fn encode(&mut self, coder: &Coder, piece: &Piece) -> CorpusResult<()> {
        let mut coverage = SegmentCount::default();
        let ids = coder.encode_piece(&piece.text, &mut coverage, &mut |_| {})?;
        require(
            coverage.dropped_bytes == 0 && coverage.special_in_text == 0,
            "original ingest piece contains dropped bytes or special tokens",
        )?;
        for id in ids {
            require(
                id != BOS && id != EOS && id < VOCAB,
                "unexpected frame token in donor text",
            )?;
            self.tokens.push(Token {
                id: u16::try_from(id)?,
                segment: piece.segment as u8,
                mask: piece.mask,
            });
        }
        Ok(())
    }

    fn orders(&self) -> CorpusResult<PairedOrders> {
        let auxiliary: Vec<_> = self
            .turns
            .iter()
            .map(|turn| &turn.detached)
            .filter(|range| !range.is_empty())
            .collect();
        let mut a: Vec<_> = (0..self.tokens.len()).map(Some).collect();
        a.resize(a.len() + auxiliary.len(), None);
        let mut b: Vec<_> = (0..self.tokens.len())
            .filter(|index| auxiliary.iter().all(|range| !range.contains(index)))
            .map(Some)
            .collect();
        let visible = b.len();
        require(
            b.iter()
                .flatten()
                .all(|&index| self.tokens[index].segment != Segment::Reasoning as u8)
                && b.last() == Some(&Some(self.tokens.len() - 1)),
            "reasoning or missing terminal EOS in answer-first visible document",
        )?;
        for range in auxiliary {
            b.push(None);
            b.extend(range.clone().map(Some));
        }
        Ok(PairedOrders {
            reasoning_first: a,
            answer_first: b,
            answer_first_visible: visible,
        })
    }
}

fn geometry_selection(train: usize, targets: &[usize]) -> Vec<usize> {
    let mut ranked: Vec<_> = targets
        .iter()
        .enumerate()
        .map(|(event, &source)| {
            let mut hash = Sha256::new();
            hash.update(GEOMETRY_DOMAIN);
            hash.update(GEOMETRY_SEED.to_le_bytes());
            hash.update((train as u64).to_le_bytes());
            hash.update((event as u64).to_le_bytes());
            (<[u8; 32]>::from(hash.finalize()), event, source)
        })
        .collect();
    if ranked.len() > GEOMETRY_PER_GROUP {
        ranked.select_nth_unstable(GEOMETRY_PER_GROUP);
        ranked.truncate(GEOMETRY_PER_GROUP);
    }
    ranked.sort_unstable_by_key(|entry| entry.1);
    ranked.into_iter().map(|entry| entry.2).collect()
}

struct Arm {
    name: &'static str,
    tokens: BufWriter<File>,
    mask: BufWriter<File>,
    segment: BufWriter<File>,
    geometry: BufWriter<File>,
    targets: BufWriter<File>,
    geometry_targets: BufWriter<File>,
    count: [u64; VOCAB as usize],
    selected: [u64; VOCAB as usize],
    geometry_selected: [u64; VOCAB as usize],
    segments: [u64; 4],
    payload: usize,
}

impl Arm {
    fn new(path: &Path, name: &'static str) -> CorpusResult<Self> {
        let new = |suffix| create(path, &format!("{name}.{suffix}"));
        let mut tokens = new("tokens.u16")?;
        tokens.write_all(&[0; UORT_HEADER])?;
        Ok(Self {
            name,
            tokens,
            mask: new("mask.u8")?,
            segment: new("segment.u8")?,
            geometry: new("geometry-mask.u8")?,
            targets: new("targets.u32")?,
            geometry_targets: new("geometry-targets.u32")?,
            count: [0; VOCAB as usize],
            selected: [0; VOCAB as usize],
            geometry_selected: [0; VOCAB as usize],
            segments: [0; 4],
            payload: 0,
        })
    }

    fn add(
        &mut self,
        canonical: &[Token],
        order: &[Option<usize>],
        canonical_targets: &[usize],
        geometry: &[usize],
    ) -> CorpusResult<()> {
        let mut mapped = vec![None; canonical.len()];
        for (position, source) in order.iter().enumerate() {
            if let Some(source) = source {
                require(
                    mapped[*source].replace(position).is_none(),
                    "duplicate canonical token occurrence",
                )?;
            }
        }
        require(
            mapped.iter().all(Option::is_some),
            "missing canonical token occurrence",
        )?;
        let position_of = |source| {
            mapped
                .get(source)
                .copied()
                .flatten()
                .ok_or("missing source position")
        };
        let mut geometry_here = vec![0u8; order.len()];
        for &source in geometry {
            let local = position_of(source)?;
            geometry_here[local] = 1;
            let address = u32::try_from(self.payload + local)?;
            self.geometry_targets.write_all(&address.to_le_bytes())?;
        }
        for &source in canonical_targets {
            let address = u32::try_from(self.payload + position_of(source)?)?;
            self.targets.write_all(&address.to_le_bytes())?;
        }
        for (source, selected_geometry) in order.iter().zip(geometry_here) {
            let token = source.map_or(Token::BOS, |index| canonical[index]);
            require(
                selected_geometry <= token.mask,
                "geometry selection included a masked token",
            )?;
            let id = usize::from(token.id);
            self.count[id] += 1;
            self.selected[id] += u64::from(token.mask);
            self.geometry_selected[id] += u64::from(selected_geometry);
            self.segments[usize::from(token.segment)] += 1;
            self.tokens.write_all(&token.id.to_le_bytes())?;
            self.mask.write_all(&[token.mask])?;
            self.segment.write_all(&[token.segment])?;
            self.geometry.write_all(&[selected_geometry])?;
        }
        self.payload += order.len();
        Ok(())
    }

    fn finish(mut self) -> CorpusResult<()> {
        let mut header = [0u8; UORT_HEADER];
        header[..8].copy_from_slice(b"UORT\x01\x00\x00\x00");
        header[8..16].copy_from_slice(&(self.payload as u64).to_le_bytes());
        header[16..20].copy_from_slice(&VOCAB.to_le_bytes());
        self.tokens.seek(SeekFrom::Start(0))?;
        self.tokens.write_all(&header)?;
        self.tokens.flush()?;
        self.mask.flush()?;
        self.segment.flush()?;
        self.geometry.flush()?;
        self.targets.flush()?;
        self.geometry_targets.flush()?;
        Ok(())
    }

    fn binding(&self) -> Value {
        let histogram = |count: &[u64]| {
            sha(&count
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect::<Vec<_>>())
        };
        json!({
            "name": self.name,
            "payload_tokens": self.payload,
            "bos": self.count[BOS as usize],
            "eos": self.count[EOS as usize],
            "eligible_targets": self.selected.iter().sum::<u64>(),
            "geometry_targets": self.geometry_selected.iter().sum::<u64>(),
            "segment_counts_frame_prompt_reasoning_answer": self.segments,
            "payload_id_histogram_sha256": histogram(&self.count),
            "eligible_id_histogram_sha256": histogram(&self.selected),
            "geometry_id_histogram_sha256": histogram(&self.geometry_selected),
        })
    }
}

fn main() -> CorpusResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if let [flag, out] = args.as_slice() {
        require(
            flag == "--verify",
            "usage: uor-visible-answer-donor --verify OUT",
        )?;
        return verify::run(Path::new(out));
    }
    let [raw, split, d87_bpe, p4_bpe, train, r, out, threads] = args.as_slice() else {
        return Err("usage: uor-visible-answer-donor RAW_ROOT SPLIT_SUMMARY D87_BPE P4_BPE FULL_TRAIN R OUT THREADS".into());
    };
    let inputs = Inputs {
        raw: raw.into(),
        split: split.into(),
        d87_bpe: d87_bpe.into(),
        p4_bpe: p4_bpe.into(),
        train: train.into(),
        r: r.into(),
    };
    let threads = threads
        .to_str()
        .ok_or("invalid threads")?
        .parse::<usize>()?;
    require((1..=16).contains(&threads), "threads must be 1..=16")?;
    let out = PathBuf::from(out);
    let (coder, r) = inputs.verify()?;
    eprintln!(
        "verified full canonical TRAIN, R stride selection, and the common tokenizer vocabulary"
    );
    fs::DirBuilder::new().mode(0o700).create(&out)?;
    let mut a = Arm::new(&out, "reasoning-first")?;
    let mut b = Arm::new(&out, "answer-first")?;
    let mut groups = create(&out, "groups.jsonl")?;
    let mut raw_selected = create(&out, "records.jsonl")?;
    let mut r_position = UORT_HEADER;
    let mut target_begin = 0usize;
    let mut geometry_begin = 0usize;
    let mut group_count = 0;
    let mut turn_count = 0;
    let mut reasoning_turn_count = 0;
    let mut reasoning_first_openings = BTreeMap::new();
    let mut answer_first_openings = BTreeMap::new();
    inputs.each_record(threads, &mut |train, line| {
        let parsed = CanonicalGroup::parse(&coder, line)?;
        for token in &parsed.tokens {
            require(r.get(r_position..r_position + 2) == Some(&token.id.to_le_bytes()),
                format!("raw re-encoding differs from exact R at TRAIN group {train}"))?;
            r_position += 2;
        }
        let PairedOrders {
            reasoning_first: a_order,
            answer_first: b_order,
            answer_first_visible: b_visible,
        } = parsed.orders()?;
        require(a_order.len() == b_order.len() && a.payload == b.payload,
            "matched group framing differs")?;
        let reasoning_turns = parsed.turns.iter().filter(|turn| !turn.detached.is_empty()).count();
        for turn in &parsed.turns {
            *reasoning_first_openings.entry(parsed.tokens[turn.detached.start].id).or_insert(0u64) += 1;
            *answer_first_openings.entry(parsed.tokens[turn.answer.start].id).or_insert(0u64) += 1;
        }
        let targets: Vec<_> = parsed.tokens.iter().enumerate().filter(|(_, token)| token.mask == 1)
            .map(|(pos, _)| pos).collect();
        let selected_geometry = geometry_selection(train, &targets);
        serde_json::to_writer(&mut groups, &json!({
            "id": parsed.id, "raw_record_sha256": sha(line.as_bytes()),
            "group": group_count, "canonical_train_group": train,
            "payload_start": a.payload, "payload_tokens_both_arms": a_order.len(),
            "visible_tokens_reasoning_first": parsed.tokens.len(), "visible_tokens_answer_first": b_visible,
            "target_start": target_begin, "eligible_targets_both_arms": targets.len(),
            "geometry_target_start": geometry_begin, "geometry_targets_both_arms": selected_geometry.len(),
            "turns": parsed.turns.len(), "reasoning_turns": reasoning_turns,
        }))?;
        groups.write_all(b"\n")?;
        raw_selected.write_all(line.as_bytes())?;
        raw_selected.write_all(b"\n")?;
        a.add(&parsed.tokens, &a_order, &targets, &selected_geometry)?;
        b.add(&parsed.tokens, &b_order, &targets, &selected_geometry)?;
        target_begin += targets.len();
        geometry_begin += selected_geometry.len();
        group_count += 1;
        turn_count += parsed.turns.len();
        reasoning_turn_count += reasoning_turns;
        Ok(())
    })?;
    require(
        group_count == GROUPS && turn_count == TURNS && r_position == r.len(),
        "raw reconstruction ended with an unexpected group, turn, or token count",
    )?;
    require(
        a.payload == b.payload
            && a.count == b.count
            && a.selected == b.selected
            && a.geometry_selected == b.geometry_selected
            && a.segments == b.segments
            && a.count[BOS as usize] == (group_count + reasoning_turn_count) as u64
            && a.count[EOS as usize] == group_count as u64
            && a.selected[EOS as usize] == group_count as u64,
        "arm token/target identity histograms, segment frequencies, or BOS/EOS counts differ",
    )?;
    let arms = [a.binding(), b.binding()];
    a.finish()?;
    b.finish()?;
    groups.flush()?;
    raw_selected.flush()?;
    let files = fs::read_dir(&out)?
        .map(|entry| {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| "invalid output file name")?;
            Ok((
                name,
                json!({"sha256":file_sha(&entry.path())?, "bytes": entry.metadata()?.len()}),
            ))
        })
        .collect::<CorpusResult<BTreeMap<_, _>>>()?;
    let openings = |histogram: &BTreeMap<u16, u64>| {
        json!({
            "distinct_ids": histogram.len(),
            "id_799": histogram.get(&799).copied().unwrap_or(0),
        })
    };
    let receipt = json!({
        "schema": "uor-r4.visible-answer-donor/1",
        "input": inputs.binding(),
        "groups": group_count, "turns": turn_count, "reasoning_turns": reasoning_turn_count,
        "visible_assistant_opening": {
            "reasoning_first": openings(&reasoning_first_openings),
            "answer_first": openings(&answer_first_openings),
        },
        "same_ordered_original_groups": true,
        "reconstructed_original_r_byte_equal": true,
        "answer_first_visible_training_reasoning_tokens": 0,
        "terminal_eos_per_visible_document": 1,
        "auxiliary_eos_either_arm": 0,
        "geometry_seed": GEOMETRY_SEED, "geometry_max_targets_per_group": GEOMETRY_PER_GROUP,
        "geometry_order": "canonical TRAIN group, increasing original mask-selected event ordinal",
        "geometry_target_count": geometry_begin,
        "arms": arms,
        "files": files,
    });
    let mut out = create(&out, "receipt.json")?;
    serde_json::to_writer_pretty(&mut out, &receipt)?;
    out.write_all(b"\n")?;
    out.flush()?;
    eprintln!("verified and prepared {group_count} paired TRAIN groups, {turn_count} assistant turns, {target_begin} full and {geometry_begin} geometry targets per arm");
    Ok(())
}
