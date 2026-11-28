#![forbid(unsafe_code)]

mod probe;

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
    ByteBpeTokenizer,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const WINDOW: usize = 192;
const EOS: u32 = 1;
const SPACE: u32 = 223;

#[derive(Deserialize)]
struct Record {
    id: String,
    messages: Vec<User>,
    turns: Vec<Reply>,
}

#[derive(Deserialize)]
struct User {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Reply {
    answer: String,
}

#[derive(Deserialize)]
struct Group {
    group: usize,
    canonical_train_group: usize,
    payload_start: usize,
    visible_end: usize,
    payload_end: usize,
}

#[derive(Deserialize, PartialEq, Eq)]
struct Opening {
    group: usize,
    turn: usize,
    prefix_start: usize,
    answer_start: usize,
    answer_end: usize,
    immediate_id: u32,
    lexical_id: u32,
    lexical_offset: usize,
    lexical_answer_offset: usize,
}

type Counts = BTreeMap<&'static str, usize>;

fn add(counts: &mut Counts, field: &'static str, amount: usize) {
    *counts.entry(field).or_default() += amount;
}

fn require(good: bool, message: &str) -> Result<()> {
    if !good {
        return Err(message.into());
    }
    Ok(())
}

fn rows<T: DeserializeOwned>(path: &Path) -> Result<impl Iterator<Item = Result<T>>> {
    Ok(BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?)))
}

fn group_window(file: &mut File, offset: usize, len: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; len];
    file.seek(SeekFrom::Start(u64::try_from(offset)?))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn tape_len(file: &mut File) -> Result<usize> {
    let header = group_window(file, 0, 64)?;
    let len = usize::try_from(u64::from_le_bytes(header[8..16].try_into()?))?;
    require(
        &header[..8] == b"UORT\x01\0\0\0"
            && &header[16..20] == 4096u32.to_le_bytes().as_slice()
            && header[20..].iter().all(|&byte| byte == 0)
            && file.metadata()?.len() == u64::try_from(64 + 2 * len)?,
        "invalid corrected donor UORT header",
    )?;
    Ok(len)
}

fn spaced(content: &str) -> String {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    format!(" {}", normalized.trim())
}

struct Turn<'a> {
    record: &'a str,
    ordinal: usize,
    index: usize,
    user: &'a str,
    answer: &'a str,
    history: &'a [Message<'a>],
    history_ids: &'a [u32],
    user_token_start: usize,
    prefix: &'a [u32],
    answer_ids: &'a [u32],
    terminal: bool,
}

fn visible_group(
    tokenizer: &ByteBpeTokenizer,
    encoder: &DialogueEncoder<'_>,
    census: &mut probe::Census,
    group: &Group,
    record: &Record,
    ids: &[u32],
    mask: &[u8],
    openings: &mut impl Iterator<Item = Result<Opening>>,
) -> Result<()> {
    let mut history = Vec::new();
    let mut prior_end = 1;
    for (index, (user, answer)) in record.messages.iter().zip(&record.turns).enumerate() {
        require(user.role == "user", "unexpected D87 source role")?;
        let opening = openings.next().ok_or("missing sealed opening")??;
        let earlier = encoder.encode_open_history(&history);
        let request = encoder.encode_user_prefix(&user.content, index != 0);
        let source = tokenizer.encode(&spaced(&user.content));
        let prefix = [&earlier.tokens[..], &request.tokens[..]].concat();
        let (source_end, source_start) = prefix
            .len()
            .checked_sub(MARKER.len())
            .and_then(|end| end.checked_sub(source.len()).map(|start| (end, start)))
            .ok_or("source outside the original prefix")?;
        let answer_bytes = spaced(&answer.answer);
        let answer_ids = tokenizer.encode(&answer_bytes);
        let turn = opening_turn(tokenizer, &answer_ids, answer_bytes.as_bytes())?;
        let start = prefix.len();
        let end = start + answer_ids.len();
        let terminal = index + 1 == record.turns.len();
        require(
            earlier.tokens.len() == prior_end
                && request.emitted_turns == 1
                && request.special_token_occurrences == 0
                && prefix.ends_with(&MARKER)
                && prefix[source_start..source_end] == source
                && tokenizer.decode_bytes(&source) == spaced(&user.content).as_bytes()
                && ids.starts_with(&prefix)
                && mask
                    .get(prior_end..start)
                    .is_some_and(|m| m.iter().all(|&n| n == 0))
                && ids.get(start..end) == Some(answer_ids.as_slice())
                && mask
                    .get(start..end)
                    .is_some_and(|m| m.iter().all(|&n| n == 1))
                && ids.get(end) == Some(&EOS)
                && mask.get(end) == Some(&u8::from(terminal))
                && opening
                    == (Opening {
                        group: group.group,
                        turn: index,
                        prefix_start: group.payload_start,
                        answer_start: group.payload_start + start,
                        answer_end: group.payload_start + end,
                        immediate_id: answer_ids[0],
                        lexical_id: answer_ids[turn],
                        lexical_offset: group.payload_start + start + turn,
                        lexical_answer_offset: turn,
                    }),
            "visible prefix, original answer, opening, or EOS mask differs from #4314",
        )?;
        let turn = Turn {
            record: &record.id,
            ordinal: group.canonical_train_group,
            index,
            user: &user.content,
            answer: &answer.answer,
            history: &history,
            history_ids: &earlier.tokens,
            user_token_start: source_start,
            prefix: &prefix,
            answer_ids: &answer_ids,
            terminal,
        };
        census.scan(tokenizer, &turn, &source)?;
        prior_end = end + 1;
        history.extend([
            Message {
                role: "user",
                content: &user.content,
            },
            Message {
                role: "assistant",
                content: &answer.answer,
            },
        ]);
    }
    require(
        ids.len() == prior_end && ids[0] == 0 && mask[0] == 0,
        "visible group range differs",
    )
}

fn opening_turn(tokenizer: &ByteBpeTokenizer, ids: &[u32], answer: &[u8]) -> Result<usize> {
    require(
        !ids.iter().any(|&id| id <= 2) && tokenizer.decode_bytes(ids) == answer,
        "visible answer encoding cannot be replayed exactly",
    )?;
    ids.iter()
        .position(|id| {
            tokenizer
                .decode_bytes(&[*id])
                .iter()
                .any(|byte| !byte.is_ascii_whitespace())
        })
        .ok_or_else(|| "empty visible answer".into())
}

fn replay(token: &Path, donor: &Path, records: &Path, review: &Path) -> Result<Value> {
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(token)?)
        .ok_or("invalid product tokenizer")?;
    require(
        tokenizer.vocab_size() == 4096
            && tokenizer.encode(" ") == [SPACE]
            && [tokenizer.encode("\n"), tokenizer.encode("Assistant:")].concat() == MARKER,
        "unsupported product tokenizer",
    )?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let mut ids = File::open(donor.join("answer-first.tokens.u16"))?;
    let mut masks = File::open(donor.join("answer-first.mask.u8"))?;
    let payload_len = tape_len(&mut ids)?;
    require(
        masks.metadata()?.len() == u64::try_from(payload_len)?,
        "mask size differs",
    )?;
    let mut records = rows::<Record>(records)?;
    let mut openings = rows::<Opening>(&donor.join("openings.jsonl"))?;
    let mut count = 0;
    let mut next_group = 0;
    let mut visible_len = 0;
    let mut seen = BTreeSet::new();
    let mut census = probe::Census::default();
    for group in rows::<Group>(&donor.join("groups.jsonl"))? {
        let group = group?;
        let record = records.next().ok_or("missing sealed visible record")??;
        require(
            group.group == count
                && group.canonical_train_group == 128 * count
                && group.payload_start == next_group
                && group.payload_start < group.visible_end
                && group.visible_end <= group.payload_end
                && group.payload_end <= payload_len
                && (1..=2).contains(&record.turns.len())
                && record.messages.len() == record.turns.len()
                && seen.insert(record.id.clone()),
            "invalid visible group membership",
        )?;
        let len = group.visible_end - group.payload_start;
        let bytes = group_window(&mut ids, 64 + 2 * group.payload_start, 2 * len)?;
        let original_ids: Vec<_> = bytes
            .chunks_exact(2)
            .map(|pair| u32::from(u16::from_le_bytes([pair[0], pair[1]])))
            .collect();
        let visible_mask = group_window(&mut masks, group.payload_start, len)?;
        visible_group(
            &tokenizer,
            &encoder,
            &mut census,
            &group,
            &record,
            &original_ids,
            &visible_mask,
            &mut openings,
        )?;
        count += 1;
        next_group = group.payload_end;
        visible_len += len;
    }
    require(
        count == 4_060
            && next_group == payload_len
            && visible_len == 5_731_000
            && openings.next().is_none()
            && records.next().is_none()
            && census.counts.get("openings") == Some(&4_345)
            && census.counts.get("visible_answer_ids") == Some(&3_763_868)
            && census.counts.get("supervised_eos") == Some(&4_060)
            && census.counts.get("visible_history_eos") == Some(&285),
        "fixed #4314 visible population differs",
    )?;
    let mut private = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(review)?,
    );
    let matching = census.finish(&mut private)?;
    private.flush()?;
    Ok(json!({
        "schema": "uor-r4.train-copy-release-inventory/1",
        "groups": count,
        "visible_token_count": visible_len,
        "counts": census.counts,
        "matched": matching,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [tokenizer, donor, records, review] = args.as_slice() else {
        return Err(
            "expected tokenizer, DONOR directory, records JSONL and new REVIEW file".into(),
        );
    };
    let summary = replay(
        Path::new(tokenizer),
        Path::new(donor),
        Path::new(records),
        Path::new(review),
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
