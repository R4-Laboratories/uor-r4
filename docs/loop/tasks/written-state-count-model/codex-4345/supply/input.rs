use crate::{require, Result, EOS};
use serde::{de::DeserializeOwned, Deserialize};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::Path,
};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
    ByteBpeTokenizer,
};

pub const GROUP_COUNT: usize = 4_060;
pub const TURN_COUNT: usize = 4_345;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const HEADER: usize = 64;
const VCB: u32 = 4096;

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

pub struct RecordKey {
    pub id: String,
    pub ordinal: usize,
}

pub struct Turn {
    pub record: usize,
    pub ordinal: usize,
    pub user: String,
    pub opening_offset: usize,
    pub opening: Vec<u32>,
    pub answer: Vec<u32>,
    pub answer_text: String,
    pub eos_supervised: bool,
}

pub struct Corpus {
    pub records: Vec<RecordKey>,
    pub turns: Vec<Turn>,
    pub visible_tokens: usize,
    pub token_bytes: Vec<Vec<u8>>,
}

fn rows<T: DeserializeOwned>(path: &Path) -> Result<impl Iterator<Item = Result<T>>> {
    Ok(BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?)))
}

fn read_window(file: &mut File, offset: usize, len: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; len];
    file.seek(SeekFrom::Start(u64::try_from(offset)?))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn tape_len(file: &mut File) -> Result<usize> {
    let header = read_window(file, 0, HEADER)?;
    let len = usize::try_from(u64::from_le_bytes(header[8..16].try_into()?))?;
    let expected_bytes = len
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(HEADER))
        .ok_or("UORT size overflow")?;
    require(
        &header[..8] == b"UORT\x01\0\0\0"
            && header[16..20] == VCB.to_le_bytes()
            && header[20..].iter().all(|&byte| byte == 0)
            && file.metadata()?.len() == u64::try_from(expected_bytes)?,
        "invalid corrected donor UORT header",
    )?;
    Ok(len)
}

fn spaced(content: &str) -> String {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    format!(" {}", normalized.trim())
}

struct VisibleGroup<'a> {
    group: &'a Group,
    record: &'a Record,
    ids: &'a [u32],
    mask: &'a [u8],
}

fn replay_group(
    tokenizer: &ByteBpeTokenizer,
    encoder: &DialogueEncoder<'_>,
    input: VisibleGroup<'_>,
    openings: &mut impl Iterator<Item = Result<Opening>>,
    turns: &mut Vec<Turn>,
) -> Result<()> {
    let VisibleGroup {
        group,
        record,
        ids,
        mask,
    } = input;
    require(
        ids.first() == Some(&0) && mask.first() == Some(&0),
        "missing visible BOS",
    )?;
    let mut history = Vec::new();
    let mut prior_end = 1;
    for (index, (user, reply)) in record.messages.iter().zip(&record.turns).enumerate() {
        require(user.role == "user", "unexpected D87 source role")?;
        let saved = openings.next().ok_or("missing sealed opening")??;
        let earlier = encoder.encode_open_history(&history);
        let request = encoder.encode_user_prefix(&user.content, index != 0);
        let opening = [&earlier.tokens[..], &request.tokens[..]].concat();
        let answer_text = spaced(&reply.answer);
        let answer = tokenizer.encode(&answer_text);
        let lexical = answer
            .iter()
            .position(|id| {
                tokenizer
                    .decode_bytes(&[*id])
                    .iter()
                    .any(|byte| !byte.is_ascii_whitespace())
            })
            .ok_or("empty visible answer")?;
        let start = opening.len();
        let end = start + answer.len();
        let eos_supervised = index + 1 == record.turns.len();
        require(
            earlier.tokens.len() == prior_end
                && earlier.special_token_occurrences == 0
                && request.emitted_turns == 1
                && request.special_token_occurrences == 0
                && opening.ends_with(&MARKER)
                && ids.starts_with(&opening)
                && mask
                    .get(prior_end..start)
                    .is_some_and(|bytes| bytes.iter().all(|&b| b == 0))
                && ids.get(start..end) == Some(answer.as_slice())
                && !answer.iter().any(|&id| id <= 2)
                && tokenizer.decode_bytes(&answer) == answer_text.as_bytes()
                && mask
                    .get(start..end)
                    .is_some_and(|bytes| bytes.iter().all(|&b| b == 1))
                && ids.get(end) == Some(&EOS)
                && mask.get(end) == Some(&u8::from(eos_supervised))
                && saved
                    == (Opening {
                        group: group.group,
                        turn: index,
                        prefix_start: group.payload_start,
                        answer_start: group.payload_start + start,
                        answer_end: group.payload_start + end,
                        immediate_id: answer[0],
                        lexical_id: answer[lexical],
                        lexical_offset: group.payload_start + start + lexical,
                        lexical_answer_offset: lexical,
                    }),
            "ordinary protocol opening, original answer, offsets or mask disagree with #4314",
        )?;
        turns.push(Turn {
            record: group.group,
            ordinal: index,
            user: user.content.clone(),
            opening_offset: group.payload_start,
            opening,
            answer,
            answer_text,
            eos_supervised,
        });
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
        prior_end = end + 1;
    }
    require(
        prior_end == ids.len(),
        "corrected visible group range differs",
    )
}

pub fn replay(tokenizer: &Path, donor: &Path, records: &Path) -> Result<Corpus> {
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or("invalid product tokenizer")?;
    require(
        tokenizer.vocab_size() == VCB as usize
            && tokenizer.encode(" ") == [223]
            && [tokenizer.encode("\n"), tokenizer.encode("Assistant:")].concat() == MARKER,
        "unsupported product tokenizer",
    )?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let mut tape = File::open(donor.join("answer-first.tokens.u16"))?;
    let mut masks = File::open(donor.join("answer-first.mask.u8"))?;
    let payload_len = tape_len(&mut tape)?;
    require(
        masks.metadata()?.len() == u64::try_from(payload_len)?,
        "mask size differs",
    )?;
    let mut source = rows::<Record>(records)?;
    let mut openings = rows::<Opening>(&donor.join("openings.jsonl"))?;
    let mut records = Vec::with_capacity(GROUP_COUNT);
    let mut turns = Vec::with_capacity(TURN_COUNT);
    let mut next_group = 0;
    let mut visible_tokens = 0;
    let mut seen = BTreeSet::new();
    for group in rows::<Group>(&donor.join("groups.jsonl"))? {
        let group = group?;
        let record = source.next().ok_or("missing sealed visible record")??;
        require(
            group.group == records.len()
                && group.canonical_train_group == 128 * group.group
                && group.payload_start == next_group
                && group.payload_start < group.visible_end
                && group.visible_end <= group.payload_end
                && group.payload_end <= payload_len
                && (1..=2).contains(&record.turns.len())
                && record.messages.len() == record.turns.len()
                && seen.insert(record.id.clone()),
            "invalid corrected visible group membership",
        )?;
        let len = group.visible_end - group.payload_start;
        let bytes = read_window(&mut tape, HEADER + 2 * group.payload_start, 2 * len)?;
        let ids: Vec<_> = bytes
            .chunks_exact(2)
            .map(|pair| u32::from(u16::from_le_bytes([pair[0], pair[1]])))
            .collect();
        let mask = read_window(&mut masks, group.payload_start, len)?;
        replay_group(
            &tokenizer,
            &encoder,
            VisibleGroup {
                group: &group,
                record: &record,
                ids: &ids,
                mask: &mask,
            },
            &mut openings,
            &mut turns,
        )?;
        records.push(RecordKey {
            id: record.id,
            ordinal: group.canonical_train_group,
        });
        next_group = group.payload_end;
        visible_tokens += len;
    }
    let total_answers = turns.iter().map(|turn| turn.answer.len()).sum::<usize>();
    require(
        records.len() == GROUP_COUNT
            && turns.len() == TURN_COUNT
            && visible_tokens == 5_731_000
            && total_answers == 3_763_868
            && turns.iter().filter(|turn| turn.eos_supervised).count() == GROUP_COUNT
            && next_group == payload_len
            && openings.next().is_none()
            && source.next().is_none(),
        "fixed #4314 visible population differs",
    )?;
    let token_bytes = (0..VCB).map(|id| tokenizer.decode_bytes(&[id])).collect();
    Ok(Corpus {
        records,
        turns,
        visible_tokens,
        token_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn ordinary_replay_refuses_a_missing_prior_assistant_eos() -> Result<()> {
        let path =
            env::var_os("UOR_SUPPLY_TEST_TOKENIZER").ok_or("missing bound test tokenizer")?;
        let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(path)?)
            .ok_or("invalid bound test tokenizer")?;
        let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
        let msgs = [
            Message {
                role: "user",
                content: "What did the dog find?",
            },
            Message {
                role: "assistant",
                content: "A stick.",
            },
            Message {
                role: "user",
                content: "Where was it?",
            },
            Message {
                role: "assistant",
                content: "Under a tree.",
            },
        ];
        let record = Record {
            id: "fixture".to_owned(),
            messages: msgs
                .iter()
                .step_by(2)
                .map(|msg| User {
                    role: msg.role.to_owned(),
                    content: msg.content.to_owned(),
                })
                .collect(),
            turns: msgs
                .iter()
                .skip(1)
                .step_by(2)
                .map(|msg| Reply {
                    answer: msg.content.to_owned(),
                })
                .collect(),
        };
        let mut encoded = encoder.encode_document(&msgs);
        let eos_at = encoder.encode_open_history(&msgs[..2]).tokens.len() - 1;
        encoded.response_mask[eos_at] = 0;
        let group = Group {
            group: 0,
            canonical_train_group: 0,
            payload_start: 0,
            visible_end: encoded.tokens.len(),
            payload_end: encoded.tokens.len(),
        };
        let make_openings = || {
            (0..2).map(|turn| {
                let user = turn * 2;
                let prefix = [
                    &encoder.encode_open_history(&msgs[..user]).tokens[..],
                    &encoder
                        .encode_user_prefix(msgs[user].content, turn != 0)
                        .tokens[..],
                ]
                .concat();
                let answer = tokenizer.encode(&spaced(msgs[user + 1].content));
                let lexical = answer
                    .iter()
                    .position(|id| *id != 223)
                    .ok_or("fixture answer is whitespace")?;
                Ok(Opening {
                    group: 0,
                    turn,
                    prefix_start: 0,
                    answer_start: prefix.len(),
                    answer_end: prefix.len() + answer.len(),
                    immediate_id: answer[0],
                    lexical_id: answer[lexical],
                    lexical_offset: prefix.len() + lexical,
                    lexical_answer_offset: lexical,
                })
            })
        };
        let mut turns = Vec::new();
        replay_group(
            &tokenizer,
            &encoder,
            VisibleGroup {
                group: &group,
                record: &record,
                ids: &encoded.tokens,
                mask: &encoded.response_mask,
            },
            &mut make_openings(),
            &mut turns,
        )?;
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[1].opening[eos_at], EOS);
        assert!(!turns[0].eos_supervised && turns[1].eos_supervised);
        encoded.tokens.remove(eos_at);
        encoded.response_mask.remove(eos_at);
        assert!(replay_group(
            &tokenizer,
            &encoder,
            VisibleGroup {
                group: &group,
                record: &record,
                ids: &encoded.tokens,
                mask: &encoded.response_mask,
            },
            &mut make_openings(),
            &mut Vec::new()
        )
        .is_err());
        Ok(())
    }
}
