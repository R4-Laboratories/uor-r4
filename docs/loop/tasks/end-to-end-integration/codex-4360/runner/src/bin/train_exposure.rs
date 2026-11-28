#![forbid(unsafe_code)]
use std::{
    env,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

const BOS: u16 = 0;
const EOS: u16 = 1;
const PROMPT: u8 = 1;
const ANSWER: u8 = 3;
const CHUNK_IDS: u64 = 257;

#[derive(Clone, Copy)]
struct Opening {
    response: u64,
    first_prompt: u64,
    own_prompt: u64,
    turn: u64,
}

#[derive(Default)]
struct Openings {
    count: u64,
    first_chunk: u64,
    later_chunk: u64,
    second_or_later_turn: u64,
    at_dropped_boundary: u64,
    same_chunk_own_prompt: u64,
    same_chunk_first_and_own_prompt: u64,
    same_chunk_first_and_own_prompt_later_turn: u64,
}
impl Openings {
    fn add(&mut self, row: Opening) {
        self.count += 1;
        let chunk = |offset| offset / CHUNK_IDS;
        let opening_chunk = chunk(row.response);
        self.first_chunk += u64::from(opening_chunk == 0);
        self.later_chunk += u64::from(opening_chunk != 0);
        self.second_or_later_turn += u64::from(row.turn > 1);
        self.at_dropped_boundary += u64::from(row.response % CHUNK_IDS == 0);
        self.same_chunk_own_prompt += u64::from(chunk(row.own_prompt) == opening_chunk);
        let both =
            chunk(row.first_prompt) == opening_chunk && chunk(row.own_prompt) == opening_chunk;
        self.same_chunk_first_and_own_prompt += u64::from(both);
        self.same_chunk_first_and_own_prompt_later_turn += u64::from(both && row.turn > 1);
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn main() -> io::Result<()> {
    let dir = env::args()
        .nth(1)
        .ok_or_else(|| invalid("missing directory"))?;
    let file = |name: &str| File::open(Path::new(&dir).join(name));
    let mut tokens = BufReader::new(file("train-100m.tokens.u16")?);
    let mut mask = BufReader::new(file("train-100m.mask.u8")?);
    let mut segments = BufReader::new(file("train-100m.segment.u8")?);
    let count = mask.get_ref().metadata()?.len();
    if tokens.get_ref().metadata()?.len() != 64 + 2 * count
        || segments.get_ref().metadata()?.len() != count
    {
        return Err(invalid("unaligned train streams"));
    }
    let mut header = [0u8; 64];
    tokens.read_exact(&mut header)?;
    if &header[..4] != b"UORT" {
        return Err(invalid("invalid UORT stream"));
    }

    let mut records = 0;
    let mut offset = 0;
    let mut turn = 0;
    let mut first_prompt = None;
    let mut own_prompt = None;
    let mut last_opening = None;
    let mut previous_mask = 0;
    let mut previous_segment = 0;
    let mut all = Openings::default();
    let mut initial = Openings::default();
    let mut last = Openings::default();
    let mut answer_or_eos = 0;
    let mut answer_or_eos_boundary = 0;
    let mut answer_content_boundary = 0;
    let mut eos_boundary = 0;
    let mut non_bos_boundary = 0;
    let mut completed_in_opening_chunk = 0;
    let mut completed_with_own_prompt = 0;
    let mut completed_with_both_prompts = 0;
    let mut completed_later_turn_with_both_prompts = 0;
    let mut multiturn_records = 0;

    let mut tb = vec![0; 2 * 65536];
    let mut mb = vec![0; 65536];
    let mut sb = vec![0; 65536];
    let mut remain = count;
    while remain > 0 {
        let size = remain.min(mb.len() as u64) as usize;
        tokens.read_exact(&mut tb[..2 * size])?;
        mask.read_exact(&mut mb[..size])?;
        segments.read_exact(&mut sb[..size])?;
        for ((pair, &m), &s) in tb[..2 * size]
            .chunks_exact(2)
            .zip(&mb[..size])
            .zip(&sb[..size])
        {
            let id = u16::from_le_bytes([pair[0], pair[1]]);
            if m > 1 || s > 3 || id >= 4096 {
                return Err(invalid("invalid token, mask or segment"));
            }
            if id == BOS {
                if m != 0 || s != 0 || (records > 0 && previous_mask != 1) {
                    return Err(invalid("invalid BOS boundary"));
                }
                if let Some(row) = last_opening.take() {
                    last.add(row);
                }
                multiturn_records += u64::from(turn > 1);
                records += 1;
                offset = 0;
                turn = 0;
                first_prompt = None;
                own_prompt = None;
            } else {
                if records == 0 {
                    return Err(invalid("missing initial BOS"));
                }
                offset += 1;
                let boundary = offset % CHUNK_IDS == 0;
                non_bos_boundary += u64::from(boundary);
                if s == PROMPT && previous_segment != PROMPT {
                    first_prompt.get_or_insert(offset);
                    own_prompt = Some(offset);
                }
                if m == 1 && previous_mask == 0 {
                    if s != ANSWER || id == EOS {
                        return Err(invalid("response rising edge is not answer content"));
                    }
                    turn += 1;
                    let row = Opening {
                        response: offset,
                        first_prompt: first_prompt
                            .ok_or_else(|| invalid("missing first prompt"))?,
                        own_prompt: own_prompt
                            .take()
                            .ok_or_else(|| invalid("missing prompt for opening"))?,
                        turn,
                    };
                    all.add(row);
                    if turn == 1 {
                        initial.add(row);
                    }
                    last_opening = Some(row);
                }
                if m == 1 {
                    answer_or_eos += 1;
                    answer_or_eos_boundary += u64::from(boundary);
                    if id == EOS {
                        eos_boundary += u64::from(boundary);
                        let row =
                            last_opening.ok_or_else(|| invalid("EOS before answer opening"))?;
                        let chunk = |offset| offset / CHUNK_IDS;
                        let complete =
                            row.response % CHUNK_IDS != 0 && chunk(row.response) == chunk(offset);
                        let with_own = complete && chunk(row.own_prompt) == chunk(offset);
                        let with_both = with_own && chunk(row.first_prompt) == chunk(offset);
                        completed_in_opening_chunk += u64::from(complete);
                        completed_with_own_prompt += u64::from(with_own);
                        completed_with_both_prompts += u64::from(with_both);
                        completed_later_turn_with_both_prompts +=
                            u64::from(with_both && row.turn > 1);
                    } else if s == ANSWER {
                        answer_content_boundary += u64::from(boundary);
                    } else {
                        return Err(invalid("unexpected supervised segment"));
                    }
                }
            }
            previous_mask = m;
            previous_segment = s;
        }
        remain -= size as u64;
    }
    if let Some(row) = last_opening {
        last.add(row);
    }
    multiturn_records += u64::from(turn > 1);
    println!("train_ids={count} records={records} multi_turn_records={multiturn_records}");
    println!("all_non_bos_targets_lost_at_boundaries={non_bos_boundary} train_answer_or_eos_ids={answer_or_eos}");
    println!("answer_or_eos_targets_lost_at_boundaries={answer_or_eos_boundary} answer_content_lost={answer_content_boundary} eos_lost={eos_boundary}");
    for (name, stats) in [("all", all), ("initial", initial), ("final", last)] {
        println!("{name}_openings={} first_chunk={} later_chunk={} later_turn={} opening_lost={} own_prompt_same_chunk={} first_and_own_prompt_same_chunk={} first_and_own_prompt_same_chunk_later_turn={}",
            stats.count, stats.first_chunk, stats.later_chunk, stats.second_or_later_turn, stats.at_dropped_boundary, stats.same_chunk_own_prompt, stats.same_chunk_first_and_own_prompt, stats.same_chunk_first_and_own_prompt_later_turn);
    }
    println!("complete_answer_eos_same_chunk={completed_in_opening_chunk} complete_answer_eos_own_prompt_same_chunk={completed_with_own_prompt} complete_answer_eos_both_prompts_same_chunk={completed_with_both_prompts} complete_later_answer_eos_both_prompts_same_chunk={completed_later_turn_with_both_prompts}");
    Ok(())
}
