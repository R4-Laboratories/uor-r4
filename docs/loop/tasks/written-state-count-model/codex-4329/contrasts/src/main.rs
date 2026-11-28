//! Author-labelled contrast tape. No model scoring or training.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
use uor_r4_tokenizer::{
    dialogue::{DialogueProtocol, Message},
    ByteBpeTokenizer,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Family {
    id: String,
    split: String,
    people: [String; 2],
    values: [String; 2],
    asserted: String,
    denied: String,
    question: String,
    ask: String,
}
const INSTRUCTION: &str =
    " Reply with the word and a period. If the text doesn't tell you, reply unknown.";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [tokenizer, out] = args.as_slice() else {
        return Err("usage: whole-turn-contrasts TOKENIZER NEW_OUT".into());
    };
    let bytes = fs::read(tokenizer)?;
    if format!("{:x}", Sha256::digest(&bytes)) != TOKENIZER {
        return Err("tokenizer digest mismatch".into());
    }
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes).ok_or("invalid tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_version(&tokenizer, 2)?;
    let bound = protocol.bind(&tokenizer)?;
    let families: Vec<Family> = serde_json::from_str(include_str!("../families.json"))?;
    fs::create_dir(out)?;
    for split in ["train", "dev"] {
        let path = Path::new(out).join(format!("{split}.jsonl"));
        let mut file = BufWriter::new(File::create_new(path)?);
        let mut seen = HashSet::new();
        for f in families.iter().filter(|f| f.split == split) {
            for first in 0..2 {
                for assignment in [[0, 1], [1, 0], [0, 0]] {
                    for (status, negative) in [
                        ("asserted", None),
                        ("denied", Some(0)),
                        ("denied", Some(1)),
                        ("question", Some(0)),
                        ("question", Some(1)),
                    ] {
                        let mut facts = String::new();
                        let mut bounds = [(0, 0); 2];
                        for who in [first, 1 - first] {
                            let template = match (negative == Some(who), status) {
                                (true, "denied") => &f.denied,
                                (true, "question") => &f.question,
                                _ => &f.asserted,
                            };
                            if !facts.is_empty() {
                                facts.push(' ');
                            }
                            let clause = template.replace("{p}", &f.people[who]);
                            let (lead, tail) = clause
                                .split_once(" {v}")
                                .ok_or("missing value placeholder")?;
                            facts.push_str(lead);
                            let start = facts.len();
                            facts.push(' ');
                            facts.push_str(&f.values[assignment[who]]);
                            bounds[who] = (start, facts.len());
                            facts.push_str(tail);
                        }
                        for asked in 0..2 {
                            let prompt = format!(
                                "{} {}{}",
                                facts,
                                f.ask.replace("{p}", &f.people[asked]),
                                INSTRUCTION
                            );
                            let ids = bound
                                .encode_assistant_prefix(&[Message {
                                    role: "user",
                                    content: &prompt,
                                }])
                                .tokens;
                            if ids.len() + 4 > 192 || !seen.insert(ids.clone()) {
                                return Err("duplicate or overlong input".into());
                            }
                            let body = ids
                                .strip_prefix(&[0, 55, 2728, 28])
                                .and_then(|b| b.strip_suffix(&[201, 35, 560, 652, 714, 28]))
                                .ok_or("bad role framing")?;
                            let raw = format!(" {prompt}");
                            let mut offset = 0;
                            let mut positions = [None; 2];
                            for (i, &id) in body.iter().enumerate() {
                                let bytes = tokenizer.decode_bytes(&[id]);
                                let end = offset + bytes.len();
                                if raw.as_bytes().get(offset..end) != Some(&bytes) {
                                    return Err("body did not round-trip".into());
                                }
                                for (who, &(left, right)) in bounds.iter().enumerate() {
                                    if (left + 1, right + 1) == (offset, end) {
                                        positions[who] = Some((4 + i, id));
                                    }
                                }
                                offset = end;
                            }
                            if offset != raw.len() {
                                return Err("incomplete tokenizer coverage".into());
                            }
                            let [Some(a), Some(b)] = positions else {
                                return Err(format!("value in {} was split", f.id).into());
                            };
                            let sites = [a, b];
                            let supported = negative != Some(asked);
                            let expected = if supported {
                                &f.values[assignment[asked]]
                            } else {
                                "unknown"
                            };
                            let mut reply = tokenizer.encode(&format!(" {expected}."));
                            reply.push(1);
                            if reply.len() != 3
                                || reply[1] != 16
                                || (supported && reply[0] != sites[asked].1)
                            {
                                return Err("reply disagrees with source".into());
                            }
                            let row = json!({"family":f.id,"split":split,"fact_order":first,"value_assignment":assignment,"changed_side":negative,"statement_type":status,"asked":asked,
                                "prompt":prompt,"prefix":ids,"supported":supported,"source_pos":supported.then_some(sites[asked].0),"site_positions":[a.0,b.0],"site_ids":[a.1,b.1],"expected_ids":reply});
                            serde_json::to_writer(&mut file, &row)?;
                            writeln!(file)?;
                        }
                    }
                }
            }
        }
        file.flush()?;
    }
    Ok(())
}
