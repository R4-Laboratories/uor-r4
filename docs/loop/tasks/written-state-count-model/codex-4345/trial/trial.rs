//! Prepare label-blind client inputs, serialize the two GPU panels, and grade replies.
#![forbid(unsafe_code)]
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
mod outcomes;
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const EOS: u32 = 1;
const CAP: usize = 192;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];

fn check(ok: bool, msg: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(msg.into())
    }
}
fn rows<T: DeserializeOwned>(file: &Path) -> Result<Vec<T>> {
    BufReader::new(fs::File::open(file)?)
        .lines()
        .map(|x| Ok(serde_json::from_str(&x?)?))
        .collect()
}
fn output(file: &Path) -> Result<BufWriter<fs::File>> {
    Ok(BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(file)?,
    ))
}
fn line(w: &mut impl Write, v: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *w, v)?;
    w.write_all(b"\n")?;
    Ok(())
}
#[derive(Deserialize, Clone)]
struct Event {
    record: String,
    turn: usize,
    prefix: Vec<u32>,
    opening_len: usize,
    completed_answer_ids: usize,
    target: u32,
    component: String,
    has_history: bool,
    last_char_class: String,
    payload_offset: usize,
}
#[derive(Deserialize)]
struct Pair {
    split: String,
    stop: Event,
    continuation: Event,
}
#[derive(Deserialize)]
struct Truth {
    user: Option<String>,
    record: String,
    turn: usize,
    answer_ids: Vec<u32>,
    answer_start: usize,
    opening_start: usize,
    answer_end: usize,
    eos: u32,
    eos_supervised: bool,
}

fn prepare(pairs: &Path, truth: &Path, dst: &Path) -> Result<()> {
    fs::create_dir(dst)?;
    let pairs: Vec<Pair> = rows(pairs)?;
    let truths: Vec<Truth> = rows(truth)?;
    let mut answers = BTreeMap::new();
    for t in truths {
        check(
            t.eos == EOS
                && t.answer_end.checked_sub(t.answer_start) == Some(t.answer_ids.len())
                && !t.answer_ids.contains(&EOS),
            "invalid reference EOS or offsets",
        )?;
        check(
            answers.insert((t.record.clone(), t.turn), t).is_none(),
            "duplicate truth",
        )?;
    }
    let mut components = BTreeMap::new();
    for p in &pairs {
        check(
            matches!(p.split.as_str(), "train" | "dev" | "held"),
            "unknown partition",
        )?;
        check(
            p.stop.component != p.continuation.component
                && p.stop.prefix != p.continuation.prefix
                && p.stop.target == EOS
                && p.continuation.target != EOS
                && p.stop.opening_len / 8 == p.continuation.opening_len / 8
                && p.stop.has_history == p.continuation.has_history
                && p.stop.last_char_class == p.continuation.last_char_class
                && p.stop.completed_answer_ids == p.continuation.completed_answer_ids,
            "pair violates matched conditions",
        )?;
        for e in [&p.stop, &p.continuation] {
            let prior_split = components.insert(e.component.clone(), &p.split);
            check(
                prior_split.is_none_or(|s| s == &p.split),
                "component crosses splits",
            )?;
        }
    }
    for split in ["train", "held"] {
        let mut boundary = output(&dst.join(format!("{split}.prefix.jsonl")))?;
        let mut labels = output(&dst.join(format!("{split}.truth.jsonl")))?;
        let mut open = output(&dst.join(format!("{split}.open.jsonl")))?;
        let mut refs = output(&dst.join(format!("{split}.answers.jsonl")))?;
        let mut turns = BTreeMap::new();
        let mut count = 0;
        for p in pairs.iter().filter(|p| p.split == split) {
            for e in [&p.stop, &p.continuation] {
                let key = (e.record.clone(), e.turn);
                let t = answers.get(&key).ok_or("missing truth")?;
                let n = e.opening_len;
                check(
                    n >= MARKER.len()
                        && n <= e.prefix.len()
                        && e.prefix[..n].ends_with(&MARKER)
                        && t.answer_start.checked_sub(t.opening_start) == Some(n)
                        && t.answer_start.checked_add(e.completed_answer_ids)
                            == Some(e.payload_offset)
                        && e.prefix.len() == n + e.completed_answer_ids
                        && e.prefix.len() < CAP
                        && t.answer_ids.get(..e.completed_answer_ids) == Some(&e.prefix[n..]),
                    "boundary not the complete original causal prefix",
                )?;
                let target = t
                    .answer_ids
                    .get(e.completed_answer_ids)
                    .copied()
                    .unwrap_or(t.eos);
                check(
                    target == e.target && (target != EOS || t.eos_supervised),
                    "target identity or mask",
                )?;
                let id = if let Some(id) = turns.get(&key) {
                    *id
                } else {
                    let id = turns.len();
                    line(
                        &mut open,
                        &json!({"index":id,"prefix":&e.prefix[..n],"user":if e.has_history {None} else {Some(t.user.as_ref().ok_or("missing first user text")?)} }),
                    )?;
                    line(
                        &mut refs,
                        &json!({"index":id,"answer_ids":t.answer_ids,"eos_supervised":t.eos_supervised,"has_history":e.has_history}),
                    )?;
                    turns.insert(key, id);
                    id
                };
                line(&mut boundary, &json!({"prefix":e.prefix}))?;
                line(
                    &mut labels,
                    &json!({"index":count,"turn_index":id,"target":e.target,"last_char_class":e.last_char_class}),
                )?;
                count += 1;
            }
        }
        check(count >= 2 && count % 2 == 0, "empty strict split")?;
        boundary.flush()?;
        labels.flush()?;
        open.flush()?;
        refs.flush()?;
    }
    Ok(())
}
#[derive(Deserialize)]
struct Captured {
    ordinal: usize,
    prefix_sha256: String,
    geometry_keys: Vec<u16>,
    ordered_keys: Vec<u16>,
    ordinary_id: u16,
    prefix_len: usize,
}
fn wire(stage: &Path, split: &str, captures: &Path, arm: &str, dest: &Path) -> Result<()> {
    check(
        matches!(split, "train" | "held") && matches!(arm, "geometry" | "ordered"),
        "bad wire mode",
    )?;
    let cf: Vec<Captured> = rows(captures)?;
    let labels: Vec<Value> = rows(&stage.join(format!("{split}.truth.jsonl")))?;
    let prefixes: Vec<Value> = rows(&stage.join(format!("{split}.prefix.jsonl")))?;
    check(
        cf.len() == labels.len()
            && cf.len() == prefixes.len()
            && !cf.is_empty()
            && cf.len() % 2 == 0,
        "capture size mismatch",
    )?;
    let mut w = output(dest)?;
    w.write_all(if split == "train" {
        b"URELPAI1"
    } else {
        b"URELEVT1"
    })?;
    w.write_all(&u32::try_from(cf.len())?.to_le_bytes())?;
    for (i, c) in cf.iter().enumerate() {
        let prefix: Vec<u32> = serde_json::from_value(prefixes[i]["prefix"].clone())?;
        let mut hash = Sha256::new();
        for id in prefix {
            hash.update(id.to_le_bytes());
        }
        check(
            c.ordinal == i && c.prefix_sha256 == format!("{:x}", hash.finalize()),
            "wrong exact causal ID sequence",
        )?;
        let label = labels[i]["target"].as_u64().ok_or("target")? == u64::from(EOS);
        check(
            label == (i % 2 == 0)
                && labels[i]["index"] == i
                && prefixes[i]["prefix"]
                    .as_array()
                    .is_some_and(|v| v.len() == c.prefix_len),
            "row order or prefix length",
        )?;
        let keys = if arm == "geometry" {
            &c.geometry_keys
        } else {
            &c.ordered_keys
        };
        check(
            keys.len() == 16
                && keys
                    .iter()
                    .enumerate()
                    .all(|(b, k)| usize::from(*k) / 256 == b),
            "bank keys",
        )?;
        for k in keys {
            w.write_all(&k.to_le_bytes())?;
        }
        w.write_all(&c.ordinary_id.to_le_bytes())?;
        w.write_all(&[u8::from(split == "train" && label), 0])?;
    }
    w.flush()?;
    Ok(())
}

#[derive(Deserialize)]
struct Client {
    artifact_sha256: String,
    generated_ids: Vec<u32>,
    prompt_ids: Vec<u32>,
}
fn divergence(reference: &[u32], emitted: &[u32]) -> Option<(usize, Option<u32>, Option<u32>)> {
    (0..reference.len().max(emitted.len()))
        .find(|&i| reference.get(i) != emitted.get(i))
        .map(|i| (i + 1, reference.get(i).copied(), emitted.get(i).copied()))
}
// Incomplete UTF-8 at a token boundary contributes no visible character yet.
// A complete invalid sequence in generated bytes renders as a replacement character.
fn sentence_end(mut bytes: &[u8]) -> bool {
    let mut last = None;
    loop {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                last = valid.chars().next_back().or(last);
                break;
            }
            Err(error) => {
                let valid =
                    std::str::from_utf8(&bytes[..error.valid_up_to()]).expect("valid prefix");
                last = valid.chars().next_back().or(last);
                let Some(bad) = error.error_len() else {
                    break;
                };
                last = Some(char::REPLACEMENT_CHARACTER);
                bytes = &bytes[error.valid_up_to() + bad..];
            }
        }
    }
    matches!(last, Some('.' | '?' | '!'))
}

fn fixed_stop(tok: &ByteBpeTokenizer, ids: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut visible = Vec::new();
    for &id in ids {
        if sentence_end(&visible) {
            out.push(EOS);
            break;
        }
        out.push(id);
        if id == EOS {
            break;
        }
        visible.extend(tok.decode_bytes(&[id]));
    }
    out
}
fn grade(
    stage: &Path,
    records: &Path,
    artifact: &str,
    tok: &Path,
    fixed: bool,
    dest: &Path,
) -> Result<()> {
    let opens: Vec<Value> = rows(&stage.join("held.open.jsonl"))?;
    let refs: Vec<Value> = rows(&stage.join("held.answers.jsonl"))?;
    check(opens.len() == refs.len(), "truth rows")?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tok)?).ok_or("tokenizer")?;
    let mut seen = BTreeSet::new();
    let mut out = output(dest)?;
    for (i, (op, r)) in opens.iter().zip(refs).enumerate() {
        check(
            op["index"] == i && r["index"] == i && seen.insert(i),
            "wrong opening",
        )?;
        let c: Client = serde_json::from_slice(&fs::read(records.join(format!("{i}.json")))?)?;
        check(
            c.artifact_sha256 == artifact && serde_json::to_value(&c.prompt_ids)? == op["prefix"],
            "client identity",
        )?;
        check(
            !c.generated_ids.is_empty()
                && c.generated_ids.len() <= CAP
                && match c.generated_ids.iter().position(|x| *x == EOS) {
                    Some(p) => p + 1 == c.generated_ids.len(),
                    None => c.generated_ids.len() == CAP,
                },
            "client must finish on actual EOS or cap",
        )?;
        let ids = if fixed {
            fixed_stop(&tok, &c.generated_ids)
        } else {
            c.generated_ids
        };
        let eos = ids.last() == Some(&EOS);
        let mut reference: Vec<u32> = serde_json::from_value(r["answer_ids"].clone())?;
        reference.push(EOS);
        let diff = divergence(&reference, &ids);
        let first = diff.map(
            |(step, expected, emitted)| json!({ "step":step,"expected":expected,"emitted":emitted}),
        );
        line(
            &mut out,
            &json!({"index":i,"observed_eos_supervised":r["eos_supervised"],"genuine_history":r["has_history"],
          "reference_ids":reference.len(),"over_cap":reference.len()>CAP,"generated":ids.len(),"eos":eos,
          "early_eos":eos && ids.len()<reference.len(),"late_eos":eos && ids.len()>reference.len(),
          "exact":diff.is_none() && eos,"first_wrong":first}),
        )?;
    }
    out.flush()?;
    Ok(())
}
fn main() -> Result<()> {
    let a = env::args().skip(1).collect::<Vec<_>>();
    let p = |i: usize| PathBuf::from(&a[i]);
    match a.first().map(String::as_str){
        Some("decisions") if a.len()==10 => outcomes::decisions(&p(1), &p(2), &p(3), &p(4), &p(5), &a[6], &a[7], &p(8), &p(9)),
        Some("prepare") if a.len()==4=>prepare(&p(1),&p(2),&p(3)),
        Some("wire") if a.len()==6=>wire(&p(1),&a[2],&p(3),&a[4],&p(5)),
        Some("grade") if a.len()==7=>grade(&p(1),&p(2),&a[3],&p(4),match a[5].as_str(){"fixed"=>true,"native"=>false,_=>return Err("grade arm".into())},&p(6)),
        _=>Err("trial prepare PAIRS TRUTH NEW_STAGE | wire STAGE train/held CAPTURE geometry/ordered NEW_BINARY | grade STAGE RECORDS ARTIFACT_SHA TOKENIZER fixed/native NEW_JSONL".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incomplete_utf8_does_not_hide_a_completed_period() {
        assert!(sentence_end(&[b'.', 0xc3]));
        assert!(!sentence_end(&[b'.', 0xc3, 0xa9]));
        assert!(!sentence_end(&[b'.', 0xff]));
        assert!(sentence_end(&[0xff, b'?']));
    }
    #[test]
    fn distinguishes_missing_id_wrong_id_and_exact() {
        assert_eq!(divergence(&[8, 9, 1], &[8, 1]), Some((2, Some(9), Some(1))));
        assert_eq!(divergence(&[8, 1], &[8]), Some((2, Some(1), None)));
        assert_eq!(divergence(&[8, 1], &[8, 1]), None);
    }
}
