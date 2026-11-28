#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;
use std::process::Command;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SOURCE_RECEIPT: &[u8] = include_bytes!("../../../codex-4314/donor/RECEIPT.json");
const REGISTRATION: &[u8] = include_bytes!("../REGISTRATION.md");
const SOURCE_SHA: &str = "61ba19783fd29b39ee935e1dfa21ae14c5de1caddd3d380c7ed1bedb6b39f6e3";
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const MARKER: [u16; 6] = [201, 35, 560, 652, 714, 28];
const VOCABULARY: u16 = 4096;
const BOS: u16 = 0;
const EOS: u16 = 1;
const SPACE: u16 = 223;
const CAP: usize = 2048;
const GROUP_COUNT: usize = 4_060;
const TURN_COUNT: usize = 4_345;
const FILES: [&str; 3] = ["inputs.jsonl", "labels.jsonl", "refused.jsonl"];

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(reason.into())
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_binding(path: &Path) -> Result<Value> {
    let mut file = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let bytes = io::copy(&mut file, &mut hasher)?;
    Ok(json!({"bytes": bytes, "sha256": format!("{:x}", hasher.finalize())}))
}

fn read_rows<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|row| Ok(serde_json::from_str(&row?)?))
        .collect()
}

fn write_rows<T: Serialize>(out: &Path, rows: &[T]) -> Result<()> {
    let mut file = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(out)?);
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(())
}

fn verify_source(verifier: &Path, tokenizer: &Path, prior: &Path, donor: &Path) -> Result<Value> {
    let verifier_pin = file_binding(verifier)?;
    require(
        verifier_pin["sha256"]
            == "d25a69c86e7f23791c52c88a3b9bf93cc7ce7b09b0cf0a1be2b1b205fee2b13d",
        "wrong independent source verifier",
    )?;
    let tokenizer_pin = file_binding(tokenizer)?;
    require(tokenizer_pin["sha256"] == TOKENIZER_SHA, "wrong tokenizer")?;
    require(
        digest(SOURCE_RECEIPT) == SOURCE_SHA,
        "wrong registered donor receipt",
    )?;
    require(
        fs::read(donor.join("receipt.json"))? == SOURCE_RECEIPT,
        "wrong saved donor receipt",
    )?;

    let output = Command::new(verifier)
        .arg("verify")
        .arg(tokenizer)
        .arg(prior)
        .arg(donor)
        .output()?;
    require(
        output.status.success(),
        "independent #4314 source verifier failed",
    )?;
    require(
        output.stdout == SOURCE_RECEIPT,
        "source reconstruction or file hashes differ",
    )?;
    let receipt: Value = serde_json::from_slice(SOURCE_RECEIPT)?;
    Ok(json!({
        "canonical_readback_sha256": digest(&output.stdout),
        "independent_verifier": verifier_pin,
        "tokenizer": tokenizer_pin,
        "prior_receipt_sha256": receipt["source"]["historical_receipt_sha256"],
        "prior_structured_records_sha256": receipt["source"]["historical_records_sha256"],
        "prior_groups_sha256": receipt["source"]["historical_groups_sha256"],
        "protocol_identity": receipt["source"]["protocol_identity"],
        "source_files": receipt["files"],
    }))
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
struct TurnKey {
    group: usize,
    turn: usize,
}

#[derive(Deserialize)]
struct Opening {
    #[serde(flatten)]
    key: TurnKey,
    prefix_start: usize,
    answer_start: usize,
    answer_end: usize,
    immediate_id: u16,
    lexical_id: u16,
    lexical_offset: usize,
    lexical_answer_offset: usize,
}

#[derive(Deserialize)]
struct Group {
    group: usize,
    payload_start: usize,
    visible_end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Boundary {
    Immediate,
    StandaloneSpace,
    OtherPrecursor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Refusal {
    FullOpeningOverCap,
    LexicalPrefixOverCap,
    UnsupportedPrecursor,
}

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
struct Input {
    #[serde(flatten)]
    key: TurnKey,
    boundary: Boundary,
    full_opening: Vec<u16>,
    lexical_prefix: Vec<u16>,
}

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
struct Label {
    #[serde(flatten)]
    key: TurnKey,
    target: u16,
    lexical_payload_offset: usize,
}

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
struct Refused {
    #[serde(flatten)]
    key: TurnKey,
    boundary: Boundary,
    reason: Refusal,
    opening_len: usize,
    lexical_prefix_len: usize,
}

#[derive(Clone, Default, Serialize)]
struct Count {
    seen: usize,
    admitted: usize,
    refused: usize,
    immediate_seen: usize,
    immediate_admitted: usize,
    standalone_space_seen: usize,
    standalone_space_admitted: usize,
    other_precursor_seen: usize,
    masked_earlier_eos_seen: usize,
    masked_earlier_eos_admitted: usize,
    target_id_absent_from_lexical_prefix: usize,
    refusal_reasons: BTreeMap<Refusal, usize>,
}

impl Count {
    fn observe(
        &mut self,
        boundary: Boundary,
        earlier_eos: usize,
        refusal: Option<Refusal>,
        target_absent: bool,
    ) {
        self.seen += 1;
        self.masked_earlier_eos_seen += earlier_eos;
        self.immediate_seen += usize::from(boundary == Boundary::Immediate);
        self.standalone_space_seen += usize::from(boundary == Boundary::StandaloneSpace);
        self.other_precursor_seen += usize::from(boundary == Boundary::OtherPrecursor);
        if let Some(reason) = refusal {
            self.refused += 1;
            *self.refusal_reasons.entry(reason).or_default() += 1;
        } else {
            self.admitted += 1;
            self.masked_earlier_eos_admitted += earlier_eos;
            self.immediate_admitted += usize::from(boundary == Boundary::Immediate);
            self.standalone_space_admitted += usize::from(boundary == Boundary::StandaloneSpace);
            self.target_id_absent_from_lexical_prefix += usize::from(target_absent);
        }
    }
}

#[derive(Default)]
struct Packet {
    inputs: Vec<Input>,
    labels: Vec<Label>,
    refused: Vec<Refused>,
    all: Count,
    first: Count,
    second: Count,
}

fn decode_tape(bytes: &[u8]) -> Result<Vec<u16>> {
    require(
        bytes.len() >= 64
            && &bytes[..8] == b"UORT\x01\0\0\0"
            && bytes[16..20] == u32::from(VOCABULARY).to_le_bytes()
            && bytes[20..64].iter().all(|&b| b == 0)
            && (bytes.len() - 64).is_multiple_of(2),
        "invalid donor UORT header",
    )?;
    let data = &bytes[64..];
    require(
        u64::from_le_bytes(bytes[8..16].try_into()?) == (data.len() / 2) as u64,
        "wrong donor UORT payload length",
    )?;
    Ok(data
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

fn classify(full: &[u16], precursor: &[u16]) -> (Boundary, Option<Refusal>) {
    let boundary = match precursor {
        [] => Boundary::Immediate,
        [SPACE] => Boundary::StandaloneSpace,
        _ => Boundary::OtherPrecursor,
    };
    let refusal = if full.len() > CAP {
        Some(Refusal::FullOpeningOverCap)
    } else if boundary == Boundary::OtherPrecursor {
        Some(Refusal::UnsupportedPrecursor)
    } else if full.len() + precursor.len() > CAP {
        Some(Refusal::LexicalPrefixOverCap)
    } else {
        None
    };
    (boundary, refusal)
}

fn reconstruct(donor: &Path) -> Result<Packet> {
    let tape = decode_tape(&fs::read(donor.join("answer-first.tokens.u16"))?)?;
    let mask = fs::read(donor.join("answer-first.mask.u8"))?;
    let openings: Vec<Opening> = read_rows(&donor.join("openings.jsonl"))?;
    let groups: Vec<Group> = read_rows(&donor.join("groups.jsonl"))?;
    require(
        tape.len() == mask.len()
            && mask.iter().all(|&m| m <= 1)
            && tape.iter().all(|&id| id < VOCABULARY)
            && openings.len() == TURN_COUNT
            && groups.len() == GROUP_COUNT,
        "wrong authenticated donor shape",
    )?;

    let mut packet = Packet::default();
    let mut next_opening = (0, 0);
    for row in openings {
        let TurnKey { group, turn } = row.key;
        let group_spec = groups.get(group).ok_or("unknown source group")?;
        require(
            (group, turn) == next_opening
                && turn <= 1
                && group_spec.group == group
                && row.prefix_start == group_spec.payload_start
                && row.answer_end < group_spec.visible_end
                && group_spec.visible_end <= tape.len()
                && row.prefix_start < row.answer_start
                && row.answer_start <= row.lexical_offset
                && row.lexical_offset < row.answer_end
                && row.lexical_offset - row.answer_start == row.lexical_answer_offset,
            "source order or lexical span mismatch",
        )?;
        let full = &tape[row.prefix_start..row.answer_start];
        let precursor = &tape[row.answer_start..row.lexical_offset];
        let earlier_eos = full
            .iter()
            .zip(&mask[row.prefix_start..row.answer_start])
            .filter(|&(id, label)| *id == EOS && *label == 0)
            .count();
        let ends_group = row.answer_end + 1 == group_spec.visible_end;
        require(
            full.first() == Some(&BOS)
                && !full[1..].contains(&BOS)
                && full.ends_with(&MARKER)
                && earlier_eos == turn
                && full.iter().filter(|&&id| id == EOS).count() == earlier_eos
                && tape[row.answer_start] == row.immediate_id
                && tape[row.lexical_offset] == row.lexical_id
                && row.lexical_id > EOS
                && mask[row.lexical_offset] == 1
                && tape[row.answer_end] == EOS
                && mask[row.answer_end] == u8::from(ends_group),
            "source prefix, gold, or EOS-mask mismatch",
        )?;
        next_opening = if ends_group {
            (group + 1, 0)
        } else {
            (group, turn + 1)
        };

        let (boundary, refusal) = classify(full, precursor);
        let lexical = &tape[row.prefix_start..row.lexical_offset];
        let absent = !lexical.contains(&row.lexical_id);
        packet.all.observe(boundary, earlier_eos, refusal, absent);
        let turn_counts = if turn == 0 {
            &mut packet.first
        } else {
            &mut packet.second
        };
        turn_counts.observe(boundary, earlier_eos, refusal, absent);
        if let Some(reason) = refusal {
            packet.refused.push(Refused {
                key: row.key,
                boundary,
                reason,
                opening_len: full.len(),
                lexical_prefix_len: lexical.len(),
            });
        } else {
            packet.inputs.push(Input {
                key: row.key,
                boundary,
                full_opening: full.to_vec(),
                lexical_prefix: lexical.to_vec(),
            });
            packet.labels.push(Label {
                key: row.key,
                target: row.lexical_id,
                lexical_payload_offset: row.lexical_offset,
            });
        }
    }
    require(
        next_opening == (GROUP_COUNT, 0)
            && packet.first.seen == GROUP_COUNT
            && packet.second.seen == TURN_COUNT - GROUP_COUNT
            && packet.all.immediate_seen == 2_149
            && packet.all.standalone_space_seen == 2_196,
        "incomplete registered opening census",
    )?;
    Ok(packet)
}

fn verify_rows<T: DeserializeOwned + PartialEq>(
    path: &Path,
    expected: &[T],
    reason: &str,
) -> Result<()> {
    let actual: Vec<T> = read_rows(path)?;
    require(actual == expected, reason)
}

fn prove_corruptions_rejected(out: &Path, packet: &Packet) -> Result<()> {
    // Keep corruptions in a separate private directory and check content before
    // computing any digest check so stale receipt failure cannot mask a bad reader.
    let dir = out.join("controls");
    fs::DirBuilder::new().mode(0o700).create(&dir)?;
    let mut inputs = packet.inputs.clone();
    let first = inputs
        .first_mut()
        .ok_or("no admitted row for BOS control")?;
    first.full_opening.remove(0);
    first.lexical_prefix.remove(0);
    let path = dir.join("deleted-bos.jsonl");
    write_rows(&path, &inputs)?;
    require(
        verify_rows(&path, &packet.inputs, "deleted BOS")
            .is_err_and(|err| err.to_string() == "deleted BOS"),
        "deleted BOS was accepted or could not be parsed",
    )?;

    let mut inputs = packet.inputs.clone();
    let spaced = inputs
        .iter_mut()
        .find(|row| row.boundary == Boundary::StandaloneSpace)
        .ok_or("no admitted standalone-space row")?;
    require(
        spaced.lexical_prefix.pop() == Some(SPACE),
        "bad space control fixture",
    )?;
    let path = dir.join("deleted-space.jsonl");
    write_rows(&path, &inputs)?;
    require(
        verify_rows(&path, &packet.inputs, "deleted space")
            .is_err_and(|err| err.to_string() == "deleted space"),
        "deleted space was accepted or could not be parsed",
    )?;

    let mut labels = packet.labels.clone();
    let gold = labels
        .first()
        .ok_or("no admitted label for swap control")?
        .target;
    let other = labels
        .iter()
        .position(|row| row.target != gold)
        .ok_or("no different gold to swap")?;
    labels[0].target = labels[other].target;
    labels[other].target = gold;
    let path = dir.join("swapped-gold.jsonl");
    write_rows(&path, &labels)?;
    require(
        verify_rows(&path, &packet.labels, "swapped gold")
            .is_err_and(|err| err.to_string() == "swapped gold"),
        "swapped gold was accepted or could not be parsed",
    )
}

fn manifest(out: &Path, packet: &Packet, source: Value) -> Result<Value> {
    let files = FILES
        .into_iter()
        .map(|name| Ok((name, file_binding(&out.join(name))?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(json!({
        "schema": "uor-r4.p4-lexical-train-inputs/1",
        "registration_sha256": digest(REGISTRATION),
        "source": source,
        "files": files,
        "cap": CAP,
        "all": packet.all,
        "first_turn": packet.first,
        "second_turn": packet.second,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [mode, verifier, tokenizer, prior, donor, out, rest @ ..] = args.as_slice() else {
        return Err("usage: uor-p4-lexical-train (build|verify|controls) DONOR_VERIFIER TOKENIZER PRIOR DONOR PACKET [NEW_CONTROLS_ROOT for controls]".into());
    };
    let (verifier, tokenizer, prior, donor, out) = (
        Path::new(verifier),
        Path::new(tokenizer),
        Path::new(prior),
        Path::new(donor),
        Path::new(out),
    );
    require(
        (mode == "build" || mode == "verify") && rest.is_empty()
            || mode == "controls" && rest.len() == 1,
        "bad mode or arguments",
    )?;
    let source = verify_source(verifier, tokenizer, prior, donor)?;
    let packet = reconstruct(donor)?;

    if mode == "build" {
        fs::DirBuilder::new().mode(0o700).create(out)?;
        write_rows(&out.join(FILES[0]), &packet.inputs)?;
        write_rows(&out.join(FILES[1]), &packet.labels)?;
        write_rows(&out.join(FILES[2]), &packet.refused)?;
        let manifest = manifest(out, &packet, source)?;
        let mut receipt = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("receipt.json"))?;
        serde_json::to_writer_pretty(&mut receipt, &manifest)?;
        receipt.write_all(b"\n")?;
        return Ok(());
    }

    verify_rows(
        &out.join(FILES[0]),
        &packet.inputs,
        "causal prefixes differ",
    )?;
    verify_rows(
        &out.join(FILES[1]),
        &packet.labels,
        "original target IDs differ",
    )?;
    verify_rows(&out.join(FILES[2]), &packet.refused, "refused rows differ")?;
    let received: Value = serde_json::from_reader(File::open(out.join("receipt.json"))?)?;
    require(
        received == manifest(out, &packet, source)?,
        "packet receipt differs",
    )?;
    if let [controls] = rest {
        let controls = Path::new(controls);
        fs::DirBuilder::new().mode(0o700).create(controls)?;
        prove_corruptions_rejected(controls, &packet)?;
        println!("deleted BOS, deleted standalone space, and swapped gold refused");
    } else {
        println!("all causal prefixes, original targets, exclusions, and file hashes verified");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_prefix_cap_never_crops_and_observed_space_is_distinct() {
        let fits = vec![7; CAP];
        let short = &fits[..CAP - 1];
        assert_eq!(classify(&fits, &[]), (Boundary::Immediate, None));
        assert_eq!(classify(short, &[SPACE]), (Boundary::StandaloneSpace, None));
        assert_eq!(
            classify(&fits, &[SPACE]),
            (
                Boundary::StandaloneSpace,
                Some(Refusal::LexicalPrefixOverCap)
            )
        );
        assert_eq!(
            classify(&[7; CAP + 1], &[]),
            (Boundary::Immediate, Some(Refusal::FullOpeningOverCap))
        );
        assert_eq!(
            classify(short, &[7]),
            (
                Boundary::OtherPrecursor,
                Some(Refusal::UnsupportedPrecursor)
            )
        );
    }
}
