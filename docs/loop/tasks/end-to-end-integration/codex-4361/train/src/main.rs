use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fs, path::Path};

#[cfg(feature = "cuda")]
include!(concat!(env!("OUT_DIR"), "/backend.rs"));
#[cfg(feature = "cuda")]
mod device;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const CHUNK_IDS: usize = 257;
const MAX_IDS: usize = 1_088;
const MAX_TRANSITIONS: usize = MAX_IDS - 1;
const DOCUMENTS_PER_UPDATE: usize = 4;
const UPDATES: usize = 64;
const TRAIN_ROWS: usize = DOCUMENTS_PER_UPDATE * UPDATES;
const TRAIN_SHA256: &str = "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a";
const VOCAB: u16 = 4_096;
const BOS: u16 = 0;
const EOS: u16 = 1;
const LR_Q8: i32 = 512;
const WD_SHIFT: u32 = 19;
const EPS: u64 = 16;
const ARM2_SHA256: &str = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";

#[derive(Debug, Deserialize)]
struct Row {
    id: u64,
    prefix: Vec<u16>,
    expected: Vec<u16>,
    stratum: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Sequence {
    tokens: Vec<u16>,
    mask: Vec<u8>,
}

#[derive(Debug)]
struct Document {
    id: u64,
    full: Sequence,
    stratum: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Arm {
    Full,
    Reset,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
struct Work {
    documents: usize,
    unsupervised_sequences: usize,
    semantic_transitions: usize,
    padding_transitions: usize,
    targets: usize,
}

impl Work {
    fn record(&mut self, sequence: &Sequence, padding: bool) {
        let transitions = sequence.tokens.len().saturating_sub(1);
        let targets = sequence
            .mask
            .iter()
            .skip(1)
            .map(|&m| usize::from(m))
            .sum::<usize>();
        if padding {
            self.padding_transitions += transitions;
        } else {
            self.semantic_transitions += transitions;
            self.unsupervised_sequences += usize::from(targets == 0);
        }
        self.targets += targets;
    }

    fn total_transitions(&self) -> usize {
        self.semantic_transitions + self.padding_transitions
    }
}

struct Batch {
    forward: Vec<Sequence>,
    padding: Vec<Sequence>,
    work: Work,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn compile(row: Row) -> Result<Document> {
    let Row {
        id,
        mut prefix,
        expected,
        stratum,
    } = row;
    let start = prefix.len();
    let total = start
        .checked_add(expected.len())
        .ok_or("sequence length overflow")?;
    if prefix.first() != Some(&BOS)
        || prefix.iter().skip(1).any(|&t| t == BOS || t >= VOCAB)
        || prefix.iter().filter(|&&t| t == EOS).count() != 2
        || expected.len() < 2
        || expected.last() != Some(&EOS)
        || expected[..expected.len().saturating_sub(1)]
            .iter()
            .any(|&t| t == BOS || t == EOS || t >= VOCAB)
        || total > MAX_IDS
        || !matches!(stratum.as_str(), "near" | "far" | "intermediate")
        || start / CHUNK_IDS != (total - 1) / CHUNK_IDS
        || start % CHUNK_IDS == 0
    {
        return Err(
            format!("invalid authentic third-turn row {id}, bound or target position").into(),
        );
    }
    prefix.extend_from_slice(&expected);
    let mut mask = vec![0; start];
    mask.resize(total, 1);
    Ok(Document {
        id,
        full: Sequence {
            tokens: prefix,
            mask,
        },
        stratum,
    })
}

fn prepare(docs: &[Document], arm: Arm) -> Result<Batch> {
    let mut out = Batch {
        forward: Vec::new(),
        padding: Vec::new(),
        work: Work {
            documents: docs.len(),
            ..Work::default()
        },
    };
    for doc in docs {
        let first = out.forward.len();
        match arm {
            Arm::Full => out.forward.push(doc.full.clone()),
            Arm::Reset => {
                for (tokens, mask) in doc
                    .full
                    .tokens
                    .chunks(CHUNK_IDS)
                    .zip(doc.full.mask.chunks(CHUNK_IDS))
                {
                    if mask.first() != Some(&0) {
                        return Err(
                            format!("document {} would lose a target at reset", doc.id).into()
                        );
                    }
                    out.forward.push(Sequence {
                        tokens: tokens.to_vec(),
                        mask: mask.to_vec(),
                    });
                }
                for cut in (CHUNK_IDS..doc.full.tokens.len()).step_by(CHUNK_IDS) {
                    // Execute the omitted transition as a zero-state, zero-loss isolated launch.
                    // No state or adjoint from this accounting row reaches a training document.
                    out.padding.push(Sequence {
                        tokens: doc.full.tokens[cut - 1..=cut].to_vec(),
                        mask: vec![0; 2],
                    });
                }
            }
        }
        for sequence in &out.forward[first..] {
            out.work.record(sequence, false);
        }
    }
    for sequence in &out.padding {
        out.work.record(sequence, true);
    }
    Ok(out)
}

fn positive_ids(batch: &Batch) -> impl Iterator<Item = u16> + '_ {
    batch.forward.iter().flat_map(|seq| {
        seq.tokens
            .iter()
            .copied()
            .zip(&seq.mask)
            .skip(1)
            .filter_map(|(token, &mask)| (mask == 1).then_some(token))
    })
}

fn match_batches(full: &Batch, reset: &Batch) -> Result<()> {
    match_work(&full.work, &reset.work)?;
    if !positive_ids(full).eq(positive_ids(reset)) {
        return Err("ordered supervised original IDs differ".into());
    }
    Ok(())
}

fn match_work(full: &Work, reset: &Work) -> Result<()> {
    if full.documents != reset.documents
        || full.targets == 0
        || full.targets != reset.targets
        || full.total_transitions() != reset.total_transitions()
    {
        return Err(format!("unequal update work: full {full:?}, reset {reset:?}").into());
    }
    Ok(())
}

struct Tape {
    sha256: String,
    docs: Vec<Document>,
    near: usize,
    far: usize,
    intermediate: usize,
}

fn load_tape(path: &Path, pinned_sha: &str) -> Result<Tape> {
    let bytes = fs::read(path)?;
    let sha256 = sha256(&bytes);
    if sha256 != pinned_sha || sha256 != TRAIN_SHA256 {
        return Err(format!("TRAIN tape SHA-256 mismatch: {sha256}").into());
    }
    let mut ids = BTreeSet::new();
    let mut docs = Vec::new();
    for (index, line) in std::str::from_utf8(&bytes)?.lines().enumerate() {
        let row: Row = serde_json::from_str(line)
            .map_err(|error| format!("TRAIN row {}: {error}", index + 1))?;
        if !ids.insert(row.id) {
            return Err(format!("TRAIN duplicate document ID {}", row.id).into());
        }
        docs.push(compile(row)?);
    }
    let near = docs.iter().filter(|doc| doc.stratum == "near").count();
    let far = docs.iter().filter(|doc| doc.stratum == "far").count();
    if docs.len() != TRAIN_ROWS || near == 0 || far == 0 {
        return Err(format!(
            "need {TRAIN_ROWS} rows and both strata, got {} ({near} near and {far} far)",
            docs.len()
        )
        .into());
    }
    for update in docs.chunks_exact(DOCUMENTS_PER_UPDATE) {
        match_batches(&prepare(update, Arm::Full)?, &prepare(update, Arm::Reset)?)?;
    }
    let intermediate = docs.len() - near - far;
    Ok(Tape {
        sha256,
        docs,
        near,
        far,
        intermediate,
    })
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [flag, tape, digest] if flag == "--validate" => {
            let tape = load_tape(Path::new(tape), digest)?;
            let rows = &tape.docs;
            let total = rows.iter().map(|row| row.full.tokens.len() - 1).sum::<usize>();
            let cuts = rows.iter().map(|row| (row.full.tokens.len() - 1) / CHUNK_IDS).sum::<usize>();
            let targets = rows.iter().flat_map(|row| &row.full.mask).filter(|&&mask| mask == 1).count();
            let max_ids = rows.iter().map(|row| row.full.tokens.len()).max().ok_or("empty pinned tape")?;
            let max_targets = rows.iter().map(|row| row.full.mask.iter().filter(|&&mask| mask == 1).count()).max().ok_or("empty pinned tape")?;
            println!("{}", json!({
                "validation":"PASS", "train_sha256":tape.sha256, "documents":tape.docs.len(),
                "near":tape.near, "far":tape.far, "intermediate":tape.intermediate, "updates":UPDATES, "documents_per_update":DOCUMENTS_PER_UPDATE,
                "max_ids":MAX_IDS, "max_transitions":MAX_TRANSITIONS, "initial_checkpoint_sha256":ARM2_SHA256,
                "lr_q8":LR_Q8, "wd_shift":WD_SHIFT, "eps":EPS,
                "total_transitions_per_arm":total,"reset_padding_transitions":cuts,"supervised_targets_per_arm":targets,
                "maximum_document_ids":max_ids,"maximum_document_targets":max_targets,
            }));
        }
        #[cfg(feature = "cuda")]
        [flag, output] if flag == "--finite" => device::finite(Path::new(output))?,
        #[cfg(feature = "cuda")]
        [flag, checkpoint, tape, digest, output, finite] if flag == "--train" => {
            device::train(Path::new(checkpoint), Path::new(tape), digest, Path::new(output), Path::new(finite))?;
        }
        _ => return Err("usage: --validate TRAIN.jsonl SHA256 | --finite FRESH_RECEIPT.json | --train ARM2 TRAIN.jsonl SHA256 FRESH_OUTPUT_DIR FINITE_RECEIPT.json (training requires --features cuda)".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(prefix_len: usize) -> Row {
        let mut prefix = vec![9; prefix_len];
        prefix[0] = BOS;
        prefix[5] = EOS;
        prefix[10] = EOS;
        Row {
            id: 17,
            prefix,
            expected: vec![11, 12, EOS],
            stratum: "far".into(),
        }
    }

    #[test]
    fn matched_padding_and_targets_at_real_boundary() -> Result<()> {
        let docs = [compile(row(CHUNK_IDS + 1))?];
        let full = prepare(&docs, Arm::Full)?;
        let reset = prepare(&docs, Arm::Reset)?;
        match_batches(&full, &reset)?;
        assert_eq!(
            reset
                .forward
                .iter()
                .map(|s| s.tokens.len())
                .collect::<Vec<_>>(),
            [257, 4]
        );
        assert_eq!(
            reset.work.semantic_transitions + 1,
            full.work.semantic_transitions
        );
        assert_eq!(reset.work.padding_transitions, 1);
        assert_eq!(reset.work.targets, 3);
        assert_eq!(reset.work.unsupervised_sequences, 1);
        assert_eq!(reset.padding[0].mask, [0, 0]);
        for invalid in [255, 256, 257] {
            assert!(compile(row(invalid)).is_err(), "bad boundary {invalid}");
        }
        Ok(())
    }
}
