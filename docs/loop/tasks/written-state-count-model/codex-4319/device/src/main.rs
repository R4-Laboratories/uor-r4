use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

mod gpu;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CONTEXT: usize = 192;
const VOCAB: u32 = 4096;
const NONE: u32 = u32::MAX;
const EOS: u32 = 1;
const COLON: u32 = 28;
const PERIOD: u32 = 16;
const LF: u32 = 201;
const QUESTION: u32 = 33;
const MARKER: [u32; 6] = [LF, 35, 560, 652, 714, COLON];
const WORDS: usize = 4096;
const CAP: usize = 3;
const ENDING_ARMS: [&str; 5] = ["native", "learned", "swapped", "oracle", "wrong_oracle"];

#[derive(Deserialize)]
struct PanelRow {
    pair_id: String,
    side: usize,
    prefix: Vec<u32>,
    gold_pos: u32,
    target: u32,
    expected: Vec<u32>,
    family: String,
}

impl PanelRow {
    fn spans(&self) -> Result<(&[u32], &[u32])> {
        let user = self
            .prefix
            .strip_suffix(&MARKER)
            .ok_or("prefix missing canonical assistant marker")?;
        let separator = user
            .windows(2)
            .rposition(|window| window == [LF, LF])
            .ok_or("prefix missing fact/question boundary")?;
        let question = user[separator + 2..]
            .iter()
            .rposition(|&token| token == QUESTION)
            .ok_or("question missing raw question mark")?;
        let instruction = separator + 2 + question + 1;
        if instruction == user.len() || self.gold_pos as usize >= separator {
            return Err("empty instruction or source lies outside facts".into());
        }
        Ok(user.split_at(instruction))
    }

    fn ideal(&self) -> Result<u32> {
        match self.expected.as_slice() {
            [target, EOS] if *target == self.target => Ok(EOS),
            [target, PERIOD, EOS] if *target == self.target => Ok(PERIOD),
            _ => Err("expected IDs do not describe a copied word and registered ending".into()),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.pair_id.is_empty()
            || self.family.is_empty()
            || self.side > 1
            || self.prefix.len() > CONTEXT
            || self.prefix.iter().any(|&id| id >= VOCAB)
            || self.target == EOS
            || self.target == PERIOD
            || self.prefix.get(self.gold_pos as usize) != Some(&self.target)
        {
            return Err("invalid causal prefix, matched side, or asserted source ID".into());
        }
        self.spans()?;
        self.ideal()?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Step {
    #[serde(flatten)]
    writer: gpu::WriterEvent,
    selected_source: &'static str,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Trajectory {
    steps: Vec<Step>,
    terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct EndingObservation<'a> {
    pair_id: &'a str,
    side: usize,
    prefix: &'a [u32],
    gold_pos: u32,
    target: u32,
    expected: &'a [u32],
    family: &'a str,
    score: i32,
    peer_score: i32,
    arms: BTreeMap<&'static str, Trajectory>,
}

fn peers(rows: &[&PanelRow]) -> Result<Vec<u32>> {
    let mut pairs = BTreeMap::<&str, [Option<usize>; 2]>::new();
    for (index, row) in rows.iter().enumerate() {
        if pairs.entry(&row.pair_id).or_default()[row.side]
            .replace(index)
            .is_some()
        {
            return Err(format!("duplicate side {} of {}", row.side, row.pair_id).into());
        }
    }
    let mut peers = vec![0; rows.len()];
    for (pair, indices) in pairs {
        let [Some(a), Some(b)] = indices else {
            return Err(format!("pair {pair} missing matched opposite instruction").into());
        };
        let (first, second) = (rows[a], rows[b]);
        let (first_base, first_instruction) = first.spans()?;
        let (second_base, second_instruction) = second.spans()?;
        if first_base != second_base
            || first_instruction == second_instruction
            || first.target != second.target
            || first.gold_pos != second.gold_pos
            || first.family != second.family
            || first.ideal()? == second.ideal()?
        {
            return Err(format!("pair {pair} must differ only in ending instructions").into());
        }
        peers[a] = u32::try_from(b)?;
        peers[b] = u32::try_from(a)?;
    }
    Ok(peers)
}

fn load_rows(path: &str) -> Result<Vec<PanelRow>> {
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let row: PanelRow = serde_json::from_str(&line?)?;
        row.validate()
            .map_err(|err| format!("{} row {} side {}: {err}", row.pair_id, index + 1, row.side))?;
        rows.push(row);
    }
    if rows.is_empty() || rows.len() > i32::MAX as usize {
        return Err("panel must contain between 1 and i32::MAX rows".into());
    }
    peers(&rows.iter().collect::<Vec<_>>())?;
    Ok(rows)
}

fn write_rows(path: &Path, observations: &[EndingObservation<'_>]) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut file = BufWriter::new(file);
    for row in observations {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(())
}

fn train(image: &str, base: &str, input: &str, output: &str) -> Result<()> {
    let start = Instant::now();
    let rows = load_rows(input)?;
    let rows: Vec<_> = rows.iter().collect();
    let device = gpu::EndingDevice::open(image, base)?;
    let (weights, stop, period) = device.fit(&rows)?;
    let serial: Vec<u8> = weights.iter().flat_map(|word| word.to_le_bytes()).collect();
    let output = Path::new(output);
    fs::create_dir(output)?;
    let weight_path = output.join("weights.i16le");
    fs::write(&weight_path, serial)?;
    let receipt = serde_json::json!({
        "schema": "uor-r4.ending-device-train/1",
        "rows": rows.len(),
        "stop": stop,
        "period": period,
        "passes": 1,
        "i16_words": weights.len(),
        "nonzero_words": weights.iter().filter(|&&weight| weight != 0).count(),
        "input_sha256": gpu::digest(input)?,
        "ending_cubin_sha256": gpu::digest(image)?,
        "weights_sha256": gpu::digest(weight_path.to_str().ok_or("non-UTF8 weight path")?)?,
        "elapsed_ms": start.elapsed().as_millis(),
    });
    fs::write(
        output.join("train.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

fn score(
    image: &str,
    files: &gpu::Files<'_>,
    weights: &str,
    input: &str,
    output: &str,
) -> Result<()> {
    let start = Instant::now();
    let rows = load_rows(input)?;
    let output = Path::new(output);
    fs::create_dir(output)?;
    let forward = gpu::replay_endings(image, files, weights, rows.iter())?;
    write_rows(&output.join("observations.jsonl"), &forward)?;
    let reverse = gpu::replay_endings(image, files, weights, rows.iter().rev())?;
    write_rows(&output.join("reverse.jsonl"), &reverse)?;
    for (first, second) in forward.iter().zip(reverse.iter().rev()) {
        if first != second {
            return Err(format!(
                "cold replay mismatch in {} side {}",
                first.pair_id, first.side
            )
            .into());
        }
    }
    let receipt = serde_json::json!({
        "schema": "uor-r4.ending-device-score/1",
        "rows": forward.len(),
        "arms_per_row": ENDING_ARMS.len(),
        "raw_cap": CAP,
        "reverse_reloaded": true,
        "reverse_replay": "identical",
        "input_sha256": gpu::digest(input)?,
        "copy_cubin_sha256": gpu::digest(files.copy)?,
        "ending_cubin_sha256": gpu::digest(image)?,
        "weights_sha256": gpu::digest(weights)?,
        "elapsed_ms": start.elapsed().as_millis(),
    });
    fs::write(
        output.join("replay.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["check-rows", input] => {
            println!("{}", load_rows(input)?.len());
            Ok(())
        }
        ["finite", image] => gpu::finite(image),
        ["train", image, base, input, output] => train(image, base, input, output),
        ["score", image, copy, occurrence, baseline, base, classes, reader, marker, weights, input, output] => {
            score(
                image,
                &gpu::Files {
                    copy,
                    occurrence,
                    baseline,
                    base,
                    classes,
                    weights: reader,
                    marker,
                },
                weights,
                input,
                output,
            )
        }
        _ => Err(concat!(
            "usage: ending-device check-rows PANEL.jsonl\n",
            "       ending-device finite ENDING.cubin\n",
            "       ending-device train ENDING.cubin BASE_FULL_DIR TRAIN.jsonl NEW_OUTPUT_DIR\n",
            "       ending-device score ENDING.cubin COPY.cubin OCCURRENCE.cubin BASELINE.cubin ",
            "BASE_FULL_DIR CLASSES.u8 READER_WEIGHTS.i16le MARKER.tsv END_WEIGHTS.i16le ",
            "PANEL.jsonl NEW_OUTPUT_DIR"
        )
        .into()),
    }
}
