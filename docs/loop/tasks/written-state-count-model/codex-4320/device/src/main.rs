use serde::{Deserialize, Serialize};
use std::{
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
const PERIOD: u32 = 16;
const COLON: u32 = 28;
const LF: u32 = 201;
const QUESTION: u32 = 33;
const MARKER: [u32; 6] = [LF, 35, 560, 652, 714, COLON];
const CAP: usize = 4;
const COMPLETION_ARMS: [&str; 4] = ["native", "correct", "swapped", "supplied_next"];

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
struct PanelRow {
    pair_id: String,
    side: u32,
    family: String,
    prefix: Vec<u32>,
    gold_pos: u32,
    target: u32,
    next_pos: u32,
    next_id: u32,
    expected: Vec<u32>,
}

impl PanelRow {
    fn boundaries(&self) -> Result<(usize, usize)> {
        let user = self
            .prefix
            .strip_suffix(&MARKER)
            .ok_or("prefix missing canonical assistant marker")?;
        let fence = user
            .windows(2)
            .rposition(|window| window == [LF, LF])
            .ok_or("prefix missing fact/question boundary")?;
        let instruction = user[fence + 2..]
            .iter()
            .rposition(|&token| token == QUESTION)
            .map(|index| fence + 3 + index)
            .ok_or("question missing raw question mark")?;
        if instruction == user.len()
            || self.gold_pos as usize >= fence
            || self.next_pos as usize >= fence
        {
            return Err("empty instruction or original source lies outside facts".into());
        }
        Ok((fence, instruction))
    }

    fn validate(&self) -> Result<()> {
        if self.pair_id.is_empty()
            || self.family.is_empty()
            || self.prefix.len() > CONTEXT - CAP
            || self.prefix.iter().any(|&id| id >= VOCAB)
            || self.gold_pos == self.next_pos
            || self.target == self.next_id
            || matches!(self.target, EOS | PERIOD)
            || matches!(self.next_id, EOS | PERIOD)
            || self.prefix.get(self.gold_pos as usize) != Some(&self.target)
            || self.prefix.get(self.next_pos as usize) != Some(&self.next_id)
        {
            return Err("invalid prefix or original source identity".into());
        }
        self.boundaries()?;
        let correct = match self.side {
            0 => self.expected == [self.target, PERIOD, EOS],
            1 => self.expected == [self.target, PERIOD, self.next_id, EOS],
            _ => false,
        };
        if !correct {
            return Err("expected IDs do not describe the registered stop or continuation".into());
        }
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

impl Trajectory {
    fn new(steps: Vec<Step>, expected: &[u32]) -> Result<Self> {
        if !(3..=CAP).contains(&steps.len())
            || steps[..steps.len() - 1]
                .iter()
                .any(|step| step.writer.committed_id == EOS)
        {
            return Err("replay continued after EOS or violated the cap".into());
        }
        let terminal = steps
            .last()
            .is_some_and(|step| step.writer.committed_id == EOS);
        let first_mismatch = steps
            .iter()
            .enumerate()
            .find(|(index, step)| expected.get(*index) != Some(&step.writer.committed_id))
            .map(|(index, _)| index + 1)
            .or_else(|| (steps.len() < expected.len()).then_some(steps.len() + 1));
        Ok(Self {
            steps,
            terminal,
            first_mismatch,
        })
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct CompletionObservation<'a> {
    #[serde(flatten)]
    row: &'a PanelRow,
    third_trace: gpu::ThirdTrace,
    arms: std::collections::BTreeMap<&'static str, Trajectory>,
}

fn peers(rows: &[&PanelRow]) -> Result<Vec<u32>> {
    let mut pairs = std::collections::BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let entry = pairs.entry(row.pair_id.as_str()).or_insert([None; 2]);
        if entry[row.side as usize].replace(index).is_some() {
            return Err(format!("duplicate pair side for {}", row.pair_id).into());
        }
    }
    let mut peers = vec![NONE; rows.len()];
    for (pair_id, sides) in pairs {
        let [Some(left), Some(right)] = sides else {
            return Err(format!("missing opposite side for {pair_id}").into());
        };
        let (stop, continuation) = (rows[left], rows[right]);
        let (_, stop_instruction) = stop.boundaries()?;
        let (_, continue_instruction) = continuation.boundaries()?;
        if stop.family != continuation.family
            || (stop.gold_pos, stop.target, stop.next_pos, stop.next_id)
                != (
                    continuation.gold_pos,
                    continuation.target,
                    continuation.next_pos,
                    continuation.next_id,
                )
            || stop.prefix[..stop_instruction] != continuation.prefix[..continue_instruction]
            || stop.prefix == continuation.prefix
        {
            return Err(format!("unaligned factual positions or suffixes for {pair_id}").into());
        }
        peers[left] = u32::try_from(right)?;
        peers[right] = u32::try_from(left)?;
    }
    Ok(peers)
}

fn load_rows(path: &str) -> Result<Vec<PanelRow>> {
    let rows: Vec<PanelRow> = BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let row: PanelRow = serde_json::from_str(&line?)?;
            row.validate()
                .map_err(|error| format!("row {}: {error}", index + 1))?;
            Ok(row)
        })
        .collect::<Result<_>>()?;
    if rows.is_empty() {
        return Err("empty matched panel".into());
    }
    peers(&rows.iter().collect::<Vec<_>>())?;
    Ok(rows)
}

fn write_rows(path: &Path, rows: &[CompletionObservation<'_>]) -> Result<()> {
    let mut output = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    for row in rows {
        serde_json::to_writer(&mut output, row)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args[1..].iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["check-rows", input] => {
            let rows = load_rows(input)?;
            println!("{}", serde_json::json!({"rows": rows.len(), "pairs": rows.len() / 2,
                "input_sha256": gpu::digest(input)? }));
        }
        ["finite", image, image_sha] => {
            gpu::finite(image, image_sha)?;
            println!("{}", serde_json::json!({"pass": true, "image_sha256": image_sha,
                "controls": ["supplied_period", "stop", "continue_native", "actual_swap",
                "validated_supplied_word", "missing_address", "wrong_id", "outside_fact_fence",
                "invalid_peer", "native_eos_terminates"] }));
        }
        ["score", image, image_sha, copy, occurrence, baseline, base, classes, weights,
            marker, input, output] => {
            let input_sha256 = gpu::digest(input)?;
            let rows = load_rows(input)?;
            if gpu::digest(input)? != input_sha256 {
                return Err("input changed while loading".into());
            }
            let output = Path::new(output);
            fs::create_dir(output)?;
            let files = gpu::Files { copy, occurrence, baseline, base, classes, weights, marker };
            let started = Instant::now();
            let forward = gpu::replay_completion(&files, image, image_sha, rows.iter())?;
            write_rows(&output.join("observations.jsonl"), &forward)?;
            let mut reverse = gpu::replay_completion(&files, image, image_sha, rows.iter().rev())?;
            write_rows(&output.join("reverse.jsonl"), &reverse)?;
            reverse.reverse();
            if forward != reverse {
                return Err("forward and reverse exact GPU traces differ".into());
            }
            let receipt = serde_json::json!({"pass": true, "rows": rows.len(),
                "pairs": rows.len() / 2, "arms": COMPLETION_ARMS, "input_sha256": input_sha256,
                "image_sha256": image_sha, "elapsed_ms": started.elapsed().as_millis()});
            fs::write(output.join("replay.json"), serde_json::to_vec_pretty(&receipt)?)?;
            println!("{receipt}");
        }
        _ => return Err("usage: postperiod-device check-rows PANEL | finite CUBIN SHA256 | score CUBIN SHA256 COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL NEW_OUT".into()),
    }
    Ok(())
}
