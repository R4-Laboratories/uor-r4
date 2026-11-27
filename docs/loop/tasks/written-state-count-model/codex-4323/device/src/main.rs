use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

#[path = "../../../codex-4321/device/src/gpu.rs"]
mod gpu;
mod scope;
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
const CAP: usize = 5;
const STAGES: usize = 3;
const WORDS: usize = 4096;
const BASE_WORDS: usize = STAGES * (WORDS + 1);
const POSITION_WORDS: usize = BASE_WORDS + WORDS;
const STATE_WORDS: usize = POSITION_WORDS + WORDS;
const ARMS_NAMES: [&str; 5] = ["native", "learned", "swapped", "correct", "opposite"];

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
struct PanelRow {
    pair_id: String,
    side: u32,
    family: String,
    contrast: u32,
    prefix: Vec<u32>,
    positions: [u32; 4],
    ids: [u32; 4],
    expected: Vec<u32>,
}

impl PanelRow {
    fn boundaries(&self) -> Result<(usize, usize, &[u32])> {
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
        if instruction == user.len() {
            return Err("empty directive".into());
        }
        Ok((fence, instruction, &user[instruction..]))
    }

    fn validate(&self) -> Result<()> {
        if self.pair_id.is_empty()
            || self.family.is_empty()
            || self.side > 1
            || self.contrast > 1
            || self.prefix.len() > CONTEXT - CAP
            || self.prefix.iter().any(|&id| id >= VOCAB)
        {
            return Err("invalid panel metadata or prefix".into());
        }
        let (fence, _, _) = self.boundaries()?;
        for (index, (&position, &id)) in self.positions.iter().zip(&self.ids).enumerate() {
            if position as usize >= fence
                || matches!(id, 0 | EOS | PERIOD)
                || self.ids[..index].contains(&id)
                || self.prefix[position as usize] != id
            {
                return Err("invalid or repeated original source identity".into());
            }
        }
        let mut expected = vec![self.ids[0], PERIOD];
        expected.extend_from_slice(&self.ids[1..1 + self.requested() as usize]);
        expected.push(EOS);
        if self.expected != expected {
            return Err("expected trajectory does not match declared contrast".into());
        }
        Ok(())
    }

    fn requested(&self) -> u32 {
        self.contrast + self.side
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Step {
    #[serde(flatten)]
    writer: gpu::WriterEvent,
    selected_source: &'static str,
    stage: Option<u32>,
    action_score: Option<i32>,
    requested_pos: Option<u32>,
    requested_id: Option<u32>,
    refusal: u32,
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
struct Common {
    stage: u32,
    own_score: i32,
    peer_score: i32,
    own_choice: &'static str,
    peer_choice: &'static str,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct ChoiceObservation<'a> {
    #[serde(flatten)]
    row: &'a PanelRow,
    common: Common,
    arms: std::collections::BTreeMap<&'static str, Trajectory>,
}

fn peers(rows: &[&PanelRow]) -> Result<Vec<u32>> {
    paired_rows(rows, false)
}

fn paired_rows(rows: &[&PanelRow], allow_identical: bool) -> Result<Vec<u32>> {
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
        let (_, stop_instruction, stop_wording) = stop.boundaries()?;
        let (_, continue_instruction, continue_wording) = continuation.boundaries()?;
        if stop.family != continuation.family
            || stop.contrast != continuation.contrast
            || stop.positions != continuation.positions
            || stop.ids != continuation.ids
            || stop.prefix[..stop_instruction] != continuation.prefix[..continue_instruction]
            || (!allow_identical && stop_wording == continue_wording)
        {
            return Err(format!("unaligned factual positions or suffixes for {pair_id}").into());
        }
        peers[left] = u32::try_from(right)?;
        peers[right] = u32::try_from(left)?;
    }
    Ok(peers)
}

fn load_rows(path: &str, allow_identical: bool) -> Result<(String, Vec<PanelRow>)> {
    let hash = gpu::digest(path)?;
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
    if rows.is_empty() || gpu::digest(path)? != hash {
        return Err("empty or changing panel".into());
    }
    paired_rows(&rows.iter().collect::<Vec<_>>(), allow_identical)?;
    Ok((hash, rows))
}

fn write_rows(path: &Path, rows: &[ChoiceObservation<'_>]) -> Result<()> {
    let mut output = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    for row in rows {
        serde_json::to_writer(&mut output, row)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    #[cfg(feature = "release-factorial")]
    if let Some(result) = gpu::factorial::command() {
        return result;
    }
    let args: Vec<_> = std::env::args().collect();
    match args[1..].iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["check-rows", input] => {
            let (hash, rows) = load_rows(input, false)?;
            println!("{}", serde_json::json!({"rows": rows.len(), "pairs": rows.len() / 2,
                "input_sha256": hash}));
        }
        ["finite", image, image_sha, frozen, frozen_sha] => {
            let receipt = scope::finite(image, image_sha, frozen, frozen_sha)?;
            println!("{receipt}");
        }
        ["trace", image, image_sha, frozen, frozen_sha, base, input, weights, output] => {
            let (hash, rows) = load_rows(input, true)?;
            scope::trace([image, image_sha, frozen, frozen_sha], base, &rows, &hash, weights, Path::new(output))?;
        }
        ["fit", image, image_sha, base, input, weights, arm, output] => {
            let (hash, rows) = load_rows(input, false)?;
            scope::fit([image, image_sha], base, &rows, &hash, weights, arm, Path::new(output))?;
        }
        ["score", image, image_sha, copy, occurrence, baseline, base, classes, weights,
            marker, input, choice_weights, output] => {
            let (input_sha256, rows) = load_rows(input, false)?;
            let (choice_weights_sha256, state, fitted_words) = scope::load_weights(choice_weights)?;
            let output = Path::new(output);
            fs::create_dir(output)?;
            let files = gpu::Files { copy, occurrence, baseline, base, classes, weights, marker };
            let started = Instant::now();
            let mut replay = gpu::Replayer::open(&files, image, image_sha, &state)?;
            let forward = replay.replay(rows.iter())?;
            write_rows(&output.join("observations.jsonl"), &forward)?;
            let mut reverse = replay.replay(rows.iter().rev())?;
            write_rows(&output.join("reverse.jsonl"), &reverse)?;
            reverse.reverse();
            if forward != reverse {
                return Err("forward and reverse exact GPU traces differ".into());
            }
            let receipt = serde_json::json!({"pass": true, "rows": rows.len(),
                "pairs": rows.len() / 2, "arms": ARMS_NAMES, "input_sha256": input_sha256,
                "choice_weights_sha256": choice_weights_sha256, "image_sha256": image_sha,
                "fitted_words": fitted_words, "elapsed_ms": started.elapsed().as_millis()});
            fs::write(output.join("replay.json"), serde_json::to_vec_pretty(&receipt)?)?;
            println!("{receipt}");
        }
        _ => return Err("usage: scope-device check-rows PANEL | finite CUBIN SHA256 FROZEN_CUBIN FROZEN_SHA256 | trace CUBIN SHA256 FROZEN_CUBIN FROZEN_SHA256 BASE_DIR PANEL WEIGHTS NEW_OUT | fit CUBIN SHA256 BASE_DIR TRAIN BALANCED_WEIGHTS link|scope|difference NEW_OUT | score CUBIN SHA256 COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL CHOICE_WEIGHTS NEW_OUT".into()),
    }
    Ok(())
}
