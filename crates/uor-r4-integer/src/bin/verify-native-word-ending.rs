use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uor_r4_integer::discrete_d11::{native_word_ending, IntegerDiscreteModel};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize, Serialize)]
struct Row {
    pair_id: String,
    side: usize,
    prefix: Vec<u32>,
    gold_pos: u16,
    target: u32,
    expected: Vec<u32>,
    family: String,
}

#[derive(Serialize)]
struct Observation<'a> {
    #[serde(flatten)]
    row: &'a Row,
    #[serde(flatten)]
    replay: native_word_ending::Replay,
}

fn main() -> CliResult<()> {
    const USAGE: &str = "usage: verify-native-word-ending ARTIFACT PANEL.jsonl [DEVICE.jsonl]";
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (artifact, panel, mut device) = match args.as_slice() {
        [artifact, panel] => (artifact, panel, None),
        [artifact, panel, device] => (artifact, panel, Some(BufReader::new(File::open(device)?))),
        _ => return Err(USAGE.into()),
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    let identity =
        native_word_ending::identity(&model).ok_or("missing packed word-ending state")?;
    let mut line = Vec::new();
    let mut panel = BufReader::new(File::open(panel)?);
    let mut rows = Vec::new();
    while read_line(&mut panel, &mut line)? {
        let row = serde_json::from_slice(&line)
            .map_err(|error| format!("invalid panel row {}: {error}", rows.len() + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("empty word-ending panel".into());
    }
    let pairs = match_peers(&rows)?;
    let mut output = BufWriter::new(io::stdout().lock());
    let mut trajectories = 0usize;
    let mut transitions = 0usize;
    let mut stops = 0usize;
    for (index, row) in rows.iter().enumerate() {
        let pair = pairs
            .get(row.pair_id.as_str())
            .ok_or("row missing from peer index")?;
        let peer = pair[1 - row.side].ok_or("missing matched peer")?;
        let replay = native_word_ending::observe(
            &model,
            &row.prefix,
            &rows[peer].prefix,
            row.gold_pos,
            &row.expected,
        )
        .map_err(|error| format!("row {} ({:?}): {error}", index + 1, row.pair_id))?;
        for arm in [
            &replay.arms.native,
            &replay.arms.learned,
            &replay.arms.swapped,
            &replay.arms.oracle,
            &replay.arms.wrong_oracle,
        ] {
            trajectories += 1;
            transitions += arm.steps.len();
            stops += usize::from(arm.terminal);
        }
        let observation = serde_json::to_value(Observation { row, replay })?;
        if let Some(device) = device.as_mut() {
            if !read_line(device, &mut line)? {
                return Err(format!("device replay ended before row {}", index + 1).into());
            }
            let expected: Value = serde_json::from_slice(&line)?;
            if observation != expected {
                let (path, cold, gpu) = first_difference("$".into(), &observation, &expected);
                return Err(format!(
                    "row {} ({:?}, side {}) {path} differs, cold {cold:?}, device {gpu:?}",
                    index + 1,
                    row.pair_id,
                    row.side,
                )
                .into());
            }
        }
        serde_json::to_writer(&mut output, &observation)?;
        output.write_all(b"\n")?;
    }
    if let Some(device) = device.as_mut() {
        if read_line(device, &mut line)? {
            return Err(
                format!("device replay has extra rows after {} matches", rows.len()).into(),
            );
        }
    }
    output.flush()?;
    eprintln!(
        "{}",
        serde_json::json!({
            "artifact_sha256": model.artifact_sha256(),
            "word_ending": identity,
            "decoding": native_word_ending::DECODING,
            "pairs": pairs.len(),
            "rows": rows.len(),
            "device_exact_rows": device.map(|_| rows.len()),
            "trajectories": trajectories,
            "writer_transitions": transitions,
            "terminal_trajectories": stops,
        })
    );
    Ok(())
}

fn match_peers(rows: &[Row]) -> CliResult<BTreeMap<&str, [Option<usize>; 2]>> {
    let mut pairs: BTreeMap<&str, [Option<usize>; 2]> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        if row.prefix.get(usize::from(row.gold_pos)) != Some(&row.target)
            || row.expected.first() != Some(&row.target)
        {
            return Err(format!("row {} does not bind its expected factual ID", index + 1).into());
        }
        let sides = pairs.entry(&row.pair_id).or_default();
        let slot = sides.get_mut(row.side).ok_or("pair side must be 0 or 1")?;
        if slot.replace(index).is_some() {
            return Err(format!("duplicate pair side for {:?}", row.pair_id).into());
        }
    }
    for (pair, sides) in &pairs {
        let [Some(left), Some(right)] = sides else {
            return Err(format!("missing matched peer for {pair:?}").into());
        };
        let (left, right) = (&rows[*left], &rows[*right]);
        if left.gold_pos != right.gold_pos
            || left.target != right.target
            || left.family != right.family
            || left.prefix == right.prefix
            || left.expected.len() == right.expected.len()
        {
            return Err(format!("divergent or identical pair inputs for {pair:?}").into());
        }
    }
    Ok(pairs)
}

fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> io::Result<bool> {
    const MAX_LINE_BYTES: usize = 64 * 1024;
    line.clear();
    let read = reader
        .take((MAX_LINE_BYTES + 1) as u64)
        .read_until(b'\n', line)?;
    if read > MAX_LINE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay row exceeds the byte limit",
        ));
    }
    Ok(read != 0)
}

fn first_difference<'a>(
    path: String,
    cold: &'a Value,
    gpu: &'a Value,
) -> (String, Option<&'a Value>, Option<&'a Value>) {
    if let (Some(cold), Some(gpu)) = (cold.as_object(), gpu.as_object()) {
        for (key, value) in cold {
            let child = format!("{path}.{key}");
            match gpu.get(key) {
                None => return (child, Some(value), None),
                Some(other) if value != other => return first_difference(child, value, other),
                Some(_) => (),
            }
        }
        if let Some((key, value)) = gpu.iter().find(|(key, _)| !cold.contains_key(*key)) {
            return (format!("{path}.{key}"), None, Some(value));
        }
    } else if let (Some(cold), Some(gpu)) = (cold.as_array(), gpu.as_array()) {
        for index in 0..cold.len().max(gpu.len()) {
            let child = format!("{path}[{index}]");
            match (cold.get(index), gpu.get(index)) {
                (Some(left), Some(right)) if left != right => {
                    return first_difference(child, left, right);
                }
                (None, Some(right)) => return (child, None, Some(right)),
                (Some(left), None) => return (child, Some(left), None),
                _ => (),
            }
        }
    }
    (path, Some(cold), Some(gpu))
}
