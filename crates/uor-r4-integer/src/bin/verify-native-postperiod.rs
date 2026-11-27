//! Cold-load the frozen #4319 writer and compare paired post-period replay.

use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde::Serialize;
use serde_json::Value;
use uor_r4_integer::discrete_d11::{
    native_postperiod::{self, Row},
    native_word_ending, IntegerDiscreteModel,
};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Serialize)]
struct Observation<'a> {
    #[serde(flatten)]
    row: &'a Row,
    #[serde(flatten)]
    replay: native_postperiod::Replay,
}

fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> CliResult<bool> {
    const MAX_LINE_BYTES: usize = 64 * 1024;
    line.clear();
    let read = reader
        .take((MAX_LINE_BYTES + 1) as u64)
        .read_until(b'\n', line)?;
    if read > MAX_LINE_BYTES {
        return Err("post-period replay row exceeds 64 KiB".into());
    }
    Ok(read != 0)
}

fn compare(cold: &Value, device: &Value, path: &str) -> CliResult<()> {
    if cold == device {
        return Ok(());
    }
    match (cold, device) {
        (Value::Object(left), Value::Object(right)) if left.keys().eq(right.keys()) => {
            for (key, value) in left {
                compare(value, &right[key], &format!("{path}.{key}"))?;
            }
        }
        (Value::Array(left), Value::Array(right)) if left.len() == right.len() => {
            for (index, (l, r)) in left.iter().zip(right).enumerate() {
                compare(l, r, &format!("{path}[{index}]"))?;
            }
        }
        _ => return Err(format!("{path} differs, cold {cold}, device {device}").into()),
    }
    Ok(())
}

fn compare_row(cold: &Value, device: &Value) -> CliResult<()> {
    let cold = cold.as_object().ok_or("invalid cold replay row")?;
    let device = device.as_object().ok_or("invalid device replay row")?;
    for (field, value) in cold {
        if field == "attribution" {
            continue;
        }
        let actual = device
            .get(field)
            .ok_or_else(|| format!("missing device {field}"))?;
        if field == "third_trace" {
            let trace = value.as_object().ok_or("invalid cold third trace")?;
            for (key, expected) in trace {
                let actual = actual
                    .get(key)
                    .ok_or_else(|| format!("missing device {field}.{key}"))?;
                compare(expected, actual, &format!("{field}.{key}"))?;
            }
        } else {
            compare(value, actual, field)?;
        }
    }
    Ok(())
}

fn main() -> CliResult<()> {
    const USAGE: &str = "usage: verify-native-postperiod ARTIFACT PANEL.jsonl [DEVICE.jsonl]";
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (artifact, panel, mut device) = match args.as_slice() {
        [artifact, panel] => (artifact, panel, None),
        [artifact, panel, device] => (artifact, panel, Some(BufReader::new(File::open(device)?))),
        _ => return Err(USAGE.into()),
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    if model.artifact_sha256() != native_postperiod::ARTIFACT_SHA256 {
        return Err("post-period replay requires the identical #4319 qualified artifact".into());
    }
    let identity =
        native_word_ending::identity(&model).ok_or("missing packed word-ending state")?;
    let mut line = Vec::new();
    let mut panel = BufReader::new(File::open(panel)?);
    let mut rows: Vec<Row> = Vec::new();
    while read_line(&mut panel, &mut line)? {
        let row = serde_json::from_slice(&line)
            .map_err(|error| format!("invalid panel row {}: {error}", rows.len() + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("empty post-period panel".into());
    }
    let mut pairs: BTreeMap<_, [Option<usize>; 2]> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let slot = pairs
            .entry(row.pair_id.as_str())
            .or_default()
            .get_mut(row.side)
            .ok_or("post-period side must be 0 or 1")?;
        if slot.replace(index).is_some() {
            return Err(format!("duplicate side for {:?}", row.pair_id).into());
        }
    }
    if pairs.values().any(|pair| pair.iter().any(Option::is_none)) {
        return Err("post-period tape has an unpaired prompt".into());
    }
    let mut output = BufWriter::new(io::stdout().lock());
    let mut trajectories = 0usize;
    let mut transitions = 0usize;
    let mut terminals = 0usize;
    for (index, row) in rows.iter().enumerate() {
        let peer = pairs[row.pair_id.as_str()][1 - row.side].ok_or("missing matched peer")?;
        let replay = native_postperiod::observe(&model, row, &rows[peer]).map_err(|error| {
            format!(
                "row {} ({}, side {}) {error}",
                index + 1,
                row.pair_id,
                row.side
            )
        })?;
        for arm in [
            &replay.arms.native,
            &replay.arms.correct,
            &replay.arms.swapped,
            &replay.arms.supplied_next,
        ] {
            trajectories += 1;
            transitions += arm.steps.len();
            terminals += usize::from(arm.terminal);
        }
        let observation = serde_json::to_value(Observation { row, replay })?;
        if let Some(input) = device.as_mut() {
            if !read_line(input, &mut line)? {
                return Err(format!("device replay ended before row {}", index + 1).into());
            }
            compare_row(&observation, &serde_json::from_slice(&line)?).map_err(|error| {
                format!(
                    "row {} ({}, side {}) {error}",
                    index + 1,
                    row.pair_id,
                    row.side
                )
            })?;
        }
        serde_json::to_writer(&mut output, &observation)?;
        output.write_all(b"\n")?;
    }
    if let Some(input) = device.as_mut() {
        if read_line(input, &mut line)? {
            return Err("device replay contains extra rows".into());
        }
    }
    output.flush()?;
    eprintln!(
        "{}",
        serde_json::json!({
            "artifact_sha256": model.artifact_sha256(),
            "word_ending": identity,
            "pairs": pairs.len(),
            "rows": rows.len(),
            "device_exact_rows": device.map(|_| rows.len()),
            "trajectories": trajectories,
            "writer_transitions": transitions,
            "terminal_trajectories": terminals,
        })
    );
    Ok(())
}
