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
    native_source_choice::{self, Row},
    IntegerDiscreteModel,
};

#[cfg(feature = "release-factorial-diagnostic")]
#[path = "shared/verify_release_factorial.rs"]
mod verify_release_factorial;

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Serialize)]
struct Observation<'a> {
    #[serde(flatten)]
    row: &'a Row,
    #[serde(flatten)]
    replay: native_source_choice::Replay,
}

fn main() -> CliResult<()> {
    const USAGE: &str = "usage: verify-native-source-choice ARTIFACT PANEL.jsonl [DEVICE.jsonl]";
    let args: Vec<_> = env::args_os().skip(1).collect();
    #[cfg(feature = "release-factorial-diagnostic")]
    if let Some((mode, args)) = args.split_first() {
        if mode == "--release-factorial" {
            return verify_release_factorial::run(
                args,
                "verify-native-source-choice --release-factorial",
            );
        }
    }
    let (artifact, panel, mut device) = match args.as_slice() {
        [artifact, panel] => (artifact, panel, None),
        [artifact, panel, device] => (artifact, panel, Some(BufReader::new(File::open(device)?))),
        _ => return Err(USAGE.into()),
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    let identity =
        native_source_choice::identity(&model).ok_or("missing packed source-choice state")?;
    let mut line = Vec::new();
    let mut panel = BufReader::new(File::open(panel)?);
    let mut rows = Vec::new();
    while read_line(&mut panel, &mut line)? {
        let row = serde_json::from_slice(&line)
            .map_err(|error| format!("invalid panel row {}: {error}", rows.len() + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("empty source-choice panel".into());
    }
    let pairs = match_peers(&rows)?;
    let mut output = BufWriter::new(io::stdout().lock());
    for (index, row) in rows.iter().enumerate() {
        let pair = pairs
            .get(row.pair_id.as_str())
            .ok_or("row missing from peer index")?;
        let peer = pair[1 - row.side].ok_or("missing matched peer")?;
        let replay = native_source_choice::observe(&model, row, &rows[peer])
            .map_err(|error| format!("row {} ({:?}): {error}", index + 1, row.pair_id))?;
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
            "source_choice": identity,
            "decoding": native_source_choice::DECODING,
            "pairs": pairs.len(),
            "rows": rows.len(),
            "device_exact_rows": device.map(|_| rows.len()),
        })
    );
    Ok(())
}

fn match_peers(rows: &[Row]) -> CliResult<BTreeMap<&str, [Option<usize>; 2]>> {
    let mut pairs: BTreeMap<&str, [Option<usize>; 2]> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let sides = pairs.entry(&row.pair_id).or_default();
        let slot = sides.get_mut(row.side).ok_or("pair side must be 0 or 1")?;
        if slot.replace(index).is_some() {
            return Err(format!("duplicate pair side for {:?}", row.pair_id).into());
        }
    }
    if pairs.values().any(|sides| sides.contains(&None)) {
        return Err("pair missing a matched side".into());
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
