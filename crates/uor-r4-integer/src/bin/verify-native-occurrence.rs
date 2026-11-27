use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uor_r4_integer::discrete_d11::{native_occurrence::Replay, IntegerDiscreteModel};

#[derive(Deserialize)]
struct PanelRow {
    prefix: Vec<u32>,
    raw_target: Option<u32>,
    target: u32,
    gold_pos: Option<u16>,
    rival_pos: Option<u16>,
    #[serde(default)]
    pair_id: Option<String>,
    #[serde(default)]
    group_id: Option<String>,
    #[serde(default)]
    side: Option<u8>,
    context_start: Option<u16>,
    context_end: Option<u16>,
}

#[derive(Serialize)]
struct Observation<'a> {
    index: usize,
    pair_id: &'a Option<String>,
    group_id: &'a Option<String>,
    side: Option<u8>,
    context_start: Option<u16>,
    context_end: Option<u16>,
    prefix: &'a [u32],
    raw_target: Option<u32>,
    target: u32,
    gold_pos: Option<u16>,
    rival_pos: Option<u16>,
    #[serde(flatten)]
    replay: Replay,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: verify-native-occurrence ARTIFACT PANEL.jsonl [DEVICE.jsonl]";
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (artifact, panel, device) = match args.as_slice() {
        [artifact, panel] => (artifact, panel, None),
        [artifact, panel, device] => (artifact, panel, Some(BufReader::new(File::open(device)?))),
        _ => return Err(USAGE.into()),
    };
    let mut device = device;
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    let identity = model
        .native_occurrence_identity()
        .ok_or("artifact has no native occurrence state")?;
    eprintln!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "artifact_sha256": model.artifact_sha256(),
            "occurrence": identity,
        }))?
    );
    let file = File::open(panel)?;
    let mut panel = BufReader::new(file);
    let mut output = BufWriter::new(io::stdout().lock());
    let mut line = Vec::new();
    let mut expected_line = Vec::new();
    let mut index = 0;
    while read_line(&mut panel, &mut line)? {
        let row: PanelRow = serde_json::from_slice(&line)
            .map_err(|error| format!("invalid panel row {index}: {error}"))?;
        let replay = model
            .observe_native_occurrence(
                &row.prefix,
                row.raw_target,
                row.target,
                row.gold_pos,
                row.rival_pos,
            )
            .map_err(|error| format!("invalid panel row {index}: {error}"))?;
        let observation = serde_json::to_value(Observation {
            index,
            pair_id: &row.pair_id,
            group_id: &row.group_id,
            side: row.side,
            context_start: row.context_start,
            context_end: row.context_end,
            prefix: &row.prefix,
            raw_target: row.raw_target,
            target: row.target,
            gold_pos: row.gold_pos,
            rival_pos: row.rival_pos,
            replay,
        })?;
        serde_json::to_writer(&mut output, &observation)?;
        output.write_all(b"\n")?;
        if let Some(device) = device.as_mut() {
            if !read_line(device, &mut expected_line)? {
                return Err(format!("device replay ended before row {index}").into());
            }
            let expected: Value = serde_json::from_slice(&expected_line)?;
            compare(index, &observation, &expected)?;
        }
        index += 1;
    }
    output.flush()?;
    if let Some(device) = device.as_mut() {
        if read_line(device, &mut expected_line)? {
            return Err(format!("device replay has extra rows after {index} matches").into());
        }
        eprintln!("{{\"device_exact_rows\":{index}}}");
    }
    Ok(())
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

fn compare(index: usize, cold: &Value, device: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let observed = cold.as_object().ok_or("invalid cold observation")?;
    for (field, actual) in observed {
        if field == "index" {
            continue;
        }
        let expected = device
            .get(field)
            .ok_or_else(|| format!("device row {index} omits {field}"))?;
        if actual != expected {
            if field == "candidates" {
                let difference = actual
                    .as_array()
                    .zip(expected.as_array())
                    .and_then(|(a, b)| {
                        a.iter()
                            .zip(b)
                            .position(|(actual, expected)| actual != expected)
                    });
                if let Some(at) = difference {
                    return Err(format!(
                        "row {index} candidate index {at} disagrees\ncold: {}\ndevice: {}",
                        actual[at], expected[at]
                    )
                    .into());
                }
                return Err(format!("row {index} candidate lengths disagree").into());
            }
            return Err(format!(
                "row {index} {field} disagrees\ncold: {actual}\ndevice: {expected}"
            )
            .into());
        }
    }
    Ok(())
}
