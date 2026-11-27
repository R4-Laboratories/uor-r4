use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;
use uor_r4_integer::discrete_d11::{native_copy, IntegerDiscreteModel};

#[derive(Deserialize)]
struct PanelRow {
    pair_id: String,
    prefix: Vec<u32>,
    gold_pos: Option<u16>,
    rival_pos: Option<u16>,
    wrong_same_id_pos: Option<u16>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: verify-native-copy ARTIFACT PANEL.jsonl [DEVICE.jsonl]";
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (artifact, panel, mut device) = match args.as_slice() {
        [artifact, panel] => (artifact, panel, None),
        [artifact, panel, device] => (artifact, panel, Some(BufReader::new(File::open(device)?))),
        _ => return Err(USAGE.into()),
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    let identity = model
        .native_occurrence_identity()
        .ok_or("authenticated copy requires a native occurrence artifact")?;
    let mut panel = BufReader::new(File::open(panel)?);
    let mut output = BufWriter::new(io::stdout().lock());
    let mut line = Vec::new();
    let mut expected_line = Vec::new();
    let mut count = 0;
    while read_line(&mut panel, &mut line)? {
        let row: PanelRow = serde_json::from_slice(&line)
            .map_err(|error| format!("invalid panel row {count}: {error}"))?;
        let arms = native_copy::observe(
            &model,
            &row.prefix,
            row.gold_pos,
            row.rival_pos,
            row.wrong_same_id_pos,
        )
        .map_err(|error| format!("invalid panel row {count} ({:?}): {error}", row.pair_id))?;
        let observation =
            serde_json::json!({"pair_id":row.pair_id,"prefix":row.prefix,"arms":arms});
        if let Some(device) = device.as_mut() {
            if !read_line(device, &mut expected_line)? {
                return Err(format!("device replay ended before row {count}").into());
            }
            let expected: Value = serde_json::from_slice(&expected_line)?;
            if observation != expected {
                let (path, cold, gpu) = first_difference("$".into(), &observation, &expected);
                return Err(format!(
                    "row {count} ({:?}) {path} differs, cold {cold:?}, device {gpu:?}",
                    row.pair_id
                )
                .into());
            }
        }
        serde_json::to_writer(&mut output, &observation)?;
        output.write_all(b"\n")?;
        count += 1;
    }
    if count == 0 {
        return Err("empty panel".into());
    }
    if let Some(device) = device.as_mut() {
        if read_line(device, &mut expected_line)? {
            return Err(format!("device replay has extra rows after {count} matches").into());
        }
    }
    output.flush()?;
    eprintln!(
        "{}",
        serde_json::json!({
            "artifact_sha256":model.artifact_sha256(),
            "occurrence":identity,
            "action_schema":native_copy::SCHEMA,
            "rows":count,
            "device_exact_rows":device.map(|_| count),
        })
    );
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
    }
    (path, Some(cold), Some(gpu))
}
