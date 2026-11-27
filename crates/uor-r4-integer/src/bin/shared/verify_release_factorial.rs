use std::{
    ffi::OsString,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde_json::{Map, Value};
use uor_r4_integer::discrete_d11::{
    native_source_choice::{self, release_factorial, Row},
    IntegerDiscreteModel,
};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

const ROW_FIELDS: [&str; 8] = [
    "pair_id",
    "side",
    "family",
    "contrast",
    "prefix",
    "positions",
    "ids",
    "expected",
];

pub(super) fn run(args: &[OsString], command: &str) -> CliResult<()> {
    let [artifact, panel, forward, reverse, client] = args else {
        return Err(format!(
            "usage: {command} ARTIFACT PANEL.jsonl DEVICE_FORWARD.jsonl DEVICE_REVERSE.jsonl CLIENT.jsonl"
        )
        .into());
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    let identity =
        native_source_choice::identity(&model).ok_or("missing packed source-choice state")?;

    let mut reverse = BufReader::new(File::open(reverse)?);
    let mut reverse_rows = Vec::new();
    let mut line = Vec::new();
    while let Some(value) = read(&mut reverse, &mut line)? {
        reverse_rows.push(value);
    }
    let mut panel = BufReader::new(File::open(panel)?);
    let mut forward = BufReader::new(File::open(forward)?);
    let mut client = BufReader::new(File::open(client)?);
    let mut output = BufWriter::new(io::stdout().lock());
    let mut rows = 0;
    let mut learned_steps = 0;
    let mut diagnostic_steps = 0;
    while let Some(metadata) = read(&mut panel, &mut line)? {
        rows += 1;
        let mut metadata = metadata
            .as_object()
            .cloned()
            .ok_or("panel row must be a JSON object")?;
        let core = ROW_FIELDS
            .iter()
            .map(|&field| {
                let value = metadata
                    .get(field)
                    .ok_or_else(|| format!("row {rows} is missing {field}"))?;
                Ok((field.to_owned(), value.clone()))
            })
            .collect::<CliResult<Map<String, Value>>>()?;
        let row: Row = serde_json::from_value(Value::Object(core))?;
        let context = format!("row {rows} ({:?}, side {})", row.pair_id, row.side);
        let arms = release_factorial::observe(&model, &row)
            .map_err(|error| format!("{context}: {error}"))?;
        let arms = serde_json::to_value(arms)?;
        for arm in ["learned", "first", "post_b", "both"] {
            diagnostic_steps += steps(&arms[arm])?.len();
        }
        let mut learned = arms["learned"].clone();
        let ordinary_count = steps(&learned)?.len();
        for step in learned["steps"]
            .as_array_mut()
            .ok_or("missing learned steps")?
        {
            step.as_object_mut()
                .ok_or("invalid learned step")?
                .remove("release");
        }
        learned["first_mismatch"] = Value::Null;
        let ordinary = read(&mut client, &mut line)?;
        verify(&context, "ordinary client", &learned, ordinary.as_ref())?;
        learned_steps += ordinary_count;

        if metadata.insert("arms".into(), arms).is_some() {
            return Err(format!("{context} has a reserved arms field").into());
        }
        let observation = Value::Object(metadata);
        let forward_row = read(&mut forward, &mut line)?;
        let reverse_row = reverse_rows.pop();
        verify(
            &context,
            "forward device",
            &observation,
            forward_row.as_ref(),
        )?;
        verify(
            &context,
            "reverse device",
            &observation,
            reverse_row.as_ref(),
        )?;
        serde_json::to_writer(&mut output, &observation)?;
        output.write_all(b"\n")?;
    }
    if rows == 0 {
        return Err("release factorial panel is empty".into());
    }
    if !reverse_rows.is_empty()
        || read(&mut forward, &mut line)?.is_some()
        || read(&mut client, &mut line)?.is_some()
    {
        return Err(format!("comparison contains extra rows after {rows} matches").into());
    }
    output.flush()?;
    eprintln!(
        "{}",
        serde_json::json!({
            "artifact_sha256": model.artifact_sha256(),
            "source_choice": identity,
            "decoding": native_source_choice::DECODING,
            "matched_rows_per_device_direction": rows,
            "matched_arm_transitions_per_device_direction": diagnostic_steps,
            "matched_client_rows": rows,
            "matched_client_transitions": learned_steps,
        })
    );
    Ok(())
}

fn steps(trajectory: &Value) -> CliResult<&[Value]> {
    trajectory["steps"]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "missing trajectory steps".into())
}

fn read(reader: &mut impl BufRead, line: &mut Vec<u8>) -> CliResult<Option<Value>> {
    const MAX_LINE_BYTES: u64 = 1024 * 1024;
    line.clear();
    let bytes = reader.take(MAX_LINE_BYTES + 1).read_until(b'\n', line)? as u64;
    if bytes > MAX_LINE_BYTES {
        return Err("release factorial replay row exceeds the byte limit".into());
    }
    if bytes == 0 {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(line)?))
}

fn verify(context: &str, source: &str, cold: &Value, other: Option<&Value>) -> CliResult<()> {
    let other = other.ok_or_else(|| format!("{context} missing from {source}"))?;
    if cold != other {
        let (path, expected, observed) = first_difference("$".into(), cold, other);
        return Err(format!(
            "{context} {source} differs at {path}, cold {expected:?}, observed {observed:?}"
        )
        .into());
    }
    Ok(())
}

fn first_difference<'a>(
    path: String,
    cold: &'a Value,
    other: &'a Value,
) -> (String, Option<&'a Value>, Option<&'a Value>) {
    match (cold, other) {
        (Value::Object(cold), Value::Object(other)) => {
            for (key, value) in cold {
                let child = format!("{path}.{key}");
                match other.get(key) {
                    None => return (child, Some(value), None),
                    Some(found) if value != found => return first_difference(child, value, found),
                    Some(_) => (),
                }
            }
            if let Some((key, value)) = other.iter().find(|(key, _)| !cold.contains_key(*key)) {
                return (format!("{path}.{key}"), None, Some(value));
            }
        }
        (Value::Array(cold), Value::Array(other)) => {
            for index in 0..cold.len().max(other.len()) {
                let left = cold.get(index);
                let right = other.get(index);
                if let (Some(left), Some(right)) = (left, right) {
                    if left != right {
                        return first_difference(format!("{path}[{index}]"), left, right);
                    }
                } else if left != right {
                    return (format!("{path}[{index}]"), left, right);
                }
            }
        }
        _ => (),
    }
    (path, Some(cold), Some(other))
}
