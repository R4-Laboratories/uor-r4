#![forbid(unsafe_code)]
use serde::Deserialize;
use serde_json::{json, Value};
use std::{error::Error, fs, path::Path};
use uor_r4_integer::{discrete_d11::IntegerDiscreteModel, paired_role::RoleSpan};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Row {
    case_id: String,
    prefix: Vec<u32>,
    user_bounds: [usize; 2],
    question_start: usize,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [artifact, mode, tape] = args.as_slice() else {
        return Err("usage: delimiter-relaxation-study D26 [capture|control] TAPE".into());
    };
    let carrier = IntegerDiscreteModel::load(Path::new(artifact))?;
    let tape = fs::read_to_string(tape)?;
    match mode.as_str() {
        "capture" => {
            for line in tape.lines() {
                let row: Row = serde_json::from_str(line)?;
                let [start, end] = row.user_bounds;
                let user = RoleSpan { start, end };
                let tail = start.max(end.saturating_sub(32));
                println!(
                    "{}",
                    json!({
                        "case_id": row.case_id, "carrier": carrier.artifact_sha256(),
                        "separator": carrier.delimiter_relaxation_study(&row.prefix, user, row.question_start)?,
                        "tail32": carrier.delimiter_relaxation_study(&row.prefix, user, tail)?,
                    })
                );
            }
        }
        "control" => {
            let original: Value = serde_json::from_str(&tape)?;
            let ids: Vec<u32> = serde_json::from_value(original["prompt_ids"].clone())?;
            let decision = carrier
                .paired_copy_decision(&ids, true)?
                .ok_or("no control binding")?;
            if serde_json::to_value(&decision)? != original["paired"]
                || decision.choice.source != Some(51)
                || original["artifact_sha256"] != carrier.artifact_sha256()
            {
                return Err("historical native positive differs".into());
            }
            let cut = decision.choice.query.ok_or("no control query")? as usize;
            let user = RoleSpan {
                start: 1,
                end: ids.len(),
            };
            let census = carrier.delimiter_relaxation_study(&ids, user, cut)?;
            let traces = census["traces"].as_array().ok_or("no control traces")?;
            let match_count = traces
                .iter()
                .filter(|trace| {
                    trace["preceding_delimiter"] == true
                        && trace["query"][0] == cut
                        && trace["selection"]["source"] == 51
                })
                .count();
            if match_count == 0 {
                return Err("public native trace did not reproduce the historical source".into());
            }
            println!(
                "{}",
                json!({
                    "carrier": carrier.artifact_sha256(),
                    "historical_source": 51,
                    "native_traces_at_historical_query_selecting_source": match_count,
                    "native_equal_traces": census["native_assertions"],
                })
            );
        }
        _ => return Err("unknown mode".into()),
    }
    Ok(())
}
