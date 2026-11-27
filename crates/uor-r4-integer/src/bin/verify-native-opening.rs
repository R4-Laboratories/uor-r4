//! Cold-load the fitted artifact and replay rows through the product writer.

use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use uor_r4_integer::discrete_d11::{native_opening::Observation, IntegerDiscreteModel};

#[derive(Deserialize, Serialize)]
struct PanelRow {
    pair_id: String,
    side: u32,
    assignment: Option<u32>,
    group_id: Option<String>,
    cohort: Option<String>,
    #[serde(skip_serializing)]
    prefix: Vec<u32>,
    #[serde(skip_serializing)]
    target: u32,
    #[serde(skip_serializing)]
    rival: u32,
}

#[derive(Serialize)]
struct RowReport<'a> {
    #[serde(flatten)]
    panel: &'a PanelRow,
    #[serde(flatten)]
    opening: Option<Observation>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: verify-native-opening ARTIFACT PANEL.jsonl";
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [artifact, panel] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    let identity = model
        .native_opening_identity()
        .ok_or("artifact has no opening state")?;
    let panel = BufReader::new(File::open(panel)?);
    let mut output = BufWriter::new(io::stdout().lock());
    for (line_index, line) in panel.lines().enumerate() {
        let line = line?;
        let report_line = || -> Result<_, Box<dyn std::error::Error>> {
            let panel: PanelRow = serde_json::from_str(&line)?;
            let start = panel
                .prefix
                .len()
                .saturating_sub(model.shape().context as usize);
            let prefix = &panel.prefix[start..];
            let opening = model.observe_native_opening(prefix, panel.target, panel.rival)?;
            let in_domain = opening.is_some();
            let mut record = serde_json::to_value(RowReport {
                panel: &panel,
                opening,
            })?;
            if !in_domain {
                let served = model.evaluate_prefix_traced(prefix)?;
                record["selected"] =
                    serde_json::to_value(served.trunk_pointer.map(|gate| gate.selected_token))?;
                record["emitted"] = served.evaluation.prediction.into();
            }
            record["state_sha256"] = identity.state_sha256.into();
            Ok(record)
        };
        let record = report_line().map_err(|e| format!("panel row {}: {e}", line_index + 1))?;
        serde_json::to_writer(&mut output, &record)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}
