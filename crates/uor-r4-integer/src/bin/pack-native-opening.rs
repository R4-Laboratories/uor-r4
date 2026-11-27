use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{BufReader, Seek},
    path::Path,
};

use uor_r4_integer::discrete_d11::{
    native_opening::{self, Mode},
    IntegerDiscreteModel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: pack-native-opening BASE colon|marker STATE.tsv OUT FIT_SHA256";
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [base, mode, state, output, fit] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let mode = match mode.to_str() {
        Some("colon") => Mode::Colon,
        Some("marker") => Mode::Marker,
        _ => return Err(USAGE.into()),
    };
    let fit = hex::decode(fit.to_str().ok_or(USAGE)?)?
        .try_into()
        .map_err(|_| "FIT_SHA256 must decode to 32 bytes")?;
    let base = File::open(base)?;
    let state = BufReader::new(File::open(state)?);
    let output = Path::new(output);
    let mut stage_name = output
        .file_name()
        .ok_or("missing output filename")?
        .to_owned();
    stage_name.push(format!(".stage-{}", std::process::id()));
    let stage = output.with_file_name(stage_name);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&stage)?;
    let publish = (|| -> Result<_, Box<dyn std::error::Error>> {
        native_opening::write_artifact(base, state, mode, fit, &mut file)?;
        file.sync_all()?;
        let artifact_bytes = file.metadata()?.len();
        file.rewind()?;
        let model = IntegerDiscreteModel::from_reader(BufReader::new(&mut file))?;
        let identity = model
            .native_opening_identity()
            .ok_or("artifact has no opening state")?;
        let receipt = serde_json::json!({
            "schema": model.artifact_schema(),
            "artifact_sha256": model.artifact_sha256(),
            "artifact_bytes": artifact_bytes,
            "opening": identity,
        });
        fs::hard_link(&stage, output)?;
        Ok(receipt)
    })();
    drop(file);
    let cleanup = fs::remove_file(stage);
    let receipt = publish?;
    cleanup?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
