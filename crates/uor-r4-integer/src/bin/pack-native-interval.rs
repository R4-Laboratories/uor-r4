use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, Cursor, Read, Seek},
    path::Path,
};

use uor_r4_integer::discrete_d11::{
    native_interval::{self, Arm, WEIGHT_BYTES},
    native_termination, IntegerDiscreteModel,
};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> CliResult<()> {
    const USAGE: &str =
        "usage: pack-native-interval ARM BALANCED.d11 WORDS.i16le OUTPUT.d11 FIT_SHA256\n\
        ARM: geometry, count, directed_geometry, directed_count, or swapped_geometry";
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments
        .first()
        .is_some_and(|arg| arg == "ordinary-termination")
    {
        return pack_termination(&arguments[1..]);
    }
    let [arm, base, words, output, fit] = arguments.as_slice() else {
        return Err(USAGE.into());
    };
    let arm = match arm.to_str() {
        Some("geometry") => Arm::Geometry,
        Some("count") => Arm::Count,
        Some("directed_geometry") => Arm::DirectedGeometry,
        Some("directed_count") => Arm::DirectedCount,
        Some("swapped_geometry") => Arm::SwappedGeometry,
        _ => return Err(USAGE.into()),
    };
    let fit = hex::decode(fit.to_str().ok_or(USAGE)?)?
        .try_into()
        .map_err(|_| "FIT_SHA256 must decode to 32 bytes")?;
    let mut word_file = File::open(words)?;
    let mut words = [0; WEIGHT_BYTES];
    word_file.read_exact(&mut words)?;
    if word_file.read(&mut [0])? != 0 {
        return Err("interval words have trailing bytes".into());
    }

    let output = Path::new(output);
    let mut staged_name = output
        .file_name()
        .ok_or("missing output filename")?
        .to_owned();
    staged_name.push(format!(".stage-{}", std::process::id()));
    let staged = output.with_file_name(staged_name);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&staged)?;
    let publish = (|| -> CliResult<_> {
        let base = BufReader::new(File::open(base)?);
        native_interval::write_artifact(base, Cursor::new(words), arm, fit, &mut file)?;
        file.sync_all()?;
        let artifact_bytes = file.metadata()?.len();
        file.rewind()?;
        let model = IntegerDiscreteModel::from_reader(BufReader::new(&mut file))?;
        let identity = native_interval::identity(&model).ok_or("missing interval state")?;
        if identity.arm != arm || native_interval::persisted_words(&model)? != words {
            return Err("interval arm or words differ after cold reload".into());
        }
        let receipt = serde_json::json!({
            "schema": model.artifact_schema(),
            "artifact_sha256": model.artifact_sha256(),
            "artifact_bytes": artifact_bytes,
            "interval": identity,
        });
        fs::hard_link(&staged, output)?;
        Ok(receipt)
    })();
    drop(file);
    let cleanup = fs::remove_file(staged);
    let receipt = publish?;
    cleanup?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

fn pack_termination(arguments: &[std::ffi::OsString]) -> CliResult<()> {
    const USAGE: &str = "usage: pack-native-interval ordinary-termination BASE.d11 BASE_SHA256 (geometry|ordered) WORDS.i16le OUTPUT.d11 FIT_SHA256";
    let [base, base_sha256, arm, words, output, fit] = arguments else {
        return Err(USAGE.into());
    };
    let arm = match arm.to_str() {
        Some("geometry") => native_termination::Arm::Geometry,
        Some("ordered") => native_termination::Arm::Ordered,
        _ => return Err(USAGE.into()),
    };
    let base_sha256 = base_sha256.to_str().ok_or(USAGE)?;
    let fit = hex::decode(fit.to_str().ok_or(USAGE)?)?
        .try_into()
        .map_err(|_| "FIT_SHA256 must decode to 32 bytes")?;
    let mut word_file = File::open(words)?;
    let mut words = [0; native_termination::WEIGHT_BYTES];
    word_file.read_exact(&mut words)?;
    if word_file.read(&mut [0])? != 0 {
        return Err("ordinary termination words have trailing bytes".into());
    }

    let output = Path::new(output);
    let mut staged_name = output
        .file_name()
        .ok_or("missing output filename")?
        .to_owned();
    staged_name.push(format!(".stage-{}", std::process::id()));
    let staged = output.with_file_name(staged_name);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&staged)?;
    let publish = (|| -> CliResult<_> {
        let base = BufReader::new(File::open(base)?);
        native_termination::write_artifact(
            base,
            base_sha256,
            Cursor::new(words),
            arm,
            fit,
            &mut file,
        )?;
        file.sync_all()?;
        let artifact_bytes = file.metadata()?.len();
        file.rewind()?;
        let model = IntegerDiscreteModel::from_reader(BufReader::new(&mut file))?;
        let identity = native_termination::identity(&model).ok_or("missing termination state")?;
        if identity.arm != arm || native_termination::persisted_words(&model)? != words {
            return Err("ordinary termination arm or words differ after cold reload".into());
        }
        let receipt = serde_json::json!({
            "schema": model.artifact_schema(),
            "artifact_sha256": model.artifact_sha256(),
            "artifact_bytes": artifact_bytes,
            "ordinary_termination": identity,
        });
        fs::hard_link(&staged, output)?;
        Ok(receipt)
    })();
    drop(file);
    let cleanup = fs::remove_file(staged);
    let receipt = publish?;
    cleanup?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
