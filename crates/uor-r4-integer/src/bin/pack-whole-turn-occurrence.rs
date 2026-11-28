use std::{
    env,
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{BufReader, Cursor, Read, Seek, Write},
    path::Path,
};

use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::{
    native_occurrence::WEIGHT_BYTES, native_whole_turn, IntegerDiscreteModel,
};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> CliResult<()> {
    const USAGE: &str = "usage: pack-whole-turn-occurrence extract BALANCED.d11 OUT.i16le\n       pack-whole-turn-occurrence pack BALANCED.d11 WORDS.i16le OUT.d11 FIT_SHA256";
    let args: Vec<_> = env::args_os().skip(1).collect();
    match args.as_slice() {
        [command, base, output] if command == OsStr::new("extract") => extract(base, output),
        [command, base, words, output, fit] if command == OsStr::new("pack") => {
            let fit = hex::decode(fit.to_str().ok_or(USAGE)?)?
                .try_into()
                .map_err(|_| "FIT_SHA256 must decode to 32 bytes")?;
            pack(base, words, Path::new(output), fit)
        }
        _ => Err(USAGE.into()),
    }
}

fn extract(base: &OsStr, output: &OsStr) -> CliResult<()> {
    let model = IntegerDiscreteModel::from_reader(BufReader::new(File::open(base)?))?;
    let words = native_whole_turn::balanced_words(&model)?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    out.write_all(&words)?;
    out.sync_all()?;
    println!("{}", hex::encode(Sha256::digest(words)));
    Ok(())
}

fn pack(base: &OsStr, words: &OsStr, output: &Path, fit: [u8; 32]) -> CliResult<()> {
    let mut word_file = File::open(words)?;
    let mut words = [0; WEIGHT_BYTES];
    word_file.read_exact(&mut words)?;
    if word_file.read(&mut [0])? != 0 {
        return Err("expected exactly 513 little-endian signed words".into());
    }
    let base = BufReader::new(File::open(base)?);
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
    let publish = (|| -> CliResult<_> {
        native_whole_turn::write_artifact(base, Cursor::new(words), fit, &mut file)?;
        file.sync_all()?;
        let artifact_bytes = file.metadata()?.len();
        file.rewind()?;
        let model = IntegerDiscreteModel::from_reader(BufReader::new(&mut file))?;
        let identity = native_whole_turn::identity(&model).ok_or("missing whole-turn state")?;
        if native_whole_turn::expanded_words(&model)? != words {
            return Err("whole-turn words differ after cold reload".into());
        }
        let receipt = serde_json::json!({
            "schema": model.artifact_schema(),
            "artifact_sha256": model.artifact_sha256(),
            "artifact_bytes": artifact_bytes,
            "occurrence": identity,
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
