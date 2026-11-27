use std::{
    env,
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{BufReader, Seek, Write},
    path::Path,
};

use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::{native_occurrence, native_opening, IntegerDiscreteModel};
use uor_r4_tokenizer::ByteBpeTokenizer;

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> CliResult<()> {
    const USAGE: &str = "usage: pack-native-occurrence classes TOKENIZER.json OUT.u8\n       pack-native-occurrence pack MARKER.d11 WEIGHTS.i16le CLASSES.u8 OUT.d11 FIT_SHA256";
    let args: Vec<_> = env::args_os().skip(1).collect();
    match args.as_slice() {
        [command, tokenizer, output] if command == OsStr::new("classes") => {
            write_classes(Path::new(tokenizer), Path::new(output))
        }
        [command, marker, weights, classes, output, fit] if command == OsStr::new("pack") => {
            let fit = hex::decode(fit.to_str().ok_or(USAGE)?)?
                .try_into()
                .map_err(|_| "FIT_SHA256 must decode to 32 bytes")?;
            pack(marker, weights, classes, Path::new(output), fit)
        }
        _ => Err(USAGE.into()),
    }
}

fn write_classes(tokenizer: &Path, output: &Path) -> CliResult<()> {
    let bytes = fs::read(tokenizer)?;
    if hex::encode(Sha256::digest(&bytes)) != native_opening::CANONICAL_TOKENIZER_SHA256 {
        return Err("token classes require the canonical tokenizer".into());
    }
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes)
        .ok_or("could not load the canonical tokenizer")?;
    if tokenizer.vocab_size() != native_occurrence::VOCABULARY {
        return Err("unexpected canonical vocabulary".into());
    }
    let mut classes = [0; native_occurrence::VOCABULARY];
    for (id, class) in classes.iter_mut().enumerate() {
        let id = id as u32;
        *class = native_occurrence::classify_token(id, &tokenizer.decode_bytes(&[id]));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&classes)?;
    file.sync_all()?;
    println!("{}", hex::encode(Sha256::digest(classes)));
    Ok(())
}

fn pack(
    marker: &OsStr,
    weights: &OsStr,
    classes: &OsStr,
    output: &Path,
    fit: [u8; 32],
) -> CliResult<()> {
    let marker = BufReader::new(File::open(marker)?);
    let weights = BufReader::new(File::open(weights)?);
    let classes = BufReader::new(File::open(classes)?);
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
        native_occurrence::write_artifact(marker, weights, classes, fit, &mut file)?;
        file.sync_all()?;
        let artifact_bytes = file.metadata()?.len();
        file.rewind()?;
        let model = IntegerDiscreteModel::from_reader(BufReader::new(&mut file))?;
        let identity = model
            .native_occurrence_identity()
            .ok_or("artifact has no native occurrence state")?;
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
