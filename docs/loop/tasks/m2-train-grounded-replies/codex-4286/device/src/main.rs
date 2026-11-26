#![forbid(unsafe_code)]

mod fixture;
mod packet;
mod prepare;
mod rank;

use sha2::{Digest, Sha256};
use std::{env, fs, io::Write, path::Path};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MODEL_SHA: &str = "5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8";
const FIT_SHA: &str = "ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663";
const CODEC_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(reason.to_owned().into())
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked(path: &str, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        sha(&bytes) == expected,
        &format!("identity changed for {path}"),
    )?;
    Ok(bytes)
}

fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::File::create_new(path)?.write_all(bytes)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["prepare", reference, artifact, codec, fit, out] => {
            prepare::run(reference, artifact, codec, fit, Path::new(out))
        }
        ["grade", prepared, artifact, fit, out] => {
            let fit = rank::pooled(&checked(fit, FIT_SHA)?)?;
            prepare::verify_model_banks(artifact, &fit)?;
            rank::grade(Path::new(prepared), &fit, Path::new(out))
        }
        ["fixture", out] => fixture::write(Path::new(out)),
        ["grade-fixture", root] => fixture::check(Path::new(root)),
        _ => Err("usage: prepare REFERENCE D11 TOKENIZER FIT NEW_DIR | grade PREPARED D11 FIT SCORED | fixture NEW_DIR | grade-fixture ROOT".into()),
    }
}
