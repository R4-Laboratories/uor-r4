#![forbid(unsafe_code)]
use serde::Deserialize;
use std::{error::Error, fs, path::Path};
use uor_r4_tokenizer::ByteBpeTokenizer;

#[derive(Deserialize)]
struct Capture {
    emitted: Vec<u32>,
    stop: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [tokenizer, directory] = args.as_slice() else {
        return Err("check_client_bytes TOKENIZER CALIBRATION_DIRECTORY".into());
    };
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or("invalid tokenizer")?;
    let dir = Path::new(directory);
    for id in 1..=3 {
        let capture: Capture =
            serde_json::from_slice(&fs::read(dir.join(format!("{id}.gpu.json")))?)?;
        let ids = match capture.stop.as_str() {
            "eos" => capture
                .emitted
                .strip_suffix(&[1])
                .ok_or("missing terminal EOS")?,
            "invalid_bos" => capture
                .emitted
                .strip_suffix(&[0])
                .ok_or("missing terminal BOS")?,
            "cap" if capture.emitted.len() == 32 => capture.emitted.as_slice(),
            _ => return Err("unknown stop or inconsistent token limit".into()),
        };
        if ids
            .iter()
            .any(|&i| i <= 1 || i as usize >= tok.vocab_size())
            || tok.decode_bytes(ids) != fs::read(dir.join(format!("{id}.response.bytes")))?
        {
            return Err(format!("cold client byte mismatch at article {id}").into());
        }
    }
    Ok(())
}
