//! Lossless transport-operator binding for the frozen UORSTT01 research reader.
#![forbid(unsafe_code)]
use std::{env, error::Error, fs, io::Write, path::Path};

const HEADER: usize = 16;
const STATE_BYTES: usize = 13_336;

fn state_ok(bytes: &[u8]) -> bool {
    bytes.len() == STATE_BYTES && bytes.starts_with(b"UORSTT01")
}

fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)
}

fn pack(name: &str, state: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let tag = match name {
        "endpoint" => 0,
        "ordered" => 1,
        _ => return Err("unknown transport operator".into()),
    };
    if !state_ok(state) {
        return Err("invalid native state".into());
    }
    let mut bytes = Vec::with_capacity(HEADER + STATE_BYTES);
    bytes.extend_from_slice(b"UORBRG01");
    bytes.push(tag);
    bytes.resize(HEADER, 0);
    bytes.extend_from_slice(state);
    Ok(bytes)
}

fn unpack(bytes: &[u8]) -> Result<(&str, &[u8]), Box<dyn Error>> {
    if bytes.len() != HEADER + STATE_BYTES
        || !bytes.starts_with(b"UORBRG01")
        || bytes[9..HEADER].iter().any(|byte| *byte != 0)
        || !state_ok(&bytes[HEADER..])
    {
        return Err("invalid transport bundle".into());
    }
    match bytes[8] {
        0 => Ok(("endpoint", &bytes[HEADER..])),
        1 => Ok(("ordered", &bytes[HEADER..])),
        _ => Err("unrecognized transport version".into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [op, name, input, output] if op == "pack" => {
            write_new(Path::new(output), &pack(name, &fs::read(input)?)?)?;
        }
        [op, input, output] if op == "unpack" => {
            let bytes = fs::read(input)?;
            let (name, state) = unpack(&bytes)?;
            write_new(Path::new(output), state)?;
            println!("{name}");
        }
        _ => return Err("use pack OPERATOR STATE OUT or unpack BUNDLE STATE_OUT".into()),
    }
    Ok(())
}
