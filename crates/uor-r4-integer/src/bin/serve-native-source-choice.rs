use std::{
    env,
    io::{self, BufRead, BufWriter, Read, Write},
    path::Path,
};

use serde::Deserialize;
use uor_r4_integer::discrete_d11::{native_source_choice, IntegerDiscreteModel};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    prefix: Vec<u32>,
    positions: [u32; 4],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [artifact] = args.as_slice() else {
        return Err("usage: serve-native-source-choice ARTIFACT < REQUESTS.jsonl".into());
    };
    let model = IntegerDiscreteModel::load(Path::new(artifact))?;
    if native_source_choice::identity(&model).is_none() {
        return Err("missing packed native source-choice state".into());
    }
    let mut input = io::stdin().lock();
    let mut output = BufWriter::new(io::stdout().lock());
    let mut line = Vec::new();
    const LIMIT: u64 = 64 * 1024;
    loop {
        line.clear();
        let bytes = (&mut input).take(LIMIT + 1).read_until(b'\n', &mut line)?;
        if bytes == 0 {
            break;
        }
        if bytes as u64 > LIMIT {
            return Err("source-choice request exceeds 64 KiB".into());
        }
        let request: Request = serde_json::from_slice(&line)?;
        let served = native_source_choice::serve(&model, &request.prefix, request.positions)?;
        serde_json::to_writer(&mut output, &served)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
