// Extract the two immutable native tables used by the proof readers.
use std::{env, error::Error, fs, io::Write, path::Path};

fn table(source: &str, name: &str, length: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let marker = format!("const {name}: [u8; {length}] = [");
    let (_, body) = source.split_once(&marker).ok_or("missing native table")?;
    let (values, _) = body.split_once("];").ok_or("unterminated table")?;
    let values: Vec<u8> = values
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if values.len() != length || values.iter().any(|&v| v >= 64) {
        return Err("invalid native table".into());
    }
    Ok(values)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let [source, out] = args.as_slice() else {
        return Err("usage: table-bytes discrete_d11.rs FRESH_OUTPUT".into());
    };
    let source = fs::read_to_string(source)?;
    let mut file = fs::File::create_new(Path::new(out))?;
    file.write_all(&table(&source, "BFS", 256)?)?;
    file.write_all(&table(&source, "COMPOSE", 4096)?)?;
    file.sync_all()?;
    Ok(())
}
