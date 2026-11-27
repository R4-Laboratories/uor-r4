//! Time warm, sequential requests through the unchanged ordinary client on Wick.
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    process::{Command, Stdio},
    time::Instant,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [client, artifact, input, output] = args.as_slice() else {
        return Err("time-client SERVE ARTIFACT REQUESTS NEW_RESPONSES".into());
    };
    let requests: Vec<String> = BufReader::new(File::open(input)?)
        .lines()
        .collect::<Result<_, _>>()?;
    let warm = requests.first().ok_or("empty request tape")?;
    let mut child = Command::new(client)
        .arg(artifact)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut tx = child.stdin.take().ok_or("no client stdin")?;
    let mut rx = BufReader::new(child.stdout.take().ok_or("no client stdout")?);
    let mut buffer = Vec::new();
    let mut out = BufWriter::new(File::create_new(output)?);
    let mut elapsed = 0u128;
    let mut warm_elapsed = 0u128;
    for (index, prompt) in std::iter::once(warm).chain(requests.iter()).enumerate() {
        buffer.clear();
        let now = Instant::now();
        writeln!(tx, "{prompt}")?;
        tx.flush()?;
        if rx.read_until(b'\n', &mut buffer)? == 0 || buffer.last() != Some(&b'\n') {
            return Err("client returned an incomplete response".into());
        }
        let nanos = now.elapsed().as_nanos();
        if index != 0 {
            elapsed += nanos;
            out.write_all(&buffer)?;
        } else {
            warm_elapsed = nanos;
        }
    }
    out.flush()?;
    drop(tx);
    if !child.wait()?.success() {
        return Err("ordinary client failed".into());
    }
    println!(
        "{{\"requests\":{},\"startup_and_first_request_ns\":{},\"warm_request_ns\":{}}}",
        requests.len(),
        warm_elapsed,
        elapsed
    );
    Ok(())
}
