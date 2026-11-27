use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[cfg(feature = "cuda")]
mod gpu;

const VOCAB: u32 = 4096;
const CONTEXT: usize = 192;
const EOS: u32 = 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prompt {
    id: String,
    domain: String,
    prefix: Vec<u32>,
    cap: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Teacher {
    id: String,
    domain: String,
    prefix: Vec<u32>,
    gold: Vec<u32>,
    new_teacher_tokens: Vec<bool>,
    cap: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    id: String,
    arm: String,
    tokens: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct Witness {
    id: String,
    offset: usize,
    prefix: Vec<u32>,
    cells: Vec<u32>,
    read: Vec<i32>,
    top8: Vec<u32>,
    prediction: u32,
    pointer: [i32; 13],
}

fn hex_id(id: &str) -> bool {
    id.len() == 24
        && id
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn read_rows<T: DeserializeOwned>(file: &str) -> Result<Vec<T>, String> {
    BufReader::new(File::open(file).map_err(|e| format!("{file}: {e}"))?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.map_err(|e| e.to_string())?;
            serde_json::from_str(&line).map_err(|e| format!("{file} row {}: {e}", i + 1))
        })
        .collect()
}

fn writer(path: &Path) -> Result<BufWriter<File>, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map(BufWriter::new)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn line(file: &mut BufWriter<File>, value: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())
}

fn check_prompt(id: &str, domain: &str, prefix: &[u32], cap: usize) -> Result<(), String> {
    if !hex_id(id)
        || !matches!(domain, "conversation" | "recall" | "prose" | "open")
        || prefix.is_empty()
        || prefix.len() > CONTEXT
        || prefix.iter().any(|&v| v >= VOCAB)
        || cap == 0
        || cap > 256
    {
        return Err(format!("invalid evaluation prompt {id}"));
    }
    Ok(())
}

fn check_set(file: &str) -> Result<Vec<Prompt>, String> {
    let prompts: Vec<Prompt> = read_rows(file)?;
    let mut ids = BTreeSet::new();
    for p in &prompts {
        check_prompt(&p.id, &p.domain, &p.prefix, p.cap)?;
        if !ids.insert(&p.id) {
            return Err(format!("duplicate question {}", p.id));
        }
    }
    if prompts
        .iter()
        .filter(|p| p.domain == "conversation")
        .count()
        != 200
    {
        return Err("expected 200 conversations".into());
    }
    if prompts.iter().filter(|p| p.domain == "open").count() != 32 {
        return Err("expected 32 separate open-ended conversations".into());
    }
    if ["recall", "prose"]
        .iter()
        .any(|d| !prompts.iter().any(|p| &p.domain == d))
    {
        return Err("missing recall or prose checks".into());
    }
    Ok(prompts)
}

fn slide(prefix: &mut Vec<u32>, token: u32) {
    if prefix.len() == CONTEXT {
        prefix.remove(0);
    }
    prefix.push(token);
}

// Ordinary clean-room client replay of recorded, preselected device witnesses.
fn replay(artifact: &str, witness_file: &str, prompts: &str, replies: &str) -> Result<(), String> {
    let bytes = fs::read(artifact).map_err(|e| e.to_string())?;
    let (_, client) = d11::artifact::ArtifactTables::from_bytes(&bytes)?;
    let rows: Vec<Witness> = read_rows(witness_file)?;
    let prompts = check_set(prompts)?;
    let generated: Vec<Reply> = read_rows(replies)?;
    let by_id: BTreeMap<_, _> = generated.iter().map(|r| (&r.id, &r.tokens)).collect();
    if by_id.len() != prompts.len() || generated.len() != prompts.len() {
        return Err("incomplete reply roster".into());
    }
    let mut planned = BTreeMap::new();
    for domain in ["conversation", "open", "prose", "recall"] {
        let mut candidates: Vec<_> = prompts.iter().filter(|p| p.domain == domain).collect();
        candidates.sort_by(|a, b| a.id.cmp(&b.id));
        for p in [candidates[0], candidates[candidates.len() - 1]] {
            let tokens = by_id.get(&p.id).ok_or("missing anchor reply")?;
            if tokens.is_empty()
                || tokens.len() > p.cap
                || tokens
                    .iter()
                    .position(|&t| t == EOS)
                    .map_or(tokens.len() != p.cap, |i| i + 1 != tokens.len())
            {
                return Err("invalid anchor completion".into());
            }
            let mut prefix = p.prefix.clone();
            for (offset, &token) in tokens.iter().enumerate() {
                planned.insert((p.id.as_str(), offset), (prefix.clone(), token));
                slide(&mut prefix, token);
            }
        }
    }
    for w in &rows {
        let (prefix, token) = planned
            .remove(&(w.id.as_str(), w.offset))
            .ok_or("unexpected or duplicate witness")?;
        if prefix != w.prefix || token != w.prediction {
            return Err("witness differs from anonymous rollout".into());
        }
        let observed = client
            .evaluate_prefix_traced(&w.prefix)
            .map_err(|e| e.to_string())?;
        let o = observed.evaluation;
        if o.prediction != w.prediction
            || o.read_vector != w.read
            || o.top8 != w.top8
            || o.admitted_cells != w.cells
        {
            return Err(format!(
                "ordinary client disagrees at {} offset {}",
                w.id, w.offset
            ));
        }
        let site = w.pointer[0];
        let native = observed
            .trunk_pointer
            .ok_or("missing ordinary client pointer gate")?;
        let source = native.successor.map(|(pos, _)| pos as i32).unwrap_or(-1);
        if site != w.pointer[3]
            || site < -1
            || site >= w.prefix.len() as i32
            || w.pointer[6] != w.pointer[7]
            || source != site
            || i32::from(native.read_code) != w.pointer[4]
            || i32::from(native.recurrence_code) != w.pointer[5]
            || i32::from(native.selected_token) != w.pointer[6]
        {
            return Err(format!(
                "non-native pointer trace at {} offset {}",
                w.id, w.offset
            ));
        }
    }
    if !planned.is_empty() {
        return Err("missing registered cold trajectory steps".into());
    }
    println!(
        "{}",
        serde_json::json!({"checks":rows.len(),"read_vector":true,"admitted_cells":true,
        "ordered_top8":true,"emission":true,"read_code":true,"recurrence_code":true,
        "successor_address":true,"gate_token":true,"artifact":client.artifact_sha256()})
    );
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    match args {
        [op, file] if op == "check-prompts" => { println!("{}", check_set(file)?.len()); Ok(()) }
        [op, artifact, witnesses, prompts, replies] if op == "replay" => replay(artifact, witnesses, prompts, replies),
        #[cfg(feature = "cuda")]
        [op, artifact, opaque_arm, input, out] if op == "generate" || op == "teacher" => {
            if !hex_id(opaque_arm) { return Err("require an opaque arm identifier".into()); }
            gpu::run(op, artifact, opaque_arm, input, Path::new(out))
        }
        _ => Err("expected check-prompts INPUT, replay D11 WITNESSES, or generate|teacher D11 OPAQUE INPUT NEW_OUTPUT_DIR".into())
    }
}

fn main() {
    if let Err(e) = run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
