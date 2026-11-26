#![forbid(unsafe_code)]

#[allow(dead_code)]
mod controls;
#[allow(dead_code)]
mod gate;
#[allow(dead_code)]
mod identity;
mod median;
mod packet;
#[allow(dead_code)]
mod panel;
#[allow(dead_code)]
mod replay;
#[allow(dead_code)]
mod signed;
mod study;

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use identity::address;
use packet::{Bank, Row};
use serde_json::Value;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn check(ok: bool, detail: impl std::fmt::Display) -> Result<()> {
    if !ok {
        return Err(detail.to_string().into());
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)
        .map_err(|err| format!("{}: {err}", path.display()))?)
}

fn read_lines(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            Ok(serde_json::from_str(line)
                .map_err(|err| format!("{} line {}: {err}", path.display(), i + 1))?)
        })
        .collect()
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    Ok(())
}

fn same(expected: &Value, actual: &Value, location: &str) -> Result<u64> {
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => {
            a.iter().try_fold(0, |n, (key, value)| {
                Ok(n + same(value, &b[key], &format!("{location}.{key}"))?)
            })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().try_fold(0, |n, (i, (a, b))| {
                Ok(n + same(a, b, &format!("{location}[{i}]"))?)
            })
        }
        _ => {
            check(
                expected == actual,
                format_args!("{location}: expected {expected}, observed {actual}"),
            )?;
            Ok(1)
        }
    }
}

fn equal_rows(wanted: &[Value], observed: &[Value], location: &str) -> Result<u64> {
    check(
        wanted.len() == observed.len(),
        format!("{location}: row count changed"),
    )?;
    wanted
        .iter()
        .zip(observed)
        .enumerate()
        .try_fold(0, |n, (i, (want, got))| {
            Ok(n + same(want, got, &format!("{location}[{i}]"))?)
        })
}

fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn run(args: &[PathBuf]) -> Result<()> {
    match args {
        [mode, pool, parent, panel, original, reference, out] if mode == Path::new("premise") => {
            study::premise(pool, parent, panel, original, reference, out)
        }
        [mode, endpoint, ordered, labels, parent, physical, out]
            if mode == Path::new("fixture") =>
        {
            let (e, o, labels) = median::read_tapes(endpoint, ordered, labels)?;
            let base = fs::read(parent)?;
            let audited = median::audit(&e, &o, &labels, &base, Some(physical), true)?;
            write_json(out, &audited.value())
        }
        [mode, pool, parent, panel, reference, physical, out] if mode == Path::new("train") => {
            study::train(pool, parent, panel, reference, physical, out)
        }
        [mode, pool, parent, far_only, panel, train, audit, physical, grades, out]
            if mode == Path::new("held") =>
        {
            study::held(
                pool, parent, far_only, panel, train, audit, physical, grades, out,
            )
        }
        [mode, core, out] if mode == Path::new("gate-fixture") => gate::fixture(core, out),
        [mode, panel_path, original, out, previous @ ..] if mode == Path::new("panel") => {
            panel::audit(panel_path, original, out, previous)
        }
        _ => Err("usage: median-audit {premise|fixture|train|held|gate-fixture|panel} ...".into()),
    }
}

fn main() -> ExitCode {
    match run(&env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>())
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}
