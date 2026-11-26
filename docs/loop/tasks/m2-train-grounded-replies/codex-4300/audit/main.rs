#![forbid(unsafe_code)]

mod enrollment;
#[allow(dead_code)]
mod packet;
mod ranking;

use std::{
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Request {
    row: usize,
    sign: usize,
    word: usize,
    site: usize,
    rival: usize,
}
impl Request {
    fn signed(self) -> i32 {
        if self.sign == 0 {
            1
        } else {
            -1
        }
    }
}

fn check(condition: bool, error: impl std::fmt::Display) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(error.to_string().into())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| format!("{}: {error}", path.display()))?)
}

fn read_lines(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            Ok(serde_json::from_str(line)
                .map_err(|error| format!("{} line {}: {error}", path.display(), i + 1))?)
        })
        .collect()
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn same(expected: &Value, observed: &Value, at: &str) -> Result<usize> {
    match (expected, observed) {
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => {
            a.iter().try_fold(0, |n, (key, value)| {
                Ok(n + same(value, &b[key], &format!("{at}.{key}"))?)
            })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().try_fold(0, |n, (i, (x, y))| {
                Ok(n + same(x, y, &format!("{at}[{i}]"))?)
            })
        }
        _ => {
            check(
                expected == observed,
                format!("{at}: expected {expected}, observed {observed}"),
            )?;
            Ok(1)
        }
    }
}

fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn opts(values: &Value) -> Option<Vec<u64>> {
    values.as_array()?.iter().map(Value::as_u64).collect()
}

fn ordering(a: Option<Vec<u64>>, b: Option<Vec<u64>>) -> &'static str {
    match (a, b) {
        (None, None) => "both_missing",
        (None, Some(_)) => "requested_missing",
        (Some(_), None) => "rival_missing",
        (Some(a), Some(b)) => match a.cmp(&b) {
            std::cmp::Ordering::Less => "requested_better",
            std::cmp::Ordering::Equal => "equal",
            std::cmp::Ordering::Greater => "rival_better",
        },
    }
}

fn classification(req: Request, trace: &Value, info: &Value) -> Value {
    let local = |site: usize| {
        let row = &trace["local_sites"][site];
        (row["eligible"] == true)
            .then(|| opts(&row["cost"]))
            .flatten()
    };
    let local_order = ordering(local(req.site), local(req.rival));
    let joint_order = ordering(
        opts(&trace["requested"]["tuple"]["best"]),
        opts(&trace["rival"]["tuple"]["best"]),
    );
    let tuple = &trace["native"]["tuple"];
    let binding = &trace["native"]["links"][req.sign];
    let selected = tuple["sites"][req.sign]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok());
    let selection = match selected {
        Some(site) if site == req.site => "requested",
        Some(site) if site == req.rival => "rival",
        Some(_) => "third",
        None if tuple["ties"].as_u64().unwrap_or(0) > 0 => "tied",
        None => "absent",
    };
    let consensus = tuple["min_sites"][req.sign]
        .as_array()
        .filter(|sites| sites.len() == 1)
        .and_then(|sites| sites[0].as_u64());
    let governor = binding["word"]
        .as_u64()
        .filter(|_| tuple["ties"].as_u64().is_some_and(|ties| ties > 0))
        .map(|ordinal| ordinal == req.word as u64);
    json!({
        "request":trace["request"],"row":req.row,"case_id":info["case_id"],"pair_id":info["pair_id"],
        "sign":req.signed(),"family":info["family"],"distance":info["distance"],"governor_order":info["governor_order"],
        "selector":info["selector"],"historical_lost":info["historical_lost"],
        "local":local_order,"joint":joint_order,"native":selection,
        "native_site":selected,"native_consensus_site":consensus,"native_governor_correct":governor,
        "requested_local_cost":local(req.site),"rival_local_cost":local(req.rival),
        "requested_joint_cost":opts(&trace["requested"]["tuple"]["best"]),
        "rival_joint_cost":opts(&trace["rival"]["tuple"]["best"]),
        "requested_ties":trace["requested"]["tuple"]["ties"],"rival_ties":trace["rival"]["tuple"]["ties"],
        "native_site_unique_across_minima":tuple["min_sites"][req.sign].as_array().is_some_and(|sites|sites.len()==1),
        "native_min_sites":tuple["min_sites"][req.sign],"third_sites":trace["native_third_sites"],
        "requested_opposite":trace["requested"]["tuple"]["addresses"][1-req.sign],
        "requested_heading":trace["requested"]["tuple"]["addresses"][2],
        "rival_opposite":trace["rival"]["tuple"]["addresses"][1-req.sign],
        "rival_heading":trace["rival"]["tuple"]["addresses"][2],
    })
}

fn census(cohort: &Path, codec: &Path, out: &Path) -> Result<()> {
    let panel = enrollment::Panel::load(cohort, codec)?;
    write_json(out, &panel.census())
}

fn run_replay(
    cohort: &Path,
    codec: &Path,
    state: &Path,
    banks: &Path,
    gpu: &Path,
    archive: &Path,
    out: &Path,
) -> Result<()> {
    let panel = enrollment::Panel::load(cohort, codec)?;
    let state_bytes = fs::read(state)?;
    let states = packet::state(&state_bytes)?;
    let frozen = packet::pooled(&fs::read(banks)?)?;
    let raw_native = read_lines(&gpu.join("native.jsonl"))?;
    let raw_requests = read_lines(&gpu.join("requests.jsonl"))?;
    let archived = if archive == Path::new("-") {
        None
    } else {
        Some(read_lines(archive)?)
    };
    check(
        raw_native.len() == panel.rows.len()
            && raw_requests.len() == panel.requests.len()
            && archived
                .as_ref()
                .is_none_or(|rows| rows.len() == panel.rows.len()),
        "wrong GPU or archive enrollment",
    )?;
    let mut scalar_fields = 0;
    let mut archived_fields = 0;
    let mut classified = Vec::new();
    let mut by_row = vec![Vec::new(); panel.rows.len()];
    for (i, request) in panel.requests.iter().enumerate() {
        by_row[request.row].push(i);
    }
    for (i, row) in panel.rows.iter().enumerate() {
        let replay = ranking::Replay::new(row, &states, &frozen)?;
        let native = replay.native(i);
        scalar_fields += same(&native, &raw_native[i], &format!("native row {i}"))?;
        if let Some(archive) = &archived {
            archived_fields += same(
                &archive[i],
                &raw_native[i],
                &format!("archived native row {i}"),
            )?;
        }
        for &request in &by_row[i] {
            let input = panel.requests[request];
            let expected = replay.request(request, input);
            scalar_fields += same(
                &expected,
                &raw_requests[request],
                &format!("request {request} row {i}"),
            )?;
            classified.push(classification(input, &expected, &panel.info[request]));
        }
    }
    classified.sort_by_key(|row| row["request"].as_u64());
    write_json(
        out,
        &json!({
            "census":panel.census(),"state_sha256":sha(&state_bytes),"scalar_fields":scalar_fields,
            "archived_native_fields":archived_fields,"classifications":classified
        }),
    )
}

fn run(args: &[PathBuf]) -> Result<()> {
    match args {
        [action, cohort, codec, endpoint, out] if action == Path::new("geometry") => {
            write_json(out, &enrollment::geometry(cohort, codec, endpoint)?)
        }
        [action, cohort, codec, before, after, out] if action == Path::new("historical") => {
            historical(cohort,codec,before,after,out)
        }
        [action, text, state, banks, requests, gpu, out] if action == Path::new("fixture") => {
            run_fixture(text,state,banks,requests,gpu,out)
        }
        [action,cohort,codec,out] if action==Path::new("census") => census(cohort,codec,out),
        [action,cohort,codec,state,banks,gpu,archive,out] if action==Path::new("replay") =>
            run_replay(cohort,codec,state,banks,gpu,archive,out),
        _ => Err("usage: switch-audit census COHORT CODEC OUT | geometry COHORT CODEC ENDPOINT OUT | historical COHORT CODEC BEFORE AFTER OUT | fixture TEXT STATE BANKS REQUESTS GPU OUT | replay COHORT CODEC STATE BANKS GPU ARCHIVE_OR_DASH OUT".into()),
    }
}

fn main() -> ExitCode {
    match run(&env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>())
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("switch-audit: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_fixture(
    text: &Path,
    state: &Path,
    banks: &Path,
    requests: &Path,
    gpu: &Path,
    out: &Path,
) -> Result<()> {
    let rows = packet::text(&fs::read(text)?)?;
    let states = packet::state(&fs::read(state)?)?;
    let frozen = packet::pooled(&fs::read(banks)?)?;
    let requests = enrollment::parse_requests(requests)?;
    let native = read_lines(&gpu.join("native.jsonl"))?;
    let traces = read_lines(&gpu.join("requests.jsonl"))?;
    check(
        rows.len() == native.len() && requests.len() == traces.len(),
        "finite GPU enrollment mismatch",
    )?;
    let mut words = 0;
    for (ordinal, row) in rows.iter().enumerate() {
        let replay = ranking::Replay::new(row, &states, &frozen)?;
        words += same(
            &replay.native(ordinal),
            &native[ordinal],
            &format!("finite native {ordinal}"),
        )?;
        for (i, request) in requests
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, r)| r.row == ordinal)
        {
            check(
                request.word < row.words.len()
                    && request.site < row.sites.len()
                    && request.rival < row.sites.len(),
                "finite request outside packet bounds",
            )?;
            words += same(
                &replay.request(i, request),
                &traces[i],
                &format!("finite request {i}"),
            )?;
        }
    }
    check(
        requests.iter().all(|req| req.row < rows.len()),
        "finite request row missing",
    )?;
    write_json(
        out,
        &json!({"rows":rows.len(),"requests":requests.len(),"equal_scalar_fields":words}),
    )
}

fn historical(cohort: &Path, codec: &Path, before: &Path, after: &Path, out: &Path) -> Result<()> {
    let panel = enrollment::Panel::load(cohort, codec)?;
    let frozen = read_lines(before)?;
    let refit = read_lines(after)?;
    check(
        frozen.len() == panel.rows.len() && refit.len() == panel.rows.len(),
        "prior physical trace cardinality differs",
    )?;
    let mut cells = std::collections::BTreeMap::<String, Vec<(Request, &Value)>>::new();
    for (&request, raw) in panel.requests.iter().zip(&panel.info) {
        cells
            .entry(
                raw["pair_id"]
                    .as_str()
                    .ok_or("missing historic pair")?
                    .to_owned(),
            )
            .or_default()
            .push((request, raw));
    }
    let mut pairs = Vec::new();
    for (pair, members) in cells {
        check(members.len() == 2, "prior contrast pair incomplete")?;
        let correct =
            |arm: &[Value], req: &Request| arm[req.row]["tuple"]["sites"][req.sign] == req.site;
        let follows = |arm: &[Value]| members.iter().all(|(r, _)| correct(arm, r));
        let before = follows(&frozen);
        let after = follows(&refit);
        check(
            members
                .iter()
                .all(|(_, v)| v["historical_lost"] == (before && !after)),
            "focus flag differs from archived native switch loss",
        )?;
        if before && !after {
            pairs.push(json!({"pair":pair,"rows":members.iter().map(|(r,_)|r.row).collect::<Vec<_>>(),
                "both_native_governors_before":members.iter().all(|(r,_)|frozen[r.row]["links"][r.sign]["word"]==r.word)}));
        }
    }
    check(
        pairs.len() == 4,
        "did not recover the four registered switch losses",
    )?;
    write_json(
        out,
        &json!({"archived_lost_pairs":pairs,"archived_lost_count":pairs.len(),
        "status":"prior physical observations only"}),
    )
}
