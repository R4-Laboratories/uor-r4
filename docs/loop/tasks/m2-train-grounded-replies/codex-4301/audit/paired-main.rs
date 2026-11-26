#![forbid(unsafe_code)]

mod controls;
#[allow(dead_code)]
mod enrollment {
    include!("enrollment.rs");
    include!("identities.rs");
}
#[allow(dead_code)]
mod packet;
#[allow(dead_code)]
mod ranking {
    include!("ranking.rs");
    include!("rival.rs");
}

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

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
        [1, -1][self.sign]
    }
}

fn check(condition: bool, issue: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(issue.into().into())
    }
}

fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_lines(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            Ok(serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), i + 1))?)
        })
        .collect()
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut writer = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    );
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn exact(a: &Value, b: &Value, path: &str) -> Result<usize> {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => {
            a.iter().try_fold(0, |n, (key, value)| {
                Ok(n + exact(value, &b[key], &format!("{path}.{key}"))?)
            })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().try_fold(0, |n, (i, (a, b))| {
                Ok(n + exact(a, b, &format!("{path}[{i}]"))?)
            })
        }
        _ => {
            check(a == b, format!("{path}: expected {a}, observed {b}"))?;
            Ok(1)
        }
    }
}

fn native_address(input: &Value, byte_origin: u64) -> Result<Value> {
    let pair = |key: &str, offset| -> Result<Value> {
        let value = &input[key];
        let endpoints = if value.is_array() {
            [&value[0], &value[1]]
        } else {
            [&value["start"], &value["end"]]
        };
        let checked = |v: &Value| -> Result<u64> {
            Ok(v.as_u64()
                .and_then(|n| n.checked_sub(offset))
                .ok_or("invalid original coordinates")?)
        };
        Ok(json!([checked(endpoints[0])?, checked(endpoints[1])?]))
    };
    Ok(json!({"bytes": pair("bytes", byte_origin)?, "tokens":pair("tokens", 0)?}))
}

fn request_grade(
    request: Request,
    metadata: &Value,
    native: &Value,
    costs: &Value,
    mut thirds: Value,
) -> Result<Value> {
    let tuple = &native["tuple"];
    let binding = &native["links"][request.sign];
    let unique = tuple["ties"] == 1;
    let follows = unique
        && binding["word"] == request.word
        && binding["governor"] == native_address(&metadata["governor"], 0)?
        && binding["site"] == request.site
        && tuple["sites"][request.sign] == request.site
        && binding["noun"] == native_address(&metadata["requested"], 0)?
        && tuple["addresses"][request.sign] == binding["noun"];
    let other = &native["links"][1 - request.sign];
    let full = follows
        && other["word"] == metadata["opposite_governor"]["index"]
        && other["governor"] == native_address(&metadata["opposite_governor"], 0)?
        && other["site"] == metadata["opposite"]["index"]
        && other["noun"] == native_address(&metadata["opposite"], 0)?
        && tuple["sites"][2] == metadata["heading"]["index"]
        && tuple["addresses"][2] == native_address(&metadata["heading"], 0)?;
    let relation = |path: &str| {
        let requested = &costs["sites"][request.site];
        let rival = &costs["sites"][request.rival];
        let (a, b) = if path == "local" {
            (&requested[path], &rival[path])
        } else {
            (&requested[path]["best"], &rival[path]["best"])
        };
        let values = |v: &Value| {
            v.as_array()
                .and_then(|a| a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>())
        };
        match (values(a), values(b)) {
            (None, None) => "both_absent",
            (None, Some(_)) => "requested_absent",
            (Some(_), None) => "rival_absent",
            (Some(a), Some(b)) => match a.cmp(&b) {
                std::cmp::Ordering::Less => "requested_cheaper",
                std::cmp::Ordering::Equal => "tie",
                std::cmp::Ordering::Greater => "rival_cheaper",
            },
        }
    };
    let r = &mut thirds;
    r["case_id"] = metadata["case_id"].clone();
    r["pair_id"] = metadata["pair_id"].clone();
    r["family"] = metadata["family"].clone();
    r["selector"] = metadata["selector"].clone();
    r["follows"] = json!(follows);
    r["full"] = json!(full);
    r["native_ties"] = tuple["ties"].clone();
    r["native_signed_site"] = binding["site"].clone();
    r["native_signed_governor"] = binding["word"].clone();
    r["local_same_spelling"] = json!(relation("local"));
    r["fixed_same_spelling"] = json!(relation("fixed"));
    r["free_same_spelling"] = json!(relation("free"));
    Ok(thirds)
}

fn pairs(grades: &[Value]) -> Result<Vec<Value>> {
    let mut index = BTreeMap::<&str, Vec<&Value>>::new();
    for row in grades {
        index
            .entry(row["pair_id"].as_str().ok_or("missing pair ID")?)
            .or_default()
            .push(row);
    }
    index
        .into_iter()
        .map(|(id, rows)| {
            let [a, b] = rows.as_slice() else {
                return Err(format!("pair {id} is incomplete").into());
            };
            check(
                a["family"] == b["family"]
                    && a["selector"] != b["selector"]
                    && a["case_id"] != b["case_id"],
                "a paired loss or win silently substituted a row",
            )?;
            Ok(
                json!({"id":id,"family":a["family"],"case_ids":[a["case_id"],b["case_id"]],
            "follows":a["follows"] == true && b["follows"] == true,
            "full":a["full"] == true && b["full"] == true,
            "unresolved": a["native_ties"] != 1 || b["native_ties"] != 1}),
            )
        })
        .collect()
}

fn replay(
    cohort: &Path,
    codec: &Path,
    state: &Path,
    banks: &Path,
    native: &Path,
    costs: &Path,
    out: &Path,
) -> Result<()> {
    let panel = enrollment::Panel::load(cohort, codec)?;
    let state_bytes = fs::read(state)?;
    let states = packet::state(&state_bytes)?;
    let banks = packet::pooled(&fs::read(banks)?)?;
    let native = read_lines(native)?;
    let costs = read_lines(costs)?;
    check(
        native.len() == panel.rows.len() && costs.len() == panel.requests.len(),
        "physical cohort size differs",
    )?;
    let mut scalar_fields = 0;
    let mut grades = Vec::new();
    let mut by_row = vec![Vec::new(); panel.rows.len()];
    for (i, request) in panel.requests.iter().enumerate() {
        by_row[request.row].push(i);
    }
    for (i, row) in panel.rows.iter().enumerate() {
        let ranking = ranking::Replay::new(row, &states, &banks)?;
        scalar_fields += exact(&ranking.native(i), &native[i], &format!("native row {i}"))?;
        for &j in &by_row[i] {
            let request = panel.requests[j];
            let (expected, witness) = ranking.competition(request);
            scalar_fields += exact(&expected, &costs[j], &format!("cost request {j}"))?;
            grades.push(request_grade(
                request,
                &panel.info[j],
                &native[i],
                &expected,
                witness,
            )?);
        }
    }
    let paired = pairs(&grades)?;
    write_json(
        out,
        &json!({"census":panel.census(),"state_sha256":sha(&state_bytes),
        "scalar_fields":scalar_fields,"rows":grades,"pairs":paired}),
    )
}

fn replay_near(
    rows: &[packet::Row],
    members: &mut [Value],
    state: &Path,
    banks: &Path,
    physical: &Path,
    out: &Path,
) -> Result<()> {
    let state = fs::read(state)?;
    let states = packet::state(&state)?;
    let banks = packet::pooled(&fs::read(banks)?)?;
    let records = read_lines(physical)?;
    let native: BTreeMap<_, _> = records
        .iter()
        .map(|record| (record["row"].as_u64(), record))
        .collect();
    check(
        (records.len() == members.len() || records.len() == rows.len())
            && native.len() == records.len(),
        "protected physical row enrollment is wrong",
    )?;
    let mut scalar_fields = 0;
    let mut protected = BTreeMap::<String, bool>::new();
    for member in members.iter_mut() {
        let i = member["row_index"]
            .as_u64()
            .ok_or("missing original index")?;
        let replay = ranking::Replay::new(&rows[i as usize], &states, &banks)?;
        let expected = replay.native(i as usize);
        let observed = native
            .get(&Some(i))
            .ok_or("protected original row missing")?;
        scalar_fields += exact(&expected, observed, &format!("protected native row {i}"))?;
        let ok = observed["tuple"]["ties"] == 1
            && observed["tuple"]["sites"] == member["sites"]
            && observed["tuple"]["addresses"] == member["site_addresses"]
            && (0..2).all(|sign| {
                observed["links"][sign]["word"] == member["governors"][sign]
                    && observed["links"][sign]["governor"] == member["governor_addresses"][sign]
                    && observed["links"][sign]["site"] == member["sites"][sign]
                    && observed["links"][sign]["noun"] == member["site_addresses"][sign]
            });
        let pair = member["pair_id"]
            .as_str()
            .ok_or("missing protected pair label")?;
        *protected.entry(pair.to_owned()).or_insert(true) &= ok;
        member["full"] = json!(ok);
    }
    let misses: Vec<_> = protected
        .iter()
        .filter(|(_, full)| !**full)
        .map(|(id, _)| id)
        .collect();
    write_json(
        out,
        &json!({"state_sha256":sha(&state),"scalar_fields":scalar_fields,
        "protected_pairs":protected,"lost_protected":misses,"members":members}),
    )
}

fn compare_fresh(paths: [&Path; 3], out: &Path) -> Result<()> {
    let raw = paths
        .map(read_json)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut indexed = Vec::new();
    for audit in &raw {
        let rows = audit["pairs"]
            .as_array()
            .ok_or("missing fresh paired readout")?;
        let mut pairs = BTreeMap::new();
        for row in rows {
            let id = row["id"].as_str().ok_or("missing compared pair ID")?;
            check(
                pairs.insert(id, row).is_none(),
                "duplicate fresh pair identity",
            )?;
        }
        indexed.push(pairs);
    }
    check(
        indexed
            .iter()
            .skip(1)
            .all(|set| indexed[0].keys().eq(set.keys())),
        "fresh paired identities differ by model",
    )?;
    let mut families = BTreeMap::<&str, Vec<Value>>::new();
    let mut all = Vec::new();
    for (id, row) in &indexed[0] {
        let family = row["family"].as_str().ok_or("missing fresh family")?;
        let models = [&indexed[0][id], &indexed[1][id], &indexed[2][id]];
        check(
            models
                .iter()
                .all(|m| m["case_ids"] == row["case_ids"] && m["family"] == family),
            "model read different original prompt identities",
        )?;
        let full = models.map(|model| model["full"] == true);
        let signed = models.map(|model| model["follows"] == true);
        let value =
            json!({"id":id,"case_ids":row["case_ids"],"family":family,"signed":signed,"full":full});
        all.push(value.clone());
        families.entry(family).or_default().push(value);
    }
    let reduce = |rows: &[Value], metric: &str| {
        let mut intersections: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for row in rows {
            let mask: String = (0..3)
                .map(|i| if row[metric][i] == true { '1' } else { '0' })
                .collect();
            if let Some(id) = row["id"].as_str() {
                intersections.entry(mask).or_default().insert(id.to_owned());
            }
        }
        json!(intersections)
    };
    let by_family: BTreeMap<_, _> = families
        .iter()
        .map(|(id, rows)| {
            (
                id,
                json!({"n":rows.len(),
        "signed_intersections":reduce(rows, "signed"),"full_intersections":reduce(rows, "full")}),
            )
        })
        .collect();
    write_json(
        out,
        &json!({"order":["elected_or_diagnostic", "frozen_ordered", "ordered_median"],"n":all.len(),
        "signed_intersections":reduce(&all, "signed"),"full_intersections":reduce(&all, "full"),"family":by_family}),
    )
}

fn fixture(
    text: &Path,
    state: &Path,
    banks: &Path,
    requests: &Path,
    native: &Path,
    costs: &Path,
    out: &Path,
) -> Result<()> {
    let rows = packet::text(&fs::read(text)?)?;
    let states = packet::state(&fs::read(state)?)?;
    let banks = packet::pooled(&fs::read(banks)?)?;
    let requests = enrollment::parse_requests(requests)?;
    let native = read_lines(native)?;
    let costs = read_lines(costs)?;
    check(
        rows.len() == native.len() && costs.len() == requests.len(),
        "physical finite enrollment differs",
    )?;
    let mut scalar_fields = 0;
    let mut replays = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let replay = ranking::Replay::new(row, &states, &banks)?;
        scalar_fields += exact(
            &replay.native(i),
            &native[i],
            &format!("finite native row {i}"),
        )?;
        replays.push(replay);
    }
    let mut witness = Vec::new();
    for (i, request) in requests.iter().copied().enumerate() {
        let replay = replays
            .get(request.row)
            .ok_or("invalid fixture row index")?;
        let (expected, thirds) = replay.competition(request);
        scalar_fields += exact(&expected, &costs[i], &format!("finite request {i}"))?;
        witness.push(thirds);
    }
    write_json(
        out,
        &json!({"scalar_fields":scalar_fields,"witnesses":witness}),
    )
}

fn run(args: &[PathBuf]) -> Result<()> {
    match args {
        [action, cohort, codec, out] if action == Path::new("census") => {
            write_json(out, &enrollment::Panel::load(cohort, codec)?.census())
        }
        [action, cohort, codec, out] if action == Path::new("control-census") => {
            controls::run(cohort, codec, None, out)
        }
        [action, cohort, codec, gold, swapped, out] if action == Path::new("physical-controls") => {
            controls::run(cohort, codec, Some([gold, swapped]), out)
        }
        [action, cohort, codec, endpoint, out] if action == Path::new("geometry") => {
            write_json(out, &enrollment::geometry(cohort, codec, endpoint)?)
        }
        [action, root, codec, premise, gold, out] if action == Path::new("near-census") => {
            let (_, members) = enrollment::protected_near(root, codec, premise, gold)?;
            write_json(out, &json!({"members":members}))
        }
        [action, root, codec, premise, gold, state, banks, native, out] if action == Path::new("near-replay") => {
            let (rows, mut members) = enrollment::protected_near(root, codec, premise, gold)?;
            replay_near(&rows, &mut members, state, banks, native, out)
        }
        [action, cohort, codec, state, banks, native, costs, out] if action == Path::new("replay") => {
            replay(cohort, codec, state, banks, native, costs, out)
        }
        [action, text, state, banks, requests, native, costs, out] if action == Path::new("fixture") => {
            fixture(text, state, banks, requests, native, costs, out)
        }
        [action, a, b, c, out] if action == Path::new("compare-fresh") => compare_fresh([a, b, c], out),
        _ => Err("expected paired-audit census, geometry, near-census, near-replay, fixture, replay or compare-fresh".into()),
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
            eprintln!("paired-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
