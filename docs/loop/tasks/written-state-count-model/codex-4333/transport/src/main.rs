#![forbid(unsafe_code)]

mod raw_bfs;

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_integer::discrete_d11::{
    cl06_action, cl06_bfs, orbit_root_coords, spiralcore_orbit_roots, spiralcore_signed_axes,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const ROOTS: usize = 16;
const STATES: usize = 64;
const AXES: usize = 8;
const BLOCKS: usize = 4;

fn signed_block(query: usize, source: usize) -> [i32; AXES] {
    cl06_action(cl06_bfs(query, 0), &orbit_root_coords(source)).map(|cell| cell / 2)
}

fn signed_site(query: u16, source: u16) -> [i32; BLOCKS * AXES] {
    let mut features = [0; BLOCKS * AXES];
    for (block, feature) in features.chunks_exact_mut(AXES).enumerate() {
        let request = usize::from((query >> (4 * block)) & 15);
        let site = usize::from((source >> (4 * block)) & 15);
        feature.copy_from_slice(&signed_block(request, site));
    }
    features
}

fn raw_axes(action: usize) -> &'static [u8] {
    &spiralcore_signed_axes()[action * AXES..(action + 1) * AXES]
}

fn raw_apply(action: usize, coordinates: [i8; AXES]) -> [i32; AXES] {
    std::array::from_fn(|axis| {
        let code = raw_axes(action)[axis];
        let sign = if code >= 8 { -1 } else { 1 };
        sign * i32::from(coordinates[usize::from(code % 8)])
    })
}

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message().into())
    }
}

fn verify_operator() -> Result<Value> {
    let roots = spiralcore_orbit_roots();
    let anchor = roots[0].map(|cell| i32::from(cell) / 2);
    for action in 0..STATES {
        let mut axes = raw_axes(action)
            .iter()
            .map(|code| code & 7)
            .collect::<Vec<_>>();
        axes.sort_unstable();
        require(axes == [0, 1, 2, 3, 4, 5, 6, 7], || {
            format!("action {action} is not a permutation")
        })?;
        for (index, raw) in roots.iter().enumerate() {
            let root = raw.map(i32::from);
            require(orbit_root_coords(index) == root, || {
                format!("root {index} differs")
            })?;
            require(
                cl06_action(action, &root) == raw_apply(action, *raw),
                || format!("signed axis mismatch on action {action}, root {index}"),
            )?;
        }
    }
    for (source, raw) in roots.iter().enumerate() {
        for (target, want) in roots.iter().enumerate() {
            let action = usize::from(raw_bfs::BFS[source * ROOTS + target]);
            require(
                cl06_bfs(source, target) == action
                    && raw_apply(action, *raw) == want.map(i32::from),
                || format!("BFS transition mismatch {source}->{target}"),
            )?;
            let independently_transported =
                raw_apply(usize::from(raw_bfs::BFS[target * ROOTS]), *raw);
            require(
                signed_block(target, source) == independently_transported.map(|cell| cell / 2)
                    && independently_transported
                        .iter()
                        .all(|cell| matches!(cell, -2 | 0 | 2))
                    && independently_transported
                        .iter()
                        .filter(|cell| **cell != 0)
                        .count()
                        == 2,
                || format!("normalized block mismatch: query {target}, site {source}"),
            )?;
        }
        require(signed_block(source, source) == anchor, || {
            format!("query {source} did not reach the normalized anchor")
        })?;
    }

    // Derive active <- fixed directly from raw signed axes. A signed bijection of
    // score coefficients gives the same inverse weight substitution for every
    // site in a row, preserving the whole integer box with any fixed offset.
    for active_query in 0..ROOTS {
        let active_axes = raw_axes(usize::from(raw_bfs::BFS[active_query * ROOTS]));
        for fixed_query in 0..ROOTS {
            let fixed_axes = raw_axes(usize::from(raw_bfs::BFS[fixed_query * ROOTS]));
            let mut matched_axes = [(0, 0); AXES];
            for (active_axis, active_code) in active_axes.iter().copied().enumerate() {
                let fixed_axis = fixed_axes
                    .iter()
                    .position(|fixed_code| fixed_code & 7 == active_code & 7)
                    .ok_or("fixed gauge misses a source axis")?;
                let sign = if (active_code ^ fixed_axes[fixed_axis]) & 8 == 0 {
                    1
                } else {
                    -1
                };
                matched_axes[active_axis] = (fixed_axis, sign);
            }
            for site in 0..ROOTS {
                let active = signed_block(active_query, site);
                let fixed = signed_block(fixed_query, site);
                for (active_axis, (fixed_axis, sign)) in matched_axes.into_iter().enumerate() {
                    require(active[active_axis] == sign * fixed[fixed_axis], || {
                        format!("gauge mismatch: active {active_query}, fixed {fixed_query}, site {site}")
                    })?;
                }
            }
        }
    }
    Ok(json!({
        "signed_action_root_cells": STATES * ROOTS,
        "bfs_transitions": ROOTS * ROOTS,
        "normalized_query_site_cells": ROOTS * ROOTS,
        "query_anchor_identities": ROOTS,
        "fixed_gauge_query_query_site_cells": ROOTS * ROOTS * ROOTS,
    }))
}

fn case_id(row: &Value) -> Result<&str> {
    row["case_id"]
        .as_str()
        .ok_or_else(|| "capture has no case_id".into())
}

fn read_code(value: &Value) -> Result<u16> {
    Ok(u16::try_from(
        value.as_u64().ok_or("invalid recurrence code")?,
    )?)
}

fn read_captures(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn augment(capture: &Path, opposing: &Path) -> Result<()> {
    let mut captures = read_captures(capture)?;
    let opposing: BTreeMap<String, String> = serde_json::from_reader(File::open(opposing)?)?;
    let queries: BTreeMap<String, u16> = captures
        .iter()
        .map(|row| Ok((case_id(row)?.to_owned(), read_code(&row["query_code"])?)))
        .collect::<Result<_>>()?;
    if captures.is_empty() || queries.len() != captures.len() || queries.keys().ne(opposing.keys())
    {
        return Err("capture IDs and opposing map must cover the same nonempty, unique set".into());
    }
    for row in &mut captures {
        let id = case_id(row)?;
        let partner = &opposing[id];
        if partner == id || opposing.get(partner).map(String::as_str) != Some(id) {
            return Err(
                format!("opposing map is not a fixed-point-free involution at {id}").into(),
            );
        }
        let real = queries[id];
        let swapped = queries[partner];
        let fixed = read_code(&row["fixed_query_code"])?;
        if fixed != 0 {
            return Err("fixed query must use the recorded anchor gauge 0".into());
        }
        let removed = row
            .get("no_question_code")
            .ok_or("capture lacks no_question_code")?;
        let removed = (!removed.is_null())
            .then(|| read_code(removed))
            .transpose()?;
        for candidate in row["candidates"]
            .as_array_mut()
            .ok_or("capture lacks candidates")?
        {
            let source = read_code(&candidate["recurrence_code"])?;
            let fields = candidate
                .as_object_mut()
                .ok_or("invalid captured candidate")?;
            for (field, query) in [
                ("real_signed", Some(real)),
                ("fixed_signed", Some(fixed)),
                ("swapped_signed", Some(swapped)),
                ("no_question_signed", removed),
            ] {
                if fields
                    .insert(
                        field.into(),
                        json!(query.map(|code| signed_site(code, source))),
                    )
                    .is_some()
                {
                    return Err(format!("capture already has field {field}").into());
                }
            }
        }
    }
    let mut out = BufWriter::new(io::stdout().lock());
    for row in captures {
        serde_json::to_writer(&mut out, &row)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [check] if check == "--check" => println!("{}", verify_operator()?),
        [capture, opposing] => {
            verify_operator()?;
            augment(Path::new(capture), Path::new(opposing))?;
        }
        _ => {
            return Err(
                "usage: check-signed-site-transport --check | CAPTURE.jsonl OPPOSING.json".into(),
            )
        }
    }
    Ok(())
}
