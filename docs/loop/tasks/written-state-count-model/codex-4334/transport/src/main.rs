#![forbid(unsafe_code)]

#[path = "../../../codex-4333/transport/src/raw_bfs.rs"]
mod raw_bfs;
#[path = "../../../codex-4332/probe/src/recurrence.rs"]
mod recurrence;

use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
use uor_r4_integer::discrete_d11::{
    cl06_action, cl06_bfs, native_copy, native_opening::ASSISTANT_MARKER, orbit_root_coords,
    spiralcore_orbit_roots, spiralcore_signed_axes, IntegerDiscreteModel,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Signed = [i32; 32];

const CONTROLS: [&str; 4] = [
    "real_signed",
    "fixed_signed",
    "swapped_signed",
    "no_question_signed",
];
const FRAMES: [&str; 3] = ["zero", "separator", "tail32"];

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if !condition {
        return Err(message().into());
    }
    Ok(())
}

fn checked_state(
    model: &IntegerDiscreteModel,
    oracle: &recurrence::Recurrence,
    prefix: &[u32],
) -> Result<u16> {
    let product = native_copy::observe_generate(model, prefix)?.recurrence_code;
    let reference = oracle.code(prefix)?;
    require(product == reference, || {
        format!(
            "cut {} product={product:#06x} oracle={reference:#06x}",
            prefix.len()
        )
    })?;
    Ok(product)
}

fn block_word(word: u16, block: usize) -> usize {
    usize::from((word >> (block * 4)) & 15)
}

fn signed_site(query: u16, target: u16, source: u16) -> Signed {
    let mut features = [0; 32];
    for (block, cells) in features.chunks_exact_mut(8).enumerate() {
        let action = cl06_bfs(block_word(query, block), block_word(target, block));
        let root = orbit_root_coords(block_word(source, block));
        cells.copy_from_slice(&cl06_action(action, &root).map(|cell| cell / 2));
    }
    features
}

fn check_transport() -> Result<Value> {
    let roots = spiralcore_orbit_roots();
    let axes = spiralcore_signed_axes();
    for query in 0..16 {
        for target in 0..16 {
            let raw_action = usize::from(raw_bfs::BFS[query * 16 + target]);
            require(
                cl06_bfs(query, target) == raw_action
                    && cl06_action(raw_action, &orbit_root_coords(query))
                        == roots[target].map(i32::from),
                || format!("canonical action differs for query {query}, target {target}"),
            )?;
            for (site, root) in roots.iter().enumerate() {
                let expected = std::array::from_fn(|axis| {
                    let signed_axis = axes[raw_action * 8 + axis];
                    let sign = if signed_axis & 8 == 0 { 1 } else { -1 };
                    sign * i32::from(root[usize::from(signed_axis & 7)])
                });
                let got = cl06_action(raw_action, &orbit_root_coords(site));
                require(
                    got == expected
                        && (query != target || got == root.map(i32::from))
                        && got.iter().filter(|&&cell| cell != 0).count() == 2
                        && got.iter().all(|cell| matches!(cell, -2 | 0 | 2)),
                    || format!("action differs for query {query}, target {target}, site {site}"),
                )?;
            }
        }
    }
    Ok(json!({
        "transition_cells": 256,
        "action_query_target_site_cells": 4096,
        "identity_query_site_cells": 256
    }))
}

fn id(row: &Value) -> Result<&str> {
    row["case_id"].as_str().ok_or("missing case ID".into())
}

fn word(value: &Value) -> Result<u16> {
    Ok(u16::try_from(value.as_u64().ok_or("missing native word")?)?)
}

fn write_record(output: &mut impl Write, record: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, record)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn emit(
    model: &IntegerDiscreteModel,
    oracle: &recurrence::Recurrence,
    row: &Value,
    swapped: u16,
    frames: &mut [BufWriter<File>],
    evidence: &mut BufWriter<File>,
) -> Result<()> {
    let prefix: Vec<u32> = serde_json::from_value(row["prefix"].clone())?;
    let [start, end]: [usize; 2] = serde_json::from_value(row["user_bounds"].clone())?;
    let qstart: usize = serde_json::from_value(row["question_start"].clone())?;
    let query_end: usize = serde_json::from_value(row["query_prefix_end"].clone())?;
    let query = prefix
        .strip_suffix(&ASSISTANT_MARKER)
        .ok_or("opening lacks assistant marker")?;
    require(
        start > 0 && (start..end).contains(&qstart) && query.len() == end && query_end == end,
        || "invalid original user bounds, separator or query boundary".into(),
    )?;
    let tail_cut = start.max(end.saturating_sub(32));
    let checked = |cut: usize| checked_state(model, oracle, &prefix[..cut]);
    let real = checked(end)?;
    let passage = checked(qstart)?;
    let tail = checked(tail_cut)?;
    let full = checked(prefix.len())?;
    require(
        real == word(&row["query_code"])?
            && row["marker"]["question_state"].as_u64() == Some(u64::from(real))
            && passage == word(&row["no_question_code"])?
            && word(&row["fixed_query_code"])? == 0
            && full == word(&row["ordinary_opening"]["recurrence_code"])?
            && full == word(&row["expanded_opening"]["recurrence_code"])?,
        || "captured query, passage, fixed or full word differs".into(),
    )?;

    let originals = row["candidates"].as_array().ok_or("missing candidates")?;
    let expanded = row["expanded_frozen"]["candidates"]
        .as_array()
        .ok_or("missing expanded candidates")?;
    require(originals.len() == expanded.len(), || {
        "expanded frozen membership differs".into()
    })?;
    let mut records = [row.clone(), row.clone(), row.clone()];
    let mut seen = BTreeSet::new();
    let mut changed_from_zero = [[0usize; 4]; 2];
    let mut changed_between_cuts = [0usize; 4];
    let mut first_difference = [[None; 4]; 3];
    for (index, (candidate, native)) in originals.iter().zip(expanded).enumerate() {
        let at = usize::from(word(&candidate["pos"])?);
        let token = u32::from(word(&candidate["id"])?);
        require(
            (start..end).contains(&at)
                && seen.insert(at)
                && prefix[at] == token
                && ["pos", "id", "score", "features"]
                    .iter()
                    .all(|field| candidate[*field] == native[*field]),
            || format!("original membership differs at site {at}"),
        )?;
        let site = checked(at + 1)?;
        require(site == word(&candidate["recurrence_code"])?, || {
            format!("captured original site word differs at {at}")
        })?;
        let feature_set = |target, removed| {
            [real, 0, swapped, removed].map(|query| signed_site(query, target, site))
        };
        let features = [
            feature_set(0, passage),
            feature_set(passage, passage),
            feature_set(tail, tail),
        ];
        for control in 0..CONTROLS.len() {
            for frame in 1..FRAMES.len() {
                if let Some(axis) = features[0][control]
                    .iter()
                    .zip(&features[frame][control])
                    .position(|(before, after)| before != after)
                {
                    changed_from_zero[frame - 1][control] += 1;
                    first_difference[frame - 1][control].get_or_insert((
                        at,
                        axis,
                        features[0][control][axis],
                        features[frame][control][axis],
                    ));
                }
            }
            if let Some(axis) = features[1][control]
                .iter()
                .zip(&features[2][control])
                .position(|(before, after)| before != after)
            {
                changed_between_cuts[control] += 1;
                first_difference[2][control].get_or_insert((
                    at,
                    axis,
                    features[1][control][axis],
                    features[2][control][axis],
                ));
            }
        }
        for (record, features) in records.iter_mut().zip(features) {
            let entry = record["candidates"][index]
                .as_object_mut()
                .ok_or("invalid candidate object")?;
            for (field, cells) in CONTROLS.into_iter().zip(features) {
                if entry.insert(field.into(), json!(cells)).is_some() {
                    return Err(format!("duplicate feature field {field}").into());
                }
            }
        }
    }

    for (output, record) in frames.iter_mut().zip(&records) {
        write_record(output, record)?;
    }
    write_record(
        evidence,
        &json!({
            "case_id": id(row)?,
            "user_start": start,
            "user_end": end,
            "separator_cut": qstart,
            "tail32_cut": tail_cut,
            "tail32_clamp_to_user_start": end - start <= 32,
            "real_word": real,
            "swapped_word": swapped,
            "separator_word": passage,
            "tail32_word": tail,
            "full_word": full,
            "original_sites": originals.len(),
            "native_state_comparisons": 4 + originals.len(),
            "control_order": CONTROLS,
            "comparison_order": ["zero_to_separator", "zero_to_tail32", "separator_to_tail32"],
            "changed_from_zero": changed_from_zero,
            "changed_between_cuts": changed_between_cuts,
            "first_difference_position_axis_and_signed_cells": first_difference,
        }),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--check" {
        println!("{}", check_transport()?);
        return Ok(());
    }
    let [model, flat, captures, opposing, out] = args.as_slice() else {
        return Err("usage: boundary-relative-transport --check | BALANCED.d11 FLAT-TRUNK CAPTURE.jsonl OPPOSING.json OUTPUT-STEM".into());
    };
    check_transport()?;
    let model = IntegerDiscreteModel::load(Path::new(model))?;
    let oracle = recurrence::Recurrence::new(&fs::read(flat)?, &model.packed_codebook_roots())?;
    let captures = BufReader::new(File::open(captures)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<Vec<Value>>>()?;
    let opposing: BTreeMap<String, String> = serde_json::from_reader(File::open(opposing)?)?;
    let queries = captures
        .iter()
        .map(|row| Ok((id(row)?, word(&row["query_code"])?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    require(
        !captures.is_empty()
            && captures.len() == queries.len()
            && queries
                .keys()
                .copied()
                .eq(opposing.keys().map(String::as_str)),
        || "case IDs are nonunique or differ from the opposing map".into(),
    )?;
    let open = |suffix| {
        let mut path = out.clone();
        path.push(format!("-{suffix}.jsonl"));
        File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .map(BufWriter::new)
    };
    let mut frames = FRAMES
        .into_iter()
        .map(open)
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut evidence = open("cuts")?;
    for row in &captures {
        let case = id(row)?;
        let partner = &opposing[case];
        require(
            partner != case && opposing.get(partner).map(String::as_str) == Some(case),
            || format!("opposing map is not a fixed-point-free involution at {case}"),
        )?;
        emit(
            &model,
            &oracle,
            row,
            queries[partner.as_str()],
            &mut frames,
            &mut evidence,
        )
        .map_err(|error| format!("case {case}: {error}"))?;
    }
    for output in frames.iter_mut().chain(std::iter::once(&mut evidence)) {
        output.flush()?;
    }
    Ok(())
}
