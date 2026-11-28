#![forbid(unsafe_code)]

#[path = "../../../codex-4333/transport/src/raw_bfs.rs"]
mod raw_bfs;
#[path = "../../../codex-4332/probe/src/recurrence.rs"]
mod recurrence;

use serde::Deserialize;
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
type Coordinates = [i32; 8];
type Banks = [Coordinates; 4];

const EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
const FIELDS: [&str; 4] = [
    "real_signed",
    "fixed_signed",
    "swapped_signed",
    "no_question_signed",
];

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if !condition {
        return Err(message().into());
    }
    Ok(())
}

fn raw_action(from: usize, to: usize, root: usize) -> Coordinates {
    let action = usize::from(raw_bfs::BFS[16 * from + to]);
    let axes = spiralcore_signed_axes();
    let root = spiralcore_orbit_roots()[root];
    std::array::from_fn(|axis| {
        let code = axes[8 * action + axis];
        let coordinate = i32::from(root[usize::from(code & 7)]);
        if code & 8 == 0 {
            coordinate
        } else {
            -coordinate
        }
    })
}

fn product_action(from: usize, to: usize, root: usize) -> Coordinates {
    cl06_action(cl06_bfs(from, to), &orbit_root_coords(root))
}

fn unit_coordinates(raw: Coordinates) -> Coordinates {
    raw.map(|coordinate| coordinate / 2)
}

fn check_operator() -> Result<Value> {
    let roots = spiralcore_orbit_roots();
    let mut asymmetric_pairs = 0;
    for left in 0..roots.len() {
        require(
            cl06_bfs(left, 0) == usize::from(raw_bfs::BFS[16 * left])
                && product_action(left, 0, left) == roots[0].map(i32::from),
            || format!("action from root {left} does not match the canonical transition to zero"),
        )?;
        for right in 0..roots.len() {
            let product = product_action(left, 0, right);
            require(
                product == raw_action(left, 0, right)
                    && roots.iter().any(|root| root.map(i32::from) == product)
                    && product.iter().filter(|&&cell| cell != 0).count() == 2
                    && product.iter().all(|cell| matches!(cell, -2 | 0 | 2))
                    && (left != 0 || product == roots[right].map(i32::from)),
                || format!("raw edge replay differs for ordered root pair {left},{right}"),
            )?;
            asymmetric_pairs += usize::from(product != product_action(right, 0, left));
        }
    }
    let axis = |left, right| unit_coordinates(product_action(left, 0, right))[0];
    let first_axis = [axis(0, 0), axis(0, 1), axis(1, 0), axis(1, 1)];
    let [aa, ab, ba, bb] = first_axis;
    let mixed_difference = aa - ab - ba + bb;
    require(
        first_axis == [1, -1, -1, 1] && mixed_difference == 4,
        || format!("unexpected edge interaction witness {first_axis:?}"),
    )?;
    Ok(json!({
        "canonical_left_transitions": roots.len(),
        "ordered_root_pairs": roots.len() * roots.len(),
        "reversal_changes_ordered_root_pairs": asymmetric_pairs,
        "left_zero_identity_pairs": roots.len(),
        "first_axis_00_01_10_11": first_axis,
        "mixed_difference": mixed_difference,
        "edges": EDGES,
    }))
}

struct Replay {
    model: IntegerDiscreteModel,
    oracle: recurrence::Recurrence,
}

impl Replay {
    fn state(&self, prefix: &[u32], cut: usize) -> Result<u16> {
        let prefix = prefix.get(..cut).ok_or("replay cut outside the prefix")?;
        let product = native_copy::observe_generate(&self.model, prefix)?.recurrence_code;
        let reference = self.oracle.code(prefix)?;
        require(product == reference, || {
            format!("prefix cut {cut}: product={product:#06x}, oracle={reference:#06x}")
        })?;
        Ok(product)
    }
}

#[derive(Deserialize)]
struct Cut {
    case_id: String,
    user_start: usize,
    user_end: usize,
    tail32_cut: usize,
    tail32_word: u16,
    real_word: u16,
    swapped_word: u16,
    full_word: u16,
    original_sites: usize,
}

fn root_index(word: u16, bank: usize) -> usize {
    usize::from((word >> (4 * bank)) & 15)
}

fn transported_roots(query: u16, anchor: u16, site: u16, additive: Banks) -> Result<[usize; 4]> {
    let mut transported = [0; 4];
    for (bank, old) in additive.into_iter().enumerate() {
        let query = root_index(query, bank);
        let anchor = root_index(anchor, bank);
        let site = root_index(site, bank);
        let raw = raw_action(query, anchor, site);
        let product = product_action(query, anchor, site);
        require(raw == product && unit_coordinates(product) == old, || {
            format!("prior signed coordinates or independent raw transport differs in bank {bank}")
        })?;
        transported[bank] = spiralcore_orbit_roots()
            .iter()
            .position(|root| root.map(i32::from) == product)
            .ok_or("transported block outside the native orbit")?;
    }
    Ok(transported)
}

fn transform(
    candidate: &Value,
    edge: &mut Value,
    clamp: &mut Value,
    queries: [u16; 4],
    anchor: u16,
    site: u16,
) -> Result<()> {
    for (field, query) in FIELDS.into_iter().zip(queries) {
        let additive: [i32; 32] = serde_json::from_value(candidate[field].clone())?;
        let additive: Banks =
            std::array::from_fn(|bank| std::array::from_fn(|axis| additive[8 * bank + axis]));
        let roots = transported_roots(query, anchor, site, additive)?;
        let mut edge_banks = [[0; 8]; 4];
        let mut clamp_banks = [[0; 8]; 4];
        for (bank, (left, right)) in EDGES.into_iter().enumerate() {
            let product = product_action(roots[left], 0, roots[right]);
            let clamped = product_action(0, 0, roots[right]);
            require(
                product == raw_action(roots[left], 0, roots[right])
                    && unit_coordinates(clamped) == additive[right],
                || format!("raw edge or clamped cyclic identity differs for {field} bank {bank}"),
            )?;
            edge_banks[bank] = unit_coordinates(product);
            clamp_banks[bank] = unit_coordinates(clamped);
        }
        edge[field] = json!(edge_banks.concat());
        clamp[field] = json!(clamp_banks.concat());
    }
    Ok(())
}

fn id(row: &Value) -> Result<&str> {
    row["case_id"].as_str().ok_or("missing case ID".into())
}

fn decode<T: serde::de::DeserializeOwned>(value: &Value) -> Result<T> {
    Ok(serde_json::from_value(value.clone())?)
}

fn read_rows<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn output_file(stem: &Path, suffix: &str) -> Result<BufWriter<File>> {
    let mut path = stem.as_os_str().to_owned();
    path.push(suffix);
    let output = File::options().create_new(true).write(true).open(path)?;
    Ok(BufWriter::new(output))
}

fn write_json_line(output: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    Ok(output.write_all(b"\n")?)
}

fn extract_row(replay: &Replay, row: &Value, cut: &Cut, swapped: u16) -> Result<(Value, Value)> {
    let prefix: Vec<u32> = decode(&row["prefix"])?;
    let [start, end]: [usize; 2] = decode(&row["user_bounds"])?;
    let tail32 = start.max(end.saturating_sub(32));
    let candidates = row["candidates"].as_array().ok_or("missing candidates")?;
    let frozen = row["expanded_frozen"]["candidates"]
        .as_array()
        .ok_or("missing expanded candidates")?;
    require(
        id(row)? == cut.case_id
            && start > 0
            && start < end
            && (start, end, tail32) == (cut.user_start, cut.user_end, cut.tail32_cut)
            && prefix.strip_suffix(&ASSISTANT_MARKER).map(<[u32]>::len) == Some(end)
            && decode::<usize>(&row["query_prefix_end"])? == end
            && candidates.len() == frozen.len()
            && candidates.len() == cut.original_sites,
        || "pinned user bounds, tail cut or candidate cardinality differs".into(),
    )?;
    let real = replay.state(&prefix, end)?;
    let anchor = replay.state(&prefix, tail32)?;
    let full = replay.state(&prefix, prefix.len())?;
    require(
        real == cut.real_word
            && real == decode::<u16>(&row["query_code"])?
            && real == decode::<u16>(&row["marker"]["question_state"])?
            && decode::<u16>(&row["fixed_query_code"])? == 0
            && swapped == cut.swapped_word
            && anchor == cut.tail32_word
            && full == cut.full_word
            && full == decode::<u16>(&row["ordinary_opening"]["recurrence_code"])?
            && full == decode::<u16>(&row["expanded_opening"]["recurrence_code"])?,
        || "pinned real, fixed, swapped, tail32 or full recurrence differs".into(),
    )?;
    let queries = [real, 0, swapped, anchor];
    let mut edge = row.clone();
    let mut clamp = row.clone();
    let mut seen = BTreeSet::new();
    for (index, (candidate, original)) in candidates.iter().zip(frozen).enumerate() {
        let pos: usize = decode(&candidate["pos"])?;
        let token: u32 = decode(&candidate["id"])?;
        require(
            (start..end).contains(&pos)
                && seen.insert(pos)
                && prefix.get(pos) == Some(&token)
                && ["pos", "id", "score", "features"].into_iter().all(|field| {
                    candidate
                        .get(field)
                        .is_some_and(|value| original.get(field) == Some(value))
                }),
            || format!("pinned eligible occurrence differs at position {pos}"),
        )?;
        let site = replay.state(&prefix, pos + 1)?;
        require(
            site == decode::<u16>(&candidate["recurrence_code"])?,
            || format!("pinned site recurrence differs at position {pos}"),
        )?;
        transform(
            candidate,
            &mut edge["candidates"][index],
            &mut clamp["candidates"][index],
            queries,
            anchor,
            site,
        )?;
    }
    Ok((edge, clamp))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let operator = check_operator()?;
    if args.len() == 1 && args[0] == "--check" {
        println!("{operator}");
        return Ok(());
    }
    let [model, flat, prior_features, cuts, opposing, stem] = args.as_slice() else {
        return Err("usage: neighboring-root-source --check | BALANCED.d11 FLAT_TRUNK TAIL32.jsonl CUTS.jsonl OPPOSING.json OUTPUT_STEM".into());
    };
    let model = IntegerDiscreteModel::load(Path::new(model))?;
    let oracle = recurrence::Recurrence::new(&fs::read(flat)?, &model.packed_codebook_roots())?;
    let replay = Replay { model, oracle };
    let rows = read_rows::<Value>(prior_features)?;
    let cuts = read_rows::<Cut>(cuts)?;
    let opposing: BTreeMap<String, String> = serde_json::from_reader(File::open(opposing)?)?;
    let queries = rows
        .iter()
        .map(|row| Ok((id(row)?, decode::<u16>(&row["query_code"])?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    require(
        !rows.is_empty()
            && rows.len() == cuts.len()
            && rows.len() == queries.len()
            && queries
                .keys()
                .copied()
                .eq(opposing.keys().map(String::as_str)),
        || "feature, cut and opposing panel identities differ".into(),
    )?;

    let mut edge_output = output_file(Path::new(stem), "-edge.jsonl")?;
    let mut clamp_output = output_file(Path::new(stem), "-clamp.jsonl")?;
    for (row, cut) in rows.iter().zip(&cuts) {
        let case = id(row)?;
        let partner = &opposing[case];
        require(
            partner != case && opposing.get(partner).map(String::as_str) == Some(case),
            || format!("opposing map is not a fixed-point-free involution at {case}"),
        )?;
        let (edge, clamp) = extract_row(&replay, row, cut, queries[partner.as_str()])
            .map_err(|error| format!("case {case}: {error}"))?;
        write_json_line(&mut edge_output, &edge)?;
        write_json_line(&mut clamp_output, &clamp)?;
    }
    edge_output.flush()?;
    clamp_output.flush()?;
    let sites: usize = cuts.iter().map(|cut| cut.original_sites).sum();
    println!(
        "{}",
        json!({
            "rows": rows.len(),
            "sites": sites,
            "retained_sites_per_frame": sites,
            "checked_recurrence_words": 3 * rows.len() + sites,
            "signed_blocks_checked_per_frame": sites * FIELDS.len() * EDGES.len(),
        })
    );
    Ok(())
}
