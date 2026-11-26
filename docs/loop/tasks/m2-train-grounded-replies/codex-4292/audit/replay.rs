#![forbid(unsafe_code)]

#[allow(dead_code)]
mod packet;
#[allow(dead_code)]
mod replay;
#[allow(dead_code)]
mod signed;

use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    thread,
};

use packet::{Bank, Row, Sign, EDGE};
use replay::{Arm, Coverage};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const STATE_BYTES: usize = 13_336;
const CENTERS: usize = 8;

fn check(ok: bool, why: impl FnOnce() -> String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(why().into())
    }
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(fs::File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

struct Move {
    sign: usize,
    slot: usize,
    axis: Option<usize>,
    delta: [i16; EDGE],
}

fn offset(sign: usize, slot: usize, axis: usize) -> usize {
    8 + 6664 * sign + 6152 + 2 * (slot * EDGE + axis)
}

fn write_axis(state: &mut [u8], sign: usize, slot: usize, axis: usize, value: i16) {
    let off = offset(sign, slot, axis);
    state[off..off + 2].copy_from_slice(&value.to_le_bytes());
}

fn replay_menu(
    parent: &[u8],
    origin: &[Sign; 2],
    votes: [[[i32; EDGE]; CENTERS]; 2],
) -> (Vec<u8>, Vec<Move>) {
    let mut sweep = parent.to_vec();
    let mut coordinates = Vec::new();
    let mut centers = Vec::new();
    for (sign, bank) in origin.iter().enumerate() {
        for (slot, center) in bank.transport.iter().enumerate() {
            let delta = std::array::from_fn(|axis| {
                let old = center[axis];
                old.saturating_add(votes[sign][slot][axis].signum() as i16) - old
            });
            let mut movement = 0;
            for (axis, amount) in delta.iter().copied().enumerate() {
                if amount != 0 {
                    movement += 1;
                    write_axis(&mut sweep, sign, slot, axis, center[axis] + amount);
                    let mut single = [0; EDGE];
                    single[axis] = amount;
                    coordinates.push(Move {
                        sign,
                        slot,
                        axis: Some(axis),
                        delta: single,
                    });
                }
            }
            if movement > 1 {
                centers.push(Move {
                    sign,
                    slot,
                    axis: None,
                    delta,
                });
            }
        }
    }
    coordinates.extend(centers);
    (sweep, coordinates)
}

struct ReferenceRow {
    pair: String,
    admission: String,
    case_id: String,
    distance: String,
    gold: [Value; 3],
    governors: [Value; 2],
}

fn string_field(value: &Value, field: &str) -> Result<String> {
    Ok(value[field]
        .as_str()
        .ok_or_else(|| format!("missing reference {field}"))?
        .to_string())
}

fn range(span: &Value, shift: i64) -> Result<[i64; 2]> {
    let start = span["start"].as_i64().ok_or("missing reference start")? + shift;
    let end = span["end"].as_i64().ok_or("missing reference end")? + shift;
    check(start >= 0 && end > start, || {
        "invalid reference span".into()
    })?;
    Ok([start, end])
}

fn references(path: &Path, rows: usize) -> Result<Vec<ReferenceRow>> {
    if path == Path::new("-") {
        return Ok(Vec::new());
    }
    let values: Vec<Value> = serde_json::from_slice(&fs::read(path)?)?;
    check(values.len() == rows, || {
        "reference row count differs from the packet".into()
    })?;
    let mut case_ids = BTreeSet::new();
    values.iter().map(|value| {
        let case_id = string_field(value, "id")?;
        check(case_ids.insert(case_id.clone()), || "duplicate case ID".into())?;
        let sites = value["native_scope"].as_array().ok_or("missing gold scope")?;
        let [a, e, h] = sites.as_slice() else { return Err("expected three original gold addresses".into()); };
        let gold = |site: &Value| -> Result<Value> {
            Ok(json!({"bytes": range(&site["bytes"], -1)?, "tokens": range(&site["tokens"], 0)?}))
        };
        let governor = |name: &str| -> Result<Value> {
            let site = &value["governors"][name];
            let tokens = site["tokens"].as_array().ok_or("missing governor tokens")?;
            check(tokens.len() == 2 && tokens[0].as_u64().zip(tokens[1].as_u64()).is_some_and(|(a,b)| a < b),
                || "invalid governor tokens".into())?;
            Ok(json!({"bytes": range(&site["bytes"], 0)?, "tokens": tokens}))
        };
        Ok(ReferenceRow {
            pair: string_field(value, "pair_id")?, admission: string_field(value, "admission")?,
            case_id, distance: string_field(value, "distance")?,
            gold: [gold(a)?, gold(e)?, gold(h)?],
            governors: [governor("affirmative")?, governor("excluded")?],
        })
    }).collect()
}

#[derive(Default)]
struct Pairs {
    near: Vec<String>,
    far: Vec<String>,
    exact_near: Vec<String>,
    exact_far: Vec<String>,
    first_bad_rows: BTreeMap<String, Value>,
}

fn pair_readback(refs: &[ReferenceRow], predictions: &[Value], training: bool) -> Result<Pairs> {
    let mut pairs = Pairs::default();
    let mut opposed = BTreeMap::<&str, Vec<(usize, bool, bool)>>::new();
    for (ordinal, (gold, got)) in refs.iter().zip(predictions).enumerate() {
        let tuple = got["tuple"]["addresses"].as_array();
        let exact = got["tuple"]["ties"] == 1 && tuple.is_some_and(|actual| actual == &gold.gold);
        let links = &got["links"];
        let governor_ok = (0..2).all(|i| links[i]["governor"] == gold.governors[i]);
        let full = exact && governor_ok && (0..2).all(|i| links[i]["noun"] == gold.gold[i]);
        opposed
            .entry(&gold.pair)
            .or_default()
            .push((ordinal, exact, full));
        if full || gold.distance != "near" {
            continue;
        }
        pairs
            .first_bad_rows
            .entry(gold.pair.clone())
            .or_insert_with(|| {
                json!({
                    "row": ordinal, "case_id": gold.case_id, "admission": gold.admission,
                    "original_nouns": gold.gold, "original_governors": gold.governors,
                    "observed": {"tuple": got["tuple"], "links": got["links"]}
                })
            });
    }
    for (id, members) in opposed {
        let [(i, a_exact, a_full), (j, b_exact, b_full)] = members.as_slice() else {
            return Err(format!("pair {id} does not have two endpoints").into());
        };
        let distance = refs[*i].distance.as_str();
        check(
            refs[*j].distance == distance
                && matches!(distance, "near" | "far")
                && ((refs[*i].admission == "take" && refs[*j].admission == "defer")
                    || (refs[*j].admission == "take" && refs[*i].admission == "defer"))
                && (!training
                    || ((*i < 128) == (distance == "far") && (*j < 128) == (distance == "far"))),
            || format!("bad opposed pair identity at {id}"),
        )?;
        if *a_exact && *b_exact {
            if distance == "near" {
                &mut pairs.exact_near
            } else {
                &mut pairs.exact_far
            }
            .push(id.into());
        }
        if *a_full && *b_full {
            if distance == "near" {
                &mut pairs.near
            } else {
                &mut pairs.far
            }
            .push(id.into());
        }
    }
    Ok(pairs)
}

fn joint_readback(
    rows: &[Row],
    banks: &[Bank; 3],
    state: &[Sign; 2],
    observed: &[Value],
    name: &str,
) -> Result<Vec<Value>> {
    check(observed.len() == rows.len(), || {
        format!("{name} joint row count")
    })?;
    let mut coverage = Coverage::default();
    rows.iter()
        .zip(observed)
        .enumerate()
        .map(|(ordinal, (row, actual))| {
            let wanted = replay::predict(ordinal, row, state, banks, Arm::Learned, &mut coverage);
            check(wanted == *actual, || {
                format!("{name} joint row {ordinal} diverged")
            })?;
            Ok(wanted)
        })
        .collect()
}

struct Batch<'a> {
    rows: &'a [Row],
    banks: &'a [Bank; 3],
    refs: &'a [ReferenceRow],
    signed: &'a signed::Observer<'a>,
    physical: &'a Path,
    far_rows: usize,
}

impl Batch<'_> {
    fn candidate(
        &self,
        id: Option<usize>,
        wanted: &[u8],
        changed: usize,
        score: &Value,
    ) -> Result<Value> {
        let name = id.map_or_else(|| "base".to_owned(), |id| id.to_string());
        let observed = fs::read(self.physical.join(format!("{name}.state")))?;
        check(wanted == observed, || {
            format!("{name} complete state bytes disagree")
        })?;
        let state = packet::state(&observed)?;
        let reads = self.signed.observations(&state)?;
        let edge_trace = read_jsonl(&self.physical.join(format!("{name}.edges.jsonl")))?;
        check(edge_trace.len() == reads.len(), || {
            format!("{name} signed edge count")
        })?;
        let mut hinge = 0;
        let mut wins = 0;
        let mut losses = 0;
        let mut eligible = 0;
        for (i, (wanted, actual)) in reads.iter().zip(edge_trace).enumerate() {
            for key in [
                "row",
                "sign",
                "governor_word",
                "gold_site",
                "eligible_for_fit",
                "eligible_rivals",
                "before",
            ] {
                check(wanted[key] == actual[key], || {
                    format!("{name} signed edge {i} field {key} diverged")
                })?;
            }
            if i < 2 * self.far_rows && wanted["eligible_for_fit"] == true {
                let margin = wanted["before"]["margin"]
                    .as_i64()
                    .ok_or("eligible edge has no margin")?;
                hinge += (1 - margin).max(0);
                wins += usize::from(margin > 0);
                losses += usize::from(margin < 0);
                eligible += 1;
            }
        }
        let computed = json!({"id": id, "hinge": hinge, "strict_wins": wins, "strict_losses": losses,
            "eligible": eligible, "changed": changed});
        for (key, value) in computed.as_object().ok_or("bad computed metrics")? {
            check(*value == score[key], || {
                format!("{name} summary field {key} diverged")
            })?;
        }
        let joint = read_jsonl(&self.physical.join(format!("{name}.reader.jsonl")))?;
        let predicted = joint_readback(self.rows, self.banks, &state, &joint, &name)?;
        let pairs = pair_readback(self.refs, &predicted, true)?;
        Ok(
            json!({"score": computed, "near_full": pairs.near, "far_full": pairs.far,
            "near_failures": pairs.first_bad_rows}),
        )
    }
}

fn batch(args: &[PathBuf]) -> Result<()> {
    let [words, text, labels, parent, expected_step, round_base, physical, reference, receipt] =
        args
    else {
        return Err("usage: batch WORDS TEXT LABELS PARENT EXPECTED_STEP1 ROUND_BASE PHYSICAL REFERENCE_OR_DASH NEW_RECEIPT".into());
    };
    let parent = fs::read(parent)?;
    let round_base = fs::read(round_base)?;
    let rows = packet::text(&fs::read(text)?)?;
    let banks = packet::pooled(&fs::read(words)?)?;
    let origin = packet::state(&parent)?;
    let base = packet::state(&round_base)?;
    check(
        parent.len() == STATE_BYTES && round_base.len() == STATE_BYTES,
        || "wrong complete state size".into(),
    )?;
    let labels = signed::read_labels(labels, rows.len())?;
    let observer = signed::Observer::new(&rows, labels, &origin)?;
    let (step, expected) = replay_menu(&parent, &origin, observer.steps(&origin)?);
    check(step == fs::read(expected_step)?, || {
        "independent old signed sweep differs".into()
    })?;
    check(
        step == fs::read(physical.join("step1-replayed.state"))?,
        || "new physical first signed sweep differs".into(),
    )?;
    let menu = read_jsonl(&physical.join("menu.jsonl"))?;
    let scores = read_jsonl(&physical.join("scores.jsonl"))?;
    let complete: Value = serde_json::from_slice(&fs::read(physical.join("complete.json"))?)?;
    let far_rows: usize = complete["common_far_rows"]
        .as_u64()
        .ok_or("missing far row count")?
        .try_into()?;
    let coordinates = expected
        .iter()
        .filter(|movement| movement.axis.is_some())
        .count();
    let centers = expected.len() - coordinates;
    let singleton_centers = expected[..coordinates]
        .iter()
        .fold(BTreeMap::new(), |mut count, movement| {
            *count.entry((movement.sign, movement.slot)).or_insert(0) += 1;
            count
        })
        .into_values()
        .filter(|&count| count == 1)
        .count();
    let menu_summary: Value =
        serde_json::from_slice(&fs::read(physical.join("menu-summary.json"))?)?;
    check(
        menu_summary
            == json!({"coordinates":coordinates,"centers":centers,
        "duplicate_centers":singleton_centers,"proposals":expected.len()}),
        || "independent menu summary differs".into(),
    )?;
    check(
        complete
            == json!({"rows":rows.len(),"common_far_rows":far_rows,"proposals":expected.len()})
            && far_rows > 0
            && far_rows < rows.len()
            && (rows.len() != 256 || far_rows == 128),
        || "physical batch row scope differs".into(),
    )?;
    check(
        menu.len() == expected.len() && scores.len() == expected.len() + 1,
        || "physical menu size or score count differs".into(),
    )?;
    let refs = references(reference, rows.len())?;
    let batch = Batch {
        rows: &rows,
        banks: &banks,
        refs: &refs,
        signed: &observer,
        physical,
        far_rows,
    };
    let mut metrics = Vec::new();
    metrics.push(batch.candidate(None, &round_base, 0, &scores[0])?);
    let check_candidate = |id: usize, movement: &Move| -> Result<Value> {
        let Move {
            sign,
            slot,
            axis,
            delta,
        } = movement;
        let kind = if axis.is_some() {
            "coordinate"
        } else {
            "center"
        };
        check(
            menu[id]
                == json!({"id": id, "kind": kind, "sign": sign, "slot": slot, "axis": axis, "delta": delta}),
            || format!("menu entry {id} differs from independent vote"),
        )?;
        let mut candidate = round_base.clone();
        let mut changed = 0;
        for (axis, amount) in delta.iter().copied().enumerate() {
            let old = base[*sign].transport[*slot][axis];
            let next = old.saturating_add(amount);
            changed += usize::from(old != next);
            write_axis(&mut candidate, *sign, *slot, axis, next);
        }
        batch.candidate(Some(id), &candidate, changed, &scores[id + 1])
    };
    // The oracles read the same immutable packet. Preserve canonical proposal
    // order while bounding Wick to four reference-analysis threads.
    let chunk_size = expected.len().div_ceil(4).max(1);
    let observations = thread::scope(|scope| -> Result<_> {
        let workers: Vec<_> = expected
            .chunks(chunk_size)
            .enumerate()
            .map(|(part, moves)| {
                scope.spawn(move || -> std::result::Result<Vec<_>, String> {
                    moves
                        .iter()
                        .enumerate()
                        .map(|(i, movement)| {
                            check_candidate(part * chunk_size + i, movement)
                                .map_err(|error| error.to_string())
                        })
                        .collect()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| Ok(worker.join().map_err(|_| "audit worker panicked")??))
            .collect::<Result<Vec<_>>>()
    })?;
    metrics.extend(observations.into_iter().flatten());
    let out = json!({"pass": true, "candidate_count": expected.len(), "comparison_rows_per_state": rows.len(),
        "complete_state_bytes_verified": STATE_BYTES * (expected.len() + 2),
        "signed_edges_verified": rows.len() * 2 * (expected.len() + 1),
        "joint_rows_verified": rows.len() * (expected.len() + 1), "metrics": metrics});
    save(receipt, &out)
}

fn held(args: &[PathBuf]) -> Result<()> {
    let [words, text, state, trace, reference, receipt] = args else {
        return Err("usage: held WORDS TEXT STATE READER_JSONL REFERENCE NEW_RECEIPT".into());
    };
    let rows = packet::text(&fs::read(text)?)?;
    let banks = packet::pooled(&fs::read(words)?)?;
    let state = packet::state(&fs::read(state)?)?;
    let refs = references(reference, rows.len())?;
    let observed = read_jsonl(trace)?;
    let joint = joint_readback(&rows, &banks, &state, &observed, "held")?;
    let pair = pair_readback(&refs, &joint, false)?;
    save(
        receipt,
        &json!({"pass": true, "rows": rows.len(), "near_full": pair.near, "far_full": pair.far,
        "near_exact": pair.exact_near, "far_exact": pair.exact_far}),
    )
}

fn save(path: &Path, data: &Value) -> Result<()> {
    let mut file = BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?,
    );
    serde_json::to_writer(&mut file, data)?;
    writeln!(file)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    let (mode, inputs) = args.split_first().ok_or("missing mode")?;
    match mode.to_str() {
        Some("batch") => batch(inputs),
        Some("held") => held(inputs),
        _ => Err("expected batch or held mode".into()),
    }
}
