//! Proofs that two authentic targets cannot simultaneously be strictly first.
use crate::device::{AddressObservation, Capture, Gpu, Support};
use crate::{fail, Result};
use serde::Serialize;
use std::{
    cmp::Reverse,
    fs,
    io::{BufWriter, Write},
    path::Path,
};

const IDS: usize = 4095;
const UNIT: i64 = 1 << 32;

#[derive(Clone, Copy, Debug)]
struct Constraint {
    key: u64,
    row: u16,
    rival: u16,
    bound: i32,
}

fn incidence(gold: u32, rival: u32) -> (u64, u64) {
    let mut a = 0;
    let mut reverse = 0;
    for bank in 0..4 {
        let g = u64::from((gold >> (bank * 6)) & 63);
        let r = u64::from((rival >> (bank * 6)) & 63);
        if g != r {
            a |= ((g << 6) | r) << (bank * 12);
            reverse |= ((r << 6) | g) << (bank * 12);
        }
    }
    (a, reverse)
}
fn strict_bound(gold: u64, rival: u64) -> Result<i32> {
    if gold > 1 << 48 || rival > 1 << 48 {
        return Err(fail("observation outside Q48 prior range"));
    }
    Ok(i32::try_from(
        (rival as i64 - gold as i64).div_euclid(UNIT) + 1,
    )?)
}
fn make(row: u16, rival: u16, gold: u64, sample: AddressObservation) -> Result<Option<Constraint>> {
    let (a, rev) = incidence(sample.gold, sample.rival);
    if a == 0 {
        return Ok(None);
    }
    Ok(Some(Constraint {
        key: (a.min(rev) << 1) | u64::from(a > rev),
        row,
        rival,
        bound: strict_bound(gold, sample.prior)?,
    }))
}
fn opposite(x: Constraint, y: Constraint, groups: &[u64]) -> bool {
    x.key ^ 1 == y.key
        && x.row != y.row
        && groups[usize::from(x.row)] != groups[usize::from(y.row)]
        && i64::from(x.bound) + i64::from(y.bound) > 0
}
fn cover(side: &[Constraint], other: &[Constraint], groups: &[u64], covered: &mut [bool]) {
    // The two highest bounds from different source groups suffice to decide
    // whether a row has any admissible opposite in this signature bucket.
    let Some(best) = other.iter().max_by_key(|c| c.bound) else {
        return;
    };
    let next = other
        .iter()
        .filter(|c| groups[c.row as usize] != groups[best.row as usize])
        .max_by_key(|c| c.bound);
    for &x in side {
        if opposite(x, *best, groups) || next.is_some_and(|y| opposite(x, *y, groups)) {
            covered[x.row as usize] = true;
        }
    }
}
fn matching(edges: &mut Vec<Constraint>, groups: &[u64]) -> (Vec<(Constraint, Constraint)>, usize) {
    edges.sort_unstable_by_key(|x| (x.key, x.row, Reverse(x.bound), x.rival));
    edges.dedup_by_key(|x| (x.key, x.row));
    let mut matches = Vec::new();
    let mut used = vec![false; groups.len()];
    let mut covered = used.clone();
    for bucket in edges.chunk_by(|a, b| a.key >> 1 == b.key >> 1) {
        let (left, right) = bucket.split_at(bucket.partition_point(|c| c.key & 1 == 0));
        if left.is_empty() || right.is_empty() {
            continue;
        }
        cover(left, right, groups, &mut covered);
        cover(right, left, groups, &mut covered);
        let mut available: Vec<_> = right
            .iter()
            .copied()
            .filter(|v| !used[usize::from(v.row)])
            .collect();
        for &x in left {
            if used[usize::from(x.row)] {
                continue;
            }
            if let Some(j) = available.iter().position(|&y| opposite(x, y, groups)) {
                let y = available.swap_remove(j);
                used[usize::from(x.row)] = true;
                used[usize::from(y.row)] = true;
                matches.push((x, y));
            }
        }
    }
    (matches, covered.into_iter().filter(|&v| v).count())
}

#[derive(Clone, Serialize)]
struct Side {
    ordinal: u32,
    group: u64,
    gold_id: u16,
    rival_id: u16,
    gold_prior: u64,
    rival_prior: u64,
    gold_slots: [u32; 4],
    rival_slots: [u32; 4],
    required_difference: i32,
}
fn slots(packed: u32) -> [u32; 4] {
    std::array::from_fn(|b| (b as u32 * 64) + ((packed >> (6 * b)) & 63))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn audit(
    gpu: &Gpu,
    capture: &Capture,
    targets: &[u16],
    ordinals: &[u32],
    train_groups: &[u64],
    common: &[bool],
    support: &Support,
    out: &Path,
) -> Result<()> {
    if ordinals.len() != targets.len() || common.len() != targets.len() {
        return Err(fail("constraint cohort shape changed"));
    }
    fs::create_dir(out)?;
    let groups: Vec<_> = ordinals
        .iter()
        .map(|&i| {
            train_groups
                .get(i as usize)
                .copied()
                .ok_or("invalid group ordinal")
        })
        .collect::<std::result::Result<_, _>>()?;
    let domain = common.iter().filter(|&&v| v).count();
    // At most 16 million 16-byte packed measurements for this fixed cohort.
    let mut edges = Vec::with_capacity(
        domain
            .checked_mul(IDS - 1)
            .ok_or("constraint count overflow")?,
    );
    let mut aliases = 0usize;
    for start in (0..targets.len()).step_by(128) {
        let count = (targets.len() - start).min(128);
        let block = gpu.constraint_block(capture, targets, support, start, count)?;
        for (offset, row) in block.chunks_exact(IDS).enumerate() {
            let i = start + offset;
            if !common[i] {
                continue;
            }
            let gold = targets[i];
            let same = row[usize::from(gold) - 1];
            if same.rival != same.gold {
                return Err(fail("gold did not map to itself"));
            }
            for (j, &obs) in row.iter().enumerate() {
                let rival = (j + 1) as u16;
                if rival == gold {
                    continue;
                }
                if obs.gold != same.gold {
                    return Err(fail("inconsistent device gold addresses"));
                }
                if let Some(c) = make(i as u16, rival, same.prior, obs)? {
                    edges.push(c);
                } else {
                    if same.prior <= obs.prior {
                        return Err(fail("infeasible alias on strictly attainable row"));
                    }
                    aliases += 1;
                }
            }
        }
    }
    let observations = edges.len();
    let (pairs, covered) = matching(&mut edges, &groups);
    let distinct = edges.len();
    drop(edges);
    let mut wanted = vec![None; targets.len()];
    let mut proofs = vec![[None, None]; pairs.len()];
    let mut raw = BufWriter::new(fs::File::create_new(out.join("observations.jsonl"))?);
    for (p, &(x, y)) in pairs.iter().enumerate() {
        for (s, c) in [x, y].into_iter().enumerate() {
            if wanted[c.row as usize].replace((p, s, c)).is_some() {
                return Err(fail("overlapping certificate"));
            }
        }
    }
    for start in (0..targets.len()).step_by(128) {
        let count = (targets.len() - start).min(128);
        if wanted[start..start + count].iter().all(Option::is_none) {
            continue;
        }
        let block = gpu.constraint_block(capture, targets, support, start, count)?;
        for (d, row) in block.chunks_exact(IDS).enumerate() {
            let i = start + d;
            let Some((p, s, c)) = wanted[i] else { continue };
            let gold = targets[i];
            let mass = row[usize::from(gold) - 1].prior;
            let obs = row[usize::from(c.rival) - 1];
            let x = make(i as u16, c.rival, mass, obs)?
                .ok_or("certificate pair collapsed on reread")?;
            if x.key != c.key || x.bound != c.bound {
                return Err(fail("GPU certificate did not reproduce"));
            }
            let g = row[usize::from(gold) - 1];
            serde_json::to_writer(&mut raw, &serde_json::json!({
                "ordinal":ordinals[i], "gold_id":gold, "rival_id":c.rival,
                "gold":{"prior":g.prior,"gold":g.gold,"rival":g.rival},
                "rival":{"prior":obs.prior,"gold":obs.gold,"rival":obs.rival}
            }))?;
            raw.write_all(b"\n")?;
            proofs[p][s] = Some(Side {
                ordinal: ordinals[i],
                group: groups[i],
                gold_id: gold,
                rival_id: c.rival,
                gold_prior: mass,
                rival_prior: obs.prior,
                gold_slots: slots(obs.gold),
                rival_slots: slots(obs.rival),
                required_difference: c.bound,
            });
        }
    }
    let mut f = BufWriter::new(fs::File::create_new(out.join("certificates.jsonl"))?);
    raw.flush()?;
    for pair in proofs {
        let [Some(x), Some(y)] = pair else {
            return Err(fail("missing physical certificate side"));
        };
        serde_json::to_writer(&mut f, &[x, y])?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    let bound = domain - pairs.len();
    let summary = serde_json::json!({"addressing":"geometry","population":"jointly strictly attainable TRAIN rows",
        "rows":domain,"original_rivals_per_row":IDS-1,"nonalias_inequalities":observations,"weaker_full_aliases":aliases,
        "row_signature_directions":distinct,"opposed_rows_across_distinct_groups":covered,"disjoint_certified_pairs":pairs.len(),
        "shared_strict_win_upper_bound":bound,"full_policy_retention_bar":(4*domain).div_ceil(5),
        "certificates_exclude_full_retention_bar":bound < (4*domain).div_ceil(5),
        "matching":"deterministic greedy; disjoint-pair lower bound, not a maximum matching"});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}

pub(crate) fn finite_device() -> Result<()> {
    let (a, b) = crate::device::constraint_fixture()?;
    let x = make(0, 7, a[7].prior, a[6])?.ok_or("missing first fixture inequality")?;
    let y = make(1, 8, b[6].prior, b[7])?.ok_or("missing second fixture inequality")?;
    if x.bound != 2
        || y.bound != 1
        || !opposite(x, y, &[10, 11])
        || opposite(x, y, &[10, 10])
        || opposite(x, Constraint { bound: -2, ..y }, &[10, 11])
        || strict_bound(UNIT as u64 + 1, 0)? != -1
    {
        return Err(fail("bad transfer of signed strict inequalities"));
    }
    Ok(())
}
