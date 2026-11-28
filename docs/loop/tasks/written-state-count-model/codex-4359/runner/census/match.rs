//! Deterministic row-disjoint opposite inequalities and independent witness re-read.
use std::cmp::Reverse;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

const ROWS: usize = 3886;
const IDS: usize = 4096;
const ROW_BYTES: usize = 12 + 8 * IDS;
const UNIT: i64 = 1 << 32;
const BAR: usize = 3109;
const SUPPORT: [usize; 16] = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0];

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy)]
struct Row {
    ordinal: u32,
    group: u32,
    gold: u16,
    carry: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edge {
    key: u64,
    row: u16,
    rival: u16,
    bound: i32,
}

fn opposite(left: Edge, right: Edge, rows: &[Row]) -> bool {
    left.key ^ 1 == right.key
        && rows[usize::from(left.row)].group != rows[usize::from(right.row)].group
        && left.bound + right.bound > 0
}

fn matching(
    mut edges: Vec<Edge>,
    killed: &[Option<Edge>],
    rows: &[Row],
) -> (Vec<[Edge; 2]>, usize) {
    edges.retain(|e| killed[usize::from(e.row)].is_none());
    edges.sort_unstable_by_key(|e| (e.key, e.row, Reverse(e.bound), e.rival));
    edges.dedup_by_key(|e| (e.key, e.row));
    let unique = edges.len();
    let mut used = vec![false; rows.len()];
    let mut selected = Vec::new();
    for bucket in edges.chunk_by(|a, b| a.key >> 1 == b.key >> 1) {
        let (direct, reverse) = bucket.split_at(bucket.partition_point(|e| e.key & 1 == 0));
        if direct.is_empty() || reverse.is_empty() {
            continue;
        }
        let mut available: Vec<_> = reverse
            .iter()
            .copied()
            .filter(|e| !used[usize::from(e.row)])
            .collect();
        for &left in direct {
            if used[usize::from(left.row)] {
                continue;
            }
            if let Some(index) = available
                .iter()
                .position(|&right| !used[usize::from(right.row)] && opposite(left, right, rows))
            {
                let right = available.swap_remove(index);
                used[usize::from(left.row)] = true;
                used[usize::from(right.row)] = true;
                selected.push([left, right]);
            }
        }
    }
    (selected, unique)
}

fn address(carry: u16, root: u16, tables: &[u8; 4352], arm: &str) -> [u8; 5] {
    let mut slots = [0; 5];
    for (bank, slot) in slots[..4].iter_mut().enumerate() {
        let source = usize::from((carry >> (4 * bank)) & 15);
        let target = usize::from((root >> (4 * bank)) & 15);
        *slot = tables[16 * source + target];
    }
    slots[4] = match arm {
        "product" => tables[256 + 64 * usize::from(slots[0]) + usize::from(slots[1])],
        "unary" => (4 * usize::from(carry & 15) + SUPPORT[usize::from(root & 15)]) as u8,
        _ => 0,
    };
    slots
}

fn read_mass(tape: &mut File, e: Edge, id: u16) -> Result<u64> {
    let offset = usize::from(e.row) * ROW_BYTES + 12 + 8 * usize::from(id);
    tape.seek(SeekFrom::Start(offset as u64))?;
    let mut bytes = [0; 8];
    tape.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn witness(
    e: Edge,
    kind: &str,
    pair: usize,
    rows: &[Row],
    roots: &[u16; IDS],
    tape: &mut File,
    tables: &[u8; 4352],
    arm: &str,
    out: &mut impl Write,
) -> Result<()> {
    let row = rows[usize::from(e.row)];
    let gold = address(row.carry, roots[usize::from(row.gold)], tables, arm);
    let rival = address(row.carry, roots[usize::from(e.rival)], tables, arm);
    let gold_prior = read_mass(tape, e, row.gold)?;
    let rival_prior = read_mass(tape, e, e.rival)?;
    let bound = (rival_prior as i64 - gold_prior as i64).div_euclid(UNIT) + 1;
    let mut signed_terms = Vec::new();
    let mut reverse_terms = Vec::new();
    for (i, (&g, &r)) in gold.iter().zip(&rival).enumerate().rev() {
        if g != r {
            signed_terms.push((i, g, r));
            reverse_terms.push((i, r, g));
        }
    }
    let reversed = signed_terms > reverse_terms;
    let packed = signed_terms
        .iter()
        .map(|&(bank, g, r)| {
            let (left, right) = if reversed { (r, g) } else { (g, r) };
            ((u64::from(left) << 6) | u64::from(right)) << (12 * bank)
        })
        .sum::<u64>();
    if ((packed << 1) | u64::from(reversed)) != e.key || i64::from(e.bound) != bound {
        return Err("selected witness differs on host re-read".into());
    }

    writeln!(
        out,
        "{arm}\t{kind}\t{pair}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{gold_prior}\t{rival_prior}\t{bound}\t{:016x}",
        row.ordinal,
        row.group,
        row.carry,
        row.gold,
        e.rival,
        roots[usize::from(row.gold)],
        roots[usize::from(e.rival)],
        gold,
        rival,
        e.key
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let [_, table_file, roots_file, row_file, arm, gpu_edges, output] = args.as_slice() else {
        return Err("usage: match TABLE ROOTS ROW_TAPE ARM CONSTRAINT_IN CERTIFICATE_OUT".into());
    };
    if !["baseline", "product", "unary"].contains(&arm.as_str()) {
        return Err("unknown arm".into());
    }
    let tables: [u8; 4352] = std::fs::read(table_file)?
        .try_into()
        .map_err(|_| "invalid native table length")?;
    if tables.iter().any(|&value| value >= 64) {
        return Err("invalid native state".into());
    }
    let root_bytes = std::fs::read(roots_file)?;
    if root_bytes.len() != IDS * 2 {
        return Err("invalid original-ID root vector length".into());
    }
    let roots =
        std::array::from_fn(|i| u16::from_le_bytes([root_bytes[2 * i], root_bytes[2 * i + 1]]));
    let mut tape = File::open(row_file)?;
    if tape.metadata()?.len() != (ROW_BYTES * ROWS) as u64 {
        return Err("invalid row tape length".into());
    }
    let mut rows = Vec::with_capacity(ROWS);
    for index in 0..ROWS {
        tape.seek(SeekFrom::Start((index * ROW_BYTES) as u64))?;
        let mut header = [0; 12];
        tape.read_exact(&mut header)?;
        let row = Row {
            ordinal: u32::from_le_bytes(header[0..4].try_into()?),
            group: u32::from_le_bytes(header[4..8].try_into()?),
            gold: u16::from_le_bytes(header[8..10].try_into()?),
            carry: u16::from_le_bytes(header[10..12].try_into()?),
        };
        if row.gold == 0
            || usize::from(row.gold) >= IDS
            || rows
                .last()
                .is_some_and(|previous: &Row| previous.ordinal >= row.ordinal)
        {
            return Err("invalid ordinal or gold ID".into());
        }
        rows.push(row);
    }

    let mut edges = Vec::with_capacity(ROWS * (IDS - 2));
    let mut killed: Vec<Option<Edge>> = vec![None; ROWS];
    let mut input = BufReader::new(File::open(gpu_edges)?);
    let mut bytes = [0; 16];
    for (row_index, row) in rows.iter().enumerate() {
        for rival in 1..IDS {
            input.read_exact(&mut bytes)?;
            let edge = Edge {
                key: u64::from_le_bytes(bytes[..8].try_into()?),
                row: u16::from_le_bytes(bytes[8..10].try_into()?),
                rival: u16::from_le_bytes(bytes[10..12].try_into()?),
                bound: i32::from_le_bytes(bytes[12..16].try_into()?),
            };
            if usize::from(edge.row) != row_index
                || usize::from(edge.rival) != rival
                || edge.key >= 1 << 61
                || !(-65535..=65537).contains(&edge.bound)
            {
                return Err("invalid or incomplete GPU constraint tape".into());
            }
            if edge.rival == row.gold {
                if edge.key != 0 || edge.bound != 0 {
                    return Err("nonzero self comparison".into());
                }
            } else if edge.key != 0 {
                edges.push(edge);
            } else if edge.bound > 0 && killed[row_index].is_none_or(|old| old.bound < edge.bound) {
                killed[row_index] = Some(edge);
            }
        }
    }
    if input.read(&mut bytes)? != 0 {
        return Err("trailing GPU constraint bytes".into());
    }
    let (pairs, unique) = matching(edges, &killed, &rows);
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    writeln!(
        out,
        "arm\tkind\tpair\tordinal\tgroup\tcarry\tgold\trival\tgold_root\trival_root\tgold_slots\trival_slots\tgold_prior_q48\trival_prior_q48\tstrict_min\tkey"
    )?;
    for e in killed.iter().flatten().copied() {
        witness(
            e,
            "singleton",
            0,
            &rows,
            &roots,
            &mut tape,
            &tables,
            arm,
            &mut out,
        )?;
    }
    let mut claimed: Vec<_> = killed.iter().map(Option::is_some).collect();
    for (pair, &[left, right]) in pairs.iter().enumerate() {
        if !opposite(left, right, &rows) {
            return Err("selected pair is not contradictory".into());
        }
        for e in [left, right] {
            if std::mem::replace(&mut claimed[usize::from(e.row)], true) {
                return Err("selected certificate reuses an opening".into());
            }
            witness(
                e,
                "pair",
                pair + 1,
                &rows,
                &roots,
                &mut tape,
                &tables,
                arm,
                &mut out,
            )?;
        }
    }
    out.flush()?;
    let singletons = killed.iter().flatten().count();
    let upper = ROWS - singletons - pairs.len();
    println!(
        "{arm} rows={ROWS} rivals_per_row={} comparisons={} row_signature_directions={unique} singletons={singletons} disjoint_pairs={} strict_win_upper_bound={upper} bar={BAR}",
        IDS - 2,
        ROWS * (IDS - 2),
        pairs.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cannot_pair_an_occupied_direct_row_as_a_later_reverse() {
        let rows = [1, 2, 3].map(|group| Row {
            ordinal: group,
            group,
            gold: 1,
            carry: 0,
        });
        let edge = |key, row| Edge {
            key,
            row,
            rival: 2,
            bound: 1,
        };
        let (pairs, _) = matching(
            vec![edge(2, 0), edge(2, 1), edge(3, 0), edge(3, 2)],
            &[None; 3],
            &rows,
        );
        assert_eq!(pairs, [[edge(2, 0), edge(3, 2)]]);
    }

    #[test]
    fn finite_native_address_masks_and_order() {
        let bytes = std::fs::read(std::env::var("D11_TABLES").unwrap()).unwrap();
        let tables: &[u8; 4352] = bytes.as_slice().try_into().unwrap();
        let mut masks = [[0u64; 5]; 3];
        for source in 0..256 {
            for target in 0..256 {
                for (arm_index, arm) in ["baseline", "product", "unary"].iter().enumerate() {
                    for (bank, slot) in address(source, target, tables, arm).into_iter().enumerate()
                    {
                        masks[arm_index][bank] |= 1 << slot;
                    }
                }
            }
        }
        for mask in masks {
            assert_eq!(mask[..2], [0x0001860c0d01ffff; 2]);
        }
        assert_eq!(masks[0][4], 1);
        assert_eq!(masks[1][4], u64::MAX);
        assert_eq!(masks[2][4], u64::MAX);
        assert_eq!(tables[256 + 64 + 2], 6);
        assert_eq!(tables[256 + 128 + 1], 27);
    }

    #[test]
    fn matching_preserves_strict_cross_group_disjointness() {
        let row = Row {
            ordinal: 0,
            group: 4,
            gold: 17,
            carry: 2,
        };
        let rows = [row, row, Row { group: 9, ..row }, Row { group: 10, ..row }];
        let e = Edge {
            key: 2,
            row: 0,
            rival: 3,
            bound: -1,
        };
        let reverse = Edge {
            key: 3,
            row: 1,
            rival: 17,
            bound: 9,
        };
        let equality_only = Edge {
            row: 2,
            bound: 1,
            ..reverse
        };
        let contradiction = Edge {
            row: 2,
            bound: 2,
            ..reverse
        };
        let late = Edge { row: 3, ..e };
        assert!(!opposite(e, reverse, &rows));
        assert!(!opposite(e, equality_only, &rows));
        assert!(opposite(e, contradiction, &rows));
        let (pairs, unique) = matching(
            vec![e, reverse, equality_only, contradiction, late],
            &[None; 4],
            &rows,
        );
        assert_eq!(unique, 4);
        assert_eq!(pairs, [[e, contradiction], [late, reverse]]);
    }
}
