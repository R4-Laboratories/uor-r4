#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::{error::Error, fs, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Pair {
    pair_id: usize,
    a: Side,
    b: Side,
}

#[derive(Deserialize)]
struct Side {
    native: Arm,
    swap: Arm,
    zero: Arm,
}

#[derive(Deserialize)]
struct Arm {
    own: Target,
    other: Target,
}

#[derive(Deserialize)]
struct Target {
    target_mass_q48: u64,
    token: u16,
    cell: usize,
    diagnostic_within_cell_rank: usize,
}

#[derive(Default, Serialize)]
struct ZeroAudit {
    own: usize,
    other: usize,
    both: usize,
}

#[derive(Default, Serialize)]
struct Comparisons {
    native_higher_rows: usize,
    native_lower_rows: usize,
    tied_rows: usize,
    both_native_higher_pairs: usize,
    one_higher_one_lower_pairs: usize,
    both_native_lower_pairs: usize,
    one_higher_one_tied_pairs: usize,
    one_lower_one_tied_pairs: usize,
    both_tied_pairs: usize,
}

impl Comparisons {
    fn record(&mut self, [a, b]: [Ordering; 2]) {
        use Ordering::{Equal, Greater, Less};
        for sign in [a, b] {
            match sign {
                Greater => self.native_higher_rows += 1,
                Less => self.native_lower_rows += 1,
                Equal => self.tied_rows += 1,
            }
        }
        match (a, b) {
            (Greater, Greater) => self.both_native_higher_pairs += 1,
            (Greater, Less) | (Less, Greater) => self.one_higher_one_lower_pairs += 1,
            (Less, Less) => self.both_native_lower_pairs += 1,
            (Greater, Equal) | (Equal, Greater) => self.one_higher_one_tied_pairs += 1,
            (Less, Equal) | (Equal, Less) => self.one_lower_one_tied_pairs += 1,
            (Equal, Equal) => self.both_tied_pairs += 1,
        }
    }
}

fn prefer(native: &Arm, changed: &Arm) -> Ordering {
    let intact = u128::from(native.own.target_mass_q48) * u128::from(changed.other.target_mass_q48);
    let intervened =
        u128::from(changed.own.target_mass_q48) * u128::from(native.other.target_mass_q48);
    intact.cmp(&intervened)
}

fn same_cell_rank(arm: &Arm) -> Ordering {
    arm.other
        .diagnostic_within_cell_rank
        .cmp(&arm.own.diagnostic_within_cell_rank)
}

fn validate_target(target: &Target, native: &Target) -> Result<()> {
    if target.target_mass_q48 > (1 << 48)
        || target.token == 0
        || target.token >= 4096
        || target.token != native.token
        || target.cell != native.cell
        || target.cell >= 64
        || !(1..=64).contains(&target.diagnostic_within_cell_rank)
    {
        return Err("invalid target mass, identity, cell or rank".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [raw] = args.as_slice() else {
        return Err("usage: odds ARCHIVED_GPU_ROWS.jsonl".into());
    };
    let raw = fs::read_to_string(raw)?;
    let mut seen = [false; 64];
    let mut swap = Comparisons::default();
    let mut zero = Comparisons::default();
    let mut zero_counts: [ZeroAudit; 3] = std::array::from_fn(|_| ZeroAudit::default());
    let mut same_cell_rows = 0;
    let mut same_cell_pairs = 0;
    let mut same_cell_preferences = [[0; 3]; 3];
    let mut same_cell_native_own_swap_other = 0;
    for line in raw.lines() {
        let pair: Pair = serde_json::from_str(line)?;
        if !(1..=64).contains(&pair.pair_id) || std::mem::replace(&mut seen[pair.pair_id - 1], true)
        {
            return Err("duplicate or out-of-range pair ID".into());
        }
        if pair.a.native.own.token != pair.b.native.other.token
            || pair.a.native.other.token != pair.b.native.own.token
            || pair.a.native.own.token == pair.b.native.own.token
        {
            return Err("mismatched cross-reference token identities".into());
        }
        let mut rows_with_same_cell = 0;
        for side in [&pair.a, &pair.b] {
            for (index, arm) in [&side.native, &side.swap, &side.zero]
                .into_iter()
                .enumerate()
            {
                validate_target(&arm.own, &side.native.own)?;
                validate_target(&arm.other, &side.native.other)?;
                let own_zero = arm.own.target_mass_q48 == 0;
                let other_zero = arm.other.target_mass_q48 == 0;
                zero_counts[index].own += usize::from(own_zero);
                zero_counts[index].other += usize::from(other_zero);
                zero_counts[index].both += usize::from(own_zero && other_zero);
                if side.native.own.cell == side.native.other.cell {
                    let pref = match same_cell_rank(arm) {
                        Ordering::Greater => 0,
                        Ordering::Less => 1,
                        Ordering::Equal => 2,
                    };
                    same_cell_preferences[index][pref] += 1;
                }
            }
            if side.native.own.cell == side.native.other.cell {
                rows_with_same_cell += 1;
                same_cell_native_own_swap_other += usize::from(
                    same_cell_rank(&side.native) == Ordering::Greater
                        && same_cell_rank(&side.swap) == Ordering::Less,
                );
            }
        }
        same_cell_rows += rows_with_same_cell;
        same_cell_pairs += usize::from(rows_with_same_cell == 2);
        swap.record([
            prefer(&pair.a.native, &pair.a.swap),
            prefer(&pair.b.native, &pair.b.swap),
        ]);
        zero.record([
            prefer(&pair.a.native, &pair.a.zero),
            prefer(&pair.b.native, &pair.b.zero),
        ]);
    }
    if !seen.into_iter().all(|present| present) {
        return Err("missing archived pair IDs".into());
    }
    serde_json::to_writer_pretty(
        io::stdout().lock(),
        &serde_json::json!({
            "analysis": "Exploratory post hoc; sign of exact u128 comparison native_own*changed_other vs changed_own*native_other",
            "pairs": 64,
            "rows": 128,
            "native_vs_swap": swap,
            "native_vs_zero": zero,
            "zero_mass_rows": {"native": zero_counts[0], "swap": zero_counts[1], "zero": zero_counts[2]},
            "same_cell": {
                "pairs": same_cell_pairs,
                "rows": same_cell_rows,
                "preference_array_order": ["own better rank", "other better rank", "tie"],
                "native": same_cell_preferences[0],
                "swap": same_cell_preferences[1],
                "zero": same_cell_preferences[2],
                "rows_native_own_better_swap_other_better": same_cell_native_own_swap_other,
            },
        }),
    )?;
    println!();
    Ok(())
}
