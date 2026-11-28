//! Paired, integer grading of the common-attainable lexical election cohort.
use serde::Serialize;
use std::collections::BTreeMap;

const RESAMPLES: usize = 4096;
const PPM: i128 = 1_000_000;

#[derive(Clone, Copy, Debug, Default)]
pub struct ElectionRow {
    pub group: u64,
    pub zero: bool,
    pub old: bool,
    pub geometry: bool,
    pub ordered: bool,
    pub geometry_immediate: bool,
    pub ordered_immediate: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Band {
    pub observed_ppm: Option<i64>,
    pub lower_ppm: Option<i64>,
    pub upper_ppm: Option<i64>,
    pub defined_resamples: usize,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct Comparison {
    pub pass: bool,
    pub meets_endpoints: bool,
    pub meets_retention: bool,
    pub rows: u64,
    pub groups: usize,
    pub resamples: usize,
    pub zero_strict: u64,
    pub old_strict: u64,
    pub geometry_strict: u64,
    pub ordered_strict: u64,
    pub geometry_immediate: u64,
    pub geometry_retained: u64,
    pub ordered_immediate: u64,
    pub ordered_retained: u64,
    pub geometry_minus_old: Band,
    pub geometry_minus_zero: Band,
    pub geometry_minus_ordered: Band,
    pub geometry_retention: Band,
    pub ordered_retention: Band,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    rows: u64,
    zero: u64,
    old: u64,
    geometry: u64,
    ordered: u64,
    geometry_immediate: u64,
    geometry_retained: u64,
    ordered_immediate: u64,
    ordered_retained: u64,
}

impl Counts {
    fn observe(&mut self, row: ElectionRow) {
        self.rows += 1;
        self.zero += u64::from(row.zero);
        self.old += u64::from(row.old);
        self.geometry += u64::from(row.geometry);
        self.ordered += u64::from(row.ordered);
        self.geometry_immediate += u64::from(row.geometry_immediate);
        self.geometry_retained += u64::from(row.geometry_immediate && row.geometry);
        self.ordered_immediate += u64::from(row.ordered_immediate);
        self.ordered_retained += u64::from(row.ordered_immediate && row.ordered);
    }

    fn add(&mut self, group: Self) {
        self.rows += group.rows;
        self.zero += group.zero;
        self.old += group.old;
        self.geometry += group.geometry;
        self.ordered += group.ordered;
        self.geometry_immediate += group.geometry_immediate;
        self.geometry_retained += group.geometry_retained;
        self.ordered_immediate += group.ordered_immediate;
        self.ordered_retained += group.ordered_retained;
    }

    fn geometry_minus(self, control: u64) -> i128 {
        i128::from(self.geometry) - i128::from(control)
    }

    fn rates(self) -> [Option<i64>; 5] {
        [
            ratio_ppm(self.geometry_minus(self.old), self.rows),
            ratio_ppm(self.geometry_minus(self.zero), self.rows),
            ratio_ppm(self.geometry_minus(self.ordered), self.rows),
            ratio_ppm(i128::from(self.geometry_retained), self.geometry_immediate),
            ratio_ppm(i128::from(self.ordered_retained), self.ordered_immediate),
        ]
    }
}

fn ratio_ppm(numerator: i128, denominator: u64) -> Option<i64> {
    if denominator == 0 {
        return None;
    }
    // Round receipts to the nearest ppm. The gate uses exact cross-products.
    let denominator = i128::from(denominator);
    let magnitude = (numerator.abs() * PPM + denominator / 2) / denominator;
    Some((numerator.signum() * magnitude) as i64)
}

fn draw(state: &mut u64, bound: u64) -> usize {
    // SplitMix64, with the same seed and rejection rule as #4354.
    let threshold = bound.wrapping_neg() % bound;
    loop {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut bits = *state;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        bits ^= bits >> 31;
        if bits >= threshold {
            return (bits % bound) as usize;
        }
    }
}

/// Each row must belong to the structural intersection for both address operators.
pub fn compare(rows: &[ElectionRow], require_retention: bool) -> Result<Comparison, String> {
    let row_count = u64::try_from(rows.len()).map_err(|_| "row count exceeds u64".to_owned())?;
    let mut total = Counts::default();
    let mut groups = BTreeMap::<u64, Counts>::new();
    for &row in rows {
        total.observe(row);
        groups.entry(row.group).or_default().observe(row);
    }
    let groups: Vec<_> = groups.into_values().collect();
    let resamples = if groups.is_empty() { 0 } else { RESAMPLES };
    let mut state = 0x606f_35d3_8e69_eb4d;
    let mut distributions: [Vec<i64>; 5] = std::array::from_fn(|_| Vec::with_capacity(resamples));
    for _ in 0..resamples {
        let mut sample = Counts::default();
        // Draw whole conversations together, retaining each conversation's row count.
        for _ in &groups {
            sample.add(groups[draw(&mut state, groups.len() as u64)]);
        }
        for (distribution, rate) in distributions.iter_mut().zip(sample.rates()) {
            if let Some(rate) = rate {
                distribution.push(rate);
            }
        }
    }
    let observed = total.rates();
    let [geometry_minus_old, geometry_minus_zero, geometry_minus_ordered, geometry_retention, ordered_retention] =
        std::array::from_fn(|i| {
            let values = &mut distributions[i];
            values.sort_unstable();
            let quantile = |thousandths| {
                values
                    .len()
                    .checked_sub(1)
                    .map(|last| values[last * thousandths / 1000])
            };
            Band {
                observed_ppm: observed[i],
                lower_ppm: quantile(25),
                upper_ppm: quantile(975),
                defined_resamples: values.len(),
            }
        });

    let meets_endpoints = row_count != 0
        && [total.old, total.zero, total.ordered]
            .into_iter()
            .all(|control| 20 * total.geometry_minus(control) >= i128::from(row_count));
    let meets_retention = total.geometry_immediate != 0
        && 5 * u128::from(total.geometry_retained) >= 4 * u128::from(total.geometry_immediate);
    Ok(Comparison {
        pass: meets_endpoints && (!require_retention || meets_retention),
        meets_endpoints,
        meets_retention,
        rows: row_count,
        groups: groups.len(),
        resamples,
        zero_strict: total.zero,
        old_strict: total.old,
        geometry_strict: total.geometry,
        ordered_strict: total.ordered,
        geometry_immediate: total.geometry_immediate,
        geometry_retained: total.geometry_retained,
        ordered_immediate: total.ordered_immediate,
        ordered_retained: total.ordered_retained,
        geometry_minus_old,
        geometry_minus_zero,
        geometry_minus_ordered,
        geometry_retention,
        ordered_retention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary() -> Vec<ElectionRow> {
        (0..20)
            .map(|i| ElectionRow {
                group: i / 2,
                zero: i < 3,
                old: i < 3,
                geometry: i < 4,
                ordered: i < 3,
                geometry_immediate: i < 5,
                ordered_immediate: i < 6,
            })
            .collect()
    }

    #[test]
    fn passes_at_exact_checkpoint_and_retention_boundaries() {
        let mut rows = boundary();
        let passed = compare(&rows, true).unwrap();
        assert!(passed.pass);
        assert_eq!(passed.geometry_minus_old.observed_ppm, Some(50_000));
        assert_eq!(passed.geometry_minus_zero.observed_ppm, Some(50_000));
        assert_eq!(passed.geometry_minus_ordered.observed_ppm, Some(50_000));
        assert_eq!(passed.geometry_retention.observed_ppm, Some(800_000));
        assert_eq!(passed.ordered_retention.observed_ppm, Some(500_000));
        rows.reverse();
        assert_eq!(compare(&rows, true).unwrap(), passed);
    }

    #[test]
    fn each_control_independently_blocks_advancement() {
        let mut rows = boundary();
        rows[4].zero = true;
        assert!(!compare(&rows, true).unwrap().pass);
        rows[4].zero = false;
        rows[4].old = true;
        assert!(!compare(&rows, true).unwrap().pass);
        rows[4].old = false;
        rows[4].ordered = true;
        assert!(!compare(&rows, true).unwrap().pass);
    }

    #[test]
    fn retention_deficit_blocks_fit_but_endpoints_can_confirm() {
        let mut rows = boundary();
        rows[7].geometry_immediate = true;
        let failed = compare(&rows, true).unwrap();
        assert!(!failed.pass);
        assert!(failed.meets_endpoints);
        assert_eq!(failed.geometry_retained, 4);
        assert_eq!(failed.geometry_immediate, 6);
        for row in &mut rows {
            row.geometry_immediate = false;
            row.ordered_immediate = false;
        }
        assert!(!compare(&rows, true).unwrap().pass);
        let confirmation = compare(&rows, false).unwrap();
        assert!(confirmation.pass);
        assert_eq!(confirmation.geometry_retention, Band::default());
        assert_eq!(confirmation.ordered_retention, Band::default());
    }

    #[test]
    fn empty_cohort_has_no_pass_or_defined_bands() {
        for require_retention in [false, true] {
            let empty = compare(&[], require_retention).unwrap();
            assert!(!empty.pass);
            assert_eq!((empty.rows, empty.groups, empty.resamples), (0, 0, 0));
            assert_eq!(empty.geometry_minus_old, Band::default());
            assert_eq!(empty.geometry_retention, Band::default());
        }
    }

    #[test]
    fn paired_resampling_preserves_unequal_group_weights() {
        let winning = ElectionRow {
            group: 11,
            geometry: true,
            geometry_immediate: true,
            ..ElectionRow::default()
        };
        let losing = ElectionRow {
            group: 99,
            ..ElectionRow::default()
        };
        let comparison = compare(&[winning, winning, winning, losing], true).unwrap();
        assert_eq!(
            comparison.geometry_minus_old,
            Band {
                observed_ppm: Some(750_000),
                lower_ppm: Some(0),
                upper_ppm: Some(1_000_000),
                defined_resamples: RESAMPLES,
            }
        );
        assert_eq!(comparison.geometry_retention.lower_ppm, Some(1_000_000));
        assert!(comparison.geometry_retention.defined_resamples < RESAMPLES);
    }
}
