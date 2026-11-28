//! Integer checkpoint gates and paired whole-conversation bootstrap receipts.
use serde::Serialize;
use std::collections::BTreeMap;

const RESAMPLES: usize = 4096;
const PPM: i128 = 1_000_000;

#[derive(Clone, Copy, Debug, Default)]
pub struct ElectionRow {
    pub group: u64,
    pub centered: bool,
    pub frozen: bool,
    pub historical: bool,
    pub uncentered: bool,
    pub ordered: bool,
    pub centered_immediate: bool,
    pub centered_newly_corrected: bool,
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
    pub centered_strict: u64,
    pub frozen_strict: u64,
    pub historical_strict: u64,
    pub uncentered_strict: u64,
    pub ordered_strict: u64,
    pub centered_immediate: u64,
    pub centered_retained: u64,
    pub centered_newly_corrected: u64,
    pub centered_newly_retained: u64,
    pub centered_minus_historical: Band,
    pub centered_minus_frozen: Band,
    pub centered_minus_uncentered: Band,
    pub centered_minus_ordered: Band,
    pub centered_retention: Band,
    pub centered_newly_retention: Band,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    rows: u64,
    centered: u64,
    frozen: u64,
    historical: u64,
    uncentered: u64,
    ordered: u64,
    immediate: u64,
    retained: u64,
    newly_corrected: u64,
    newly_retained: u64,
}

impl Counts {
    fn observe(&mut self, row: ElectionRow) {
        self.rows += 1;
        self.centered += u64::from(row.centered);
        self.frozen += u64::from(row.frozen);
        self.historical += u64::from(row.historical);
        self.uncentered += u64::from(row.uncentered);
        self.ordered += u64::from(row.ordered);
        self.immediate += u64::from(row.centered_immediate);
        self.retained += u64::from(row.centered_immediate && row.centered);
        self.newly_corrected += u64::from(row.centered_newly_corrected);
        self.newly_retained += u64::from(row.centered_newly_corrected && row.centered);
    }

    fn add(&mut self, group: Self) {
        self.rows += group.rows;
        self.centered += group.centered;
        self.frozen += group.frozen;
        self.historical += group.historical;
        self.uncentered += group.uncentered;
        self.ordered += group.ordered;
        self.immediate += group.immediate;
        self.retained += group.retained;
        self.newly_corrected += group.newly_corrected;
        self.newly_retained += group.newly_retained;
    }

    fn centered_minus(self, control: u64) -> i128 {
        i128::from(self.centered) - i128::from(control)
    }

    fn rates(self) -> [Option<i64>; 6] {
        [
            ratio_ppm(self.centered_minus(self.historical), self.rows),
            ratio_ppm(self.centered_minus(self.frozen), self.rows),
            ratio_ppm(self.centered_minus(self.uncentered), self.rows),
            ratio_ppm(self.centered_minus(self.ordered), self.rows),
            ratio_ppm(i128::from(self.retained), self.immediate),
            ratio_ppm(i128::from(self.newly_retained), self.newly_corrected),
        ]
    }
}

fn ratio_ppm(numerator: i128, denominator: u64) -> Option<i64> {
    if denominator == 0 {
        return None;
    }
    let denominator = i128::from(denominator);
    let magnitude = (numerator.abs() * PPM + denominator / 2) / denominator;
    Some((numerator.signum() * magnitude) as i64)
}

fn draw(state: &mut u64, bound: u64) -> usize {
    // SplitMix64 with #4355's seed and rejection rule.
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

/// Grade a single seed and epoch on its complete common-attainable cohort.
/// Held-group grades use endpoint-only decisions and carry no visit observations.
pub fn compare(rows: &[ElectionRow], require_retention: bool) -> Result<Comparison, String> {
    let mut total = Counts::default();
    let mut groups = BTreeMap::<u64, Counts>::new();
    for &row in rows {
        if row.centered_newly_corrected && !row.centered_immediate {
            return Err("newly corrected row lacks an immediate strict win".to_owned());
        }
        if !require_retention && row.centered_immediate {
            return Err("held row has an immediate fit observation".to_owned());
        }
        total.observe(row);
        groups.entry(row.group).or_default().observe(row);
    }
    let groups: Vec<_> = groups.into_values().collect();
    let resamples = if groups.is_empty() { 0 } else { RESAMPLES };
    let mut state = 0x606f_35d3_8e69_eb4d;
    let mut distributions: [Vec<i64>; 6] = std::array::from_fn(|_| Vec::with_capacity(resamples));
    for _ in 0..resamples {
        let mut sample = Counts::default();
        for _ in &groups {
            sample.add(groups[draw(&mut state, groups.len() as u64)]);
        }
        for (distribution, rate) in distributions.iter_mut().zip(sample.rates()) {
            distribution.extend(rate);
        }
    }
    let observed = total.rates();
    let [centered_minus_historical, centered_minus_frozen, centered_minus_uncentered, centered_minus_ordered, centered_retention, centered_newly_retention] =
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

    let meets_endpoints = total.rows > 0
        && [
            total.historical,
            total.frozen,
            total.uncentered,
            total.ordered,
        ]
        .into_iter()
        .all(|control| 20 * total.centered_minus(control) >= i128::from(total.rows));
    let meets_retention =
        total.immediate > 0 && 5 * u128::from(total.retained) >= 4 * u128::from(total.immediate);
    Ok(Comparison {
        pass: meets_endpoints && (!require_retention || meets_retention),
        meets_endpoints,
        meets_retention,
        rows: total.rows,
        groups: groups.len(),
        resamples,
        centered_strict: total.centered,
        frozen_strict: total.frozen,
        historical_strict: total.historical,
        uncentered_strict: total.uncentered,
        ordered_strict: total.ordered,
        centered_immediate: total.immediate,
        centered_retained: total.retained,
        centered_newly_corrected: total.newly_corrected,
        centered_newly_retained: total.newly_retained,
        centered_minus_historical,
        centered_minus_frozen,
        centered_minus_uncentered,
        centered_minus_ordered,
        centered_retention,
        centered_newly_retention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_exact_endpoint_and_retention_inequalities() {
        let mut rows = (0..20)
            .map(|i| ElectionRow {
                group: i / 2,
                centered: i < 4,
                frozen: i < 3,
                historical: i < 3,
                uncentered: i < 3,
                ordered: i < 3,
                centered_immediate: i < 5,
                centered_newly_corrected: (2..5).contains(&i),
            })
            .collect::<Vec<_>>();
        let exact = compare(&rows, true).unwrap();
        assert!(exact.pass);
        assert_eq!(exact.centered_minus_historical.observed_ppm, Some(50_000));
        assert_eq!(exact.centered_retention.observed_ppm, Some(800_000));
        assert_eq!(
            (
                exact.centered_newly_corrected,
                exact.centered_newly_retained
            ),
            (3, 2)
        );
        rows.reverse();
        assert_eq!(compare(&rows, true).unwrap(), exact);

        for control in [
            |row: &mut ElectionRow, value| row.historical = value,
            |row: &mut ElectionRow, value| row.frozen = value,
            |row: &mut ElectionRow, value| row.uncentered = value,
            |row: &mut ElectionRow, value| row.ordered = value,
        ] {
            control(&mut rows[0], true);
            let failed = compare(&rows, true).unwrap();
            assert!(!failed.pass);
            assert!(!failed.meets_endpoints);
            assert!(failed.meets_retention);
            control(&mut rows[0], false);
        }
        rows[0].centered_immediate = true;
        let failed = compare(&rows, true).unwrap();
        assert!(!failed.pass);
        assert!(failed.meets_endpoints);
        assert!(!failed.meets_retention);
    }

    #[test]
    fn needs_195_wins_on_the_registered_cohort() {
        let mut rows = (0..3886)
            .map(|i| ElectionRow {
                group: 1,
                centered: i < 194,
                ..ElectionRow::default()
            })
            .collect::<Vec<_>>();
        assert!(!compare(&rows, false).unwrap().pass);
        rows[194].centered = true;
        assert!(compare(&rows, false).unwrap().pass);
        assert!(!compare(&rows, true).unwrap().pass);
    }

    #[test]
    fn empty_and_invalid_cohorts_are_handled_explicitly() {
        let empty = compare(&[], false).unwrap();
        assert!(!empty.pass);
        assert_eq!((empty.rows, empty.groups, empty.resamples), (0, 0, 0));
        assert_eq!(empty.centered_minus_historical, Band::default());
        let mut invalid = ElectionRow {
            centered_newly_corrected: true,
            ..ElectionRow::default()
        };
        assert!(compare(&[invalid], true).is_err());
        invalid.centered_immediate = true;
        assert!(compare(&[invalid], false).is_err());
    }

    #[test]
    fn preserves_unequal_group_weights_in_paired_samples() {
        let winning = ElectionRow {
            group: 11,
            centered: true,
            centered_immediate: true,
            ..ElectionRow::default()
        };
        let losing = ElectionRow {
            group: 99,
            ..ElectionRow::default()
        };
        let grade = compare(&[winning, winning, winning, losing], true).unwrap();
        assert_eq!(
            grade.centered_minus_uncentered,
            Band {
                observed_ppm: Some(750_000),
                lower_ppm: Some(0),
                upper_ppm: Some(1_000_000),
                defined_resamples: RESAMPLES,
            }
        );
        assert_eq!(grade.centered_retention.lower_ppm, Some(1_000_000));
        assert!(grade.centered_retention.defined_resamples < RESAMPLES);
        assert_eq!(grade.centered_newly_retention, Band::default());
    }
}
