//! Integer reduction of device observations. This module does not run a model.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const SEEDS: [u32; 2] = [435300, 435301];
const MAX_ID: u32 = 4095;
const ONE_Q48: u64 = 1 << 48;
const RESAMPLES: usize = 4096;
const PPM: i128 = 1_000_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Election {
    pub margin_q48: i64,
    pub rival: u32,
    pub rank: u32,
}

impl Election {
    fn strict(self) -> bool {
        self.margin_q48 > 0
    }

    fn validate(self, gold: u32) -> Result<(), &'static str> {
        if !(1..=MAX_ID).contains(&self.rival) || self.rival == gold {
            return Err("invalid original rival ID");
        }
        let elects_gold = self.strict() || (self.margin_q48 == 0 && gold < self.rival);
        if !(1..=MAX_ID).contains(&self.rank) || (self.rank == 1) != elects_gold {
            return Err("rank disagrees with margin or original-ID tie priority");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RowImpact {
    pub ordinal: u32,
    pub group: u64,
    pub gold_id: u32,
    pub prior_rank: u32,
    pub alias_count: u32,
    pub ceiling: Election,
    pub zero: Election,
    pub before: Election,
    pub before_rival_prior_q48: u64,
    pub old_rival_margin_q48: i64,
    pub after: Election,
    pub clipped_words: u32,
    pub checkpoint: Election,
}

pub struct SeedInput<'a> {
    pub seed: u32,
    pub geometry: &'a [RowImpact],
    pub ordered: &'a [RowImpact],
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ConfirmationRow {
    pub ordinal: u32,
    pub group: u64,
    pub gold_id: u32,
    pub ceiling: Election,
    pub zero: Election,
    pub checkpoint: Election,
}

pub struct ConfirmationSeedInput<'a> {
    pub seed: u32,
    pub geometry: &'a [ConfirmationRow],
    pub ordered: &'a [ConfirmationRow],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Loss {
    Structural,
    ClippedOldRival,
    OldRival,
    Replacement,
    EpochEnd,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct LossCounts {
    pub structural: u64,
    pub clipped_old_rival: u64,
    pub old_rival: u64,
    pub replacement: u64,
    pub epoch_end: u64,
}

impl LossCounts {
    fn observe(&mut self, loss: Loss) {
        match loss {
            Loss::Structural => self.structural += 1,
            Loss::ClippedOldRival => self.clipped_old_rival += 1,
            Loss::OldRival => self.old_rival += 1,
            Loss::Replacement => self.replacement += 1,
            Loss::EpochEnd => self.epoch_end += 1,
        }
    }

    fn largest(self) -> Vec<Loss> {
        let tally = [
            (Loss::Structural, self.structural),
            (Loss::ClippedOldRival, self.clipped_old_rival),
            (Loss::OldRival, self.old_rival),
            (Loss::Replacement, self.replacement),
            (Loss::EpochEnd, self.epoch_end),
        ];
        let maximum = tally.iter().map(|&(_, count)| count).max().unwrap_or(0);
        tally
            .into_iter()
            .filter_map(|(loss, count)| (count != 0 && count == maximum).then_some(loss))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ElectionCounts {
    pub strict: u64,
    pub ties_won: u64,
    pub ties_lost: u64,
}

impl ElectionCounts {
    fn observe(&mut self, election: Election) {
        self.strict += u64::from(election.strict());
        self.ties_won += u64::from(election.margin_q48 == 0 && election.rank == 1);
        self.ties_lost += u64::from(election.margin_q48 == 0 && election.rank != 1);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Trajectory {
    pub rows: u64,
    pub ceiling: ElectionCounts,
    pub zero: ElectionCounts,
    pub before: ElectionCounts,
    pub old_rival_strict: u64,
    pub old_rival_ties_won: u64,
    pub old_rival_ties_lost: u64,
    pub immediate: ElectionCounts,
    pub checkpoint: ElectionCounts,
    pub retained: u64,
    pub later_gains: u64,
    pub immediate_gains: u64,
    pub clipped_rows: u64,
    pub clipped_words: u64,
    pub aliased_rows: u64,
    pub first_losses: LossCounts,
}

impl Trajectory {
    fn observe(&mut self, row: &RowImpact) {
        self.rows += 1;
        self.ceiling.observe(row.ceiling);
        self.zero.observe(row.zero);
        self.before.observe(row.before);
        self.old_rival_strict += u64::from(row.old_rival_margin_q48 > 0);
        self.old_rival_ties_won +=
            u64::from(row.old_rival_margin_q48 == 0 && row.gold_id < row.before.rival);
        self.old_rival_ties_lost +=
            u64::from(row.old_rival_margin_q48 == 0 && row.gold_id > row.before.rival);
        self.immediate.observe(row.after);
        self.checkpoint.observe(row.checkpoint);
        self.retained += u64::from(row.after.strict() && row.checkpoint.strict());
        self.later_gains += u64::from(!row.after.strict() && row.checkpoint.strict());
        self.immediate_gains += u64::from(!row.before.strict() && row.after.strict());
        self.clipped_rows += u64::from(row.clipped_words != 0);
        self.clipped_words += u64::from(row.clipped_words);
        self.aliased_rows += u64::from(row.alias_count != 0);

        let loss = if !row.ceiling.strict() {
            Loss::Structural
        } else if row.old_rival_margin_q48 <= 0 {
            if row.clipped_words == 0 {
                Loss::OldRival
            } else {
                Loss::ClippedOldRival
            }
        } else if !row.after.strict() {
            Loss::Replacement
        } else if !row.checkpoint.strict() {
            Loss::EpochEnd
        } else {
            return;
        };
        self.first_losses.observe(loss);
    }
}

#[derive(Debug, Default, Serialize)]
pub struct ArmGrade {
    pub all: Trajectory,
    pub common_attainable: Trajectory,
    pub by_prior_rank: BTreeMap<&'static str, Trajectory>,
    pub by_alias: BTreeMap<&'static str, Trajectory>,
    pub by_clipping: BTreeMap<&'static str, Trajectory>,
    pub common_by_prior_rank: BTreeMap<&'static str, Trajectory>,
    pub common_by_alias: BTreeMap<&'static str, Trajectory>,
    pub common_by_clipping: BTreeMap<&'static str, Trajectory>,
    pub dominant_losses: Vec<Loss>,
}

impl ArmGrade {
    fn observe(&mut self, row: &RowImpact, common: bool) {
        let rank = match row.prior_rank {
            1 => "1",
            2..=10 => "2-10",
            11..=100 => "11-100",
            101..=1000 => "101-1000",
            _ => "1001-4095",
        };
        let alias = if row.alias_count == 0 { "none" } else { "any" };
        let clip = if row.clipped_words == 0 {
            "none"
        } else {
            "any"
        };
        self.all.observe(row);
        self.by_prior_rank.entry(rank).or_default().observe(row);
        self.by_alias.entry(alias).or_default().observe(row);
        self.by_clipping.entry(clip).or_default().observe(row);
        if common {
            self.common_attainable.observe(row);
            self.common_by_prior_rank
                .entry(rank)
                .or_default()
                .observe(row);
            self.common_by_alias.entry(alias).or_default().observe(row);
            self.common_by_clipping
                .entry(clip)
                .or_default()
                .observe(row);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PairedCounts {
    rows: u64,
    zero: u64,
    geometry: u64,
    ordered: u64,
    geometry_immediate: u64,
    geometry_retained: u64,
    ordered_immediate: u64,
    ordered_retained: u64,
}

impl PairedCounts {
    fn add(&mut self, other: Self) {
        self.rows += other.rows;
        self.zero += other.zero;
        self.geometry += other.geometry;
        self.ordered += other.ordered;
        self.geometry_immediate += other.geometry_immediate;
        self.geometry_retained += other.geometry_retained;
        self.ordered_immediate += other.ordered_immediate;
        self.ordered_retained += other.ordered_retained;
    }

    fn endpoints(self) -> bool {
        let rows = i128::from(self.rows);
        let geometry = i128::from(self.geometry);
        rows != 0
            && 20 * (geometry - i128::from(self.zero)) >= rows
            && 20 * (geometry - i128::from(self.ordered)) >= rows
    }

    fn retains(self) -> bool {
        self.geometry_immediate != 0
            && 5 * u128::from(self.geometry_retained) >= 4 * u128::from(self.geometry_immediate)
    }

    fn rates(self) -> [Option<i64>; 4] {
        [
            ratio_ppm(i128::from(self.geometry) - i128::from(self.zero), self.rows),
            ratio_ppm(
                i128::from(self.geometry) - i128::from(self.ordered),
                self.rows,
            ),
            ratio_ppm(i128::from(self.geometry_retained), self.geometry_immediate),
            ratio_ppm(i128::from(self.ordered_retained), self.ordered_immediate),
        ]
    }
}

fn ratio_ppm(numerator: i128, denominator: u64) -> Option<i64> {
    (denominator != 0).then(|| {
        let denominator = i128::from(denominator);
        // Round to nearest ppm for the receipt only. Gate checks use exact counts.
        let rounded = (numerator.abs() * PPM + denominator / 2) / denominator;
        (numerator.signum() * rounded) as i64
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Band {
    pub observed_ppm: Option<i64>,
    pub lower_ppm: Option<i64>,
    pub upper_ppm: Option<i64>,
    pub defined_resamples: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PairedGrade {
    pub rows: u64,
    pub groups: usize,
    pub zero_strict: u64,
    pub geometry_strict: u64,
    pub ordered_strict: u64,
    pub geometry_immediate: u64,
    pub geometry_retained: u64,
    pub ordered_immediate: u64,
    pub ordered_retained: u64,
    pub resamples: usize,
    pub geometry_minus_zero: Band,
    pub geometry_minus_ordered: Band,
    pub geometry_retention: Band,
    pub ordered_retention: Band,
    pub meets_endpoints: bool,
    pub meets_retention: bool,
}

#[derive(Debug, Serialize)]
pub struct SeedGrade {
    pub seed: u32,
    pub geometry: ArmGrade,
    pub ordered: ArmGrade,
    pub common_attainable: PairedGrade,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NextAction {
    GroupDisjointTrainConfirmation,
    RepairAddressCoverage,
    RepairClippedUpdate,
    InspectChosenRivalCorrection,
    TestFullElectionCorrection,
    TestSharedWeightInterference,
    NoMatchedDegreeBenefit,
    InspectSeedTrajectories,
    CheckCommonAttainableDomain,
}

#[derive(Debug, Serialize)]
pub struct EpochGrade {
    pub epoch: u32,
    pub seeds: [SeedGrade; 2],
    pub advance: Option<bool>,
    pub next_action: Option<NextAction>,
}

#[derive(Debug, Serialize)]
pub struct ConfirmationGrade {
    pub seeds: [u32; 2],
    pub common_attainable: [PairedGrade; 2],
    pub advance: bool,
}

fn draw(state: &mut u64, bound: u64) -> usize {
    // SplitMix64 with rejection to keep every cluster equiprobable.
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

fn paired(groups: BTreeMap<u64, PairedCounts>) -> PairedGrade {
    let groups: Vec<_> = groups.into_values().collect();
    let mut total = PairedCounts::default();
    for &group in &groups {
        total.add(group);
    }

    let mut state = 0x606f_35d3_8e69_eb4d;
    let mut distributions: [Vec<i64>; 4] = std::array::from_fn(|_| Vec::with_capacity(RESAMPLES));
    let resamples = if groups.is_empty() { 0 } else { RESAMPLES };
    for _ in 0..resamples {
        let mut sample = PairedCounts::default();
        for _ in 0..groups.len() {
            sample.add(groups[draw(&mut state, groups.len() as u64)]);
        }
        for (values, value) in distributions.iter_mut().zip(sample.rates()) {
            if let Some(value) = value {
                values.push(value);
            }
        }
    }
    let bands: [Band; 4] = std::array::from_fn(|i| {
        let values = &mut distributions[i];
        values.sort_unstable();
        let band_end = |thousandths| {
            values
                .len()
                .checked_sub(1)
                .map(|last| values[last * thousandths / 1000])
        };
        Band {
            observed_ppm: total.rates()[i],
            lower_ppm: band_end(25),
            upper_ppm: band_end(975),
            defined_resamples: values.len(),
        }
    });
    PairedGrade {
        rows: total.rows,
        groups: groups.len(),
        zero_strict: total.zero,
        geometry_strict: total.geometry,
        ordered_strict: total.ordered,
        geometry_immediate: total.geometry_immediate,
        geometry_retained: total.geometry_retained,
        ordered_immediate: total.ordered_immediate,
        ordered_retained: total.ordered_retained,
        resamples,
        geometry_minus_zero: bands[0],
        geometry_minus_ordered: bands[1],
        geometry_retention: bands[2],
        ordered_retention: bands[3],
        meets_endpoints: total.endpoints(),
        meets_retention: total.retains(),
    }
}

fn valid_endpoint(row: &ConfirmationRow) -> Result<(), &'static str> {
    if !(1..=MAX_ID).contains(&row.gold_id) {
        return Err("invalid original gold ID");
    }
    for election in [row.ceiling, row.zero, row.checkpoint] {
        election.validate(row.gold_id)?;
        if election.margin_q48 > row.ceiling.margin_q48 || election.rank < row.ceiling.rank {
            return Err("election exceeds exact row ceiling");
        }
    }
    Ok(())
}

impl From<RowImpact> for ConfirmationRow {
    fn from(row: RowImpact) -> Self {
        Self {
            ordinal: row.ordinal,
            group: row.group,
            gold_id: row.gold_id,
            ceiling: row.ceiling,
            zero: row.zero,
            checkpoint: row.checkpoint,
        }
    }
}

fn valid_visit(row: &RowImpact) -> Result<(), &'static str> {
    valid_endpoint(&(*row).into())?;
    for election in [row.before, row.after] {
        election.validate(row.gold_id)?;
        if election.margin_q48 > row.ceiling.margin_q48 || election.rank < row.ceiling.rank {
            return Err("visit exceeds exact row ceiling");
        }
    }
    if row.prior_rank != row.zero.rank || row.alias_count >= MAX_ID {
        return Err("inconsistent prior rank or full-address alias count");
    }
    if row.clipped_words > 8 || !(1..=ONE_Q48).contains(&row.before_rival_prior_q48) {
        return Err("invalid clipping counter or strongest rival prior");
    }
    if row.after.margin_q48 < row.before.margin_q48
        || row.after.rank > row.before.rank
        || row.after.margin_q48 > row.old_rival_margin_q48
        || (row.after.rival == row.before.rival && row.after.margin_q48 != row.old_rival_margin_q48)
        || (row.after.rival > row.before.rival && row.after.margin_q48 == row.old_rival_margin_q48)
    {
        return Err("post-update rival contradicts the recorded old-rival score");
    }
    if row.before.strict()
        && (row.before != row.after
            || row.before.margin_q48 != row.old_rival_margin_q48
            || row.clipped_words != 0)
    {
        return Err("already-strict row changed during its visit");
    }
    Ok(())
}

fn index<T>(rows: &[T], ordinal: impl Fn(&T) -> u32) -> Result<BTreeMap<u32, &T>, String> {
    let mut indexed = BTreeMap::new();
    for row in rows {
        let key = ordinal(row);
        if indexed.insert(key, row).is_some() {
            return Err(format!("duplicate ordinal {key}"));
        }
    }
    Ok(indexed)
}

fn same_electorate(a: &ConfirmationRow, b: &ConfirmationRow) -> bool {
    (a.ordinal, a.group, a.gold_id, a.zero) == (b.ordinal, b.group, b.gold_id, b.zero)
}

fn validate_seeds(seeds: [u32; 2]) -> Result<(), String> {
    if seeds != SEEDS {
        return Err(format!(
            "expected registered seeds {SEEDS:?}, got {seeds:?}"
        ));
    }
    Ok(())
}

fn verify_domain<T>(
    indexed: &[BTreeMap<u32, &T>; 4],
    endpoints: impl Fn(&T) -> ConfirmationRow,
    same_arm: impl Fn(&T, &T) -> bool,
) -> Result<(), String> {
    let [g0, o0, g1, o1] = indexed;
    if [o0, g1, o1]
        .into_iter()
        .any(|map| !map.keys().eq(g0.keys()))
    {
        return Err("registered arms and seeds have different ordinal sets".to_owned());
    }
    for (&ordinal, &reference) in g0 {
        let base = endpoints(reference);
        let o0_row = o0[&ordinal];
        let g1_row = g1[&ordinal];
        let o1_row = o1[&ordinal];
        if [o0_row, g1_row, o1_row]
            .iter()
            .any(|row| !same_electorate(&base, &endpoints(row)))
            || !same_arm(reference, g1_row)
            || !same_arm(o0_row, o1_row)
        {
            return Err(format!(
                "input, zero arm or fixed support drift at ordinal {ordinal}"
            ));
        }
    }
    Ok(())
}

fn pair_updates(geometry: Election, ordered: Election, zero: Election) -> PairedCounts {
    PairedCounts {
        rows: 1,
        zero: u64::from(zero.strict()),
        geometry: u64::from(geometry.strict()),
        ordered: u64::from(ordered.strict()),
        ..PairedCounts::default()
    }
}

fn grade_seed(
    seed: u32,
    geometry: &BTreeMap<u32, &RowImpact>,
    ordered: &BTreeMap<u32, &RowImpact>,
) -> Result<SeedGrade, String> {
    let mut geometric_grade = ArmGrade::default();
    let mut ordered_grade = ArmGrade::default();
    let mut groups = BTreeMap::<u64, PairedCounts>::new();
    for (&ordinal, &geometry) in geometry {
        let ordered = ordered[&ordinal];
        for (name, row) in [("geometry", geometry), ("ordered", ordered)] {
            valid_visit(row)
                .map_err(|error| format!("seed {seed} {name} row {ordinal}: {error}"))?;
        }

        let common = geometry.ceiling.strict() && ordered.ceiling.strict();
        geometric_grade.observe(geometry, common);
        ordered_grade.observe(ordered, common);
        if common {
            let mut pair = pair_updates(geometry.checkpoint, ordered.checkpoint, geometry.zero);
            pair.geometry_immediate = u64::from(geometry.after.strict());
            pair.geometry_retained =
                u64::from(geometry.after.strict() && geometry.checkpoint.strict());
            pair.ordered_immediate = u64::from(ordered.after.strict());
            pair.ordered_retained =
                u64::from(ordered.after.strict() && ordered.checkpoint.strict());
            groups.entry(geometry.group).or_default().add(pair);
        }
    }
    geometric_grade.dominant_losses = geometric_grade.all.first_losses.largest();
    ordered_grade.dominant_losses = ordered_grade.all.first_losses.largest();
    Ok(SeedGrade {
        seed,
        geometry: geometric_grade,
        ordered: ordered_grade,
        common_attainable: paired(groups),
    })
}

fn next_action(seeds: &[SeedGrade; 2], advance: bool) -> NextAction {
    if advance {
        return NextAction::GroupDisjointTrainConfirmation;
    }
    if seeds.iter().any(|seed| seed.common_attainable.rows == 0) {
        return NextAction::CheckCommonAttainableDomain;
    }
    if seeds.iter().all(|seed| {
        let paired = &seed.common_attainable;
        let zero_gain = i128::from(paired.geometry_strict) - i128::from(paired.zero_strict);
        paired.meets_retention
            && 20 * zero_gain >= i128::from(paired.rows)
            && paired.geometry_strict <= paired.ordered_strict
    }) {
        return NextAction::NoMatchedDegreeBenefit;
    }
    match (
        seeds[0].geometry.dominant_losses.as_slice(),
        seeds[1].geometry.dominant_losses.as_slice(),
    ) {
        ([Loss::Structural], [Loss::Structural]) => NextAction::RepairAddressCoverage,
        ([Loss::ClippedOldRival], [Loss::ClippedOldRival]) => NextAction::RepairClippedUpdate,
        ([Loss::OldRival], [Loss::OldRival]) => NextAction::InspectChosenRivalCorrection,
        ([Loss::Replacement], [Loss::Replacement]) => NextAction::TestFullElectionCorrection,
        ([Loss::EpochEnd], [Loss::EpochEnd]) => NextAction::TestSharedWeightInterference,
        _ => NextAction::InspectSeedTrajectories,
    }
}

pub fn grade_epoch(epoch: u32, seeds: &[SeedInput<'_>; 2]) -> Result<EpochGrade, String> {
    if !(1..=8).contains(&epoch) {
        return Err(format!(
            "epoch {epoch} is outside the registered range 1..=8"
        ));
    }
    validate_seeds([seeds[0].seed, seeds[1].seed])?;
    let indexed = [
        index(seeds[0].geometry, |row| row.ordinal)?,
        index(seeds[0].ordered, |row| row.ordinal)?,
        index(seeds[1].geometry, |row| row.ordinal)?,
        index(seeds[1].ordered, |row| row.ordinal)?,
    ];
    verify_domain(
        &indexed,
        |row| (*row).into(),
        |left, right| {
            (left.ceiling, left.prior_rank, left.alias_count)
                == (right.ceiling, right.prior_rank, right.alias_count)
        },
    )?;

    let seeds = [
        grade_seed(seeds[0].seed, &indexed[0], &indexed[1])?,
        grade_seed(seeds[1].seed, &indexed[2], &indexed[3])?,
    ];
    let advance = (epoch == 8).then(|| {
        seeds.iter().all(|seed| {
            seed.common_attainable.meets_endpoints && seed.common_attainable.meets_retention
        })
    });
    let next_action = advance.map(|advance| next_action(&seeds, advance));
    Ok(EpochGrade {
        epoch,
        seeds,
        advance,
        next_action,
    })
}

fn grade_confirmation_seed(
    seed: u32,
    geometry: &BTreeMap<u32, &ConfirmationRow>,
    ordered: &BTreeMap<u32, &ConfirmationRow>,
) -> Result<PairedGrade, String> {
    let mut groups = BTreeMap::<u64, PairedCounts>::new();
    for (&ordinal, &geometry) in geometry {
        let ordered = ordered[&ordinal];
        for (name, row) in [("geometry", geometry), ("ordered", ordered)] {
            valid_endpoint(row)
                .map_err(|error| format!("seed {seed} {name} row {ordinal}: {error}"))?;
        }
        if geometry.ceiling.strict() && ordered.ceiling.strict() {
            groups.entry(geometry.group).or_default().add(pair_updates(
                geometry.checkpoint,
                ordered.checkpoint,
                geometry.zero,
            ));
        }
    }
    Ok(paired(groups))
}

/// Grades the separately frozen internal split supplied by the caller.
pub fn grade_confirmation(
    seeds: &[ConfirmationSeedInput<'_>; 2],
) -> Result<ConfirmationGrade, String> {
    let seed_ids = [seeds[0].seed, seeds[1].seed];
    validate_seeds(seed_ids)?;
    let indexed = [
        index(seeds[0].geometry, |row| row.ordinal)?,
        index(seeds[0].ordered, |row| row.ordinal)?,
        index(seeds[1].geometry, |row| row.ordinal)?,
        index(seeds[1].ordered, |row| row.ordinal)?,
    ];
    verify_domain(
        &indexed,
        |row| *row,
        |left, right| left.ceiling == right.ceiling,
    )?;
    let common_attainable = [
        grade_confirmation_seed(seed_ids[0], &indexed[0], &indexed[1])?,
        grade_confirmation_seed(seed_ids[1], &indexed[2], &indexed[3])?,
    ];
    let advance = common_attainable.iter().all(|seed| seed.meets_endpoints);
    Ok(ConfirmationGrade {
        seeds: seed_ids,
        common_attainable,
        advance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn election(margin_q48: i64, rival: u32) -> Election {
        let elects_gold = margin_q48 > 0 || (margin_q48 == 0 && rival > 2);
        Election {
            margin_q48,
            rival,
            rank: if elects_gold { 1 } else { 2 },
        }
    }

    fn row(ordinal: u32, group: u64, immediate: bool, checkpoint: bool) -> RowImpact {
        RowImpact {
            ordinal,
            group,
            gold_id: 2,
            prior_rank: 2,
            alias_count: 0,
            ceiling: election(10, 3),
            zero: election(-2, 3),
            before: election(-1, 3),
            before_rival_prior_q48: 100,
            old_rival_margin_q48: if immediate { 2 } else { -1 },
            after: election(if immediate { 2 } else { -1 }, 3),
            clipped_words: 0,
            checkpoint: election(if checkpoint { 1 } else { -2 }, 3),
        }
    }

    fn seed<'a>(id: u32, geometry: &'a [RowImpact], ordered: &'a [RowImpact]) -> SeedInput<'a> {
        SeedInput {
            seed: id,
            geometry,
            ordered,
        }
    }

    fn input<'a>(
        g0: &'a [RowImpact],
        o0: &'a [RowImpact],
        g1: &'a [RowImpact],
        o1: &'a [RowImpact],
    ) -> [SeedInput<'a>; 2] {
        [seed(SEEDS[0], g0, o0), seed(SEEDS[1], g1, o1)]
    }

    #[test]
    fn gate_passes_at_exact_five_point_gain_and_eighty_percent_retention() {
        let mut geometry: Vec<_> = (0..20)
            .map(|i| row(i, u64::from(i / 2), i < 5, i < 4))
            .collect();
        let mut ordered: Vec<_> = (0..20)
            .map(|i| row(i, u64::from(i / 2), i < 3, i < 3))
            .collect();
        for row in geometry
            .iter_mut()
            .take(3)
            .chain(ordered.iter_mut().take(3))
        {
            row.zero = election(1, 3);
            row.prior_rank = 1;
        }
        let mut reordered = geometry.clone();
        reordered.reverse();
        let early = grade_epoch(7, &input(&geometry, &ordered, &reordered, &ordered)).unwrap();
        assert_eq!(early.advance, None);
        assert_eq!(early.next_action, None);
        let last = grade_epoch(8, &input(&geometry, &ordered, &reordered, &ordered)).unwrap();
        assert_eq!(last.advance, Some(true));
        assert_eq!(
            last.next_action,
            Some(NextAction::GroupDisjointTrainConfirmation)
        );
        assert_eq!(
            last.seeds[0]
                .common_attainable
                .geometry_minus_ordered
                .observed_ppm,
            Some(50_000)
        );
        assert_eq!(
            last.seeds[0]
                .common_attainable
                .geometry_retention
                .observed_ppm,
            Some(800_000)
        );
        assert_eq!(
            last.seeds[0].common_attainable,
            last.seeds[1].common_attainable
        );

        geometry[3].checkpoint = election(-1, 3);
        let failed = grade_epoch(8, &input(&geometry, &ordered, &reordered, &ordered)).unwrap();
        assert!(!failed.seeds[0].common_attainable.meets_endpoints);
        assert!(!failed.seeds[0].common_attainable.meets_retention);
        assert_eq!(failed.advance, Some(false));
    }

    #[test]
    fn replacement_routes_both_seeds_and_other_failures_stay_visible() {
        let mut geometry: Vec<_> = (0..8).map(|i| row(i, u64::from(i), false, false)).collect();
        for row in geometry.iter_mut().take(3) {
            row.old_rival_margin_q48 = 2;
            row.after = election(0, 4);
        }
        geometry[0].checkpoint = election(1, 4);
        geometry[3].ceiling = election(-2, 3);
        geometry[3].before = election(-2, 3);
        geometry[3].after = election(-2, 3);
        geometry[3].old_rival_margin_q48 = -2;
        geometry[3].alias_count = 1;
        geometry[4].clipped_words = 1;
        geometry[5] = row(5, 5, true, false);
        geometry[6] = row(6, 6, true, true);
        let ordered: Vec<_> = (0..8).map(|i| row(i, u64::from(i), false, true)).collect();
        let last = grade_epoch(8, &input(&geometry, &ordered, &geometry, &ordered)).unwrap();
        assert_eq!(last.advance, Some(false));
        assert_eq!(
            last.next_action,
            Some(NextAction::TestFullElectionCorrection)
        );
        let arm = &last.seeds[0].geometry;
        assert_eq!(arm.all.immediate.ties_won, 3);
        assert_eq!(arm.all.immediate.strict, 2);
        assert_eq!(arm.all.retained, 1);
        assert_eq!(arm.all.later_gains, 1);
        assert_eq!(
            arm.all.first_losses,
            LossCounts {
                structural: 1,
                clipped_old_rival: 1,
                old_rival: 1,
                replacement: 3,
                epoch_end: 1,
            }
        );
        assert_eq!(arm.by_alias["any"].first_losses.structural, 1);
        assert_eq!(arm.by_clipping["any"].first_losses.clipped_old_rival, 1);
        assert_eq!(arm.common_attainable.rows, 7);
    }

    #[test]
    fn seed_disagreement_and_matched_ordered_win_produce_distinct_actions() {
        let immediate = [row(0, 0, true, false)];
        let missed = [row(0, 0, false, false)];
        let perfect = [row(0, 0, true, true)];
        let split = grade_epoch(8, &input(&immediate, &missed, &missed, &missed)).unwrap();
        assert_eq!(split.advance, Some(false));
        assert_eq!(split.next_action, Some(NextAction::InspectSeedTrajectories));
        let tied = grade_epoch(8, &input(&perfect, &perfect, &perfect, &perfect)).unwrap();
        assert_eq!(tied.advance, Some(false));
        assert_eq!(tied.next_action, Some(NextAction::NoMatchedDegreeBenefit));
    }

    #[test]
    fn missing_duplicate_or_shifted_rows_and_corrupt_visits_error() {
        let base = [row(0, 88, false, false)];
        let mut shifted = base;
        shifted[0].ordinal = 1;
        assert!(grade_epoch(8, &input(&base, &shifted, &base, &base)).is_err());
        assert!(grade_epoch(8, &input(&base, &[], &base, &base)).is_err());
        let duplicate = [base[0]; 2];
        assert!(grade_epoch(8, &input(&base, &duplicate, &base, &base)).is_err());
        shifted = base;
        shifted[0].group = 99;
        assert!(grade_epoch(8, &input(&base, &shifted, &base, &base)).is_err());
        shifted = base;
        shifted[0].zero = election(-3, 3);
        assert!(grade_epoch(8, &input(&base, &shifted, &base, &base)).is_err());
        shifted = base;
        shifted[0].ceiling = election(11, 3);
        assert!(grade_epoch(8, &input(&base, &base, &shifted, &base)).is_err());
        shifted = base;
        shifted[0].after = election(0, 1);
        assert!(grade_epoch(8, &input(&shifted, &base, &shifted, &base)).is_err());
    }

    #[test]
    fn cluster_resampling_keeps_unequal_group_sizes_and_reports_undefined_retention() {
        let geometry = [
            row(0, 11, true, true),
            row(1, 11, true, true),
            row(2, 11, true, true),
            row(3, 99, false, false),
        ];
        let ordered = geometry.map(|item| row(item.ordinal, item.group, false, false));
        let grade = grade_epoch(8, &input(&geometry, &ordered, &geometry, &ordered)).unwrap();
        let paired = grade.seeds[0].common_attainable;
        assert_eq!((paired.rows, paired.groups), (4, 2));
        assert_eq!(
            paired.geometry_minus_ordered,
            Band {
                observed_ppm: Some(750_000),
                lower_ppm: Some(0),
                upper_ppm: Some(1_000_000),
                defined_resamples: RESAMPLES,
            }
        );
        // Selecting only cluster 99 twice leaves this retention draw undefined.
        assert!(paired.geometry_retention.defined_resamples < RESAMPLES);
        assert_eq!(paired.geometry_retention.lower_ppm, Some(1_000_000));
        assert_eq!(paired.ordered_retention, Band::default());
        assert_eq!(paired, grade.seeds[1].common_attainable);
    }

    #[test]
    fn strict_common_domain_excludes_tie_winners_and_blocks_empty_gate() {
        let mut geometry = [row(0, 88, false, false)];
        geometry[0].ceiling = election(0, 3);
        geometry[0].alias_count = 1;
        let ordered = [row(0, 88, true, true)];
        let grade = grade_epoch(8, &input(&geometry, &ordered, &geometry, &ordered)).unwrap();
        assert_eq!(grade.advance, Some(false));
        assert_eq!(
            grade.next_action,
            Some(NextAction::CheckCommonAttainableDomain)
        );
        assert_eq!(grade.seeds[0].geometry.all.ceiling.ties_won, 1);
        assert_eq!(grade.seeds[0].common_attainable.rows, 0);
        assert_eq!(
            grade.seeds[0].common_attainable.geometry_minus_ordered,
            Band::default()
        );
    }

    #[test]
    fn group_confirmation_checks_both_seeds_without_a_retention_denominator() {
        let geometry = [row(26, 88, false, true).into()];
        let ordered = [row(26, 88, false, false).into()];
        let seeds = [
            ConfirmationSeedInput {
                seed: SEEDS[0],
                geometry: &geometry,
                ordered: &ordered,
            },
            ConfirmationSeedInput {
                seed: SEEDS[1],
                geometry: &geometry,
                ordered: &ordered,
            },
        ];
        let grade = grade_confirmation(&seeds).unwrap();
        assert!(grade.advance);
        assert!(!grade.common_attainable[0].meets_retention);
        let failed = [
            ConfirmationSeedInput {
                seed: SEEDS[0],
                geometry: &geometry,
                ordered: &ordered,
            },
            ConfirmationSeedInput {
                seed: SEEDS[1],
                geometry: &ordered,
                ordered: &ordered,
            },
        ];
        assert!(!grade_confirmation(&failed).unwrap().advance);
    }
}
