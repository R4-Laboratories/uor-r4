//! Independently verify the sealed visit and checkpoint streams on Wick.
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SEEDS: [u32; 2] = [435300, 435301];
const TRAIN_ROWS: usize = 3891;
const COMMON_ROWS: usize = 3886;
const COMMON_GROUPS: usize = 3703;
const HELD_ROWS: usize = 787;
const PRIOR_VISITS_SHA: &str = "1fec709219e9d7f4d80d38359c79bea616354f2847feb42ad121e2df53fee537";
const PRIOR_WORDS_SHA: &str = "233162fe0834a8641aec6e6e65cfdfdfd73ffef44ed8645038c6793ec4c4cd71";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Policy {
    Full,
    GoldOnly,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::GoldOnly => "gold-only",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::GoldOnly => "gold_only",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "kebab-case")]
enum Addressing {
    Geometry,
    OrderedMatched,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
struct Fit {
    seed: u32,
    epoch: u32,
    policy: Policy,
}

impl Fit {
    fn check(self) -> Result<()> {
        require(
            SEEDS.contains(&self.seed) && (1..=8).contains(&self.epoch),
            "unexpected seed or epoch",
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
struct Decision {
    margin_q48: i64,
    rival_prior_q48: u64,
    rival: u32,
    rank: u32,
}

impl Decision {
    fn strict(self) -> bool {
        self.margin_q48 > 0
    }

    fn valid(self, gold: u32) -> bool {
        (1..=4095).contains(&self.rival)
            && (1..=4095).contains(&self.rank)
            && self.rival != gold
            && (self.rank == 1) == (self.strict() || (self.margin_q48 == 0 && self.rival > gold))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
struct Learning {
    before: Decision,
    after: Decision,
    radius: u32,
    changed_words: u32,
    l1: u32,
    saturated_words: u32,
    outcome: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
struct Electorate {
    group: u64,
    gold: u32,
    common: bool,
    old_geometry: Decision,
    zero: Decision,
    ceiling: Decision,
    checkpoint: Decision,
}

#[derive(Debug, Deserialize)]
struct PriorVisit {
    #[serde(flatten)]
    fit: Fit,
    addressing: Addressing,
    ordinal: u32,
    #[serde(flatten)]
    election: Electorate,
    visit: Learning,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Gauge {
    score_q48: i64,
    offsets: [i32; 4],
    changed_words: u32,
    l1: u32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct CenterVisit {
    learning: Learning,
    before_center: Decision,
    gauge: Gauge,
    gold_before_q48: i64,
    gold_centered_q48: i64,
    clipped_learning_words: u32,
    shift_failure: u32,
}

#[derive(Debug, Deserialize)]
struct Visit {
    #[serde(flatten)]
    fit: Fit,
    addressing: Addressing,
    centered: bool,
    ordinal: u32,
    #[serde(flatten)]
    election: Electorate,
    visit: CenterVisit,
    visit_index: usize,
    age_visits: usize,
    age_learning_updates: usize,
    visits_since_prior: Option<usize>,
}

fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.to_owned().into())
    }
}

fn stream<T: DeserializeOwned>(
    path: &Path,
    mut observe: impl FnMut(T) -> Result<()>,
) -> Result<usize> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut rows = 0;
    while reader.read_line(&mut line)? != 0 {
        rows += 1;
        let item = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{rows}: {error}", path.display()))?;
        observe(item).map_err(|error| format!("{}:{rows}: {error}", path.display()))?;
        line.clear();
    }
    Ok(rows)
}

fn arm_index(centered: bool, addressing: Addressing) -> usize {
    usize::from(centered) * 2 + usize::from(addressing == Addressing::OrderedMatched)
}

fn validate_visit(row: &Visit) -> Result<()> {
    let visit = row.visit;
    let learning = visit.learning;
    let election = row.election;
    require(
        (1..=4095).contains(&election.gold)
            && [
                learning.before,
                learning.after,
                visit.before_center,
                election.old_geometry,
                election.zero,
                election.ceiling,
                election.checkpoint,
            ]
            .into_iter()
            .all(|decision| decision.valid(election.gold)),
        "invalid original-ID election",
    )?;
    let gauge = visit.gauge;
    let offset_sum: i64 = gauge.offsets.into_iter().map(i64::from).sum();
    let absolute_offset_sum: u64 = gauge
        .offsets
        .into_iter()
        .map(|offset| u64::from(offset.unsigned_abs()))
        .sum();
    let shifted_banks = gauge.offsets.into_iter().filter(|&n| n != 0).count();
    require(
        (row.centered || shifted_banks == 0)
            && gauge
                .offsets
                .into_iter()
                .all(|n| (-32768..=32767).contains(&n))
            && gauge.score_q48 == offset_sum * (1_i64 << 32)
            && u64::from(gauge.l1) == 26 * absolute_offset_sum
            && gauge.changed_words as usize == 26 * shifted_banks
            && visit.gold_before_q48.checked_add(gauge.score_q48) == Some(visit.gold_centered_q48)
            && visit.shift_failure == 0
            && visit.before_center == learning.before,
        "invalid gauge movement or gauge changed the election",
    )?;
    require(
        learning.before.margin_q48 <= learning.after.margin_q48
            && learning.before.rank >= learning.after.rank
            && learning.after.margin_q48 <= election.ceiling.margin_q48
            && election.checkpoint.margin_q48 <= election.ceiling.margin_q48
            && (row.fit.policy != Policy::Full
                || learning.after.strict() == election.ceiling.strict()),
        "learning regressed or disagreed with the structural ceiling",
    )?;
    let cap = match row.fit.policy {
        Policy::Full => 104,
        Policy::GoldOnly => 4,
    };
    let valid_update = !learning.before.strict()
        && learning.after.strict()
        && (1..=65535).contains(&learning.radius)
        && (1..=cap).contains(&learning.changed_words)
        && u64::from(learning.l1) >= u64::from(learning.changed_words)
        && u64::from(learning.l1) <= u64::from(learning.radius) * u64::from(learning.changed_words)
        && learning.saturated_words <= cap
        && visit.clipped_learning_words <= learning.saturated_words;
    let no_update = learning.before == learning.after
        && learning.radius == 0
        && learning.changed_words == 0
        && learning.l1 == 0
        && learning.saturated_words == 0
        && visit.clipped_learning_words == 0;
    require(
        match learning.outcome {
            0 => learning.before.strict() && no_update,
            1 => valid_update,
            2 => !learning.before.strict() && no_update,
            _ => false,
        },
        "invalid learning outcome or write counters",
    )?;
    require(
        row.visit_index.checked_add(row.age_visits) == Some(TRAIN_ROWS - 1),
        "inconsistent visit age",
    )
}

#[derive(Clone, Copy, Debug)]
struct CompactVisit {
    election: Electorate,
    immediate: bool,
    newly_corrected: bool,
}

#[derive(Default)]
struct Epoch {
    rows: BTreeMap<u32, [Option<CompactVisit>; 4]>,
    order: [BTreeMap<usize, (bool, usize)>; 4],
    movement: [Movement; 4],
}

impl Epoch {
    fn observe(&mut self, row: &Visit) -> Result<()> {
        let arm = arm_index(row.centered, row.addressing);
        let slot = &mut self.rows.entry(row.ordinal).or_default()[arm];
        let learning = row.visit.learning;
        require(
            slot.is_none()
                && self.order[arm]
                    .insert(
                        row.visit_index,
                        (learning.outcome == 1, row.age_learning_updates),
                    )
                    .is_none(),
            "duplicate row identity or visit index",
        )?;
        *slot = Some(CompactVisit {
            election: row.election,
            immediate: learning.after.strict(),
            newly_corrected: learning.outcome == 1,
        });
        self.movement[arm].observe(row);
        Ok(())
    }

    fn grade(self) -> Result<Value> {
        require(self.rows.len() == TRAIN_ROWS, "incomplete fit population")?;
        for order in self.order {
            let mut seen = 0;
            for (updated, age) in order.into_values().rev() {
                require(
                    seen == age,
                    "learning-update age disagrees with decoded order",
                )?;
                seen += usize::from(updated);
            }
        }
        let mut groups = GroupCounts::default();
        for arms in self.rows.into_values() {
            let [Some(off_geometry), Some(off_ordered), Some(on_geometry), Some(on_ordered)] = arms
            else {
                return Err("missing fit arm".into());
            };
            let g = on_geometry.election;
            require(
                [off_geometry, off_ordered, on_ordered]
                    .into_iter()
                    .all(|arm| {
                        let e = arm.election;
                        (e.group, e.gold, e.common, e.old_geometry, e.zero)
                            == (g.group, g.gold, g.common, g.old_geometry, g.zero)
                    })
                    && off_geometry.election.ceiling == g.ceiling
                    && off_ordered.election.ceiling == on_ordered.election.ceiling
                    && g.common == (g.ceiling.strict() && on_ordered.election.ceiling.strict()),
                "paired electorate or common-domain drift",
            )?;
            if g.common {
                groups.observe(
                    g.group,
                    [
                        g.checkpoint.strict(),
                        g.old_geometry.strict(),
                        g.zero.strict(),
                        off_geometry.election.checkpoint.strict(),
                        on_ordered.election.checkpoint.strict(),
                    ],
                    on_geometry.immediate,
                    on_geometry.newly_corrected,
                );
            }
        }
        let grade = groups.grade(true);
        require(
            grade["rows"] == COMMON_ROWS && grade["groups"] == COMMON_GROUPS,
            "common population disagrees with the registered cohort",
        )?;
        Ok(grade)
    }
}

#[derive(Default)]
struct GroupCounts(BTreeMap<u64, [i64; 10]>);

impl GroupCounts {
    fn observe(&mut self, group: u64, elections: [bool; 5], immediate: bool, newly: bool) {
        let counters = self.0.entry(group).or_default();
        counters[0] += 1;
        for (counter, won) in counters[1..=5].iter_mut().zip(elections) {
            *counter += i64::from(won);
        }
        counters[6] += i64::from(immediate);
        counters[7] += i64::from(immediate && elections[0]);
        counters[8] += i64::from(newly);
        counters[9] += i64::from(newly && elections[0]);
    }

    fn grade(self, fit: bool) -> Value {
        let groups: Vec<_> = self.0.into_values().collect();
        let sum = groups
            .iter()
            .fold([0; 10], |a, b| std::array::from_fn(|i| a[i] + b[i]));
        let [rows, centered, old, zero, off, ordered, immediate, retained, newly, newly_retained] =
            sum;
        let endpoints = rows != 0
            && [old, zero, off, ordered]
                .into_iter()
                .all(|n| centered >= n + (rows + 19) / 20);
        let retention = immediate != 0 && retained >= (immediate * 4 + 4) / 5;
        let bands = intervals(&groups, sum);
        json!({
            "rows": rows, "groups": groups.len(), "resamples": if rows == 0 { 0 } else { 4096 },
            "centered_strict": centered, "historical_strict": old, "frozen_strict": zero,
            "uncentered_strict": off, "ordered_strict": ordered,
            "centered_immediate": immediate, "centered_retained": retained,
            "centered_newly_corrected": newly, "centered_newly_retained": newly_retained,
            "meets_endpoints": endpoints, "meets_retention": retention,
            "pass": endpoints && (!fit || retention),
            "centered_minus_historical": bands[0], "centered_minus_frozen": bands[1],
            "centered_minus_uncentered": bands[2], "centered_minus_ordered": bands[3],
            "centered_retention": bands[4], "centered_newly_retention": bands[5],
        })
    }
}

fn rates(
    [rows, center, old, zero, off, ordered, immediate, retained, newly, newly_retained]: [i64; 10],
) -> [Option<i64>; 6] {
    [
        (center - old, rows),
        (center - zero, rows),
        (center - off, rows),
        (center - ordered, rows),
        (retained, immediate),
        (newly_retained, newly),
    ]
    .map(|(n, d)| {
        (d > 0).then(|| {
            let signed_millionths = n * 1_000_000;
            (signed_millionths + n.signum() * (d / 2)) / d
        })
    })
}

fn intervals(groups: &[[i64; 10]], total: [i64; 10]) -> [Value; 6] {
    let mut rng = 0x606f_35d3_8e69_eb4d_u64;
    let draws = std::iter::from_fn(move || {
        rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let x = (rng ^ (rng >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let y = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Some(y ^ (y >> 31))
    });
    let bound = groups.len() as u64;
    let mut samples: [Vec<i64>; 6] = std::array::from_fn(|_| Vec::new());
    if bound > 0 {
        let cutoff = ((1_u128 << 64) % u128::from(bound)) as u64;
        let mut selected = draws
            .filter(|&x| x >= cutoff)
            .map(|x| groups[(x % bound) as usize]);
        for _ in 0..4096 {
            let mut resampled = [0; 10];
            for group in selected.by_ref().take(groups.len()) {
                for (count, add) in resampled.iter_mut().zip(group) {
                    *count += add;
                }
            }
            for (values, rate) in samples.iter_mut().zip(rates(resampled)) {
                values.extend(rate);
            }
        }
    }
    let observed = rates(total);
    std::array::from_fn(|i| {
        let sample = &mut samples[i];
        sample.sort_unstable();
        let interval = sample
            .len()
            .checked_sub(1)
            .map(|last| (sample[last / 40], sample[last * 39 / 40]));
        json!({
            "observed_ppm": observed[i],
            "lower_ppm": interval.map(|pair| pair.0),
            "upper_ppm": interval.map(|pair| pair.1),
            "defined_resamples": sample.len(),
        })
    })
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

#[derive(Default)]
struct Movement {
    totals: BTreeMap<&'static str, u64>,
    quartiles: [[u64; 5]; 4],
}

impl Movement {
    fn observe(&mut self, row: &Visit) {
        let v = row.visit;
        let l = v.learning;
        let initially_won = l.before.strict();
        let won = l.after.strict();
        let end = row.election.checkpoint;
        let landed = end.strict();
        let correction = l.outcome == 1;
        for (name, value) in [
            ("rows", 1),
            ("strict_before", u64::from(initially_won)),
            ("immediate", u64::from(won)),
            ("updated", u64::from(correction)),
            ("infeasible", u64::from(l.outcome == 2)),
            ("retained", u64::from(won && landed)),
            ("erased", u64::from(won && !landed)),
            ("later_gains", u64::from(!won && landed)),
            ("newly_corrected", u64::from(correction)),
            ("newly_corrected_retained", u64::from(correction && landed)),
            ("checkpoint_strict", u64::from(landed)),
            ("ties_won", u64::from(end.margin_q48 == 0 && end.rank == 1)),
            ("ties_lost", u64::from(end.margin_q48 == 0 && end.rank != 1)),
            ("learning_changed_words", u64::from(l.changed_words)),
            ("learning_l1", u64::from(l.l1)),
            (
                "learning_directional_rail_hits",
                u64::from(l.saturated_words),
            ),
            (
                "learning_clipped_words",
                u64::from(v.clipped_learning_words),
            ),
            ("gauge_visits", u64::from(v.gauge.changed_words != 0)),
            ("gauge_changed_words", u64::from(v.gauge.changed_words)),
            ("gauge_l1", u64::from(v.gauge.l1)),
        ] {
            *self.totals.entry(name).or_default() += value;
        }
        let radius = self.totals.entry("max_radius").or_default();
        *radius = (*radius).max(u64::from(l.radius));
        let quartile = &mut self.quartiles[row.age_visits * 4 / TRAIN_ROWS];
        for (count, condition) in quartile.iter_mut().zip([
            won,
            won && landed,
            won && !landed,
            correction,
            correction && landed,
        ]) {
            *count += u64::from(condition);
        }
    }

    fn receipt(&self, words: &[i16], masks: [u64; 4]) -> Value {
        let rails = |rail| -> [usize; 4] {
            std::array::from_fn(|bank| {
                (0..64)
                    .filter(|&slot| {
                        masks[bank] & (1 << slot) != 0 && words[64 * bank + slot] == rail
                    })
                    .count()
            })
        };
        let mut result = json!(self.totals);
        result["endpoint_low_rail"] = json!(rails(i16::MIN));
        result["endpoint_high_rail"] = json!(rails(i16::MAX));
        result["age_quartiles"] = json!(self.quartiles.map(
            |[immediate, retained, erased, newly_corrected, newly_corrected_retained]| json!({
                "immediate":immediate, "retained":retained, "erased":erased,
                "newly_corrected":newly_corrected,
                "newly_corrected_retained":newly_corrected_retained,
            })
        ));
        result
    }
}

#[derive(Deserialize)]
struct WordRecord {
    #[serde(flatten)]
    fit: Fit,
    addressing: Addressing,
    #[serde(default)]
    centered: bool,
    words: Vec<i16>,
}

#[derive(Deserialize)]
struct Support {
    geometry: [u64; 4],
    ordered: [u64; 4],
}

type Words = BTreeMap<(Fit, usize), Vec<i16>>;
type PriorVisits = BTreeMap<(Fit, u32, Addressing), (Electorate, Learning)>;

fn read_words(path: &Path, masks: [u64; 4]) -> Result<Words> {
    let mut out = Words::new();
    stream(path, |row: WordRecord| {
        row.fit.check()?;
        require(row.words.len() == 256, "invalid word count")?;
        require(
            row.words
                .iter()
                .enumerate()
                .all(|(i, &word)| word == 0 || masks[i / 64] & (1 << (i % 64)) != 0),
            "checkpoint wrote a structurally unreachable slot",
        )?;
        require(
            out.insert(
                (row.fit, arm_index(row.centered, row.addressing)),
                row.words,
            )
            .is_none(),
            "duplicate checkpoint",
        )
    })?;
    Ok(out)
}

fn file_sha(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    loop {
        let buffer = file.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        digest.update(buffer);
        let len = buffer.len();
        file.consume(len);
    }
    Ok(hex::encode(digest.finalize()))
}

#[derive(Deserialize)]
struct SparseElection {
    before: Decision,
    centered: Decision,
    before_ceiling: Decision,
    centered_ceiling: Decision,
    shift_failures: u32,
    tied_aliases: u32,
    blocking_aliases: u32,
}

#[derive(Deserialize)]
struct FrozenSparseRow {
    seed: u32,
    addressing: Addressing,
    ordinal: u32,
    group: u64,
    gold: u32,
    common: bool,
    attainable: Decision,
    classification: String,
    precheck: SparseElection,
}

fn verify_frozen_sparse(
    path: &Path,
    old_words: &Words,
    old_visits: &PriorVisits,
    masks: [u64; 4],
) -> Result<Value> {
    let mut keys = BTreeSet::new();
    let mut totals = BTreeMap::<_, [BTreeMap<&str, u64>; 2]>::new();
    stream(&path.join("frozen-sparse.jsonl"), |row: FrozenSparseRow| {
        require(
            keys.insert((row.seed, row.addressing, row.ordinal)),
            "duplicate frozen sparse row",
        )?;
        let key = (
            Fit {
                seed: row.seed,
                epoch: 8,
                policy: Policy::GoldOnly,
            },
            row.ordinal,
            row.addressing,
        );
        let (old, _) = old_visits
            .get(&key)
            .ok_or("missing prior sparse checkpoint row")?;
        let event = row.precheck;
        require(
            (row.group, row.gold, row.common, row.attainable)
                == (old.group, old.gold, old.common, old.ceiling)
                && event.before == old.checkpoint
                && event.before == event.centered
                && event.shift_failures == 0
                && [event.before_ceiling, event.centered_ceiling]
                    .into_iter()
                    .all(|ceiling| {
                        ceiling.valid(row.gold)
                            && event.before.margin_q48 <= ceiling.margin_q48
                            && ceiling.margin_q48 <= row.attainable.margin_q48
                            && (event.blocking_aliases == 0 || !ceiling.strict())
                    }),
            "frozen sparse decisions or cohort disagree with the archived checkpoint",
        )?;
        let class = match (
            event.before.strict(),
            event.before_ceiling.strict(),
            event.centered_ceiling.strict(),
        ) {
            (true, _, _) => "already_winning",
            (false, true, true) => "still_feasible",
            (false, false, true) => "rescued",
            (false, true, false) => "newly_blocked",
            (false, false, false) => "still_blocked",
        };
        require(
            row.classification == class,
            "frozen sparse classification differs",
        )?;
        for m in totals
            .entry((row.seed, row.addressing))
            .or_default()
            .iter_mut()
            .take(1 + usize::from(row.common))
        {
            for (field, n) in [
                ("rows", 1),
                ("common_rows", u64::from(row.common)),
                ("already_winning", u64::from(class == "already_winning")),
                ("still_feasible", u64::from(class == "still_feasible")),
                ("rescued", u64::from(class == "rescued")),
                ("newly_blocked", u64::from(class == "newly_blocked")),
                ("still_blocked", u64::from(class == "still_blocked")),
                ("rows_with_tied_alias", u64::from(event.tied_aliases != 0)),
                ("tied_aliases", u64::from(event.tied_aliases)),
                (
                    "rows_with_blocking_alias",
                    u64::from(event.blocking_aliases != 0),
                ),
                ("blocking_aliases", u64::from(event.blocking_aliases)),
                (
                    "higher_ceiling",
                    u64::from(event.centered_ceiling.margin_q48 > event.before_ceiling.margin_q48),
                ),
                (
                    "lower_ceiling",
                    u64::from(event.centered_ceiling.margin_q48 < event.before_ceiling.margin_q48),
                ),
            ] {
                *m.entry(field).or_default() += n;
            }
        }
        Ok(())
    })?;
    require(
        keys.len() == 4 * TRAIN_ROWS && totals.len() == 4,
        "incomplete frozen sparse cohort",
    )?;
    let reported = read_json(&path.join("frozen-sparse-summary.json"))?;
    let mut summaries = Vec::new();
    for ((seed, address), [all, common]) in totals {
        let arm = arm_index(false, address);
        let words = old_words
            .get(&(
                Fit {
                    seed,
                    epoch: 8,
                    policy: Policy::GoldOnly,
                },
                arm,
            ))
            .ok_or("missing prior sparse words")?;
        let offsets = std::array::from_fn(|bank| {
            let mut low = i32::from(i16::MAX);
            let mut high = i32::from(i16::MIN);
            for i in 0..64 {
                if masks[bank] & (1 << i) != 0 {
                    low = low.min(i32::from(words[64 * bank + i]));
                    high = high.max(i32::from(words[64 * bank + i]));
                }
            }
            let room_left = low - i32::from(i16::MIN);
            let room_right = i32::from(i16::MAX) - high;
            (room_right - room_left) / 2
        });
        let gauge = Gauge {
            offsets,
            score_q48: (1_i64 << 32) * offsets.into_iter().map(i64::from).sum::<i64>(),
            changed_words: 26 * offsets.into_iter().filter(|&n| n != 0).count() as u32,
            l1: 26 * offsets.into_iter().map(i32::unsigned_abs).sum::<u32>(),
        };
        let address = if arm == 0 {
            "geometry"
        } else {
            "ordered-matched"
        };
        let summary =
            json!({"seed":seed,"addressing":address,"gauge":gauge,"all":all,"common":common});
        require(
            all["rows"] == TRAIN_ROWS as u64
                && common["rows"] == COMMON_ROWS as u64
                && reported
                    .as_array()
                    .is_some_and(|rows| rows.len() == 4 && rows.contains(&summary)),
            "frozen sparse counts or bank offset disagree with report",
        )?;
        summaries.push(summary);
    }
    Ok(json!(summaries))
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
struct Held {
    seed: u32,
    ordinal: u32,
    group: u64,
    gold: u32,
    common: bool,
    zero: Decision,
    old_geometry: Decision,
    uncentered_geometry: Decision,
    centered_geometry: Decision,
    centered_ordered: Decision,
}

fn withheld(group: u64) -> bool {
    let mut hash = Sha256::new();
    hash.update(b"uor-r4/lexical-margin/group-v1\0");
    hash.update(group.to_le_bytes());
    let digest = hash.finalize();
    let head = std::array::from_fn(|i| digest[i]);
    u64::from_le_bytes(head) % 5 == 0
}

fn verify_held(path: &Path, population: &BTreeMap<u32, (u64, u32, bool)>) -> Result<Value> {
    let mut rows = BTreeMap::<(u32, u32), Held>::new();
    stream(&path.join("endpoints.jsonl"), |row: Held| {
        require(
            SEEDS.contains(&row.seed)
                && population
                    .get(&row.ordinal)
                    .is_some_and(|&(group, gold, _)| {
                        row.group == group && row.gold == gold && withheld(group)
                    })
                && [
                    row.zero,
                    row.old_geometry,
                    row.uncentered_geometry,
                    row.centered_geometry,
                    row.centered_ordered,
                ]
                .into_iter()
                .all(|decision| decision.valid(row.gold)),
            "invalid held endpoint or held-group membership",
        )?;
        require(
            rows.insert((row.seed, row.ordinal), row).is_none(),
            "duplicate held ordinal",
        )
    })?;
    require(
        rows.len() == HELD_ROWS * SEEDS.len(),
        "incomplete held endpoints",
    )?;
    let reported = read_json(&path.join("confirmation.json"))?;
    require(
        reported["groups_disjoint"] == true
            && reported["held_rows"] == HELD_ROWS
            && reported["fit_rows"] == TRAIN_ROWS - HELD_ROWS,
        "held-fit partition summary disagrees",
    )?;
    let mut pass = true;
    let mut verified = Vec::new();
    for seed in SEEDS {
        let mut counts = GroupCounts::default();
        for (&ordinal, &(group, _, _)) in population {
            if !withheld(group) {
                continue;
            }
            let row = rows.get(&(seed, ordinal)).ok_or("missing held ordinal")?;
            let reference = rows
                .get(&(SEEDS[0], ordinal))
                .ok_or("missing reference held ordinal")?;
            require(
                row.common == reference.common && row.zero == reference.zero,
                "held common mask or zero decision differed by seed",
            )?;
            if row.common {
                counts.observe(
                    row.group,
                    [
                        row.centered_geometry.strict(),
                        row.old_geometry.strict(),
                        row.zero.strict(),
                        row.uncentered_geometry.strict(),
                        row.centered_ordered.strict(),
                    ],
                    false,
                    false,
                );
            }
        }
        let grade = counts.grade(false);
        require(
            reported["seeds"].as_array().is_some_and(|s| {
                s.len() == SEEDS.len()
                    && s.iter()
                        .any(|item| item["seed"] == seed && item["grade"] == grade)
            }),
            "independent held grade differs from report",
        )?;
        pass &= grade["pass"] == true;
        verified.push(json!({"seed":seed, "grade":grade}));
    }
    require(
        reported["pass"] == pass,
        "reported held two-seed gate differs",
    )?;
    Ok(json!({"pass":pass, "seeds":verified}))
}

#[derive(Deserialize)]
struct Certificate {
    ordinal: u32,
    group: u64,
    gold_id: u32,
    rival_id: u32,
    gold_prior: u64,
    rival_prior: u64,
    gold_slots: [u32; 4],
    rival_slots: [u32; 4],
    required_difference: i64,
}

#[derive(Deserialize)]
struct DeviceObservation {
    ordinal: u32,
    gold_id: u32,
    rival_id: u32,
    gold: CandidateObservation,
    rival: CandidateObservation,
}

#[derive(Deserialize)]
struct CandidateObservation {
    prior: u64,
    gold: u32,
    rival: u32,
}

fn unpack(packed: u32) -> [u32; 4] {
    std::array::from_fn(|bank| {
        let bank = bank as u32;
        64 * bank + ((packed >> (6 * bank)) & 63)
    })
}

fn certificate_vector(side: &Certificate, support: [u64; 4]) -> Result<[i8; 256]> {
    require(
        (1..=4095).contains(&side.gold_id)
            && (1..=4095).contains(&side.rival_id)
            && side.gold_id != side.rival_id
            && side.gold_prior <= 1 << 48
            && side.rival_prior <= 1 << 48,
        "certificate identity or Q48 prior out of range",
    )?;
    let valid_slot = |bank: u32, slot: u32| {
        slot / 64 == bank && support[bank as usize] & (1_u64 << (slot % 64)) != 0
    };
    let mut vector = [0; 256];
    for bank in 0..4 {
        let (gold, rival) = (side.gold_slots[bank], side.rival_slots[bank]);
        require(
            valid_slot(bank as u32, gold) && valid_slot(bank as u32, rival),
            "certificate references a slot outside the supported bank",
        )?;
        vector[gold as usize] += 1;
        vector[rival as usize] -= 1;
    }
    let target = i128::from(side.gold_prior);
    let rival = i128::from(side.rival_prior);
    let threshold = i128::from(side.required_difference);
    let step = 1_i128 << 32;
    require(
        vector.iter().any(|&k| k != 0)
            && target + step * (threshold - 1) <= rival
            && target + step * threshold > rival,
        "certificate bound is not the exact strict Q48 threshold",
    )?;
    Ok(vector)
}

fn verify_certificates(
    path: &Path,
    population: &BTreeMap<u32, (u64, u32, bool)>,
    masks: [u64; 4],
) -> Result<Value> {
    let observations_path = path.join("observations.jsonl");
    let mut observations = BTreeMap::new();
    stream(&observations_path, |obs: DeviceObservation| {
        require(
            observations
                .insert((obs.ordinal, obs.gold_id, obs.rival_id), obs)
                .is_none(),
            "duplicate raw certificate observation",
        )
    })?;
    let mut used = BTreeSet::new();
    let count = stream(
        &path.join("certificates.jsonl"),
        |pair: [Certificate; 2]| {
            let [left, right] = &pair;
            require(
                left.group != right.group,
                "certificate pair shares a source group",
            )?;
            for side in &pair {
                require(
                    used.insert(side.ordinal)
                        && population.get(&side.ordinal) == Some(&(side.group, side.gold_id, true)),
                    "certificate rows overlap or are outside the common TRAIN cohort",
                )?;
                let key = (side.ordinal, side.gold_id, side.rival_id);
                let sample = observations
                    .remove(&key)
                    .ok_or("missing raw certificate observation")?;
                require(
                    [
                        sample.gold.gold,
                        sample.gold.rival,
                        sample.rival.gold,
                        sample.rival.rival,
                    ]
                    .into_iter()
                    .all(|address| address < (1 << 24))
                        && sample.gold.gold == sample.gold.rival
                        && sample.gold.gold == sample.rival.gold
                        && sample.gold.prior == side.gold_prior
                        && sample.rival.prior == side.rival_prior
                        && unpack(sample.gold.gold) == side.gold_slots
                        && unpack(sample.rival.rival) == side.rival_slots,
                    "certificate differs from the captured GPU addresses or priors",
                )?;
            }
            let a = certificate_vector(left, masks)?;
            let b = certificate_vector(right, masks)?;
            require(
                a.into_iter().zip(b).all(|(l, r)| l + r == 0)
                    && i128::from(left.required_difference) + i128::from(right.required_difference)
                        > 0,
                "certificate inequalities can coexist or have different incidence vectors",
            )
        },
    )?;
    require(
        observations.is_empty(),
        "orphan raw certificate observation",
    )?;
    let upper_bound = COMMON_ROWS
        .checked_sub(count)
        .ok_or("too many certificates")?;
    let lower_bar = (COMMON_ROWS * 4).div_ceil(5);
    let summary = read_json(&path.join("summary.json"))?;
    require(
        summary["addressing"] == "geometry"
            && summary["rows"] == COMMON_ROWS
            && summary["original_rivals_per_row"] == 4094
            && summary["disjoint_certified_pairs"] == count
            && summary["shared_strict_win_upper_bound"] == upper_bound
            && summary["full_policy_retention_bar"] == lower_bar
            && summary["certificates_exclude_full_retention_bar"] == (upper_bound < lower_bar),
        "certificate-derived bound disagrees with experiment summary",
    )?;
    Ok(json!({
        "disjoint_pairs":count,
        "certified_rows":used.len(),
        "raw_gpu_observations_compared":used.len(),
        "shared_strict_win_upper_bound":upper_bound,
        "full_policy_retention_bar":lower_bar,
        "excludes_full_policy_retention_bar":upper_bound < lower_bar,
    }))
}

fn verify(prior: &Path, current: &Path) -> Result<Value> {
    let support: Support =
        serde_json::from_reader(BufReader::new(File::open(current.join("support.json"))?))?;
    require(
        support.geometry == support.ordered
            && support.geometry.iter().all(|mask| mask.count_ones() == 26),
        "invalid address support",
    )?;
    require(
        file_sha(&prior.join("checkpoints.jsonl"))? == PRIOR_WORDS_SHA
            && file_sha(&prior.join("visits.jsonl"))? == PRIOR_VISITS_SHA,
        "historical packet members fail their pinned hashes",
    )?;
    let old_words = read_words(&prior.join("checkpoints.jsonl"), support.geometry)?;
    let new_words = read_words(&current.join("checkpoints.jsonl"), support.geometry)?;
    require(
        old_words.len() == 64
            && new_words.len() == 128
            && old_words
                .iter()
                .all(|(key, old)| key.1 < 2 && new_words.get(key) == Some(old)),
        "uncentered replay failed for an epoch checkpoint",
    )?;
    let mut old_visits = PriorVisits::new();
    let old_count = stream(&prior.join("visits.jsonl"), |row: PriorVisit| {
        row.fit.check()?;
        require(
            old_visits
                .insert(
                    (row.fit, row.ordinal, row.addressing),
                    (row.election, row.visit),
                )
                .is_none(),
            "duplicate historical row",
        )
    })?;
    require(old_count == 64 * TRAIN_ROWS, "wrong historical visit count")?;
    let frozen_sparse = verify_frozen_sparse(current, &old_words, &old_visits, support.geometry)?;
    let mut population = BTreeMap::new();
    let mut reported = BTreeMap::<Fit, Value>::new();
    let epochs = read_json(&current.join("epochs.json"))?;
    for epoch in epochs.as_array().ok_or("expected epoch array")? {
        let fit = serde_json::from_value::<Fit>(epoch.clone())?;
        fit.check()?;
        require(
            reported.insert(fit, epoch.clone()).is_none(),
            "duplicate reported epoch",
        )?;
    }
    require(
        reported.len() == 32,
        "wrong number of reported epoch groups",
    )?;
    let mut summaries = Vec::new();
    let mut batch: Option<(Fit, Epoch)> = None;
    let mut shared_positions = BTreeMap::<(u32, u32, u32), usize>::new();
    let mut gates = BTreeMap::from([(Policy::Full, true), (Policy::GoldOnly, true)]);
    let mut finish = |fit: Fit, epoch: Epoch| -> Result<()> {
        let report = reported
            .remove(&fit)
            .ok_or("missing or noncontiguous epoch report")?;
        let mut arms = Vec::new();
        for (arm, observed) in epoch.movement.iter().enumerate() {
            let centered = arm >= 2;
            let addressing = if arm % 2 == 0 {
                "geometry"
            } else {
                "ordered-matched"
            };
            let words = new_words
                .get(&(fit, arm))
                .ok_or("missing epoch checkpoint")?;
            let movement = observed.receipt(words, support.geometry);
            let matches = report["arms"].as_array().is_some_and(|arms| {
                arms.len() == 4
                    && arms.iter().any(|item| {
                        item["centered"] == centered
                            && item["addressing"] == addressing
                            && item["movement"] == movement
                            && item["checkpoint_bytes"] == 512
                    })
            });
            require(matches, "independent movement recount differs from report")?;
            arms.push(json!({"addressing":addressing,"centered":centered,"movement":movement}));
        }
        let paired = epoch.grade()?;
        require(
            report["paired"] == paired,
            "independent paired grade differs from report",
        )?;
        if fit.epoch == 8 {
            let passed = gates.get_mut(&fit.policy).ok_or("unexpected policy")?;
            *passed &= paired["pass"] == true;
        }
        summaries.push(json!({"seed":fit.seed,"policy":fit.policy,"epoch":fit.epoch,"paired":paired,"arms":arms}));
        Ok(())
    };
    let current_count = stream(&current.join("visits.jsonl"), |row: Visit| {
        row.fit.check()?;
        validate_visit(&row)?;
        if !row.centered {
            require(
                old_visits.remove(&(row.fit, row.ordinal, row.addressing))
                    == Some((row.election, row.visit.learning)),
                &format!(
                    "uncentered visit differs from history at {:?} {:?} ordinal {}",
                    row.fit, row.addressing, row.ordinal
                ),
            )?;
        }
        let common = (row.election.group, row.election.gold, row.election.common);
        let expected = population.entry(row.ordinal).or_insert(common);
        require(
            *expected == common,
            "population identity or common mask drift",
        )?;
        let position = shared_positions
            .entry((row.fit.seed, row.fit.epoch, row.ordinal))
            .or_insert(row.visit_index);
        require(
            *position == row.visit_index,
            "locked schedule differs across arms",
        )?;
        let prior_index = if row.fit.epoch == 1 {
            None
        } else {
            Some(
                *shared_positions
                    .get(&(row.fit.seed, row.fit.epoch - 1, row.ordinal))
                    .ok_or("missing prior visit position")?,
            )
        };
        let elapsed = prior_index.map(|index| TRAIN_ROWS + row.visit_index - index);
        require(row.visits_since_prior == elapsed, "visit spacing differs")?;
        if batch.as_ref().is_some_and(|(fit, _)| *fit != row.fit) {
            if let Some((fit, epoch)) = batch.take() {
                finish(fit, epoch)?;
            }
        }
        batch
            .get_or_insert_with(|| (row.fit, Epoch::default()))
            .1
            .observe(&row)
    })?;
    if let Some((fit, epoch)) = batch {
        finish(fit, epoch)?;
    }
    require(
        old_visits.is_empty()
            && reported.is_empty()
            && current_count == 128 * TRAIN_ROWS
            && population.len() == TRAIN_ROWS
            && population
                .values()
                .filter(|&&(group, _, _)| withheld(group))
                .count()
                == HELD_ROWS,
        "visit tape or locked population coverage differs",
    )?;
    let run = read_json(&current.join("run.json"))?;
    require(
        run["historical_visits_verified"] == old_count
            && run["controls_verified"] == old_words.len(),
        "run does not report complete historical replay",
    )?;
    let mut gate_receipt = BTreeMap::new();
    let mut held_receipt = BTreeMap::new();
    for (policy, passed) in gates {
        require(
            run["two_seed_gate"][policy.key()] == passed,
            "reported fit gate differs from independent recount",
        )?;
        gate_receipt.insert(policy.key(), passed);
        let path = current.join(format!("confirmation-{}", policy.name()));
        let held = if passed {
            let verified = verify_held(&path, &population)?;
            require(
                run["confirmation"][policy.key()] == verified["pass"],
                "run's held gate differs",
            )?;
            verified
        } else {
            require(
                !path.try_exists()? && run["confirmation"][policy.key()].is_null(),
                "held group ran despite failing fit gate",
            )?;
            Value::Null
        };
        held_receipt.insert(policy.key(), held);
    }
    let audit_path = current.join("joint-conflicts");
    let conflicts_required = gate_receipt.values().any(|&p| !p);
    require(
        run["joint_conflicts_audit"] == conflicts_required
            && audit_path.try_exists()? == conflicts_required,
        "joint conflict audit dispatch disagrees with the preregistered gate",
    )?;
    let certificates = if conflicts_required {
        verify_certificates(&audit_path, &population, support.geometry)?
    } else {
        Value::Null
    };
    Ok(json!({
        "historical_visits_verified":old_count,
        "historical_visit_mismatches":0,
        "historical_checkpoints_verified":old_words.len(),
        "historical_checkpoint_mismatches":0,
        "visits_recounted":current_count,
        "epochs":summaries,
        "two_seed_gate":gate_receipt,
        "confirmation":held_receipt,
        "frozen_sparse":frozen_sparse,
        "joint_conflicts":certificates,
    }))
}

fn main() -> Result<()> {
    let args: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    let [prior, current] = args.as_slice() else {
        return Err("usage: recount PRIOR_TRIAL CURRENT_TRIAL".into());
    };
    let receipt = verify(prior, current)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &receipt)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Visit {
        let decision = Decision {
            margin_q48: 1,
            rival_prior_q48: 0,
            rival: 2,
            rank: 1,
        };
        Visit {
            fit: Fit {
                seed: SEEDS[0],
                epoch: 1,
                policy: Policy::GoldOnly,
            },
            addressing: Addressing::Geometry,
            centered: true,
            ordinal: 0,
            election: Electorate {
                group: 1,
                gold: 1,
                common: true,
                old_geometry: decision,
                zero: decision,
                ceiling: decision,
                checkpoint: decision,
            },
            visit: CenterVisit {
                learning: Learning {
                    before: decision,
                    after: decision,
                    radius: 0,
                    changed_words: 0,
                    l1: 0,
                    saturated_words: 0,
                    outcome: 0,
                },
                before_center: decision,
                gauge: Gauge {
                    score_q48: 1_i64 << 32,
                    offsets: [1, 0, 0, 0],
                    changed_words: 26,
                    l1: 26,
                },
                gold_before_q48: 0,
                gold_centered_q48: 1_i64 << 32,
                clipped_learning_words: 0,
                shift_failure: 0,
            },
            visit_index: TRAIN_ROWS - 1,
            age_visits: 0,
            age_learning_updates: 0,
            visits_since_prior: None,
        }
    }

    #[test]
    fn rejects_gauge_ranking_and_age_corruption() {
        let row = fixture();
        validate_visit(&row).unwrap();
        let mut altered = fixture();
        altered.visit.gauge.l1 += 1;
        assert!(validate_visit(&altered).is_err());
        let mut altered = fixture();
        altered.visit.before_center.margin_q48 -= 1;
        assert!(validate_visit(&altered).is_err());
        let mut altered = fixture();
        altered.age_visits += 1;
        assert!(validate_visit(&altered).is_err());
    }

    #[test]
    fn restores_complete_paired_groups_and_signed_bands() {
        let mut groups = GroupCounts::default();
        for _ in 0..3 {
            groups.observe(11, [false, true, false, false, false], false, false);
        }
        groups.observe(99, [false, false, false, false, false], false, false);
        let defeat = groups.grade(true);
        assert_eq!(
            defeat["centered_minus_historical"],
            json!({
                "observed_ppm": -750000, "lower_ppm": -1000000, "upper_ppm": 0,
                "defined_resamples": 4096,
            })
        );
        assert_eq!(defeat["centered_retention"]["defined_resamples"], 0);
        assert_eq!(defeat["pass"], false);
    }

    #[test]
    fn certificates_validate_strict_integer_boundary_and_support() {
        let mut side = Certificate {
            ordinal: 0,
            group: 11,
            gold_id: 2,
            rival_id: 3,
            gold_prior: 3,
            rival_prior: 2,
            gold_slots: [0, 64, 128, 192],
            rival_slots: [1, 64, 128, 192],
            required_difference: 0,
        };
        let vector = certificate_vector(&side, [3; 4]).unwrap();
        assert_eq!(&vector[..2], &[1, -1]);
        assert!(vector[2..].iter().all(|&v| v == 0));
        side.required_difference = 1;
        assert!(certificate_vector(&side, [3; 4]).is_err());
        side.rival_prior = side.gold_prior;
        assert!(certificate_vector(&side, [3; 4]).is_ok());
        side.rival_slots[0] = 64;
        assert!(certificate_vector(&side, [3; 4]).is_err());
    }
}
