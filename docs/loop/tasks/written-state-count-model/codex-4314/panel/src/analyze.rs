use crate::*;
use std::cmp::Ordering;
use std::fmt::Display;

const ARMS: [&str; 2] = ["colon", "marker"];
const ANSWER: usize = 0;
const RIVAL: usize = 1;
const WHITESPACE: usize = 2;

#[derive(Deserialize)]
struct Trace {
    group_id: String,
    assignment: usize,
    side: usize,
    pair_id: String,
    cohort: String,
    key: Option<u32>,
    support: u32,
    question_state: Option<u32>,
    scores: [i32; 3],
    ranks: [u32; 3],
    selected: u32,
    emitted: u32,
    winning_score: i32,
    trained_candidates: u32,
    first_admitted_cell: Option<u32>,
    writer_runner_up: Option<u32>,
    target: u32,
    rival: u32,
    raw_target: u32,
}

struct Opening {
    row: Row,
    key: Key,
    arms: [Trace; 2],
}

fn require(condition: bool, reason: &str) -> Result<()> {
    if !condition {
        return Err(reason.to_owned().into());
    }
    Ok(())
}

impl Opening {
    fn validate_trace(&self, trace: &Trace, marker: bool) -> Result<()> {
        let row = &self.row;
        let metadata_matches = trace.group_id == row.group_id
            && trace.assignment == row.assignment
            && trace.side == row.side
            && trace.pair_id == row.pair_id
            && trace.cohort == row.cohort
            && trace.target == row.target
            && trace.rival == row.rival
            && trace.raw_target == row.raw_target;
        if !metadata_matches {
            return Err(format!(
                "trace disagrees with panel at {} side {}",
                row.pair_id, row.side
            )
            .into());
        }
        let candidates = [row.target, row.rival, row.space_token];
        for (i, &id) in candidates.iter().enumerate() {
            require(
                trace.scores[i] <= trace.winning_score
                    && (trace.ranks[i] == 1) == (trace.selected == id)
                    && (trace.selected != id || trace.scores[i] == trace.winning_score)
                    && (0..i).all(|j| {
                        trace.ranks[i] != trace.ranks[j]
                            && (trace.scores[i] <= trace.scores[j]
                                || trace.ranks[i] < trace.ranks[j])
                            && (trace.scores[i] >= trace.scores[j]
                                || trace.ranks[i] > trace.ranks[j])
                    }),
                "inconsistent candidate score, rank or selection",
            )?;
        }
        require(
            trace.question_state.is_some() == marker
                && trace.ranks.iter().all(|&rank| rank > 0 && rank <= 4096)
                && (trace.key.is_some() || trace.support == 0)
                && [trace.selected, trace.emitted]
                    .into_iter()
                    .all(|id| id < 4096),
            "trace contains invalid state, rank, or original ID",
        )
    }

    fn probe_raw(&self) -> usize {
        if self.row.raw_target == self.row.space_token {
            WHITESPACE
        } else {
            ANSWER
        }
    }

    fn probe_raw_rival(&self) -> usize {
        if self.row.raw_rival == self.row.space_token {
            WHITESPACE
        } else {
            RIVAL
        }
    }
}

#[derive(Default, Serialize)]
struct MatchedOutcome {
    neither: usize,
    colon_only: usize,
    marker_only: usize,
    both: usize,
}

impl MatchedOutcome {
    fn add(&mut self, arms: [bool; 2]) {
        match arms {
            [false, false] => self.neither += 1,
            [true, false] => self.colon_only += 1,
            [false, true] => self.marker_only += 1,
            [true, true] => self.both += 1,
        }
    }
}

#[derive(Default, Serialize)]
struct MatchedArms {
    strict_two_way_pairs: MatchedOutcome,
    strict_two_way_both_assignments_groups: MatchedOutcome,
    raw_emission_correct: MatchedOutcome,
    lexical_target_above_space: MatchedOutcome,
}

fn matched_arms(groups: &BTreeMap<String, Vec<Opening>>) -> BTreeMap<String, MatchedArms> {
    let mut by_cohort = BTreeMap::<String, MatchedArms>::new();
    for group in groups.values() {
        let output = by_cohort.entry(group[0].row.cohort.clone()).or_default();
        let mut both_assignments = [true; 2];
        for pair in group.chunks_exact(2) {
            let wins = [0, 1].map(|arm| {
                pair.iter().all(|opening| {
                    let trace = &opening.arms[arm];
                    trace.scores[ANSWER] > trace.scores[RIVAL]
                })
            });
            output.strict_two_way_pairs.add(wins);
            both_assignments[0] &= wins[0];
            both_assignments[1] &= wins[1];
        }
        if group.len() == 4 {
            output
                .strict_two_way_both_assignments_groups
                .add(both_assignments);
        }
        for opening in group {
            output.raw_emission_correct.add(
                opening
                    .arms
                    .each_ref()
                    .map(|trace| trace.emitted == opening.row.raw_target),
            );
            output.lexical_target_above_space.add(
                opening
                    .arms
                    .each_ref()
                    .map(|trace| trace.scores[ANSWER] > trace.scores[WHITESPACE]),
            );
        }
    }
    by_cohort
}

// Offline statistics on independent groups. Expand the Bernoulli sum directly
// so that the tail does not lose precision by subtraction from one.
fn binomial_lower_tail(n: usize, k: usize, probability: f64) -> f64 {
    let mut masses = vec![0.0; k + 1];
    masses[0] = 1.0;
    for draw in 1..=n {
        for hits in (1..=k.min(draw)).rev() {
            masses[hits] = masses[hits] * (1.0 - probability) + masses[hits - 1] * probability;
        }
        masses[0] *= 1.0 - probability;
    }
    masses.into_iter().sum()
}

fn binomial_bound(n: usize, successes: usize, tail: f64, lower: bool) -> f64 {
    if lower && successes == 0 {
        return 0.0;
    }
    if !lower && successes == n {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..55 {
        let mid = (lo + hi) / 2.0;
        let move_right = if lower {
            binomial_lower_tail(n, n - successes, 1.0 - mid) < tail
        } else {
            binomial_lower_tail(n, successes, mid) > tail
        };
        if move_right {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

fn group_statistics(n: usize, successes: usize) -> serde_json::Value {
    json!({
        "successes":successes,
        "groups":n,
        "fraction":successes as f64 / n as f64,
        "two_sided_95_clopper_pearson":[
            binomial_bound(n, successes, 0.025, true),
            binomial_bound(n, successes, 0.025, false),
        ],
        "one_sided_95_lower_clopper_pearson":binomial_bound(n, successes, 0.05, true),
    })
}

fn primary(
    groups: &BTreeMap<String, Vec<Opening>>,
    matched: &BTreeMap<String, MatchedArms>,
) -> serde_json::Value {
    let pair = &matched["controlled"].strict_two_way_both_assignments_groups;
    let n = pair.neither + pair.colon_only + pair.marker_only + pair.both;
    let discordant = pair.colon_only + pair.marker_only;
    let p = if discordant == 0 {
        1.0
    } else {
        (2.0 * binomial_lower_tail(discordant, pair.colon_only.min(pair.marker_only), 0.5)).min(1.0)
    };
    let joint = groups
        .values()
        .filter(|group| {
            group.len() == 4
                && group.iter().all(|opening| {
                    let scores = opening.arms[1].scores;
                    scores[ANSWER] > scores[RIVAL] && scores[ANSWER] > scores[WHITESPACE]
                })
        })
        .count();
    let joint_lower = binomial_bound(n, joint, 0.05, true);
    json!({
        "unit":"independent controlled group, both questions under both fact assignments",
        "strict_two_way_both_assignments":{
            "matched_outcomes":pair,
            "exact_two_sided_discordance_p_value":p,
            "colon":group_statistics(n,pair.colon_only+pair.both),
            "marker":group_statistics(n,pair.marker_only+pair.both),
        },
        "marker_all_four_targets_beat_rival_and_space":{
            "interval":group_statistics(n,joint),
            "advancement_bound_exceeds_one_half":joint_lower > 0.5,
        },
    })
}

fn load(root: &Path, traces: [&Path; 2]) -> Result<BTreeMap<String, Vec<Opening>>> {
    let rows: Vec<Row> = read_jsonl(&root.join("held/rows.jsonl"))?;
    let keys: Vec<Key> = read_jsonl(&root.join("held/keys.jsonl"))?;
    let [colon, marker] = traces.map(read_jsonl::<Trace>);
    let (colon, marker) = (colon?, marker?);
    require(
        rows.len() == keys.len() && rows.len() == colon.len() && rows.len() == marker.len(),
        "arm trace, row and key counts differ",
    )?;
    let mut groups = BTreeMap::<String, Vec<_>>::new();
    let mut identities = HashSet::new();
    for (((row, key), colon), marker) in rows.into_iter().zip(keys).zip(colon).zip(marker) {
        require(
            row.group_id == key.group_id
                && row.pair_id == key.pair_id
                && row.assignment == key.assignment
                && row.side == key.side
                && key.gold.get(row.lexical_offset) == Some(&row.target)
                && key.gold.first() == Some(&row.raw_target)
                && row.prefix.ends_with(&MARKER)
                && identities.insert((row.pair_id.clone(), row.side)),
            "invalid row/key identity or boundary",
        )?;
        let opening = Opening {
            row,
            key,
            arms: [colon, marker],
        };
        opening.validate_trace(&opening.arms[0], false)?;
        opening.validate_trace(&opening.arms[1], true)?;
        groups
            .entry(opening.row.group_id.clone())
            .or_default()
            .push(opening);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|opening| (opening.row.assignment, opening.row.side));
        for (assignment, pair) in group.chunks(2).enumerate() {
            let [a, b] = pair else {
                return Err("incomplete held pair".into());
            };
            require(
                a.row.assignment == assignment
                    && b.row.assignment == assignment
                    && a.row.side == 0
                    && b.row.side == 1
                    && a.row.pair_id == b.row.pair_id
                    && a.row.cohort == b.row.cohort
                    && a.row.target == b.row.rival
                    && b.row.target == a.row.rival
                    && a.row.target != b.row.target
                    && a.row.raw_target == b.row.raw_rival
                    && b.row.raw_target == a.row.raw_rival,
                "held pair has wrong candidate direction",
            )?;
        }
        match group.as_slice() {
            [a, b, c, d] if a.row.cohort == "controlled" => require(
                c.row.cohort == "controlled"
                    && d.row.cohort == "controlled"
                    && a.key.question == c.key.question
                    && b.key.question == d.key.question
                    && a.row.target == d.row.target
                    && b.row.target == c.row.target
                    && group.iter().all(|opening| opening.row.lexical_offset == 0),
                "controlled held group failed the independent assignment swap check",
            )?,
            [a, b] if a.row.cohort == "coqa" && b.row.cohort == "coqa" => {}
            _ => return Err("unexpected held group shape".into()),
        }
    }
    Ok(groups)
}

#[derive(Default, Serialize)]
struct Directions {
    less: usize,
    equal: usize,
    greater: usize,
}

impl Directions {
    fn add(&mut self, comparison: Ordering) {
        match comparison {
            Ordering::Less => self.less += 1,
            Ordering::Equal => self.equal += 1,
            Ordering::Greater => self.greater += 1,
        }
    }
}

#[derive(Default, Serialize)]
struct Collisions {
    both_present: usize,
    equal: usize,
    distinct: usize,
    one_absent: usize,
    both_absent: usize,
}

impl Collisions {
    fn add(&mut self, a: Option<u32>, b: Option<u32>) {
        match (a, b) {
            (Some(a), Some(b)) => {
                self.both_present += 1;
                self.equal += usize::from(a == b);
                self.distinct += usize::from(a != b);
            }
            (None, None) => self.both_absent += 1,
            _ => self.one_absent += 1,
        }
    }
}

#[derive(Default, Serialize)]
struct Rows {
    openings: usize,
    configured_key_levels: BTreeMap<u32, usize>,
    key_absent: usize,
    support_zero: usize,
    no_trained_candidates: usize,
    target_vs_rival: Directions,
    target_vs_space: Directions,
    target_score_vs_zero: Directions,
    rival_score_vs_zero: Directions,
    space_score_vs_zero: Directions,
    target_wins_with_positive_score: usize,
    target_wins_with_zero_score: usize,
    raw_target_vs_raw_rival: Directions,
    raw_target_is_space: usize,
    selected_target: usize,
    emitted_target: usize,
    selected_raw_target: usize,
    emitted_raw_target: usize,
    selected_space: usize,
    emitted_space: usize,
    writer_changed_selection: usize,
    selected_ids: BTreeMap<u32, usize>,
    emitted_ids: BTreeMap<u32, usize>,
    #[serde(skip)]
    target_ranks: Vec<u32>,
    #[serde(skip)]
    rival_ranks: Vec<u32>,
    #[serde(skip)]
    space_ranks: Vec<u32>,
    #[serde(skip)]
    raw_ranks: Vec<u32>,
    target_rank_range_and_middle: Option<[u32; 4]>,
    rival_rank_range_and_middle: Option<[u32; 4]>,
    space_rank_range_and_middle: Option<[u32; 4]>,
    raw_rank_range_and_middle: Option<[u32; 4]>,
}

fn range_and_middle(values: &mut [u32]) -> Option<[u32; 4]> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let last = values.len() - 1;
    Some([
        values[0],
        values[last / 2],
        values[(last + 1) / 2],
        values[last],
    ])
}

impl Rows {
    fn add(&mut self, opening: &Opening, arm: usize) {
        let row = &opening.row;
        let trace = &opening.arms[arm];
        let raw_probe = opening.probe_raw();
        self.openings += 1;
        if let Some(key) = trace.key {
            *self.configured_key_levels.entry(key >> 20).or_default() += 1;
        } else {
            self.key_absent += 1;
        }
        self.support_zero += usize::from(trace.support == 0);
        self.no_trained_candidates += usize::from(trace.trained_candidates == 0);
        self.target_vs_rival
            .add(trace.scores[ANSWER].cmp(&trace.scores[RIVAL]));
        self.target_vs_space
            .add(trace.scores[ANSWER].cmp(&trace.scores[WHITESPACE]));
        self.target_score_vs_zero.add(trace.scores[ANSWER].cmp(&0));
        self.rival_score_vs_zero.add(trace.scores[RIVAL].cmp(&0));
        self.space_score_vs_zero
            .add(trace.scores[WHITESPACE].cmp(&0));
        let target_wins = trace.scores[ANSWER] > trace.scores[RIVAL];
        self.target_wins_with_positive_score +=
            usize::from(target_wins && trace.scores[ANSWER] > 0);
        self.target_wins_with_zero_score += usize::from(target_wins && trace.scores[ANSWER] == 0);
        self.raw_target_vs_raw_rival
            .add(trace.scores[raw_probe].cmp(&trace.scores[opening.probe_raw_rival()]));
        self.raw_target_is_space += usize::from(row.raw_target == row.space_token);
        self.selected_target += usize::from(trace.selected == row.target);
        self.emitted_target += usize::from(trace.emitted == row.target);
        self.selected_raw_target += usize::from(trace.selected == row.raw_target);
        self.emitted_raw_target += usize::from(trace.emitted == row.raw_target);
        self.selected_space += usize::from(trace.selected == row.space_token);
        self.emitted_space += usize::from(trace.emitted == row.space_token);
        self.writer_changed_selection += usize::from(trace.selected != trace.emitted);
        *self.selected_ids.entry(trace.selected).or_default() += 1;
        *self.emitted_ids.entry(trace.emitted).or_default() += 1;
        self.target_ranks.push(trace.ranks[ANSWER]);
        self.rival_ranks.push(trace.ranks[RIVAL]);
        self.space_ranks.push(trace.ranks[WHITESPACE]);
        self.raw_ranks.push(trace.ranks[raw_probe]);
    }

    fn finish(&mut self) {
        self.target_rank_range_and_middle = range_and_middle(&mut self.target_ranks);
        self.rival_rank_range_and_middle = range_and_middle(&mut self.rival_ranks);
        self.space_rank_range_and_middle = range_and_middle(&mut self.space_ranks);
        self.raw_rank_range_and_middle = range_and_middle(&mut self.raw_ranks);
    }
}

#[derive(Default, Serialize)]
struct Aggregate {
    rows: Rows,
    groups: usize,
    question_pairs: usize,
    strict_two_way_pairs: usize,
    wrong_both_pairs: usize,
    tied_either_side_pairs: usize,
    strict_two_way_raw_pairs: usize,
    raw_two_way_equal_id_pairs: usize,
    both_targets_beat_space_pairs: usize,
    both_emitted_target_pairs: usize,
    both_emitted_raw_target_pairs: usize,
    both_lexical_references_at_marker_pairs: usize,
    strict_two_way_lexical_references_at_marker_pairs: usize,
    strict_two_way_both_assignments_groups: usize,
    both_raw_emissions_on_both_assignments_groups: usize,
    target_score_change_own_vs_crossed_question: Directions,
    target_rank_change_own_vs_crossed_question: Directions,
    both_fixed_target_scores_higher_with_own_question_pairs: usize,
    all_fixed_target_scores_higher_with_own_question_both_assignments_groups: usize,
    target_score_change_own_vs_crossed_facts: Directions,
    key_collision_across_question: Collisions,
    question_state_collision_across_question: Collisions,
    key_collision_across_assignment: Collisions,
    question_state_collision_across_assignment: Collisions,
    actual_selection_changed_across_question: usize,
    emission_changed_across_question: usize,
}

impl Aggregate {
    fn add(&mut self, group: &[Opening], arm: usize) {
        self.groups += 1;
        let mut both_assignments_win = true;
        let mut both_assignments_emitted = true;
        let mut both_assignments_fixed_scores = true;
        for opening in group {
            self.rows.add(opening, arm);
        }
        for pair in group.chunks_exact(2) {
            let [a, b] = [&pair[0], &pair[1]];
            let [x, y] = [&a.arms[arm], &b.arms[arm]];
            let margins = [x, y].map(|t| t.scores[ANSWER].cmp(&t.scores[RIVAL]));
            let wins = margins == [Ordering::Greater; 2];
            let emitted_raw = x.emitted == a.row.raw_target && y.emitted == b.row.raw_target;
            let fixed = [
                x.scores[ANSWER].cmp(&y.scores[RIVAL]),
                y.scores[ANSWER].cmp(&x.scores[RIVAL]),
            ];
            let fixed_ranks = [
                y.ranks[RIVAL].cmp(&x.ranks[ANSWER]),
                x.ranks[RIVAL].cmp(&y.ranks[ANSWER]),
            ];
            let fixed_wins = fixed == [Ordering::Greater; 2];
            self.question_pairs += 1;
            self.strict_two_way_pairs += usize::from(wins);
            self.wrong_both_pairs += usize::from(margins == [Ordering::Less; 2]);
            self.tied_either_side_pairs += usize::from(margins.contains(&Ordering::Equal));
            self.strict_two_way_raw_pairs += usize::from(
                x.scores[a.probe_raw()] > x.scores[a.probe_raw_rival()]
                    && y.scores[b.probe_raw()] > y.scores[b.probe_raw_rival()],
            );
            self.raw_two_way_equal_id_pairs += usize::from(a.row.raw_target == b.row.raw_target);
            self.both_targets_beat_space_pairs += usize::from(
                [x, y]
                    .into_iter()
                    .all(|t| t.scores[ANSWER] > t.scores[WHITESPACE]),
            );
            self.both_emitted_target_pairs +=
                usize::from(x.emitted == a.row.target && y.emitted == b.row.target);
            self.both_emitted_raw_target_pairs += usize::from(emitted_raw);
            let at_marker = a.row.lexical_offset == 0 && b.row.lexical_offset == 0;
            self.both_lexical_references_at_marker_pairs += usize::from(at_marker);
            self.strict_two_way_lexical_references_at_marker_pairs +=
                usize::from(at_marker && wins);
            self.both_fixed_target_scores_higher_with_own_question_pairs += usize::from(fixed_wins);
            self.key_collision_across_question.add(x.key, y.key);
            self.question_state_collision_across_question
                .add(x.question_state, y.question_state);
            self.actual_selection_changed_across_question += usize::from(x.selected != y.selected);
            self.emission_changed_across_question += usize::from(x.emitted != y.emitted);
            for ordering in fixed {
                self.target_score_change_own_vs_crossed_question
                    .add(ordering);
            }
            for ordering in fixed_ranks {
                self.target_rank_change_own_vs_crossed_question
                    .add(ordering);
            }
            both_assignments_win &= wins;
            both_assignments_emitted &= emitted_raw;
            both_assignments_fixed_scores &= fixed_wins;
        }
        if let [a, b, c, d] = group {
            self.strict_two_way_both_assignments_groups += usize::from(both_assignments_win);
            self.both_raw_emissions_on_both_assignments_groups +=
                usize::from(both_assignments_emitted);
            self.all_fixed_target_scores_higher_with_own_question_both_assignments_groups +=
                usize::from(both_assignments_fixed_scores);
            for (left, right) in [(a, c), (b, d)] {
                let x = &left.arms[arm];
                let y = &right.arms[arm];
                self.key_collision_across_assignment.add(x.key, y.key);
                self.question_state_collision_across_assignment
                    .add(x.question_state, y.question_state);
                self.target_score_change_own_vs_crossed_facts
                    .add(x.scores[ANSWER].cmp(&y.scores[RIVAL]));
                self.target_score_change_own_vs_crossed_facts
                    .add(y.scores[ANSWER].cmp(&x.scores[RIVAL]));
            }
        }
    }
}

fn exposure_name(count: usize) -> &'static str {
    [
        "neither_candidate_seen",
        "one_candidate_seen",
        "both_candidates_seen",
    ][count]
}

fn summary(
    groups: &BTreeMap<String, Vec<Opening>>,
    train: &HashSet<u32>,
    arm: usize,
) -> Result<serde_json::Value> {
    let mut strata = BTreeMap::<String, Aggregate>::new();
    let mut rows_by_target_exposure = BTreeMap::<String, Rows>::new();
    for group in groups.values() {
        let first = &group[0];
        let cohort = first.row.cohort.as_str();
        let count = [first.row.target, first.row.rival]
            .into_iter()
            .filter(|id| train.contains(id))
            .count();
        let mut domains = vec![
            cohort.to_owned(),
            format!("{cohort}/{}", exposure_name(count)),
        ];
        if let Some(family) = first.key.source["family"].as_str() {
            domains.push(format!("controlled/wording/{family}"));
            domains.push(format!(
                "controlled/wording/{family}/{}",
                exposure_name(count)
            ));
        } else {
            let kind = if group.iter().any(|opening| opening.row.lexical_offset > 0) {
                "mixed_whitespace"
            } else {
                "both_lexical_at_marker"
            };
            domains.push(format!("coqa/{kind}"));
        }
        for name in domains {
            strata.entry(name).or_default().add(group, arm);
        }
        for opening in group {
            let exposure = if train.contains(&opening.row.target) {
                "seen"
            } else {
                "unseen"
            };
            rows_by_target_exposure
                .entry(format!("{cohort}/{exposure}"))
                .or_default()
                .add(opening, arm);
        }
    }
    for stats in strata.values_mut() {
        stats.rows.finish();
    }
    for stats in rows_by_target_exposure.values_mut() {
        stats.finish();
    }
    Ok(json!({"groups":strata,"rows_by_contrast_target_exposure":rows_by_target_exposure}))
}

fn csv_line(writer: &mut BufWriter<File>, fields: &[String]) -> Result<()> {
    for (index, field) in fields.iter().enumerate() {
        if index != 0 {
            write!(writer, ",")?;
        }
        let escaped = field.replace('"', "\"\"");
        write!(writer, "\"{escaped}\"")?;
    }
    writeln!(writer)?;
    Ok(())
}

fn value<T: Display>(values: &mut Vec<String>, value: T) {
    values.push(value.to_string());
}

const ARM_COLUMNS: &[&str] = &[
    "key",
    "key_level",
    "support",
    "question_state",
    "trained_candidates",
    "winning_score",
    "label_a_score",
    "label_a_rank",
    "label_b_score",
    "label_b_rank",
    "space_score",
    "space_rank",
    "immediate_score",
    "immediate_rank",
    "selected",
    "emitted",
    "first_admitted_cell",
    "writer_runner_up",
];

fn cells(row: &Opening, label_a: u32) -> Vec<String> {
    let mut fields = vec![
        row.row.pair_id.clone(),
        row.key.question.clone(),
        row.key.reply.clone(),
    ];
    value(&mut fields, row.row.target);
    value(&mut fields, row.row.rival);
    value(&mut fields, row.row.raw_target);
    value(&mut fields, row.row.lexical_offset);
    value(&mut fields, row.row.prefix.len());
    for arm in &row.arms {
        let a = if row.row.target == label_a {
            ANSWER
        } else {
            RIVAL
        };
        let b = if a == ANSWER { RIVAL } else { ANSWER };
        fields.push(arm.key.map(|id| id.to_string()).unwrap_or_default());
        fields.push(
            arm.key
                .map(|key| (key >> 20).to_string())
                .unwrap_or_default(),
        );
        value(&mut fields, arm.support);
        fields.push(
            arm.question_state
                .map(|id| id.to_string())
                .unwrap_or_default(),
        );
        value(&mut fields, arm.trained_candidates);
        value(&mut fields, arm.winning_score);
        for index in [a, b, WHITESPACE, row.probe_raw()] {
            value(&mut fields, arm.scores[index]);
            value(&mut fields, arm.ranks[index]);
        }
        value(&mut fields, arm.selected);
        value(&mut fields, arm.emitted);
        for optional in [arm.first_admitted_cell, arm.writer_runner_up] {
            fields.push(
                optional
                    .filter(|&word| word != u32::MAX)
                    .map(|word| word.to_string())
                    .unwrap_or_default(),
            );
        }
    }
    fields
}

fn cell_columns() -> Vec<String> {
    let mut headers = [
        "pair_id",
        "question",
        "reference",
        "lexical_target_id",
        "lexical_rival_id",
        "immediate_target_id",
        "lexical_offset",
        "prefix_tokens",
    ]
    .map(str::to_owned)
    .to_vec();
    for arm in ARMS {
        headers.extend(ARM_COLUMNS.iter().map(|name| format!("{arm}_{name}")));
    }
    headers
}

fn context(opening: &Opening) -> Result<&str> {
    opening
        .key
        .user
        .rsplit_once("\n\n")
        .map(|(text, _)| text)
        .ok_or_else(|| "private key lost its shared context".into())
}

fn write_outputs(
    groups: &BTreeMap<String, Vec<Opening>>,
    train: &HashSet<u32>,
    out: &Path,
) -> Result<(String, serde_json::Value)> {
    let headings = cell_columns();
    let mut openings = writer(&out.join("openings.csv"))?;
    let mut wide = writer(&out.join("controlled-matrix.csv"))?;
    let mut coqa = writer(&out.join("natural-openings.csv"))?;
    let mut long_header = [
        "group_id",
        "cohort",
        "family_or_source",
        "assignment",
        "side",
        "context",
        "label_a_text",
        "label_a_id",
        "label_a_in_contrast",
        "label_b_text",
        "label_b_id",
        "label_b_in_contrast",
    ]
    .map(str::to_owned)
    .to_vec();
    long_header.extend(headings.iter().cloned());
    csv_line(&mut openings, &long_header)?;
    csv_line(&mut coqa, &long_header)?;
    let mut wide_header = [
        "group_id",
        "family",
        "assignment_0_context",
        "assignment_1_context",
        "label_a_text",
        "label_a_id",
        "label_a_in_contrast",
        "label_b_text",
        "label_b_id",
        "label_b_in_contrast",
    ]
    .map(str::to_owned)
    .to_vec();
    for (assignment, side) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        wide_header.extend(
            headings
                .iter()
                .map(|name| format!("assignment_{assignment}_side_{side}_{name}")),
        );
    }
    csv_line(&mut wide, &wide_header)?;
    let mut example = None;
    for (id, group) in groups {
        let (a, b) = (&group[0], &group[1]);
        let [label_a, label_b] = [a.row.target, b.row.target];
        let labels: Vec<_> = [a, b]
            .into_iter()
            .flat_map(|opening| {
                [
                    opening.key.reply.clone(),
                    opening.row.target.to_string(),
                    train.contains(&opening.row.target).to_string(),
                ]
            })
            .collect();
        let source = a.key.source["family"]
            .as_str()
            .or(a.key.source["source"].as_str())
            .ok_or("missing wording family or natural source")?;
        for opening in group {
            let mut fields = vec![
                id.clone(),
                opening.row.cohort.clone(),
                source.to_owned(),
                opening.row.assignment.to_string(),
                opening.row.side.to_string(),
                context(opening)?.to_owned(),
            ];
            fields.extend(labels.iter().cloned());
            fields.extend(cells(opening, label_a));
            csv_line(&mut openings, &fields)?;
            if opening.row.cohort == "coqa" {
                csv_line(&mut coqa, &fields)?;
            }
        }
        if let [_, _, c, _] = group.as_slice() {
            let mut fields = vec![
                id.clone(),
                source.to_owned(),
                context(a)?.to_owned(),
                context(c)?.to_owned(),
            ];
            fields.extend(labels);
            for opening in group {
                fields.extend(cells(opening, label_a));
            }
            csv_line(&mut wide, &fields)?;
            if example.is_none() {
                let details = fields
                    .iter()
                    .zip(&wide_header)
                    .map(|(value, name)| (name.clone(), value.clone()))
                    .collect::<BTreeMap<_, _>>();
                let private = out.join("example.json");
                let mut writer = writer(&private)?;
                serde_json::to_writer_pretty(&mut writer, &details)?;
                writeln!(writer)?;
                writer.flush()?;
                example = Some((
                    id.clone(),
                    json!({
                        "family":source,
                        "label_ids":[label_a,label_b],
                        "prompts": group.iter().map(|opening| {
                            json!({
                                "pair_id": opening.row.pair_id,
                                "assignment":opening.row.assignment,
                                "side":opening.row.side,
                                "target":opening.row.target,
                                "rival":opening.row.rival,
                                "arms":opening.arms.iter().enumerate().map(|(idx,t)| (ARMS[idx], json!({
                                    "key":t.key,"support":t.support,"scores":t.scores,"ranks":t.ranks,
                                    "question_state":t.question_state,"selected":t.selected,"emitted":t.emitted,
                                }))).collect::<BTreeMap<_,_>>(),
                            })
                        }).collect::<Vec<_>>(),
                    }),
                ));
            }
        }
    }
    for file in [&mut openings, &mut wide, &mut coqa] {
        file.flush()?;
    }
    example.ok_or_else(|| "no controlled example".into())
}

pub(crate) fn run(args: &[&str]) -> Result<()> {
    let [panel, colon, marker, out] = args else {
        return Err("usage: marker-question-panel --summarize SEALED_PANEL COLON_TRACE MARKER_TRACE NEW_OUTPUT".into());
    };
    let root = Path::new(panel);
    let traces = [Path::new(colon), Path::new(marker)];
    let train: HashSet<_> = read_jsonl::<Row>(&root.join("train/rows.jsonl"))?
        .into_iter()
        .map(|row| row.target)
        .collect();
    let groups = load(root, traces)?;
    require(
        groups.len() == 205 && groups.values().flatten().count() == 810,
        "held result has unexpected coverage",
    )?;
    let out = Path::new(out);
    fs::create_dir(out)?;
    let (example_id, example) = write_outputs(&groups, &train, out)?;
    let matched = matched_arms(&groups);
    let mut report = writer(&out.join("summary.json"))?;
    serde_json::to_writer_pretty(
        &mut report,
        &json!({
            "schema":"uor-r4.marker-question-panel-aggregate/1",
            "panel_manifest_sha256":sha(&root.join("sha256sums.txt"))?,
            "trace_hashes":{"colon":sha(traces[0])?, "marker":sha(traces[1])?},
            "ranks": "ranges contain minimum, lower middle, upper middle and maximum; smaller rank is better",
            "comparison_direction": "greater is higher score or a better numerical rank for the correct target",
            "exposure": "seen means a lexical target in the controlled contrast tape",
            "scope":"scores are measured at the colon; a mixed whitespace pair's lexical probes are counterfactual at that step",
            "example_group":example_id,
            "example":example,
        "primary_group_statistics":primary(&groups,&matched),
        "matched_arms":matched,
            "arms":{"colon":summary(&groups, &train, 0)?, "marker":summary(&groups, &train, 1)?},
        }),
    )?;
    writeln!(report)?;
    report.flush()?;
    let mut manifest = writer(&out.join("sha256sums.txt"))?;
    for name in [
        "summary.json",
        "example.json",
        "openings.csv",
        "controlled-matrix.csv",
        "natural-openings.csv",
    ] {
        writeln!(manifest, "{}  {name}", sha(&out.join(name))?)?;
    }
    manifest.flush()?;
    Ok(())
}
