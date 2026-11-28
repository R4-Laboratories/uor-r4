use crate::{verify::*, wire::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    numerator: i128,
    denominator: i128,
}

impl Rate {
    fn new(numerator: i128, denominator: i128) -> Self {
        debug_assert!(denominator > 0);
        let (mut divisor, mut rest) = (numerator.abs(), denominator);
        while rest != 0 {
            (divisor, rest) = (rest, divisor % rest);
        }
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }
    fn plus(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )
    }
    fn minus(self, other: Self) -> Self {
        self.plus(Self {
            numerator: -other.numerator,
            ..other
        })
    }
    fn at_least(self, numerator: i128, denominator: i128) -> bool {
        self.numerator * denominator >= numerator * self.denominator
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Tally {
    pub wins: usize,
    pub total: usize,
}

impl Tally {
    fn record(&mut self, win: bool) {
        self.wins += usize::from(win);
        self.total += 1;
    }
    fn merge(self, other: Self) -> Self {
        Self {
            wins: self.wins + other.wins,
            total: self.total + other.total,
        }
    }
    fn rate(self) -> Option<Rate> {
        (self.total > 0).then(|| Rate::new(self.wins as i128, self.total as i128))
    }
}

fn title_macro(values: impl Iterator<Item = Tally>) -> Option<Rate> {
    let mut sum = Rate::new(0, 1);
    let mut count = 0;
    for tally in values {
        sum = sum.plus(tally.rate()?);
        count += 1;
    }
    (count > 0).then(|| Rate::new(sum.numerator, sum.denominator * count))
}

#[derive(Default, Serialize)]
struct PairedChanges {
    wins: usize,
    losses: usize,
    ties: usize,
}

impl PairedChanges {
    fn record(&mut self, first: bool, second: bool) {
        match (first, second) {
            (true, false) => self.wins += 1,
            (false, true) => self.losses += 1,
            _ => self.ties += 1,
        }
    }
}

#[derive(Serialize)]
struct CaseGrade<'a> {
    case_id: &'a str,
    target: Option<Span>,
    selected: Option<Span>,
    gold_margin: i32,
    strict: bool,
}

impl<'a> CaseGrade<'a> {
    fn from(observation: Observation<'a>, scores: &Scores) -> Self {
        let gold_margin = strict_margin(observation.span(), observation, scores);
        Self {
            case_id: &observation.case.id,
            target: observation.span(),
            selected: scores.selection.span,
            gold_margin,
            strict: gold_margin >= 1,
        }
    }
}

#[derive(Serialize)]
struct PairGrade<'a> {
    passage_sha256: &'a str,
    left_id: &'a str,
    right_id: &'a str,
    stage: &'static str,
    reason: Option<&'a str>,
    qualified: bool,
    solved: bool,
    cases: Vec<CaseGrade<'a>>,
}

#[derive(Serialize)]
struct TitleGrade<'a> {
    title: &'a str,
    family: Family,
    mechanical: usize,
    passages: usize,
    original_selected: Tally,
    original_qualified: Tally,
    selected: Tally,
    qualified: Tally,
    positive: Tally,
    no_read: Tally,
    unanswered_pairs: Tally,
    contradicted_premise_pairs: Tally,
    rows: Vec<PairGrade<'a>>,
}

fn title<'a>(
    title: &'a Title,
    roster: &CheckedRoster<'a>,
    scores: &ScoresByCase<'_>,
) -> TitleGrade<'a> {
    let mut report = TitleGrade {
        title: &title.name,
        family: title.family,
        mechanical: title.mechanical_candidates,
        passages: title.passages,
        original_selected: Tally::default(),
        original_qualified: Tally::default(),
        selected: Tally::default(),
        qualified: Tally::default(),
        positive: Tally::default(),
        no_read: Tally::default(),
        unanswered_pairs: Tally::default(),
        contradicted_premise_pairs: Tally::default(),
        rows: Vec::new(),
    };
    for (i, pair) in title.selected.iter().enumerate() {
        let (stage, reason) = match &pair.qualification {
            Qualification::SourceExcluded { reason } => ("source_excluded", Some(reason.as_str())),
            Qualification::CaptureExcluded { reason } => {
                ("capture_excluded", Some(reason.as_str()))
            }
            Qualification::RebindExcluded { reason } => ("rebind_excluded", Some(reason.as_str())),
            Qualification::Qualified { .. } => ("qualified", None),
        };
        let mut row = PairGrade {
            passage_sha256: &pair.passage_sha256,
            left_id: &pair.left_id,
            right_id: &pair.right_id,
            stage,
            reason,
            qualified: reason.is_none(),
            solved: false,
            cases: Vec::new(),
        };
        if let Qualification::Qualified { opening } = &pair.qualification {
            for case in &opening.cases {
                let grade = CaseGrade::from(
                    roster.validation[case.id.as_str()],
                    scores[case.id.as_str()],
                );
                if grade.target.is_some() {
                    report.positive.record(grade.strict);
                } else {
                    report.no_read.record(grade.strict);
                }
                row.cases.push(grade);
            }
            row.solved = row.cases.iter().all(|case| case.strict);
            report.qualified.record(row.solved);
            match opening.cases[1].target {
                Target::NoRead {
                    group: Negative::Unanswered,
                } => report.unanswered_pairs.record(row.solved),
                Target::NoRead {
                    group: Negative::ContradictedPremise,
                } => report.contradicted_premise_pairs.record(row.solved),
                Target::Source { .. } => {}
            }
        }
        if i < title.passages.min(4) {
            report.original_selected.record(row.solved);
            if row.qualified {
                report.original_qualified.record(row.solved);
            }
        }
        report.selected.record(row.solved);
        report.rows.push(row);
    }
    report
}

#[derive(Serialize)]
struct Metric {
    pooled: Tally,
    macro_rate: Option<Rate>,
}

fn metric<'a, 't: 'a>(
    titles: &'a [&'a TitleGrade<'t>],
    field: impl Fn(&TitleGrade<'t>) -> Tally,
) -> Metric {
    let pooled = titles
        .iter()
        .map(|title| field(title))
        .fold(Tally::default(), Tally::merge);
    let macro_rate = title_macro(titles.iter().map(|title| field(title)));
    Metric { pooled, macro_rate }
}

#[derive(Serialize)]
struct FamilyGrade {
    original_selected_pairs: Metric,
    original_qualified_pairs: Metric,
    selected_pairs: Metric,
    qualified_pairs: Metric,
    qualified_positives: Metric,
    qualified_no_read: Metric,
    unanswered_pairs: Tally,
    contradicted_premise_pairs: Tally,
    singleton_title: Option<Tally>,
    excluding_singleton_qualified_pairs: Option<Metric>,
}

fn family(titles: &[TitleGrade<'_>], family: Family) -> FamilyGrade {
    let selected: Vec<_> = titles
        .iter()
        .filter(|title| title.family == family)
        .collect();
    let singleton = selected.iter().find(|title| title.passages == 1);
    FamilyGrade {
        original_selected_pairs: metric(&selected, |title| title.original_selected),
        original_qualified_pairs: metric(&selected, |title| title.original_qualified),
        selected_pairs: metric(&selected, |title| title.selected),
        qualified_pairs: metric(&selected, |title| title.qualified),
        qualified_positives: metric(&selected, |title| title.positive),
        qualified_no_read: metric(&selected, |title| title.no_read),
        unanswered_pairs: selected
            .iter()
            .map(|title| title.unanswered_pairs)
            .fold(Tally::default(), Tally::merge),
        contradicted_premise_pairs: selected
            .iter()
            .map(|title| title.contradicted_premise_pairs)
            .fold(Tally::default(), Tally::merge),
        singleton_title: singleton.map(|title| title.qualified),
        excluding_singleton_qualified_pairs: singleton.map(|_| {
            let rest: Vec<_> = selected
                .iter()
                .filter(|title| title.passages > 1)
                .copied()
                .collect();
            metric(&rest, |title| title.qualified)
        }),
    }
}

#[derive(Serialize)]
struct ModelGrade<'a> {
    arm: Arm,
    titles: Vec<TitleGrade<'a>>,
    two_answer: FamilyGrade,
    answerability: FamilyGrade,
    protected: Vec<CaseGrade<'a>>,
}

#[derive(Serialize)]
struct TrainControl<'a> {
    actual: Arm,
    paired_pairs: usize,
    unpaired_cases: usize,
    strict_actual_pairs: usize,
    strict_permuted_original_pairs: usize,
    strict_permuted_reversed_pairs: usize,
    zero_strict_pairs: usize,
    strict_reversal_pairs: Vec<[&'a str; 2]>,
}

fn train_control<'a>(
    training: &'a Training,
    roster: &CheckedRoster<'a>,
    models: &BTreeMap<Arm, CheckedModel<'_>>,
    key: Key,
) -> TrainControl<'a> {
    let actual = Arm {
        mixture: training.mixture,
        key,
        fit: Fit::Actual,
    };
    let mut control = TrainControl {
        actual,
        paired_pairs: 0,
        unpaired_cases: 0,
        strict_actual_pairs: 0,
        strict_permuted_original_pairs: 0,
        strict_permuted_reversed_pairs: 0,
        zero_strict_pairs: 0,
        strict_reversal_pairs: Vec::new(),
    };
    let cases = &roster.train[&training.mixture];
    let fits = |fit: Fit| &models[&Arm { fit, ..actual }].train;
    for source in &training.sources {
        let [left_opening, right_opening] = source.openings.as_slice() else {
            control.unpaired_cases += 1;
            continue;
        };
        let (a, b) = (&left_opening.cases[0], &right_opening.cases[0]);
        control.paired_pairs += 1;
        let (left, right) = (cases[a.id.as_str()], cases[b.id.as_str()]);
        let strict = |fit: Fit, swapped: bool| {
            let (left_label, right_label) = if swapped {
                (right.span(), left.span())
            } else {
                (left.span(), right.span())
            };
            strict_margin(left_label, left, fits(fit)[a.id.as_str()]) >= 1
                && strict_margin(right_label, right, fits(fit)[b.id.as_str()]) >= 1
        };
        let actual_strict = strict(Fit::Actual, false);
        let reversed = strict(Fit::Permuted, true);
        control.strict_actual_pairs += usize::from(actual_strict);
        control.strict_permuted_original_pairs += usize::from(strict(Fit::Permuted, false));
        control.strict_permuted_reversed_pairs += usize::from(reversed);
        control.zero_strict_pairs += usize::from(strict(Fit::Zero, false));
        if actual_strict && reversed {
            control.strict_reversal_pairs.push([&a.id, &b.id]);
        }
    }
    control
}

#[derive(Serialize)]
struct TitleComparison<'a> {
    title: &'a str,
    family: Family,
    vs_historical_geometry: PairedChanges,
    vs_natural_count: PairedChanges,
    vs_historical_count: PairedChanges,
}

#[derive(Serialize)]
struct Contrasts {
    natural_geometry_vs_historical_geometry: Option<Rate>,
    natural_geometry_vs_natural_count: Option<Rate>,
    natural_supervision_effect_within_count: Option<Rate>,
    geometric_key_effect_within_historical: Option<Rate>,
    factorial_difference_in_differences: Option<Rate>,
}

fn contrasts(field: impl Fn(Arm) -> Option<Rate>) -> Contrasts {
    let geometry = field(PROPOSED);
    let count = field(NATURAL_COUNT);
    let historical_geometry = field(HISTORICAL_GEOMETRY);
    let historical_count = field(HISTORICAL_COUNT);
    let delta = |a: Option<Rate>, b: Option<Rate>| Some(a?.minus(b?));
    let geometry_effect = delta(geometry, count);
    let historical_effect = delta(historical_geometry, historical_count);
    Contrasts {
        natural_geometry_vs_historical_geometry: delta(geometry, historical_geometry),
        natural_geometry_vs_natural_count: geometry_effect,
        natural_supervision_effect_within_count: delta(count, historical_count),
        geometric_key_effect_within_historical: historical_effect,
        factorial_difference_in_differences: delta(geometry_effect, historical_effect),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceStatus {
    InstrumentInconclusive,
    Negative,
    Pass,
}

#[derive(Serialize)]
pub struct Report<'a> {
    schema: &'static str,
    pub source_status: SourceStatus,
    pub roster_sha256: &'a str,
    pub evidence_sha256: &'a str,
    coverage_floors: bool,
    natural_geometry_protected_anchors: bool,
    train_reversal_and_zero_controls: bool,
    train_controls: Vec<TrainControl<'a>>,
    two_answer_10_points_over_both: bool,
    two_answer_titles_strictly_above_both: usize,
    two_answer_no_title_declines: bool,
    answerability_all_macros_noninferior: bool,
    answerability_no_title_declines: bool,
    two_answer_contrasts: Contrasts,
    answerability_pair_contrasts: Contrasts,
    answerability_positive_contrasts: Contrasts,
    answerability_no_read_contrasts: Contrasts,
    comparisons: Vec<TitleComparison<'a>>,
    arms: Vec<ModelGrade<'a>>,
    pub writer_grade_required: bool,
    pub locked_access_authorized: bool,
}

pub fn score<'a>(
    sealed: &'a Roster,
    checked: &CheckedRoster<'a>,
    models: &BTreeMap<Arm, CheckedModel<'_>>,
    roster_sha: &'a str,
    evidence_sha: &'a str,
) -> Report<'a> {
    let mut grades = BTreeMap::new();
    for (arm, model) in models {
        let titles: Vec<_> = sealed
            .titles
            .iter()
            .map(|row| title(row, checked, &model.validation))
            .collect();
        let protected = checked
            .protected
            .iter()
            .map(|(id, observation)| CaseGrade::from(*observation, model.protected[id]))
            .collect();
        grades.insert(
            *arm,
            ModelGrade {
                arm: *arm,
                protected,
                two_answer: family(&titles, Family::TwoAnswer),
                answerability: family(&titles, Family::Answerability),
                titles,
            },
        );
    }
    let mut controls = Vec::new();
    for training in &sealed.training {
        for key in [Key::Count, Key::Geometry] {
            controls.push(train_control(training, checked, models, key));
        }
    }
    let train_pass = controls
        .iter()
        .all(|c| !c.strict_reversal_pairs.is_empty() && c.zero_strict_pairs == 0);
    let proposed = &grades[&PROPOSED];
    let anchor_pass = proposed.protected.iter().all(|row| row.strict);
    let two = contrasts(|arm| grades[&arm].two_answer.qualified_pairs.macro_rate);
    let answer = contrasts(|arm| grades[&arm].answerability.qualified_pairs.macro_rate);
    let positive = contrasts(|arm| grades[&arm].answerability.qualified_positives.macro_rate);
    let no_read = contrasts(|arm| grades[&arm].answerability.qualified_no_read.macro_rate);
    let over_both = |comparison: &Contrasts, numerator, denominator| {
        [
            comparison.natural_geometry_vs_historical_geometry,
            comparison.natural_geometry_vs_natural_count,
        ]
        .into_iter()
        .all(|rate| rate.is_some_and(|rate| rate.at_least(numerator, denominator)))
    };
    let ten = over_both(&two, 1, 10);
    let answer_noninferior = [&answer, &positive, &no_read]
        .iter()
        .all(|metric| over_both(metric, 0, 1));
    let mut two_up = 0;
    let mut two_no_declines = true;
    let mut answer_no_declines = true;
    let mut comparisons = Vec::new();
    for (i, title) in proposed.titles.iter().enumerate() {
        let compare = |arm| {
            let other: &ModelGrade<'_> = &grades[&arm];
            let other = &other.titles[i];
            let mut changes = PairedChanges::default();
            for (ours, theirs) in title.rows.iter().zip(&other.rows) {
                if ours.qualified {
                    changes.record(ours.solved, theirs.solved);
                }
            }
            (other.qualified.wins, changes)
        };
        let (historical, vs_historical_geometry) = compare(HISTORICAL_GEOMETRY);
        let (natural, vs_natural_count) = compare(NATURAL_COUNT);
        let (_, vs_historical_count) = compare(HISTORICAL_COUNT);
        match title.family {
            Family::TwoAnswer => {
                two_up += usize::from(title.qualified.wins > historical.max(natural));
                two_no_declines &= title.qualified.wins >= historical.max(natural);
            }
            Family::Answerability => {
                answer_no_declines &= title.qualified.wins >= historical.max(natural)
            }
        }
        comparisons.push(TitleComparison {
            title: title.title,
            family: title.family,
            vs_historical_geometry,
            vs_natural_count,
            vs_historical_count,
        });
    }
    let source_status = if !checked.coverage || !train_pass {
        SourceStatus::InstrumentInconclusive
    } else if anchor_pass
        && ten
        && two_up >= 5
        && two_no_declines
        && answer_noninferior
        && answer_no_declines
    {
        SourceStatus::Pass
    } else {
        SourceStatus::Negative
    };
    Report {
        schema: "uor-r4.reciprocal-transfer-validation-source-report/1",
        source_status,
        roster_sha256: roster_sha,
        evidence_sha256: evidence_sha,
        coverage_floors: checked.coverage,
        natural_geometry_protected_anchors: anchor_pass,
        train_reversal_and_zero_controls: train_pass,
        train_controls: controls,
        two_answer_10_points_over_both: ten,
        two_answer_titles_strictly_above_both: two_up,
        two_answer_no_title_declines: two_no_declines,
        answerability_all_macros_noninferior: answer_noninferior,
        answerability_no_title_declines: answer_no_declines,
        two_answer_contrasts: two,
        answerability_pair_contrasts: answer,
        answerability_positive_contrasts: positive,
        answerability_no_read_contrasts: no_read,
        comparisons,
        arms: grades.into_values().collect(),
        writer_grade_required: true,
        locked_access_authorized: false,
    }
}

pub fn coverage(
    roster: &Roster,
    checked: &CheckedRoster<'_>,
    roster_sha256: &str,
) -> serde_json::Value {
    let titles: Vec<_> = roster.titles.iter().map(|title| {
        let selected = title.selected.len();
        let mut counts = BTreeMap::from([
            ("source_excluded", 0usize), ("capture_excluded", 0),
            ("rebind_excluded", 0), ("qualified", 0),
        ]);
        for pair in &title.selected {
            let stage = match pair.qualification {
                Qualification::SourceExcluded { .. } => "source_excluded",
                Qualification::CaptureExcluded { .. } => "capture_excluded",
                Qualification::RebindExcluded { .. } => "rebind_excluded",
                Qualification::Qualified { .. } => "qualified",
            };
            *counts.entry(stage).or_default() += 1;
        }
        serde_json::json!({ "title":title.name, "family":title.family, "mechanical":title.mechanical_candidates,
            "passages":title.passages, "selected":selected, "dispositions":counts })
    }).collect();
    serde_json::json!({
        "schema":"uor-r4.reciprocal-transfer-validation-coverage-report/1",
        "roster_sha256":roster_sha256,
        "coverage":if checked.coverage { "pass" } else { "instrument_inconclusive" },
        "titles":titles,
        "locked_access_authorized":false,
    })
}
