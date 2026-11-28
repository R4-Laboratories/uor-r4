// The host reference is limited to the declared synthetic finite cases below.
use crate::{
    gpu::{Decision, Device, Fit, FitEvent},
    input::{self, Geometry, Row, ASSISTANT, USER},
    require, write_json, write_jsonl, Result, EPOCHS, MARGIN, STATE_WORDS, WEIGHT_LIMIT,
};
use serde_json::json;
use std::{fs, path::Path};

fn row(case_id: &str, body: &[u32], target: Option<u16>) -> Row {
    Row {
        case_id: case_id.to_owned(),
        prefix: [&USER[..], body, &ASSISTANT].concat(),
        user_bounds: [USER.len() as u16, (USER.len() + body.len()) as u16],
        target,
    }
}

fn expected(rows: &[Row], sites: &[Decision], initial: &[i16]) -> Fit {
    const NO_READ: usize = STATE_WORDS - 1;
    let mut weights = initial.to_vec();
    let mut events = Vec::new();
    let mut snapshots = Vec::new();
    for epoch in 1..=EPOCHS {
        for (row, decision) in rows.iter().zip(sites) {
            let sites = &decision.expanded_frozen.candidates;
            let score = |position: u16| -> i32 {
                sites
                    .iter()
                    .find(|site| site.pos == position)
                    .into_iter()
                    .flat_map(|site| site.features)
                    .map(|word| i32::from(weights[usize::from(word)]))
                    .sum()
            };
            let threshold = weights[NO_READ];
            let no_read = (None, i32::from(threshold));
            let desired_score = row.target.map_or(no_read.1, score);
            let best_other = sites
                .iter()
                .filter(|candidate| Some(candidate.pos) != row.target)
                .map(|candidate| (Some(candidate.pos), score(candidate.pos)))
                .max_by_key(|&(position, value)| (value, std::cmp::Reverse(position)));
            let competitor = if row.target.is_some() {
                best_other
                    .filter(|value| value.1 > no_read.1)
                    .or(Some(no_read))
            } else {
                best_other
            };
            let margin = competitor.map(|(_, value)| desired_score - value);
            let update = margin.is_some_and(|delta| delta < MARGIN);
            if update {
                let mut net = [0i16; STATE_WORDS];
                let mut add = |position, value| {
                    if let Some(site) = sites.iter().find(|site| Some(site.pos) == position) {
                        for word in site.features {
                            net[usize::from(word)] += value;
                        }
                    } else {
                        net[NO_READ] += value;
                    }
                };
                add(row.target, 1);
                add(competitor.and_then(|value| value.0), -1);
                for (value, delta) in weights.iter_mut().zip(net) {
                    *value = (*value + delta).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
                }
            }
            events.push(FitEvent {
                epoch,
                case_id: row.case_id.clone(),
                gold_pos: row.target,
                competing_pos: competitor.and_then(|value| value.0),
                desired_score,
                competitor_score: competitor.map(|value| value.1),
                no_read_score: threshold,
                margin,
                update,
            });
        }
        snapshots.push(weights.clone());
    }
    Fit { snapshots, events }
}

pub fn run(image: &Path, geometry: Geometry, out: &Path) -> Result<()> {
    let zero = vec![0i16; STATE_WORDS];
    let nonlexical = geometry
        .classes
        .iter()
        .enumerate()
        .skip(3)
        .find(|&(_, &class)| !input::eligible(class))
        .ok_or("no nonlexical fixture ID in the canonical table")?
        .0 as u32;
    let rows = [
        row("01-one-positive", &[1042], Some(4)),
        row("02-two-candidates", &[1042, 201, 201, 671], Some(7)),
        row("03-no-read", &[1042, 201, 201, 671], None),
        row("04-empty-NoRead", &[nonlexical], None),
        row(
            "05-embedded-role",
            &[1042, 201, 55, 2728, 28, 201, 201, 671],
            None,
        ),
        row("06-query-suffix", &[1042; 44], None),
        row("07-duplicate-original-id", &[1042, 201, 201, 1042], None),
    ];
    for row in &rows {
        row.validate(&geometry.classes)?;
    }
    fs::create_dir(out)?;
    let gpu = Device::open(image, &geometry)?;
    let tape = rows.iter().collect::<Vec<_>>();
    let sites = gpu.score(&tape, &zero, &zero, &geometry.classes)?;
    require(
        sites[0].old_framed.fact_bounds.is_none()
            && sites[0].expanded_frozen.best_ties == 1
            && sites[0].expanded_frozen.winner_pos.is_none()
            && sites[1].old_framed.fact_bounds == Some([4, 5])
            && sites[1].old_framed.query_bounds == Some([7, 8])
            && sites[1].expanded_frozen.fact_bounds == Some([4, 8])
            && sites[1].expanded_frozen.best_ties == 2
            && sites[1].expanded_frozen.winner_pos.is_none()
            && sites[3].expanded_frozen.candidates.is_empty()
            && sites[4].old_framed.fact_bounds.is_none()
            && sites[4].expanded_frozen.fact_bounds == Some([4, 12])
            && sites[5].expanded_frozen.query_bounds == Some([16, 48]),
        "unexpected finite scope, threshold tie, or duplicate candidate",
    )?;
    let one_row = &rows[..1];
    let one = gpu.fit(one_row, &zero)?;
    require(
        one == expected(one_row, &sites[..1], &zero)
            && one.events[0].update
            && one.events[1].margin == Some(17)
            && one.events[1..].iter().all(|event| !event.update),
        "single positive should update once against NoRead",
    )?;
    let trained = gpu.score(&tape[..1], &zero, &one.snapshots[0], &geometry.classes)?;
    require(
        trained[0].expanded_frozen.winner_pos.is_none()
            && trained[0].expanded_fitted.winner_pos == Some(4),
        "the finite update must turn a feasible abstention into a source read",
    )?;
    let duplicate = &sites[6].expanded_frozen.candidates;
    require(
        duplicate.len() == 2
            && duplicate[0].id == duplicate[1].id
            && duplicate[0].pos == 4
            && duplicate[1].pos == 7,
        "finite same-ID addresses differ from their original positions",
    )?;
    let distinguishing = duplicate[0]
        .features
        .iter()
        .zip(duplicate[1].features)
        .find_map(|(a, b)| (*a != b).then_some(usize::from(b)))
        .ok_or("same-ID occurrences have no distinguishing feature")?;
    let mut choose_later = zero.clone();
    choose_later[distinguishing] = 1;
    let read = gpu.score(&tape[6..], &zero, &choose_later, &geometry.classes)?;
    require(
        read[0].expanded_frozen.winner_pos.is_none()
            && read[0].expanded_frozen.best_ties == 2
            && read[0].expanded_fitted.winner_pos == Some(7),
        "the finite read must distinguish the selected later same-ID address",
    )?;
    let two_rows = &rows[1..2];
    let tie = gpu.fit(two_rows, &zero)?;
    require(
        tie == expected(two_rows, &sites[1..2], &zero)
            && tie.events[0].competing_pos.is_none()
            && tie.events[0].competitor_score == Some(0),
        "exact positive competitor tie should favor NoRead",
    )?;
    let no_read = gpu.fit(&rows[2..4], &zero)?;
    require(
        no_read == expected(&rows[2..4], &sites[2..4], &zero)
            && no_read.events[0].competing_pos == Some(4)
            && no_read.events[0].update
            && no_read.events[1].margin.is_none()
            && !no_read.events[1].update,
        "negative tie should update against earliest and empty negative should do nothing",
    )?;
    let sharing = &sites[1].expanded_frozen.candidates;
    let first = sharing.first().ok_or("missing earliest finite candidate")?;
    let second = sharing.last().ok_or("missing latest finite candidate")?;
    let mut overlap = None;
    let mut distinct = None;
    for (left, right) in first.features.into_iter().zip(second.features) {
        if left == right {
            overlap.get_or_insert(left);
        } else {
            distinct.get_or_insert([left, right]);
        }
    }
    let common = usize::from(overlap.ok_or("finite candidates need a shared word")?);
    let [rival, gold] = distinct
        .ok_or("finite candidates need distinct words")?
        .map(usize::from);
    let mut limit = zero.clone();
    limit[STATE_WORDS - 1] = -WEIGHT_LIMIT;
    limit[common] = -WEIGHT_LIMIT;
    limit[rival] = WEIGHT_LIMIT;
    limit[gold] = WEIGHT_LIMIT;
    let clamp = gpu.fit(two_rows, &limit)?;
    require(
        clamp == expected(two_rows, &sites[1..2], &limit)
            && clamp.events[0].competing_pos == Some(4)
            && clamp.snapshots[0][common] == -WEIGHT_LIMIT
            && clamp.snapshots[0][gold] == WEIGHT_LIMIT
            && clamp.snapshots[0][rival] < WEIGHT_LIMIT,
        "signed update should cancel shared features before clamping once",
    )?;
    let chronological = gpu.fit(&rows, &zero)?;
    require(
        chronological == expected(&rows, &sites, &zero)
            && chronological == gpu.fit(&rows, &zero)?,
        "finite native fit and independent chronological i16 readback disagree",
    )?;
    let digest = write_jsonl(&out.join("decisions.jsonl"), &sites)?;
    let events = write_jsonl(&out.join("fit-events.jsonl"), &chronological.events)?;
    let selection = write_jsonl(&out.join("selection.jsonl"), trained.iter().chain(&read))?;
    write_json(
        &out.join("finite.json"),
        &json!({
            "schema": "uor-r4.whole-turn-finite/1",
            "gpu": gpu.name,
            "cubin_sha256": gpu.image_sha,
            "geometry": Geometry::identity(),
            "decisions_sha256": digest,
            "fit_events_sha256": events,
            "selection_sha256": selection,
            "tests": [
                "unframed single candidate", "framed paragraph and expanded scope",
                "strict abstention at equal scores", "nested markers remain user content",
                "32-ID query suffix", "positive comparison favors NoRead on tie",
                "negative comparison favors earliest position on tie",
                "empty NoRead prompt", "overlap cancels and i16 is clamped once",
                "four chronological epochs with exact repeat fit",
                "finite update turns abstention into an actual source read",
                "same-ID exact tie abstains and a distinguishing feature selects the later address"
            ],
            "pass": true,
        }),
    )
}
