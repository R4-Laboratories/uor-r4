#![forbid(unsafe_code)]

mod native;

use native::{invalid, State, AXES, SIGNS, SLOTS, TAIL};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;

const TAIL_WORDS: usize = AXES - TAIL;
const COORDINATES: usize = SIGNS * SLOTS * TAIL_WORDS;
const INIT: usize = 57;
const PROPOSALS: usize = 54;
const PATHS: usize = 8;
const ROUNDS: usize = 6;

fn number<T: std::str::FromStr>(value: &str) -> io::Result<T> {
    value
        .parse()
        .map_err(|_| invalid(format!("invalid numerical field {value}")))
}

struct Geometry {
    bytes: Vec<u8>,
    rows: Vec<Row>,
}

struct Row {
    words: usize,
    sites: usize,
    transport: usize,
}

fn field(bytes: &[u8], at: usize) -> io::Result<usize> {
    let slice = bytes
        .get(at..at + 4)
        .ok_or_else(|| invalid("truncated packet field"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as usize)
}

impl Geometry {
    fn load(path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.get(..8) != Some(b"UORTXT01") {
            return Err(invalid("wrong ordered packet magic"));
        }
        let count = field(&bytes, 8)?;
        if !(2..=512).contains(&count) || count % 2 != 0 {
            return Err(invalid("invalid paired packet row count"));
        }
        let mut offset = 12;
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let words = field(&bytes, offset)?;
            let sites = field(&bytes, offset + 4)?;
            let triples = field(&bytes, offset + 8)?;
            if !(1..=128).contains(&words) || !(1..=128).contains(&sites) || triples > 1 << 22 {
                return Err(invalid("ordered packet row exceeds archive format bounds"));
            }
            let transport = offset + 12 + (words + sites) * 208;
            rows.push(Row {
                words,
                sites,
                transport,
            });
            offset = transport + words * sites * AXES * 2 + triples * 12;
        }
        if offset != bytes.len() {
            return Err(invalid("ordered packet does not read to its declared end"));
        }
        Ok(Self { bytes, rows })
    }

    fn point(&self, row: usize, word: usize, site: usize, axis: usize) -> io::Result<i16> {
        let location = self
            .rows
            .get(row)
            .ok_or_else(|| invalid("invalid objective row"))?;
        if word >= location.words || site >= location.sites {
            return Err(invalid(
                "objective transport address outside ordered packet",
            ));
        }
        let index = location.transport + ((word * location.sites + site) * AXES + axis) * 2;
        Ok(i16::from_le_bytes([
            self.bytes[index],
            self.bytes[index + 1],
        ]))
    }
}

struct Assignment {
    word: usize,
    site: usize,
    center: usize,
}

struct Term {
    row: usize,
    side: usize,
    local_debt: u32,
    local_worst: u32,
    joint_debt: u32,
    joint_worst: u32,
    focus: [Assignment; 2],
    opposite: [Assignment; 2],
}

fn terms(path: &Path, n: usize) -> io::Result<BTreeMap<String, Vec<Term>>> {
    let mut records = BTreeMap::<String, Vec<Term>>::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let [name, pair, selector, row, sign, governor, requested, rival, request_center, rival_center, local_debt, local_worst, joint_debt, joint_worst, ow0, os0, oc0, ow1, os1, oc1] =
            fields.as_slice()
        else {
            return Err(invalid("unexpected objective audit TSV"));
        };
        let batch = records.entry((*name).to_owned()).or_default();
        if batch.len() / 2 != number::<usize>(pair)?
            || *selector
                != if batch.len() % 2 == 0 {
                    "first"
                } else {
                    "later"
                }
        {
            return Err(invalid("objective rows out of paired order"));
        }
        let side = match *sign {
            "1" => 0,
            "-1" => 1,
            _ => return Err(invalid("invalid objective sign")),
        };
        let point = |word, site, center| -> io::Result<Assignment> {
            Ok(Assignment {
                word: number(word)?,
                site: number(site)?,
                center: number(center)?,
            })
        };
        batch.push(Term {
            row: number(row)?,
            side,
            local_debt: number(local_debt)?,
            local_worst: number(local_worst)?,
            joint_debt: number(joint_debt)?,
            joint_worst: number(joint_worst)?,
            focus: [
                point(governor, requested, request_center)?,
                point(governor, rival, rival_center)?,
            ],
            opposite: [point(ow0, os0, oc0)?, point(ow1, os1, oc1)?],
        });
    }
    for batch in records.values() {
        let physical: BTreeSet<_> = batch.iter().map(|row| row.row).collect();
        if physical.iter().copied().ne(0..n) || batch.len() != n {
            return Err(invalid(
                "objective rows lack exact physical packet coverage",
            ));
        }
        for pair in batch.chunks_exact(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if a.side != b.side
                || a.focus[0].site != b.focus[1].site
                || a.focus[1].site != b.focus[0].site
                || a.local_worst != a.local_debt.max(b.local_debt)
                || b.local_worst != a.local_worst
                || a.joint_worst != a.joint_debt.max(b.joint_debt)
                || b.joint_worst != a.joint_worst
            {
                return Err(invalid("objective pair identities or worst debts disagree"));
            }
        }
    }
    Ok(records)
}

fn initial_name(index: usize) -> String {
    match index {
        0 => "original".to_owned(),
        1 => "median".to_owned(),
        2 => "median-affirmative".to_owned(),
        3 => "median-excluded".to_owned(),
        4 => "blend-quarter".to_owned(),
        5 => "blend-half".to_owned(),
        6 => "blend-three-quarters".to_owned(),
        7..=38 => {
            let offset = index - 7;
            format!(
                "{}-slot-{}",
                if offset < 16 { "original" } else { "median" },
                offset % 16
            )
        }
        _ => {
            let offset = index - 39;
            let slot = offset % 9;
            format!(
                "{}-bilateral-{}",
                if offset < 9 { "original" } else { "median" },
                if slot == SLOTS {
                    "0-3".to_owned()
                } else {
                    slot.to_string()
                }
            )
        }
    }
}

fn initial(original: &State, median: &State, i: usize) -> io::Result<State> {
    let reverse = matches!(i, 1 | 23..=38 | 48..=56);
    let (base, donor) = if reverse {
        (median, original)
    } else {
        (original, median)
    };
    let mut state = base.copy();
    for side in 0..SIGNS {
        for slot in 0..state.active_slots(side) {
            match i {
                2..=3 if side == i - 2 => state.replace_slot(donor, side, slot)?,
                4..=6 => {
                    for axis in TAIL..AXES {
                        let a = i32::from(original.coordinate(side, slot, axis));
                        let b = i32::from(median.coordinate(side, slot, axis));
                        state.set_coordinate(
                            side,
                            slot,
                            axis,
                            (a + (b - a) * (i as i32 - 3) / 4) as i16,
                        );
                    }
                }
                7..=38 if (i - 7) % 16 == side * SLOTS + slot => {
                    state.replace_slot(donor, side, slot)?
                }
                39..=56
                    if (i - 39) % 9 == slot || ((i - 39) % 9 == SLOTS && matches!(slot, 0 | 3)) =>
                {
                    state.replace_slot(donor, side, slot)?;
                }
                _ => {}
            }
        }
    }
    Ok(state)
}

fn votes(
    geometry: &Geometry,
    terms: &[Term],
    current: &State,
) -> io::Result<[[i32; 6]; COORDINATES]> {
    let mut votes = [[0_i32; 6]; COORDINATES];
    for (index, values) in votes.iter_mut().enumerate() {
        let side = index / (SLOTS * TAIL_WORDS);
        let slot = index / TAIL_WORDS % SLOTS;
        let axis = TAIL + index % TAIL_WORDS;
        if slot >= current.active_slots(side) {
            continue;
        }
        let coordinate = current.coordinate(side, slot, axis);
        for term in terms {
            let direction = |point: &Assignment| -> io::Result<i32> {
                if point.center >= current.active_slots(side) {
                    return Err(invalid("vote selects an inactive center"));
                }
                if point.center != slot {
                    return Ok(0);
                }
                Ok(
                    (i32::from(geometry.point(term.row, point.word, point.site, axis)?)
                        - i32::from(coordinate))
                    .signum(),
                )
            };
            let mut local = 0;
            if term.side == side {
                local = direction(&term.focus[0])? - direction(&term.focus[1])?;
                if term.local_debt != 0 {
                    values[0] += local;
                    if term.local_debt == term.local_worst {
                        values[1] += local;
                    }
                }
            }
            if term.joint_debt != 0 && term.joint_debt == term.joint_worst {
                values[2] += if term.side == side {
                    local
                } else {
                    direction(&term.opposite[0])? - direction(&term.opposite[1])?
                };
            }
        }
        values[3] = values[1] + values[2];
        values[4 + side] = values[3];
    }
    Ok(votes)
}

fn advance(
    geometry: &Geometry,
    terms: &[Term],
    parent: &State,
    choice: usize,
) -> io::Result<State> {
    let weights = votes(geometry, terms, parent)?;
    let mut favorite_slots = [None; SIGNS];
    let mut favorite_axes = [None; SIGNS];
    for side in 0..SIGNS {
        let begin = side * SLOTS * TAIL_WORDS;
        let bank = &weights[begin..begin + SLOTS * TAIL_WORDS];
        let axis_weights = bank.iter().map(|directions| directions[3].abs());
        favorite_axes[side] = axis_weights
            .enumerate()
            .filter(|(_, weight)| *weight != 0)
            .max_by_key(|(index, weight)| (*weight, std::cmp::Reverse(*index)))
            .map(|(index, _)| begin + index);
        favorite_slots[side] = bank
            .chunks_exact(TAIL_WORDS)
            .map(|slot| slot.iter().map(|d| d[3].abs()).sum::<i32>())
            .enumerate()
            .filter(|(_, weight)| *weight != 0)
            .max_by_key(|(index, weight)| (*weight, std::cmp::Reverse(*index)))
            .map(|(index, _)| index);
    }

    let mut updated = parent.copy();
    for (index, weight) in weights.iter().enumerate() {
        let side = index / (SLOTS * TAIL_WORDS);
        let slot = index / TAIL_WORDS % SLOTS;
        let axis = TAIL + index % TAIL_WORDS;
        if slot >= parent.active_slots(side) {
            continue;
        }
        let step = match choice {
            0..=35 => weight[choice / 6].signum() * (1 << (choice % 6)),
            36..=41 if (choice - 36) / 3 == side && favorite_slots[side] == Some(slot) => {
                weight[3].signum() * (2 << (2 * ((choice - 36) % 3)))
            }
            42..=53 if (choice - 42) / 6 == side && favorite_axes[side] == Some(index) => {
                let step = 1 << (2 * (((choice - 42) / 2) % 3));
                if choice % 2 == 0 {
                    -step
                } else {
                    step
                }
            }
            _ => 0,
        };
        let coordinate = i32::from(parent.coordinate(side, slot, axis)) + step;
        updated.set_coordinate(
            side,
            slot,
            axis,
            coordinate.clamp(i16::MIN.into(), i16::MAX.into()) as i16,
        );
    }
    Ok(updated)
}

fn candidates(dir: &Path, original: &State) -> io::Result<(Vec<String>, BTreeMap<String, State>)> {
    let mut order = Vec::new();
    let mut candidates = BTreeMap::new();
    for line in std::fs::read_to_string(dir.join("candidates.tsv"))?.lines() {
        let (name, filename) = line
            .split_once('\t')
            .ok_or_else(|| invalid("invalid candidate manifest"))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
            || filename != format!("{name}.state")
        {
            return Err(invalid("candidate path and basename disagree"));
        }
        let state = State::load(&dir.join(filename))?;
        original.verify_immutable(&state)?;
        if !dir.join(format!("{name}.native.jsonl")).is_file()
            || candidates.insert(name.to_owned(), state).is_some()
        {
            return Err(invalid("candidate is repeated or lacks native replay"));
        }
        order.push(name.to_owned());
    }
    let disk: BTreeSet<_> = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.file_name()))
        .collect::<io::Result<_>>()?;
    for file in disk {
        let file = file.to_string_lossy();
        if let Some(state) = file.strip_suffix(".state") {
            if !candidates.contains_key(state) {
                return Err(invalid("unlisted native state payload"));
            }
        }
    }
    Ok((order, candidates))
}

fn append_name(order: &mut Vec<String>, name: &str) {
    if !order.iter().any(|current| current == name) {
        order.push(name.to_owned());
    }
}

struct Replay<'a> {
    original: &'a State,
    median: &'a State,
    geometry: &'a Geometry,
    states: &'a BTreeMap<String, State>,
    objectives: &'a BTreeMap<String, Vec<Term>>,
}

impl Replay<'_> {
    fn state(&self, name: &str) -> io::Result<&State> {
        self.states
            .get(name)
            .ok_or_else(|| invalid(format!("missing state {name}")))
    }

    fn verify(&self, schedule: &Path, budget: usize, manifest_order: &[String]) -> io::Result<()> {
        let mut aliases = BTreeMap::<String, String>::new();
        let mut observed_order = Vec::new();
        let mut initials = 0;
        let mut evaluated = INIT;
        let mut accepted = 0;
        let mut stopped = false;
        let mut track = None;
        let mut round = 0;
        let mut expected_parent = String::new();
        let mut path_count = 0;
        for line in std::fs::read_to_string(schedule)?.lines() {
            let fields: Vec<_> = line.split('\t').collect();
            match fields.as_slice() {
                ["initial", index, name, artifact, counted, movement, changed] => {
                    let i = number::<usize>(index)?;
                    if i != initials
                        || i >= INIT
                        || number::<usize>(counted)? != INIT
                        || *name != initial_name(i)
                    {
                        return Err(invalid("initial candidate schedule drift"));
                    }
                    let expected = initial(self.original, self.median, i)?;
                    let named = self.state(artifact)?;
                    let (words, distance) = self.original.tail_distance(named);
                    let reused = observed_order.iter().find(|previous| {
                        self.states
                            .get(*previous)
                            .is_some_and(|seen| named.same(seen))
                    });
                    let correct_name = if i < 2 {
                        *artifact == *name
                    } else {
                        match reused {
                            Some(prev) => *artifact == prev,
                            None => *artifact == *name,
                        }
                    };
                    if !correct_name
                        || !expected.same(named)
                        || words != number(changed)?
                        || distance != number(movement)?
                    {
                        return Err(invalid(format!(
                            "incorrect initial state or alias at {name}"
                        )));
                    }
                    aliases.insert((*name).to_owned(), (*artifact).to_owned());
                    append_name(&mut observed_order, artifact);
                    initials += 1;
                }
                ["fit", path, step, priority, parent, proposal, status, artifact, counted, before_movement, before_changed, after_movement, after_changed] =>
                {
                    if initials != INIT {
                        return Err(invalid(
                            "fit path started before initial placements completed",
                        ));
                    }
                    let index = number::<usize>(path)?;
                    let step = number::<usize>(step)?;
                    let choice = number::<usize>(proposal)?;
                    let expected_priority = [
                        "balanced",
                        "balanced",
                        "affirmative",
                        "excluded",
                        "local",
                        "joint",
                        "balanced",
                        "local",
                    ];
                    if index >= PATHS
                        || choice >= PROPOSALS
                        || *priority != expected_priority[index]
                        || !matches!(*status, "true" | "false")
                    {
                        return Err(invalid("unregistered fit branch"));
                    }
                    if track != Some(index) {
                        let exact_start = [
                            "original",
                            "median",
                            "median-affirmative",
                            "median-excluded",
                            "blend-half",
                        ];
                        if index != path_count
                            || track.is_some() && !stopped && round != ROUNDS
                            || index < exact_start.len() && *parent != exact_start[index]
                            || !aliases.contains_key(*parent)
                        {
                            return Err(invalid("fit path start drift"));
                        }
                        track = Some(index);
                        path_count += 1;
                        round = 0;
                        stopped = false;
                        expected_parent = (*parent).to_owned();
                    }
                    if stopped || step != round + 1 || step > ROUNDS || *parent != expected_parent {
                        return Err(invalid("fit ancestry or round order drift"));
                    }
                    round = step;
                    evaluated += PROPOSALS;
                    if number::<usize>(counted)? != evaluated {
                        return Err(invalid("fit proposal counter drift"));
                    }
                    let canonical = aliases.get(*parent).map_or(*parent, String::as_str);
                    let state = self.state(canonical)?;
                    let terms = self
                        .objectives
                        .get(canonical)
                        .ok_or_else(|| invalid("missing parent objective replay"))?;
                    let updated = advance(self.geometry, terms, state, choice)?;
                    let (before_words, before_distance) = self.original.tail_distance(state);
                    let (after_words, after_distance) = self.original.tail_distance(&updated);
                    if before_words != number(before_changed)?
                        || before_distance != number(before_movement)?
                        || after_words != number(after_changed)?
                        || after_distance != number(after_movement)?
                    {
                        return Err(invalid(format!(
                            "proposal displacement receipt mismatch at path {path}, round {step}"
                        )));
                    }
                    if *status == "true" {
                        let actual = self.state(artifact)?;
                        let reused = observed_order.iter().find(|previous| {
                            self.states
                                .get(*previous)
                                .is_some_and(|seen| actual.same(seen))
                        });
                        let canonical_name = format!("fit-{index}-{step}");
                        let correct_name = match reused {
                            Some(prev) => *artifact == prev,
                            None => *artifact == canonical_name,
                        };
                        if !correct_name || !updated.same(actual) || state.same(actual) {
                            return Err(invalid(format!("learned state failed exact proposal replay at path {path}, round {step}")));
                        }
                        append_name(&mut observed_order, artifact);
                        expected_parent = (*artifact).to_owned();
                        accepted += 1;
                    } else {
                        stopped = true;
                        if *artifact != "-" {
                            return Err(invalid("rejected proposal unexpectedly exported"));
                        }
                    }
                }
                _ => return Err(invalid("invalid schedule audit TSV")),
            }
        }
        if initials != INIT
            || evaluated != budget
            || path_count != PATHS
            || manifest_order != observed_order
            || self.states.keys().ne(self.objectives.keys())
            || !stopped && round != ROUNDS
        {
            return Err(invalid("incomplete schedule or output census"));
        }
        println!(
            r#"{{"pass":true,"packet_rows":{},"immutable_states":{},"replayed_initial":{initials},"accepted_updates_replayed":{accepted},"tested_proposals":{evaluated},"paths":{path_count}}}"#,
            self.geometry.rows.len(),
            manifest_order.len()
        );
        Ok(())
    }
}

fn verify(
    original: &Path,
    median: &Path,
    packet: &Path,
    dir: &Path,
    audits: &Path,
    budget: usize,
) -> io::Result<()> {
    let original = State::load(original)?;
    let median = State::load(median)?;
    original.verify_immutable(&median)?;
    let geometry = Geometry::load(packet)?;
    let (manifest_order, states) = candidates(dir, &original)?;
    let objectives = terms(&audits.join("objectives.tsv"), geometry.rows.len())?;
    Replay {
        original: &original,
        median: &median,
        geometry: &geometry,
        states: &states,
        objectives: &objectives,
    }
    .verify(&audits.join("schedule.tsv"), budget, &manifest_order)
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [original, median, packet, dir, audits, budget] => verify(
            Path::new(original),
            Path::new(median),
            Path::new(packet),
            Path::new(dir),
            Path::new(audits),
            number(budget)?,
        ),
        _ => Err(invalid(
            "expected ORIGINAL MEDIAN TRAIN_PACKET FIT_OUTPUT AUDIT_TSVS EVALUATED",
        )),
    }
}
