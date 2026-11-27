//! Aggregate already-measured device rows. A token's analytical full rank and its
//! optional deployed rank are kept separate throughout.
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

type Rows<T> = BTreeMap<(String, u8), T>;
type Result<T> = std::result::Result<T, String>;

#[derive(Deserialize)]
struct Panel {
    pair_id: String,
    group_id: String,
    side: u8,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
    raw_target: u32,
    lexical_offset: usize,
    leading_whitespace: Vec<u32>,
    space_token: u32,
}

#[derive(Deserialize, Serialize, PartialEq)]
struct Terms {
    c_q: u32,
    c_qr: u32,
    c_r: u32,
    c_qc: u32,
    c_rr: u32,
    pointer_bonus: u32,
}
impl Terms {
    fn weighted(&self) -> [u64; 6] {
        [
            u64::from(self.c_q),
            u64::from(self.c_qr) << 6,
            u64::from(self.c_r) << 1,
            u64::from(self.c_qc) << 5,
            u64::from(self.c_rr) << 3,
            u64::from(self.pointer_bonus),
        ]
    }
}
#[derive(Deserialize, Serialize, PartialEq)]
struct Probe {
    id: u32,
    mass: u64,
    pool_member: bool,
    gate_rank: Option<u32>,
    gate_full_rank: u32,
    prior_count: u32,
    prior_rank: u32,
    writer_admitted: bool,
    writer_rank: Option<u32>,
    gate_score_terms: Terms,
}
impl Probe {
    fn eligible_mass(&self) -> u64 {
        if self.pool_member {
            self.mass
        } else {
            0
        }
    }
}

#[derive(Deserialize)]
struct Probes {
    gold: Probe,
    rival: Probe,
    space: Probe,
}

#[derive(Deserialize)]
struct Trace {
    pair_id: String,
    side: u8,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    artifact_sha256: String,
    read_vector: Vec<i32>,
    read_code: u32,
    recurrence_code: u32,
    recurrent_frames: Vec<u32>,
    recurrent_vector: Vec<i32>,
    #[serde(alias = "recurrent_sum", alias = "recurrent_raw_sum")]
    recurrence_sum: Vec<i32>,
    writer_vector: Vec<i32>,
    admitted_cells: Vec<u32>,
    source_position: Value,
    source_token: Value,
    gate_selected: u32,
    emitted: u32,
    prior_selected: u32,
    top8: Vec<u32>,
    probes: Probes,
    query_token: u32,
    routes: Vec<Value>,
    duplicate_replay: bool,
}

fn lines<T: DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path).map_err(|e| format!("{path}: {e}"))?)
        .lines()
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(&l.map_err(|e| e.to_string())?)
                .map_err(|e| format!("{path} line {}: {e}", i + 1))
        })
        .collect()
}

fn indexed<T>(list: Vec<T>, key: impl Fn(&T) -> (String, u8)) -> Result<Rows<T>> {
    let mut rows = BTreeMap::new();
    for row in list {
        let k = key(&row);
        if k.0.is_empty() || k.1 > 1 || rows.insert(k, row).is_some() {
            return Err("empty, invalid or duplicate row identity".into());
        }
    }
    Ok(rows)
}

#[derive(Default)]
struct Acc {
    counts: BTreeMap<&'static str, usize>,
    ranks: BTreeMap<&'static str, Vec<u32>>,
    hist: BTreeMap<&'static str, BTreeMap<u32, usize>>,
    term_mass: BTreeMap<&'static str, [u64; 6]>,
}
impl Acc {
    fn count(&mut self, name: &'static str, yes: bool) {
        *self.counts.entry(name).or_default() += usize::from(yes);
    }
    fn rank(&mut self, name: &'static str, n: Option<u32>) {
        let values = self.ranks.entry(name).or_default();
        if let Some(n) = n {
            values.push(n);
        }
    }
    fn hit(&mut self, name: &'static str, id: u32) {
        *self.hist.entry(name).or_default().entry(id).or_default() += 1;
    }
    fn finish(self) -> Value {
        let ranks: BTreeMap<_,_> = self.ranks.into_iter().map(|(k, mut v)| {
            v.sort_unstable();
            let quantile = |p: usize| if v.is_empty(){None}else{Some(v[(p*v.len()).div_ceil(100)-1])};
            (k,json!({"n":v.len(),"min":v.first(),"q25":quantile(25),"median":quantile(50),"q75":quantile(75),"max":v.last()}))
        }).collect();
        json!({"counts":self.counts,"ranks":ranks,"token_histograms":self.hist,
            "weighted_terms_for_eligible_candidates":self.term_mass,
            "term_order":["last_token","last_token_and_read","read","last_token_and_recurrence","read_and_recurrence","successor"]})
    }
}

fn check_probe(p: &Probe, row: &Trace) -> Result<()> {
    if p.gate_full_rank == 0
        || p.gate_full_rank > 4096
        || p.prior_rank == 0
        || p.prior_rank > 4096
        || p.writer_admitted != p.writer_rank.is_some()
        || p.mass != p.gate_score_terms.weighted().iter().sum::<u64>()
        || (p.writer_rank == Some(1)) != (row.emitted == p.id)
        || (p.gate_rank == Some(1)) != (row.gate_selected == p.id)
        || (p.gate_full_rank == 1) != (row.gate_selected == p.id)
        || p.writer_rank
            .is_some_and(|r| r == 0 || r > 4096 || (r == 1) != (row.emitted == p.id))
        || p.gate_rank
            .is_some_and(|r| r == 0 || r > 4096 || (r == 1) != (row.gate_selected == p.id))
        || (p.prior_rank == 1) != (p.id == row.prior_selected)
    {
        return Err(format!(
            "rank inconsistency for {} side {} token {}",
            row.pair_id, row.side, p.id
        ));
    }
    Ok(())
}
fn check(p: &Panel, r: &Trace, other: &Panel, sha: &str) -> Result<()> {
    if p.pair_id != r.pair_id
        || p.side != r.side
        || p.cohort != r.cohort
        || p.cohort != other.cohort
        || p.group_id != other.group_id
        || p.target == other.target
        || p.target == 223
        || p.space_token != 223
        || p.raw_target != p.target
        || p.lexical_offset != 0
        || !p.leading_whitespace.is_empty()
        || p.prefix != r.prefix
        || r.target != p.target
        || r.rival != other.target
        || r.probes.gold.id != p.target
        || r.probes.rival.id != other.target
        || r.probes.space.id != 223
        || r.artifact_sha256 != sha
        || r.top8.first() != Some(&r.emitted)
        || r.top8.len() != 8
        || r.prefix.last() != Some(&r.query_token)
        || p.prefix.last() != other.prefix.last()
        || p.prefix == other.prefix
        || p.prefix.len() > 192
        || r.read_vector.len() != 32
        || r.recurrent_vector.len() != 32
        || r.recurrence_sum.len() != 32
        || r.writer_vector.len() != 32
        || r.recurrent_frames.len() != 4
        || r.admitted_cells.is_empty()
        || r.routes.is_empty()
        || !r.duplicate_replay
        || (p.cohort == "controlled" && p.prefix.len() != other.prefix.len())
    {
        return Err(format!(
            "panel, state or checkpoint mismatch for {} side {}",
            p.pair_id, p.side
        ));
    }
    for pr in [&r.probes.gold, &r.probes.rival, &r.probes.space] {
        check_probe(pr, r)?;
    }
    Ok(())
}

fn tally_row(a: &mut Acc, r: &Trace, other: &Trace) {
    let g = &r.probes.gold;
    let space = &r.probes.space;
    let gold = g.eligible_mass();
    let sm = space.eligible_mass();
    a.count("rows", true);
    for (name, probe) in [("gold", g), ("rival", &r.probes.rival), ("space", space)] {
        let sum = a.term_mass.entry(name).or_default();
        if probe.pool_member {
            for (s, v) in sum.iter_mut().zip(probe.gate_score_terms.weighted()) {
                *s += v;
            }
        }
    }
    for (name, yes) in [
        ("gold_in_pool", g.pool_member),
        ("gold_positive_eligible", gold > 0),
        ("gold_zero_eligible", gold == 0),
        ("gold_raw_positive_excluded", g.mass > 0 && !g.pool_member),
        ("gold_positive_above_space_mass", gold > sm),
        ("gold_tied_space_mass", gold == sm),
        ("gold_space_both_unsupported", gold == 0 && sm == 0),
        ("gold_space_positive_mass_tie", gold > 0 && gold == sm),
        (
            "gold_no_support_under_either_question",
            gold == 0 && other.probes.rival.eligible_mass() == 0,
        ),
        (
            "rank_moves_without_gold_mass_change",
            gold == other.probes.rival.eligible_mass()
                && g.gate_full_rank != other.probes.rival.gate_full_rank,
        ),
        ("gold_below_space_mass", gold < sm),
        (
            "gold_above_space_full_gate_rank",
            g.gate_full_rank < space.gate_full_rank,
        ),
        ("gold_above_space_prior", g.prior_rank < space.prior_rank),
        ("gold_improved_vs_prior", g.gate_full_rank < g.prior_rank),
        ("gold_equal_to_prior_rank", g.gate_full_rank == g.prior_rank),
        ("gold_worse_than_prior", g.gate_full_rank > g.prior_rank),
        (
            "gold_improved_vs_other_question",
            g.gate_full_rank < other.probes.rival.gate_full_rank,
        ),
        (
            "gold_unchanged_vs_other_question",
            g.gate_full_rank == other.probes.rival.gate_full_rank,
        ),
        (
            "gold_worse_vs_other_question",
            g.gate_full_rank > other.probes.rival.gate_full_rank,
        ),
        (
            "gold_effective_mass_better_own_question",
            gold > other.probes.rival.eligible_mass(),
        ),
        (
            "gold_effective_mass_equal_own_question",
            gold == other.probes.rival.eligible_mass(),
        ),
        (
            "gold_effective_mass_worse_own_question",
            gold < other.probes.rival.eligible_mass(),
        ),
        ("gate_selects_gold", r.gate_selected == g.id),
        ("gate_selects_space", r.gate_selected == 223),
        (
            "fixed_query_alone_strictly_favors_space_over_gold",
            space.gate_score_terms.c_q > g.gate_score_terms.c_q,
        ),
        (
            "context_adds_more_mass_to_space_than_gold",
            sm.saturating_sub(if space.pool_member {
                u64::from(space.gate_score_terms.c_q)
            } else {
                0
            }) > gold.saturating_sub(if g.pool_member {
                u64::from(g.gate_score_terms.c_q)
            } else {
                0
            }),
        ),
        ("gate_selects_prior", r.gate_selected == r.prior_selected),
        ("writer_emits_gold", r.emitted == g.id),
        ("writer_emits_space", r.emitted == 223),
        ("writer_emits_prior", r.emitted == r.prior_selected),
        ("writer_keeps_gate", r.emitted == r.gate_selected),
        (
            "writer_changes_correct_gate",
            r.gate_selected == g.id && r.emitted != g.id,
        ),
        ("writer_admits_gold", g.writer_admitted),
        ("writer_admits_space", space.writer_admitted),
        (
            "gold_native_writer_rank_above_admitted_space",
            g.writer_rank
                .zip(space.writer_rank)
                .is_some_and(|(g, s)| g < s),
        ),
        ("writer_includes_gold_in_top8", r.top8.contains(&g.id)),
        (
            "gate_includes_gold_in_top8",
            g.gate_rank.is_some_and(|n| n <= 8),
        ),
    ] {
        a.count(name, yes);
    }
    for (name, val) in [
        ("gold_gate_full", Some(g.gate_full_rank)),
        ("gold_gate_native", g.gate_rank),
        ("gold_prior", Some(g.prior_rank)),
        ("gold_writer_native", g.writer_rank),
        ("space_gate_full", Some(space.gate_full_rank)),
        ("space_gate_native", space.gate_rank),
        ("space_prior", Some(space.prior_rank)),
        ("space_writer_native", space.writer_rank),
    ] {
        a.rank(name, val);
    }
    for (name, val) in [
        ("gate", r.gate_selected),
        ("writer", r.emitted),
        ("prior", r.prior_selected),
        ("target", g.id),
        ("query_token", r.query_token),
    ] {
        a.hit(name, val);
    }
}

#[derive(Clone, Copy)]
struct PairOutcome {
    strict: bool,
    full: bool,
    writer: bool,
}

fn tally_pair(a: &mut Acc, l: &Trace, r: &Trace) -> PairOutcome {
    let read = l.read_code != r.read_code;
    let rec = l.recurrence_code != r.recurrence_code;
    let sg = |t: &Trace| {
        t.probes
            .gold
            .eligible_mass()
            .cmp(&t.probes.rival.eligible_mass())
    };
    let (left, right) = (sg(l), sg(r));
    let strict = left.is_gt() && right.is_gt();
    let full = [l, r]
        .iter()
        .all(|t| t.probes.gold.gate_full_rank < t.probes.rival.gate_full_rank);
    let writer = [l, r].iter().all(|t| t.emitted == t.target);
    for (name, v) in [
        ("pairs", true),
        ("read_changes", read),
        ("raw_read_changes", l.read_vector != r.read_vector),
        ("active_route_records_change", l.routes != r.routes),
        ("recurrence_changes", rec),
        (
            "terminal_frames_change",
            l.recurrent_frames != r.recurrent_frames,
        ),
        (
            "normalized_recurrent_vector_changes",
            l.recurrent_vector != r.recurrent_vector,
        ),
        (
            "raw_recurrence_sum_changes",
            l.recurrence_sum != r.recurrence_sum,
        ),
        ("both_codes_change", read && rec),
        ("read_only_changes", read && !rec),
        ("recurrence_only_changes", !read && rec),
        ("neither_code_changes", !read && !rec),
        (
            "successor_address_changes",
            l.source_position != r.source_position,
        ),
        ("successor_id_changes", l.source_token != r.source_token),
        (
            "any_gate_input_changes",
            read || rec || l.source_token != r.source_token,
        ),
        ("gate_winner_changes", l.gate_selected != r.gate_selected),
        ("writer_emission_changes", l.emitted != r.emitted),
        (
            "post_gate_vector_changes",
            l.writer_vector != r.writer_vector,
        ),
        (
            "admitted_cells_change",
            l.admitted_cells != r.admitted_cells,
        ),
        ("both_gold_strict_mass_wins", strict),
        (
            "both_crossed_rivals_strictly_win",
            left.is_lt() && right.is_lt(),
        ),
        ("mass_tie_both_legs", left.is_eq() && right.is_eq()),
        ("mass_tie_at_least_one_leg", left.is_eq() || right.is_eq()),
        ("both_gold_full_rank_wins", full),
        (
            "both_gate_winners_correct",
            l.gate_selected == l.target && r.gate_selected == r.target,
        ),
        ("both_writer_emissions_correct", writer),
        (
            "both_priors_prefer_requested",
            l.probes.gold.prior_rank < l.probes.rival.prior_rank
                && r.probes.gold.prior_rank < r.probes.rival.prior_rank,
        ),
    ] {
        a.count(name, v);
    }
    tally_row(a, l, r);
    tally_row(a, r, l);
    PairOutcome {
        strict,
        full,
        writer,
    }
}

fn example(row: &Trace) -> Value {
    json!({"target":row.target,"rival":row.rival,"query_token":row.query_token,"read_code":row.read_code,
        "recurrence_code":row.recurrence_code,"frames":row.recurrent_frames,"successor":[row.source_position,row.source_token],
        "gate":row.gate_selected,"emitted":row.emitted,"prior":row.prior_selected,"probes":[row.probes.gold,row.probes.rival,row.probes.space]})
}

fn summarize(panel: &Rows<Panel>, tr: &Rows<Trace>, sha: &str) -> Result<Value> {
    if tr.keys().ne(panel.keys()) {
        return Err("incomplete or additional trace rows".into());
    }
    let mut strata = BTreeMap::<String, Acc>::new();
    let mut groups: BTreeMap<(&str, &str), Vec<PairOutcome>> = BTreeMap::new();
    let mut examples = BTreeMap::<String, Value>::new();
    for (id, _) in panel.keys().filter(|(_, side)| *side == 0) {
        let key0 = (id.clone(), 0);
        let key1 = (id.clone(), 1);
        let pl = &panel[&key0];
        let pr = panel.get(&key1).ok_or("missing partner")?;
        let l = &tr[&key0];
        let r = &tr[&key1];
        check(pl, l, pr, sha)?;
        check(pr, r, pl, sha)?;
        if l.read_code == r.read_code
            && l.recurrence_code == r.recurrence_code
            && l.query_token == r.query_token
            && l.source_token == r.source_token
            && (l.gate_selected != r.gate_selected
                || l.writer_vector != r.writer_vector
                || l.top8 != r.top8
                || l.admitted_cells != r.admitted_cells
                || l.probes.gold != r.probes.rival
                || l.probes.rival != r.probes.gold
                || l.probes.space != r.probes.space)
        {
            return Err(format!(
                "identical gate inputs produced different outputs on {id}"
            ));
        }
        for (g, as_rival) in [
            (&l.probes.gold, &r.probes.rival),
            (&r.probes.gold, &l.probes.rival),
            (&l.probes.space, &r.probes.space),
        ] {
            if g.id != as_rival.id
                || g.prior_rank != as_rival.prior_rank
                || g.prior_count != as_rival.prior_count
            {
                return Err("prior differs under swap".into());
            }
        }
        let cohort = pl.cohort.as_str();
        let result = tally_pair(strata.entry(cohort.into()).or_default(), l, r);
        if cohort != "controlled" {
            let length = if l.prefix.len() == r.prefix.len() {
                "equal_length"
            } else {
                "unequal_length"
            };
            tally_pair(
                strata.entry(format!("{cohort}_{length}")).or_default(),
                l,
                r,
            );
        }
        groups
            .entry((cohort, &pl.group_id))
            .or_default()
            .push(result);
        for (label, condition) in [
            ("first", true),
            ("strict_mass_flip", result.strict),
            (
                "both_codes_changed_same_wrong_token",
                l.read_code != r.read_code
                    && l.recurrence_code != r.recurrence_code
                    && l.emitted == r.emitted
                    && l.emitted != l.target
                    && r.emitted != r.target,
            ),
        ] {
            if condition {
                examples
                    .entry(format!("{cohort}_{label}"))
                    .or_insert_with(|| json!({"pair":id,"rows":[example(l),example(r)]}));
            }
        }
    }
    let mut gc = BTreeMap::<String, Acc>::new();
    for ((cohort, _), outcomes) in groups {
        if cohort == "controlled" && outcomes.len() != 2 {
            return Err("controlled group lacks its two reversed assignments".into());
        }
        let a = gc.entry(cohort.into()).or_default();
        a.count("groups", true);
        a.count("all_pairs_strict_mass", outcomes.iter().all(|p| p.strict));
        a.count("all_pairs_full_rank", outcomes.iter().all(|p| p.full));
        a.count(
            "all_first_emissions_correct",
            outcomes.iter().all(|p| p.writer),
        );
    }
    let stats: BTreeMap<_, _> = strata.into_iter().map(|(k, v)| (k, v.finish())).collect();
    let group_stats: BTreeMap<_, _> = gc.into_iter().map(|(k, v)| (k, v.counts)).collect();
    Ok(
        json!({"artifact_sha256":sha,"strata":stats,"independent_groups":group_stats,"examples":examples}),
    )
}
fn run() -> Result<Value> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [panel_path, ap, bp] = args.as_slice() else {
        return Err("usage: question-swap-counts PANEL REASONING.jsonl ANSWER.jsonl".into());
    };
    let panel = indexed(lines::<Panel>(panel_path)?, |p| (p.pair_id.clone(), p.side))?;
    if panel.len() < 800
        || panel.len() % 2 != 0
        || panel
            .keys()
            .any(|(id, s)| !panel.contains_key(&(id.clone(), 1 - *s)))
    {
        return Err("incomplete primary panel".into());
    }
    let a = indexed(lines::<Trace>(ap)?, |r| (r.pair_id.clone(), r.side))?;
    let b = indexed(lines::<Trace>(bp)?, |r| (r.pair_id.clone(), r.side))?;
    let ids: BTreeSet<_> = panel
        .values()
        .filter(|p| p.cohort == "controlled")
        .map(|p| &p.group_id)
        .collect();
    if ids.len() != 200 {
        return Err("expected 200 independent controlled groups".into());
    }
    let mut cross = BTreeMap::<String, Acc>::new();
    for (key, p) in &panel {
        let ra = a.get(key).ok_or("missing reasoning row")?;
        let rb = b.get(key).ok_or("missing answer row")?;
        if ra.prior_selected != rb.prior_selected
            || ra.probes.gold.prior_rank != rb.probes.gold.prior_rank
            || ra.probes.space.prior_rank != rb.probes.space.prior_rank
        {
            return Err("shared unigram differs between models".into());
        }
        let acc = cross.entry(p.cohort.clone()).or_default();
        acc.count("rows", true);
        acc.count("same_emitted", ra.emitted == rb.emitted);
        acc.count("same_gate", ra.gate_selected == rb.gate_selected);
        acc.count(
            "answer_has_better_gold_gate_rank",
            rb.probes.gold.gate_full_rank < ra.probes.gold.gate_full_rank,
        );
        acc.count(
            "answer_has_worse_gold_gate_rank",
            rb.probes.gold.gate_full_rank > ra.probes.gold.gate_full_rank,
        );
    }
    let cross: BTreeMap<_, _> = cross.into_iter().map(|(k, v)| (k, v.counts)).collect();
    Ok(
        json!({"rank_summary":"1-based, nearest-rank quartiles, lower median; optional native ranks summarize admitted rows only",
    "reasoning_first":summarize(&panel,&a,"1b8e1240149b8058b5e79c88c9d42d7c4990339f1b98452a1b34d33b4e132398")?,
    "answer_first":summarize(&panel,&b,"47214eda18119b1e555cec858f9356c93246a23c9f34686962bc19ba05716049")?,
    "between_checkpoints":cross}),
    )
}
fn main() {
    match run().and_then(|v| serde_json::to_string_pretty(&v).map_err(|e| e.to_string())) {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
