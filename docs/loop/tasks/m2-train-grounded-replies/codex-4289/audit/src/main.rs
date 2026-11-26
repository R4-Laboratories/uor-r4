#![forbid(unsafe_code)]
// Read exactly the inherited text and state ABI. Only this verifier uses supervised targets.
#[allow(dead_code)]
#[path = "../../../codex-4288/parity/src/packet.rs"]
mod packet;
use packet::{Address, Row, Sign, EDGE};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    env, fs,
    io::{BufWriter, Write},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Bank = [Vec<[i16; EDGE]>; 2];
fn check(ok: bool, msg: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(msg.into())
    }
}
fn nearest<const D: usize>(source: &[i16; D], bank: &[[i16; D]]) -> Result<(i32, usize)> {
    bank.iter()
        .enumerate()
        .map(|(ordinal, center)| {
            let cost = center
                .iter()
                .zip(source)
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .sum();
            (cost, ordinal)
        })
        .min()
        .ok_or_else(|| "empty inherited bank".into())
}

fn labels(path: &Path, rows: usize) -> Result<Vec<[[usize; 2]; 2]>> {
    let buf = fs::read(path)?;
    check(
        buf.len() == 12 + 16 * rows && buf.get(..8) == Some(b"UORLNK01"),
        "label format or length",
    )?;
    let mut words = buf[8..]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize);
    check(words.next() == Some(rows), "label row count")?;
    let mut out = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut links = [[0; 2]; 2];
        for link in &mut links {
            for field in link {
                *field = words.next().ok_or("truncated label")?;
            }
        }
        out.push(links);
    }
    Ok(out)
}

struct Example<'a> {
    sign: usize,
    word: usize,
    gold: usize,
    sites: Vec<usize>,
    admissible: bool,
    gold_in_slot: bool,
    row: &'a Row,
}
impl<'a> Example<'a> {
    fn new(row: &'a Row, sign: usize, [word, gold]: [usize; 2], state: &[Sign; 2]) -> Result<Self> {
        let governor = row.words.get(word).ok_or("governor ordinal outside row")?;
        let target = row.sites.get(gold).ok_or("gold ordinal outside row")?;
        check(
            !governor.address.overlaps(target.address),
            "supervised word overlaps gold",
        )?;
        let allowed: BTreeSet<_> = row
            .tuples
            .iter()
            .map(|t| t[sign])
            .filter(|&s| !governor.address.overlaps(row.sites[s].address))
            .collect();
        let own = nearest(&governor.point, &state[sign].lexicon)?.0;
        let other = nearest(&governor.point, &state[1 - sign].lexicon)?.0;
        Ok(Self {
            sign,
            word,
            gold,
            admissible: own < other,
            gold_in_slot: allowed.contains(&gold),
            sites: allowed.into_iter().filter(|&s| s != gold).collect(),
            row,
        })
    }
    fn edge(&self, s: usize) -> &[i16; EDGE] {
        &self.row.edges[self.word * self.row.sites.len() + s]
    }
    fn eligible(&self) -> bool {
        self.admissible && self.gold_in_slot && !self.sites.is_empty()
    }
    fn read(&self, banks: &Bank) -> Result<Read> {
        let (gold_cost, gold_center) = nearest(self.edge(self.gold), &banks[self.sign])?;
        let mut costs = Vec::with_capacity(self.sites.len());
        for &site in &self.sites {
            let (cost, center) = nearest(self.edge(site), &banks[self.sign])?;
            costs.push(Candidate { site, cost, center });
        }
        let min = costs.iter().map(|c| c.cost).min();
        let rivals: Vec<_> = costs
            .iter()
            .filter(|c| Some(c.cost) == min)
            .map(|c| c.site)
            .collect();
        Ok(Read {
            gold_cost,
            gold_center,
            in_slot: self.gold_in_slot,
            margin: min.filter(|_| self.gold_in_slot).map(|c| c - gold_cost),
            rivals,
            costs,
        })
    }
}
struct Candidate {
    site: usize,
    cost: i32,
    center: usize,
}
struct Read {
    gold_cost: i32,
    gold_center: usize,
    in_slot: bool,
    margin: Option<i32>,
    rivals: Vec<usize>,
    costs: Vec<Candidate>,
}
fn state_label(r: &Read) -> &'static str {
    if !r.in_slot {
        return "no_gold";
    }
    match r.margin {
        Some(m) if m < 0 => "loss",
        Some(0) => "tie",
        Some(_) => "win",
        None => "no_rival",
    }
}
fn addr(a: Address) -> Value {
    json!({"bytes":a.bytes,"tokens":a.tokens})
}
fn brief(r: &Read) -> Value {
    let pivot = r
        .rivals
        .first()
        .and_then(|first| r.costs.iter().find(|c| c.site == *first));
    json!({"gold_cost":r.gold_cost,"gold_center":r.gold_center,"rivals":r.rivals,"margin":r.margin,
      "outcome":state_label(r),"loss":r.margin.map(|m|(1-m).max(0)), "rival_ties":r.rivals.len(),
      "rival_cost":pivot.map(|c|c.cost),"rival_site":pivot.map(|c|c.site),"rival_center":pivot.map(|c|c.center)})
}
fn hinge_and_wins(ex: &[Example], reads: &[Read]) -> (i64, usize) {
    let mut sum = 0i64;
    let mut wins = 0;
    for (e, r) in ex.iter().zip(reads).filter(|(e, _)| e.eligible()) {
        if let Some(m) = r.margin {
            sum += i64::from((1 - m).max(0));
            wins += usize::from(m > 0);
        }
        debug_assert!(e.gold_in_slot);
    }
    (sum, wins)
}
fn update(ex: &[Example], reads: &[Read], banks: &mut Bank, repel: bool) -> Result<()> {
    let mut votes: [Vec<[i32; EDGE]>; 2] = std::array::from_fn(|s| vec![[0; EDGE]; banks[s].len()]);
    for (e, r) in ex.iter().zip(reads) {
        if !e.eligible() || r.margin.is_none_or(|m| m > 0) {
            continue;
        }
        let mut vote = |site: usize, center: usize, weight: i32| {
            for (d, &value) in e.edge(site).iter().enumerate() {
                let sign = (i32::from(value) - i32::from(banks[e.sign][center][d])).signum();
                votes[e.sign][center][d] += weight * sign;
            }
        };
        vote(e.gold, r.gold_center, 1);
        if repel {
            let first = r.rivals.first().ok_or("violator has no rival")?;
            let rival = r
                .costs
                .iter()
                .find(|c| c.site == *first)
                .ok_or("winning competitor outside scored list")?;
            vote(rival.site, rival.center, -1);
        }
    }
    for (bank, vote) in banks.iter_mut().zip(votes) {
        for (center, counts) in bank.iter_mut().zip(vote) {
            for (c, v) in center.iter_mut().zip(counts) {
                *c = c.saturating_add(v.signum() as i16);
            }
        }
    }
    Ok(())
}
fn bank(state: &[Sign; 2]) -> Bank {
    std::array::from_fn(|s| state[s].transport.clone())
}
fn same_state(path: &Path, initial: &[u8], wanted: &Bank) -> Result<()> {
    let raw = fs::read(path)?;
    check(
        initial.len() == 13336 && raw.len() == initial.len(),
        "native state bytes",
    )?;
    let actual = packet::state(&raw)?;
    check(
        bank(&actual) == *wanted,
        "replayed integer transport state disagrees",
    )?;
    for s in 0..2 {
        let off = 8 + 6664 * s;
        check(
            initial[off..off + 6148] == raw[off..off + 6148],
            "lexical bank bytes changed",
        )?;
        check(
            initial[off + 6148..off + 6152] == raw[off + 6148..off + 6152],
            "transport slot count changed",
        )?;
        let unused = off + 6152 + wanted[s].len() * EDGE * 2;
        check(
            initial[unused..off + 6664] == raw[unused..off + 6664],
            "inactive transport slots changed",
        )?;
    }
    Ok(())
}

fn run(args: &[std::path::PathBuf]) -> Result<()> {
    let [mode, text, supervision, initial, picked, last, events, summary] = args else {
        return Err("usage: audit contrastive|positive|diagnose TEXT LABELS PARENT POCKET FINAL NEW_EVENTS NEW_SUMMARY".into());
    };
    let mode = mode.to_str().ok_or("invalid mode")?;
    check(
        matches!(mode, "contrastive" | "positive" | "diagnose"),
        "unknown mode",
    )?;
    let tape = packet::text(&fs::read(text)?)?;
    let initial_bytes = fs::read(initial)?;
    let state = packet::state(&initial_bytes)?;
    let mut ex = Vec::with_capacity(2 * tape.len());
    for (row, links) in tape.iter().zip(labels(supervision, tape.len())?) {
        for (s, l) in links.into_iter().enumerate() {
            ex.push(Example::new(row, s, l, &state)?);
        }
    }
    let mut trajectory = Vec::new();
    let mut saved = Vec::new();
    let mut scores = Vec::new();
    let mut metrics = Vec::new();
    let mut centers = bank(&state);
    let epochs = if mode == "diagnose" { 0 } else { 8 };
    for step in 0..=epochs {
        let reads = ex
            .iter()
            .map(|e| e.read(&centers))
            .collect::<Result<Vec<_>>>()?;
        let (hinge, wins) = hinge_and_wins(&ex, &reads);
        metrics.push(json!({"step":step,"hinge":hinge,"wins":wins}));
        scores.push((hinge, std::cmp::Reverse(wins), step));
        saved.push(centers.clone());
        if step < epochs {
            update(&ex, &reads, &mut centers, mode == "contrastive")?;
        }
        trajectory.push(reads);
    }
    let chosen = scores.into_iter().min().ok_or("no scored state")?.2;
    same_state(picked, &initial_bytes, &saved[chosen])?;
    same_state(last, &initial_bytes, &centers)?;
    let mut out = BufWriter::new(fs::File::create_new(events)?);
    for (i, e) in ex.iter().enumerate() {
        let a = &trajectory[0][i];
        let b = &trajectory[chosen][i];
        let initial_rivals_after: Vec<_> = b
            .costs
            .iter()
            .filter(|c| a.rivals.contains(&c.site))
            .map(|c| {
                json!({"site":c.site,"cost":c.cost,"center":c.center,
                "margin":b.in_slot.then_some(c.cost-b.gold_cost)})
            })
            .collect();
        let pool: Vec<_> = a
            .costs
            .iter()
            .zip(&b.costs)
            .map(|(x, y)| {
                json!({"site":x.site,
            "address":addr(e.row.sites[x.site].address),"before_cost":x.cost,"after_cost":y.cost,
            "before_center":x.center,"after_center":y.center})
            })
            .collect();
        let h: Vec<_> = trajectory.iter().map(|r| brief(&r[i])).collect();
        serde_json::to_writer(
            &mut out,
            &json!({"row":i/2,"sign":e.sign,"governor_word":e.word,"gold_site":e.gold,
            "governor_address":addr(e.row.words[e.word].address),"gold_address":addr(e.row.sites[e.gold].address),
            "governor_admitted":e.admissible,"gold_in_slot":e.gold_in_slot,"eligible_for_fit":e.eligible(),
            "before":brief(a),"after":brief(b),"final":brief(&trajectory[epochs][i]),
            "initial_rivals_after":initial_rivals_after,"pool":pool,"history":h,
            "transition":[state_label(a),state_label(b)],"chosen_step":chosen}),
        )?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    let mut out = BufWriter::new(fs::File::create_new(summary)?);
    serde_json::to_writer_pretty(
        &mut out,
        &json!({"mode":mode,"rows":tape.len(),"signed_edges":ex.len(),
        "eligible":ex.iter().filter(|e|e.eligible()).count(),"chosen_step":chosen,"trajectory":metrics,
        "reconstructed_selected_state":true,"reconstructed_final_state":true,"lexical_bytes_preserved":true}),
    )?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn main() -> Result<()> {
    run(&env::args_os()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>())
}
