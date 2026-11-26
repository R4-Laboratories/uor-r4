use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::packet::{Row, Sign, EDGE};
use crate::{check, Result};

pub type Votes = [[[i32; EDGE]; 8]; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Strongest,
    All,
}

impl Objective {
    pub fn read(value: &str) -> Result<Self> {
        match value {
            "strongest" => Ok(Self::Strongest),
            "all" | "ordered-all" => Ok(Self::All),
            _ => Err(format!("unknown objective {value}").into()),
        }
    }
}

pub fn labels(raw: &[u8], rows: usize) -> Result<Vec<[[usize; 2]; 2]>> {
    check(
        raw.len() == 12 + 16 * rows && raw.starts_with(b"UORLNK01"),
        "invalid signed label packet",
    )?;
    let fields: Vec<_> = raw[8..]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .collect();
    check(fields[0] == rows, "signed label row count differs")?;
    Ok(fields[1..]
        .chunks_exact(4)
        .map(|r| [[r[0], r[1]], [r[2], r[3]]])
        .collect())
}

fn nearest<const DIM: usize>(point: &[i16; DIM], bank: &[[i16; DIM]]) -> Result<(u64, usize)> {
    bank.iter()
        .enumerate()
        .map(|(slot, center)| {
            let cost = point
                .iter()
                .zip(center)
                .map(|(a, b)| u64::from(a.abs_diff(*b)))
                .sum();
            (cost, slot)
        })
        .min()
        .ok_or_else(|| "empty signed bank".into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub site: usize,
    pub cost: u64,
    pub center: usize,
}

impl Candidate {
    fn compact(self) -> Value {
        json!([self.site, self.cost, self.center])
    }
}

pub struct Edge<'a> {
    row: &'a Row,
    pub sign: usize,
    pub word: usize,
    pub gold_site: usize,
    pub admitted: bool,
    gold_offered: bool,
    wrong_sites: Vec<usize>,
}

impl<'a> Edge<'a> {
    pub fn new(
        row: &'a Row,
        sign: usize,
        [word, gold_site]: [usize; 2],
        state: &[Sign; 2],
    ) -> Result<Self> {
        let gov = row
            .words
            .get(word)
            .ok_or("signed word ordinal outside row")?;
        let noun = row
            .sites
            .get(gold_site)
            .ok_or("signed gold ordinal outside row")?;
        check(
            !gov.address.overlaps(noun.address),
            "gold overlaps its governor",
        )?;
        let by_role: BTreeSet<_> = row.tuples.iter().map(|tuple| tuple[sign]).collect();
        let gold_offered = by_role.contains(&gold_site);
        let wrong_sites = by_role
            .into_iter()
            .filter(|&site| site != gold_site && !gov.address.overlaps(row.sites[site].address))
            .collect();
        let admitted = nearest(&gov.point, &state[sign].lexicon)?.0
            < nearest(&gov.point, &state[1 - sign].lexicon)?.0;
        Ok(Self {
            row,
            sign,
            word,
            gold_site,
            admitted,
            gold_offered,
            wrong_sites,
        })
    }

    fn point(&self, site: usize) -> &[i16; EDGE] {
        &self.row.edges[self.word * self.row.sites.len() + site]
    }

    pub fn read(&self, state: &[Sign; 2]) -> Result<Read> {
        let bank = &state[self.sign].transport;
        let candidate = |site| -> Result<Candidate> {
            let (cost, center) = nearest(self.point(site), bank)?;
            Ok(Candidate { site, cost, center })
        };
        Ok(Read {
            gold: candidate(self.gold_site)?,
            wrong: self
                .wrong_sites
                .iter()
                .copied()
                .map(candidate)
                .collect::<Result<_>>()?,
            gold_offered: self.gold_offered,
            eligible: self.admitted && self.gold_offered && !self.wrong_sites.is_empty(),
        })
    }

    pub fn vote(&self, read: &Read, state: &[Sign; 2], objective: Objective, votes: &mut Votes) {
        if !read.eligible {
            return;
        }
        let mut add = |candidate: Candidate, sign: i32| {
            let center = &state[self.sign].transport[candidate.center];
            let delta = self.point(candidate.site).iter().zip(center);
            for (axis, (x, c)) in delta.enumerate() {
                votes[self.sign][candidate.center][axis] +=
                    sign * (i32::from(*x) - i32::from(*c)).signum();
            }
        };
        let selected: Vec<_> = match objective {
            Objective::All => read
                .wrong
                .iter()
                .copied()
                .filter(|r| r.cost < read.gold.cost)
                .collect(),
            Objective::Strongest => read
                .pivot()
                .filter(|r| r.cost < read.gold.cost)
                .into_iter()
                .collect(),
        };
        for rival in selected {
            add(read.gold, 1);
            add(rival, -1);
        }
    }

    pub fn record(&self, ordinal: usize, read: &Read, legacy: bool) -> Value {
        let mut value = json!({
            "row": ordinal, "sign": self.sign, "governor_word": self.word,
            "gold_site": self.gold_site, "eligible_for_fit": read.eligible,
            "eligible_rivals": read.wrong.len(), "before": read.brief(),
        });
        if !legacy {
            let totals = read.totals();
            value["strongest_loss"] = json!(totals.strongest_loss);
            value["all_loss"] = json!(totals.all_loss);
            value["strictly_cheaper"] = json!(totals.cheaper);
            value["equal_cost_rivals"] = json!(totals.equal);
            value["candidates"] =
                Value::Array(read.wrong.iter().copied().map(Candidate::compact).collect());
        }
        value
    }
}

pub struct Read {
    pub gold: Candidate,
    pub wrong: Vec<Candidate>,
    pub gold_offered: bool,
    pub eligible: bool,
}

impl Read {
    fn pivot(&self) -> Option<Candidate> {
        self.wrong
            .iter()
            .copied()
            .min_by_key(|rival| (rival.cost, rival.site))
    }

    pub fn totals(&self) -> Totals {
        if !self.eligible {
            return Totals::default();
        }
        let mut values = Totals {
            eligible: 1,
            ..Totals::default()
        };
        for rival in &self.wrong {
            let deficit = self.gold.cost.saturating_sub(rival.cost);
            values.strongest_loss = values.strongest_loss.max(deficit);
            values.all_loss += deficit;
            values.cheaper += usize::from(deficit > 0);
            values.equal += usize::from(rival.cost == self.gold.cost);
        }
        values.losses = usize::from(values.cheaper > 0);
        values.ties = usize::from(values.cheaper == 0 && values.equal > 0);
        values.wins = usize::from(values.cheaper == 0 && values.equal == 0);
        values
    }

    fn brief(&self) -> Value {
        let pivot = self.pivot();
        let winners: Vec<_> = self
            .wrong
            .iter()
            .filter(|r| Some(r.cost) == pivot.map(|p| p.cost))
            .map(|r| r.site)
            .collect();
        let margin = pivot
            .filter(|_| self.gold_offered)
            .map(|r| r.cost as i64 - self.gold.cost as i64);
        let outcome = match margin {
            _ if !self.gold_offered => "no_gold",
            Some(m) if m < 0 => "loss",
            Some(0) => "tie",
            Some(_) => "win",
            None => "no_rival",
        };
        json!({
            "gold_cost": self.gold.cost, "gold_center": self.gold.center,
            "rival_cost": pivot.map(|r| r.cost), "rival_site": pivot.map(|r| r.site),
            "rival_center": pivot.map(|r| r.center), "rival_ties": winners.len(),
            "rivals": winners, "margin": margin, "loss": margin.map(|m| (1-m).max(0)), "outcome": outcome,
        })
    }
}

#[derive(Clone, Copy, Default)]
pub struct Totals {
    pub strongest_loss: u64,
    pub all_loss: u64,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    pub cheaper: usize,
    pub equal: usize,
    pub eligible: usize,
}

impl Totals {
    pub fn accumulate(&mut self, read: Self) {
        self.strongest_loss += read.strongest_loss;
        self.all_loss += read.all_loss;
        self.wins += read.wins;
        self.losses += read.losses;
        self.ties += read.ties;
        self.cheaper += read.cheaper;
        self.equal += read.equal;
        self.eligible += read.eligible;
    }

    pub fn value(self, id: Option<usize>, changed: usize) -> Value {
        json!({"id":id,"changed":changed,"strongest_loss":self.strongest_loss,"all_loss":self.all_loss,
            "strict_wins":self.wins,"strict_losses":self.losses,"tied_edges":self.ties,
            "strictly_cheaper":self.cheaper,"equal_cost_rivals":self.equal,"eligible":self.eligible})
    }
}

pub fn proposal(
    base: &[u8],
    state: &[Sign; 2],
    votes: &Votes,
    id: usize,
) -> Result<(Vec<u8>, usize)> {
    check(
        base.len() == 13_336 && id < 68,
        "invalid proposal or native state length",
    )?;
    let step = [1, 2, 4, 8][id % 4];
    let group = id / 4;
    let mut proposed = base.to_vec();
    let mut changed = 0;
    for (sign, bank) in state.iter().enumerate() {
        for (slot, center) in bank.transport.iter().enumerate() {
            if group != 16 && group != 8 * sign + slot {
                continue;
            }
            for (axis, value) in center.iter().enumerate() {
                let next = value.saturating_add(step * (votes[sign][slot][axis].signum() as i16));
                changed += usize::from(next != *value);
                let off = 8 + 6_664 * sign + 6_152 + 2 * (EDGE * slot + axis);
                proposed[off..off + 2].copy_from_slice(&next.to_le_bytes());
            }
        }
    }
    Ok((proposed, changed))
}
