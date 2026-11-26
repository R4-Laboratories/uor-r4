use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::packet::{Address, Row, Sign};
use crate::{check, number, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mask {
    pub sign: usize,
    pub winner: Option<usize>,
    pub control: Option<usize>,
}

impl Mask {
    pub fn requested(self, arm: usize) -> Option<usize> {
        [None, self.winner, self.control][arm]
    }
}

pub fn masks(blob: &[u8], rows: &[Row]) -> Result<Vec<Option<Mask>>> {
    let lines: Vec<_> = std::str::from_utf8(blob)?.lines().collect();
    check(lines.len() == rows.len(), "wrong mask row count")?;
    lines
        .iter()
        .zip(rows)
        .map(|(line, row)| {
            let fields = line
                .split_ascii_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<Vec<i64>, _>>()?;
            match fields.as_slice() {
                [-1, -1, -1] => Ok(None),
                [sign @ (0 | 1), winner, control] => {
                    let checked = |site: i64| -> Result<Option<usize>> {
                        if site == -1 {
                            return Ok(None);
                        }
                        let site = usize::try_from(site)?;
                        check(site < row.sites.len(), "mask site outside tape")?;
                        Ok(Some(site))
                    };
                    let winner = checked(*winner)?;
                    let control = checked(*control)?;
                    check(
                        winner.is_none() || winner != control,
                        "same winner and control site",
                    )?;
                    Ok(Some(Mask {
                        sign: *sign as usize,
                        winner,
                        control,
                    }))
                }
                _ => Err("malformed signed mask".into()),
            }
        })
        .collect()
}

pub fn labels(bytes: &[u8], rows: &[Row]) -> Result<Vec<[[usize; 2]; 2]>> {
    check(
        bytes.get(..8) == Some(b"UORLNK01") && bytes.len() == 12 + 16 * rows.len(),
        "wrong label format or length",
    )?;
    let mut fields = bytes[8..]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize);
    check(fields.next() == Some(rows.len()), "wrong label row count")?;
    rows.iter()
        .map(|row| {
            let mut labels = [[0; 2]; 2];
            for label in &mut labels {
                let word = fields.next().ok_or("truncated governor label")?;
                let gold = fields.next().ok_or("truncated gold label")?;
                check(
                    word < row.words.len() && gold < row.sites.len(),
                    "original label outside tape",
                )?;
                *label = [word, gold];
            }
            Ok(labels)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub site: usize,
    pub cost: [u64; 2],
    pub transport_center: usize,
}

impl Candidate {
    pub fn json(self, row: &Row) -> Value {
        json!({"site": self.site, "noun": address(row.sites[self.site].address), "cost": self.cost,
            "transport_center": self.transport_center})
    }
}

pub fn address(span: Address) -> Value {
    json!({"bytes": span.bytes, "tokens": span.tokens})
}

pub fn spans(span: &Value) -> Value {
    json!({"bytes": [span["bytes"]["start"], span["bytes"]["end"]],
        "tokens": [span["tokens"]["start"], span["tokens"]["end"]]})
}

fn closest<const N: usize>(point: &[i16; N], bank: &[[i16; N]]) -> Result<(u64, usize)> {
    bank.iter()
        .enumerate()
        .map(|(slot, center)| {
            let cost = center
                .iter()
                .zip(point)
                .map(|(&a, &b)| u64::from((i32::from(a) - i32::from(b)).unsigned_abs()))
                .sum();
            (cost, slot)
        })
        .min()
        .ok_or_else(|| "empty frozen signed bank".into())
}

pub fn candidates(
    row: &Row,
    state: &[Sign; 2],
    sign: usize,
    word: usize,
) -> Result<Vec<Candidate>> {
    let governor = &row.words[word];
    let lexical = closest(&governor.point, &state[sign].lexicon)?.0;
    if lexical >= closest(&governor.point, &state[1 - sign].lexicon)?.0 {
        return Ok(Vec::new());
    }
    let slots: BTreeSet<_> = row.tuples.iter().map(|tuple| tuple[sign]).collect();
    slots
        .into_iter()
        .filter(|&site| !governor.address.overlaps(row.sites[site].address))
        .map(|site| {
            let edge = &row.edges[word * row.sites.len() + site];
            let (transport, transport_center) = closest(edge, &state[sign].transport)?;
            Ok(Candidate {
                site,
                cost: [lexical, transport],
                transport_center,
            })
        })
        .collect()
}

pub fn raw_rank(
    row: &Row,
    ordinal: usize,
    sign: usize,
    word: usize,
    candidates: &[Candidate],
    requested: Option<usize>,
    omitted: Option<usize>,
) -> Value {
    json!({"row": ordinal, "sign": sign, "word": word, "governor": address(row.words[word].address),
        "requested_site": requested, "omitted_site": omitted,
        "candidates": candidates.iter().map(|c| c.json(row)).collect::<Vec<_>>()})
}

fn inside(interval: [u32; 2], a: [u32; 2], b: [u32; 2]) -> bool {
    (a[1] <= interval[0] && interval[1] <= b[0]) || (b[1] <= interval[0] && interval[1] <= a[0])
}

fn between(site: Address, governor: Address, gold: Address) -> bool {
    inside(site.bytes, governor.bytes, gold.bytes)
        && inside(site.tokens, governor.tokens, gold.tokens)
}

pub fn strongest(candidates: &[Candidate], excluding: Option<usize>) -> Vec<&Candidate> {
    let eligible = candidates.iter().filter(|c| Some(c.site) != excluding);
    let minimum = eligible.clone().map(|c| c.cost).min();
    eligible.filter(|c| Some(c.cost) == minimum).collect()
}

pub fn plan(
    row: &Row,
    sign: usize,
    [word, gold]: [usize; 2],
    gold_tuple: [usize; 3],
    original: [usize; 2],
    signed: [&[Candidate]; 2],
) -> Result<(Mask, Vec<(usize, [u64; 5])>)> {
    let governor = row.words[word].address;
    let target = row.sites[gold].address;
    let winner = original[1];
    let rival = row.sites[winner].address;
    let prior_strongest = signed.map(|ranks| strongest(ranks, Some(gold)));
    for (stage, candidates) in prior_strongest.iter().enumerate() {
        check(
            candidates.iter().any(|c| c.site == original[stage]),
            "historical joint choice was not a strongest signed wrong site",
        )?;
    }
    check(
        between(rival, governor, target),
        "prior winner was not fully intervening in bytes and BPE",
    )?;
    let mut eligible: Vec<_> = signed[1]
        .iter()
        .filter(|c| {
            let site = row.sites[c.site].address;
            between(site, governor, target)
                && !gold_tuple.contains(&c.site)
                && !original.contains(&c.site)
                && !prior_strongest.iter().flatten().any(|r| r.site == c.site)
        })
        .map(|c| {
            let site = row.sites[c.site].address;
            (
                c.site,
                [
                    site.gap(rival),
                    u64::from(site.tokens[0].abs_diff(rival.tokens[0])),
                    u64::from(site.bytes[0]),
                    u64::from(site.bytes[1]),
                    c.site as u64,
                ],
            )
        })
        .collect();
    eligible.sort_by_key(|(_, order)| *order);
    let control = eligible.first().map(|(site, _)| *site);
    Ok((
        Mask {
            sign,
            winner: Some(winner),
            control,
        },
        eligible,
    ))
}

pub fn decorate(mut value: Value, text: &str) -> Result<Value> {
    if value.is_null() {
        return Ok(value);
    }
    let start = number(&value["bytes"][0])?;
    let end = number(&value["bytes"][1])?;
    let surface = text
        .get(start..end)
        .ok_or("source address outside original UTF-8 text")?;
    check(!surface.is_empty(), "empty original surface")?;
    value["surface"] = json!(surface);
    Ok(value)
}

pub fn grade(row: &Row, text: &str, candidates: &[Candidate], gold: usize) -> Result<Value> {
    let describe = |c: &Candidate| -> Result<Value> {
        let mut value = c.json(row);
        value["noun"] = decorate(value["noun"].take(), text)?;
        Ok(value)
    };
    let gold = candidates
        .iter()
        .find(|c| c.site == gold)
        .ok_or("gold unavailable in signed read")?;
    let winners = strongest(candidates, None)
        .into_iter()
        .map(describe)
        .collect::<Result<Vec<_>>>()?;
    let wrong = strongest(candidates, Some(gold.site))
        .into_iter()
        .map(describe)
        .collect::<Result<Vec<_>>>()?;
    let ranking = candidates
        .iter()
        .map(|c| {
            let mut value = describe(c)?;
            value["rank"] = json!(1 + candidates.iter().filter(|r| r.cost < c.cost).count());
            value["ties"] = json!(candidates.iter().filter(|r| r.cost == c.cost).count());
            Ok(value)
        })
        .collect::<Result<Vec<Value>>>()?;
    let mut ranking = ranking;
    ranking.sort_by_key(|c| (c["rank"].as_u64(), c["site"].as_u64()));
    Ok(
        json!({"gold": describe(gold)?, "gold_rank": 1 + candidates.iter().filter(|c| c.cost < gold.cost).count(),
        "gold_ties": candidates.iter().filter(|c| c.cost == gold.cost).count(), "candidates": ranking,
        "winners": winners, "strongest_wrong": wrong}),
    )
}
