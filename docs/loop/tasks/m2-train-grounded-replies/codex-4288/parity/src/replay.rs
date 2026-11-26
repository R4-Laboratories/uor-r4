use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::packet::{Address, Bank, Row, Sign, EDGE};

#[derive(Clone, Copy)]
pub enum Arm {
    Learned,
    GovernorGap,
}

impl Arm {
    pub fn label(self) -> &'static str {
        match self {
            Self::Learned => "learned",
            Self::GovernorGap => "governor_gap",
        }
    }
}

#[derive(Default)]
pub struct Coverage {
    pub lexical_costs: u64,
    pub transport_costs: u64,
    pub gap_costs: u64,
    pub frozen_costs: u64,
    pub coordinate_differences: u64,
    pub eligible_links: u64,
    pub candidate_triples: u64,
    pub eligible_triples: u64,
    pub minimal_witnesses: u64,
    pub unique_winners: u64,
    pub tied_rows: u64,
    pub no_witness_rows: u64,
}

type LinkCost = [u64; 2];
type TupleCost = [u64; 3];

#[derive(Clone, Copy)]
struct Link {
    word: usize,
    site: usize,
    cost: LinkCost,
}

fn distance<const N: usize>(
    centers: &[[i16; N]],
    point: &[i16; N],
    coverage: &mut Coverage,
) -> Option<u64> {
    coverage.coordinate_differences += (centers.len() * N) as u64;
    centers
        .iter()
        .map(|center| {
            point
                .iter()
                .zip(center)
                .map(|(a, b)| u64::from(a.abs_diff(*b)))
                .sum()
        })
        .min()
}

fn candidates(
    row: &Row,
    signs: &[Sign; 2],
    arm: Arm,
    coverage: &mut Coverage,
) -> [Vec<Vec<Link>>; 2] {
    let mut uses_site = [BTreeSet::new(), BTreeSet::new()];
    for &[affirmed, excluded, _] in &row.tuples {
        uses_site[0].insert(affirmed);
        uses_site[1].insert(excluded);
    }
    let mut candidates = std::array::from_fn(|_| vec![Vec::new(); row.sites.len()]);
    for (word, governor) in row.words.iter().enumerate() {
        coverage.lexical_costs += 2;
        let plus = distance(&signs[0].lexicon, &governor.point, coverage);
        let minus = distance(&signs[1].lexicon, &governor.point, coverage);
        let (head, lexical) = match (plus, minus) {
            (Some(a), Some(b)) if a < b => (0, a),
            (Some(a), Some(b)) if b < a => (1, b),
            (Some(a), None) => (0, a),
            (None, Some(b)) => (1, b),
            _ => continue,
        };
        for &site in &uses_site[head] {
            let noun = row.sites[site].address;
            if governor.address.overlaps(noun) {
                continue;
            }
            let transport = match arm {
                Arm::Learned => {
                    coverage.transport_costs += 1;
                    let edge = &row.edges[word * row.sites.len() + site];
                    distance::<EDGE>(&signs[head].transport, edge, coverage)
                }
                Arm::GovernorGap => {
                    coverage.gap_costs += 1;
                    Some(governor.address.gap(noun))
                }
            };
            if let Some(transport) = transport {
                coverage.eligible_links += 1;
                candidates[head][site].push(Link {
                    word,
                    site,
                    cost: [lexical, transport],
                });
            }
        }
    }
    for links in candidates.iter_mut().flatten() {
        links.sort_by_key(|link| link.cost);
    }
    candidates
}

fn address(span: Address) -> Value {
    json!({"bytes": span.bytes, "tokens": span.tokens})
}

fn sole(values: &BTreeSet<usize>) -> Option<usize> {
    values.first().copied().filter(|_| values.len() == 1)
}

fn pair(cost: Option<LinkCost>) -> [Option<u64>; 2] {
    cost.map(|cost| cost.map(Some)).unwrap_or([None; 2])
}

fn triple(cost: Option<TupleCost>) -> [Option<u64>; 3] {
    cost.map(|cost| cost.map(Some)).unwrap_or([None; 3])
}

fn describe_links(row: &Row, all: &[Link], winners: &[Link]) -> Value {
    let words: BTreeSet<_> = winners.iter().map(|link| link.word).collect();
    let sites: BTreeSet<_> = winners.iter().map(|link| link.site).collect();
    let word = sole(&words);
    let site = sole(&sites);
    json!({
        "word": word, "site": site,
        "governor": word.map(|index| address(row.words[index].address)),
        "noun": site.map(|index| address(row.sites[index].address)),
        "word_ties": words.len(), "site_ties": sites.len(), "link_ties": winners.len(),
        "eligible": all.len(),
        "best": pair(all.first().map(|link| link.cost)),
        "rival": pair(all.get(1).map(|link| link.cost))
    })
}

fn minima(links: &[Link]) -> &[Link] {
    let count = links
        .first()
        .map(|first| links.partition_point(|link| link.cost == first.cost))
        .unwrap_or(0);
    &links[..count]
}

fn combined(plus: LinkCost, minus: LinkCost, bank: u64) -> TupleCost {
    [plus[0] + minus[0], plus[1] + minus[1], bank]
}

fn frozen_costs(row: &Row, banks: &[Bank; 3], coverage: &mut Coverage) -> Vec<[u64; 3]> {
    row.sites
        .iter()
        .map(|site| {
            std::array::from_fn(|head| {
                coverage.frozen_costs += 1;
                // An empty frozen bank is rejected by the fit packet reader.
                distance(&banks[head], &site.point, coverage).unwrap_or(u64::MAX / 4)
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct TupleCandidate {
    ordinal: usize,
    sites: [usize; 3],
    cost: TupleCost,
    witnesses: u64,
}

pub fn predict(
    ordinal: usize,
    row: &Row,
    signs: &[Sign; 2],
    banks: &[Bank; 3],
    arm: Arm,
    coverage: &mut Coverage,
) -> Value {
    let by_site = candidates(row, signs, arm, coverage);
    let costs = frozen_costs(row, banks, coverage);
    let independent: [Value; 2] = std::array::from_fn(|head| {
        let mut all: Vec<_> = by_site[head].iter().flatten().copied().collect();
        all.sort_by_key(|link| link.cost);
        describe_links(row, &all, minima(&all))
    });
    let collision =
        independent[0]["site"].is_number() && independent[0]["site"] == independent[1]["site"];
    coverage.candidate_triples += row.tuples.len() as u64;
    let mut tuples: Vec<_> = row
        .tuples
        .iter()
        .enumerate()
        .filter_map(|(ordinal, &sites)| {
            let [a, e, h] = sites;
            let plus = minima(&by_site[0][a]);
            let minus = minima(&by_site[1][e]);
            let bank = costs[a][0] + costs[e][1] + costs[h][2];
            Some(TupleCandidate {
                ordinal,
                sites,
                cost: combined(plus.first()?.cost, minus.first()?.cost, bank),
                witnesses: (plus.len() * minus.len()) as u64,
            })
        })
        .collect();
    tuples.sort_unstable_by_key(|tuple| tuple.cost);
    coverage.eligible_triples += tuples.len() as u64;
    let n_best = tuples
        .first()
        .map(|first| tuples.partition_point(|tuple| tuple.cost == first.cost))
        .unwrap_or(0);
    let winners = &tuples[..n_best];
    let ties: u64 = winners.iter().map(|tuple| tuple.witnesses).sum();
    coverage.minimal_witnesses += ties;
    match ties {
        0 => coverage.no_witness_rows += 1,
        1 => coverage.unique_winners += 1,
        _ => coverage.tied_rows += 1,
    };
    let best = tuples.first().map(|tuple| tuple.cost);
    // Each tuple contributes its per-site minimum. A second governor at the
    // same minimum still contributes a distinct witness and ties the rival.
    let rival = if ties > 1 {
        best
    } else {
        tuples.get(1).map(|tuple| tuple.cost)
    };
    let unique = winners.first().filter(|_| ties == 1);
    let links = if winners.is_empty() {
        independent.clone()
    } else {
        std::array::from_fn(|head| {
            let sites: BTreeSet<_> = winners.iter().map(|tuple| tuple.sites[head]).collect();
            let mut all = Vec::new();
            let mut minima_in_winning_tuples = Vec::new();
            for site in sites {
                let choices = &by_site[head][site];
                all.extend_from_slice(choices);
                minima_in_winning_tuples.extend_from_slice(minima(choices));
            }
            all.sort_by_key(|link| link.cost);
            describe_links(row, &all, &minima_in_winning_tuples)
        })
    };
    json!({
        "row": ordinal, "arm": arm.label(), "links": links,
        "independent_links": independent, "collision": collision,
        "tuple": {
            "ordinal": unique.map(|tuple| tuple.ordinal),
            "sites": unique.map(|tuple| tuple.sites),
            "addresses": unique.map(|tuple| tuple.sites.map(|i| address(row.sites[i].address))),
            "ties": ties, "eligible": tuples.len(),
            "bank": best.map(|cost| cost[2]), "rival_bank": rival.map(|cost| cost[2]),
            "best": triple(best), "rival": triple(rival)
        }
    })
}
