use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::packet::{Address, Bank, Row, Sign};
use crate::{check, Request, Result};

pub type Cost = [u64; 2];
type JointCost = [u64; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Link {
    word: usize,
    site: usize,
    cost: Cost,
}

type Ranks = [Vec<Vec<Link>>; 2];

fn address(span: Address) -> Value {
    json!({"bytes": span.bytes, "tokens": span.tokens})
}

fn nearest<const N: usize>(bank: &[[i16; N]], point: &[i16; N]) -> Option<(u64, usize)> {
    bank.iter()
        .enumerate()
        .map(|(slot, center)| {
            let distance = center
                .iter()
                .zip(point)
                .map(|(&x, &y)| u64::from((i32::from(x) - i32::from(y)).unsigned_abs()))
                .sum();
            (distance, slot)
        })
        .min()
}

fn lexical(row: &Row, signs: &[Sign; 2], word: usize, sign: usize) -> Option<(u64, usize)> {
    nearest(&signs[sign].lexicon, &row.words[word].point)
}

fn transport(
    row: &Row,
    signs: &[Sign; 2],
    word: usize,
    site: usize,
    sign: usize,
) -> Option<(u64, usize)> {
    nearest(
        &signs[sign].transport,
        &row.edges[word * row.sites.len() + site],
    )
}

fn signed_cost(want: Option<(u64, usize)>, opposite: Option<(u64, usize)>) -> Option<u64> {
    want.map(|p| p.0)
        .filter(|&cost| opposite.is_none_or(|other| cost < other.0))
}

fn rank_links(row: &Row, signs: &[Sign; 2]) -> Ranks {
    std::array::from_fn(|sign| {
        let mut used = vec![false; row.sites.len()];
        for tuple in &row.tuples {
            used[tuple[sign]] = true;
        }
        used.iter()
            .enumerate()
            .map(|(site, &allowed)| {
                if !allowed {
                    return Vec::new();
                }
                let mut links: Vec<_> = row
                    .words
                    .iter()
                    .enumerate()
                    .filter_map(|(word, governor)| {
                        if governor.address.overlaps(row.sites[site].address) {
                            return None;
                        }
                        let own = lexical(row, signs, word, sign);
                        let other = lexical(row, signs, word, 1 - sign);
                        let lex = signed_cost(own, other)?;
                        let (transport, _) = transport(row, signs, word, site, sign)?;
                        Some(Link {
                            word,
                            site,
                            cost: [lex, transport],
                        })
                    })
                    .collect();
                links.sort_by_key(|link| link.cost);
                links
            })
            .collect()
    })
}

fn rank_len(links: &[Link]) -> usize {
    links.first().map_or(0, |first| {
        links.partition_point(|link| link.cost == first.cost)
    })
}

fn scalar_pair(cost: Option<Cost>) -> Value {
    cost.map_or(json!([null, null]), |words| json!(words))
}
fn scalar_triple(cost: Option<JointCost>) -> Value {
    cost.map_or(json!([null, null, null]), |words| json!(words))
}

fn describe(row: &Row, mut all: Vec<Link>, winners: Vec<Link>) -> Value {
    let words: BTreeSet<_> = winners.iter().map(|link| link.word).collect();
    let sites: BTreeSet<_> = winners.iter().map(|link| link.site).collect();
    let sole = |ids: &BTreeSet<usize>| ids.first().copied().filter(|_| ids.len() == 1);
    let word = sole(&words);
    let site = sole(&sites);
    all.sort_by_key(|link| link.cost);
    json!({
        "word":word,"site":site,
        "governor":word.map(|i| address(row.words[i].address)),
        "noun":site.map(|i| address(row.sites[i].address)),
        "word_ties":words.len(),"site_ties":sites.len(),"link_ties":winners.len(),
        "eligible":all.len(),
        "best":scalar_pair(all.first().map(|link| link.cost)),
        "rival":scalar_pair(all.get(1).map(|link| link.cost))
    })
}

fn independent(row: &Row, ranks: &[Vec<Link>]) -> Value {
    let mut all: Vec<_> = ranks.iter().flatten().copied().collect();
    all.sort_by_key(|link| link.cost);
    let winners = all[..rank_len(&all)].to_vec();
    describe(row, all, winners)
}

fn site_costs(row: &Row, banks: &[Bank; 3]) -> Result<Vec<[u64; 3]>> {
    check(
        banks.iter().all(|bank| !bank.is_empty()),
        "empty frozen tuple bank",
    )?;
    row.sites
        .iter()
        .map(|site| {
            let mut costs = [0; 3];
            for (sign, bank) in banks.iter().enumerate() {
                costs[sign] = nearest(bank, &site.point)
                    .ok_or("frozen bank has no center")?
                    .0;
            }
            Ok(costs)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Witness {
    ordinal: usize,
    sites: [usize; 3],
    cost: JointCost,
    ties: usize,
}

fn joint_cost(a: Link, e: Link, bank: u64) -> JointCost {
    [a.cost[0] + e.cost[0], a.cost[1] + e.cost[1], bank]
}

fn read(row: &Row, ranks: &Ranks, costs: &[[u64; 3]], include_sites: bool) -> Value {
    let independent: [Value; 2] = std::array::from_fn(|sign| independent(row, &ranks[sign]));
    let mut candidates = Vec::new();
    for (ordinal, &[a, e, h]) in row.tuples.iter().enumerate() {
        let plus = &ranks[0][a];
        let minus = &ranks[1][e];
        if let (Some(&left), Some(&right)) = (plus.first(), minus.first()) {
            let bank = costs[a][0] + costs[e][1] + costs[h][2];
            candidates.push(Witness {
                ordinal,
                sites: [a, e, h],
                cost: joint_cost(left, right, bank),
                ties: rank_len(plus) * rank_len(minus),
            });
        }
    }
    candidates.sort_unstable_by_key(|candidate| candidate.cost);
    let best = candidates.first().map(|candidate| candidate.cost);
    let minima =
        &candidates[..candidates.partition_point(|candidate| Some(candidate.cost) == best)];
    let ties: usize = minima.iter().map(|candidate| candidate.ties).sum();
    let unique = minima.first().filter(|_| ties == 1);
    let rival = if ties > 1 {
        best
    } else {
        let mut other = candidates.get(1).map(|candidate| candidate.cost);
        if let Some(winner) = unique {
            let [a, e, _] = winner.sites;
            let plus = &ranks[0][a];
            let minus = &ranks[1][e];
            let alternatives = [
                plus.get(1)
                    .map(|&link| joint_cost(link, minus[0], winner.cost[2])),
                minus
                    .get(1)
                    .map(|&link| joint_cost(plus[0], link, winner.cost[2])),
            ];
            other = other
                .into_iter()
                .chain(alternatives.into_iter().flatten())
                .min();
        }
        other
    };
    let minimum_sites: [BTreeSet<usize>; 3] =
        std::array::from_fn(|i| minima.iter().map(|result| result.sites[i]).collect());
    let links = if minima.is_empty() {
        independent.clone()
    } else {
        std::array::from_fn(|sign| {
            let choices: Vec<_> = minimum_sites[sign]
                .iter()
                .map(|&site| &ranks[sign][site])
                .collect();
            let all = choices
                .iter()
                .flat_map(|part| part.iter().copied())
                .collect();
            let winners = choices
                .iter()
                .flat_map(|part| part[..rank_len(part)].iter().copied())
                .collect();
            describe(row, all, winners)
        })
    };
    let mut tuple = json!({
        "ordinal":unique.map(|r| r.ordinal),"sites":unique.map(|r| r.sites),
        "addresses":unique.map(|r| r.sites.map(|site| address(row.sites[site].address))),
        "ties":ties,"eligible":candidates.len(),
        "bank":best.map(|cost|cost[2]),"rival_bank":rival.map(|cost|cost[2]),
        "best":scalar_triple(best),"rival":scalar_triple(rival)
    });
    if include_sites {
        tuple["min_sites"] = json!(minimum_sites);
    }
    json!({"links":links,"independent_links":independent,"tuple":tuple,
        "collision":independent[0]["site"].is_number() && independent[0]["site"]==independent[1]["site"]})
}

fn conditioned(row: &Row, all: &Ranks, costs: &[[u64; 3]], request: Request, site: usize) -> Value {
    let mut ranks = all.clone();
    let pin = ranks[request.sign][site]
        .iter()
        .find(|link| link.word == request.word)
        .copied();
    for candidates in &mut ranks[request.sign] {
        candidates.clear();
    }
    ranks[request.sign][site].extend(pin);
    let mut view = read(row, &ranks, costs, true);
    view.as_object_mut().map(|value| value.remove("collision"));
    view
}

fn fixed_minimum(
    row: &Row,
    ranks: &Ranks,
    costs: &[[u64; 3]],
    sign: usize,
    pin: Option<Link>,
) -> (Option<JointCost>, usize, usize) {
    let Some(pin) = pin else { return (None, 0, 0) };
    let mut results = row
        .tuples
        .iter()
        .filter(|tuple| tuple[sign] == pin.site)
        .filter_map(|&[a, e, h]| {
            let opposite = &ranks[1 - sign][if sign == 0 { e } else { a }];
            let cost = joint_cost(
                pin,
                *opposite.first()?,
                costs[a][0] + costs[e][1] + costs[h][2],
            );
            Some((cost, rank_len(opposite)))
        });
    let mut best = None;
    let mut ties = 0;
    let mut eligible = 0;
    for (cost, count) in &mut results {
        eligible += 1;
        if best.is_none_or(|old| cost < old) {
            best = Some(cost);
            ties = count;
        } else if Some(cost) == best {
            ties += count;
        }
    }
    (best, eligible, ties)
}

fn rank_at<T: Ord + Copy>(target: Option<T>, costs: &[Option<T>]) -> Value {
    if let Some(target) = target {
        json!({
            "cheaper":costs.iter().flatten().filter(|&&cost| cost<target).count(),
            "equal":costs.iter().flatten().filter(|&&cost| cost==target).count()
        })
    } else {
        Value::Null
    }
}

pub struct Replay<'a> {
    row: &'a Row,
    signs: &'a [Sign; 2],
    ranks: Ranks,
    costs: Vec<[u64; 3]>,
}

impl<'a> Replay<'a> {
    pub fn new(row: &'a Row, signs: &'a [Sign; 2], banks: &[Bank; 3]) -> Result<Self> {
        Ok(Self {
            row,
            signs,
            ranks: rank_links(row, signs),
            costs: site_costs(row, banks)?,
        })
    }

    pub fn native(&self, ordinal: usize) -> Value {
        let mut native = read(self.row, &self.ranks, &self.costs, false);
        native["row"] = json!(ordinal);
        native["arm"] = json!("learned");
        native
    }

    pub fn request(&self, index: usize, requested: Request) -> Value {
        let row = self.row;
        let all = &self.ranks;
        let word = requested.word;
        let sign = requested.sign;
        let own = lexical(row, self.signs, word, sign);
        let other = lexical(row, self.signs, word, 1 - sign);
        let lex = signed_cost(own, other);
        let links: Vec<_> = all[sign]
            .iter()
            .map(|ranks| ranks.iter().find(|link| link.word == word).copied())
            .collect();
        let local_costs: Vec<_> = links
            .iter()
            .map(|link| link.map(|link| link.cost))
            .collect();
        let joint: Vec<_> = links
            .iter()
            .map(|&link| fixed_minimum(row, all, &self.costs, sign, link))
            .collect();
        let joint_costs: Vec<_> = joint.iter().map(|outcome| outcome.0).collect();
        let local: Vec<_> = row.sites.iter().enumerate().map(|(site, occurrence)| {
            let ranks = &all[sign][site];
            let role = row.tuples.iter().any(|tuple| tuple[sign]==site);
            let nonoverlap = !row.words[word].address.overlaps(occurrence.address);
            let eligible = links[site];
            let transport = transport(row,self.signs,word,site,sign);
            let (best, count, ties) = joint[site];
            json!({
                "site":site,"noun":address(occurrence.address),"role":role,"nonoverlap":nonoverlap,
                "strict_sign":lex.is_some(), "eligible":eligible.is_some(),
                "cost":[own.map(|pair| pair.0),transport.map(|pair| pair.0)],
                "opposite_lexical":other.map(|pair| pair.0),
                "lexical_center":own.map(|pair| pair.1),
                "transport_center":transport.map(|pair| pair.1),
                "attachment":describe(row,ranks.clone(),ranks[..rank_len(ranks)].to_vec()),
                "local_rank":rank_at(local_costs[site],&local_costs),
                "joint":{"best":scalar_triple(best),"eligible":count,"ties":ties,"rank":rank_at(best,&joint_costs)}
            })
        }).collect();
        let native = read(row, all, &self.costs, true);
        let third: Vec<_> = native["tuple"]["min_sites"][sign]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|&id| id != &json!(requested.site) && id != &json!(requested.rival))
            .collect();
        json!({
            "request":index,"row":requested.row,"sign":requested.signed(),
            "word":word,"governor":address(row.words[word].address),"requested_site":requested.site,
            "rival_site":requested.rival,"local_sites":local,
            "requested":conditioned(row,all,&self.costs,requested,requested.site),
            "rival":conditioned(row,all,&self.costs,requested,requested.rival),
            "native":native,"native_third_sites":third
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::{Occurrence, EDGE, POINT};

    fn point<const D: usize>(first: i16) -> [i16; D] {
        let mut coords = [0; D];
        coords[0] = first;
        coords
    }

    fn occurrence(position: u32, value: i16) -> Occurrence {
        Occurrence {
            address: Address {
                bytes: [position, position + 1],
                tokens: [position, position + 1],
            },
            point: point::<POINT>(value),
        }
    }

    fn model() -> (Row, [Sign; 2], [Bank; 3]) {
        let row = Row {
            words: vec![occurrence(0, 1), occurrence(1, 10), occurrence(7, 0)],
            sites: vec![
                occurrence(2, 0),
                occurrence(3, 0),
                occurrence(4, 0),
                occurrence(5, 0),
                occurrence(6, 0),
            ],
            edges: [1, 3, 15, 15, 15, 15, 15, 9, 1, 15, 0, 15, 15, 15, 15]
                .into_iter()
                .map(point::<EDGE>)
                .collect(),
            tuples: vec![[0, 2, 4], [1, 3, 4]],
        };
        let states = [
            Sign {
                lexicon: vec![point(0)],
                transport: vec![point(0)],
            },
            Sign {
                lexicon: vec![point(10)],
                transport: vec![point(0)],
            },
        ];
        (row, states, std::array::from_fn(|_| vec![point(0)]))
    }

    #[test]
    fn pin_fixes_governor_and_reversal_is_in_joint_cost() -> Result<()> {
        let (row, states, banks) = model();
        let request = Request {
            row: 0,
            sign: 0,
            word: 0,
            site: 0,
            rival: 1,
        };
        let trace = Replay::new(&row, &states, &banks)?.request(0, request);
        assert_eq!(trace["local_sites"][0]["attachment"]["word"], 2);
        assert_eq!(trace["requested"]["links"][0]["word"], 0);
        assert_eq!(trace["local_sites"][0]["cost"], json!([1, 1]));
        assert_eq!(trace["local_sites"][1]["cost"], json!([1, 3]));
        assert_eq!(trace["requested"]["tuple"]["best"], json!([1, 10, 0]));
        assert_eq!(trace["rival"]["tuple"]["best"], json!([1, 4, 0]));
        Ok(())
    }

    #[test]
    fn exact_tuple_ties_and_missing_opposite_attachment_are_distinct() -> Result<()> {
        let (mut row, states, banks) = model();
        let request = Request {
            row: 0,
            sign: 0,
            word: 0,
            site: 0,
            rival: 1,
        };
        row.tuples.push([0, 3, 4]);
        row.edges[5 + 2] = point(1);
        let tied = Replay::new(&row, &states, &banks)?.request(0, request);
        assert_eq!(tied["requested"]["tuple"]["ties"], 2);
        assert_eq!(
            tied["requested"]["tuple"]["min_sites"],
            json!([[0], [2, 3], [4]])
        );
        row.tuples.pop();
        row.sites[2].address = row.words[1].address;
        let missing = Replay::new(&row, &states, &banks)?.request(0, request);
        assert_eq!(missing["local_sites"][0]["eligible"], true);
        assert_eq!(
            missing["requested"]["tuple"]["best"],
            json!([null, null, null])
        );
        assert_eq!(missing["rival"]["tuple"]["ties"], 1);
        Ok(())
    }
}
