// Included after the sealed #4300 integer-ranking implementation.

#[derive(Default)]
struct SiteMinimum {
    best: Option<JointCost>,
    eligible: usize,
    ties: usize,
    tuples: Vec<usize>,
}

impl SiteMinimum {
    fn insert(&mut self, cost: JointCost, ordinal: usize, multiplicity: usize) {
        self.eligible += 1;
        if self.best.is_none_or(|best| cost < best) {
            self.best = Some(cost);
            self.ties = 0;
            self.tuples.clear();
        }
        if Some(cost) == self.best {
            self.ties += multiplicity;
            self.tuples.push(ordinal);
        }
    }

    fn physical(&self) -> Value {
        json!({"best":self.best,"eligible":self.eligible,"ties":self.ties,
            "first_tuple":self.tuples.first()})
    }
}

fn compare_third(requested: Option<JointCost>, third: Option<JointCost>) -> &'static str {
    match (requested, third) {
        (None, None) => "both_absent",
        (None, Some(_)) => "requested_absent",
        (Some(_), None) => "third_absent",
        (Some(a), Some(b)) => match a.cmp(&b) {
            std::cmp::Ordering::Less => "requested_cheaper",
            std::cmp::Ordering::Equal => "tie",
            std::cmp::Ordering::Greater => "third_cheaper",
        },
    }
}

fn strongest_third(sites: &[SiteMinimum], request: Request, correct: Option<JointCost>) -> Value {
    let rivals = || {
        sites
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != request.site && *i != request.rival)
    };
    let best = rivals().filter_map(|(_, site)| site.best).min();
    let winners: Vec<_> = rivals()
        .filter(|(_, site)| best.is_some() && site.best == best)
        .map(|(i, site)| json!({"site":i,"ties":site.ties,"tuple_ordinals":site.tuples}))
        .collect();
    json!({"best":best,"sites":winners,"relation":compare_third(sites[request.site].best, best),
        "correct_governor_relation":compare_third(correct, best)})
}

impl Replay<'_> {
    pub fn competition(&self, request: Request) -> (Value, Value) {
        let row = self.row;
        let sign = request.sign;
        let mut fixed = Vec::new();
        let mut free = Vec::new();
        let local: Vec<_> = self.ranks[sign]
            .iter()
            .map(|links| links.iter().find(|link| link.word == request.word).copied())
            .collect();
        fixed.resize_with(row.sites.len(), SiteMinimum::default);
        free.resize_with(row.sites.len(), SiteMinimum::default);

        for (ordinal, &[a, e, h]) in row.tuples.iter().enumerate() {
            let [target, other] = if sign == 0 { [a, e] } else { [e, a] };
            let opposite = &self.ranks[1 - sign][other];
            let Some(&opposite_link) = opposite.first() else {
                continue;
            };
            let allowed = &self.ranks[sign][target];
            let Some(&allowed_link) = allowed.first() else {
                continue;
            };
            let bank = self.costs[a][0] + self.costs[e][1] + self.costs[h][2];
            let opposite_ties = rank_len(opposite);
            free[target].insert(
                joint_cost(allowed_link, opposite_link, bank),
                ordinal,
                rank_len(allowed) * opposite_ties,
            );
            if let Some(pin) = local[target] {
                fixed[target].insert(joint_cost(pin, opposite_link, bank), ordinal, opposite_ties);
            }
        }

        let sites: Vec<_> = self.ranks[sign]
            .iter()
            .enumerate()
            .map(|(i, links)| {
                let governors: Vec<_> = links[..rank_len(links)]
                    .iter()
                    .map(|link| link.word)
                    .collect();
                json!({"i":i,"local":local[i].map(|link| link.cost),
                "fixed":fixed[i].physical(),"free":free[i].physical(),"governors":governors})
            })
            .collect();
        let correct = fixed[request.site].best;
        let mut strongest_free = strongest_third(&free, request, correct);
        let mut strongest_fixed = strongest_third(&fixed, request, correct);
        for (group, is_fixed) in [(&mut strongest_free, false), (&mut strongest_fixed, true)] {
            if let Some(thirds) = group["sites"].as_array_mut() {
                for site in thirds {
                    if let Some(i) = site["site"].as_u64().map(|i| i as usize) {
                        site["address"] = address(row.sites[i].address);
                        let links = &self.ranks[sign][i];
                        let governor_ids: Vec<_> = if is_fixed {
                            vec![request.word]
                        } else {
                            links[..rank_len(links)]
                                .iter()
                                .map(|link| link.word)
                                .collect()
                        };
                        site["governors"] = json!(governor_ids);
                        site["governor_addresses"] = json!(governor_ids
                            .into_iter()
                            .map(|word| address(row.words[word].address))
                            .collect::<Vec<_>>());
                    }
                }
            }
        }
        let raw = json!({"row":request.row,"sign":request.signed(),"word":request.word,
            "requested":request.site,"rival":request.rival,"sites":sites});
        let witness = json!({"row":request.row,"sign":request.signed(),
            "requested_address":address(row.sites[request.site].address),"rival_address":address(row.sites[request.rival].address),
            "fixed_third":strongest_fixed,"free_third":strongest_free});
        (raw, witness)
    }
}

#[cfg(test)]
mod competition_tests {
    use super::*;
    use crate::packet::Occurrence;

    fn point<const N: usize>(x: i16) -> [i16; N] {
        let mut result = [0; N];
        result[0] = x;
        result
    }

    fn occurrence(index: u32, coordinate: i16) -> Occurrence {
        let span = [index, index + 1];
        Occurrence {
            address: Address {
                bytes: span,
                tokens: span,
            },
            point: point(coordinate),
        }
    }

    #[test]
    fn competing_site_uses_another_governor_and_keeps_all_tied_assignments() -> Result<()> {
        let mut row = Row {
            words: vec![occurrence(0, 0), occurrence(2, 0), occurrence(4, 10)],
            sites: (10..16).map(|i| occurrence(i, 0)).collect(),
            edges: [4, 7, 10, 10, 0, 0, 8, 9, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0]
                .into_iter()
                .map(point)
                .collect(),
            tuples: vec![[0, 4, 5], [1, 4, 5], [2, 4, 5], [2, 4, 3], [3, 4, 5]],
        };
        let signs = [
            Sign {
                lexicon: vec![point(0)],
                transport: vec![point(0)],
            },
            Sign {
                lexicon: vec![point(10)],
                transport: vec![point(0)],
            },
        ];
        let banks = std::array::from_fn(|_| vec![point(0)]);
        let request = Request {
            row: 0,
            sign: 0,
            word: 0,
            site: 0,
            rival: 1,
        };
        let (physical, witness) = Replay::new(&row, &signs, &banks)?.competition(request);
        assert_eq!(witness["fixed_third"]["relation"], "requested_cheaper");
        assert_eq!(witness["free_third"]["relation"], "third_cheaper");
        let third = &witness["free_third"]["sites"];
        assert_eq!(third[0]["tuple_ordinals"], json!([2, 3]));
        assert_eq!(third[0]["ties"], 2);
        assert_eq!(third[0]["governors"], json!([1]));
        assert_eq!(third[1]["tuple_ordinals"], json!([4]));
        assert_eq!(physical["sites"][2]["free"]["first_tuple"], 2);
        row.edges[6] = point(4);
        row.edges[8] = point(4);
        row.tuples.pop();
        let (physical, witness) = Replay::new(&row, &signs, &banks)?.competition(request);
        assert_eq!(physical["sites"][0]["governors"], json!([0, 1]));
        assert_eq!(physical["sites"][0]["free"]["ties"], 2);
        assert_eq!(witness["free_third"]["relation"], "tie");
        row.tuples.truncate(2);
        let (_, witness) = Replay::new(&row, &signs, &banks)?.competition(request);
        assert_eq!(witness["free_third"]["relation"], "third_absent");
        row.tuples.clear();
        let (_, witness) = Replay::new(&row, &signs, &banks)?.competition(request);
        assert_eq!(witness["free_third"]["relation"], "both_absent");
        Ok(())
    }
}
