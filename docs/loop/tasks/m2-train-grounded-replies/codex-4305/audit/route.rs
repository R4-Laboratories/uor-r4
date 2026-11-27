// Included inside the sealed integer ranker.
pub struct ArgumentMask<'a> {
    pub sign: usize,
    pub governor: Option<usize>,
    pub sites: Option<&'a BTreeSet<usize>>,
}

impl Replay<'_> {
    fn admitted(&self, mask: &ArgumentMask<'_>) -> Ranks {
        let mut routes = self.ranks.clone();
        let sign = mask.sign;
        for (site, links) in routes[sign].iter_mut().enumerate() {
            if mask.sites.is_some_and(|sites| !sites.contains(&site)) {
                links.clear();
                continue;
            }
            if let Some(word) = mask.governor {
                links.clear();
                let role = self.row.tuples.iter().any(|tuple| tuple[sign] == site);
                if role
                    && !self.row.words[word]
                        .address
                        .overlaps(self.row.sites[site].address)
                {
                    if let (Some((lex, _)), Some((edge, _))) = (
                        lexical(self.row, self.signs, word, sign),
                        transport(self.row, self.signs, word, site, sign),
                    ) {
                        links.push(Link {
                            word,
                            site,
                            cost: [lex, edge],
                        });
                    }
                }
            }
        }
        routes
    }

    pub fn argument_native(&self, ordinal: usize, mask: Option<&ArgumentMask<'_>>) -> Value {
        let admitted = mask.map(|mask| self.admitted(mask));
        let ranks = admitted.as_ref().unwrap_or(&self.ranks);
        let mut result = read(self.row, ranks, &self.costs, false);
        let independent = &result["independent_links"];
        // Allow the supplied word to occupy both signs, as the native observer does.
        // The strict native gate makes this case unreachable in the inherited audit.
        if independent[0]["word"].is_number() && independent[0]["word"] == independent[1]["word"] {
            result["collision"] = json!(true);
        }
        result["row"] = json!(ordinal);
        result["arm"] = json!("learned");
        result
    }

    pub fn argument_routes(&self, ordinal: usize, mask: Option<&ArgumentMask<'_>>) -> Value {
        let admitted = mask.map(|mask| self.admitted(mask));
        let ranks = admitted.as_ref().unwrap_or(&self.ranks);
        let mut compatible_tuples = 0;
        let mut compatible_assignments = 0;
        let mut best_tuples = 0;
        let mut best_assignments = 0;
        let mut compatible_same_word_assignments = 0;
        let mut best_same_word_assignments = 0;
        let mut best = None;
        for &[a, e, h] in &self.row.tuples {
            let plus = &ranks[0][a];
            let minus = &ranks[1][e];
            let (Some(&p), Some(&m)) = (plus.first(), minus.first()) else {
                continue;
            };
            compatible_tuples += 1;
            compatible_assignments += plus.len() * minus.len();
            let count_same = |p: &[Link], m: &[Link]| {
                let a: BTreeSet<_> = p.iter().map(|link| link.word).collect();
                let e: BTreeSet<_> = m.iter().map(|link| link.word).collect();
                a.intersection(&e).count()
            };
            compatible_same_word_assignments += count_same(plus, minus);
            let cost = joint_cost(p, m, self.costs[a][0] + self.costs[e][1] + self.costs[h][2]);
            if best.is_none_or(|old| cost < old) {
                best = Some(cost);
                best_tuples = 0;
                best_assignments = 0;
                best_same_word_assignments = 0;
            }
            if Some(cost) == best {
                let plus = &plus[..rank_len(plus)];
                let minus = &minus[..rank_len(minus)];
                best_tuples += 1;
                best_assignments += plus.len() * minus.len();
                best_same_word_assignments += count_same(plus, minus);
            }
        }
        let rank = |links: &Vec<Link>| {
            let tied: BTreeSet<_> = links[..rank_len(links)]
                .iter()
                .map(|link| link.word)
                .collect();
            json!({"eligible":links.len(),"best":scalar_pair(links.first().map(|link| link.cost)),
                "rival":scalar_pair(links.get(1).map(|link| link.cost)), "governors":tied})
        };
        let forced_cost = mask
            .and_then(|mask| mask.governor.map(|word| (mask.sign, word)))
            .map(|(sign, word)| {
                [sign, 1 - sign].map(|role| lexical(self.row, self.signs, word, role).map(|p| p.0))
            })
            .unwrap_or([None, None]);
        json!({
            "row":ordinal,
            "mask":mask.map(|mask| json!({"sign":([1,-1][mask.sign]),
                "forced_word":mask.governor,"sites":mask.sites})),
            "forced_lexical":forced_cost,
            "original_tuples":self.row.tuples.len(),"compatible_tuples":compatible_tuples,
            "compatible_assignments":compatible_assignments,"best_tuples":best_tuples,
            "best_assignments":best_assignments,"compatible_same_word_assignments":compatible_same_word_assignments,
            "best_same_word_assignments":best_same_word_assignments,
            "ranks":ranks.iter().map(|by_site| by_site.iter().map(rank).collect::<Vec<_>>()).collect::<Vec<_>>()
        })
    }

    pub fn argument_provenance(&self, sign: usize, word: usize, sites: &BTreeSet<usize>) -> Value {
        let own = lexical(self.row, self.signs, word, sign);
        let opposite = lexical(self.row, self.signs, word, 1 - sign);
        let edges: Vec<_> = sites.iter().map(|&site| {
            let native = self.ranks[sign][site].iter().any(|link| link.word == word);
            let distance = transport(self.row, self.signs, word, site, sign);
            json!({"site":site,"transport":distance,"native_admitted":native,
                "word_site_overlap":self.row.words[word].address.overlaps(self.row.sites[site].address),
                "role_tuples":self.row.tuples.iter().filter(|tuple| tuple[sign]==site).count()})
        }).collect();
        json!({"word":word,"own_lexical":own,"opposite_lexical":opposite,
            "strict_signed_membership":signed_cost(own,opposite).is_some(),"sites":edges})
    }
}
