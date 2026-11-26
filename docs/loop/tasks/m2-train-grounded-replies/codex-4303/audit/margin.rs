// Compiled beside the sealed #4302 ranking implementation. Independently
// enumerates legal tuples for the authored governor at the contested sign.
struct ObjectiveArm {
    selected: Option<(JointCost, usize, Link, Link)>,
    eligible: usize,
    ties: usize,
    min_sites: BTreeSet<usize>,
}

impl ObjectiveArm {
    fn new() -> Self {
        Self {
            selected: None,
            eligible: 0,
            ties: 0,
            min_sites: BTreeSet::new(),
        }
    }

    fn offer(&mut self, cost: JointCost, ordinal: usize, own: Link, other: Link, ties: usize) {
        self.eligible += 1;
        if self.selected.is_none_or(|prior| cost < prior.0) {
            self.selected = Some((cost, ordinal, own, other));
            self.ties = 0;
            self.min_sites.clear();
        }
        if self.selected.is_some_and(|prior| cost == prior.0) {
            self.ties += ties;
            self.min_sites.insert(own.site);
        }
    }
}

impl Replay<'_> {
    fn objective_arm(&self, request: Request, score: &ObjectiveArm, target: bool) -> Value {
        let focus = score.selected.map(|(_, _, own, _)| own.site);
        let local_site = if target { Some(request.site) } else { focus };
        let local = local_site
            .filter(|&site| {
                !self.row.words[request.word]
                    .address
                    .overlaps(self.row.sites[site].address)
                    && self.row.tuples.iter().any(|t| t[request.sign] == site)
            })
            .and_then(|site| transport(self.row, self.signs, request.word, site, request.sign));
        let opposite: [Option<usize>; 3] = score.selected.map_or([None; 3], |(_, _, _, link)| {
            let center = transport(self.row, self.signs, link.word, link.site, 1 - request.sign)
                .map(|p| p.1);
            [Some(link.word), Some(link.site), center]
        });
        json!({"cost":scalar_triple(score.selected.map(|p| p.0)),"site":focus,"tuple":score.selected.map(|p| p.1),
            "local":local.map(|p| p.0),"center":local.map(|p| p.1),"ties":score.ties,"eligible":score.eligible,
            "opposite":opposite,"min_sites":score.min_sites})
    }

    pub fn margin_objective(&self, request: Request, name: &str, i: usize) -> Value {
        let mut target = ObjectiveArm::new();
        let mut third = ObjectiveArm::new();
        let sign = request.sign;
        for (ordinal, t) in self.row.tuples.iter().enumerate() {
            let site = t[sign];
            if site == request.rival {
                continue;
            }
            let opposite = &self.ranks[1 - sign][t[1 - sign]];
            let Some(&other) = opposite.first() else {
                continue;
            };
            let Some(&own) = self.ranks[sign][site]
                .iter()
                .find(|link| link.word == request.word)
            else {
                continue;
            };
            let bank = t
                .iter()
                .enumerate()
                .map(|(role, &selected)| self.costs[selected][role])
                .sum();
            let score = if site == request.site {
                &mut target
            } else {
                &mut third
            };
            score.offer(
                joint_cost(own, other, bank),
                ordinal,
                own,
                other,
                rank_len(opposite),
            );
        }
        let authored = self.row.words[request.word].address;
        let a = target.selected.map(|p| p.0);
        let b = third.selected.map(|p| p.0);
        let eligible = a.zip(b);
        let opportunity = if signed_cost(
            lexical(self.row, self.signs, request.word, sign),
            lexical(self.row, self.signs, request.word, 1 - sign),
        )
        .is_none()
        {
            "target_non_strict_sign"
        } else if authored.overlaps(self.row.sites[request.site].address) {
            "target_overlap"
        } else if !self.row.tuples.iter().any(|t| t[sign] == request.site) {
            "target_no_role"
        } else if a.is_none() {
            "target_no_completion"
        } else if b.is_none() {
            "no_third"
        } else {
            "eligible"
        };
        let margin = eligible.map(|(a, b)| b[1] as i64 - a[1] as i64);
        let movable = eligible.is_some_and(|(a, b)| a[0] == b[0]);
        let debt = if movable {
            margin.map_or(0, |m| (8 - m).max(0))
        } else {
            0
        };
        let selector = if i % 2 == 0 { "first" } else { "later" };
        json!({"candidate":name,"pair":i/2,"selector":selector,"row":request.row,"sign":request.signed(),
            "word":request.word,"requested":request.site,"rival":request.rival,"opportunity":opportunity,
            "margin":margin,"movable":movable,"debt":debt,"strict_win":eligible.is_some_and(|(a,b)| a < b),
            "joint":[self.objective_arm(request,&target,true),self.objective_arm(request,&third,false)]})
    }
}
