use crate::device::{ElectionFit, ElectionPolicy};
use crate::measure::{compare, ElectionRow};
use std::collections::BTreeMap;

const PRIOR_TRAJECTORIES: &str = "8c787f0d954ab4ae2ecd37cb644e67de1dc33aa4904067edc2ce6797751b3a5a";
const POLICIES: [(ElectionPolicy, &str); 2] = [
    (ElectionPolicy::GoldOnly, "gold-only"),
    (ElectionPolicy::Full, "full"),
];
type History = BTreeMap<(String, u64), serde_json::Value>;

fn verified_trajectory(path: &Path) -> Result<History> {
    let bytes = fs::read(path)?;
    if hex::encode(Sha256::digest(&bytes)) != PRIOR_TRAJECTORIES {
        return Err(fail("historical trajectory digest mismatch"));
    }
    let mut index = History::new();
    for line in std::str::from_utf8(&bytes)?.lines() {
        let rec: serde_json::Value = serde_json::from_str(line)?;
        let key = (
            rec["arm"].as_str().ok_or("missing arm")?.to_owned(),
            rec["epoch"].as_u64().ok_or("missing epoch")?,
        );
        if index.insert(key, rec).is_some() {
            return Err(fail("duplicate historical epoch"));
        }
    }
    if index.len() != 32 {
        return Err(fail("expected 32 historical records"));
    }
    Ok(index)
}

fn verify_old_arm(a: &Arm, history: &History) -> Result<()> {
    for (i, epoch) in a.result.epochs.iter().enumerate() {
        let arm = format!("{}-{}", tag(a.mode), a.seed);
        let observed =
            json!({"arm":arm,"epoch":i+1,"stats":epoch.stats,"words":epoch.weights.as_slice()});
        if history.get(&(arm, i as u64 + 1)) != Some(&observed) {
            return Err(fail(
                "old control did not reproduce its exact words and counters",
            ));
        }
    }
    Ok(())
}

fn load_engines(path: &Path) -> Result<(Engines, [u8; 32])> {
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(path, false)?
    else {
        return Err(fail("expected frozen embedded P4"));
    };
    if identity.artifact_sha256 != ARTIFACT || expected_state_sha256 != STATE {
        return Err(fail("P4 identity mismatch"));
    }
    let model =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha256, identity)?;
    let inner = &model.p4.as_ref().ok_or("missing P4")?.inner;
    let bind = inner.state_sha256().ok_or("missing bound state")?;
    if hex::encode(bind) != STATE {
        return Err(fail("bound state mismatch"));
    }
    let raw = split_uorp4a01_sections(&state_bytes)?;
    let engines = Engines {
        lexical: Gpu::new(&model.model.p4_lexical_trunk_words()?, p4_lexical_bfs())?,
        p4: P4GpuEngine::new(
            raw.counts_v3,
            raw.s3_compact,
            raw.s2_skip_compact,
            raw.s4_table,
            &inner.s6.weights_q32,
            &inner.codes,
        )?,
    };
    Ok((engines, bind))
}

struct Domain {
    geometry: Baselines,
    ordered: Baselines,
}
impl Domain {
    fn new(gpu: &Gpu, c: &Cohort, s: &Support) -> Result<Self> {
        let geometry = baselines(gpu, c, Mode::Geometry, s)?;
        let ordered = baselines(gpu, c, Mode::Ordered, s)?;
        if geometry.zero != ordered.zero {
            return Err(fail("zero elections disagree"));
        }
        Ok(Self { geometry, ordered })
    }
    fn common(&self, i: usize) -> bool {
        self.geometry.ceiling[i].ceiling.margin_q48 > 0
            && self.ordered.ceiling[i].ceiling.margin_q48 > 0
    }
}

struct NewPair {
    geometry: ElectionFit,
    ordered: ElectionFit,
}
fn fit_new(
    gpu: &Gpu,
    c: &Cohort,
    s: &Support,
    seed: u32,
    policy: ElectionPolicy,
) -> Result<NewPair> {
    let orders = c.schedule(seed)?;
    Ok(NewPair {
        geometry: gpu.fit_all(&c.capture, &c.targets, &orders, Mode::Geometry, s, policy)?,
        ordered: gpu.fit_all(&c.capture, &c.targets, &orders, Mode::Ordered, s, policy)?,
    })
}
fn new_witness(
    gpu: &Gpu,
    sentinel: &Cohort,
    s: &Support,
    name: &str,
    mode: Mode,
    words: &[i16; 256],
    out: &mut BufWriter<File>,
) -> Result<()> {
    let w = gpu.witness(&sentinel.capture, &sentinel.targets, mode, s, words)?;
    row(
        out,
        &json!({"name":name,"addressing":tag(mode),"carry":sentinel.capture.features()[0].carry_code,"device":w[0]}),
    )
}
#[derive(Default, Serialize)]
struct Movement {
    rows: u32,
    strict_before: u32,
    immediate: u32,
    updated: u32,
    infeasible: u32,
    retained: u32,
    erased: u32,
    later_gains: u32,
    checkpoint_strict: u32,
    ties_won: u32,
    ties_lost: u32,
    changed_words: u64,
    l1: u64,
    directional_saturations: u64,
    max_radius: u32,
}
struct EpochEvidence<'a> {
    old: &'a Arm,
    new: &'a NewPair,
    domain: &'a Domain,
    cohort: &'a Cohort,
    train: &'a Train,
    name: &'a str,
    seed: u32,
}
impl EpochEvidence<'_> {
    fn write(
        &self,
        epoch: usize,
        visits: &mut BufWriter<File>,
        checkpoints: &mut BufWriter<File>,
    ) -> Result<(serde_json::Value, bool)> {
        let mut common = Vec::new();
        let mut totals = [Movement::default(), Movement::default()];
        let g = &self.new.geometry.epochs[epoch];
        let o = &self.new.ordered.epochs[epoch];
        for (i, &ordinal) in self.cohort.ordinals.iter().enumerate() {
            let group = self.train.groups[ordinal as usize];
            let old = self.old.result.epochs[epoch].checkpoint[i];
            for (j, mode, e, base) in [
                (0, Mode::Geometry, g, &self.domain.geometry),
                (1, Mode::Ordered, o, &self.domain.ordered),
            ] {
                let v = &e.visits[i];
                let end = e.checkpoint[i];
                let ceiling = base.ceiling[i].ceiling;
                let instant = v.after.margin_q48 > 0;
                if self.name == "full" && instant != (ceiling.margin_q48 > 0) {
                    return Err(fail("two-sided update disagreed with attainable ceiling"));
                }
                let m = &mut totals[j];
                m.rows += 1;
                m.strict_before += u32::from(v.before.margin_q48 > 0);
                m.immediate += u32::from(instant);
                m.updated += u32::from(v.outcome == 1);
                m.infeasible += u32::from(v.outcome == 2);
                m.changed_words += u64::from(v.changed_words);
                m.l1 += u64::from(v.l1);
                m.directional_saturations += u64::from(v.saturated_words);
                m.max_radius = m.max_radius.max(v.radius);
                m.checkpoint_strict += u32::from(end.margin_q48 > 0);
                m.retained += u32::from(instant && end.margin_q48 > 0);
                m.erased += u32::from(instant && end.margin_q48 <= 0);
                m.later_gains += u32::from(!instant && end.margin_q48 > 0);
                m.ties_won += u32::from(end.margin_q48 == 0 && end.rank == 1);
                m.ties_lost += u32::from(end.margin_q48 == 0 && end.rank > 1);
                row(
                    visits,
                    &json!({"epoch":epoch+1,"seed":self.seed,"policy":self.name,"addressing":tag(mode),"ordinal":ordinal,"group":group,
                    "gold":self.cohort.targets[i],"common":self.domain.common(i),"old_geometry":old,"zero":base.zero[i],"ceiling":ceiling,"visit":v,"checkpoint":end}),
                )?;
            }
            if self.domain.common(i) {
                common.push(ElectionRow {
                    group,
                    zero: self.domain.geometry.zero[i].margin_q48 > 0,
                    old: old.margin_q48 > 0,
                    geometry: g.checkpoint[i].margin_q48 > 0,
                    ordered: o.checkpoint[i].margin_q48 > 0,
                    geometry_immediate: g.visits[i].after.margin_q48 > 0,
                    ordered_immediate: o.visits[i].after.margin_q48 > 0,
                });
            }
        }
        for (mode, e) in [(Mode::Geometry, g), (Mode::Ordered, o)] {
            row(
                checkpoints,
                &json!({"epoch":epoch+1,"seed":self.seed,"policy":self.name,"addressing":tag(mode),"words":e.weights.as_slice()}),
            )?;
        }
        let paired = compare(&common, true)?;
        let pass = paired.pass;
        Ok((
            json!({"epoch":epoch+1,"seed":self.seed,"policy":self.name,"geometry":totals[0],"ordered":totals[1],"paired":paired}),
            pass,
        ))
    }
}

fn confirm_policy(
    engines: &Engines,
    train: &Train,
    bind: [u8; 32],
    policy: ElectionPolicy,
    name: &str,
    out: &Path,
) -> Result<bool> {
    new_dir(out)?;
    let (held_ids, fit_ids): (Vec<_>, Vec<_>) = train
        .ordinals
        .iter()
        .copied()
        .partition(|&i| input::is_withheld(train.groups[i as usize]));
    let gpu = &engines.lexical;
    let (s, heads) = {
        let fit = Cohort::new(engines, train, fit_ids)?;
        let s = support(gpu, &fit)?;
        let mut heads = Vec::new();
        for seed in SEEDS {
            let old = gpu.fit(
                &fit.capture,
                &fit.targets,
                &fit.schedule(seed)?,
                Mode::Geometry,
                &s,
            )?;
            let pair = fit_new(gpu, &fit, &s, seed, policy)?;
            heads.push((
                seed,
                old.weights,
                pair.geometry.weights,
                pair.ordered.weights,
            ));
        }
        (s, heads)
    };
    let held = Cohort::new(engines, train, held_ids)?;
    let domain = Domain::new(gpu, &held, &s)?;
    let mut endpoints = create(&out.join("endpoints.jsonl"))?;
    let mut all_pass = true;
    let mut summaries = Vec::new();
    for (seed, old_words, geo_words, ordered_words) in heads {
        let mut ends = Vec::new();
        for (label, mode, words) in [
            ("old-geometry", Mode::Geometry, &old_words),
            ("geometry", Mode::Geometry, &geo_words),
            ("ordered-matched", Mode::Ordered, &ordered_words),
        ] {
            save_head(&out.join(format!("{label}-{seed}.lex")), bind, mode, words)?;
            ends.push(gpu.evaluate(&held.capture, &held.targets, mode, &s, words)?);
        }
        let mut rows = Vec::new();
        for (i, &ordinal) in held.ordinals.iter().enumerate() {
            let group = train.groups[ordinal as usize];
            row(
                &mut endpoints,
                &json!({"seed":seed,"ordinal":ordinal,"group":group,"gold":held.targets[i],"common":domain.common(i),
                "zero":domain.geometry.zero[i],"old":ends[0][i],"geometry":ends[1][i],"ordered":ends[2][i]}),
            )?;
            if domain.common(i) {
                rows.push(ElectionRow {
                    group,
                    zero: domain.geometry.zero[i].margin_q48 > 0,
                    old: ends[0][i].margin_q48 > 0,
                    geometry: ends[1][i].margin_q48 > 0,
                    ordered: ends[2][i].margin_q48 > 0,
                    geometry_immediate: false,
                    ordered_immediate: false,
                });
            }
        }
        let grade = compare(&rows, false)?;
        all_pass &= grade.pass;
        summaries.push(json!({"seed":seed,"grade":grade}));
    }
    endpoints.flush()?;
    save(
        &out.join("confirmation.json"),
        &json!({"policy":name,"groups_disjoint":true,"rows":held.targets.len(),"pass":all_pass,"seeds":summaries}),
    )?;
    Ok(all_pass)
}

pub fn run_election(p4: &Path, root: &Path, split: &Path, replay: &Path, out: &Path) -> Result<()> {
    new_dir(out)?;
    let start = Instant::now();
    let train = input::load(root)?;
    input::verify_historical_inputs(root)?;
    input::verify_split(&train, split)?;
    if hex::encode(Sha256::digest(fs::read(split)?)) != SPLIT {
        return Err(fail("split digest mismatch"));
    }
    let history = verified_trajectory(replay)?;
    let (engines, bind) = load_engines(p4)?;
    let gpu = &engines.lexical;
    let load_ms = start.elapsed().as_millis();
    let c = Cohort::new(&engines, &train, train.ordinals.clone())?;
    let s = support(gpu, &c)?;
    let domain = Domain::new(gpu, &c, &s)?;
    let capture_ms = start.elapsed().as_millis() - load_ms;
    save(&out.join("support.json"), &s)?;
    let sentinel = Cohort::new(&engines, &train, vec![c.ordinals[0]])?;
    save(
        &out.join("sentinel-prefix.json"),
        &train.prefixes[c.ordinals[0] as usize],
    )?;
    let mut witnesses = create(&out.join("witnesses.jsonl"))?;
    save_head(&out.join("zero.lex"), bind, Mode::Geometry, &ZERO)?;
    new_witness(
        gpu,
        &sentinel,
        &s,
        "zero",
        Mode::Geometry,
        &ZERO,
        &mut witnesses,
    )?;
    let mut passes = [true, true];
    let mut visits = create(&out.join("visits.jsonl"))?;
    let mut checkpoints = create(&out.join("checkpoints.jsonl"))?;
    let mut summaries = Vec::new();
    let mut timings = Vec::new();
    for seed in SEEDS {
        let tick = Instant::now();
        let old = fit_pair(gpu, &c, &s, seed)?;
        verify_old_arm(&old.geometry, &history)?;
        verify_old_arm(&old.ordered, &history)?;
        timings.push(json!({"policy":"old","seed":seed,"fit_ms":tick.elapsed().as_millis()}));
        for (j, (policy, name)) in POLICIES.iter().enumerate() {
            let tick = Instant::now();
            let pair = fit_new(gpu, &c, &s, seed, *policy)?;
            timings.push(json!({"policy":name,"seed":seed,"fit_ms":tick.elapsed().as_millis()}));
            for (mode, arm) in [
                (Mode::Geometry, &pair.geometry),
                (Mode::Ordered, &pair.ordered),
            ] {
                let label = format!("{}-{name}-{seed}", tag(mode));
                save_head(&out.join(format!("{label}.lex")), bind, mode, &arm.weights)?;
                new_witness(
                    gpu,
                    &sentinel,
                    &s,
                    &label,
                    mode,
                    &arm.weights,
                    &mut witnesses,
                )?;
            }
            let evidence = EpochEvidence {
                old: &old.geometry,
                new: &pair,
                domain: &domain,
                cohort: &c,
                train: &train,
                name,
                seed,
            };
            for epoch in 0..8 {
                let (summary, pass) = evidence.write(epoch, &mut visits, &mut checkpoints)?;
                summaries.push(summary);
                if epoch == 7 {
                    passes[j] &= pass;
                }
            }
        }
    }
    visits.flush()?;
    checkpoints.flush()?;
    witnesses.flush()?;
    save(&out.join("epochs.json"), &summaries)?;
    drop(c);
    let mut confirmed = [None, None];
    for (j, (policy, name)) in POLICIES.iter().enumerate() {
        if passes[j] {
            confirmed[j] = Some(confirm_policy(
                &engines,
                &train,
                bind,
                *policy,
                name,
                &out.join(format!("confirmation-{name}")),
            )?);
        }
    }
    save(
        &out.join("run.json"),
        &json!({"artifact":ARTIFACT,"state":STATE,"split":SPLIT,"rows":train.targets.len(),
        "controls_verified":32,"two_seed_full_gate":{"gold_only":passes[0],"full":passes[1]},
        "confirmation":{"gold_only":confirmed[0],"full":confirmed[1]},"old_cubin":gpu.digest(),"new_cubin":gpu.election_digest(),
        "load_ms":load_ms,"capture_ms":capture_ms,"fits":timings,"elapsed_ms":start.elapsed().as_millis()}),
    )
}
