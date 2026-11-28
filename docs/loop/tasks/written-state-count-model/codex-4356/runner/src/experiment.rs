use crate::device::{CenterFit, CenteredVisit, ElectionPolicy};
use crate::measure::{compare, ElectionRow};
use serde_json::Value;
use std::io::{BufRead, BufReader, Seek};

const POLICIES: [(ElectionPolicy, &str); 2] = [
    (ElectionPolicy::GoldOnly, "gold-only"),
    (ElectionPolicy::Full, "full"),
];
const ARMS: [(Mode, bool); 4] = [
    (Mode::Geometry, false),
    (Mode::Ordered, false),
    (Mode::Geometry, true),
    (Mode::Ordered, true),
];
const EPOCHS: usize = 8;
const TRAIN_COMMON: usize = 3886;

// Members of the immutable #4355 results packet, verified on restore before use.
const PREVIOUS_CHECKPOINTS: &str =
    "233162fe0834a8641aec6e6e65cfdfdfd73ffef44ed8645038c6793ec4c4cd71";
const PREVIOUS_VISITS: &str = "1fec709219e9d7f4d80d38359c79bea616354f2847feb42ad121e2df53fee537";
const PREVIOUS_SUPPORT: &str = "5c34f7315e5ff31e266741a80e4d7d5e25a7c4fec89bc8b21faf97c09225a013";
const PREVIOUS_HEADS: [(&str, &str); 8] = [
    (
        "geometry-full-435300",
        "e5516223485f2332f1e3c4df5a2887daa7d591bb61149fe677bdc6fa5c2dc809",
    ),
    (
        "geometry-full-435301",
        "9e24e72ca30ce4d16060da28182349cfbbac0d395044d166738e441113aa3194",
    ),
    (
        "geometry-gold-only-435300",
        "7df18e1a7abcca3f1f35a3d68f290b3e4faae080263b687cc48c5229b4968896",
    ),
    (
        "geometry-gold-only-435301",
        "74f81ba0125a8b3f48f363eb1eb82647801a6d868a64e88298eb032fd941d876",
    ),
    (
        "ordered-matched-full-435300",
        "1ff45f2a5b3e2d5dd5fbe726d83cc65e03722ea398bebea34a0ffc1c060a1c85",
    ),
    (
        "ordered-matched-full-435301",
        "402a58674728b106db085b1ad8a38b563e38f0e2a6c28b296187a00b65fb093a",
    ),
    (
        "ordered-matched-gold-only-435300",
        "7d9e2bfb1d4ad60bc02e30a51a5c292a7474bc66d18d5dc15136d77c64f5d74b",
    ),
    (
        "ordered-matched-gold-only-435301",
        "6765c184ee329982a4dc24eca29bd85a3f622e8d71efeb9c8b5c51e3d252685e",
    ),
];

fn verified_file(path: &Path, expected: &str) -> Result<BufReader<File>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    loop {
        let bytes = file.fill_buf()?;
        if bytes.is_empty() {
            break;
        }
        digest.update(bytes);
        let len = bytes.len();
        file.consume(len);
    }
    if hex::encode(digest.finalize()) != expected {
        return Err(fail(format!(
            "historical checksum mismatch at {}",
            path.display()
        )));
    }
    file.rewind()?;
    Ok(file)
}

struct HistoricRows {
    label: &'static str,
    reader: BufReader<File>,
    line: String,
    checked: usize,
}

impl HistoricRows {
    fn new(root: &Path, label: &'static str, digest: &str) -> Result<Self> {
        Ok(Self {
            label,
            reader: verified_file(&root.join(label), digest)?,
            line: String::new(),
            checked: 0,
        })
    }

    fn replay(&mut self, observed: &Value) -> Result<()> {
        self.line.clear();
        if self.reader.read_line(&mut self.line)? == 0 {
            return Err(fail(format!("historical {} ended early", self.label)));
        }
        let expected: Value = serde_json::from_str(&self.line)?;
        self.checked += 1;
        if expected != *observed {
            return Err(fail(format!(
                "historical {} replay differs at record {} (seed {}, policy {}, epoch {}, ordinal {})",
                self.label, self.checked, observed["seed"], observed["policy"],
                observed["epoch"], observed["ordinal"],
            )));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<usize> {
        if !self.reader.fill_buf()?.is_empty() {
            return Err(fail(format!(
                "historical {} has unmatched records",
                self.label
            )));
        }
        Ok(self.checked)
    }
}

fn previous_words(prior: &Path, label: &str, bind: [u8; 32], mode: Mode) -> Result<[i16; 256]> {
    let (_, digest) = PREVIOUS_HEADS
        .iter()
        .find(|(name, _)| *name == label)
        .ok_or("unknown archived lexical head")?;
    let mut file = verified_file(&prior.join(format!("{label}.lex")), digest)?;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut bytes)?;
    let encoded_words = bytes
        .get(40..552)
        .ok_or("archived lexical head is truncated")?;
    let words = std::array::from_fn(|i| {
        i16::from_le_bytes([encoded_words[i * 2], encoded_words[i * 2 + 1]])
    });
    if bytes != encoded(bind, mode, &words) {
        return Err(fail("archived lexical head did not round trip exactly"));
    }
    Ok(words)
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
    Ok((
        Engines {
            lexical: Gpu::new(&model.model.p4_lexical_trunk_words()?, p4_lexical_bfs())?,
            p4: P4GpuEngine::new(
                raw.counts_v3,
                raw.s3_compact,
                raw.s2_skip_compact,
                raw.s4_table,
                &inner.s6.weights_q32,
                &inner.codes,
            )?,
        },
        bind,
    ))
}

struct Domain {
    families: [Baselines; 2],
    common: Vec<bool>,
}

impl Domain {
    fn new(gpu: &Gpu, cohort: &Cohort, slots: &Support) -> Result<Self> {
        let geometry = baselines(gpu, cohort, Mode::Geometry, slots)?;
        let ordered = baselines(gpu, cohort, Mode::Ordered, slots)?;
        // Rank and selected rival at zero must agree across address families.
        if geometry.zero != ordered.zero {
            return Err(fail("zero elections disagree"));
        }
        let common = geometry
            .ceiling
            .iter()
            .zip(&ordered.ceiling)
            .map(|(g, o)| strict(g.ceiling) && strict(o.ceiling))
            .collect();
        Ok(Self {
            families: [geometry, ordered],
            common,
        })
    }
}

fn strict(decision: Decision) -> bool {
    decision.margin_q48 > 0
}

fn label(mode: Mode, policy: &str, seed: u32, center: bool) -> String {
    let state = if center { "centered" } else { "uncentered" };
    format!("{}-{policy}-{seed}-{state}", tag(mode))
}

struct ClientWitness<'a> {
    gpu: &'a Gpu,
    sentinel: Cohort,
    slots: &'a Support,
    bind: [u8; 32],
    out: &'a Path,
    witnesses: BufWriter<File>,
}

impl<'a> ClientWitness<'a> {
    fn new(
        engines: &'a Engines,
        train: &Train,
        slots: &'a Support,
        bind: [u8; 32],
        out: &'a Path,
    ) -> Result<Self> {
        let sentinel = Cohort::new(engines, train, vec![0])?;
        save(&out.join("sentinel-prefix.json"), &train.prefixes[0])?;
        Ok(Self {
            gpu: &engines.lexical,
            sentinel,
            slots,
            bind,
            out,
            witnesses: create(&out.join("witnesses.jsonl"))?,
        })
    }

    fn export(&mut self, name: &str, mode: Mode, words: &[i16; 256]) -> Result<()> {
        save_head(
            &self.out.join(format!("{name}.lex")),
            self.bind,
            mode,
            words,
        )?;
        let witnesses = self.gpu.witness(
            &self.sentinel.capture,
            &self.sentinel.targets,
            mode,
            self.slots,
            words,
        )?;
        row(
            &mut self.witnesses,
            &json!({
                "name":name,"addressing":tag(mode),
                "carry":self.sentinel.capture.features()[0].carry_code,
                "device":witnesses.first().ok_or("missing cold witness")?,
            }),
        )
    }
}

#[derive(Default, Serialize)]
struct FrozenTotals {
    rows: u32,
    common_rows: u32,
    already_winning: u32,
    still_feasible: u32,
    rescued: u32,
    newly_blocked: u32,
    still_blocked: u32,
    rows_with_tied_alias: u32,
    tied_aliases: u64,
    rows_with_blocking_alias: u32,
    blocking_aliases: u64,
    higher_ceiling: u32,
    lower_ceiling: u32,
}

fn frozen_sparse(
    gpu: &Gpu,
    train: &Train,
    c: &Cohort,
    s: &Support,
    domain: &Domain,
    bind: [u8; 32],
    prior: &Path,
    out: &Path,
) -> Result<()> {
    let mut raw = create(&out.join("frozen-sparse.jsonl"))?;
    let mut summaries = Vec::new();
    for seed in SEEDS {
        for (i, mode) in [Mode::Geometry, Mode::Ordered].into_iter().enumerate() {
            let name = format!("{}-gold-only-{seed}", tag(mode));
            let words = previous_words(prior, &name, bind, mode)?;
            let frozen = gpu.frozen_sparse(&c.capture, &c.targets, mode, s, &words)?;
            let mut totals = [FrozenTotals::default(), FrozenTotals::default()];
            if frozen.rows.len() != c.targets.len() {
                return Err(fail("incomplete frozen sparse precheck"));
            }
            for (j, result) in frozen.rows.iter().enumerate() {
                if result.shift_failures != 0
                    || result.before != result.centered
                    || (strict(result.before)
                        && (!strict(result.before_ceiling) || !strict(result.centered_ceiling)))
                    || (result.blocking_aliases > 0
                        && (strict(result.before_ceiling) || strict(result.centered_ceiling)))
                    || result.before_ceiling.margin_q48 < result.before.margin_q48
                    || result.centered_ceiling.margin_q48 < result.centered.margin_q48
                {
                    return Err(fail("frozen sparse election or ceiling invariant failed"));
                }
                let class = match (
                    strict(result.before),
                    strict(result.before_ceiling),
                    strict(result.centered_ceiling),
                ) {
                    (true, _, _) => "already_winning",
                    (false, true, true) => "still_feasible",
                    (false, false, true) => "rescued",
                    (false, true, false) => "newly_blocked",
                    (false, false, false) => "still_blocked",
                };
                let ordinal = c.ordinals[j];
                let common = domain.common[j];
                for m in totals.iter_mut().take(1 + usize::from(common)) {
                    m.rows += 1;
                    m.common_rows += u32::from(common);
                    m.already_winning += u32::from(class == "already_winning");
                    m.still_feasible += u32::from(class == "still_feasible");
                    m.rescued += u32::from(class == "rescued");
                    m.newly_blocked += u32::from(class == "newly_blocked");
                    m.still_blocked += u32::from(class == "still_blocked");
                    m.rows_with_tied_alias += u32::from(result.tied_aliases != 0);
                    m.tied_aliases += u64::from(result.tied_aliases);
                    m.rows_with_blocking_alias += u32::from(result.blocking_aliases != 0);
                    m.blocking_aliases += u64::from(result.blocking_aliases);
                    m.higher_ceiling += u32::from(
                        result.centered_ceiling.margin_q48 > result.before_ceiling.margin_q48,
                    );
                    m.lower_ceiling += u32::from(
                        result.centered_ceiling.margin_q48 < result.before_ceiling.margin_q48,
                    );
                }
                row(
                    &mut raw,
                    &json!({
                        "seed":seed,"addressing":tag(mode),"ordinal":ordinal,"group":train.groups[ordinal as usize],
                        "gold":c.targets[j],"common":common,"classification":class,
                        "attainable":domain.families[i].ceiling[j].ceiling,"precheck":result,
                    }),
                )?;
            }
            save_head(
                &out.join(format!("frozen-{name}-centered.lex")),
                bind,
                mode,
                &frozen.weights,
            )?;
            summaries.push(
                json!({"seed":seed,"addressing":tag(mode),"gauge":frozen.gauge,
                "all":totals[0],"common":totals[1]}),
            );
        }
    }
    raw.flush()?;
    save(&out.join("frozen-sparse-summary.json"), &summaries)
}

#[derive(Default, Serialize)]
struct Age {
    immediate: u32,
    retained: u32,
    erased: u32,
    newly_corrected: u32,
    newly_corrected_retained: u32,
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
    newly_corrected: u32,
    newly_corrected_retained: u32,
    checkpoint_strict: u32,
    ties_won: u32,
    ties_lost: u32,
    learning_changed_words: u64,
    learning_l1: u64,
    learning_directional_rail_hits: u64,
    learning_clipped_words: u64,
    gauge_visits: u32,
    gauge_changed_words: u64,
    gauge_l1: u64,
    max_radius: u32,
    endpoint_low_rail: [u32; 4],
    endpoint_high_rail: [u32; 4],
    age_quartiles: [Age; 4],
}

impl Movement {
    fn observe(&mut self, visit: &CenteredVisit, end: Decision, age: usize, rows: usize) {
        let v = visit.learning;
        let instant = strict(v.after);
        let landed = strict(end);
        let corrected = !strict(v.before) && instant;
        let retained = instant && landed;
        let erased = instant && !landed;
        self.rows += 1;
        self.strict_before += u32::from(strict(v.before));
        self.immediate += u32::from(instant);
        self.updated += u32::from(v.outcome == 1);
        self.infeasible += u32::from(v.outcome == 2);
        self.retained += u32::from(retained);
        self.erased += u32::from(erased);
        self.later_gains += u32::from(!instant && landed);
        self.newly_corrected += u32::from(corrected);
        self.newly_corrected_retained += u32::from(corrected && landed);
        self.checkpoint_strict += u32::from(landed);
        self.ties_won += u32::from(end.margin_q48 == 0 && end.rank == 1);
        self.ties_lost += u32::from(end.margin_q48 == 0 && end.rank > 1);
        self.learning_changed_words += u64::from(v.changed_words);
        self.learning_l1 += u64::from(v.l1);
        self.learning_directional_rail_hits += u64::from(v.saturated_words);
        self.learning_clipped_words += u64::from(visit.clipped_learning_words);
        self.gauge_visits += u32::from(visit.gauge.changed_words != 0);
        self.gauge_changed_words += u64::from(visit.gauge.changed_words);
        self.gauge_l1 += u64::from(visit.gauge.l1);
        self.max_radius = self.max_radius.max(v.radius);
        let bucket = &mut self.age_quartiles[(age * 4) / rows];
        bucket.immediate += u32::from(instant);
        bucket.retained += u32::from(retained);
        bucket.erased += u32::from(erased);
        bucket.newly_corrected += u32::from(corrected);
        bucket.newly_corrected_retained += u32::from(corrected && landed);
    }

    fn rails(&mut self, words: &[i16; 256]) {
        for (i, &word) in words.iter().enumerate() {
            if MATCHED_SLOT_MASK & (1u64 << (i % 64)) != 0 {
                self.endpoint_low_rail[i / 64] += u32::from(word == i16::MIN);
                self.endpoint_high_rail[i / 64] += u32::from(word == i16::MAX);
            }
        }
    }
}

struct EpochReplays {
    checkpoints: HistoricRows,
    visits: HistoricRows,
}

struct OutputRows {
    checkpoints: BufWriter<File>,
    visits: BufWriter<File>,
}

struct EpochEvidence<'a> {
    fits: &'a [CenterFit],
    old: &'a FitResult,
    domain: &'a Domain,
    cohort: &'a Cohort,
    train: &'a Train,
    seed: u32,
    policy: &'a str,
    orders: &'a [Vec<u32>],
    positions: &'a [Vec<usize>],
}

impl EpochEvidence<'_> {
    fn write(
        &self,
        epoch: usize,
        old: &mut EpochReplays,
        out: &mut OutputRows,
    ) -> Result<(Value, bool)> {
        let rows = self.cohort.ordinals.len();
        let mut frequency: [Movement; 4] = std::array::from_fn(|_| Movement::default());
        let mut common = Vec::new();
        let evidence: Vec<_> = self.fits.iter().map(|arm| &arm.epochs[epoch]).collect();
        let suffix_updates: Vec<_> = evidence
            .iter()
            .map(|e| {
                let mut suffix = vec![0; rows];
                let mut subsequent = 0;
                for &ord in self.orders[epoch].iter().rev() {
                    let ord = ord as usize;
                    suffix[ord] = subsequent;
                    subsequent += u32::from(e.visits[ord].learning.outcome == 1);
                }
                suffix
            })
            .collect();
        for (i, &ordinal) in self.cohort.ordinals.iter().enumerate() {
            let group = self.train.groups[ordinal as usize];
            let gold = self.cohort.targets[i];
            let is_common = self.domain.common[i];
            let old_end = self.old.epochs[epoch].checkpoint[i];
            let visit_index = self.positions[epoch][i];
            let age_visits = rows - 1 - visit_index;
            let visits_since_prior = epoch
                .checked_sub(1)
                .map(|prev| rows + visit_index - self.positions[prev][i]);
            for (arm, (mode, centered)) in ARMS.into_iter().enumerate() {
                let e = evidence[arm];
                let base = &self.domain.families[arm % 2];
                let ceiling = base.ceiling[i].ceiling;
                let v = &e.visits[i];
                let end = e.checkpoint[i];
                if v.before_center != v.learning.before
                    || v.shift_failure != 0
                    || v.gold_centered_q48 - v.gold_before_q48 != v.gauge.score_q48
                    || (!centered
                        && (v.gauge.changed_words != 0
                            || v.gauge.score_q48 != 0
                            || v.gauge.offsets != [0; 4]))
                    || (self.policy == "full" && strict(v.learning.after) != strict(ceiling))
                {
                    return Err(fail(format!(
                        "election invariant failed for seed {} {} {} epoch {} ordinal {ordinal}",
                        self.seed,
                        self.policy,
                        tag(mode),
                        epoch + 1
                    )));
                }
                frequency[arm].observe(v, end, age_visits, rows);
                let mut record = json!({
                    "epoch":epoch+1,"seed":self.seed,"policy":self.policy,"addressing":tag(mode),
                    "ordinal":ordinal,"group":group,"gold":gold,"common":is_common,
                    "old_geometry":old_end,"zero":base.zero[i],"ceiling":ceiling,
                    "visit":v.learning,"checkpoint":end,
                });
                if !centered {
                    old.visits.replay(&record)?;
                }
                record["visit"] = serde_json::to_value(v)?;
                record["centered"] = json!(centered);
                record["visit_index"] = json!(visit_index);
                record["age_visits"] = json!(age_visits);
                record["visits_since_prior"] = json!(visits_since_prior);
                record["age_learning_updates"] = json!(suffix_updates[arm][i]);
                row(&mut out.visits, &record)?;
            }
            if is_common {
                let centered_visit = evidence[2].visits[i].learning;
                common.push(ElectionRow {
                    group,
                    frozen: strict(self.domain.families[0].zero[i]),
                    historical: strict(old_end),
                    uncentered: strict(evidence[0].checkpoint[i]),
                    centered: strict(evidence[2].checkpoint[i]),
                    ordered: strict(evidence[3].checkpoint[i]),
                    centered_immediate: strict(centered_visit.after),
                    centered_newly_corrected: centered_visit.outcome == 1,
                });
            }
        }

        let mut arms = Vec::new();
        for (arm, (mode, centered)) in ARMS.into_iter().enumerate() {
            let e = evidence[arm];
            frequency[arm].rails(&e.weights);
            let mut checkpoint = json!({
                "epoch":epoch+1,"seed":self.seed,"policy":self.policy,
                "addressing":tag(mode),"words":e.weights.as_slice(),
            });
            if !centered {
                old.checkpoints.replay(&checkpoint)?;
            }
            checkpoint["centered"] = json!(centered);
            row(&mut out.checkpoints, &checkpoint)?;
            arms.push(json!({
                "addressing":tag(mode),"centered":centered,"movement":frequency[arm],
                "elapsed_ns":e.elapsed_ns,"checkpoint_bytes":512,"learning_stats":e.stats,
            }));
        }
        let paired = compare(&common, true)?;
        let pass = paired.pass;
        Ok((
            json!({
                "epoch":epoch+1,"seed":self.seed,"policy":self.policy,"arms":arms,"paired":paired,
            }),
            pass,
        ))
    }
}

fn positions(orders: &[Vec<u32>], rows: usize) -> Result<Vec<Vec<usize>>> {
    if orders.len() != EPOCHS {
        return Err(fail("unexpected number of locked TRAIN schedules"));
    }
    orders
        .iter()
        .map(|order| {
            let mut positions = vec![usize::MAX; rows];
            for (i, &row) in order.iter().enumerate() {
                let position = positions
                    .get_mut(row as usize)
                    .ok_or("schedule row is out of range")?;
                if std::mem::replace(position, i) != usize::MAX {
                    return Err(fail("duplicate row in sealed schedule"));
                }
            }
            if positions.contains(&usize::MAX) {
                return Err(fail("sealed schedule omitted an admitted row"));
            }
            Ok(positions)
        })
        .collect()
}

fn repeat_verified_history(
    gpu: &Gpu,
    c: &Cohort,
    slots: &Support,
    seed: u32,
    orders: &[Vec<u32>],
    root: &Path,
) -> Result<FitResult> {
    let historical = gpu.fit(&c.capture, &c.targets, orders, Mode::Geometry, slots)?;
    let stats: Vec<_> = historical.epochs.iter().map(|e| &e.stats).collect();
    input::verify_history(
        root,
        seed,
        &historical.weights,
        &serde_json::to_value(stats)?,
    )?;
    Ok(historical)
}

fn fit_arm(
    gpu: &Gpu,
    c: &Cohort,
    slots: &Support,
    orders: &[Vec<u32>],
    policy: ElectionPolicy,
    mode: Mode,
    centered: bool,
) -> Result<CenterFit> {
    let fit = gpu.fit_center(
        &c.capture, &c.targets, orders, mode, slots, policy, centered,
    )?;
    if fit.epochs.len() != EPOCHS
        || fit.epochs.iter().any(|epoch| {
            epoch.visits.len() != c.targets.len() || epoch.checkpoint.len() != c.targets.len()
        })
        || fit.epochs[EPOCHS - 1].weights != fit.weights
    {
        return Err(fail(
            "incomplete centered-fit telemetry or extra final translation",
        ));
    }
    Ok(fit)
}

fn group_confirm(
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
    if (fit_ids.len(), held_ids.len()) != (3104, 787) {
        return Err(fail("sealed group-disjoint partition denominators changed"));
    }
    let gpu = &engines.lexical;
    let (slots, fitted) = {
        let c = Cohort::new(engines, train, fit_ids)?;
        let slots = support(gpu, &c)?;
        let mut fitted = Vec::new();
        for seed in SEEDS {
            let orders = c.schedule(seed)?;
            let start = Instant::now();
            let old = gpu.fit(&c.capture, &c.targets, &orders, Mode::Geometry, &slots)?;
            let mut heads = vec![(
                "old_geometry",
                Mode::Geometry,
                old.weights,
                start.elapsed().as_nanos(),
            )];
            for (label, mode, centered) in [
                ("uncentered_geometry", Mode::Geometry, false),
                ("centered_geometry", Mode::Geometry, true),
                ("centered_ordered", Mode::Ordered, true),
            ] {
                let start = Instant::now();
                let fit = fit_arm(gpu, &c, &slots, &orders, policy, mode, centered)?;
                heads.push((label, mode, fit.weights, start.elapsed().as_nanos()));
            }
            fitted.push((seed, heads));
        }
        (slots, fitted)
    };
    save(&out.join("support.json"), &slots)?;
    let held = Cohort::new(engines, train, held_ids)?;
    let domain = Domain::new(gpu, &held, &slots)?;
    let mut raw = create(&out.join("endpoints.jsonl"))?;
    let mut pass = true;
    let mut summaries = Vec::new();
    for (seed, heads) in fitted {
        let mut decisions = Vec::new();
        let mut timings = Vec::new();
        for (label, mode, words, elapsed_ns) in heads {
            save_head(&out.join(format!("{label}-{seed}.lex")), bind, mode, &words)?;
            decisions.push(gpu.evaluate(&held.capture, &held.targets, mode, &slots, &words)?);
            timings.push(json!({"arm":label,"fit_elapsed_ns":elapsed_ns,"head_bytes":584}));
        }
        let mut common = Vec::new();
        for (i, &ordinal) in held.ordinals.iter().enumerate() {
            let group = train.groups[ordinal as usize];
            let frozen = domain.families[0].zero[i];
            row(
                &mut raw,
                &json!({
                    "seed":seed,"ordinal":ordinal,"group":group,"gold":held.targets[i],"common":domain.common[i],
                    "zero":frozen,"old_geometry":decisions[0][i],"uncentered_geometry":decisions[1][i],
                    "centered_geometry":decisions[2][i],"centered_ordered":decisions[3][i],
                }),
            )?;
            if domain.common[i] {
                common.push(ElectionRow {
                    group,
                    frozen: strict(frozen),
                    historical: strict(decisions[0][i]),
                    uncentered: strict(decisions[1][i]),
                    centered: strict(decisions[2][i]),
                    ordered: strict(decisions[3][i]),
                    centered_immediate: false,
                    centered_newly_corrected: false,
                });
            }
        }
        let grade = compare(&common, false)?;
        pass &= grade.pass;
        summaries.push(json!({"seed":seed,"grade":grade,"fits":timings}));
    }
    raw.flush()?;
    save(
        &out.join("confirmation.json"),
        &json!({
            "policy":name,"groups_disjoint":true,"fit_rows":3104,"held_rows":787,
            "pass":pass,"seeds":summaries,
        }),
    )?;
    Ok(pass)
}

pub fn run_center(p4: &Path, root: &Path, split: &Path, prior: &Path, out: &Path) -> Result<()> {
    new_dir(out)?;
    let start = Instant::now();
    let train = crate::verify_input(root, split)?;
    if hex::encode(Sha256::digest(fs::read(split)?)) != SPLIT {
        return Err(fail("split digest mismatch"));
    }
    let mut replays = EpochReplays {
        checkpoints: HistoricRows::new(prior, "checkpoints.jsonl", PREVIOUS_CHECKPOINTS)?,
        visits: HistoricRows::new(prior, "visits.jsonl", PREVIOUS_VISITS)?,
    };
    let previous_support: Value = serde_json::from_reader(verified_file(
        &prior.join("support.json"),
        PREVIOUS_SUPPORT,
    )?)?;
    let (engines, bind) = load_engines(p4)?;
    let gpu = &engines.lexical;
    let load_ns = start.elapsed().as_nanos();
    let c = Cohort::new(&engines, &train, train.ordinals.clone())?;
    let slots = support(gpu, &c)?;
    if serde_json::to_value(&slots)? != previous_support {
        return Err(fail("TRAIN support differs from the #4355 packet"));
    }
    let domain = Domain::new(gpu, &c, &slots)?;
    if domain.common.iter().filter(|&&member| member).count() != TRAIN_COMMON {
        return Err(fail(
            "registered 3,886-row common attainable domain changed",
        ));
    }
    let capture_and_baselines_ns = start.elapsed().as_nanos() - load_ns;
    save(&out.join("support.json"), &slots)?;

    let tick = Instant::now();
    frozen_sparse(gpu, &train, &c, &slots, &domain, bind, prior, out)?;
    let frozen_sparse_ns = tick.elapsed().as_nanos();

    let mut cold = ClientWitness::new(&engines, &train, &slots, bind, out)?;
    cold.export("zero", Mode::Geometry, &ZERO)?;
    let mut raw = OutputRows {
        visits: create(&out.join("visits.jsonl"))?,
        checkpoints: create(&out.join("checkpoints.jsonl"))?,
    };
    let mut passes = [true, true];
    let mut summaries = Vec::new();
    let mut fit_timings = Vec::new();
    let mut reference_timings = Vec::new();
    for seed in SEEDS {
        let orders = c.schedule(seed)?;
        let positions = positions(&orders, c.targets.len())?;
        let tick = Instant::now();
        let historical = repeat_verified_history(gpu, &c, &slots, seed, &orders, root)?;
        reference_timings
            .push(json!({"seed":seed,"fit_elapsed_ns":tick.elapsed().as_nanos(),"head_bytes":584}));
        cold.export(
            &format!("historical-geometry-{seed}"),
            Mode::Geometry,
            &historical.weights,
        )?;
        for (j, (policy, name)) in POLICIES.into_iter().enumerate() {
            let mut fits = Vec::new();
            for (mode, centered) in ARMS {
                let tick = Instant::now();
                let fit = fit_arm(gpu, &c, &slots, &orders, policy, mode, centered)?;
                fit_timings.push(json!({
                    "seed":seed,"policy":name,"addressing":tag(mode),"centered":centered,
                    "fit_elapsed_ns":tick.elapsed().as_nanos(),"head_bytes":584,
                }));
                if !centered {
                    let past_label = format!("{}-{name}-{seed}", tag(mode));
                    let past = previous_words(prior, &past_label, bind, mode)?;
                    if past != fit.weights {
                        return Err(fail(format!(
                            "archived attachment did not replay for {past_label}"
                        )));
                    }
                }
                cold.export(&label(mode, name, seed, centered), mode, &fit.weights)?;
                fits.push(fit);
            }
            let evidence = EpochEvidence {
                fits: &fits,
                old: &historical,
                domain: &domain,
                cohort: &c,
                train: &train,
                seed,
                policy: name,
                orders: &orders,
                positions: &positions,
            };
            for epoch in 0..EPOCHS {
                let (summary, pass) = evidence.write(epoch, &mut replays, &mut raw)?;
                summaries.push(summary);
                if epoch == EPOCHS - 1 {
                    passes[j] &= pass;
                }
            }
        }
    }
    let replayed_words = replays.checkpoints.finish()?;
    let replayed_visits = replays.visits.finish()?;
    if replayed_words != 64 || replayed_visits != 249024 {
        return Err(fail("unexpected number of verified historical records"));
    }
    raw.visits.flush()?;
    raw.checkpoints.flush()?;
    cold.witnesses.flush()?;
    save(&out.join("epochs.json"), &summaries)?;

    let conflicts_requested = !passes.iter().all(|&passed| passed);
    if conflicts_requested {
        crate::conflicts::audit(
            gpu,
            &c.capture,
            &c.targets,
            &c.ordinals,
            &train.groups,
            &domain.common,
            &slots,
            &out.join("joint-conflicts"),
        )?;
    }
    drop(c);

    let mut confirmed = [None, None];
    for (j, (policy, name)) in POLICIES.into_iter().enumerate() {
        if passes[j] {
            confirmed[j] = Some(group_confirm(
                &engines,
                &train,
                bind,
                policy,
                name,
                &out.join(format!("confirmation-{name}")),
            )?);
        }
    }
    save(
        &out.join("run.json"),
        &json!({
            "artifact":ARTIFACT,"state":STATE,"split":SPLIT,"rows":train.targets.len(),
            "common_rows":TRAIN_COMMON,"controls_verified":replayed_words,"historical_visits_verified":replayed_visits,
            "previous_checkpoint_sha256":PREVIOUS_CHECKPOINTS,"previous_visits_sha256":PREVIOUS_VISITS,
            "two_seed_gate":{"gold_only":passes[0],"full":passes[1]},
            "confirmation":{"gold_only":confirmed[0],"full":confirmed[1]},
            "joint_conflicts_audit":conflicts_requested,
            "retention_cubin":gpu.digest(),"full_election_cubin":gpu.election_digest(),"center_cubin":gpu.center_digest(),
            "load_ns":load_ns,"capture_and_baselines_ns":capture_and_baselines_ns,"frozen_sparse_ns":frozen_sparse_ns,
            "fits":fit_timings,"historical_reference_fits":reference_timings,
            "elapsed_ns":start.elapsed().as_nanos(),
            "visits_bytes":fs::metadata(out.join("visits.jsonl"))?.len(),
            "checkpoints_bytes":fs::metadata(out.join("checkpoints.jsonl"))?.len(),
        }),
    )
}
