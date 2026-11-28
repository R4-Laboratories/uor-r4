use crate::{
    device::{Capture, Decision, Feasibility, FitResult, Gpu, Mode, Support},
    fail,
    grade::{self, ConfirmationRow, ConfirmationSeedInput, Election, RowImpact, SeedInput},
    input::{self, Train},
    p4_cuda::{split_uorp4a01_sections, P4GpuEngine},
    Result,
};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
    time::Instant,
};
use uor_r4_integer::discrete_d11::{
    count_deployment::{load_client_artifact, ClientArtifactFile, DiscreteCountModel},
    native_occurrence::{lexical::MATCHED_SLOT_MASK, lexical_commit::LexicalHead},
    p4_lexical_bfs,
};
const ARTIFACT: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const STATE: &str = "368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778";
const SPLIT: &str = "3b8d07c8112d696d08a91651c2ce0da1ef8acb81723cfcaec01bdea7798ee17c";
const SEEDS: [u32; 2] = [435300, 435301];
const ZERO: [i16; 256] = [0; 256];
fn new_dir(path: &Path) -> Result<()> {
    Ok(fs::DirBuilder::new().mode(0o700).create(path)?)
}
fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    ))
}
fn save(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut out = create(path)?;
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn row(out: &mut BufWriter<File>, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    Ok(())
}
struct Engines {
    lexical: Gpu,
    p4: P4GpuEngine,
}
impl Engines {
    fn capture(&self, train: &Train, ordinals: &[u32]) -> Result<Capture> {
        let prefixes: Vec<_> = ordinals
            .iter()
            .map(|&i| train.prefixes[i as usize].clone())
            .collect();
        let mut priors = Vec::with_capacity(prefixes.len() * 4095);
        for batch in prefixes.chunks(64) {
            let (steps, raw) =
                self.p4
                    .run_parity_batch(batch, &vec![1; batch.len()], &vec![true; batch.len()])?;
            if steps.len() != batch.len()
                || steps
                    .iter()
                    .zip(batch)
                    .any(|(s, h)| s.status != 0 || s.context_tokens as usize != h.len())
            {
                return Err(fail("frozen P4 scored a different causal window"));
            }
            priors.extend(raw);
        }
        self.lexical.capture(&prefixes, &priors)
    }
}
struct Cohort {
    ordinals: Vec<u32>,
    targets: Vec<u16>,
    capture: Capture,
}
impl Cohort {
    fn new(engines: &Engines, train: &Train, ordinals: Vec<u32>) -> Result<Self> {
        let capture = engines.capture(train, &ordinals)?;
        let targets = ordinals
            .iter()
            .map(|&i| train.targets[i as usize])
            .collect();
        Ok(Self {
            ordinals,
            targets,
            capture,
        })
    }
    fn schedule(&self, seed: u32) -> Result<Vec<Vec<u32>>> {
        input::orders(seed, &self.ordinals)
            .into_iter()
            .map(|epoch| {
                epoch
                    .into_iter()
                    .map(|ord| {
                        self.ordinals
                            .binary_search(&ord)
                            .map(|i| i as u32)
                            .map_err(|_| fail("epoch ordinal missing from fit cohort"))
                    })
                    .collect()
            })
            .collect()
    }
}
fn election(d: Decision) -> Election {
    Election {
        margin_q48: d.margin_q48,
        rival: d.rival,
        rank: d.rank,
    }
}
fn tag(mode: Mode) -> &'static str {
    match mode {
        Mode::Geometry => "geometry",
        Mode::Ordered => "ordered-matched",
    }
}
fn encoded(bind: [u8; 32], mode: Mode, w: &[i16; 256]) -> [u8; 584] {
    match mode {
        Mode::Geometry => LexicalHead::encode_bound(bind, false, w),
        Mode::Ordered => LexicalHead::encode_matched_bound(bind, w),
    }
}
fn save_head(path: &Path, bind: [u8; 32], mode: Mode, w: &[i16; 256]) -> Result<()> {
    if w.iter()
        .enumerate()
        .any(|(i, &v)| v != 0 && MATCHED_SLOT_MASK & (1u64 << (i % 64)) == 0)
    {
        return Err(fail("unreachable deployment word was updated"));
    }
    let bytes = encoded(bind, mode, w);
    let mut file = create(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    let actual = fs::read(path)?;
    if actual != bytes
        || LexicalHead::bind(&actual, &bind)?.addressing_name() != tag(mode)
        || LexicalHead::bind(&actual, &[0; 32]).is_ok()
    {
        return Err(fail("bound head did not restore at exact address operator"));
    }
    Ok(())
}
struct Baselines {
    ceiling: Vec<Feasibility>,
    zero: Vec<Decision>,
}
fn baselines(gpu: &Gpu, c: &Cohort, mode: Mode, s: &Support) -> Result<Baselines> {
    Ok(Baselines {
        ceiling: gpu.feasibility(&c.capture, &c.targets, mode, s)?,
        zero: gpu.evaluate(&c.capture, &c.targets, mode, s, &ZERO)?,
    })
}
struct Arm {
    seed: u32,
    mode: Mode,
    result: FitResult,
}
struct SeedArms {
    geometry: Arm,
    ordered: Arm,
}
fn fit_pair(gpu: &Gpu, c: &Cohort, s: &Support, seed: u32) -> Result<SeedArms> {
    let schedule = c.schedule(seed)?;
    let fit = |mode| -> Result<Arm> {
        Ok(Arm {
            seed,
            mode,
            result: gpu.fit(&c.capture, &c.targets, &schedule, mode, s)?,
        })
    };
    Ok(SeedArms {
        geometry: fit(Mode::Geometry)?,
        ordered: fit(Mode::Ordered)?,
    })
}
fn impacts(train: &Train, c: &Cohort, base: &Baselines, a: &Arm, e: usize) -> Vec<RowImpact> {
    let epoch = &a.result.epochs[e];
    c.ordinals
        .iter()
        .enumerate()
        .map(|(i, &ord)| {
            let v = epoch.visits[i];
            let f = base.ceiling[i];
            RowImpact {
                ordinal: ord,
                group: train.groups[ord as usize],
                gold_id: c.targets[i].into(),
                prior_rank: f.prior_rank,
                alias_count: f.alias_count,
                ceiling: election(f.ceiling),
                zero: election(base.zero[i]),
                before: election(v.before),
                before_rival_prior_q48: v.before.rival_prior_q48,
                old_rival_margin_q48: v.old_rival_margin_q48,
                after: election(v.after),
                clipped_words: v.clipped_words,
                checkpoint: election(epoch.checkpoint[i]),
            }
        })
        .collect()
}
fn support(gpu: &Gpu, c: &Cohort) -> Result<Support> {
    let support = gpu.support(&c.capture)?;
    if support.geometry != [MATCHED_SLOT_MASK; 4] || support.geometry != support.ordered {
        return Err(fail(format!(
            "fit population does not match the native deployment support {support:?}"
        )));
    }
    Ok(support)
}
fn records<'a>(rows: &'a [(Vec<RowImpact>, Vec<RowImpact>); 2]) -> [SeedInput<'a>; 2] {
    std::array::from_fn(|i| SeedInput {
        seed: SEEDS[i],
        geometry: &rows[i].0,
        ordered: &rows[i].1,
    })
}
fn confirm(engines: &Engines, train: &Train, bind: [u8; 32], out: &Path) -> Result<()> {
    new_dir(out)?;
    let (reserved, fit): (Vec<_>, Vec<_>) = train
        .ordinals
        .iter()
        .copied()
        .partition(|&i| input::is_withheld(train.groups[i as usize]));
    let c = Cohort::new(engines, train, fit)?;
    let s = support(&engines.lexical, &c)?;
    let pairs = [
        fit_pair(&engines.lexical, &c, &s, SEEDS[0])?,
        fit_pair(&engines.lexical, &c, &s, SEEDS[1])?,
    ];
    drop(c);
    let held = Cohort::new(engines, train, reserved)?;
    let mut raw = create(&out.join("endpoints.jsonl"))?;
    let mut endpoint = |a: &Arm| -> Result<Vec<ConfirmationRow>> {
        let base = baselines(&engines.lexical, &held, a.mode, &s)?;
        let end = engines.lexical.evaluate(
            &held.capture,
            &held.targets,
            a.mode,
            &s,
            &a.result.weights,
        )?;
        save_head(
            &out.join(format!("{}-{}.lex", tag(a.mode), a.seed)),
            bind,
            a.mode,
            &a.result.weights,
        )?;
        let rows = held
            .ordinals
            .iter()
            .enumerate()
            .map(|(i, &ord)| ConfirmationRow {
                ordinal: ord,
                group: train.groups[ord as usize],
                gold_id: held.targets[i].into(),
                ceiling: election(base.ceiling[i].ceiling),
                zero: election(base.zero[i]),
                checkpoint: election(end[i]),
            })
            .collect::<Vec<_>>();
        for v in &rows {
            row(&mut raw, &json!({"seed":a.seed,"mode":tag(a.mode),"row":v}))?;
        }
        Ok(rows)
    };
    let mut all = Vec::new();
    for pair in &pairs {
        all.push((endpoint(&pair.geometry)?, endpoint(&pair.ordered)?));
    }
    raw.flush()?;
    let seeds = std::array::from_fn(|i| ConfirmationSeedInput {
        seed: SEEDS[i],
        geometry: &all[i].0,
        ordered: &all[i].1,
    });
    save(&out.join("grade.json"), &grade::grade_confirmation(&seeds)?)?;
    save(&out.join("fit-support.json"), &s)
}
pub fn run(p4: &Path, root: &Path, split: &Path, out: &Path) -> Result<()> {
    let started = Instant::now();
    new_dir(out)?;
    let train = input::load(root)?;
    input::verify_historical_inputs(root)?;
    input::verify_split(&train, split)?;
    if hex::encode(Sha256::digest(fs::read(split)?)) != SPLIT {
        return Err(fail("unregistered group manifest"));
    }
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(p4, false)?
    else {
        return Err(fail("expected frozen embedded P4"));
    };
    if identity.artifact_sha256 != ARTIFACT || expected_state_sha256 != STATE {
        return Err(fail("P4 identity mismatch"));
    }
    let model =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha256, identity)?;
    let inner = &model
        .p4
        .as_ref()
        .ok_or_else(|| fail("no embedded P4 state"))?
        .inner;
    let bind = inner
        .state_sha256()
        .ok_or_else(|| fail("missing P4 state SHA"))?;
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
    let load_ms = started.elapsed().as_millis();
    let c = Cohort::new(&engines, &train, train.ordinals.clone())?;
    let gpu = &engines.lexical;
    let s = support(gpu, &c)?;
    save(&out.join("support.json"), &s)?;
    let geo = baselines(gpu, &c, Mode::Geometry, &s)?;
    let ordered = baselines(gpu, &c, Mode::Ordered, &s)?;
    let capture_ms = started.elapsed().as_millis() - load_ms;
    let pairs = [
        fit_pair(gpu, &c, &s, SEEDS[0])?,
        fit_pair(gpu, &c, &s, SEEDS[1])?,
    ];
    let fit_ms = started.elapsed().as_millis() - load_ms - capture_ms;
    let sentinel = Cohort::new(&engines, &train, vec![c.ordinals[0]])?;
    save(
        &out.join("sentinel-prefix.json"),
        &train.prefixes[c.ordinals[0] as usize],
    )?;
    let mut witness_out = create(&out.join("witnesses.jsonl"))?;
    let mut witness = |name: &str, mode, words: &[i16; 256]| -> Result<()> {
        let w = gpu.witness(&sentinel.capture, &sentinel.targets, mode, &s, words)?;
        row(
            &mut witness_out,
            &json!({"name":name,"addressing":tag(mode),"carry":sentinel.capture.features()[0].carry_code,"device":w[0]}),
        )
    };
    save_head(&out.join("zero.lex"), bind, Mode::Geometry, &ZERO)?;
    witness("zero", Mode::Geometry, &ZERO)?;
    let mut trajectories = create(&out.join("trajectories.jsonl"))?;
    for pair in &pairs {
        let a = &pair.geometry;
        let stats: Vec<_> = a.result.epochs.iter().map(|e| e.stats).collect();
        input::verify_history(
            root,
            a.seed,
            &a.result.weights,
            &serde_json::to_value(stats)?,
        )?;
        for a in [&pair.geometry, &pair.ordered] {
            let name = format!("{}-{}", tag(a.mode), a.seed);
            save_head(
                &out.join(format!("{name}.lex")),
                bind,
                a.mode,
                &a.result.weights,
            )?;
            witness(&name, a.mode, &a.result.weights)?;
            for (e, epoch) in a.result.epochs.iter().enumerate() {
                row(
                    &mut trajectories,
                    &json!({"arm":name,"epoch":e+1,"stats":epoch.stats,"words":epoch.weights.as_slice()}),
                )?;
            }
        }
    }
    trajectories.flush()?;
    witness_out.flush()?;
    let mut grades = Vec::new();
    let mut visits = create(&out.join("visits.jsonl"))?;
    for e in 0..8usize {
        let observations = std::array::from_fn(|i| {
            (
                impacts(&train, &c, &geo, &pairs[i].geometry, e),
                impacts(&train, &c, &ordered, &pairs[i].ordered, e),
            )
        });
        for (i, obs) in observations.iter().enumerate() {
            for (mode, rows) in [(Mode::Geometry, &obs.0), (Mode::Ordered, &obs.1)] {
                for v in rows {
                    row(
                        &mut visits,
                        &json!({"epoch":e+1,"seed":SEEDS[i],"mode":tag(mode),"row":v}),
                    )?;
                }
            }
        }
        grades.push(grade::grade_epoch(e as u32 + 1, &records(&observations))?);
    }
    visits.flush()?;
    let advance = grades[7]
        .advance
        .ok_or_else(|| fail("epoch-eight decision missing"))?;
    save(&out.join("epochs.json"), &grades)?;
    drop(c);
    if advance {
        confirm(&engines, &train, bind, &out.join("group-confirmation"))?;
    }
    save(
        &out.join("run.json"),
        &json!({"artifact":ARTIFACT,"state":STATE,"split":SPLIT,"rows":train.targets.len(),"geometry_reproduced_seeds":SEEDS,
        "advance":advance,"load_ms":load_ms,"capture_and_baselines_ms":capture_ms,"fits_ms":fit_ms,"elapsed_ms":started.elapsed().as_millis()}),
    )
}
