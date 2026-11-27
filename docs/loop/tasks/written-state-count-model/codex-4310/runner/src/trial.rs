use crate::gpu::{Batch, Choice, Evaluation, Fit, Gpu, Mode, Output, CELLS, WORDS};
use crate::p4_cuda::{split_uorp4a01_sections, P4GpuEngine, ParityStepOutHost};
use crate::sample::{sample_digest, Corpus, Position, DEV_SHA, TRAIN_SHA};
use crate::{fail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    ffi::OsString,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
use uor_r4_count_runtime::p4_serving::{P4Session, P4State, S4SourceGate, S4SourceMode};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, ClientArtifactFile, DiscreteCountModel,
};

const P4_SHA: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const P4_STATE_SHA: &str = "368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778";
const R_TARGETS: usize = 4_096;
const DEV_ROWS: usize = 512;
const R_SEED: u64 = 431000;
const DEV_SEED: u64 = 431001;
const ROW_BATCH: usize = 128;
const ZERO: [i8; CELLS] = [0; CELLS];
const FIT_PANEL_SHA: &str = "e5b1c14eb96686b7fdee3f78bd70634f79b114e2912d5c53aca635c791edcc52";
const HELD_PANEL_SHA: &str = "d8630bf60f06cc65ff4b32732cf5b1b5aba2a2837008408a207e32b950d24ef0";

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    device_mode: Mode,
    client_mode: Option<S4SourceMode>,
    null_weights: bool,
}

const ARMS: [Arm; 6] = [
    Arm {
        name: "frozen",
        device_mode: Mode::Disabled,
        client_mode: None,
        null_weights: true,
    },
    Arm {
        name: "zero",
        device_mode: Mode::Disabled,
        client_mode: Some(S4SourceMode::Disabled),
        null_weights: false,
    },
    Arm {
        name: "learned",
        device_mode: Mode::Active,
        client_mode: Some(S4SourceMode::Active),
        null_weights: false,
    },
    Arm {
        name: "wrong-address",
        device_mode: Mode::WrongAddress,
        client_mode: Some(S4SourceMode::WrongAddress),
        null_weights: false,
    },
    Arm {
        name: "shuffled-role",
        device_mode: Mode::ShuffledRole,
        client_mode: Some(S4SourceMode::ShuffledRole),
        null_weights: false,
    },
    Arm {
        name: "role-blind",
        device_mode: Mode::RoleBlind,
        client_mode: Some(S4SourceMode::RoleBlind),
        null_weights: false,
    },
];

#[derive(Deserialize)]
struct PanelRow {
    id: String,
    domain: String,
    split: String,
    prefix: Vec<u16>,
    gold: Vec<u16>,
    #[serde(default)]
    free_run: bool,
    donor_opening_exposed: bool,
}

#[derive(Debug, Serialize)]
struct GateTrace {
    resolved_role: Option<u8>,
    effective_role: Option<u8>,
    gate_index: Option<u8>,
    source_pos: Option<u32>,
    source_token: Option<u16>,
    rival_token: Option<u16>,
    weight_i8: i8,
    transfer_q48: u64,
    source_mass_before_q48: Option<u64>,
    rival_mass_before_q48: Option<u64>,
}

fn yes_pos(value: u32) -> Option<u32> {
    (value != u32::MAX).then_some(value)
}

impl From<Choice> for GateTrace {
    fn from(gate: Choice) -> Self {
        Self {
            resolved_role: (gate.resolved_role != 0).then_some(gate.resolved_role as u8),
            effective_role: (gate.effective_role != 0).then_some(gate.effective_role as u8),
            gate_index: yes_pos(gate.gate_index).map(|i| i as u8),
            source_pos: yes_pos(gate.source_pos),
            source_token: (gate.source != 0).then_some(gate.source as u16),
            rival_token: (gate.rival != 0).then_some(gate.rival as u16),
            weight_i8: gate.weight as i8,
            transfer_q48: gate.transfer,
            source_mass_before_q48: (gate.source != 0).then_some(gate.source_before),
            rival_mass_before_q48: (gate.rival != 0).then_some(gate.rival_before),
        }
    }
}

#[derive(Clone, Default, Serialize)]
struct Counts {
    n_targets: u64,
    top1: u64,
    brier_q32: u64,
    prll_bits_q12: u64,
    present_targets: u64,
    present_top1: u64,
    present_brier_q32: u64,
    present_prll_bits_q12: u64,
    absent_targets: u64,
    absent_top1: u64,
    absent_brier_q32: u64,
    absent_prll_bits_q12: u64,
    eos_targets: u64,
    eos_top1: u64,
    role_positive: u64,
    admitted: u64,
    active: u64,
    top1_changed: u64,
    gained: u64,
    lost: u64,
    admitted_by_cell: [u64; CELLS],
    zero_weight: u64,
    no_wrong_address: u64,
    conditioned_pooled_weight_diff: u64,
    conditioned_pooled_score_diff: u64,
}

impl Counts {
    fn observe(&mut self, scored: Evaluation, baseline: Evaluation, gold: u16, present: bool) {
        let correct = scored.token == u32::from(gold);
        let was_correct = baseline.token == u32::from(gold);
        let ok = u64::from(correct);
        self.n_targets += 1;
        self.top1 += ok;
        self.brier_q32 += scored.brier;
        self.prll_bits_q12 += scored.prll_bits;
        self.role_positive += u64::from(scored.gate.resolved_role != 0);
        self.admitted += u64::from(scored.gate.source != 0);
        self.active += u64::from(scored.gate.transfer != 0);
        if let Some(index) = yes_pos(scored.gate.gate_index) {
            if scored.gate.source != 0 {
                self.admitted_by_cell[index as usize] += 1;
                self.zero_weight += u64::from(scored.gate.weight == 0);
            } else {
                self.no_wrong_address += 1;
            }
        }
        self.top1_changed += u64::from(scored.token != baseline.token);
        self.gained += u64::from(correct && !was_correct);
        self.lost += u64::from(!correct && was_correct);
        if gold == 1 {
            self.eos_targets += 1;
            self.eos_top1 += ok;
        } else if present {
            self.present_targets += 1;
            self.present_top1 += ok;
            self.present_brier_q32 += scored.brier;
            self.present_prll_bits_q12 += scored.prll_bits;
        } else {
            self.absent_targets += 1;
            self.absent_top1 += ok;
            self.absent_brier_q32 += scored.brier;
            self.absent_prll_bits_q12 += scored.prll_bits;
        }
    }
}

#[derive(Serialize)]
struct RecordResult<'a> {
    id: &'a str,
    domain: &'a str,
    arm: &'static str,
    donor_opening_exposed: bool,
    #[serde(flatten)]
    counts: Counts,
}

struct Runner<'a> {
    gpu: Gpu,
    p4_gpu: P4GpuEngine,
    p4_cpu: &'a P4State<'a>,
}

fn hash_words(words: &[u64]) -> String {
    let mut h = Sha256::new();
    for &word in words {
        h.update(word.to_le_bytes());
    }
    hex::encode(h.finalize())
}

fn prefix_sha256(prefix: &[u16]) -> String {
    let mut h = Sha256::new();
    for &id in prefix {
        h.update(id.to_le_bytes());
    }
    hex::encode(h.finalize())
}

fn open_panel(path: &Path, split: &str) -> Result<(Vec<PanelRow>, [u8; 32])> {
    let bytes = fs::read(path)?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    if hex::encode(digest)
        != if split == "fit" {
            FIT_PANEL_SHA
        } else {
            HELD_PANEL_SHA
        }
    {
        return Err(fail(format!("unexpected frozen {split} panel bytes")));
    }
    let content = std::str::from_utf8(&bytes)?;
    let mut seen = BTreeSet::new();
    let mut panel = Vec::new();
    for line in content.lines() {
        let row: PanelRow = serde_json::from_str(line)?;
        let domain_ok = match row.domain.as_str() {
            "conversation" => (8..=128).contains(&row.gold.len()) && row.gold.last() == Some(&1),
            "prose" => row.gold.len() == 64 && row.gold.iter().all(|&id| id != 1),
            _ => false,
        };
        if row.id.is_empty()
            || row.split != split
            || !domain_ok
            || !seen.insert(row.id.clone())
            || row.prefix.first() != Some(&0)
            || row.prefix.len() > 1536
            || row.prefix.last().is_some_and(|&t| t < 2)
            || row.prefix[1..].iter().any(|&t| t == 0 || t >= 4096)
            || (row.domain == "prose" && row.prefix.contains(&1))
            || !row
                .gold
                .iter()
                .any(|&token| token > 1 && !row.prefix.contains(&token))
            || (row.donor_opening_exposed
                && (row.domain != "conversation" || split != "held" || row.free_run))
            || row
                .gold
                .iter()
                .enumerate()
                .any(|(i, &t)| t == 0 || t >= 4096 || (t == 1 && i + 1 != row.gold.len()))
            || (split == "fit" && row.free_run)
        {
            return Err(fail(format!("invalid panel record {}", row.id)));
        }
        panel.push(row);
    }
    for domain in ["conversation", "prose"] {
        let n = panel.iter().filter(|row| row.domain == domain).count();
        if (split == "fit" && n != 64) || (split == "held" && ![64, 128].contains(&n)) {
            return Err(fail(format!("unregistered {split}/{domain} count {n}")));
        }
    }
    if split == "held" {
        let selected = |domain: &str, lo: usize, hi: usize| {
            panel
                .iter()
                .filter(|row| {
                    row.domain == domain && row.free_run && (lo..=hi).contains(&row.gold.len())
                })
                .count()
        };
        if selected("conversation", 8, 48) != 8
            || selected("conversation", 49, 128) != 16
            || selected("prose", 64, 64) != 24
        {
            return Err(fail(
                "held free-run preselection must be 8 short, 16 long, and 24 prose",
            ));
        }
    }
    Ok((panel, digest))
}

fn fit_identity(r_draw: &str, fit: &[u8; 32]) -> Result<[u8; 32]> {
    let mut h = Sha256::new();
    h.update(b"UORS4G01 train inputs v1\0");
    h.update(hex::decode(TRAIN_SHA)?);
    h.update(hex::decode(r_draw)?);
    h.update(fit);
    Ok(h.finalize().into())
}

fn active_window(history: &[u16]) -> &[u16] {
    &history[history.len().saturating_sub(2048)..]
}

fn teacher_inputs(row: &PanelRow) -> Vec<Vec<u16>> {
    let mut history = row.prefix.clone();
    row.gold
        .iter()
        .map(|&token| {
            let prefix = active_window(&history).to_vec();
            history.push(token);
            prefix
        })
        .collect()
}

fn corpus_inputs(corpus: &Corpus, positions: &[Position]) -> (Vec<Vec<u16>>, Vec<u16>) {
    positions
        .iter()
        .map(|&position| {
            (
                active_window(corpus.prefix(position)).to_vec(),
                corpus.ids[position.record + position.target_at],
            )
        })
        .unzip()
}

fn emit(writer: &mut impl Write, row: impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, &row)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn trace_step(
    route: &ParityStepOutHost,
    step: Evaluation,
    prefix: &[u16],
    words: &[u64],
    frozen: bool,
) -> Value {
    json!({
        "context_tokens": route.context_tokens,
        "prefix_sha256": prefix_sha256(prefix),
        "highest_order": route.highest_order,
        "pick_pos": (route.pick_has != 0).then_some(route.pick_pos),
        "pick_distance": (route.pick_has != 0).then_some(route.pick_distance),
        "pick_successor": (route.pick_has != 0).then_some(route.pick_successor),
        "pick_class": route.pick_class,
        "pick_ties": route.pick_ties,
        "s3_pkey": route.s3_pkey,
        "s3_backoff_key": route.s3_backoff_key,
        "s3_hit_kind": route.s3_hit_kind,
        "s2_skip_keys": route.s2_skip_keys,
        "s2_skip_hit_mask": route.s2_skip_hit_mask,
        "cache_hit_state": route.cache_hit_state,
        "bucket": route.bucket,
        "s4_role_scores": route.s4_role_scores,
        "s4_role_choice": (route.s4_role_choice != 0).then_some(route.s4_role_choice),
        "s4_source": (!frozen).then(|| GateTrace::from(step.gate)),
        "token_id": step.token,
        "mass_q48": step.top_mass,
        "top_ties": step.ties,
        "mass_sha256": hash_words(words),
    })
}

impl Runner<'_> {
    fn fit_corpus(&self, fit: &mut Fit, corpus: &Corpus, positions: &[Position]) -> Result<()> {
        for chunk in positions.chunks(ROW_BATCH) {
            let (prefixes, targets) = corpus_inputs(corpus, chunk);
            let batch = self.gpu.prepare(&self.p4_gpu, &prefixes, &targets)?;
            self.gpu.observe_fit(fit, &batch)?;
        }
        Ok(())
    }

    fn fit_panel(&self, fit: &mut Fit, panel: &[PanelRow]) -> Result<()> {
        for row in panel {
            let prefixes = teacher_inputs(row);
            let batch = self.gpu.prepare(&self.p4_gpu, &prefixes, &row.gold)?;
            self.gpu.observe_fit(fit, &batch)?;
        }
        Ok(())
    }

    fn teacher_record(
        &self,
        input: &[Vec<u16>],
        targets: &[u16],
        weights: &[i8; CELLS],
        per_row: &mut [[Counts; ARMS.len()]],
        offset: usize,
    ) -> Result<()> {
        let batch = self.gpu.prepare(&self.p4_gpu, input, targets)?;
        let mut frozen = Vec::new();
        let mut conditioned = Vec::new();
        for (index, arm) in ARMS.iter().enumerate() {
            let evaluation = self.gpu.evaluate(
                &batch,
                if arm.null_weights { &ZERO } else { weights },
                arm.device_mode,
                false,
                true,
            )?;
            if index == 0 {
                frozen = evaluation.rows.clone();
            } else if index == 2 {
                conditioned = evaluation.rows.clone();
            } else if index == 1
                && evaluation.rows.iter().zip(&frozen).any(|(a, b)| {
                    a.token != b.token
                        || a.top_mass != b.top_mass
                        || a.target_mass != b.target_mass
                        || a.brier != b.brier
                        || a.prll_bits != b.prll_bits
                        || a.gate.transfer != 0
                })
            {
                return Err(fail("matched zero weights changed the frozen P4 outcome"));
            }
            for (i, ((prefix, &gold), (scored, &baseline))) in input
                .iter()
                .zip(targets)
                .zip(evaluation.rows.into_iter().zip(&frozen))
                .enumerate()
            {
                per_row[offset + i][index].observe(scored, baseline, gold, prefix.contains(&gold));
                if index == 5 {
                    let learned = conditioned[i].gate;
                    let pooled = scored.gate;
                    if learned.source != pooled.source
                        || learned.source_pos != pooled.source_pos
                        || learned.rival != pooled.rival
                        || learned.resolved_role != pooled.resolved_role
                    {
                        return Err(fail(
                            "role-blind eligibility or authenticated source differs",
                        ));
                    }
                    let counts = &mut per_row[offset + i][index];
                    counts.conditioned_pooled_weight_diff +=
                        u64::from(learned.weight != pooled.weight);
                    let signed_transfer =
                        |gate: Choice| gate.transfer as i64 * i64::from(gate.weight.signum());
                    counts.conditioned_pooled_score_diff +=
                        u64::from(signed_transfer(learned) != signed_transfer(pooled));
                }
            }
        }
        Ok(())
    }

    fn teacher_split(
        &self,
        panel: &[PanelRow],
        weights: &[i8; CELLS],
        gate: &S4SourceGate,
        writer: &mut impl Write,
        witness_writer: &mut impl Write,
    ) -> Result<usize> {
        let mut observed = BTreeSet::new();
        let mut witness_words = 0;
        for row in panel {
            let prefixes = teacher_inputs(row);
            let mut counts = vec![std::array::from_fn(|_| Counts::default()); prefixes.len()];
            self.teacher_record(&prefixes, &row.gold, weights, &mut counts, 0)?;
            for (i, arm) in ARMS.iter().enumerate() {
                let mut sum = Counts::default();
                for row in &counts {
                    sum.merge(&row[i]);
                }
                emit(
                    writer,
                    RecordResult {
                        id: &row.id,
                        domain: &row.domain,
                        arm: arm.name,
                        donor_opening_exposed: row.donor_opening_exposed,
                        counts: sum,
                    },
                )?;
            }
            for (kind, position) in [
                (
                    "nonzero-transfer",
                    counts.iter().position(|r| r[2].active != 0),
                ),
                (
                    "changed-top1",
                    counts.iter().position(|r| r[2].top1_changed != 0),
                ),
                ("abstention", counts.iter().position(|r| r[2].admitted == 0)),
            ] {
                if let Some(position) = position.filter(|_| observed.insert(kind)) {
                    let input = &prefixes[position..=position];
                    let target = &row.gold[position..=position];
                    let batch = self.gpu.prepare(&self.p4_gpu, input, target)?;
                    for arm in [ARMS[0], ARMS[2]] {
                        let result =
                            self.gpu
                                .evaluate(&batch, weights, arm.device_mode, true, false)?;
                        witness_words += self.verify_ordinary(input, &batch, &result, arm, gate)?;
                        emit(
                            witness_writer,
                            json!({
                                "kind": kind, "id": row.id, "domain": row.domain, "position": position,
                                "donor_opening_exposed": row.donor_opening_exposed,
                                "prefix": input[0], "gold": target[0], "arm": arm.name,
                                "step": trace_step(&batch.route[0], result.rows[0], &input[0], &result.scores, arm.client_mode.is_none())
                            }),
                        )?;
                    }
                }
            }
        }
        Ok(witness_words)
    }

    fn teacher_anchor(
        &self,
        corpus: &Corpus,
        positions: &[Position],
        weights: &[i8; CELLS],
        writer: &mut impl Write,
    ) -> Result<()> {
        let mut counts = vec![std::array::from_fn(|_| Counts::default()); positions.len()];
        for (chunk, offset) in positions
            .chunks(ROW_BATCH)
            .zip((0..positions.len()).step_by(ROW_BATCH))
        {
            let (prefixes, targets) = corpus_inputs(corpus, chunk);
            self.teacher_record(&prefixes, &targets, weights, &mut counts, offset)?;
        }
        for (i, position) in positions.iter().enumerate() {
            let id = format!("dev-tune:{}:{}", position.record, position.target_at);
            for (arm, counts) in ARMS.iter().zip(&counts[i]) {
                emit(
                    writer,
                    RecordResult {
                        id: &id,
                        domain: "d87-dev-tune",
                        arm: arm.name,
                        donor_opening_exposed: false,
                        counts: counts.clone(),
                    },
                )?;
            }
        }
        Ok(())
    }

    fn verify_ordinary(
        &self,
        prefixes: &[Vec<u16>],
        batch: &Batch,
        output: &Output,
        arm: Arm,
        gate: &S4SourceGate,
    ) -> Result<usize> {
        for (i, (prefix, (route, step))) in prefixes
            .iter()
            .zip(batch.route.iter().zip(&output.rows))
            .enumerate()
        {
            let words = &output.scores[i * WORDS..(i + 1) * WORDS];
            let mut session = P4Session::new();
            session.sync_prompt(prefix)?;
            let (token, trace) = session.step_with_source(
                self.p4_cpu,
                true,
                true,
                arm.client_mode.map(|mode| (gate, mode)),
            )?;
            let trace = trace.ok_or_else(|| fail("ordinary P4 trace was empty"))?;
            let rust = trace.s4_source;
            let cuda = GateTrace::from(step.gate);
            let gate_matches = rust.map_or(arm.client_mode.is_none(), |host| {
                host.resolved_role == cuda.resolved_role
                    && host.effective_role == cuda.effective_role
                    && host.gate_index == cuda.gate_index
                    && host.source_pos == cuda.source_pos
                    && host.source_token == cuda.source_token
                    && host.rival_token == cuda.rival_token
                    && host.weight_i8 == cuda.weight_i8
                    && host.transfer_q48 == cuda.transfer_q48
                    && host.source_mass_before_q48 == cuda.source_mass_before_q48
                    && host.rival_mass_before_q48 == cuda.rival_mass_before_q48
            });
            let cuda_s3_mode = match route.s3_hit_kind {
                0 => "primary",
                1 => "backoff",
                2 => "starling",
                3 => "uniform_miss",
                _ => return Err(fail("unknown GPU S3 mode")),
            };
            // The baseline includes the whole frozen P4 path. Compare the raw
            // route and all output words, as well as the overlaid source trace.
            if u32::from(token) != step.token
                || trace.highest_order as u32 != route.highest_order
                || trace.context_tokens as u32 != route.context_tokens
                || trace.prefix_sha256 != prefix_sha256(prefix)
                || trace.pick_pos != (route.pick_has != 0).then_some(route.pick_pos)
                || trace.pick_successor.map(u32::from)
                    != (route.pick_has != 0).then_some(route.pick_successor)
                || trace.pick_distance != (route.pick_has != 0).then_some(route.pick_distance)
                || trace.pick_ties != route.pick_ties
                || u32::from(trace.pick_class) != route.pick_class
                || trace.s3_pkey != route.s3_pkey
                || u32::from(trace.s3_backoff_key) != route.s3_backoff_key
                || trace.s3_mode != cuda_s3_mode
                || trace.s2_skip_keys != route.s2_skip_keys
                || u32::from(trace.s2_skip_hit_mask) != route.s2_skip_hit_mask
                || u32::from(trace.cache_hit_state) != route.cache_hit_state
                || trace.bucket as u32 != route.bucket
                || trace.s4_role_scores != route.s4_role_scores
                || trace.s4_role_choice.map(u32::from)
                    != (route.s4_role_choice != 0).then_some(route.s4_role_choice)
                || trace.mass_q48 != step.top_mass
                || trace.top_ties != step.ties
                || trace.mass_sha256 != hash_words(words)
                || session.last_scores_q48().get(1..) != Some(words)
                || !gate_matches
            {
                return Err(fail(format!(
                    "ordinary parity mismatch in {} at row {i}: CUDA gate {cuda:?}, Rust gate {rust:?}, CUDA route {route:?}, Rust trace {trace:?}",
                    arm.name
                )));
            }
        }
        Ok(prefixes.len() * WORDS)
    }

    fn generate(
        &self,
        panel: &[PanelRow],
        weights: &[i8; CELLS],
        writer: &mut impl Write,
    ) -> Result<usize> {
        let selected: Vec<_> = panel.iter().filter(|row| row.free_run).collect();
        let mut all_steps = 0usize;
        // Keep independent autoregressive histories and batch each surviving arm.
        for arm in ARMS {
            let mut histories: Vec<_> = selected.iter().map(|row| row.prefix.clone()).collect();
            let mut tokens: Vec<Vec<u16>> = vec![Vec::new(); selected.len()];
            let mut traces: Vec<Vec<Value>> = vec![Vec::new(); selected.len()];
            let mut next: Vec<usize> = (0..selected.len()).collect();
            while !next.is_empty() {
                let inputs: Vec<_> = next
                    .iter()
                    .map(|&i| active_window(&histories[i]).to_vec())
                    .collect();
                // The dummy is used only for unused teacher metrics. Nothing
                // from the gold reference is seen by the free-running kernel.
                let batch = self
                    .gpu
                    .prepare(&self.p4_gpu, &inputs, &vec![1; inputs.len()])?;
                let result = self.gpu.evaluate(
                    &batch,
                    if arm.null_weights { &ZERO } else { weights },
                    arm.device_mode,
                    true,
                    false,
                )?;
                for (j, &i) in next.iter().enumerate() {
                    let evaluated = result.rows[j];
                    let words = &result.scores[j * WORDS..(j + 1) * WORDS];
                    let emitted = u16::try_from(evaluated.token)?;
                    tokens[i].push(emitted);
                    traces[i].push(trace_step(
                        &batch.route[j],
                        evaluated,
                        &inputs[j],
                        words,
                        arm.client_mode.is_none(),
                    ));
                    if emitted != 1 {
                        histories[i].push(emitted);
                    }
                }
                next.retain(|&i| {
                    tokens[i].last() != Some(&1) && tokens[i].len() < cap(selected[i])
                });
            }
            for (i, row) in selected.iter().enumerate() {
                all_steps += tokens[i].len();
                let exact = if row.domain == "conversation" {
                    tokens[i] == row.gold
                } else {
                    tokens[i].as_slice().get(..row.gold.len()) == Some(row.gold.as_slice())
                };
                emit(
                    writer,
                    json!({
                        "id": row.id, "domain": row.domain, "arm": arm.name, "tokens": tokens[i],
                        "donor_opening_exposed": row.donor_opening_exposed,
                        "ended_eos": tokens[i].last() == Some(&1), "gold": row.gold,
                        "cap": cap(row), "steps": traces[i], "exact": exact,
                    }),
                )?;
            }
            writer.flush()?;
            eprintln!(
                "generated {} for {} fixed prompts",
                arm.name,
                selected.len()
            );
        }
        Ok(all_steps)
    }
}

fn cap(row: &PanelRow) -> usize {
    if row.domain == "prose" {
        64
    } else {
        128
    }
}

impl Counts {
    fn merge(&mut self, other: &Self) {
        self.n_targets += other.n_targets;
        self.top1 += other.top1;
        self.brier_q32 += other.brier_q32;
        self.prll_bits_q12 += other.prll_bits_q12;
        self.present_targets += other.present_targets;
        self.present_top1 += other.present_top1;
        self.present_brier_q32 += other.present_brier_q32;
        self.present_prll_bits_q12 += other.present_prll_bits_q12;
        self.absent_targets += other.absent_targets;
        self.absent_top1 += other.absent_top1;
        self.absent_brier_q32 += other.absent_brier_q32;
        self.absent_prll_bits_q12 += other.absent_prll_bits_q12;
        self.eos_targets += other.eos_targets;
        self.eos_top1 += other.eos_top1;
        self.role_positive += other.role_positive;
        self.admitted += other.admitted;
        self.active += other.active;
        self.top1_changed += other.top1_changed;
        self.gained += other.gained;
        self.lost += other.lost;
        self.zero_weight += other.zero_weight;
        self.no_wrong_address += other.no_wrong_address;
        self.conditioned_pooled_weight_diff += other.conditioned_pooled_weight_diff;
        self.conditioned_pooled_score_diff += other.conditioned_pooled_score_diff;
        for (sum, &count) in self
            .admitted_by_cell
            .iter_mut()
            .zip(&other.admitted_by_cell)
        {
            *sum += count;
        }
    }
}

pub fn run(args: Vec<OsString>) -> Result<()> {
    let started = Instant::now();
    match args.as_slice() {
        [cmd] if cmd == "finite" => {
            let gpu = Gpu::new()?;
            gpu.finite()?;
            println!(
                "{}",
                json!({"status": "PASS", "cubin_sha256": gpu.digest()})
            );
            return Ok(());
        }
        [cmd, p4, r, dev, fit, held, out] if cmd == "run" => {
            let out = Path::new(out);
            fs::create_dir(out)?;
            let train = Corpus::load(Path::new(r), TRAIN_SHA)?;
            let dev = Corpus::load(Path::new(dev), DEV_SHA)?;
            let positions = train
                .train_sample(1, R_TARGETS, R_SEED)?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let dev_draw = dev.dev_sample(DEV_ROWS, DEV_SEED)?;
            let r_draw_sha = sample_digest(&train, &positions);
            let dev_draw_sha = sample_digest(&dev, &dev_draw);
            let (fit_panel, fit_panel_sha) = open_panel(Path::new(fit), "fit")?;
            let (held_panel, held_panel_sha) = open_panel(Path::new(held), "held")?;
            let fit_ids: BTreeSet<_> = fit_panel.iter().map(|row| &row.id).collect();
            if held_panel.iter().any(|row| fit_ids.contains(&row.id)) {
                return Err(fail("fit and held panel IDs intersect"));
            }

            let ClientArtifactFile::Embedded {
                model,
                state_bytes,
                expected_state_sha256,
                identity,
            } = load_client_artifact(Path::new(p4), false)?
            else {
                return Err(fail("pinned P4 did not load as embedded deployment"));
            };
            if identity.artifact_sha256 != P4_SHA || expected_state_sha256 != P4_STATE_SHA {
                return Err(fail("complete P4 artifact or inner state digest is wrong"));
            }
            let deployment = DiscreteCountModel::bind_embedded(
                model,
                &state_bytes,
                &expected_state_sha256,
                identity,
            )?;
            let p4 = &deployment
                .p4
                .as_ref()
                .ok_or_else(|| fail("pinned artifact does not bind P4 state"))?
                .inner;
            let bind = p4
                .state_sha256()
                .ok_or_else(|| fail("P4 state lacks serving identity"))?;
            if hex::encode(bind) != P4_STATE_SHA {
                return Err(fail("ordinary P4 runtime bound an unexpected state"));
            }
            let raw = split_uorp4a01_sections(&state_bytes)?;
            let runner = Runner {
                gpu: Gpu::new()?,
                p4_gpu: P4GpuEngine::new(
                    raw.counts_v3,
                    raw.s3_compact,
                    raw.s2_skip_compact,
                    raw.s4_table,
                    &p4.s6.weights_q32,
                    &p4.codes,
                )?,
                p4_cpu: p4,
            };
            runner.gpu.finite()?;
            let load_ms = started.elapsed().as_millis();

            let fit_started = Instant::now();
            let mut fit = runner.gpu.start_fit()?;
            runner.fit_corpus(&mut fit, &train, &positions)?;
            let r_fit_ms = fit_started.elapsed().as_millis();
            let r_only = runner.gpu.finish_fit(&fit)?;
            runner.fit_panel(&mut fit, &fit_panel)?;
            let fitted = runner.gpu.finish_fit(&fit)?;
            let complete_fit_ms = fit_started.elapsed().as_millis();
            let train_identity = fit_identity(&r_draw_sha, &fit_panel_sha)?;
            let saved = S4SourceGate::new(bind, train_identity, fitted.weights)?;
            let gate_path = out.join("learned.uors4g01");
            fs::write(&gate_path, saved.encode_bytes())?;
            let gate_bytes = fs::read(&gate_path)?;
            let restored = S4SourceGate::from_bytes(&gate_bytes, bind)?;
            if restored != saved || S4SourceGate::from_bytes(&gate_bytes, [0; 32]).is_ok() {
                return Err(fail(
                    "native weight read-back or wrong-state refusal failed",
                ));
            }
            let mut changed = gate_bytes.clone();
            changed[72] ^= 1;
            if S4SourceGate::from_bytes(&changed, bind).is_ok() {
                return Err(fail("tampered gate bytes were admitted"));
            }

            let parity_started = Instant::now();
            let parity_dev: Vec<_> = dev_draw
                .iter()
                .copied()
                .filter(|&pos| dev.prefix(pos).last() != Some(&1))
                .take(16)
                .collect();
            if parity_dev.len() != 16 {
                return Err(fail(
                    "fewer than sixteen open DEV-tune prefixes for ordinary replay",
                ));
            }
            let (mut prefixes, mut targets) = corpus_inputs(&dev, &parity_dev);
            let prior_eos = held_panel
                .iter()
                .find(|row| row.free_run && row.domain == "conversation" && row.prefix.contains(&1))
                .ok_or_else(|| {
                    fail("no fixed strict conversation with historical EOS for parity")
                })?;
            prefixes.push(prior_eos.prefix.clone());
            targets.push(prior_eos.gold[0]);
            let batch = runner.gpu.prepare(&runner.p4_gpu, &prefixes, &targets)?;
            let mut parity_words = 0;
            let mut parity_writer =
                BufWriter::new(File::create(out.join("parity-fixtures.jsonl"))?);
            for arm in ARMS {
                let evaluation = runner.gpu.evaluate(
                    &batch,
                    if arm.null_weights {
                        &ZERO
                    } else {
                        &fitted.weights
                    },
                    arm.device_mode,
                    true,
                    true,
                )?;
                parity_words +=
                    runner.verify_ordinary(&prefixes, &batch, &evaluation, arm, &restored)?;
                for (i, ((prefix, route), &step)) in prefixes
                    .iter()
                    .zip(&batch.route)
                    .zip(&evaluation.rows)
                    .enumerate()
                {
                    let (kind, id, domain) = parity_dev.get(i).map_or_else(
                        || ("prior-eos", prior_eos.id.clone(), "conversation"),
                        |p| {
                            (
                                "open-dev-anchor",
                                format!("dev-tune:{}:{}", p.record, p.target_at),
                                "d87-dev-tune",
                            )
                        },
                    );
                    let words = &evaluation.scores[i * WORDS..(i + 1) * WORDS];
                    emit(
                        &mut parity_writer,
                        json!({
                            "kind": kind, "id": id, "domain": domain, "prefix": prefix,
                            "gold": targets[i], "arm": arm.name,
                            "step": trace_step(route, step, prefix, words, arm.client_mode.is_none()),
                        }),
                    )?;
                }
            }
            parity_writer.flush()?;
            let parity_ms = parity_started.elapsed().as_millis();

            let teacher_started = Instant::now();
            let mut teacher = BufWriter::new(File::create(out.join("teacher-records.jsonl"))?);
            runner.teacher_anchor(&dev, &dev_draw, &fitted.weights, &mut teacher)?;
            let mut witness_writer =
                BufWriter::new(File::create(out.join("teacher-witnesses.jsonl"))?);
            let witness_words = runner.teacher_split(
                &held_panel,
                &fitted.weights,
                &restored,
                &mut teacher,
                &mut witness_writer,
            )?;
            teacher.flush()?;
            witness_writer.flush()?;
            let teacher_ms = teacher_started.elapsed().as_millis();

            let generation_started = Instant::now();
            let mut generation = BufWriter::new(File::create(out.join("generation.jsonl"))?);
            let generated_steps = runner.generate(&held_panel, &fitted.weights, &mut generation)?;
            generation.flush()?;
            let generation_ms = generation_started.elapsed().as_millis();
            let report = json!({
                "status": "PASS",
                "artifact_sha256": P4_SHA,
                "state_sha256": P4_STATE_SHA,
                "gate_sha256": hex::encode(Sha256::digest(&gate_bytes)),
                "train_identity_sha256": hex::encode(train_identity),
                "cubin_sha256": runner.gpu.digest(),
                "r_fit": {"uort_sha256": TRAIN_SHA, "draw_sha256": r_draw_sha, "seed": R_SEED, "targets": positions.len()},
                "d87_dev_tune": {"uort_sha256": DEV_SHA, "draw_sha256": dev_draw_sha, "seed": DEV_SEED, "records": dev_draw.len()},
                "fit_panel": {"sha256": hex::encode(fit_panel_sha), "records": fit_panel.len(), "targets": fit_panel.iter().map(|row| row.gold.len()).sum::<usize>()},
                "held_panel": {"sha256": hex::encode(held_panel_sha), "records": held_panel.len(), "targets": held_panel.iter().map(|row| row.gold.len()).sum::<usize>()},
                "r_only_fit": r_only,
                "complete_fit": fitted,
                "raw_ordinary_parity": {"dev_records": parity_dev.len(), "dev_draw_sha256": sample_digest(&dev, &parity_dev), "prior_eos_record": prior_eos.id, "arms": ARMS.len(), "words": parity_words, "natural_witness_words": witness_words},
                "generated_steps": generated_steps,
                "timing_ms": {
                    "load_and_finite": load_ms, "r_fit": r_fit_ms, "fit_with_panel": complete_fit_ms,
                    "parity": parity_ms, "teacher": teacher_ms, "generation": generation_ms,
                    "total": started.elapsed().as_millis(),
                },
            });
            serde_json::to_writer_pretty(File::create(out.join("measurement.json"))?, &report)?;
            println!("{report}");
            return Ok(());
        }
        _ => {}
    }
    Err(fail("usage: s4-output-runner finite | run P4_D11 R_UORT DEV_TUNE_UORT FIT_JSONL HELD_JSONL OUT_DIR"))
}
