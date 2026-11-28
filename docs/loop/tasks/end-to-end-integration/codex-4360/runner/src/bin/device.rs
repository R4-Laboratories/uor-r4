use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{self, BufRead, BufReader, BufWriter, Write},
    sync::Arc,
};
use uor_r4_native_vector::{
    butterfly_rotor::{splitmix64, LossKind, ACT_CLAMP, HIER_COARSE, HIER_FINE, PKM_TOP_K},
    butterfly_rotor_v3::{ButterflyV3Config, ButterflyV3Model, ButterflyV3Session},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize, Serialize)]
struct Row {
    prefix: Vec<u16>,
    expected: Vec<u16>,
}

#[derive(Deserialize)]
struct Pair {
    pair_id: u64,
    a: Row,
    b: Row,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct State {
    carry: Vec<Vec<i32>>,
    shift_hist: Vec<Vec<i32>>,
    tokens_seen: u64,
}

impl From<ButterflyV3Session> for State {
    fn from(session: ButterflyV3Session) -> Self {
        Self {
            carry: session.carry,
            shift_hist: session.shift_hist,
            tokens_seen: session.tokens_seen,
        }
    }
}

struct Engine {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    weights: CudaSlice<i16>,
    perm: CudaSlice<u16>,
    inv: CudaSlice<u16>,
    gpu_name: String,
    gpu_cc: (i32, i32),
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Step {
    greedy_token: u16,
    greedy_coarse: usize,
    greedy_fine: usize,
    coarse_scores: Vec<i32>,
    pkm_slots: Vec<Vec<i32>>,
    hidden_q15: Vec<i32>,
}

struct Launch {
    state: State,
    trace: Vec<i32>,
    mass: Vec<u64>,
    n_emitted: usize,
}

struct Request<'a> {
    inputs: &'a [u16],
    rollout: bool,
    steps: usize,
    eos: [u16; 2],
    targets: [u16; 2],
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Target {
    token: u16,
    cell: usize,
    within_cell_slot: usize,
    admitted: bool,
    coarse_score_q15: i32,
    fine_score_q15: i32,
    coarse_score_minus_winner_q15: i32,
    fine_score_minus_best_other_q15: Option<i32>,
    diagnostic_within_cell_rank: usize,
    admitted_within_cell_rank: Option<usize>,
    target_mass_q48: u64,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Arm {
    before: State,
    first: Step,
    greedy_fine_score_q15: i32,
    after: State,
    own: Target,
    other: Target,
    emitted: Vec<u16>,
    first_error: Option<usize>,
    donor_first_error: Option<usize>,
    donor_exact: bool,
    own_outcome: Outcome,
    donor_outcome: Outcome,
    hit_eos: bool,
    stop_reason: &'static str,
    eos_token: Option<u16>,
    exact: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Outcome {
    verdict: &'static str,
    first_mismatch_index: Option<usize>,
}

impl Engine {
    fn new(model: &ButterflyV3Model, artifact: &[u8], cubin: Vec<u8>) -> Result<Self> {
        if !(2..=4096).contains(&model.cfg.vocab) || model.cfg.dim > 512 {
            return Err("unsupported routed vocab or CUDA dim".into());
        }
        let ctx = CudaContext::new(0)?;
        let gpu_name = ctx.name()?;
        let gpu_cc = ctx.compute_capability()?;
        if gpu_cc != (8, 9) {
            return Err(format!("requires CUDA sm89, got {gpu_cc:?}").into());
        }
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_binary(cubin))?;
        let kernel = module.load_function("bfy_v3_compact_replay")?;
        // UORBFY03 ends with packed native i16 weights in the CUDA backend's exact
        // ordering, then the eight-byte artifact checksum.
        let end = artifact
            .len()
            .checked_sub(8)
            .ok_or("missing artifact checksum")?;
        let start = end
            .checked_sub(2 * model.param_count())
            .ok_or("missing artifact weights")?;
        let weights: Vec<_> = artifact[start..end]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        if !weights.starts_with(&model.embeddings) {
            return Err("artifact CUDA weight layout mismatch".into());
        }

        Ok(Self {
            weights: stream.clone_htod(&weights)?,
            perm: stream.clone_htod(&model.vocab_perm)?,
            inv: stream.clone_htod(&model.vocab_inv)?,
            stream,
            kernel,
            gpu_name,
            gpu_cc,
        })
    }

    fn launch(&self, cfg: ButterflyV3Config, start: &State, req: Request<'_>) -> Result<Launch> {
        if req.steps == 0 {
            return Ok(Launch {
                state: start.clone(),
                trace: Vec::new(),
                mass: Vec::new(),
                n_emitted: 0,
            });
        }
        let dim = cfg.dim;
        let (loss_code, loss_param) = cfg.loss_kind.code();
        let params = [
            cfg.vocab as u32,
            dim as u32,
            dim.trailing_zeros(),
            cfg.layers as u32,
            cfg.ffn_branches as u32,
            cfg.pkm_sub_keys as u32,
            cfg.pkm_key_dim as u32,
            u32::try_from(req.steps)?,
            u32::from(req.rollout),
            req.eos[0].into(),
            req.eos[1].into(),
            req.targets[0].into(),
            req.targets[1].into(),
            loss_code.into(),
            loss_param as u32,
        ];
        let d_params = self.stream.clone_htod(&params)?;
        let d_inputs = self.stream.clone_htod(req.inputs)?;
        let mut d_carry = self.stream.clone_htod(&start.carry.concat())?;
        let mut d_shift = self.stream.clone_htod(&start.shift_hist.concat())?;
        let stride = trace_stride(cfg);
        let trace_len = if req.rollout { stride * req.steps } else { 1 };
        let mut d_trace = self.stream.alloc_zeros::<i32>(trace_len)?;
        let mut d_mass = self.stream.alloc_zeros::<u64>(2)?;
        let mut d_count = self.stream.alloc_zeros::<u32>(1)?;
        let mut d_first = self.stream.alloc_zeros::<i32>(4 * dim * cfg.layers)?;
        let mut builder = self.stream.launch_builder(&self.kernel);
        builder
            .arg(&self.weights)
            .arg(&self.perm)
            .arg(&self.inv)
            .arg(&d_inputs)
            .arg(&d_params)
            .arg(&mut d_carry)
            .arg(&mut d_shift)
            .arg(&mut d_trace)
            .arg(&mut d_mass)
            .arg(&mut d_count)
            .arg(&mut d_first);
        // The launch ABI uses typed buffers and one block; each trace record's
        // stride and all workspaces are derived from the validated model.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (dim.min(256) as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        let (carry, shift_hist, consumed) = if req.rollout {
            let first = self.stream.clone_dtoh(&d_first)?;
            let (carry, shift) = first.split_at(dim * cfg.layers);
            (carry.to_vec(), shift.to_vec(), 1)
        } else {
            (
                self.stream.clone_dtoh(&d_carry)?,
                self.stream.clone_dtoh(&d_shift)?,
                req.steps,
            )
        };
        Ok(Launch {
            state: State {
                carry: carry.chunks_exact(dim).map(<[_]>::to_vec).collect(),
                shift_hist: shift_hist
                    .chunks_exact(3 * dim)
                    .map(<[_]>::to_vec)
                    .collect(),
                tokens_seen: start.tokens_seen + consumed as u64,
            },
            trace: self.stream.clone_dtoh(&d_trace)?,
            mass: self.stream.clone_dtoh(&d_mass)?,
            n_emitted: self.stream.clone_dtoh(&d_count)?[0] as usize,
        })
    }
}

fn trace_stride(cfg: ButterflyV3Config) -> usize {
    2 + cfg.dim + 3 * HIER_COARSE + cfg.layers * PKM_TOP_K
}

fn admissible(model: &ButterflyV3Model, token: u16) -> bool {
    token != 0 && usize::from(token) < model.cfg.vocab
}

fn target(
    model: &ButterflyV3Model,
    step: &Step,
    token: u16,
    fine_scores: &[i32],
    mass: u64,
) -> Target {
    let slot = usize::from(model.vocab_perm[usize::from(token)]);
    let cell = slot / HIER_FINE;
    let fine = slot % HIER_FINE;
    let score = fine_scores[fine];
    let rank = 1 + fine_scores
        .iter()
        .enumerate()
        .filter(|&(candidate, &candidate_score)| {
            let candidate_id = model.vocab_inv[cell * HIER_FINE + candidate];
            admissible(model, candidate_id)
                && (candidate_score > score || (candidate_score == score && candidate < fine))
        })
        .count();
    let admitted = cell == step.greedy_coarse;
    Target {
        token,
        cell,
        within_cell_slot: fine,
        admitted,
        coarse_score_q15: step.coarse_scores[cell],
        fine_score_q15: score,
        coarse_score_minus_winner_q15: step.coarse_scores[cell]
            - step.coarse_scores[step.greedy_coarse],
        fine_score_minus_best_other_q15: fine_scores
            .iter()
            .enumerate()
            .filter(|&(candidate, _)| {
                candidate != fine
                    && admissible(model, model.vocab_inv[cell * HIER_FINE + candidate])
            })
            .map(|(_, &candidate_score)| candidate_score)
            .max()
            .map(|best| score - best),
        diagnostic_within_cell_rank: rank,
        admitted_within_cell_rank: admitted.then_some(rank),
        target_mass_q48: mass,
    }
}

fn arm(
    engine: &Engine,
    model: &ButterflyV3Model,
    state: State,
    row: &Row,
    donor: &Row,
    eos: [u16; 2],
    cap: usize,
) -> Result<Arm> {
    let own = row.expected[0];
    let other = donor.expected[0];
    let inputs = &row.prefix[row.prefix.len() - 1..];
    let output = engine.launch(
        model.cfg,
        &state,
        Request {
            inputs,
            rollout: true,
            steps: cap,
            targets: [own, other],
            eos,
        },
    )?;
    if !(1..=cap).contains(&output.n_emitted) {
        return Err("CUDA invalid rollout length".into());
    }
    let emitted = output
        .trace
        .chunks_exact(trace_stride(model.cfg))
        .take(output.n_emitted)
        .map(|trace| model.vocab_inv[trace[0] as usize])
        .collect::<Vec<_>>();
    let first_slot = output.trace[0] as usize;
    let coarse_begin = 2 + model.cfg.dim;
    let own_begin = coarse_begin + HIER_COARSE;
    let other_begin = own_begin + HIER_FINE;
    let pkm_begin = other_begin + HIER_FINE;
    let first = Step {
        greedy_token: emitted[0],
        greedy_coarse: first_slot / HIER_FINE,
        greedy_fine: first_slot % HIER_FINE,
        coarse_scores: output.trace[coarse_begin..own_begin].to_vec(),
        pkm_slots: output.trace[pkm_begin..trace_stride(model.cfg)]
            .chunks_exact(PKM_TOP_K)
            .map(<[_]>::to_vec)
            .collect(),
        hidden_q15: output.trace[2..coarse_begin].to_vec(),
    };
    let own = target(
        model,
        &first,
        own,
        &output.trace[own_begin..other_begin],
        output.mass[0],
    );
    let other = target(
        model,
        &first,
        other,
        &output.trace[other_begin..pkm_begin],
        output.mass[1],
    );
    let eos_token = emitted.last().copied().filter(|id| eos.contains(id));
    let stop_reason = match emitted.last() {
        Some(&0) => "invalid_bos",
        Some(_) if eos_token.is_some() => "eos",
        _ => "cap",
    };
    let hit_eos = eos_token.is_some();
    let own_outcome = outcome(&emitted, &row.expected, hit_eos);
    let donor_outcome = outcome(&emitted, &donor.expected, hit_eos);
    let exact = own_outcome.verdict == "exact_through_eos";
    let donor_exact = donor_outcome.verdict == "exact_through_eos";
    Ok(Arm {
        before: state,
        first,
        greedy_fine_score_q15: output.trace[1],
        after: output.state,
        own,
        other,
        emitted,
        first_error: own_outcome.first_mismatch_index,
        donor_first_error: donor_outcome.first_mismatch_index,
        donor_exact,
        own_outcome,
        donor_outcome,
        hit_eos,
        stop_reason,
        eos_token,
        exact,
    })
}

fn outcome(emitted: &[u16], expected: &[u16], reached_eos: bool) -> Outcome {
    let first_mismatch_index = emitted
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .or_else(|| (emitted.len() > expected.len()).then_some(expected.len()));
    let verdict = if first_mismatch_index.is_some() {
        "first_wrong"
    } else if reached_eos && emitted == expected {
        "exact_through_eos"
    } else {
        "censored"
    };
    Outcome {
        verdict,
        first_mismatch_index,
    }
}

fn prefill(engine: &Engine, model: &ButterflyV3Model, row: &Row) -> Result<State> {
    let inputs = &row.prefix[..row.prefix.len() - 1];
    let start = model.new_session().into();
    Ok(engine
        .launch(
            model.cfg,
            &start,
            Request {
                inputs,
                rollout: false,
                steps: inputs.len(),
                targets: [0; 2],
                eos: [0; 2],
            },
        )?
        .state)
}

fn checked_pair(model: &ButterflyV3Model, pair: &Pair, eos: [u16; 2]) -> Result<()> {
    for row in [&pair.a, &pair.b] {
        if row.prefix.is_empty()
            || row.expected.is_empty()
            || row
                .prefix
                .iter()
                .chain(&row.expected)
                .any(|&id| usize::from(id) >= model.cfg.vocab)
            || row.expected.iter().any(|&id| !admissible(model, id))
            || !row.expected.last().is_some_and(|id| eos.contains(id))
            || row.expected[..row.expected.len() - 1]
                .iter()
                .any(|id| eos.contains(id))
        {
            return Err(format!(
                "pair {} has invalid prefix or EOS-terminated continuation",
                pair.pair_id
            )
            .into());
        }
    }
    if pair.a.prefix.last() != pair.b.prefix.last() || pair.a.expected[0] == pair.b.expected[0] {
        return Err(format!(
            "pair {} needs a common final input and distinct first targets",
            pair.pair_id
        )
        .into());
    }
    Ok(())
}

fn assignment(a: &Arm, b: &Arm) -> Value {
    let native_mass = a.own.target_mass_q48 + b.own.target_mass_q48;
    let swapped_mass = a.other.target_mass_q48 + b.other.target_mass_q48;
    let native_first = u8::from(a.first.greedy_token == a.own.token)
        + u8::from(b.first.greedy_token == b.own.token);
    let swapped_first = u8::from(a.first.greedy_token == a.other.token)
        + u8::from(b.first.greedy_token == b.other.token);
    json!({
        "native_mass_q48_sum":native_mass, "swapped_mass_q48_sum":swapped_mass,
        "native_minus_swapped_q48":native_mass as i64 - swapped_mass as i64,
        "native_greedy_matches":native_first, "swapped_greedy_matches":swapped_first,
    })
}

fn evaluate_pair(
    engine: &Engine,
    model: &ButterflyV3Model,
    pair: &Pair,
    eos: [u16; 2],
    cap: usize,
) -> Result<Value> {
    checked_pair(model, pair, eos)?;
    let a = prefill(engine, model, &pair.a)?;
    let b = prefill(engine, model, &pair.b)?;

    let mut a_swap = a.clone();
    a_swap.carry.clone_from(&b.carry);
    let mut b_swap = b.clone();
    b_swap.carry.clone_from(&a.carry);

    let mut a_zero = a.clone();
    let mut b_zero = b.clone();
    for layer in a_zero.carry.iter_mut().chain(b_zero.carry.iter_mut()) {
        layer.fill(0);
    }

    let a_run = |state| arm(engine, model, state, &pair.a, &pair.b, eos, cap);
    let b_run = |state| arm(engine, model, state, &pair.b, &pair.a, eos, cap);
    let (an, bn) = (a_run(a.clone())?, b_run(b.clone())?);
    let (asw, bsw) = (a_run(a_swap)?, b_run(b_swap)?);
    let (az, bz) = (a_run(a_zero)?, b_run(b_zero)?);
    Ok(json!({
        "pair_id":pair.pair_id,
        "a":{"row":pair.a,"before":a,"native":an,"swap":asw,"zero":az},
        "b":{"row":pair.b,"before":b,"native":bn,"swap":bsw,"zero":bz},
        "assignment":{"native":assignment(&an,&bn),"swap":assignment(&asw,&bsw),"zero":assignment(&az,&bz)},
    }))
}

fn rust_prefill(model: &ButterflyV3Model, row: &Row) -> Result<ButterflyV3Session> {
    let mut session = model.new_session();
    for &token in &row.prefix[..row.prefix.len() - 1] {
        model.step(token, &mut session)?;
    }
    Ok(session)
}

fn rust_fine_score(model: &ButterflyV3Model, hidden: &[i32], slot: usize) -> i32 {
    let dim = model.cfg.dim;
    let weight = &model.embeddings[slot * dim..(slot + 1) * dim];
    let sum: i64 = hidden
        .iter()
        .zip(weight)
        .map(|(&h, &w)| i64::from(h) * i64::from(w))
        .sum();
    let shift = ButterflyV3Model::head_dot_shift(dim);
    (((sum + (1_i64 << (shift - 1))) >> shift) as i32).clamp(-ACT_CLAMP, ACT_CLAMP)
}

fn verify_step(
    model: &ButterflyV3Model,
    input: u16,
    session: &mut ButterflyV3Session,
    gpu: &Arm,
) -> Result<u16> {
    let output = model.step(input, session)?;
    let rust = Step {
        greedy_token: output.greedy_token,
        greedy_coarse: output.greedy_coarse.into(),
        greedy_fine: output.greedy_fine.into(),
        coarse_scores: output.coarse_scores.to_vec(),
        pkm_slots: output
            .pkm_slots
            .iter()
            .map(|slots| slots.iter().map(|&id| id as i32).collect())
            .collect(),
        hidden_q15: output.hidden_q15,
    };
    let after = State::from(session.clone());
    if after != gpu.after || rust != gpu.first {
        return Err(format!(
            "transition mismatch: carry={} shift={} count={} greedy={} route={} coarse={} pkm={} hidden={}",
            after.carry == gpu.after.carry, after.shift_hist == gpu.after.shift_hist,
            after.tokens_seen == gpu.after.tokens_seen, rust.greedy_token == gpu.first.greedy_token,
            rust.greedy_coarse == gpu.first.greedy_coarse && rust.greedy_fine == gpu.first.greedy_fine,
            rust.coarse_scores == gpu.first.coarse_scores, rust.pkm_slots == gpu.first.pkm_slots,
            rust.hidden_q15 == gpu.first.hidden_q15
        ).into());
    }
    let winning_slot = rust.greedy_coarse * HIER_FINE + rust.greedy_fine;
    if rust_fine_score(model, &rust.hidden_q15, winning_slot) != gpu.greedy_fine_score_q15 {
        return Err("deployed fine winner score mismatch".into());
    }
    for gpu_target in [&gpu.own, &gpu.other] {
        let cell = usize::from(model.vocab_perm[usize::from(gpu_target.token)]) / HIER_FINE;
        let exact_scores: Vec<_> = (0..HIER_FINE)
            .map(|fine| rust_fine_score(model, &rust.hidden_q15, cell * HIER_FINE + fine))
            .collect();
        let mass = model.target_q48_mass_from_hidden(&rust.hidden_q15, gpu_target.token)?;
        if target(model, &rust, gpu_target.token, &exact_scores, mass) != *gpu_target {
            return Err(format!(
                "labeled fine ranking/score/mass mismatch for {}",
                gpu_target.token
            )
            .into());
        }
    }
    Ok(rust.greedy_token)
}

fn terminal(last: u16, eos: [u16; 2]) -> Option<&'static str> {
    if last == 0 {
        Some("invalid_bos")
    } else if eos.contains(&last) {
        Some("eos")
    } else {
        None
    }
}

struct Verification<'a> {
    engine: &'a Engine,
    model: &'a ButterflyV3Model,
    eos: [u16; 2],
    cap: usize,
    digest: Sha256,
    transitions: usize,
}

impl<'a> Verification<'a> {
    fn arm(
        &mut self,
        recipient: &Row,
        donor: &Row,
        mut rust: ButterflyV3Session,
        gpu_start: State,
    ) -> Result<Arm> {
        if State::from(rust.clone()) != gpu_start {
            return Err("pre-intervention CPU/GPU state mismatch".into());
        }
        let batch = arm(
            self.engine,
            self.model,
            gpu_start.clone(),
            recipient,
            donor,
            self.eos,
            self.cap,
        )?;
        let mut gpu_state = gpu_start;
        let mut rust_emitted = Vec::new();
        let mut stop = "cap";
        let mut input = *recipient.prefix.last().ok_or("empty parity prefix")?;
        for index in 0..self.cap {
            let step_row = Row {
                prefix: vec![input],
                expected: recipient.expected.clone(),
            };
            let gpu = arm(
                self.engine,
                self.model,
                gpu_state,
                &step_row,
                donor,
                self.eos,
                1,
            )?;
            let rust_id = verify_step(self.model, input, &mut rust, &gpu)
                .map_err(|error| format!("consumed step {index}: {error}"))?;
            if index == 0
                && (batch.first != gpu.first
                    || batch.after != gpu.after
                    || batch.own != gpu.own
                    || batch.other != gpu.other
                    || batch.greedy_fine_score_q15 != gpu.greedy_fine_score_q15)
            {
                return Err("batched/single-step first transition mismatch".into());
            }
            self.digest.update(serde_json::to_vec(&gpu)?);
            self.digest.update(b"\n");
            self.transitions += 1;
            rust_emitted.push(rust_id);
            gpu_state = gpu.after;
            input = rust_id;
            if let Some(reason) = terminal(rust_id, self.eos) {
                stop = reason;
                break;
            }
        }
        if batch.emitted != rust_emitted || batch.stop_reason != stop {
            return Err("batched/self-fed CPU trajectory or stopping-length mismatch".into());
        }
        Ok(batch)
    }
}

fn load_engine(cubin: &str, artifact: &str) -> Result<(ButterflyV3Model, Engine, String, String)> {
    let artifact = fs::read(artifact)?;
    let model = ButterflyV3Model::from_bytes(&artifact)?;
    let artifact_sha = hex::encode(Sha256::digest(&artifact));
    let cubin = fs::read(cubin)?;
    let cubin_sha = hex::encode(Sha256::digest(&cubin));
    let engine = Engine::new(&model, &artifact, cubin)?;
    Ok((model, engine, artifact_sha, cubin_sha))
}

fn parity(cubin: &str, artifact: &str, subset: &str, eos: &str, cap: &str) -> Result<Value> {
    let (model, engine, artifact_sha, cubin_sha) = load_engine(cubin, artifact)?;
    let (eos, cap) = parse_decode_parameters(&model, eos, cap)?;
    let bytes = fs::read(subset)?;
    let mut checks = Verification {
        engine: &engine,
        model: &model,
        eos,
        cap,
        digest: Sha256::new(),
        transitions: 0,
    };
    let mut checked_ids = std::collections::BTreeSet::new();
    let mut stop_counts = std::collections::BTreeMap::<&str, usize>::new();
    for line in BufReader::new(bytes.as_slice()).lines() {
        let pair: Pair = serde_json::from_str(&line?)?;
        checked_pair(&model, &pair, eos)?;
        if !checked_ids.insert(pair.pair_id) {
            return Err(format!("duplicate parity pair {}", pair.pair_id).into());
        }
        let gpu_a = prefill(&engine, &model, &pair.a)?;
        let gpu_b = prefill(&engine, &model, &pair.b)?;
        let rust_a = rust_prefill(&model, &pair.a)?;
        let rust_b = rust_prefill(&model, &pair.b)?;
        if gpu_a != State::from(rust_a.clone()) || gpu_b != State::from(rust_b.clone()) {
            return Err(format!("pair {} CPU/GPU prefix mismatch", pair.pair_id).into());
        }
        for (side, recipient, donor, gpu, gpu_donor, rust, rust_donor) in [
            ("a", &pair.a, &pair.b, &gpu_a, &gpu_b, &rust_a, &rust_b),
            ("b", &pair.b, &pair.a, &gpu_b, &gpu_a, &rust_b, &rust_a),
        ] {
            for name in ["native", "swap", "zero"] {
                let mut gpu = gpu.clone();
                let mut rust = rust.clone();
                match name {
                    "swap" => {
                        gpu.carry.clone_from(&gpu_donor.carry);
                        rust.carry.clone_from(&rust_donor.carry);
                    }
                    "zero" => {
                        for layer in gpu.carry.iter_mut().chain(rust.carry.iter_mut()) {
                            layer.fill(0);
                        }
                    }
                    _ => {}
                }
                let replay = checks.arm(recipient, donor, rust, gpu).map_err(|error| {
                    format!("pair {} side {side} arm {name}: {error}", pair.pair_id)
                })?;
                *stop_counts.entry(replay.stop_reason).or_default() += 1;
            }
        }
    }
    if checked_ids.len() != 6 {
        return Err(format!(
            "parity requires exactly six preselected pairs, got {}",
            checked_ids.len()
        )
        .into());
    }
    Ok(json!({
        "parity":"PASS", "gpu":engine.gpu_name, "gpu_cc":engine.gpu_cc,
        "artifact_sha256":artifact_sha, "cubin_sha256":cubin_sha,
        "subset_sha256":hex::encode(Sha256::digest(&bytes)), "pairs":checked_ids,
        "prefix_states_checked":12, "arms_checked":36, "consumed_transitions_checked":checks.transitions,
        "stops":stop_counts, "transition_receipt_sha256":hex::encode(checks.digest.finalize()),
        "eos":eos, "cap":cap,
    }))
}

#[derive(Deserialize)]
struct HistoricalFixture {
    artifact_sha256: String,
    context_len: usize,
    gen_len: usize,
    gpu_greedy_tokens: Vec<u16>,
    gpu_q48_masses: Vec<u64>,
}

fn historical(cubin: &str, artifact: &str, fixture: &str) -> Result<Value> {
    let (model, engine, artifact_sha, cubin_sha) = load_engine(cubin, artifact)?;
    let fixture_bytes = fs::read(fixture)?;
    let fixture: HistoricalFixture = serde_json::from_slice(&fixture_bytes)?;
    if fixture.context_len != 2048
        || fixture.gen_len != 128
        || fixture.gpu_greedy_tokens.len() != 128
        || fixture.gpu_q48_masses.len() != 128
        || fixture.artifact_sha256 != artifact_sha
        || fixture
            .gpu_greedy_tokens
            .iter()
            .any(|&id| !admissible(&model, id))
        || model.cfg.vocab != 4096
    {
        return Err("historical fixture shape, artifact identity or token mismatch".into());
    }
    // Original v3 runner's synthetic_bench_prompt(2048).
    let prefix: Vec<_> = (0..2048)
        .map(|i| {
            if i == 0 {
                0
            } else {
                (1 + splitmix64(i + 17) % 4095) as u16
            }
        })
        .collect();
    let row = Row {
        prefix,
        expected: fixture.gpu_greedy_tokens.clone(),
    };
    let mut gpu = prefill(&engine, &model, &row)?;
    let mut input = row.prefix[2047];
    for (index, (&token, &mass)) in fixture
        .gpu_greedy_tokens
        .iter()
        .zip(&fixture.gpu_q48_masses)
        .enumerate()
    {
        let step_row = Row {
            prefix: vec![input],
            expected: vec![token],
        };
        let step = arm(&engine, &model, gpu, &step_row, &step_row, [u16::MAX; 2], 1)?;
        if step.first.greedy_token != token || step.own.target_mass_q48 != mass {
            return Err(format!("historical CUDA greedy/mass mismatch at step {index}").into());
        }
        input = step.first.greedy_token;
        gpu = step.after;
    }
    let batch = arm(
        &engine,
        &model,
        prefill(&engine, &model, &row)?,
        &row,
        &row,
        [u16::MAX; 2],
        128,
    )?;
    if batch.emitted != fixture.gpu_greedy_tokens || batch.stop_reason != "cap" {
        return Err("historical batched greedy mismatch".into());
    }
    Ok(json!({
        "historical":"PASS", "gpu":engine.gpu_name, "gpu_cc":engine.gpu_cc,
        "artifact_sha256":artifact_sha, "cubin_sha256":cubin_sha,
        "fixture_sha256":hex::encode(Sha256::digest(&fixture_bytes)),
        "context_len":2048, "greedy_ids_checked":128, "q48_masses_checked":128,
        "batched_greedy_ids_checked":128,
    }))
}

fn finite(cubin: Vec<u8>) -> Result<Value> {
    let mut model = ButterflyV3Model::new(ButterflyV3Config {
        vocab: 128,
        dim: 16,
        layers: 2,
        ffn_branches: 2,
        pkm_sub_keys: 4,
        pkm_key_dim: 4,
        loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 2 },
        seed: 734,
    })?;
    let mut perm = model.vocab_perm.clone();
    perm.swap(1, 65);
    perm.swap(13, 79);
    model.set_vocab_perm(perm)?;
    let artifact = model.to_bytes();
    let engine = Engine::new(&model, &artifact, cubin)?;
    let mut a = Row {
        prefix: vec![0, 7, 19, 23],
        expected: vec![13, 1],
    };
    let mut b = Row {
        prefix: vec![0, 29, 31, 23],
        expected: vec![15, 1],
    };
    let av = prefill(&engine, &model, &a)?;
    let bv = prefill(&engine, &model, &b)?;
    let ar = rust_prefill(&model, &a)?;
    let br = rust_prefill(&model, &b)?;
    if av != State::from(ar.clone()) || bv != State::from(br.clone()) {
        return Err("finite oracle prefill carry/shift mismatch".into());
    }
    // Derive EOS and excluded-cell diagnostic labels from the independent serving oracle.
    let mut probe = ar.clone();
    let first = model.step(23, &mut probe)?;
    let mut emitted = vec![first.greedy_token];
    for _ in 1..8 {
        emitted.push(
            model
                .step(*emitted.last().ok_or("empty oracle")?, &mut probe)?
                .greedy_token,
        );
    }
    let cap_eos = (1..128)
        .find(|token| !emitted.contains(token))
        .ok_or("no cap EOS control")?;
    let rival = (1..128)
        .find(|&token| {
            usize::from(model.vocab_perm[usize::from(token)]) / HIER_FINE
                != usize::from(first.greedy_coarse)
        })
        .ok_or("no excluded cell token")?;
    a.expected = vec![first.greedy_token];
    b.expected = vec![rival];
    let mut checks = Verification {
        engine: &engine,
        model: &model,
        eos: [first.greedy_token; 2],
        cap: 8,
        digest: Sha256::new(),
        transitions: 0,
    };
    let exact = checks.arm(&a, &b, ar.clone(), av.clone())?;
    let sham = checks.arm(&a, &b, ar.clone(), av.clone())?;
    let labels_changed = checks.arm(&b_with_prefix(&a, &b), &a, ar.clone(), av.clone())?;
    if exact != sham
        || !exact.exact
        || exact.emitted.len() != 1
        || !exact.own.admitted
        || exact.other.admitted
        || exact.other.admitted_within_cell_rank.is_some()
        || exact.emitted != labels_changed.emitted
        || exact.after != labels_changed.after
        || exact.first != labels_changed.first
        || exact.stop_reason != labels_changed.stop_reason
        || exact.own != labels_changed.other
        || exact.other != labels_changed.own
    {
        return Err("finite sham, EOS, rank, or label independence check failed".into());
    }

    checks.eos = [cap_eos; 2];
    let native = checks.arm(&a, &b, ar.clone(), av.clone())?;
    let mut swap_gpu = av.clone();
    let mut swap_rust = ar.clone();
    swap_gpu.carry.clone_from(&bv.carry);
    swap_rust.carry.clone_from(&br.carry);
    let mut zero_gpu = av;
    let mut zero_rust = ar;
    for layer in zero_gpu.carry.iter_mut().chain(zero_rust.carry.iter_mut()) {
        layer.fill(0);
    }
    let swap = checks.arm(&a, &b, swap_rust, swap_gpu)?;
    let zero = checks.arm(&a, &b, zero_rust, zero_gpu)?;
    if native.stop_reason != "cap"
        || native.emitted.len() != 8
        || native.first.hidden_q15 == swap.first.hidden_q15
        || native.first.hidden_q15 == zero.first.hidden_q15
    {
        return Err("finite cap or divergent-carry check failed".into());
    }
    Ok(json!({
        "finite":"PASS", "gpu":engine.gpu_name, "gpu_cc":engine.gpu_cc,
        "artifact_sha256":hex::encode(Sha256::digest(&artifact)),
        "nonidentity_vocabulary":128, "eos_emitted":exact.emitted, "cap_emitted":native.emitted,
        "finite_transitions_checked":checks.transitions, "transition_receipt_sha256":hex::encode(checks.digest.finalize()),
        "carry_divergence":true, "identical_source_sham":true, "label_independence":true, "coarse_admission":true,
    }))
}

fn b_with_prefix(recipient: &Row, labels: &Row) -> Row {
    Row {
        prefix: recipient.prefix.clone(),
        expected: labels.expected.clone(),
    }
}

fn parse_decode_parameters(
    model: &ButterflyV3Model,
    eos: &str,
    cap: &str,
) -> Result<([u16; 2], usize)> {
    let eos_ids = eos
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<u16>, _>>()?;
    let eos = match eos_ids.as_slice() {
        [id] => [*id, *id],
        [first, second] => [*first, *second],
        _ => return Err("provide one or two EOS ids".into()),
    };
    if eos.iter().any(|&id| !admissible(model, id)) {
        return Err("EOS id inadmissible for model".into());
    }
    let cap = cap.parse::<usize>()?;
    if !(1..=1024).contains(&cap) {
        return Err("generation cap must be 1 through 1024".into());
    }
    Ok((eos, cap))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, cubin] if flag == "--finite" => {
            println!("{}", finite(fs::read(cubin)?)?);
            return Ok(());
        }
        [flag, cubin, artifact, subset, eos, cap] if flag == "--parity" => {
            println!("{}", parity(cubin, artifact, subset, eos, cap)?);
            return Ok(());
        }
        [flag, cubin, artifact, fixture] if flag == "--historical" => {
            println!("{}", historical(cubin, artifact, fixture)?);
            return Ok(());
        }
        _ => {}
    }
    let [cubin, artifact, pairs, eos, cap] = args.as_slice() else {
        return Err("usage: compact_replay CUBIN ARTIFACT PAIRS.jsonl EOS[,EOS] CAP | --finite CUBIN | --parity CUBIN ARTIFACT SIX_PAIRS.jsonl EOS[,EOS] CAP | --historical CUBIN ARTIFACT FIXTURE.json".into());
    };
    let (model, engine, artifact_sha256, cubin_sha256) = load_engine(cubin, artifact)?;
    let (eos, cap) = parse_decode_parameters(&model, eos, cap)?;
    eprintln!(
        "{}",
        json!({
            "gpu":engine.gpu_name,"gpu_cc":engine.gpu_cc,"artifact_sha256":artifact_sha256,
            "cubin_sha256":cubin_sha256,"eos":eos,"cap":cap,"pairs":pairs,
        })
    );
    let mut out = BufWriter::new(io::stdout().lock());
    let mut seen = std::collections::HashSet::new();
    for line in BufReader::new(fs::File::open(pairs)?).lines() {
        let pair: Pair = serde_json::from_str(&line?)?;
        if !seen.insert(pair.pair_id) {
            return Err(format!("duplicate pair {}", pair.pair_id).into());
        }
        writeln!(out, "{}", evaluate_pair(&engine, &model, &pair, eos, cap)?)?;
        out.flush()?;
    }
    if seen.is_empty() {
        return Err("input contains zero pairs".into());
    }
    Ok(())
}
