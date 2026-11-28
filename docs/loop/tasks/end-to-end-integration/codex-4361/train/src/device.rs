use crate::cuda_backend::{
    gpu::{CudaV3ButterflyEngine, INT_V3_CUBIN_BYTES},
    FlatV3AdamWBuffers,
};
use crate::*;
use cudarc::driver::CudaContext;
use serde_json::Value;
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::Instant,
};
use uor_r4_native_vector::{
    butterfly_rotor::{splitmix64, LossKind},
    butterfly_rotor_v3::{ButterflyV3AdamWState, ButterflyV3Config, ButterflyV3Model},
};

const LEARN_PKM: bool = true;
const SPARSE_SKIP: bool = true;
const FINITE_PROTOCOL: &str = "codex-4361-finite/v1";
const ARM2_INITIAL_STEP: u64 = 3_333;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
struct Accumulated {
    loss_q15: u128,
    targets: u64,
}

impl Accumulated {
    fn add(&mut self, loss: u128, targets: u64) -> Result<()> {
        self.loss_q15 = self
            .loss_q15
            .checked_add(loss)
            .ok_or("summed loss overflow")?;
        self.targets = self
            .targets
            .checked_add(targets)
            .ok_or("summed target count overflow")?;
        Ok(())
    }
}

fn identity() -> Result<(String, String, String)> {
    if std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() != Ok("1") {
        return Err("set CUDA_DISABLE_PTX_JIT=1 before CUDA startup".into());
    }
    let ctx = CudaContext::new(0)?;
    let cc = ctx.compute_capability()?;
    if cc != (8, 9) {
        return Err(format!("registered runner requires sm89, found {cc:?}").into());
    }
    let name = ctx.name()?;
    if !name.contains("RTX 4090") {
        return Err(format!("registered runner requires RTX 4090, found {name}").into());
    }
    let uuid = ctx.uuid()?.bytes.map(|word| word as u8);
    Ok((name, hex::encode(uuid), sha256(INT_V3_CUBIN_BYTES)))
}

fn launch_group(engine: &mut CudaV3ButterflyEngine, group: &[Sequence]) -> Result<Accumulated> {
    let stride = group
        .iter()
        .map(|s| s.tokens.len())
        .max()
        .ok_or("empty launch")?;
    let mut tokens = Vec::with_capacity(group.len() * stride);
    let mut mask = Vec::with_capacity(group.len() * stride);
    let mut lens = Vec::with_capacity(group.len());
    for seq in group {
        lens.push(u32::try_from(seq.tokens.len())?);
        tokens.extend_from_slice(&seq.tokens);
        tokens.resize(tokens.len() + stride - seq.tokens.len(), BOS);
        mask.extend_from_slice(&seq.mask);
        mask.resize(mask.len() + stride - seq.mask.len(), 0);
    }
    let (loss_q15, targets) = engine.accumulate_batch(&tokens, &mask, &lens, stride, LEARN_PKM)?;
    Ok(Accumulated { loss_q15, targets })
}

fn launch_all(engine: &mut CudaV3ButterflyEngine, rows: &[Sequence]) -> Result<Accumulated> {
    let mut total = Accumulated::default();
    for group in rows.chunks(DOCUMENTS_PER_UPDATE) {
        let count = launch_group(engine, group)?;
        total.add(count.loss_q15, count.targets)?;
    }
    Ok(total)
}

fn accumulate(
    engine: &mut CudaV3ButterflyEngine,
    batch: &Batch,
    check_padding: bool,
) -> Result<Accumulated> {
    if !batch.padding.is_empty() {
        let before = if check_padding {
            Some(engine.download_grads()?)
        } else {
            None
        };
        if launch_all(engine, &batch.padding)? != Accumulated::default() {
            return Err("zero-loss isolated padding reported supervised work".into());
        }
        if let Some(before) = before {
            compare_words("padding gradient", &before, &engine.download_grads()?)?;
        }
    }
    // Discard padding gradients independently of its finite zero-gradient proof.
    engine.zero_grads()?;
    let actual = launch_all(engine, &batch.forward)?;
    if actual.targets != batch.work.targets as u64 {
        return Err(format!(
            "CUDA reports {} targets, pinned mask requires {}",
            actual.targets, batch.work.targets
        )
        .into());
    }
    Ok(actual)
}

fn compare_words<T: Eq + std::fmt::Debug>(label: &str, expected: &[T], actual: &[T]) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{label}: {} oracle words, {} CUDA words",
            expected.len(),
            actual.len()
        )
        .into());
    }
    if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        return Err(format!(
            "{label} mismatch at {index}: oracle {:?}, CUDA {:?}",
            expected[index], actual[index]
        )
        .into());
    }
    Ok(())
}

fn compare_snapshot(
    label: &str,
    engine: &CudaV3ButterflyEngine,
    oracle: &ButterflyV3Model,
    oracle_adam: &ButterflyV3AdamWState,
) -> Result<()> {
    let mut actual = oracle.clone();
    let mut actual_adam = oracle.new_adamw_state();
    engine.download_into(&mut actual, &mut actual_adam)?;
    let cpu = FlatV3AdamWBuffers::pack(oracle, oracle_adam);
    let gpu = FlatV3AdamWBuffers::pack(&actual, &actual_adam);
    compare_words(&format!("{label} weights"), &cpu.weights, &gpu.weights)?;
    compare_words(&format!("{label} first moments"), &cpu.adam_m, &gpu.adam_m)?;
    compare_words(&format!("{label} second moments"), &cpu.adam_v, &gpu.adam_v)?;
    if actual.to_bytes() != oracle.to_bytes() {
        return Err(format!("{label} checkpoint serialization diverged").into());
    }
    Ok(())
}

fn create_private(path: &Path) -> Result<fs::File> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn save_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_private(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn configured_arm2(path: &Path) -> Result<ButterflyV3Model> {
    let bytes = fs::read(path)?;
    if sha256(&bytes) != ARM2_SHA256 {
        return Err("warm start artifact is not the registered Arm 2 SHA-256".into());
    }
    let model = ButterflyV3Model::from_bytes(&bytes)?;
    if model.step != ARM2_INITIAL_STEP || model.to_bytes() != bytes {
        return Err(format!(
            "bound Arm 2 failed exact reserialization or step readback at step {}",
            model.step
        )
        .into());
    }
    Ok(model)
}

pub(super) fn train(
    checkpoint: &Path,
    tape_path: &Path,
    tape_sha: &str,
    output: &Path,
    finite_path: &Path,
) -> Result<()> {
    let model = configured_arm2(checkpoint)?;
    let tape = load_tape(tape_path, tape_sha)?;
    if output.exists() {
        return Err("training output directory already exists".into());
    }
    let (gpu_name, gpu_uuid, cubin_sha256) = identity()?;
    let finite_bytes = fs::read(finite_path)?;
    let finite: Value = serde_json::from_slice(&finite_bytes)?;
    if finite["verdict"] != "PASS"
        || finite["protocol"] != FINITE_PROTOCOL
        || finite["cubin_sha256"] != cubin_sha256
        || finite["gpu"] != gpu_name
        || finite["gpu_uuid"] != gpu_uuid
        || finite["sm"] != "89"
        || finite["long_allocation_transitions"] != MAX_TRANSITIONS
    {
        return Err("physical finite receipt does not match this executable/device/class".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().mode(0o700).create(output)?;
    }
    #[cfg(not(unix))]
    fs::create_dir(output)?;

    let mut log = BufWriter::new(create_private(&output.join("updates.jsonl"))?);
    let run_start = Instant::now();
    let fresh = model.new_adamw_state();
    let mut full =
        CudaV3ButterflyEngine::new(&model, &fresh, DOCUMENTS_PER_UPDATE, MAX_TRANSITIONS)?;
    let mut reset =
        CudaV3ButterflyEngine::new(&model, &fresh, DOCUMENTS_PER_UPDATE, CHUNK_IDS - 1)?;
    let mut total_targets = 0usize;
    let mut total_transitions = 0usize;
    let mut padding_transitions = 0usize;
    for (index, docs) in tape.docs.chunks_exact(DOCUMENTS_PER_UPDATE).enumerate() {
        let update_start = Instant::now();
        let full_batch = prepare(docs, Arm::Full)?;
        let reset_batch = prepare(docs, Arm::Reset)?;
        match_batches(&full_batch, &reset_batch)?;
        let full_counts = accumulate(&mut full, &full_batch, false)?;
        let reset_counts = accumulate(&mut reset, &reset_batch, false)?;
        if full_counts.targets != reset_counts.targets || full_counts.targets == 0 {
            return Err(format!("different actual target counts at update {}", index + 1).into());
        }
        let step = model.step + (index + 1) as u64;
        // Both engines apply the historical summed-gradient integer AdamW once per
        // identical four-document/target set. The native seed uses post-increment step.
        full.apply_adamw_step(step, LR_Q8, WD_SHIFT, EPS, SPARSE_SKIP, LEARN_PKM)?;
        reset.apply_adamw_step(step, LR_Q8, WD_SHIFT, EPS, SPARSE_SKIP, LEARN_PKM)?;
        total_targets += full_batch.work.targets;
        total_transitions += full_batch.work.total_transitions();
        padding_transitions += reset_batch.work.padding_transitions;
        writeln!(
            log,
            "{}",
            json!({
                "update": index + 1, "step_after_increment":step,
                "doc_ids": docs.iter().map(|doc| doc.id).collect::<Vec<_>>(),
                "full": full_batch.work, "reset": reset_batch.work,
                "full_device": full_counts, "reset_device": reset_counts,
                "wall_ms": update_start.elapsed().as_millis(),
            })
        )?;
        log.flush()?;
    }
    log.get_ref().sync_all()?;

    let final_step = model.step + UPDATES as u64;
    let export = |engine: &CudaV3ButterflyEngine| -> Result<Vec<u8>> {
        let mut final_model = model.clone();
        let mut final_adam = final_model.new_adamw_state();
        engine.download_into(&mut final_model, &mut final_adam)?;
        final_model.step = final_step;
        let bytes = final_model.to_bytes();
        if ButterflyV3Model::from_bytes(&bytes)? != final_model
            || final_model.cfg != model.cfg
            || final_model.vocab_perm != model.vocab_perm
        {
            return Err("final artifact codec/config/permutation readback mismatch".into());
        }
        Ok(bytes)
    };
    let full_bytes = export(&full)?;
    let reset_bytes = export(&reset)?;
    let original = model.to_bytes();
    let same_parameters =
        |bytes: &[u8]| bytes[60..bytes.len() - 8] == original[60..original.len() - 8];

    save_new(&output.join("full.uorbfy03"), &full_bytes)?;
    save_new(&output.join("reset.uorbfy03"), &reset_bytes)?;
    let receipt = json!({
        "verdict":"COMPLETE","gpu":gpu_name,"gpu_uuid":gpu_uuid,"sm":"89","cubin_sha256":cubin_sha256,
        "finite_sha256":sha256(&finite_bytes),
        "initial_checkpoint_sha256":ARM2_SHA256,"initial_step":model.step,"final_step":final_step,
        "train_sha256":tape.sha256,"near":tape.near,"far":tape.far,"intermediate":tape.intermediate,
        "updates":UPDATES,"documents_per_update":DOCUMENTS_PER_UPDATE,"total_targets_per_arm":total_targets,
        "transitions_per_arm":total_transitions,"reset_semantic_transitions":total_transitions-padding_transitions,
        "reset_padding_transitions":padding_transitions,"learning_rate_q8":LR_Q8,"weight_decay_shift":WD_SHIFT,
        "eps":EPS,"sparse_skip":SPARSE_SKIP,"learn_pkm_keys":LEARN_PKM,"moments_initial":"zero",
        "gradient_scale":"exact native unnormalized target sum, identical documents and targets",
        "full_sha256":sha256(&full_bytes),"reset_sha256":sha256(&reset_bytes),
        "full_parameter_payload_changed":!same_parameters(&full_bytes),"reset_parameter_payload_changed":!same_parameters(&reset_bytes),
        "full_bytes":full_bytes.len(),"reset_bytes":reset_bytes.len(),"wall_ms":run_start.elapsed().as_millis(),
    });
    save_new(
        &output.join("COMPLETE.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}

fn planted(id: u64, prefix_len: usize) -> Result<Document> {
    let mut prefix: Vec<_> = (0..prefix_len)
        .map(|i| 32 + (splitmix64(id.wrapping_mul(8191) + i as u64) % 96) as u16)
        .collect();
    prefix[0] = BOS;
    prefix[5] = EOS;
    prefix[10] = EOS;
    if prefix_len > CHUNK_IDS {
        prefix[CHUNK_IDS - 2] = 4_000 + id as u16;
    }
    compile(Row {
        id,
        prefix,
        expected: vec![256 + id as u16, 269, EOS],
        stratum: "far".into(),
    })
}

fn synth_config(full_shape: bool) -> ButterflyV3Config {
    ButterflyV3Config {
        vocab: usize::from(VOCAB),
        dim: if full_shape { 512 } else { 16 },
        layers: if full_shape { 6 } else { 2 },
        ffn_branches: if full_shape { 4 } else { 2 },
        pkm_sub_keys: if full_shape { 64 } else { 4 },
        pkm_key_dim: if full_shape { 32 } else { 4 },
        loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 64 },
        seed: 20261128,
    }
}

fn synthetic_oracle() -> Result<ButterflyV3Model> {
    let mut model = ButterflyV3Model::new(synth_config(false))?;
    model.step = ARM2_INITIAL_STEP;
    let mut perm = model.vocab_perm.clone();
    perm.swap(49, 74);
    model.set_vocab_perm(perm)?;
    let dim = model.cfg.dim;
    // Slow nonzero carry and all shift taps active expose the actual reset boundary.
    for layer in &mut model.layers {
        layer.gate_bias_q15.fill(2_048);
        layer.shift_mix_q15[dim..2 * dim].fill(4_096);
        layer.shift_mix_q15[2 * dim..3 * dim].fill(-2_048);
        layer.shift_mix_q15[3 * dim..4 * dim].fill(1_024);
    }
    Ok(model)
}

fn oracle_rows(docs: &[Document], arm: Arm) -> Vec<(&[u16], &[u8])> {
    let mut sequences = Vec::new();
    for doc in docs {
        let t = &doc.full.tokens;
        let m = &doc.full.mask;
        if arm == Arm::Full {
            sequences.push((t.as_slice(), m.as_slice()));
        } else {
            let mut start = 0;
            while start + 1 < t.len() {
                let end = (start + 257).min(t.len());
                sequences.push((&t[start..end], &m[start..end]));
                start = end;
            }
        }
    }
    sequences
}

fn verify_serving(
    engine: &CudaV3ButterflyEngine,
    oracle: &ButterflyV3Model,
    tokens: &[u16],
) -> Result<usize> {
    let mut mask = vec![1; tokens.len()];
    mask[0] = 0;
    let (mass, greedy) = engine.eval_records(tokens, &mask, &[0], &[tokens.len() as u32], false)?;
    let mut session = oracle.new_session();
    for (index, pair) in tokens.windows(2).enumerate() {
        let out = oracle.step(pair[0], &mut session)?;
        let target_mass = oracle.target_q48_mass_from_hidden(&out.hidden_q15, pair[1])?;
        if greedy[index + 1] != out.greedy_token || mass[index + 1] != target_mass {
            return Err(
                format!("serving GPU/Rust greedy or target mass mismatch at {index}").into(),
            );
        }
    }
    Ok(tokens.len() - 1)
}

fn verify_forward_cells(
    engine: &mut CudaV3ButterflyEngine,
    oracle: &ButterflyV3Model,
    sequence: &Sequence,
) -> Result<usize> {
    engine.zero_grads()?;
    launch_group(engine, std::slice::from_ref(sequence))?;
    let saved = engine.stream.clone_dtoh(&engine.d_act_ws)?;
    let dim = oracle.cfg.dim;
    let max = engine.max_seq_len;
    let layer_words = oracle.cfg.layers * max * dim;
    let mut session = oracle.new_session();
    for (step, &token) in sequence.tokens[..sequence.tokens.len() - 1]
        .iter()
        .enumerate()
    {
        for layer in 0..oracle.cfg.layers {
            let start = 2 * layer_words + (layer * max + step) * dim;
            compare_words(
                "saved pre-input layer carry",
                &session.carry[layer],
                &saved[start..start + dim],
            )?;
            for tap in 1..=3 {
                let expected = &session.shift_hist[layer][(tap - 1) * dim..tap * dim];
                if let Some(prior) = step.checked_sub(tap) {
                    let start = layer_words + (layer * max + prior) * dim;
                    compare_words(
                        "saved pre-input layer shift history",
                        expected,
                        &saved[start..start + dim],
                    )?;
                } else if expected.iter().any(|&word| word != 0) {
                    return Err("fresh Rust session shift history did not begin at zero".into());
                }
            }
        }
        oracle.step(token, &mut session)?;
        if session.tokens_seen != (step + 1) as u64 {
            return Err(
                "Rust replay consumed count disagrees with CUDA activation timestep".into(),
            );
        }
        for layer in 0..oracle.cfg.layers {
            let now = layer_words + (layer * max + step) * dim;
            compare_words(
                "saved layer norm words for next shift history",
                &session.shift_hist[layer][..dim],
                &saved[now..now + dim],
            )?;
        }
    }
    Ok(sequence.tokens.len() - 1)
}

struct CheckedUpdate {
    model: ButterflyV3Model,
    gradient: Vec<i64>,
    counts: Accumulated,
    forward_checks: usize,
    serving_checks: usize,
}

fn finite_update(oracle: &ButterflyV3Model, docs: &[Document], arm: Arm) -> Result<CheckedUpdate> {
    let mut initial = oracle.new_adamw_state();
    let max = if arm == Arm::Full {
        MAX_TRANSITIONS
    } else {
        CHUNK_IDS - 1
    };
    let mut gpu = CudaV3ButterflyEngine::new(oracle, &initial, DOCUMENTS_PER_UPDATE, max)?;
    compare_snapshot("zero-update cold reload", &gpu, oracle, &initial)?;
    let batch = prepare(docs, arm)?;
    let inputs = oracle_rows(docs, arm);
    let mut cpu_counts = Accumulated::default();
    let mut cpu_grad = oracle.zero_grad();
    for (tokens, mask) in &inputs {
        let (loss, targets) =
            oracle.accumulate_sequence_grad(tokens, mask, &mut cpu_grad, LEARN_PKM)?;
        cpu_counts.add(loss, targets)?;
    }

    let mut forward_checks = 0;
    let mut serving_checks = 0;
    for seq in batch.forward.iter().filter(|s| s.tokens.len() >= 2) {
        forward_checks += verify_forward_cells(&mut gpu, oracle, seq)?;
        serving_checks += verify_serving(&gpu, oracle, &seq.tokens)?;
    }
    let gpu_counts = accumulate(&mut gpu, &batch, true)?;
    if cpu_counts != gpu_counts {
        return Err(format!(
            "finite {arm:?} target/loss mismatch: Rust {cpu_counts:?}, CUDA {gpu_counts:?}"
        )
        .into());
    }
    let expected = FlatV3AdamWBuffers::pack_grad(oracle, &cpu_grad);
    let actual = gpu.download_grads()?;
    compare_words("accumulated native gradient", &expected, &actual)?;
    compare_snapshot(
        "after gradients and zero-loss padding",
        &gpu,
        oracle,
        &initial,
    )?;
    let mut cpu_model = oracle.clone();
    cpu_model.apply_adamw_step(
        &cpu_grad,
        &mut initial,
        LR_Q8,
        WD_SHIFT,
        EPS,
        SPARSE_SKIP,
        LEARN_PKM,
    );
    gpu.apply_adamw_step(cpu_model.step, LR_Q8, WD_SHIFT, EPS, SPARSE_SKIP, LEARN_PKM)?;
    compare_snapshot("post-step native AdamW", &gpu, &cpu_model, &initial)?;

    let bytes = cpu_model.to_bytes();
    let reloaded = ButterflyV3Model::from_bytes(&bytes)?;
    if reloaded != cpu_model || reloaded.step != ARM2_INITIAL_STEP + 1 {
        return Err("post-increment synthetic checkpoint round-trip failed".into());
    }
    for (tokens, _) in inputs {
        serving_checks += verify_serving(&gpu, &reloaded, tokens)?;
    }
    Ok(CheckedUpdate {
        model: reloaded,
        gradient: actual,
        counts: gpu_counts,
        forward_checks,
        serving_checks,
    })
}

fn planted_carry_control(oracle: &ButterflyV3Model, doc: &Document) -> Result<()> {
    let final_input = doc
        .full
        .mask
        .iter()
        .position(|&m| m == 1)
        .ok_or("no synthetic answer")?
        - 1;
    let tokens = &doc.full.tokens;
    let mut uninterrupted = oracle.new_session();
    let mut reset = oracle.new_session();
    for &token in &tokens[..final_input] {
        oracle.step(token, &mut uninterrupted)?;
    }
    for &token in &tokens[CHUNK_IDS..final_input] {
        oracle.step(token, &mut reset)?;
    }
    let mut zero = uninterrupted.clone();
    for layer in &mut zero.carry {
        layer.fill(0);
    }
    let input = tokens[final_input];
    let native = oracle.step(input, &mut uninterrupted)?;
    let sham_reset = oracle.step(input, &mut reset)?;
    let zero_carry = oracle.step(input, &mut zero)?;
    if native.hidden_q15 == sham_reset.hidden_q15 || native.hidden_q15 == zero_carry.hidden_q15 {
        return Err("synthetic weak-carry test has no hidden-state intervention signal".into());
    }
    Ok(())
}

fn capacity_probe() -> Result<Value> {
    let device = CudaContext::new(0)?;
    let (free_before, physical_total) = device.mem_get_info()?;
    let model = ButterflyV3Model::new(synth_config(true))?;
    let adam = model.new_adamw_state();
    let maximum_ids_tail = MAX_IDS - 4 * CHUNK_IDS - 1;
    let prefix_len = MAX_IDS - maximum_ids_tail;
    let docs = (0..DOCUMENTS_PER_UPDATE)
        .map(|id| {
            let doc = planted(id as u64, prefix_len)?;
            let mut expected = vec![256 + id as u16; maximum_ids_tail];
            *expected.last_mut().ok_or("empty capacity target")? = EOS;
            compile(Row {
                id: id as u64,
                prefix: doc.full.tokens[..prefix_len].to_vec(),
                expected,
                stratum: "far".into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let full_batch = prepare(&docs, Arm::Full)?;
    let reset_batch = prepare(&docs, Arm::Reset)?;
    match_batches(&full_batch, &reset_batch)?;
    let mut full =
        CudaV3ButterflyEngine::new(&model, &adam, DOCUMENTS_PER_UPDATE, MAX_TRANSITIONS)?;
    let mut reset = CudaV3ButterflyEngine::new(&model, &adam, DOCUMENTS_PER_UPDATE, CHUNK_IDS - 1)?;
    let start = Instant::now();
    accumulate(&mut full, &full_batch, false)?;
    full.apply_adamw_step(1, LR_Q8, WD_SHIFT, EPS, SPARSE_SKIP, LEARN_PKM)?;
    let full_ms = start.elapsed().as_millis();
    let start = Instant::now();
    accumulate(&mut reset, &reset_batch, false)?;
    reset.apply_adamw_step(1, LR_Q8, WD_SHIFT, EPS, SPARSE_SKIP, LEARN_PKM)?;
    let reset_ms = start.elapsed().as_millis();
    let (free_after, _) = device.mem_get_info()?;
    Ok(json!({
        "shape":{"dim":512,"layers":6,"ffn_branches":4,"pkm_sub_keys":64,"pkm_key_dim":32},
        "full":full_batch.work,"reset":reset_batch.work,"full_update_wall_ms":full_ms,
        "device_total_bytes":physical_total,"device_free_before_bytes":free_before,"device_free_after_updates_bytes":free_after,
        "device_used_delta_bytes":free_before.saturating_sub(free_after),
        "reset_update_wall_ms":reset_ms,"64_update_pair_projection_ms":(full_ms+reset_ms)*UPDATES as u128,
        "base_device_buffer_bytes":2*model.param_count()*22+DOCUMENTS_PER_UPDATE*(3*model.cfg.layers+2)
            *(MAX_TRANSITIONS+CHUNK_IDS-1)*model.cfg.dim*4,
    }))
}

pub(super) fn finite(output: &Path) -> Result<()> {
    if output.exists() {
        return Err("finite receipt already exists; use a fresh path".into());
    }
    let (gpu_name, gpu_uuid, cubin_sha256) = identity()?;
    let oracle = synthetic_oracle()?;
    let short = (0..DOCUMENTS_PER_UPDATE)
        .map(|id| planted(id as u64, 18 + id))
        .collect::<Result<Vec<_>>>()?;
    let short_full = finite_update(&oracle, &short, Arm::Full)?;
    let short_reset = finite_update(&oracle, &short, Arm::Reset)?;
    if short_full.model != short_reset.model
        || short_full.gradient != short_reset.gradient
        || short_full.counts != short_reset.counts
    {
        return Err("identical short-row sham did not match byte-for-byte".into());
    }

    let long = (0..DOCUMENTS_PER_UPDATE)
        .map(|id| planted(id as u64, 260 + id))
        .collect::<Result<Vec<_>>>()?;
    let planned_full = prepare(&long, Arm::Full)?;
    let planned_reset = prepare(&long, Arm::Reset)?;
    match_batches(&planned_full, &planned_reset)?;
    planted_carry_control(&oracle, &long[0])?;
    let long_full = finite_update(&oracle, &long, Arm::Full)?;
    let long_reset = finite_update(&oracle, &long, Arm::Reset)?;
    let dim = oracle.cfg.dim;
    let mut earlier_rows_full_descent = 0;
    for i in 0..long.len() {
        let token = 4_000 + i;
        let slot = usize::from(oracle.vocab_perm[token]);
        let range = slot * dim..(slot + 1) * dim;
        if long_reset.gradient[range.clone()].iter().any(|&g| g != 0) {
            return Err("reset BPTT reached an earlier-window exclusive embedding row".into());
        }
        earlier_rows_full_descent += usize::from(long_full.gradient[range].iter().any(|&g| g != 0));
    }
    if long_full.model == long_reset.model
        || long_full.gradient == long_reset.gradient
        || long_full.counts.targets != long_reset.counts.targets
        || earlier_rows_full_descent == 0
    {
        return Err("planted physical gradient intervention is vacuous or targets differ".into());
    }

    let max_shape = capacity_probe()?;
    let receipt = json!({
        "verdict":"PASS","protocol":FINITE_PROTOCOL,"gpu":gpu_name,"gpu_uuid":gpu_uuid,"sm":"89","cubin_sha256":cubin_sha256,
        "long_allocation_transitions":MAX_TRANSITIONS,"short_sham":true,"cold_zero_update":true,
        "padding_gradient_exact":true,"weak_carry_intervention":true,
        "full_backward_earlier_exclusive_rows":earlier_rows_full_descent,
        "reset_backward_earlier_exclusive_rows":0,
        "short_targets":short_full.counts.targets,"cross_boundary_targets":long_full.counts.targets,
        "full_training_gradient_words_checked":long_full.gradient.len(),
        "reset_training_gradient_words_checked":long_reset.gradient.len(),
        "forward_state_transitions_checked":short_full.forward_checks+short_reset.forward_checks+long_full.forward_checks+long_reset.forward_checks,
        "serving_greedy_mass_transitions_checked":short_full.serving_checks+short_reset.serving_checks+long_full.serving_checks+long_reset.serving_checks,
        "full_after_step_sha256":sha256(&long_full.model.to_bytes()),"reset_after_step_sha256":sha256(&long_reset.model.to_bytes()),
        "synthetic_max_shape":max_shape,
    });
    save_new(output, &serde_json::to_vec_pretty(&receipt)?)?;
    println!("{receipt}");
    Ok(())
}
