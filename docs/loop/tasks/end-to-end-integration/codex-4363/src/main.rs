#![recursion_limit = "256"]

mod calibrated {
    #![allow(dead_code)]
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../codex-4360/runner/src/bin/device.rs"
    ));
    include!("finite.rs");

    use std::{collections::BTreeSet, path::Path};
    use uor_r4_native_vector::butterfly_rotor::loss_and_grad_64;

    const TRAIN_SHA: &str = "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a";
    const CUBIN_SHA: &str = "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713";
    const HISTORICAL_SHA: &str = "dcc3019546fd1fe6e272aa71a64d694db2754b946bc82638f64b19671e47d04d";
    const ARTIFACTS: [(&str, &str, u64); 3] = [
        (
            "frozen",
            "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589",
            3333,
        ),
        (
            "reset",
            "3656e0575bd9e590e63305000e6f03ca010e1a663d6ac45d9061961f9a356bc2",
            3397,
        ),
        (
            "preserved",
            "8e0721cc8b989f12af9c80b88dfac53c3a33701e6a3e8e7d2d26d2dec7fd1340",
            3397,
        ),
    ];
    const CHUNK_IDS: usize = 257;
    const MAX_IDS: usize = 1088;
    const DOCUMENTS_PER_UPDATE: usize = 4;
    const TRAIN_ROWS: usize = 256;
    const LABEL_COUNT: usize = 2151;
    const BOS: u16 = 0;
    const EOS: u16 = 1;
    const VOCAB: u16 = 4096;

    #[derive(Deserialize)]
    struct TrainingRow {
        id: u64,
        stratum: String,
        #[serde(flatten)]
        row: Row,
    }

    #[derive(Debug, PartialEq, Eq, Serialize)]
    struct Election {
        winner: usize,
        runner_up: usize,
        winner_score: i32,
        runner_up_score: i32,
        margin: i32,
        equal_maxima: Vec<usize>,
    }

    fn elect(scores: &[i32; 64], admitted: impl Fn(usize) -> bool) -> Result<Election> {
        let mut slots: Vec<_> = (0..scores.len()).filter(|&id| admitted(id)).collect();
        slots.sort_unstable_by_key(|&slot| (std::cmp::Reverse(scores[slot]), slot));
        let [winner, runner_up, ..] = slots.as_slice() else {
            return Err("expected at least two admissible entries".into());
        };
        let equal_maxima = slots
            .iter()
            .copied()
            .take_while(|&slot| scores[slot] == scores[*winner])
            .collect();
        Ok(Election {
            winner: *winner,
            runner_up: *runner_up,
            winner_score: scores[*winner],
            runner_up_score: scores[*runner_up],
            margin: scores[*winner]
                .checked_sub(scores[*runner_up])
                .ok_or("score margin overflow")?,
            equal_maxima,
        })
    }

    fn fine_admissible(model: &ButterflyV3Model, cell: usize, fine: usize) -> bool {
        admissible(model, model.vocab_inv[cell * HIER_FINE + fine])
    }

    fn coarse_admissible(model: &ButterflyV3Model, cell: usize) -> bool {
        (0..HIER_FINE).any(|fine| fine_admissible(model, cell, fine))
    }

    fn finite_host() -> Result<Value> {
        let mut rows = Vec::with_capacity(FIXTURES.len());
        for fixture in FIXTURES {
            let mut scores = [-1; 64];
            scores[..3].copy_from_slice(&fixture.planted);
            let result = elect(&scores, |slot| !(fixture.masked && slot == 0))?;
            check_fixture_result(fixture, &result, "host")?;

            // A single-coordinate integer witness gives a small, independently
            // calculable dot without invoking the model or initializing CUDA.
            let hidden = [32768; 16];
            let mut row = [0; 16];
            for desired in fixture.planted {
                plant_score(&mut row, &hidden, desired)?;
                let total: i32 = row.iter().map(|&word| i32::from(word)).sum();
                if total != desired * 4 {
                    return Err("host integer weight witness mismatch".into());
                }
            }
            rows.push(json!({"fixture":fixture.name,"election":result}));
        }

        let mut scores = [-1; 64];
        scores[0] = 17;
        scores[1] = 17;
        scores[9] = 17;
        let tie = elect(&scores, |_| true)?;
        if tie.equal_maxima != [0, 1, 9] || elect(&scores, |slot| slot == 0).is_ok() {
            return Err("host three-way tie or insufficient-candidate guard mismatch".into());
        }
        Ok(json!({
            "finite_host":"PASS", "scope":"synthetic score ordering and integer weight witness",
            "fixtures":rows,"three_way_tie":tie
        }))
    }

    fn sha256(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn pinned_bytes(path: &str, pin: &str) -> Result<Vec<u8>> {
        let bytes = fs::read(path)?;
        if sha256(&bytes) != pin {
            return Err(format!("SHA-256 mismatch: {path}").into());
        }
        Ok(bytes)
    }

    fn get_rows(path: &str) -> Result<Vec<TrainingRow>> {
        let bytes = pinned_bytes(path, TRAIN_SHA)?;
        let fit: Vec<TrainingRow> = BufReader::new(bytes.as_slice())
            .lines()
            .map(|line| Ok(serde_json::from_str(&line?)?))
            .collect::<Result<_>>()?;
        if fit.len() != TRAIN_ROWS
            || fit.iter().map(|row| row.row.expected.len()).sum::<usize>() != LABEL_COUNT
        {
            return Err("expected the registered 256 TRAIN rows and 2151 label positions".into());
        }
        let mut seen = BTreeSet::new();
        for row in &fit {
            let prefix = &row.row.prefix;
            let target = &row.row.expected;
            let first = prefix.len();
            let total = first + target.len();
            if !seen.insert(row.id)
                || prefix.first() != Some(&BOS)
                || prefix.iter().skip(1).any(|&id| id == BOS || id >= VOCAB)
                || prefix.iter().filter(|&&id| id == EOS).count() != 2
                || target.len() < 2
                || target.last() != Some(&EOS)
                || target[..target.len() - 1]
                    .iter()
                    .any(|&id| id <= EOS || id >= VOCAB)
                || !matches!(row.stratum.as_str(), "near" | "far" | "intermediate")
                || total > MAX_IDS
                || first / CHUNK_IDS != (total - 1) / CHUNK_IDS
                || first % CHUNK_IDS == 0
            {
                return Err(format!("invalid registered TRAIN opening {}", row.id).into());
            }
        }
        Ok(fit)
    }

    struct ScoreWords<'a> {
        hidden: &'a [i32],
        coarse: &'a [i32; HIER_COARSE],
        target_fine: &'a [i32; HIER_FINE],
        elected_fine: &'a [i32; HIER_FINE],
        pkm: &'a [i32],
    }

    impl<'a> ScoreWords<'a> {
        fn read(cfg: ButterflyV3Config, trace: &'a Launch) -> Result<Self> {
            if trace.n_emitted != 1 || trace.trace.len() != trace_stride(cfg) {
                return Err("expected exactly one device trace record".into());
            }
            let (hidden, tail) = trace.trace[2..].split_at(cfg.dim);
            let (coarse, tail) = tail.split_at(HIER_COARSE);
            let (target_fine, tail) = tail.split_at(HIER_FINE);
            let (elected_fine, pkm) = tail.split_at(HIER_FINE);
            Ok(Self {
                hidden,
                coarse: coarse.try_into()?,
                target_fine: target_fine.try_into()?,
                elected_fine: elected_fine.try_into()?,
                pkm,
            })
        }
    }

    fn first_step(
        engine: &Engine,
        model: &ButterflyV3Model,
        state: &State,
        input: u16,
        targets: [u16; 2],
    ) -> Result<Launch> {
        engine.launch(
            model.cfg,
            state,
            Request {
                inputs: &[input],
                rollout: true,
                steps: 1,
                eos: [EOS; 2],
                targets,
            },
        )
    }

    fn output_id(model: &ButterflyV3Model, run: &Launch) -> Result<u16> {
        let _ = ScoreWords::read(model.cfg, run)?;
        let slot = usize::try_from(run.trace[0])?;
        let id = *model
            .vocab_inv
            .get(slot)
            .ok_or("invalid elected slot in physical trace")?;
        if !admissible(model, id) {
            return Err(format!("device elected ineligible original ID {id}").into());
        }
        Ok(id)
    }

    fn check_label_independence(
        cfg: ButterflyV3Config,
        free: &Launch,
        rich: &Launch,
    ) -> Result<()> {
        let free_words = ScoreWords::read(cfg, free)?;
        let rich_words = ScoreWords::read(cfg, rich)?;
        if free.trace[..2] != rich.trace[..2]
            || free.state != rich.state
            || free_words.hidden != rich_words.hidden
            || free_words.coarse != rich_words.coarse
            || free_words.pkm != rich_words.pkm
        {
            return Err("diagnostic labels changed the physical transition".into());
        }
        Ok(())
    }

    fn q15_digest(words: &[i32]) -> String {
        let mut digest = Sha256::new();
        for word in words {
            digest.update(word.to_le_bytes());
        }
        hex::encode(digest.finalize())
    }

    fn state_digest(state: &State) -> Result<String> {
        Ok(sha256(&serde_json::to_vec(state)?))
    }

    fn signature(cfg: ButterflyV3Config, run: &Launch) -> Result<Value> {
        let words = ScoreWords::read(cfg, run)?;
        let after = state_digest(&run.state)?;
        let transition = json!({
            "elected_slot_and_score":&run.trace[..2],
            "hidden":words.hidden,
            "coarse":words.coarse.as_slice(),
            "pkm":words.pkm,
            "after_sha256":after,
        });
        Ok(json!({
            "native_transition_sha256":sha256(&serde_json::to_vec(&transition)?),
            "complete_trace_i32le_sha256":q15_digest(&run.trace),
            "diagnostic_mass_q48_le_sha256":sha256(
                &run.mass.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>()
            ),
        }))
    }

    fn capture(
        engine: &Engine,
        model: &ButterflyV3Model,
        example: &TrainingRow,
        index: usize,
    ) -> Result<Value> {
        let prefix = &example.row.prefix;
        let target_id = example.row.expected[0];
        let state = prefill(engine, model, &example.row)?;
        let input = *prefix.last().ok_or("empty authentic prefix")?;
        // The decision-bearing launch gets constant, non-reference placeholders.
        let free = first_step(engine, model, &state, input, [EOS; 2])?;
        let actual_id = output_id(model, &free)?;
        // The diagnostic labels address complete fine-head score words only.
        let rich = first_step(engine, model, &state, input, [target_id, actual_id])?;
        check_label_independence(model.cfg, &free, &rich)?;
        let sham = first_step(engine, model, &state, input, [EOS; 2])?;
        check_label_independence(model.cfg, &free, &sham)?;
        if free.trace != sham.trace || free.mass != sham.mass {
            return Err("repeated blind launch changed complete trace or masses".into());
        }
        let words = ScoreWords::read(model.cfg, &rich)?;
        let actual_slot = usize::from(model.vocab_perm[usize::from(actual_id)]);
        let target_slot = usize::from(model.vocab_perm[usize::from(target_id)]);
        let actual_cell = actual_slot / HIER_FINE;
        let target_cell = target_slot / HIER_FINE;
        let target_fine_slot = target_slot % HIER_FINE;
        let target_admitted = actual_cell == target_cell;
        let coarse_rank = elect(words.coarse, |cell| coarse_admissible(model, cell))?;
        let fine_rank = elect(words.elected_fine, |slot| {
            fine_admissible(model, actual_cell, slot)
        })?;
        let target_rank = elect(words.target_fine, |slot| {
            fine_admissible(model, target_cell, slot)
        })?;
        if coarse_rank.winner != actual_cell
            || fine_rank.winner != actual_slot % HIER_FINE
            || fine_rank.winner_score != free.trace[1]
            || (target_admitted && words.elected_fine != words.target_fine)
        {
            return Err(
                "raw score words disagree with the physical election or repeated cell".into(),
            );
        }

        let competitor = if target_rank.winner == target_fine_slot {
            target_rank.runner_up
        } else {
            target_rank.winner
        };
        let token_in = |cell, slot| model.vocab_inv[cell * HIER_FINE + slot];
        let (coarse_loss, coarse_dz) =
            loss_and_grad_64(words.coarse, target_cell, model.cfg.loss_kind);
        let (fine_loss, fine_dz) =
            loss_and_grad_64(words.target_fine, target_fine_slot, model.cfg.loss_kind);
        let fine_grad = |slot: usize| target_admitted.then_some(fine_dz[slot]);
        let local_rank = 1 + words
            .target_fine
            .iter()
            .enumerate()
            .filter(|&(slot, &score)| {
                fine_admissible(model, target_cell, slot)
                    && (score > words.target_fine[target_fine_slot]
                        || (score == words.target_fine[target_fine_slot]
                            && slot < target_fine_slot))
            })
            .count();

        Ok(json!({
            "id":example.id,"stratum":example.stratum,"training_document_index":index,
            "historical_update_index":index / DOCUMENTS_PER_UPDATE,
            "historical_train_step_before":ARTIFACTS[0].2 + (index / DOCUMENTS_PER_UPDATE) as u64,
            "target_id":target_id,"target_slot":target_slot,"target_cell":target_cell,
            "target_is_admitted":target_admitted,
            "target_coarse_raw_score_q15":words.coarse[target_cell],
            "target_fine_raw_score_q15":words.target_fine[target_fine_slot],
            "target_local_rank_in_true_cell":local_rank,
            "target_first_label_document_offset":prefix.len(),
            "historical_chunk":prefix.len()/CHUNK_IDS,
            "historical_reset_label_chunk_offset":prefix.len()%CHUNK_IDS,
            "first_target_mask":1,
            "input_original_id":input,"actual_first_id":actual_id,"actual_first_global_slot":actual_slot,
            "fine_runner_up_original_id":token_in(actual_cell, fine_rank.runner_up),
            "fine_equal_maxima_original_ids":fine_rank.equal_maxima.iter()
                .map(|&slot| token_in(actual_cell, slot)).collect::<Vec<_>>(),
            "target_cell_competitor_original_id":token_in(target_cell, competitor),
            "coarse":coarse_rank,"elected_fine":fine_rank,"forced_target_cell":target_rank,
            "cuda_raw_coarse_q15":words.coarse.as_slice(),
            "cuda_raw_elected_fine_q15":words.elected_fine.as_slice(),
            "cuda_raw_target_fine_q15":words.target_fine.as_slice(),
            "reference_free_transition_match":true,
            "blind_sham_complete_match":true,
            "physical_first_step_launches":3,
            "prefill_native_state_sha256":state_digest(&state)?,
            "blind_signature":signature(model.cfg, &free)?,
            "rich_signature":signature(model.cfg, &rich)?,
            "sham_signature":signature(model.cfg, &sham)?,
            "hidden_q15":words.hidden,
            "hidden_sha256":q15_digest(words.hidden),
            "pkm_slots":words.pkm,
            "score_gradients":{
                "scope":"native local score derivatives at this read-only checkpoint after the full serving prefix; host loss on physical CUDA score words",
                "coarse_loss_q15":coarse_loss,"fine_loss_q15":fine_loss,
                "target_coarse_dz_q15":coarse_dz[target_cell],"elected_coarse_dz_q15":coarse_dz[actual_cell],
                "coarse_runner_up_dz_q15":coarse_dz[coarse_rank.runner_up],
                "target_fine_dz_q15":fine_dz[target_fine_slot],"elected_fine_dz_q15":fine_grad(fine_rank.winner),
                "target_cell_competitor_dz_q15":fine_dz[competitor],
                "fine_runner_up_dz_q15":fine_grad(fine_rank.runner_up),
                "elected_fine_supervised_head":target_admitted,
                "realized_update_measured":false,
            },
        }))
    }

    fn require_aot() -> Result<()> {
        if std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() != Ok("1") {
            return Err("CUDA_DISABLE_PTX_JIT=1 required".into());
        }
        Ok(())
    }

    fn collect(cubin: &str, history: &str, tape: &str, paths: [&str; 3], out: &str) -> Result<()> {
        require_aot()?;
        if Path::new(out).exists() {
            return Err("output already exists".into());
        }
        let cubin_bytes = pinned_bytes(cubin, CUBIN_SHA)?;
        let _ = pinned_bytes(history, HISTORICAL_SHA)?;
        let fit = get_rows(tape)?;
        let mut checkpoints = Vec::with_capacity(ARTIFACTS.len());
        for ((name, sha, step), path) in ARTIFACTS.into_iter().zip(paths) {
            let bytes = pinned_bytes(path, sha)?;
            let model = ButterflyV3Model::from_bytes(&bytes)?;
            if model.step != step || model.cfg.vocab != usize::from(VOCAB) {
                return Err(format!("registered step/vocab mismatch for {name}").into());
            }
            checkpoints.push((name, sha, model, bytes));
        }
        if checkpoints
            .iter()
            .any(|(_, _, model, _)| model.vocab_perm != checkpoints[0].2.vocab_perm)
        {
            return Err("checkpoint permutation mismatch".into());
        }

        let host = finite_host()?;
        let physical = finite_device(&cubin_bytes)?;
        let historical = historical(cubin, paths[0], history)?;
        if historical["cubin_sha256"] != CUBIN_SHA
            || historical["artifact_sha256"] != ARTIFACTS[0].1
            || historical["fixture_sha256"] != HISTORICAL_SHA
        {
            return Err("same-device historical calibration reopened different input bytes".into());
        }
        println!(
            "{}",
            json!({"host":host,"physical":physical,"historical":historical})
        );

        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut output = BufWriter::new(options.open(out)?);
        for (name, sha, model, bytes) in checkpoints {
            let engine = Engine::new(&model, &bytes, cubin_bytes.clone())?;
            for (index, row) in fit.iter().enumerate() {
                let mut evidence = capture(&engine, &model, row, index)
                    .map_err(|error| format!("{name} row {}: {error}", row.id))?;
                evidence["checkpoint"] = json!(name);
                evidence["checkpoint_sha256"] = json!(sha);
                evidence["checkpoint_step"] = json!(model.step);
                evidence["train_sha256"] = json!(TRAIN_SHA);
                evidence["cubin_sha256"] = json!(CUBIN_SHA);
                writeln!(output, "{evidence}")?;
            }
            output.flush()?;
        }
        output.get_ref().sync_all()?;
        println!(
            "{}",
            json!({
                "verdict":"COMPLETE","rows":fit.len()*ARTIFACTS.len(),"input":"registered TRAIN only","out":out
            })
        );
        Ok(())
    }

    pub(super) fn entry() -> Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        match args.as_slice() {
            [flag] if flag == "--finite-host" => println!("{}", finite_host()?),
            [flag, cubin] if flag == "--finite-device" => {
                require_aot()?;
                println!("{}", finite_device(&pinned_bytes(cubin, CUBIN_SHA)?)?);
            }
            [cubin, history, tape, frozen, reset, full, out] => {
                collect(cubin, history, tape, [frozen, reset, full], out)?;
            }
            _ => {
                return Err(concat!(
                "usage: compact-v3-train-opening-margin --finite-host | --finite-device CUBIN | ",
                "CUBIN HISTORICAL TRAIN FROZEN RESET PRESERVED FRESH_JSONL"
            )
                .into())
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    calibrated::entry()
}
