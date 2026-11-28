#![recursion_limit = "256"]
mod calibrated {
    #![allow(dead_code)]
    include!("../../../codex-4360/runner/src/bin/device.rs");
    include!("../../../codex-4363/src/finite.rs");
    use std::path::Path;
    const CUBIN_SHA: &str = "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713";
    const HISTORICAL_SHA: &str = "dcc3019546fd1fe6e272aa71a64d694db2754b946bc82638f64b19671e47d04d";
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

    const ROSTER: &str = "700c0bd3e78bd2dd8c0644d0abaea3e265538205eec20f69b9da0ccddca1ba92";
    const ROLE: &str = "f8fe9561f79dc41ffb6d9c52423ce215f533e7482aa48a833f391611b3656630";
    const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
    const CLI: &str = "15d5856ebf19200021700f77166a4caa641896c5f54ef0dbec279e0cc70a77e2";
    const PARENT: &str = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";
    const HISTORICAL_CONTROL_CUBIN: &str =
        "d7594dab97fd490bee2cad35d595a12054f64b2cdebe1eb628ac96ac8f56328a";
    const EOS: u16 = 1;
    const CAP: usize = 96;

    fn roster(path: &str) -> Result<Vec<TrainingRow>> {
        let input = pinned_bytes(path, ROSTER)?;
        let rows: Vec<TrainingRow> = input
            .lines()
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect::<Result<_>>()?;
        if rows.len() != 4 || rows.iter().map(|r| r.row.expected.len()).sum::<usize>() != 167 {
            return Err("unexpected number of TRAIN rows or labels".into());
        }
        for (index, entry) in rows.iter().enumerate() {
            let Row { prefix, expected } = &entry.row;
            if entry.id != (index + 1) as u64
                || !matches!(entry.stratum.as_str(), "near" | "far" | "intermediate")
                || prefix.first() != Some(&0)
                || prefix[1..].contains(&0)
                || prefix.iter().filter(|&&id| id == EOS).count() != 2
                || !(2..CAP).contains(&expected.len())
                || expected.last() != Some(&EOS)
                || expected[..expected.len() - 1].iter().any(|&id| id <= EOS)
                || prefix.iter().chain(expected).any(|&id| id >= 4096)
                || expected[0] == rows[index ^ 1].row.expected.first().copied().unwrap_or(0)
            {
                return Err(format!("invalid TRAIN row {}", entry.id).into());
            }
        }
        Ok(rows)
    }

    fn valid_pin(pin: &str) -> bool {
        pin.len() == 64
            && pin
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    fn field<'a>(doc: &'a Value, key: &str) -> Result<&'a str> {
        let pin = doc[key].as_str().ok_or_else(|| format!("missing {key}"))?;
        if !valid_pin(pin) {
            return Err(format!("invalid {key}").into());
        }
        Ok(pin)
    }

    fn training_receipt(
        path: &str,
        pin: &str,
        training_cubin_pin: &str,
        finite_pin: &str,
    ) -> Result<Value> {
        let doc: Value = serde_json::from_slice(&pinned_bytes(path, pin)?)?;
        if doc["protocol"] != "uor-opening-real-single-update-v1"
            || !matches!(
                doc["result"].as_str(),
                Some("MEASURED_UPDATE" | "NO_EFFECTIVE_WEIGHT_CONTRAST")
            )
            || doc["parent_sha256"] != PARENT
            || doc["roster_sha256"] != ROSTER
            || field(&doc, "cubin_sha256")? != training_cubin_pin
            || field(&doc, "finite_receipt_sha256")? != finite_pin
            || doc["step_before"] != 3333
            || doc["step_after"] != 3334
            || doc["updates_per_arm"] != 1
            || doc["labels_per_arm"] != 167
            || doc["control"] != "physical original ABI #4361"
            || field(&doc, "historical_control_cubin_sha256")? != HISTORICAL_CONTROL_CUBIN
            || doc["baseline_forward_parity_transitions"] != 3938
            || doc["batch_lengths"] != json!([980, 1018, 933, 1011])
            || doc["opening_labels"] != json!([953, 957, 887, 978])
            || doc["margin_words"] != 256
            || doc["sham_gain_q15"] != 0
            || doc["active_gain_q15"] != 32768
        {
            return Err("receipt is outside the registered matched update".into());
        }
        let sham = field(&doc, "sham_checkpoint_sha256")?;
        let active = field(&doc, "active_checkpoint_sha256")?;
        let sham_words = &doc["sham_words"];
        let active_words = &doc["active_words"];
        let different_gradients = doc["different_gradient_words"]
            .as_u64()
            .ok_or("missing gradient contrast count")?;
        for digest in ["adam_m_le_sha256", "adam_v_le_sha256"] {
            field(sham_words, digest)?;
            field(active_words, digest)?;
        }
        let gradient_contrast =
            field(sham_words, "gradient_le_sha256")? != field(active_words, "gradient_le_sha256")?;
        let weight_contrast =
            field(sham_words, "weights_le_sha256")? != field(active_words, "weights_le_sha256")?;
        if sham == PARENT
            || active == PARENT
            || gradient_contrast != (different_gradients > 0)
            || weight_contrast != (sham != active)
            || (doc["result"] == "MEASURED_UPDATE") != (gradient_contrast && weight_contrast)
        {
            return Err("receipt has inconsistent checkpoint or gradient identities".into());
        }
        Ok(doc)
    }

    fn compare_rank(
        scores: &[i32; 64],
        gold: usize,
        admitted: impl Fn(usize) -> bool,
    ) -> Result<Value> {
        let elected = elect(scores, &admitted)?;
        if !admitted(gold) {
            return Err("inadmissible diagnostic target".into());
        }
        let rival = if elected.winner == gold {
            elected.runner_up
        } else {
            elected.winner
        };
        let rank = 1
            + (0..64)
                .filter(|&s| {
                    admitted(s)
                        && (scores[s] > scores[gold] || (scores[s] == scores[gold] && s < gold))
                })
                .count();
        let gap = i64::from(scores[gold]) - i64::from(scores[rival]);
        Ok(
            json!({"gold_slot":gold,"gold_score_q15":scores[gold],"rank":rank,"rival_slot":rival,"rival_score_q15":scores[rival],
            "gold_minus_rival_q15":gap,"strict_gold_win":gap>0,"gold_elected":elected.winner==gold,"election":elected}),
        )
    }

    fn capture_pilot(
        engine: &Engine,
        model: &ButterflyV3Model,
        row: &TrainingRow,
        partner: &TrainingRow,
    ) -> Result<Value> {
        let state = prefill(engine, model, &row.row)?;
        let input = *row.row.prefix.last().ok_or("empty causal prefix")?;
        // Blind deployment and its repeated sham run precede any diagnostic IDs.
        let blind = Row {
            prefix: vec![input],
            expected: vec![EOS],
        };
        let mut verifier = Verification {
            engine,
            model,
            eos: [EOS; 2],
            cap: CAP,
            digest: Sha256::new(),
            transitions: 0,
        };
        let rollout = verifier.arm(
            &blind,
            &blind,
            rust_prefill(model, &row.row)?,
            state.clone(),
        )?;
        let sham = arm(engine, model, state.clone(), &blind, &blind, [EOS; 2], CAP)?;
        if rollout != sham {
            return Err("blind rollout/sham mismatch".into());
        }
        let blind_first = first_step(engine, model, &state, input, [EOS; 2])?;
        let actual = output_id(model, &blind_first)?;
        if actual != rollout.emitted[0] || blind_first.state != rollout.after {
            return Err("batched versus single opening mismatch".into());
        }
        let gold = row.row.expected[0];
        let donor = partner.row.expected[0];
        let diagnostic = first_step(engine, model, &state, input, [gold, actual])?;
        check_label_independence(model.cfg, &blind_first, &diagnostic)?;
        let scores = ScoreWords::read(model.cfg, &diagnostic)?;
        let token_slot = |id: u16| usize::from(model.vocab_perm[usize::from(id)]);
        let (gold_slot, actual_slot, donor_slot) =
            (token_slot(gold), token_slot(actual), token_slot(donor));
        let (gold_cell, actual_cell, donor_cell) = (
            gold_slot / HIER_FINE,
            actual_slot / HIER_FINE,
            donor_slot / HIER_FINE,
        );
        let coarse = compare_rank(scores.coarse, gold_cell, |c| coarse_admissible(model, c))?;
        let fine = compare_rank(scores.target_fine, gold_slot % HIER_FINE, |s| {
            fine_admissible(model, gold_cell, s)
        })?;
        let elected_fine = elect(scores.elected_fine, |s| {
            fine_admissible(model, actual_cell, s)
        })?;
        if coarse["election"]["winner"] != actual_cell
            || elected_fine.winner != actual_slot % HIER_FINE
            || elected_fine.winner_score != blind_first.trace[1]
            || (gold_cell == actual_cell && scores.target_fine != scores.elected_fine)
        {
            return Err("raw score election disagrees with native output".into());
        }
        for (cell, score_words) in [
            (gold_cell, scores.target_fine),
            (actual_cell, scores.elected_fine),
        ] {
            if score_words.iter().enumerate().any(|(slot, &word)| {
                rust_fine_score(model, scores.hidden, cell * HIER_FINE + slot) != word
            }) {
                return Err("device diagnostic fine score differs from native Rust".into());
            }
        }
        let (source_stage, gap) = if gold_cell == donor_cell {
            (
                "fine",
                scores.target_fine[gold_slot % HIER_FINE]
                    - scores.target_fine[donor_slot % HIER_FINE],
            )
        } else {
            (
                "coarse",
                scores.coarse[gold_cell] - scores.coarse[donor_cell],
            )
        };
        let grade = outcome(&rollout.emitted, &row.row.expected, rollout.hit_eos);
        let contrast = grade.first_mismatch_index.map(|i| {
            match (rollout.emitted.get(i), row.row.expected.get(i)) {
                (Some(&EOS), Some(_)) => "early_eos",
                (Some(_), Some(&EOS)) => "late_or_missing_eos",
                _ => "wrong_id",
            }
        });
        Ok(
            json!({"id":row.id,"pair":(row.id-1)/2+1,"partner_id":partner.id,
              "prefix_u16le_sha256":prefix_sha(&row.row.prefix),"gold_opening_id":gold,"partner_opening_id":donor,
              "actual_opening_id":actual,"gold_cell_admitted":gold_cell==actual_cell,"elected_coarse_cell":actual_cell,
              "coarse":coarse,"fine_gold_cell":fine,"elected_fine":elected_fine,
              "raw_coarse_q15":scores.coarse.as_slice(),"raw_gold_fine_q15":scores.target_fine.as_slice(),
              "raw_elected_fine_q15":scores.elected_fine.as_slice(),"source_comparison_stage":source_stage,
              "source_gold_minus_partner_q15":gap,"source_gold_strict_win":gap>0,"gold_first_win":actual==gold,
              "emitted":rollout.emitted,"stop":rollout.stop_reason,"first_wrong_offset":grade.first_mismatch_index,
              "first_wrong_kind":contrast,"grade":grade.verdict,"exact_through_eos":grade.verdict=="exact_through_eos",
              "gold_length_including_eos":row.row.expected.len(),"blind_sham_match":true,"diagnostic_transition_match":true,
              "native_rust_transition_count":verifier.transitions,"native_rust_transition_sha256":hex::encode(verifier.digest.finalize()),
              "blind_opening_signature":signature(model.cfg,&blind_first)?
            }),
        )
    }

    fn prefix_sha(prefix: &[u16]) -> String {
        let mut hash = Sha256::new();
        for id in prefix {
            hash.update(id.to_le_bytes());
        }
        hex::encode(hash.finalize())
    }

    fn write_new(path: &Path, data: &[u8]) -> Result<()> {
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        output.write_all(data)?;
        output.sync_all()?;
        Ok(())
    }

    fn checked_models(
        receipt: &Value,
        paths: [&str; 3],
    ) -> Result<Vec<(&'static str, String, ButterflyV3Model, Vec<u8>)>> {
        [
            ("frozen", PARENT, 3333),
            ("sham", field(receipt, "sham_checkpoint_sha256")?, 3334),
            ("active", field(receipt, "active_checkpoint_sha256")?, 3334),
        ]
        .into_iter()
        .zip(paths)
        .map(|((name, sha, step), path)| {
            let bytes = pinned_bytes(path, sha)?;
            let model = ButterflyV3Model::from_bytes(&bytes)?;
            if model.step != step
                || model.cfg.vocab != 4096
                || model.cfg.dim != 512
                || model.cfg.layers != 6
                || model.to_bytes() != bytes
            {
                return Err(format!("wrong native artifact identity for {name}").into());
            }
            Ok((name, sha.to_owned(), model, bytes))
        })
        .collect()
    }

    fn pair_summary(rows: &[Value]) -> Vec<Value> {
        rows.chunks_exact(2).map(|r| {
            let wins=|v:&str|r.iter().filter(|x|x[v]==true).count();
            json!({"pair":r[0]["pair"],"arm":r[0]["arm"],"source_comparison_stage":r[0]["source_comparison_stage"],
              "ordinary_first_wins":wins("gold_first_win"),"both_members_first_win":wins("gold_first_win")==2,
              "strict_source_score_wins":wins("source_gold_strict_win"),"both_members_strict_source_win":wins("source_gold_strict_win")==2,
              "source_ties":r.iter().filter(|x|x["source_gold_minus_partner_q15"]==0).count(),
              "exact_through_eos":wins("exact_through_eos")})
        }).collect()
    }

    fn fresh_dir(path: &Path) -> Result<()> {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().mode(0o700).create(path)?;
        Ok(())
    }

    fn require_aot() -> Result<()> {
        if std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() != Ok("1") {
            return Err("CUDA_DISABLE_PTX_JIT=1 required".into());
        }
        Ok(())
    }

    fn physical_calibration(cubin: &str, history: &str, frozen: &str) -> Result<Value> {
        require_aot()?;
        let binary = pinned_bytes(cubin, CUBIN_SHA)?;
        pinned_bytes(history, HISTORICAL_SHA)?;
        pinned_bytes(frozen, PARENT)?;
        let host = finite_host()?;
        let physical = finite_device(&binary)?;
        let historical = historical(cubin, frozen, history)?;
        if historical["artifact_sha256"] != PARENT
            || historical["cubin_sha256"] != CUBIN_SHA
            || historical["fixture_sha256"] != HISTORICAL_SHA
        {
            return Err("historical fixture identity mismatch".into());
        }
        Ok(json!({"host":host,"physical":physical,"historical":historical}))
    }

    fn collect(args: &[String]) -> Result<()> {
        let [cubin, history, tape, receipt, receipt_pin, training_cubin_pin, finite_pin, frozen, sham, active, out] =
            args
        else {
            return Err("gpu SERVING_CUBIN HISTORICAL TRAIN RECEIPT RECEIPT_SHA TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA FROZEN SHAM ACTIVE NEW_DIR".into());
        };
        require_aot()?;
        let binary = pinned_bytes(cubin, CUBIN_SHA)?;
        pinned_bytes(history, HISTORICAL_SHA)?;
        let fit = roster(tape)?;
        let receipt = training_receipt(receipt, receipt_pin, training_cubin_pin, finite_pin)?;
        let models = checked_models(&receipt, [frozen, sham, active])?;
        if models[1..].iter().any(|(_, _, m, _)| {
            m.cfg != models[0].2.cfg
                || m.vocab_perm != models[0].2.vocab_perm
                || m.vocab_inv != models[0].2.vocab_inv
        }) {
            return Err("native checkpoint configuration or permutation mismatch".into());
        }
        let out = Path::new(out);
        fresh_dir(out)?;
        let calibration = physical_calibration(cubin, history, frozen)?;
        write_new(
            &out.join("calibration.json"),
            &serde_json::to_vec_pretty(&calibration)?,
        )?;
        let mut rows = Vec::with_capacity(12);
        for (name, sha, model, bytes) in &models {
            let engine = Engine::new(model, bytes, binary.clone())?;
            for (i, row) in fit.iter().enumerate() {
                let mut evidence = capture_pilot(&engine, model, row, &fit[i ^ 1])
                    .map_err(|e| format!("{name} row {}: {e}", row.id))?;
                evidence["arm"] = json!(name);
                evidence["checkpoint_sha256"] = json!(sha);
                evidence["checkpoint_step"] = json!(model.step);
                rows.push(evidence);
            }
        }
        let mut raw = Vec::new();
        for row in &rows {
            writeln!(raw, "{row}")?;
        }
        write_new(&out.join("rows.jsonl"), &raw)?;
        let report = json!({"schema":"uor-physical-opposed-opening-readout/1","verdict":"COMPLETE","cap":CAP,
          "population":"four registered TRAIN conversations in two pairs",
          "training_receipt_sha256":receipt_pin,"training_cubin_sha256":training_cubin_pin,"finite_receipt_sha256":finite_pin,
          "serving_cubin_sha256":CUBIN_SHA,"historical_fixture_sha256":HISTORICAL_SHA,"train_sha256":ROSTER,
          "rows_sha256":sha256(&raw),"rows":rows.len(),"pairs":pair_summary(&rows)});
        write_new(
            &out.join("report.json"),
            &serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{report}");
        Ok(())
    }

    fn cold(args: &[String]) -> Result<()> {
        let [cli, tokenizer, roles, tape, receipt, receipt_pin, training_cubin_pin, finite_pin, frozen, sham, active, gpu_rows, gpu_pin, out] =
            args
        else {
            return Err("cold CLI TOKENIZER ROLES TRAIN RECEIPT RECEIPT_SHA TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA FROZEN SHAM ACTIVE GPU_ROWS GPU_ROWS_SHA NEW_DIR".into());
        };
        pinned_bytes(cli, CLI)?;
        let tok_bytes = pinned_bytes(tokenizer, TOKENIZER)?;
        let tok = uor_r4_tokenizer::ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
            .ok_or("invalid bound tokenizer")?;
        let fit = roster(tape)?;
        let source: Vec<Value> = pinned_bytes(roles, ROLE)?
            .lines()
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect::<Result<_>>()?;
        let receipt = training_receipt(receipt, receipt_pin, training_cubin_pin, finite_pin)?;
        let gpu: Vec<Value> = pinned_bytes(gpu_rows, gpu_pin)?
            .lines()
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect::<Result<_>>()?;
        if gpu.len() != 12
            || source.len() != 4
            || source[0]["source_graph_pair"] != source[1]["source_graph_pair"]
            || source[2]["source_graph_pair"] != source[3]["source_graph_pair"]
            || source[0]["source_graph_pair"] == source[2]["source_graph_pair"]
        {
            return Err("wrong GPU row or role source pair inventory".into());
        }
        let out = Path::new(out);
        fresh_dir(out)?;
        let mut summaries = Vec::new();
        for (i, (arm, model_sha, model_path)) in [
            ("frozen", PARENT, frozen),
            ("sham", field(&receipt, "sham_checkpoint_sha256")?, sham),
            (
                "active",
                field(&receipt, "active_checkpoint_sha256")?,
                active,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            pinned_bytes(model_path, model_sha)?;
            let dir = out.join(arm);
            fresh_dir(&dir)?;
            let manifest = json!({"schema":"uor-r4.butterfly-v3-chat/1","checkpoint_sha256":model_sha,
              "tokenizer_sha256":TOKENIZER,"dialogue_protocol":2,"vocab_size":4096,"bos_id":0,"eos_id":EOS});
            let manifest_path = dir.join("manifest.json");
            let manifest_bytes = serde_json::to_vec(&manifest)?;
            let manifest_sha = sha256(&manifest_bytes);
            write_new(&manifest_path, &manifest_bytes)?;
            for (j, entry) in fit.iter().enumerate() {
                let raw = &gpu[4 * i + j];
                let prefix_pin = prefix_sha(&entry.row.prefix);
                let roles = &source[j]["role_messages"];
                if raw["arm"] != arm
                    || raw["checkpoint_sha256"] != model_sha
                    || raw["id"] != entry.id
                    || source[j]["id"] != entry.id
                    || raw["prefix_u16le_sha256"] != prefix_pin
                    || roles.as_array().map(Vec::len) != Some(5)
                {
                    return Err(format!(
                        "invalid physical or role identity for {arm} row {}",
                        entry.id
                    )
                    .into());
                }
                let messages = dir.join(format!("{}.messages.json", entry.id));
                let trace = dir.join(format!("{}.trace.json", entry.id));
                write_new(&messages, &serde_json::to_vec(roles)?)?;
                let command = std::process::Command::new(cli)
                    .arg("--butterfly-v3")
                    .arg(&manifest_path)
                    .args([&manifest_sha, model_path, tokenizer])
                    .arg(&messages)
                    .arg(CAP.to_string())
                    .arg("--trace")
                    .arg(&trace)
                    .output()?;
                write_new(
                    &dir.join(format!("{}.response.bytes", entry.id)),
                    &command.stdout,
                )?;
                write_new(&dir.join(format!("{}.stderr", entry.id)), &command.stderr)?;
                if !command.status.success() {
                    return Err(format!("ordinary CLI failed for {arm} row {}", entry.id).into());
                }
                let trace: Value = serde_json::from_slice(&fs::read(trace)?)?;
                let mut emitted: Vec<u32> = serde_json::from_value(trace["emitted"].clone())?;
                if trace["stop"] == "eos" {
                    if emitted.pop() != Some(u32::from(EOS)) {
                        return Err("invalid ordinary EOS".into());
                    }
                }
                if trace["prefix"] != json!(entry.row.prefix)
                    || trace["prefix_u16le_sha256"] != prefix_pin
                    || trace["stop"] != raw["stop"]
                    || trace["emitted"] != raw["emitted"]
                    || command.stdout != tok.decode_bytes(&emitted)
                {
                    return Err(
                        format!("ordinary cold parity mismatch {arm} row {}", entry.id).into(),
                    );
                }
                summaries.push(json!({"arm":arm,"id":entry.id,"checkpoint_sha256":model_sha,"manifest_sha256":manifest_sha,
                  "prefix_u16le_sha256":prefix_pin,"emitted_count":emitted.len(),"stop":trace["stop"],
                  "response_bytes_sha256":sha256(&command.stdout),"prefix_ids_equal":true,"raw_ids_equal":true,"stop_equal":true,"stdout_equal":true}));
            }
        }
        let report = json!({"schema":"uor-cold-opposed-opening-parity/1","verdict":"COMPLETE","checked":summaries.len(),
          "cap":CAP,"gpu_rows_sha256":gpu_pin,"training_receipt_sha256":receipt_pin,
          "cli_sha256":CLI,"tokenizer_sha256":TOKENIZER,"role_source_sha256":ROLE,"train_sha256":ROSTER,"rows":summaries});
        write_new(
            &out.join("report.json"),
            &serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{report}");
        Ok(())
    }

    #[cfg(test)]
    include!("tests.rs");

    pub(super) fn entry() -> Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        match args.as_slice() {
            [cmd] if cmd == "finite-host" => {
                println!("{}", finite_host()?);
                Ok(())
            }
            [cmd, tape, roles, tokenizer, cli] if cmd == "check-inputs" => {
                let rows = roster(tape)?;
                pinned_bytes(roles, ROLE)?;
                pinned_bytes(tokenizer, TOKENIZER)?;
                pinned_bytes(cli, CLI)?;
                println!("{}", json!({"verified":true,"rows":rows.len(),"cap":CAP}));
                Ok(())
            }
            [cmd, receipt, receipt_pin, cubin_pin, finite_pin] if cmd == "check-receipt" => {
                training_receipt(receipt, receipt_pin, cubin_pin, finite_pin)?;
                println!("{}", json!({"verified":true,"receipt_sha256":receipt_pin}));
                Ok(())
            }
            [cmd, cubin, history, frozen] if cmd == "gpu-calibration" => {
                println!("{}", physical_calibration(cubin, history, frozen)?);
                Ok(())
            }
            [cmd, args @ ..] if cmd == "gpu" => collect(args),
            [cmd, args @ ..] if cmd == "cold" => cold(args),
            _ => Err(
                "expected finite-host, check-inputs, check-receipt, gpu-calibration, gpu or cold"
                    .into(),
            ),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    calibrated::entry()
}
