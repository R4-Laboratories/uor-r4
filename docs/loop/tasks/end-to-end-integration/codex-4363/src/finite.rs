// Plant integer scores in normal UORBFY03 head rows. The only consumed embedding
// lies outside the edited cells, so all four fixtures share one native transition.
const SYNTHETIC_INPUT: u16 = 23;

#[derive(Clone, Copy)]
struct Fixture {
    name: &'static str,
    masked: bool,
    planted: [i32; 3],
    winner: usize,
    runner_up: usize,
    margin: i32,
    maxima: &'static [usize],
}

const FIXTURES: [Fixture; 4] = [
    Fixture {
        name: "exact_tie",
        masked: false,
        planted: [0, 0, -1],
        winner: 0,
        runner_up: 1,
        margin: 0,
        maxima: &[0, 1],
    },
    Fixture {
        name: "slot_zero_ahead_by_one",
        masked: false,
        planted: [1, 0, -1],
        winner: 0,
        runner_up: 1,
        margin: 1,
        maxima: &[0],
    },
    Fixture {
        name: "slot_zero_behind_by_one",
        masked: false,
        planted: [0, 1, -1],
        winner: 1,
        runner_up: 0,
        margin: 1,
        maxima: &[1],
    },
    Fixture {
        name: "strict_winner_masked",
        masked: true,
        planted: [2, 1, 0],
        winner: 1,
        runner_up: 2,
        margin: 1,
        maxima: &[1],
    },
];

fn plant_score(row: &mut [i16], hidden: &[i32], desired: i32) -> Result<()> {
    let (index, &coordinate) = hidden
        .iter()
        .enumerate()
        .max_by_key(|&(_, value)| value.unsigned_abs())
        .ok_or("empty fixture hidden state")?;
    if coordinate == 0 {
        return Err("fixture hidden state is zero".into());
    }

    let shift = ButterflyV3Model::head_dot_shift(hidden.len());
    let unit = 1_i64 << shift;
    let coordinate = i64::from(coordinate);
    let center = i64::from(desired) * unit / coordinate;
    let witness = [center, center - 1, center + 1]
        .into_iter()
        .filter_map(|word| i16::try_from(word).ok())
        .find(|&weight| (coordinate * i64::from(weight) + unit / 2) >> shift == i64::from(desired))
        .ok_or("no i16 witness for the requested exact score")?;
    row.fill(0);
    row[index] = witness;
    Ok(())
}

fn check_fixture_result(fixture: Fixture, election: &Election, head: &str) -> Result<()> {
    if election.winner != fixture.winner
        || election.runner_up != fixture.runner_up
        || election.margin != fixture.margin
        || election.equal_maxima != fixture.maxima
        || election.winner_score != fixture.planted[fixture.winner]
        || election.runner_up_score != fixture.planted[fixture.runner_up]
    {
        return Err(format!("{} {head} election mismatch: {election:?}", fixture.name).into());
    }
    Ok(())
}

fn finite_device(cubin: &[u8]) -> Result<Value> {
    let mut rows = Vec::with_capacity(FIXTURES.len());
    let mut gpu = None;
    for fixture in FIXTURES {
        let mut model = ButterflyV3Model::new(ButterflyV3Config {
            vocab: 192,
            dim: 16,
            layers: 2,
            ffn_branches: 2,
            pkm_sub_keys: 4,
            pkm_key_dim: 4,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 2 },
            seed: 734,
        })?;
        let mut perm = model.vocab_perm.clone();
        if fixture.masked {
            // Tokens 0..191 move to slots 64..255. Cell 0 is empty and BOS
            // occupies fine slot 0 of cell 1. Both strongest scores are masked.
            perm[..256].rotate_left(HIER_FINE);
        } else {
            // Make both fine slots 0 and 1 admissible in cells 0 and 1.
            perm.swap(0, 190);
        }
        // Keep the consumed input embedding outside the planted fine-head cells.
        perm.swap(usize::from(SYNTHETIC_INPUT), 191);
        model.set_vocab_perm(perm)?;
        let dim = model.cfg.dim;
        let origin = usize::from(SYNTHETIC_INPUT) * dim;
        let destination = usize::from(model.vocab_perm[usize::from(SYNTHETIC_INPUT)]) * dim;
        model
            .embeddings
            .copy_within(origin..origin + dim, destination);

        let start = model.new_session();
        let hidden = model.step(SYNTHETIC_INPUT, &mut start.clone())?.hidden_q15;
        let mut expected = [-1; 64];
        expected[..3].copy_from_slice(&fixture.planted);
        for (row, &score) in model.coarse_centroids.chunks_exact_mut(dim).zip(&expected) {
            plant_score(row, &hidden, score)?;
        }
        for cell in 0..=1 {
            let offset = cell * HIER_FINE * dim;
            for (row, &score) in model.embeddings[offset..offset + HIER_FINE * dim]
                .chunks_exact_mut(dim)
                .zip(&expected)
            {
                plant_score(row, &hidden, score)?;
            }
        }

        let artifact = model.to_bytes();
        let engine = Engine::new(&model, &artifact, cubin.to_vec())?;
        let free = first_step(
            &engine,
            &model,
            &State::from(start.clone()),
            SYNTHETIC_INPUT,
            [1; 2],
        )?;
        let actual_id = output_id(&model, &free)?;
        let id_for_slot = model.vocab_inv[fixture.winner * HIER_FINE + fixture.winner];
        if actual_id != id_for_slot || !admissible(&model, actual_id) {
            return Err(format!(
                "{} device chose {actual_id}, expected {id_for_slot}",
                fixture.name
            )
            .into());
        }

        let other = model.vocab_inv[fixture.winner * HIER_FINE + fixture.runner_up];
        let labeled = first_step(
            &engine,
            &model,
            &State::from(start.clone()),
            SYNTHETIC_INPUT,
            [other, actual_id],
        )?;
        check_label_independence(model.cfg, &free, &labeled)?;
        let sham = first_step(
            &engine,
            &model,
            &State::from(start.clone()),
            SYNTHETIC_INPUT,
            [1; 2],
        )?;
        check_label_independence(model.cfg, &free, &sham)?;
        if free.trace != sham.trace || free.mass != sham.mass {
            return Err(format!("{} label-blind sham diverged", fixture.name).into());
        }
        let scores = ScoreWords::read(model.cfg, &labeled)?;
        if scores.hidden != hidden
            || *scores.coarse != expected
            || *scores.target_fine != expected
            || *scores.elected_fine != expected
            || labeled.trace[1] != fixture.planted[fixture.winner]
        {
            return Err(format!("{} physical raw score or hidden mismatch", fixture.name).into());
        }
        let mut rust = start;
        let rust_step = model.step(SYNTHETIC_INPUT, &mut rust)?;
        let rust_pkm: Vec<i32> = rust_step
            .pkm_slots
            .iter()
            .flatten()
            .map(|&id| id as i32)
            .collect();
        if rust_step.greedy_token != actual_id
            || usize::from(rust_step.greedy_coarse) != fixture.winner
            || usize::from(rust_step.greedy_fine) != fixture.winner
            || rust_step.coarse_scores != expected
            || rust_pkm != scores.pkm
            || State::from(rust) != labeled.state
        {
            return Err(
                format!("{} physical/Rust native transition mismatch", fixture.name).into(),
            );
        }

        let coarse = elect(&scores.coarse, |cell| coarse_admissible(&model, cell))?;
        let fine = elect(&scores.elected_fine, |slot| {
            fine_admissible(&model, fixture.winner, slot)
        })?;
        check_fixture_result(fixture, &coarse, "coarse")?;
        check_fixture_result(fixture, &fine, "fine")?;
        rows.push(json!({
            "fixture":fixture.name, "artifact_sha256":sha256(&artifact),
            "coarse":coarse, "fine":fine, "actual_original_id":actual_id,
            "slot_zero_admitted_coarse":coarse_admissible(&model, 0),
            "slot_zero_admitted_fine":fine_admissible(&model, fixture.winner, 0),
            "physical_coarse_raw_q15":scores.coarse.as_slice(),
            "physical_fine_raw_q15":scores.elected_fine.as_slice(),
            "blind_signature":signature(model.cfg, &free)?,
            "rich_signature":signature(model.cfg, &labeled)?,
            "sham_signature":signature(model.cfg, &sham)?,
            "reference_free_transition_match":true, "rust_native_transition_match":true
        }));
        gpu = Some(engine.gpu_name);
    }
    Ok(json!({
        "finite_device":"PASS", "gpu":gpu, "gpu_cc":[8,9], "cubin_sha256":sha256(cubin),
        "physical_score_fixtures":rows,
    }))
}
