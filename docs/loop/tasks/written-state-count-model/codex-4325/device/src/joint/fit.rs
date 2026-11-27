use super::*;

const AUDIT_HEADER: usize = 6;
const AUDIT_WORDS: usize = AUDIT_HEADER + COORDINATE_CAP;

#[derive(Serialize)]
struct Slot<'a> {
    epoch: u32,
    train_index: u32,
    pair_id: &'a str,
    side: u32,
    first_stage: u32,
    first_copy: bool,
    joint_stage: u32,
    joint_copy: bool,
}

impl Slot<'_> {
    fn device_words(&self, arm: usize) -> [u32; 4] {
        let (stage, copy) = if arm == 0 {
            (self.first_stage, self.first_copy)
        } else {
            (self.joint_stage, self.joint_copy)
        };
        [self.train_index, stage, u32::from(copy), self.epoch]
    }
}

fn slots(rows: &[Input]) -> Result<Vec<Slot<'_>>> {
    let mut pools = [Vec::new(), Vec::new()];
    for (index, row) in rows.iter().enumerate() {
        if row.core.boundaries()?.2.len() < 3 {
            return Err("a TRAIN directive has no ordered distant root pair".into());
        }
        match row.core.requested() {
            1 => pools[0].push(index),
            2 => pools[1].push(index),
            _ => {}
        }
    }
    if rows.len() % 2 != 0 || pools.iter().any(Vec::is_empty) {
        return Err(
            "matched TRAIN schedule requires an even tape and both authenticated B targets".into(),
        );
    }
    let count = u32::try_from(rows.len() * EPOCHS * 2)?;
    let entry = |index: usize, epoch: usize, joint_stage: u32| {
        let row = &rows[index].core;
        Slot {
            epoch: epoch as u32,
            train_index: index as u32,
            pair_id: &row.pair_id,
            side: row.side,
            first_stage: 0,
            first_copy: row.requested() > 0,
            joint_stage,
            joint_copy: row.requested() > joint_stage,
        }
    };
    let mut tape = Vec::with_capacity(count as usize);
    for epoch in 0..EPOCHS {
        tape.extend((0..rows.len()).map(|row| entry(row, epoch, 0)));
        for slot in 0..rows.len() {
            let pool = &pools[slot % 2];
            let offset = (epoch * (rows.len() / 2) + slot / 2) % pool.len();
            tape.push(entry(pool[offset], epoch, 1));
        }
    }
    Ok(tape)
}

pub(super) fn schedule(panel: &str, output: &Path) -> Result<()> {
    let (input_sha256, rows) = run::load_panel(panel)?;
    let scheduled = slots(&rows)?;
    write_new(output, &scheduled)?;
    println!(
        "{}",
        serde_json::json!({
            "rows": rows.len(), "slots": scheduled.len(), "input_sha256": input_sha256,
            "schedule_sha256": file_hash(output)?,
        })
    );
    Ok(())
}

#[derive(Serialize)]
struct Authentication<'a> {
    train_index: usize,
    pair_id: &'a str,
    side: u32,
    supplied_opening: Step,
    supplied_period: Step,
    supervised_b: Option<TraceStep>,
    post_b_prefix: Option<Vec<u32>>,
}

fn authenticate<'a>(
    replay: &Replayer,
    rows: &[&'a PanelRow],
    scores: &CudaSlice<i32>,
) -> Result<Vec<Authentication<'a>>> {
    let openings = replay.released_openings(rows, scores)?;
    let eligible: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.requested() > 0)
        .collect();
    let inputs = eligible
        .iter()
        .map(|&(index, _)| &openings[index].prefix)
        .collect::<Vec<_>>();
    let writers = replay.device.events(&inputs, 1)?;
    let host_scores = replay.device.stream.clone_dtoh(scores)?;
    let mut records = vec![0; eligible.len() * COMMIT_STRIDE];
    let mut decisions = Vec::with_capacity(eligible.len());
    for ((&(index, row), writer), record) in eligible
        .iter()
        .zip(&writers)
        .zip(records.chunks_exact_mut(COMMIT_STRIDE))
    {
        native_only(writer)?;
        let address = Address {
            pos: row.positions[1],
            id: row.ids[1],
        };
        let decision = Decision {
            stage: 0,
            row: u32::try_from(index)?,
            fence: u32::try_from(row.boundaries()?.0)?,
            original: address,
            eligible: Some(Gate::First),
            intervention: Some(Gate::First),
            requested_count: 1,
            learned_score: Some(host_scores[index * STAGES]),
        };
        if !decision.prepare(
            record,
            writer.native_proposal,
            &openings[index].prefix.prefix,
            address,
        ) {
            return Err(format!("TRAIN original B address refused on row {index}").into());
        }
        decisions.push(decision);
    }
    let commits = replay
        .kernel
        .select(&replay.device.stream, &records, scores)?;
    let mut proof = BTreeMap::new();
    for ((((index, _), decision), writer), commit) in eligible
        .into_iter()
        .zip(decisions)
        .zip(writers)
        .zip(commits)
    {
        let address = decision.original;
        if commit
            != (Commit {
                id: address.id,
                source: 1,
                next: 1,
                refusal: 0,
            })
        {
            return Err(
                format!("GPU did not establish an authenticated B on TRAIN row {index}").into(),
            );
        }
        let next = openings[index].prefix.self_feed(commit.id);
        let trace = decision.trace(writer, commit, address, true)?;
        proof.insert(index, (trace, next.prefix));
    }
    Ok(rows
        .iter()
        .zip(openings)
        .enumerate()
        .map(|(train_index, (row, opening))| {
            let (supervised_b, post_b_prefix) = match proof.remove(&train_index) {
                Some((trace, prefix)) => (Some(trace), Some(prefix)),
                None => (None, None),
            };
            Authentication {
                train_index,
                pair_id: &row.pair_id,
                side: row.side,
                supplied_opening: opening.opening,
                supplied_period: opening.period,
                supervised_b,
                post_b_prefix,
            }
        })
        .collect())
}

pub(super) fn device_fit(
    stream: &Arc<CudaStream>,
    kernel: &Kernel,
    function: &CudaFunction,
    instructions: &Instructions,
    slots: &[[u32; 4]],
    initial: &[i16],
) -> Result<(Vec<i16>, Vec<i32>)> {
    let count = u32::try_from(slots.len())?;
    if count == 0
        || initial.len() != RELEASE_WORDS
        || slots.iter().any(|slot| {
            slot[0] >= instructions.count || slot[1] >= 2 || slot[2] > 1 || slot[3] >= EPOCHS as u32
        })
    {
        return Err("invalid GPU fit slot bounds".into());
    }
    let slots = stream.clone_htod(slots.as_flattened())?;
    let mut state = stream.clone_htod(initial)?;
    let mut audit = stream.alloc_zeros::<i32>(count as usize * AUDIT_WORDS)?;
    let mut call = stream.launch_builder(function);
    call.arg(&kernel.roots)
        .arg(&instructions.rows)
        .arg(&instructions.count)
        .arg(&slots)
        .arg(&count)
        .arg(&mut state)
        .arg(&mut audit);
    // The validated slot indices and pinned directive stream bound all reads.
    // One worker owns the only mutable copy of the residual table.
    unsafe { call.launch(grid(1))? };
    let audit = stream.clone_dtoh(&audit)?;
    if let Some((slot, record)) = audit
        .chunks_exact(AUDIT_WORDS)
        .enumerate()
        .find(|(_, row)| row[0] != 0)
    {
        return Err(format!("GPU fit refused slot {slot} with status {}", record[0]).into());
    }
    Ok((stream.clone_dtoh(&state)?, audit))
}

#[derive(Default, Serialize)]
struct StageAudit {
    slots: usize,
    stop_targets: u32,
    copy_targets: u32,
    link_visits: u64,
    unique_coordinate_visits: u64,
    repeated_coordinate_visits: u64,
    capped_coordinate_opportunities: u64,
    violated_margins: u64,
    actual_coordinate_writes: u64,
    repeated_coordinate_writes: u64,
}

#[derive(Serialize)]
struct SlotAudit<'a> {
    slot: usize,
    before: i32,
    after: i32,
    link_visits: u32,
    unique_coordinate_visits: u32,
    changed_coordinates: &'a [i32],
}

fn audit_fit<'a>(
    initial: &[i16],
    device: &[i16],
    slots: &[[u32; 4]],
    audit: &'a [i32],
) -> Result<([StageAudit; 2], Vec<SlotAudit<'a>>)> {
    let mut verified = initial.to_vec();
    let mut written = [false; WORDS];
    let mut stages = [StageAudit::default(), StageAudit::default()];
    let mut trace = Vec::with_capacity(slots.len());
    for (slot, (&[_, stage, copy, _], record)) in slots
        .iter()
        .zip(audit.chunks_exact(AUDIT_WORDS))
        .enumerate()
    {
        let [_, before, after, visits, support, changed, coordinates @ ..] = record else {
            return Err("malformed GPU fit audit".into());
        };
        let sign = if copy == 1 { 1 } else { -1 };
        let opportunity = (*support).min(COORDINATE_CAP as i32);
        let violated = before * sign < MARGIN;
        if *support <= 0
            || *support > WORDS as i32
            || visits < support
            || *changed != if violated { opportunity } else { 0 }
            || *after != before + sign * changed
        {
            return Err(format!("inconsistent GPU fit audit at slot {slot}").into());
        }
        let changed_coordinates = &coordinates[..*changed as usize];
        let summary = &mut stages[stage as usize];
        summary.slots += 1;
        summary.stop_targets += u32::from(copy == 0);
        summary.copy_targets += copy;
        summary.link_visits += *visits as u64;
        summary.unique_coordinate_visits += *support as u64;
        summary.repeated_coordinate_visits += (visits - support) as u64;
        summary.capped_coordinate_opportunities += opportunity as u64;
        summary.violated_margins += u64::from(violated);
        summary.actual_coordinate_writes += *changed as u64;
        for (visit, &coordinate) in changed_coordinates.iter().enumerate() {
            if !(0..WORDS as i32).contains(&coordinate)
                || changed_coordinates[..visit].contains(&coordinate)
            {
                return Err(
                    format!("invalid repeated or out-of-range coordinate at slot {slot}").into(),
                );
            }
            let coordinate = coordinate as usize;
            summary.repeated_coordinate_writes += u64::from(written[coordinate]);
            written[coordinate] = true;
            let word = &mut verified[STATE_WORDS + coordinate];
            *word = word
                .checked_add(sign as i16)
                .ok_or("overflow replaying GPU audit")?;
        }
        trace.push(SlotAudit {
            slot,
            before: *before,
            after: *after,
            link_visits: *visits as u32,
            unique_coordinate_visits: *support as u32,
            changed_coordinates,
        });
    }
    if trace.len() != slots.len() || verified != device {
        return Err("GPU write audit did not reconstruct the serialized integer words".into());
    }
    Ok((stages, trace))
}

pub(super) fn fit(
    files: &Files<'_>,
    image_spec: [&str; 2],
    panel: &str,
    link: &str,
    output: &Path,
) -> Result<()> {
    let started = Instant::now();
    let (input_sha256, rows) = run::load_panel(panel)?;
    let tape = slots(&rows)?;
    let mut initial = read_state(link, run::CANDIDATE_STATE, STATE_WORDS)?;
    initial.resize(RELEASE_WORDS, 0);
    let initial_bytes: Vec<_> = initial.iter().flat_map(|word| word.to_le_bytes()).collect();
    let initial_sha256 = format!("{:x}", Sha256::digest(&initial_bytes));
    let replay =
        Replayer::open_width(files, image_spec[0], image_spec[1], &initial, RELEASE_WORDS)?;
    let stream = &replay.device.stream;
    let core = rows.iter().map(|row| &row.core).collect::<Vec<_>>();
    let instructions = Instructions::from_panel(stream, &core)?;
    let initial_scores = replay.kernel.scores(stream, &instructions, &replay.state)?;
    let authenticated = authenticate(&replay, &core, &initial_scores)?;
    for slot in &tape {
        if slot.joint_stage == 1
            && authenticated[slot.train_index as usize]
                .supervised_b
                .is_none()
        {
            return Err("matched schedule contains an unauthenticated post-B slot".into());
        }
    }
    fs::create_dir(output)?;
    let schedule_path = output.join("schedule.jsonl");
    let authenticated_path = output.join("authenticated.jsonl");
    write_new(&schedule_path, &tape)?;
    write_new(&authenticated_path, &authenticated)?;
    let trainer = image(
        stream.context(),
        image_spec[0],
        Some(image_spec[1]),
        "joint_fit",
    )?;
    let mut outputs = BTreeMap::new();
    let mut support = None;
    for (arm, label) in ARMS.into_iter().enumerate() {
        let arm_tape: Vec<_> = tape.iter().map(|slot| slot.device_words(arm)).collect();
        let train_started = Instant::now();
        let (state, audit) = device_fit(
            stream,
            &replay.kernel,
            &trainer,
            &instructions,
            &arm_tape,
            &initial,
        )?;
        let train_elapsed_ms = train_started.elapsed().as_millis();
        let repeated = device_fit(
            stream,
            &replay.kernel,
            &trainer,
            &instructions,
            &arm_tape,
            &initial,
        )?;
        if repeated.0 != state || repeated.1 != audit {
            return Err(
                format!("{label} GPU refit from original bytes changed weights or audit").into(),
            );
        }
        let (stages, trace) = audit_fit(&initial, &state, &arm_tape, &audit)?;
        let observed_support: Vec<_> = trace
            .iter()
            .map(|row| (row.link_visits, row.unique_coordinate_visits))
            .collect();
        if support
            .as_ref()
            .is_some_and(|first| first != &observed_support)
        {
            return Err("matched slots used different coordinate opportunity budgets".into());
        }
        support = Some(observed_support);
        let state_bytes: Vec<_> = state.iter().flat_map(|word| word.to_le_bytes()).collect();
        let state_path = output.join(format!("{label}.i16le"));
        fs::write(&state_path, &state_bytes)?;
        let restored = read_state(
            state_path.to_str().ok_or("non-UTF-8 state path")?,
            &file_hash(&state_path)?,
            RELEASE_WORDS,
        )?;
        if restored != state {
            return Err("serialized integer state did not round-trip".into());
        }
        let state_device = stream.clone_htod(&state)?;
        let restored_device = stream.clone_htod(&restored)?;
        let saved_scores =
            stream.clone_dtoh(&replay.kernel.scores(stream, &instructions, &state_device)?)?;
        let restored_scores = stream.clone_dtoh(&replay.kernel.scores(
            stream,
            &instructions,
            &restored_device,
        )?)?;
        if saved_scores != restored_scores {
            return Err("restored integer state changed GPU scores".into());
        }
        let scores_path = output.join(format!("{label}-train-scores.jsonl"));
        let audit_path = output.join(format!("{label}-slots.jsonl"));
        write_new(&scores_path, saved_scores.chunks_exact(STAGES))?;
        write_new(&audit_path, trace)?;
        outputs.insert(label, serde_json::json!({
            "weights_sha256": file_hash(&state_path)?,
            "slot_audit_sha256": file_hash(&audit_path)?,
            "train_scores_sha256": file_hash(&scores_path)?,
            "train_elapsed_ms": train_elapsed_ms,
            "identical_refit": true,
            "integer_round_trip": true,
            "changed_residual_words": state[STATE_WORDS..].iter().filter(|&&word| word != 0).count(),
            "changed_residual_bytes": state_bytes.iter().zip(&initial_bytes).filter(|(left, right)| left != right).count(),
            "stages": stages,
        }));
    }
    let receipt = serde_json::json!({
        "pass": true,
        "rows": rows.len(),
        "epochs": EPOCHS,
        "margin": MARGIN,
        "coordinate_cap": COORDINATE_CAP,
        "slots_per_arm": tape.len(),
        "post_b_authenticated_rows": authenticated.iter().filter(|row| row.supervised_b.is_some()).count(),
        "state_words": RELEASE_WORDS,
        "state_bytes": RELEASE_WORDS * 2,
        "input_sha256": input_sha256,
        "linked_weights_sha256": run::CANDIDATE_STATE,
        "initial_weights_sha256": initial_sha256,
        "image_sha256": image_spec[1],
        "roots_sha256": BASE[7].1,
        "schedule_sha256": file_hash(&schedule_path)?,
        "authenticated_sha256": file_hash(&authenticated_path)?,
        "outputs": outputs,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    fs::write(
        output.join("fit.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
