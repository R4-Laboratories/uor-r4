use super::*;

pub(super) const AUDIT_HEADER: usize = 11;
pub(super) const AUDIT_WORDS: usize = AUDIT_HEADER + COORDINATE_CAP;

pub(super) fn device_fit(
    stream: &Arc<CudaStream>,
    kernel: &Kernel,
    function: &CudaFunction,
    instructions: &Instructions,
    groups: &[[u32; 5]],
    initial: &[i16],
    paired: bool,
) -> Result<(Vec<i16>, Vec<i32>)> {
    let count = u32::try_from(groups.len())?;
    if count == 0
        || initial.len() != RELEASE_WORDS
        || groups.iter().any(|&[first, second, stage, copy, epoch]| {
            first >= instructions.count
                || (second != NONE && (second >= instructions.count || first == second))
                || stage >= 2
                || copy > 1
                || epoch >= EPOCHS as u32
        })
    {
        return Err("invalid GPU fit group bounds".into());
    }
    let group_words = stream.clone_htod(groups.as_flattened())?;
    let paired = u32::from(paired);
    let mut state = stream.clone_htod(initial)?;
    let mut audit = stream.alloc_zeros::<i32>(groups.len() * 2 * AUDIT_WORDS)?;
    let mut call = stream.launch_builder(function);
    call.arg(&kernel.roots)
        .arg(&instructions.rows)
        .arg(&instructions.count)
        .arg(&group_words)
        .arg(&count)
        .arg(&paired)
        .arg(&mut state)
        .arg(&mut audit);
    // Inputs use the frozen packed instruction stride. One GPU worker owns the residual.
    unsafe { call.launch(grid(1))? };
    let audit = stream.clone_dtoh(&audit)?;
    if let Some((member, record)) = audit
        .chunks_exact(AUDIT_WORDS)
        .enumerate()
        .find(|(_, row)| row[0] != 0)
    {
        return Err(format!(
            "GPU fit refused group {} member {} with status {}",
            member / 2,
            member % 2,
            record[0]
        )
        .into());
    }
    Ok((stream.clone_dtoh(&state)?, audit))
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
struct StageAudit {
    groups: usize,
    opposed_groups: usize,
    identical_support_groups: usize,
    stop_targets: usize,
    copy_targets: usize,
    link_visits: u64,
    total_support: u64,
    shared_support: u64,
    coordinate_cap_budget: usize,
    whole_support_opportunities: u64,
    eligible_opportunities: u64,
    violated_before: usize,
    violated_settled: usize,
    actual_writes: u64,
    shared_writes: u64,
    distinguishing_writes: u64,
    unpaired_writes: u64,
    repeat_writes: u64,
}

#[derive(Debug, Serialize)]
struct Write {
    coordinate: usize,
    shared_with_peer: Option<bool>,
}

#[derive(Debug, Serialize)]
struct MemberTrace {
    train_index: u32,
    copy: bool,
    before_group: i32,
    before_own_update: i32,
    after_own_update: i32,
    after_group: i32,
    signed_margin_before_group: i32,
    signed_margin_before_own_update: i32,
    signed_margin_after_group: i32,
    raw_visits: i32,
    support: i32,
    shared_support: i32,
    update_support: i32,
    writes: Vec<Write>,
}

#[derive(Debug, Serialize)]
struct GroupTrace {
    group: usize,
    stage: u32,
    kind: schedule::Kind,
    members: Vec<MemberTrace>,
}

fn sign(copy: bool) -> i32 {
    if copy {
        1
    } else {
        -1
    }
}

fn member_trace(
    index: u32,
    copy: bool,
    has_peer: bool,
    distinguish: bool,
    peer_prior_delta: i32,
    peer_later_delta: i32,
    record: &[i32],
) -> Result<MemberTrace> {
    let [0, before, after, settled, visits, support, shared, options, writes, shared_writes, unique_writes, coordinates @ ..] =
        record
    else {
        return Err("malformed GPU audit".into());
    };
    let sign = sign(copy);
    let expected_writes = if sign * before < MARGIN {
        (*options).min(COORDINATE_CAP as i32)
    } else {
        0
    };
    if !(1..=WORDS as i32).contains(support)
        || visits < support
        || !(0..=*support).contains(shared)
        || (!has_peer && *shared != 0)
        || *options
            != if distinguish {
                support - shared
            } else {
                *support
            }
        || *writes != expected_writes
        || *after != before + sign * writes
        || *settled != after + peer_later_delta
    {
        return Err(format!("inconsistent GPU audit for TRAIN row {index}: {record:?}").into());
    }
    let mut selected: Vec<Write> = Vec::with_capacity(*writes as usize);
    for &encoding in &coordinates[..*writes as usize] {
        if !(0..2 * WORDS as i32).contains(&encoding) {
            return Err("GPU audit coordinate out of range".into());
        }
        let is_shared = encoding >= WORDS as i32;
        let coordinate = encoding as usize % WORDS;
        if (is_shared && (!has_peer || distinguish))
            || selected.iter().any(|item| item.coordinate == coordinate)
        {
            return Err("GPU audit selected duplicate or disallowed shared address".into());
        }
        selected.push(Write {
            coordinate,
            shared_with_peer: has_peer.then_some(is_shared),
        });
    }
    if selected
        .iter()
        .filter(|w| w.shared_with_peer == Some(true))
        .count()
        != *shared_writes as usize
        || selected
            .iter()
            .filter(|w| w.shared_with_peer == Some(false))
            .count()
            != *unique_writes as usize
    {
        return Err("GPU address classification counts do not match recorded writes".into());
    }
    let before_group = before - peer_prior_delta;
    Ok(MemberTrace {
        train_index: index,
        copy,
        before_group,
        before_own_update: *before,
        after_own_update: *after,
        after_group: *settled,
        signed_margin_before_group: sign * before_group,
        signed_margin_before_own_update: sign * before,
        signed_margin_after_group: sign * settled,
        raw_visits: *visits,
        support: *support,
        shared_support: *shared,
        update_support: *options,
        writes: selected,
    })
}

fn audit_fit(
    initial: &[i16],
    device: &[i16],
    tape: &[schedule::Group<'_>],
    audit: &[i32],
    paired: bool,
) -> Result<([StageAudit; 2], Vec<GroupTrace>)> {
    let mut verified = initial.to_vec();
    let mut seen = [false; WORDS];
    let mut stages: [StageAudit; 2] = Default::default();
    let mut trace = Vec::with_capacity(tape.len());
    if audit.len() != tape.len() * 2 * AUDIT_WORDS {
        return Err("GPU audit length does not fit the schedule".into());
    }
    for (group, (slot, records)) in tape
        .iter()
        .zip(audit.chunks_exact(2 * AUDIT_WORDS))
        .enumerate()
    {
        let [first, second, stage, copy, _] = slot.device_words();
        let copy = copy == 1;
        let has_peer = second != NONE;
        let distinguish = has_peer && paired;
        let (left, right) = records.split_at(AUDIT_WORDS);
        let left_trace = member_trace(
            first,
            copy,
            has_peer,
            distinguish,
            0,
            -sign(copy) * right[9],
            left,
        )?;
        let mut members = vec![left_trace];
        let stage_total = &mut stages[stage as usize];
        stage_total.groups += 1;
        if has_peer {
            let right_trace = member_trace(
                second,
                !copy,
                true,
                distinguish,
                sign(copy) * left[9],
                0,
                right,
            )?;
            if left[6] != right[6] {
                return Err("members disagree on intersection size".into());
            }
            stage_total.opposed_groups += 1;
            stage_total.identical_support_groups +=
                usize::from(left[5] == left[6] && right[5] == right[6]);
            members.push(right_trace);
        } else if right.iter().any(|&word| word != 0) {
            return Err("unpaired group has unexpected second-member GPU writes".into());
        }
        for member in &members {
            stage_total.copy_targets += usize::from(member.copy);
            stage_total.stop_targets += usize::from(!member.copy);
            stage_total.link_visits += member.raw_visits as u64;
            stage_total.total_support += member.support as u64;
            stage_total.shared_support += member.shared_support as u64;
            stage_total.coordinate_cap_budget += COORDINATE_CAP;
            stage_total.whole_support_opportunities +=
                member.support.min(COORDINATE_CAP as i32) as u64;
            stage_total.eligible_opportunities +=
                member.update_support.min(COORDINATE_CAP as i32) as u64;
            stage_total.violated_before +=
                usize::from(member.signed_margin_before_own_update < MARGIN);
            stage_total.violated_settled += usize::from(member.signed_margin_after_group < MARGIN);
            stage_total.actual_writes += member.writes.len() as u64;
            for write in &member.writes {
                match write.shared_with_peer {
                    Some(true) => stage_total.shared_writes += 1,
                    Some(false) => stage_total.distinguishing_writes += 1,
                    None => stage_total.unpaired_writes += 1,
                }
                stage_total.repeat_writes += u64::from(seen[write.coordinate]);
                seen[write.coordinate] = true;
                let word = &mut verified[STATE_WORDS + write.coordinate];
                *word = word
                    .checked_add(sign(member.copy) as i16)
                    .ok_or("integer audit overflow")?;
            }
        }
        trace.push(GroupTrace {
            group,
            stage,
            kind: slot.kind,
            members,
        });
    }
    if verified != device {
        return Err("recorded signed GPU writes did not reconstruct the complete state".into());
    }
    Ok((stages, trace))
}

pub(super) fn fit(
    files: &Files<'_>,
    image_spec: [&str; 2],
    panel: &str,
    keys: &str,
    link: &str,
    output: &Path,
) -> Result<()> {
    let started = Instant::now();
    let (input_sha256, rows) = run::load_panel(panel)?;
    let (keys_sha256, tape) = schedule::load(&rows, keys)?;
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
    let authenticated = auth::authenticate(&replay, &core, &initial_scores)?;
    for group in tape.iter().filter(|group| group.stage == 1) {
        for member in [Some(&group.first), group.second.as_ref()]
            .into_iter()
            .flatten()
        {
            if authenticated[member.train_index as usize]
                .supervised_b
                .is_none()
            {
                return Err("post-B schedule contains an unauthenticated original copy".into());
            }
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
        "paired_fit",
    )?;
    let device_tape: Vec<_> = tape.iter().map(schedule::Group::device_words).collect();
    let mut outputs = BTreeMap::new();
    let mut first_support = None;
    for (arm, label) in ARMS.into_iter().enumerate() {
        let paired = arm == 1;
        let train_started = Instant::now();
        let (state, audit) = device_fit(
            stream,
            &replay.kernel,
            &trainer,
            &instructions,
            &device_tape,
            &initial,
            paired,
        )?;
        let train_elapsed_ms = train_started.elapsed().as_millis();
        let repeated = device_fit(
            stream,
            &replay.kernel,
            &trainer,
            &instructions,
            &device_tape,
            &initial,
            paired,
        )?;
        if repeated.0 != state || repeated.1 != audit {
            return Err(format!("{label} GPU refit changed integer weights or audit").into());
        }
        let (stages, trace) = audit_fit(&initial, &state, &tape, &audit, paired)?;
        let support: Vec<_> = trace
            .iter()
            .flat_map(|group| group.members.iter())
            .map(|member| (member.raw_visits, member.support, member.shared_support))
            .collect();
        if first_support
            .as_ref()
            .is_some_and(|first| *first != support)
        {
            return Err("arms used different geometry or exposure counts".into());
        }
        first_support = Some(support);
        let bytes: Vec<_> = state.iter().flat_map(|word| word.to_le_bytes()).collect();
        let state_path = output.join(format!("{label}.i16le"));
        fs::write(&state_path, &bytes)?;
        let weights_sha256 = file_hash(&state_path)?;
        let restored = read_state(
            state_path.to_str().ok_or("non-UTF-8 state path")?,
            &weights_sha256,
            RELEASE_WORDS,
        )?;
        if restored != state {
            return Err("serialized signed state did not reproduce the GPU state".into());
        }
        let original_device = stream.clone_htod(&state)?;
        let restored_device = stream.clone_htod(&restored)?;
        let scores = stream.clone_dtoh(&replay.kernel.scores(
            stream,
            &instructions,
            &original_device,
        )?)?;
        let restored_scores = stream.clone_dtoh(&replay.kernel.scores(
            stream,
            &instructions,
            &restored_device,
        )?)?;
        if scores != restored_scores {
            return Err("restored signed words changed GPU release scores".into());
        }
        let score_path = output.join(format!("{label}-train-scores.jsonl"));
        let audit_path = output.join(format!("{label}-slots.jsonl"));
        write_new(&score_path, scores.chunks_exact(STAGES))?;
        write_new(&audit_path, trace)?;
        outputs.insert(label, serde_json::json!({
            "weights_sha256": weights_sha256,
            "slot_audit_sha256": file_hash(&audit_path)?,
            "train_scores_sha256": file_hash(&score_path)?,
            "train_elapsed_ms": train_elapsed_ms,
            "identical_refit": true,
            "integer_round_trip": true,
            "changed_residual_words": state[STATE_WORDS..].iter().filter(|&&word| word != 0).count(),
            "stages": stages,
        }));
    }
    let receipt = serde_json::json!({
        "pass": true, "rows": rows.len(), "epochs": EPOCHS, "margin": MARGIN,
        "coordinate_cap": COORDINATE_CAP, "groups_per_arm": tape.len(),
        "row_exposures_per_arm": rows.len() * EPOCHS * 2,
        "schedule": schedule::summary(&tape),
        "post_b_authenticated_rows": authenticated.iter().filter(|row| row.supervised_b.is_some()).count(),
        "state_words": RELEASE_WORDS, "state_bytes": RELEASE_WORDS * 2,
        "input_sha256": input_sha256, "keys_sha256": keys_sha256,
        "linked_weights_sha256": run::CANDIDATE_STATE, "initial_weights_sha256": initial_sha256,
        "image_sha256": image_spec[1], "roots_sha256": BASE[7].1,
        "schedule_sha256": file_hash(&schedule_path)?,
        "authenticated_sha256": file_hash(&authenticated_path)?,
        "outputs": outputs, "elapsed_ms": started.elapsed().as_millis(),
    });
    fs::write(
        output.join("fit.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
