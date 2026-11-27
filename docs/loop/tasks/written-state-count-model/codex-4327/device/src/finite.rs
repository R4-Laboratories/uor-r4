use super::*;

fn bucket(code: u32, distance: u32) -> usize {
    let phased = code ^ ((distance * 0x9e37) & 0xffff);
    ((phased ^ (phased >> 7)) & 2047) as usize
}

fn witness_commits(
    stream: &Arc<CudaStream>,
    kernel: &Kernel,
    scores: &CudaSlice<i32>,
) -> Result<()> {
    const NATIVE: u32 = 73;
    const FACTS: [u32; 6] = [75, 121, 130, 131, 121, 71];
    const B: Address = Address { pos: 1, id: 121 };
    const C: Address = Address { pos: 2, id: 130 };
    const D: Address = Address { pos: 3, id: 131 };
    let cases = [
        (0, B, B, [121, 1, 1, 0]),
        (0, B, Address { pos: 4, id: 121 }, [NATIVE, 0, NONE, 3]),
        (0, B, Address { pos: 1, id: 131 }, [NATIVE, 0, NONE, 3]),
        (0, B, Address::NONE, [NATIVE, 0, NONE, 3]),
        (1, C, C, [EOS, 2, NONE, 0]),
        (2, D, D, [EOS, 2, NONE, 0]),
    ];
    let mut records = vec![0; cases.len() * COMMIT_STRIDE];
    for (&(stage, original, attempted, _), record) in
        cases.iter().zip(records.chunks_exact_mut(COMMIT_STRIDE))
    {
        let decision = Decision {
            stage,
            row: 0,
            fence: FACTS.len() as u32,
            original,
            eligible: None,
            intervention: None,
            requested_count: 2,
            learned_score: None,
        };
        let authenticated = decision.prepare(record, NATIVE, &FACTS, attempted);
        if authenticated != (original == attempted) || record[5] != 0 {
            return Err("finite learned commit did not check address or clear target".into());
        }
    }
    for ((_, _, _, expected), commit) in cases
        .into_iter()
        .zip(kernel.select(stream, &records, scores)?)
    {
        if [commit.id, commit.source, commit.next, commit.refusal] != expected {
            return Err(format!("finite release expected {expected:?}, got {commit:?}").into());
        }
    }
    Ok(())
}

fn stage_scores(
    stream: &Arc<CudaStream>,
    kernel: &Kernel,
    instructions: &Instructions,
    state: &[i16],
) -> Result<Vec<i32>> {
    let state = stream.clone_htod(state)?;
    Ok(stream.clone_dtoh(&kernel.scores(stream, instructions, &state)?)?)
}

fn pair_actions(
    stream: &Arc<CudaStream>,
    kernel: &Kernel,
    instructions: &Instructions,
    state: &[i16],
    stage: u32,
) -> Result<[u32; 2]> {
    let state = stream.clone_htod(state)?;
    let scores = kernel.scores(stream, instructions, &state)?;
    let mut records = vec![0; 2 * COMMIT_STRIDE];
    let facts = [75, 121, 130, 131, 121, 71];
    let original = Address {
        pos: stage + 1,
        id: facts[stage as usize + 1],
    };
    for (row, record) in records.chunks_exact_mut(COMMIT_STRIDE).enumerate() {
        let decision = Decision {
            stage,
            row: row as u32,
            fence: facts.len() as u32,
            original,
            eligible: None,
            intervention: None,
            requested_count: 0,
            learned_score: None,
        };
        if !decision.prepare(record, 73, &facts, original) || record[5] != 0 {
            return Err(
                "finite pair action requires a verified address without a training label".into(),
            );
        }
    }
    let commits = kernel.select(stream, &records, &scores)?;
    Ok([commits[0].id, commits[1].id])
}

pub(super) fn verify(image_spec: [&str; 2], old: [&str; 2]) -> Result<()> {
    if old[1] != run::CANDIDATE_IMAGE {
        return Err("finite requires the historical linked scorer".into());
    }
    let stream = context()?;
    let mut roots = vec![0; VOCAB as usize];
    roots[6] = 1;
    roots[7] = 2;
    let kernel = Kernel::open(&stream, image_spec[0], image_spec[1], &roots)?;
    let frozen = Kernel::open(&stream, old[0], old[1], &roots)?;
    let trainer = image(
        stream.context(),
        image_spec[0],
        Some(image_spec[1]),
        "paired_fit",
    )?;
    let rows = [
        &[5, 5, 6, 7][..],
        &[5, 5, 7, 6][..],
        &[5, 5, 6, 7][..],
        &[5, 5, 6][..],
    ];
    let inputs = Instructions::from_slices(&stream, &rows)?;
    let mut initial = vec![0; RELEASE_WORDS];
    initial[2 * (WORDS + 1) + WORDS] = -11;
    let initial_scores = stage_scores(&stream, &kernel, &inputs, &initial)?;
    let frozen_scores = stage_scores(&stream, &frozen, &inputs, &initial)?;
    if initial_scores != [0, 0, -11].repeat(rows.len()) || initial_scores != frozen_scores {
        return Err("zero residual changed the frozen score".into());
    }
    let mut planted = initial.clone();
    let forward = bucket(1, 2);
    planted[STATE_WORDS + forward] = 9;
    planted[STATE_WORDS + (forward ^ 2048)] = -5;
    let planted = stream.clone_htod(&planted)?;
    let planted_scores = kernel.scores(&stream, &inputs, &planted)?;
    let planted_host = stream.clone_dtoh(&planted_scores)?;
    if planted_host[9..12] != [9, -5, -11] {
        return Err(format!("finite phase or stage-two score changed: {planted_host:?}").into());
    }
    // Commit fixture uses row zero with no reverse-distance collisions in its planted sign.
    if planted_host[..3] != [9, -5, -11] {
        return Err(format!("finite frozen commit has wrong score signs: {planted_host:?}").into());
    }
    witness_commits(&stream, &kernel, &planted_scores)?;

    let mut results = Vec::new();
    for (stage, copied_id) in [121, 130].into_iter().enumerate() {
        let phase = stage * 2048;
        let address_pair = |root| {
            [
                bucket(root, 3) ^ phase,
                (bucket(root << 5, 3) ^ 2048) ^ phase,
            ]
        };
        let stop_addresses = address_pair(2);
        let copy_addresses = address_pair(1);
        let groups =
            std::array::from_fn::<_, EPOCHS, _>(|epoch| [0, 1, stage as u32, 0, epoch as u32]);
        let device = |groups, initial, paired| {
            fit::device_fit(&stream, &kernel, &trainer, &inputs, groups, initial, paired)
        };
        let (correct, audit) = device(&groups[..], &initial[..], true)?;
        let mut expected = initial.clone();
        for address in stop_addresses {
            expected[STATE_WORDS + address] -= 4;
        }
        for address in copy_addresses {
            expected[STATE_WORDS + address] += 4;
        }
        if correct != expected
            || device(&groups[..], &initial[..], true)? != (correct.clone(), audit.clone())
        {
            return Err("paired finite writes or refit differed from hand-derived roots".into());
        }
        for member in audit.chunks_exact(fit::AUDIT_WORDS) {
            if member[4..11] != [6, 6, 4, 2, 2, 0, 2] {
                return Err(format!("unexpected paired address audit: {member:?}").into());
            }
        }
        let wrong_groups = groups.map(|[a, b, stage, copy, epoch]| [a, b, stage, 1 - copy, epoch]);
        let (wrong, _) = device(&wrong_groups[..], &initial[..], true)?;
        let reversed_groups =
            groups.map(|[a, b, stage, copy, epoch]| [b, a, stage, 1 - copy, epoch]);
        let (reversed, _) = device(&reversed_groups[..], &initial[..], true)?;
        let identical_groups = groups.map(|[a, _, stage, copy, epoch]| [a, 2, stage, copy, epoch]);
        let (identical, identical_audit) = device(&identical_groups[..], &initial[..], true)?;
        if reversed != correct
            || identical != initial
            || identical_audit
                .chunks_exact(fit::AUDIT_WORDS)
                .any(|row| row[5..11] != [6, 6, 0, 0, 0, 0])
        {
            return Err("paired finite swapped row order or identical-support check failed".into());
        }
        let good_scores = stage_scores(&stream, &kernel, &inputs, &correct)?;
        let wrong_scores = stage_scores(&stream, &kernel, &inputs, &wrong)?;
        let good_actions = pair_actions(&stream, &kernel, &inputs, &correct, stage as u32)?;
        let wrong_actions = pair_actions(&stream, &kernel, &inputs, &wrong, stage as u32)?;
        let signed_margins = [good_scores[stage], good_scores[STAGES + stage]];
        let wrong_assignment = [wrong_scores[stage], wrong_scores[STAGES + stage]];
        if signed_margins != [-8, 8]
            || wrong_assignment != [8, -8]
            || good_actions != [EOS, copied_id]
            || wrong_actions != [copied_id, EOS]
        {
            return Err(
                "label permutation failed to reverse both scores and committed actions".into(),
            );
        }

        let mut one_ok = initial.clone();
        one_ok[STATE_WORDS + stop_addresses[0]] = -8;
        let independent = [[0, 1, stage as u32, 0, 0]];
        let (_, partial) = device(&independent[..], &one_ok[..], true)?;
        let (_, rowwise) = device(&independent[..], &initial[..], false)?;
        if partial[8] != 0
            || partial[fit::AUDIT_WORDS + 8] != 2
            || rowwise[8..11] != [6, 4, 2]
            || rowwise[fit::AUDIT_WORDS + 1] != -4
            || rowwise[fit::AUDIT_WORDS + 8..fit::AUDIT_WORDS + 11] != [6, 4, 2]
        {
            return Err(
                "finite independent checks or sequential rowwise shared writes differ".into(),
            );
        }
        let fallback = [[0, NONE, stage as u32, 0, 0]];
        if device(&fallback[..], &initial[..], false)? != device(&fallback[..], &initial[..], true)?
        {
            return Err("finite unpaired policy differs between arms".into());
        }
        results.push(serde_json::json!({
            "stage": stage, "assigned_scores": signed_margins, "permuted_scores": wrong_assignment,
            "assigned_commit_ids": good_actions, "permuted_commit_ids": wrong_actions,
            "distinguishing_addresses": [stop_addresses, copy_addresses],
            "shared_support_per_member": 4, "rowwise_shared_writes_per_member": 4,
            "identical_support_unchanged": true, "reversed_member_order_unchanged": true,
            "independent_margin_holds_stop": true, "unpaired_policy_equal": true,
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "pass": true, "image_sha256": image_spec[1], "linked_image_sha256": old[1],
            "initial_scores": initial_scores, "planted_scores": planted_host, "finite_pairs": results,
            "commit_cases": ["original B", "same token wrong address", "wrong token at original address",
                "missing address", "stop after B", "fixed stop after C"],
        })
    );
    Ok(())
}
