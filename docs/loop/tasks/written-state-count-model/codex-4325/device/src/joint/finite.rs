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

pub(super) fn verify(image_spec: [&str; 2], old: [&str; 2]) -> Result<()> {
    if old[1] != run::CANDIDATE_IMAGE {
        return Err("finite replay requires the historical linked scorer".into());
    }
    let stream = context()?;
    let mut roots = vec![0; VOCAB as usize];
    roots[6] = 1;
    let kernel = Kernel::open(&stream, image_spec[0], image_spec[1], &roots)?;
    let frozen = Kernel::open(&stream, old[0], old[1], &roots)?;
    let trainer = image(
        stream.context(),
        image_spec[0],
        Some(image_spec[1]),
        "joint_fit",
    )?;
    let instructions = Instructions::from_slices(&stream, &[&[5, 5, 6], &[5, 5, 5, 5]])?;
    let mut initial = vec![0; RELEASE_WORDS];
    initial[2 * (WORDS + 1) + WORDS] = -11;
    let baseline = stream.clone_htod(&initial)?;
    let initial_scores = stream.clone_dtoh(&kernel.scores(&stream, &instructions, &baseline)?)?;
    let frozen_scores = stream.clone_dtoh(&frozen.scores(&stream, &instructions, &baseline)?)?;
    if initial_scores != [0, 0, -11, 0, 0, -11] || frozen_scores != initial_scores {
        return Err("finite zero residual changed the frozen per-stage score".into());
    }
    let mut planted = initial.clone();
    let forward = bucket(1, 2);
    planted[STATE_WORDS + forward] = 9;
    planted[STATE_WORDS + (forward ^ 2048)] = -5;
    let planted = stream.clone_htod(&planted)?;
    let planted_scores = kernel.scores(&stream, &instructions, &planted)?;
    let signed = stream.clone_dtoh(&planted_scores)?;
    if signed[..STAGES] != [9, -5, -11] {
        return Err(
            format!("finite stage phase or fixed after-C score changed: {signed:?}").into(),
        );
    }
    witness_commits(&stream, &kernel, &planted_scores)?;

    let slots = [
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ];
    let (fitted, audit) =
        fit::device_fit(&stream, &kernel, &trainer, &instructions, &slots, &initial)?;
    let first_bucket = bucket(0, 2);
    let second_bucket = bucket(0, 3);
    let forward_and_backward = [
        first_bucket,
        first_bucket ^ 2048,
        second_bucket,
        second_bucket ^ 2048,
    ];
    let mut expected = initial.clone();
    for coordinate in forward_and_backward {
        expected[STATE_WORDS + coordinate] = -2;
    }
    let observed: Vec<_> = audit
        .chunks_exact(22)
        .map(|row| (row[1], row[2], row[3], row[4], row[5]))
        .collect();
    if fitted != expected
        || observed
            != [
                (0, 4, 6, 4, 4),
                (4, 0, 6, 4, 4),
                (0, -4, 6, 4, 4),
                (-4, -8, 6, 4, 4),
                (-8, -8, 6, 4, 0),
            ]
    {
        return Err(
            format!("finite signed margins, shared alias or dedup differed: {observed:?}").into(),
        );
    }
    let fitted = stream.clone_htod(&fitted)?;
    let fitted_scores = stream.clone_dtoh(&kernel.scores(&stream, &instructions, &fitted)?)?;
    if fitted_scores[STAGES..] != [-8, -8, -11] {
        return Err(
            format!("finite fit and score disagreed on shared state: {fitted_scores:?}").into(),
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "pass": true, "image_sha256": image_spec[1], "linked_image_sha256": old[1],
            "initial_scores": initial_scores, "signed_scores": signed, "fitted_scores": fitted_scores,
            "margin_slots": observed, "shared_coordinates": forward_and_backward,
            "commit_cases": ["original B", "same token wrong address", "wrong token at original address",
                "missing address", "stop after B", "fixed stop after C"],
        })
    );
    Ok(())
}
