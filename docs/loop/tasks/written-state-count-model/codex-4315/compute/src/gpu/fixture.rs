use super::*;

fn probe(target: u32, rival: u32, anchor: u32) -> Row {
    Row {
        prefix: vec![
            0, 55, 2728, COLON, anchor, SPACE, target, COLON, SPACE, rival, COLON, 201, 201,
            anchor, COLON, 201, 35, 560, 652, 714, COLON,
        ],
        group_id: "finite".into(),
        pair_id: format!("probe-{target}"),
        assignment: 0,
        side: 0,
        target,
        rival,
        raw_target: Some(target),
        raw_rival: Some(rival),
        gold_pos: Some(6),
        rival_pos: Some(9),
        context_start: 4,
        context_end: 11,
        source_status: "unique".into(),
        cohort: "finite".into(),
    }
}

fn fit_fixture(device: &Device, good: &Row, bad: &Row, empty: &Row) -> Result<()> {
    let donors = [
        Donor {
            prefix: good.prefix.clone(),
            target: good.target,
            genuine_assistant: true,
        },
        Donor {
            prefix: good.prefix.clone(),
            target: good.target,
            genuine_assistant: false,
        },
        Donor {
            prefix: good.prefix.clone(),
            target: bad.target,
            genuine_assistant: true,
        },
        Donor {
            prefix: good.prefix.clone(),
            target: SPACE,
            genuine_assistant: true,
        },
    ];
    let linked = [
        good.clone(),
        bad.clone(),
        good.clone(),
        bad.clone(),
        good.clone(),
        bad.clone(),
        good.clone(),
        bad.clone(),
    ];
    let mut absent = good.clone();
    absent.gold_pos = None;
    let missing = [absent, empty.clone()];
    let forward = device.fit_batches(&donors, &linked, &missing, 1)?;
    let reversed = device.fit_batches(
        &donors.iter().rev().cloned().collect::<Vec<_>>(),
        &linked.iter().rev().cloned().collect::<Vec<_>>(),
        &missing.iter().rev().cloned().collect::<Vec<_>>(),
        1,
    )?;
    if forward.census != [1, 0, 1, 1, 0, 1, 8, 8, 1, 1, 0]
        || forward.census != reversed.census
        || forward.donor != reversed.donor
        || forward.contrast != reversed.contrast
        || forward.weights != reversed.weights
        || forward.donor[STATE_WORDS - 1] != -1
        || forward.contrast[STATE_WORDS - 1] != -8
        || forward.weights[STATE_WORDS - 1] != -9
    {
        return Err(format!("finite training schedule differs: {:?}", forward.census).into());
    }
    for slot in 0..FEATURES {
        let range = slot * 32..(slot + 1) * 32;
        if forward.donor[range.clone()].iter().sum::<i32>() != 1
            || forward.contrast[range].iter().sum::<i32>() != 8
        {
            return Err(format!("finite counter conservation failed for slot {slot}").into());
        }
    }
    Ok(())
}

pub fn fixture(image: &str, base: &str, class_path: &str, out: &str) -> Result<()> {
    let out = Path::new(out);
    fs::create_dir(out)?;
    let classes = classes(class_path)?;
    let lexical = |id: u32| (1..=6).contains(&classes[id as usize]);
    let mut bank = (256..VOCAB).filter(|&id| lexical(id));
    let anchor = bank.next().ok_or("empty fixture lexical bank")?;
    let rival = bank.next().ok_or("empty fixture lexical rival")?;
    let calibration = bank
        .take(128)
        .map(|id| (probe(id, rival, anchor), SPACE))
        .collect::<Vec<_>>();
    let device = Device::open(image, base, class_path)?;
    let writer = device.writer_state(base)?;
    let zero = vec![0; STATE_WORDS];
    let zero_device = device.stream.clone_htod(&zero)?;
    let sampled = device.score_rows(&writer, &calibration, &zero_device)?;
    let exact = sampled
        .iter()
        .position(|row| {
            let e = &row.arms.correct;
            e.immediate.gate_id == e.immediate.emitted_id
        })
        .ok_or("finite lexical gate never reaches the original ID")?;
    let alias = sampled
        .iter()
        .position(|row| {
            let e = &row.arms.correct;
            e.immediate.gate_id != e.immediate.emitted_id
        })
        .ok_or("finite alias witness not found")?;
    let good = calibration[exact].0.clone();
    let bad = calibration[alias].0.clone();
    let mut no_candidate = good.clone();
    no_candidate.prefix[4..11].copy_from_slice(&[SPACE, COLON, SPACE, COLON, SPACE, COLON, SPACE]);
    no_candidate.gold_pos = None;
    no_candidate.rival_pos = None;
    no_candidate.pair_id = "finite-empty".into();
    no_candidate.source_status = "absent".into();
    fit_fixture(&device, &good, &bad, &no_candidate)?;
    let mut delayed = good.clone();
    delayed.raw_target = Some(SPACE);
    delayed.pair_id = "finite-boundary-aligned".into();
    let mut unscoped = good.clone();
    unscoped.prefix.remove(unscoped.context_end);
    unscoped.pair_id = "finite-unscoped".into();
    unscoped.cohort = "finite_unscoped".into();
    let good_features = sampled[exact].candidates[2]
        .features
        .ok_or("ineligible finite gold")?;
    let mut planted = zero.clone();
    for feature in good_features {
        planted[feature as usize] = 1;
    }
    let inputs = vec![
        (good.clone(), SPACE),
        (bad.clone(), SPACE),
        (no_candidate, SPACE),
        (delayed, SPACE),
        (unscoped, SPACE),
    ];
    let state = out.join(STATE_FILE);
    save_weights(&state, &planted)?;
    let reloaded = read_weights(&state)?;
    if reloaded != planted {
        return Err("finite weight serialization differs".into());
    }
    let observed = device.observed(&writer, &inputs, &reloaded)?;
    let rev = inputs.iter().rev().cloned().collect::<Vec<_>>();
    let replay = device.observed(&writer, &rev, &reloaded)?;
    if !observed.iter().eq(replay.iter().rev()) {
        return Err("finite reversed device observations differ".into());
    }
    let [positive, negative, empty, aligned, unsupported] = observed.as_slice() else {
        return Err("wrong finite row count".into());
    };
    if positive.winner_pos != good.gold_pos
        || positive.gold_rank != Some(1)
        || positive.arms.learned.immediate.emitted_id != good.target
        || positive.arms.correct.immediate.emitted_id != good.target
        || positive.arms.swapped.immediate.gate_id != good.rival
        || positive.arms.absent != positive.arms.marker
        || sampled[exact].winner_pos.is_some()
        || sampled[exact].best_ties < 2
        || negative.arms.correct.immediate.gate_id != bad.target
        || negative.arms.correct.immediate.emitted_id == bad.target
        || empty.winner_pos.is_some()
        || empty.best_ties != 0
        || empty.candidates.iter().any(|c| c.eligible)
        || empty.arms.learned != empty.arms.marker
        || unsupported.parsed_start.is_some()
        || !unsupported.candidates.is_empty()
        || unsupported.arms.correct != unsupported.arms.marker
        || unsupported.arms.swapped != unsupported.arms.marker
        || aligned.arms.correct.immediate.source_pos.is_some()
        || aligned.arms.correct.immediate != aligned.arms.marker.immediate
        || aligned
            .arms
            .correct
            .lexical
            .as_ref()
            .is_some_and(|e| e.source_pos != good.gold_pos)
    {
        return Err(format!("finite device assertions differ: {observed:?}").into());
    }

    let controls = [
        "unique learned source beats all eligible occurrences",
        "source rank reads all eligible positions",
        "supplied lexical source reaches original ID",
        "supplied swapped source selects actual rival ID",
        "writer alias changes a real lexical gate ID",
        "NoRead on tied score",
        "NoRead on empty fact candidates",
        "unauthenticated fact scope fails closed for supplied sources",
        "forced NoRead reproduces frozen fallback",
        "delayed reference does not force lexical ID at colon",
        "integer state exact save/reload",
        "finite training conservation and reverse-order atomic accumulation",
        "full CUDA replay in reverse order",
    ];
    let observed = inputs
        .iter()
        .map(|(r, _)| r)
        .zip(observed)
        .map(|(row, decision)| wire(row, decision))
        .collect::<Result<Vec<_>>>()?;
    let inputs = inputs.into_iter().map(|(r, _)| r).collect::<Vec<_>>();
    write_rows(&out.join("fixture-inputs.jsonl"), &inputs)?;
    write_rows(&out.join("fixture-observations.jsonl"), &observed)?;
    write_json(
        &out.join("finite.json"),
        &json!({
            "status": "pass",
            "lexical_original_id": good.target,
            "alias_original_id": bad.target,
            "alias_emitted_id": sampled[alias].arms.correct.immediate.emitted_id,
            "probe_bank": calibration.len(),
            "controls": controls,
            "base_sha256": BASE_SHA,
            "fixture_marker_gate": SPACE,
        }),
    )
}
