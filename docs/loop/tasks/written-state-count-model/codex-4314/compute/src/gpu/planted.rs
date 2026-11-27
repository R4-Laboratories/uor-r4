use super::*;

fn probe(index: u32, near: bool) -> Row {
    let mut prefix = vec![0, 325 + index, 603 + 2 * index, 799, 741 + index];
    prefix.extend(contract::ASSISTANT_MARKER);
    if near {
        let pos = prefix.len() - 5;
        prefix[pos] = 36;
    }
    Row {
        prefix,
        target: 462,
        raw_target: None,
        rival: Some(1400),
        pair_id: format!("finite-{index}"),
        group_id: "finite".into(),
        assignment: 0,
        side: index,
        cohort: "finite".into(),
    }
}

pub fn fixture(image: &str, base: &str, out: &str) -> Result<()> {
    fs::create_dir(out)?;
    let path = Path::new(out);
    let device = Device::open(image, base)?;
    let candidates = (0..24).map(|i| probe(i, false)).collect::<Vec<_>>();
    let empty = device.score_arm(&candidates, &Weights::new(), 1)?;
    if empty.iter().any(|v| {
        v["scores"] != json!([0, 0, 0])
            || !v["key"].is_null()
            || v["support"] != 0
            || v["selected"] != 223
            || v["ranks"][2] != 1
            || v["ranks"][0] == v["ranks"][1]
    }) {
        return Err("zero table did not fall back to the exact frozen unigram order".into());
    }
    let first = 0;
    let second = (1..candidates.len())
        .find(|&i| empty[i]["question_state"] != empty[first]["question_state"])
        .ok_or("planted prefixes collide")?;
    let third = (1..candidates.len())
        .find(|&i| {
            empty[i]["question_state"] != empty[first]["question_state"]
                && empty[i]["question_state"] != empty[second]["question_state"]
        })
        .ok_or("need unseen fixture state")?;
    let a = candidates[first].clone();
    let mut b = candidates[second].clone();
    b.target = 1400;
    b.rival = Some(462);
    let near = probe(first as u32, true);
    let mut generic = near.clone();
    generic.target = contract::SPACE;
    generic.rival = None;
    let mut rows = vec![generic; 7];
    for _ in 0..4 {
        rows.push(a.clone());
        rows.push(b.clone());
    }
    let held = vec![
        a.clone(),
        b.clone(),
        near.clone(),
        candidates[third].clone(),
    ];
    let held_json = held
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    save_lines(&path.join("fixture.jsonl"), &held_json)?;
    for (arm, name) in ARM_NAMES.iter().enumerate() {
        let state = device.fit_arm(&rows, arm as u32)?;
        let reversed: Vec<_> = rows.iter().rev().cloned().collect();
        if state != device.fit_arm(&reversed, arm as u32)? {
            return Err("shuffled training replay differs".into());
        }
        let state_path = path.join(format!("{name}.tsv"));
        save(&state, &state_path)?;
        let observed = device.score_arm(&held, &load(&state_path)?, arm as u32)?;
        let score = |i: usize| &observed[i]["scores"];
        let chosen = |i: usize| observed[i]["selected"].as_u64();
        if arm == 0 {
            if score(0) != &json!([4, 4, 6])
                || score(1) != score(0)
                || chosen(0) != Some(223)
                || observed[0]["key"] != observed[2]["key"]
                || !observed[0]["question_state"].is_null()
            {
                return Err("colon fixtures failed".into());
            }
        } else {
            if score(0) != &json!([8, -4, -4])
                || score(1) != score(0)
                || score(2) != &json!([0, 0, 14])
                || chosen(0) != Some(462)
                || chosen(1) != Some(1400)
                || chosen(2) != Some(223)
                || observed[0]["ranks"][0] != 1
                || observed[1]["ranks"][0] != 1
                || !observed[2]["question_state"].is_null()
                || observed[3]["key"]
                    .as_u64()
                    .is_none_or(|k| k >> contract::LEVEL_SHIFT >= 4)
            {
                return Err("marker specificity, swapped-answer, near-marker or unseen-address fixture failed".into());
            }
        }
        save_lines(&path.join(format!("{name}.jsonl")), &observed)?;
    }
    save_lines(
        &path.join("finite.json"),
        &[json!({"status":"pass","train":15,"openings":4,
        "controls":["zero-support","trained correct token at each of two states","swapped answer loses",
        "signed rival and space negatives","matched colon scores","prior tie break","marker near miss",
        "sparse backoff","reversed training order","reloaded integer table"]})],
    )
}
