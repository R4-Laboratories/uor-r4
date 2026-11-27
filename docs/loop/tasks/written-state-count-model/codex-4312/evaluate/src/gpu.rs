use super::*;
use d11::artifact::ArtifactTables;
use d11::device::DeviceScorer;
use d11::kernel::{pack_rows, pack_rows_with_targets};
use std::collections::BTreeMap;

pub(super) fn run(
    op: &str,
    artifact: &str,
    arm: &str,
    input: &str,
    out: &Path,
) -> Result<(), String> {
    if std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() != Ok("1") {
        return Err("require ahead-of-time CUDA loading".into());
    }
    let bytes = fs::read(artifact).map_err(|e| e.to_string())?;
    let (tables, client) = ArtifactTables::from_bytes(&bytes)?;
    if !client.is_trunk_d11()
        || client.shape().context as usize != CONTEXT
        || client.shape().vocabulary != VOCAB
    {
        return Err("expected the registered complete D11 trunk with original IDs".into());
    }
    let device = DeviceScorer::new(0, &tables)?;
    if device.capability != (8, 9) {
        return Err("expected an sm_89 device".into());
    }
    fs::create_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
    if op == "generate" {
        generate(&device, arm, input, out)
    } else {
        teacher(&device, arm, input, out)
    }
}

fn generate(device: &DeviceScorer, arm: &str, input: &str, out: &Path) -> Result<(), String> {
    let prompts = check_set(input)?;
    let mut witnesses = BTreeMap::<&str, Vec<&str>>::new();
    for p in &prompts {
        witnesses.entry(&p.domain).or_default().push(&p.id);
    }
    let mut retained = BTreeSet::new();
    for ids in witnesses.values_mut() {
        ids.sort();
        retained.insert(ids[0]);
        retained.insert(ids[ids.len() - 1]);
    }
    let mut states: Vec<_> = prompts.iter().map(|p| p.prefix.clone()).collect();
    let mut replies = vec![Vec::new(); prompts.len()];
    let mut trace = writer(&out.join("witnesses.jsonl"))?;
    let mut active: Vec<_> = (0..prompts.len()).collect();
    while !active.is_empty() {
        let prefixes: Vec<_> = active.iter().map(|&i| states[i].as_slice()).collect();
        let packed = pack_rows(&prefixes)?;
        let interventions = vec![[0, -1]; active.len()];
        let (scores, pointers) = device.score_pointers(&packed, &interventions)?;
        if scores.len() != active.len() || pointers.len() != active.len() {
            return Err("missing CUDA rows".into());
        }
        let mut next = Vec::new();
        for (j, &i) in active.iter().enumerate() {
            let score = &scores[j];
            let pred = score.prediction;
            if score.status != 0 || pred >= VOCAB || score.top8.first() != Some(&pred) {
                return Err(format!(
                    "invalid CUDA result {} at {}",
                    score.status, prompts[i].id
                ));
            }
            if retained.contains(prompts[i].id.as_str()) {
                line(
                    &mut trace,
                    &Witness {
                        id: prompts[i].id.clone(),
                        offset: replies[i].len(),
                        prefix: states[i].clone(),
                        cells: score.admitted_cells.clone(),
                        read: score.read_vector.clone(),
                        top8: score.top8.clone(),
                        prediction: pred,
                        pointer: pointers[j],
                    },
                )?;
            }
            replies[i].push(pred);
            if pred != EOS && replies[i].len() < prompts[i].cap {
                slide(&mut states[i], pred);
                next.push(i);
            }
        }
        active = next;
    }
    trace.flush().map_err(|e| e.to_string())?;
    let mut file = writer(&out.join("replies.jsonl"))?;
    for (p, tokens) in prompts.iter().zip(replies) {
        line(
            &mut file,
            &Reply {
                id: p.id.clone(),
                arm: arm.into(),
                tokens,
            },
        )?;
    }
    file.flush().map_err(|e| e.to_string())
}

fn teacher(device: &DeviceScorer, arm: &str, input: &str, out: &Path) -> Result<(), String> {
    let rows: Vec<Teacher> = read_rows(input)?;
    if rows.is_empty() {
        return Err("missing teacher positions".into());
    }
    let mut ids = BTreeSet::new();
    let mut file = writer(&out.join("teacher.jsonl"))?;
    for group in rows.chunks(16) {
        let mut windows = Vec::new();
        let mut targets = Vec::new();
        for row in group {
            check_prompt(&row.id, &row.domain, &row.prefix, row.cap)?;
            if !ids.insert(&row.id)
                || row.gold.is_empty()
                || row.gold.len() != row.new_teacher_tokens.len()
                || row.gold.iter().any(|&t| t >= VOCAB)
            {
                return Err(format!("invalid teacher row {}", row.id));
            }
            let mut prefix = row.prefix.clone();
            for &target in &row.gold {
                windows.push(prefix.clone());
                targets.push(target);
                slide(&mut prefix, target);
            }
        }
        let refs: Vec<_> = windows.iter().map(Vec::as_slice).collect();
        let scores = device.score(&pack_rows_with_targets(&refs, &targets)?)?;
        let mut scores = scores.into_iter();
        for row in group {
            let mut ranks = Vec::new();
            let mut predictions = Vec::new();
            for target in &row.gold {
                let score = scores.next().ok_or("missing CUDA teacher result")?;
                if score.status != 0 {
                    return Err(format!("row {} status {}", row.id, score.status));
                }
                ranks.push(score.top8.iter().position(|t| t == target).map(|i| i + 1));
                predictions.push(score.prediction);
            }
            line(
                &mut file,
                &serde_json::json!({"id":row.id,"arm":arm,"domain":row.domain,
                "gold":row.gold,"predictions":predictions,"top8_rank":ranks,
                "new_teacher_tokens":row.new_teacher_tokens}),
            )?;
        }
        if scores.next().is_some() {
            return Err("unexpected extra CUDA result".into());
        }
    }
    file.flush().map_err(|e| e.to_string())
}
