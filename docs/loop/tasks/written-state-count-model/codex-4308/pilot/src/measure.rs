use crate::{
    engine::{MoveMode, P5Gpu, StepResult},
    fail,
    rows::NativeRow,
    sample::Draw,
    Result,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
    time::Instant,
};
use uor_r4_discrete::p5_brier_core::{P5BatchBrier, P5S1State};

pub fn fresh(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

pub fn json_file(path: &Path, value: &Value) -> Result<()> {
    fresh(path, &serde_json::to_vec_pretty(value)?)
}

pub fn verify_file(path: &Path, expected: &str) -> Result<()> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    if hex::encode(hash.finalize()) != expected {
        return Err(fail(format!("wrong SHA for {}", path.display())));
    }
    Ok(())
}

pub fn checkpoint(state: &P5S1State, path: &Path) -> Result<(P5S1State, u128, String)> {
    let start = Instant::now();
    let encoded = state.encode_bytes()?;
    fresh(path, &encoded)?;
    let disk = std::fs::read(path)?;
    let reloaded = P5S1State::from_bytes(&disk)?;
    if disk != encoded || &reloaded != state {
        return Err(fail("persisted S1 state differs"));
    }
    Ok((
        reloaded,
        start.elapsed().as_micros(),
        hex::encode(Sha256::digest(&disk)),
    ))
}

fn timed(
    gpu: &P5Gpu,
    state: &mut P5S1State,
    batch: &P5BatchBrier,
    coords: &[u32],
    mode: MoveMode,
) -> Result<(StepResult, u128)> {
    let start = Instant::now();
    let step = gpu.step(state, batch, coords, mode)?;
    Ok((step, start.elapsed().as_micros()))
}

pub fn matched_step(
    gpu: &P5Gpu,
    state: &mut P5S1State,
    batch: &P5BatchBrier,
    coords: &[u32],
    all_first: bool,
) -> Result<Value> {
    let mut other = state.clone();
    let ((all, all_us), (distinct, distinct_us)) = if all_first {
        let first = timed(gpu, state, batch, coords, MoveMode::AllGenerators)?;
        (
            first,
            timed(
                gpu,
                &mut other,
                batch,
                coords,
                MoveMode::DistinctDestinations,
            )?,
        )
    } else {
        let first = timed(
            gpu,
            &mut other,
            batch,
            coords,
            MoveMode::DistinctDestinations,
        )?;
        (
            timed(gpu, state, batch, coords, MoveMode::AllGenerators)?,
            first,
        )
    };
    if all.scores != distinct.scores
        || all.record != distinct.record
        || *state != other
        || all.affected_rows != distinct.affected_rows
    {
        return Err(fail(
            "duplicate-move lane scores, membership, commit or state differ",
        ));
    }
    let s = &all.record;
    let mut hash = Sha256::new();
    for score in &all.scores {
        hash.update(score.coord.to_le_bytes());
        hash.update(score.status.to_le_bytes());
        for loss in score.losses {
            hash.update(loss.to_le_bytes());
        }
    }
    Ok(json!({
        "all_first":all_first,"coordinates":coords,"logical_moves":15*coords.len(),
        "score_words":16*coords.len(),"score_words_sha256":hex::encode(hash.finalize()),
        "all_us":all_us,"distinct_us":distinct_us,"all_kernel_us":all.score_micros,"distinct_kernel_us":distinct.score_micros,
        "all_device_us":all.device_micros,"distinct_device_us":distinct.device_micros,
        "all_plan_us":all.plan_micros,"distinct_plan_us":distinct.plan_micros,
        "all_upload_allocate_us":all.upload_allocate_micros,"distinct_upload_allocate_us":distinct.upload_allocate_micros,
        "all_readback_validate_us":all.readback_validate_micros,"distinct_readback_validate_us":distinct.readback_validate_micros,
        "all_destinations":all.evaluated_destinations,"distinct_destinations":distinct.evaluated_destinations,"affected_rows":all.affected_rows,
        "before_q32":s.l_before,"best_q32":s.l_best,"joint_q32":s.l_joint,"committed_q32":s.committed_loss,"joint":s.joint,
        "strict_candidates":s.s_len,"committed_coordinates":s.commit_coords,"committed_lanes":s.commit_lanes,"epoch":s.epoch_after
    }))
}

pub fn same_pick(a: &NativeRow, b: &NativeRow) -> bool {
    (a.pick_pos, a.pick_distance, a.pick_successor, a.pick_ties)
        == (b.pick_pos, b.pick_distance, b.pick_successor, b.pick_ties)
}

pub fn row_json(row: &NativeRow) -> Value {
    json!({"target":row.target,"top1":row.top1,"target_mass_q48":row.target_mass_q48,"top1_mass_q48":row.top1_mass_q48,
      "prll_mass_q48":row.prll_mass_q48,"prll_bits_q12":row.prll_bits_q12,"brier_q32":row.brier_q32,
      "pick_pos":row.pick_pos,"distance":row.pick_distance,"successor":row.pick_successor,"ties":row.pick_ties,"class":row.pick_class,"bucket":row.bucket})
}

/// Independent rows are sampled one per development record. Return sum-delta percentiles in Q32 units.
pub fn compare(base: &[NativeRow], rows: &[NativeRow]) -> Result<Value> {
    if base.is_empty()
        || rows.len() != base.len()
        || base.iter().zip(rows).any(|(a, b)| a.target != b.target)
    {
        return Err(fail("development pairs are misaligned"));
    }
    let n = rows.len();
    let diffs: Vec<i64> = base
        .iter()
        .zip(rows)
        .map(|(a, b)| b.brier_q32 as i64 - a.brier_q32 as i64)
        .collect();
    let mut draw = Draw::new(4312);
    let mut reps: Vec<i64> = (0..1000)
        .map(|_| (0..n).map(|_| diffs[draw.index(n)]).sum())
        .collect();
    reps.sort_unstable();
    let mask = |pred: fn(&NativeRow, &NativeRow) -> bool| {
        base.iter().zip(rows).filter(|(a, b)| pred(a, b)).count()
    };
    Ok(
        json!({"rows":n, "brier_sum_q32":rows.iter().map(|r|r.brier_q32).sum::<u64>(),
        "delta_vs_base_q32":diffs.iter().sum::<i64>(),"paired_delta_ci95_q32":[reps[24],reps[974]],
        "prll_bits_sum_q12":rows.iter().map(|r|r.prll_bits_q12).sum::<u64>(),
        "target_mass_changed":mask(|a,b|a.target_mass_q48!=b.target_mass_q48),
        "source_changed":mask(|a,b|!same_pick(a,b)), "top1_changed":mask(|a,b|a.top1!=b.top1),
        "top1_correct":rows.iter().filter(|r|r.top1==r.target).count(), "recovered":mask(|a,b|a.top1!=a.target && b.top1==b.target),
        "lost":mask(|a,b|a.top1==a.target && b.top1!=b.target),"better_brier":diffs.iter().filter(|v|**v<0).count(),
        "worse_brier":diffs.iter().filter(|v|**v>0).count()}),
    )
}
