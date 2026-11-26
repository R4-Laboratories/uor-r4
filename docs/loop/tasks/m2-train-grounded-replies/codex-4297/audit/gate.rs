use std::{collections::BTreeSet, path::Path};

use serde_json::{json, Value};

use crate::{check, read_json, write_json, Result};

type Ids = BTreeSet<String>;

fn array(value: &Value) -> Result<&Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| "gate pair membership is missing".into())
}

fn members(value: &Value) -> Result<Ids> {
    let values = array(value)?;
    let ids = values
        .iter()
        .map(serde_json::to_string)
        .collect::<serde_json::Result<Ids>>()?;
    check(values.len() == ids.len(), "duplicate gate pair membership")?;
    Ok(ids)
}

pub fn checks(held: &Value, training: &Value, arm: &str) -> Result<Value> {
    let all = &held["all"];
    let near = &held["distance"]["near"];
    let far = &held["distance"]["far"];
    let score = |slice: &Value, name: &str| members(&slice[name]["exact"]);
    let base_near = score(near, "baseline")?;
    let base_far = score(far, "baseline")?;
    let model_near = score(near, arm)?;
    let model_far = score(far, arm)?;
    let observations = held["family"]
        .as_object()
        .ok_or("gate family membership missing")?;
    let matched_far = score(far, "far_only")?
        .difference(&base_far)
        .cloned()
        .collect::<Ids>();
    let complete = score(all, arm)?.len();
    let own_before = training["objective_before"]
        .as_u64()
        .ok_or("missing TRAIN base loss")?;
    let own_after = training["objective_after"]
        .as_u64()
        .ok_or("missing TRAIN elected loss")?;
    let mut gains_by_distance = true;
    let mut total_gain = true;
    for baseline in ["baseline", "native", "unfitted"] {
        total_gain &= complete.saturating_sub(score(all, baseline)?.len()) >= 8;
        if baseline != "baseline" {
            gains_by_distance &= model_near.len() > score(near, baseline)?.len()
                && model_far.len() > score(far, baseline)?.len();
        }
    }
    let mut families = true;
    for group in observations.values() {
        families &= score(group, arm)?.len() >= score(group, "native")?.len();
    }
    let correct = score(all, "correct_link")?.len();
    let exchanged = score(all, "exchanged_link")?.len();
    Ok(json!({
        "retained_train_sources": training["qualified"] == true && members(&training["final"]["lost_protected"])?.is_empty(),
        "train_reduced_loss": own_after < own_before,
        "train_new_far_pair": !members(&training["final"]["far_gained"])?.is_empty(),
        "retains_held_near": !base_near.is_empty() && base_near.is_subset(&model_near),
        "retains_matched_far_gains": matched_far.is_subset(&model_far),
        "improves_far": model_far.len() > base_far.len(),
        "total_gain":total_gain,
        "gains_both_distances":gains_by_distance,
        "full_links_both_distances":!members(&near[arm]["full"])?.is_empty() && !members(&far[arm]["full"])?.is_empty(),
        "no_family_below_native":families,
        "supplied_controls":members(&all["correct_link"]["earlier"])?.len() == 128 && all["exchanged_link"]["prompts"] == 0
            && correct.saturating_sub(exchanged) >= 8
            && score(near, "correct_link")?.len() > score(near, "exchanged_link")?.len()
            && score(far, "correct_link")?.len() > score(far, "exchanged_link")?.len(),
    }))
}

pub fn passed(checks: &Value) -> bool {
    checks
        .as_object()
        .is_some_and(|values| !values.is_empty() && values.values().all(|item| item == true))
}

pub fn fixture(core: &Path, out: &Path) -> Result<()> {
    let training = read_json(&core.join("training.json"))?;
    let positive = checks(&read_json(&core.join("held.json"))?, &training, "all")?;
    let substitution = checks(&read_json(&core.join("near-swap.json"))?, &training, "all")?;
    let exchanged = checks(
        &read_json(&core.join("control-leak.json"))?,
        &training,
        "all",
    )?;
    check(
        passed(&positive)
            && !passed(&substitution)
            && substitution["retains_held_near"] == false
            && !passed(&exchanged)
            && exchanged["supplied_controls"] == false,
        "independent writer-gate finite controls did not separate",
    )?;
    write_json(
        out,
        &json!({"pass":true,"positive":positive,"near_identity_substitution":substitution,"exchanged_control_leak":exchanged}),
    )
}
