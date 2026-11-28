use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::{self, BufRead, BufReader},
};
use uor_r4_native_vector::{
    butterfly_rotor::{ACT_CLAMP, HIER_COARSE, HIER_FINE},
    butterfly_rotor_v3::ButterflyV3Model,
};

mod inventory {
    #![allow(dead_code)]
    include!("weight_inventory.rs");

    pub(super) fn checkpoints(paths: [&str; 3]) -> Result<[ButterflyV3Model; 3]> {
        Ok([
            load(paths[0], PINS[0])?,
            load(paths[1], PINS[1])?,
            load(paths[2], PINS[2])?,
        ])
    }

    pub(super) const IDENTITIES: [(&str, &str, u64); 3] = PINS;
}

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CUBIN: &str = "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713";
const TRAIN: &str = "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a";
const TRAIN_ROWS: usize = 256;
const CELL: usize = 3;
const QUOTE_ID: u16 = 1228;
const QUOTE_SLOT: usize = CELL * HIER_FINE;

#[derive(Deserialize, Eq, PartialEq)]
struct Signature {
    native_transition_sha256: String,
    complete_trace_i32le_sha256: String,
    diagnostic_mass_q48_le_sha256: String,
}

#[derive(Deserialize)]
struct Row {
    id: u64,
    training_document_index: usize,
    checkpoint: String,
    checkpoint_sha256: String,
    checkpoint_step: u64,
    train_sha256: String,
    cubin_sha256: String,
    target_id: u16,
    target_cell: usize,
    actual_first_id: u16,
    actual_first_global_slot: usize,
    reference_free_transition_match: bool,
    blind_sham_complete_match: bool,
    blind_signature: Signature,
    rich_signature: Signature,
    sham_signature: Signature,
    hidden_q15: Vec<i32>,
    hidden_sha256: String,
    cuda_raw_coarse_q15: Vec<i32>,
    cuda_raw_elected_fine_q15: Vec<i32>,
    cuda_raw_target_fine_q15: Vec<i32>,
}

impl Row {
    fn check(
        &self,
        model: &ButterflyV3Model,
        identity: (&str, &str, u64),
        index: usize,
    ) -> Result<()> {
        let (name, pin, step) = identity;
        let mut digest = Sha256::new();
        for word in &self.hidden_q15 {
            digest.update(word.to_le_bytes());
        }
        if self.checkpoint != name
            || self.checkpoint_sha256 != pin
            || self.checkpoint_step != step
            || self.train_sha256 != TRAIN
            || self.cubin_sha256 != CUBIN
            || self.training_document_index != index
            || self.hidden_q15.len() != model.cfg.dim
            || self.hidden_sha256 != hex::encode(digest.finalize())
            || self.cuda_raw_coarse_q15.len() != HIER_COARSE
            || self.cuda_raw_elected_fine_q15.len() != HIER_FINE
            || self.cuda_raw_target_fine_q15.len() != HIER_FINE
            || !self.reference_free_transition_match
            || !self.blind_sham_complete_match
            || self.blind_signature != self.sham_signature
            || self.blind_signature.native_transition_sha256
                != self.rich_signature.native_transition_sha256
            || model
                .vocab_perm
                .get(usize::from(self.actual_first_id))
                .copied()
                .map(usize::from)
                != Some(self.actual_first_global_slot)
            || model
                .vocab_perm
                .get(usize::from(self.target_id))
                .copied()
                .map(|s| usize::from(s) / HIER_FINE)
                != Some(self.target_cell)
        {
            return Err(format!("invalid {name} physical receipt for document {index}").into());
        }
        Ok(())
    }

    fn fine(&self) -> Result<(&'static str, Option<&[i32]>)> {
        let elected = self.actual_first_global_slot / HIER_FINE == CELL;
        let target = self.target_cell == CELL;
        match (elected, target) {
            (true, true) => {
                if self.cuda_raw_elected_fine_q15 != self.cuda_raw_target_fine_q15 {
                    return Err("repeat physical cell-3 fine traces disagree".into());
                }
                Ok(("elected_and_target", Some(&self.cuda_raw_elected_fine_q15)))
            }
            (true, false) => Ok(("elected_only", Some(&self.cuda_raw_elected_fine_q15))),
            (false, true) => Ok(("target_only", Some(&self.cuda_raw_target_fine_q15))),
            (false, false) => Ok(("unobserved", None)),
        }
    }
}

struct Endpoint<'a> {
    hidden: &'a [i32],
    candidate: &'a [i16],
    rival: &'a [i16],
    observed_candidate: i32,
    observed_rival: i32,
}

#[derive(Serialize)]
struct Parts {
    hidden_change: i64,
    row_movement: i64,
    exact_dot_gap_delta: i64,
    rounding_and_clamp_residual: i64,
    rounded_gap_delta: i64,
    rounded_gap_left: i64,
    rounded_gap_right: i64,
    unrounded_endpoint_dots: [[i64; 2]; 2],
    clamped_endpoint_scores: [[bool; 2]; 2],
}

fn verify_round(dot: i64, observed: i32, shift: u32) -> Result<bool> {
    let unbounded = (dot + (1_i64 << (shift - 1))) >> shift;
    let bounded = unbounded.clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP));
    if bounded != i64::from(observed) {
        return Err(format!(
            "physical score {observed} disagrees with round/clamp of dot {dot}: {bounded}"
        )
        .into());
    }
    Ok(bounded != unbounded)
}

fn decompose(left: Endpoint<'_>, right: Endpoint<'_>, shift: u32) -> Result<Parts> {
    let len = left.hidden.len();
    if [
        right.hidden.len(),
        left.candidate.len(),
        left.rival.len(),
        right.candidate.len(),
        right.rival.len(),
    ]
    .iter()
    .any(|&n| n != len)
    {
        return Err("endpoint dot shape mismatch".into());
    }

    let mut hidden_change = 0i64;
    let mut row_movement = 0i64;
    let mut dots = [[0i64; 2]; 2];
    for index in 0..len {
        let h0 = i64::from(left.hidden[index]);
        let h1 = i64::from(right.hidden[index]);
        let a0 = i64::from(left.candidate[index]);
        let a1 = i64::from(right.candidate[index]);
        let b0 = i64::from(left.rival[index]);
        let b1 = i64::from(right.rival[index]);
        let w0 = a0 - b0;
        let w1 = a1 - b1;
        hidden_change += (h1 - h0) * (w1 + w0);
        row_movement += (h1 + h0) * (w1 - w0);
        dots[0][0] += h0 * a0;
        dots[0][1] += h0 * b0;
        dots[1][0] += h1 * a1;
        dots[1][1] += h1 * b1;
    }
    let clamped = [
        [
            verify_round(dots[0][0], left.observed_candidate, shift)?,
            verify_round(dots[0][1], left.observed_rival, shift)?,
        ],
        [
            verify_round(dots[1][0], right.observed_candidate, shift)?,
            verify_round(dots[1][1], right.observed_rival, shift)?,
        ],
    ];
    let exact = 2 * ((dots[1][0] - dots[1][1]) - (dots[0][0] - dots[0][1]));
    if exact != hidden_change + row_movement {
        return Err("integer endpoint decomposition did not conserve the exact gap change".into());
    }
    let gap_left = i64::from(left.observed_candidate) - i64::from(left.observed_rival);
    let gap_right = i64::from(right.observed_candidate) - i64::from(right.observed_rival);
    let rounded = gap_right - gap_left;
    Ok(Parts {
        hidden_change,
        row_movement,
        exact_dot_gap_delta: exact,
        rounding_and_clamp_residual: rounded * (1_i64 << (shift + 1)) - exact,
        rounded_gap_delta: rounded,
        rounded_gap_left: gap_left,
        rounded_gap_right: gap_right,
        unrounded_endpoint_dots: dots,
        clamped_endpoint_scores: clamped,
    })
}

fn strongest_other(
    scores: &[i32],
    candidate: usize,
    admissible: impl Fn(usize) -> bool,
) -> Result<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|&(slot, _)| slot != candidate && admissible(slot))
        .max_by_key(|&(slot, &score)| (score, std::cmp::Reverse(slot)))
        .map(|(slot, _)| slot)
        .ok_or_else(|| "no admissible right-endpoint rival".into())
}

fn weight_row(weights: &[i16], row: usize, dim: usize) -> &[i16] {
    &weights[row * dim..(row + 1) * dim]
}

fn make_endpoint<'a>(
    h: &'a [i32],
    weights: &'a [i16],
    scores: &[i32],
    candidate: usize,
    rival: usize,
    offset: usize,
) -> Endpoint<'a> {
    Endpoint {
        hidden: h,
        candidate: weight_row(weights, offset + candidate, h.len()),
        rival: weight_row(weights, offset + rival, h.len()),
        observed_candidate: scores[candidate],
        observed_rival: scores[rival],
    }
}

#[derive(Serialize)]
struct Attribution {
    id: u64,
    rival_slot: usize,
    rival_original_id: Option<u16>,
    parts: Parts,
}

fn signs(values: impl Iterator<Item = i64>) -> Value {
    let mut values: Vec<_> = values.collect();
    values.sort_unstable();
    let n = values.len();
    let median = if n == 0 {
        None
    } else {
        Some(values[(n - 1) / 2] + values[n / 2])
    };
    json!({
        "signed_total":values.iter().sum::<i64>(),
        "absolute_total":values.iter().map(|v| v.unsigned_abs()).sum::<u64>(),
        "max_absolute":values.iter().map(|v| v.unsigned_abs()).max(),
        "median_times_two":median,
        "positive":values.iter().filter(|&&v| v>0).count(),
        "zero":values.iter().filter(|&&v| v==0).count(),
        "negative":values.iter().filter(|&&v| v<0).count(),
    })
}

fn summarize(rows: &[Attribution]) -> Value {
    let mut clamped = [[0usize; 2]; 2];
    for row in rows {
        for (counts, flags) in clamped.iter_mut().zip(row.parts.clamped_endpoint_scores) {
            for (count, flag) in counts.iter_mut().zip(flags) {
                *count += usize::from(flag);
            }
        }
    }
    let abs_compare = |row: &Attribution| {
        row.parts
            .hidden_change
            .unsigned_abs()
            .cmp(&row.parts.row_movement.unsigned_abs())
    };
    json!({
        "paired_rows":rows.len(),
        "hidden_change_numerator":signs(rows.iter().map(|r|r.parts.hidden_change)),
        "output_rows_movement_numerator":signs(rows.iter().map(|r|r.parts.row_movement)),
        "exact_gap_delta_numerator":signs(rows.iter().map(|r|r.parts.exact_dot_gap_delta)),
        "rounding_and_clamp_residual_numerator":signs(rows.iter().map(|r|r.parts.rounding_and_clamp_residual)),
        "observed_rounded_gap_delta_q15_words":signs(rows.iter().map(|r|r.parts.rounded_gap_delta)),
        "observed_left_gap_q15_words":signs(rows.iter().map(|r|r.parts.rounded_gap_left)),
        "observed_right_gap_q15_words":signs(rows.iter().map(|r|r.parts.rounded_gap_right)),
        "absolute_term_dominance":{
            "hidden_change":rows.iter().filter(|r|abs_compare(r).is_gt()).count(),
            "output_row_movement":rows.iter().filter(|r|abs_compare(r).is_lt()).count(),
            "equal":rows.iter().filter(|r|abs_compare(r).is_eq()).count(),
        },
        "four_endpoint_scores_verified_per_row":true,
        "clamped_scores_left_and_right_candidate_then_rival":clamped,
    })
}

fn compare(
    left_model: &ButterflyV3Model,
    right_model: &ButterflyV3Model,
    left: &[Row],
    right: &[Row],
) -> Result<Value> {
    let shift = ButterflyV3Model::head_dot_shift(left_model.cfg.dim);
    let admissible = |slot: usize| {
        let id = usize::from(right_model.vocab_inv[slot]);
        id != 0 && id < right_model.cfg.vocab
    };
    let mut coarse = Vec::with_capacity(left.len());
    let mut fine = Vec::new();
    let mut availability = BTreeMap::new();
    for (left, right) in left.iter().zip(right) {
        if left.id != right.id || left.target_id != right.target_id {
            return Err("physical rows were not matched on TRAIN document and target".into());
        }
        let result = (|| -> Result<()> {
            let rival = strongest_other(&right.cuda_raw_coarse_q15, CELL, |cell| {
                let start = cell * HIER_FINE;
                (start..start + HIER_FINE).any(admissible)
            })?;
            let parts = decompose(
                make_endpoint(
                    &left.hidden_q15,
                    &left_model.coarse_centroids,
                    &left.cuda_raw_coarse_q15,
                    CELL,
                    rival,
                    0,
                ),
                make_endpoint(
                    &right.hidden_q15,
                    &right_model.coarse_centroids,
                    &right.cuda_raw_coarse_q15,
                    CELL,
                    rival,
                    0,
                ),
                shift,
            )?;
            coarse.push(Attribution {
                id: left.id,
                rival_slot: rival,
                rival_original_id: None,
                parts,
            });
            let (left_source, left_fine) = left.fine()?;
            let (right_source, right_fine) = right.fine()?;
            *availability
                .entry(format!("{left_source} / {right_source}"))
                .or_insert(0usize) += 1;
            if let (Some(left_fine), Some(right_fine)) = (left_fine, right_fine) {
                let rival = strongest_other(right_fine, QUOTE_SLOT % HIER_FINE, |slot| {
                    admissible(CELL * HIER_FINE + slot)
                })?;
                let parts = decompose(
                    make_endpoint(
                        &left.hidden_q15,
                        &left_model.embeddings,
                        left_fine,
                        0,
                        rival,
                        CELL * HIER_FINE,
                    ),
                    make_endpoint(
                        &right.hidden_q15,
                        &right_model.embeddings,
                        right_fine,
                        0,
                        rival,
                        CELL * HIER_FINE,
                    ),
                    shift,
                )?;
                fine.push(Attribution {
                    id: left.id,
                    rival_slot: rival,
                    rival_original_id: Some(right_model.vocab_inv[CELL * HIER_FINE + rival]),
                    parts,
                });
            }
            Ok(())
        })();
        result.map_err(|error| format!("TRAIN row {}: {error}", left.id))?;
    }
    Ok(json!({
        "coarse_cell_3":summarize(&coarse),
        "fine_id_1228":summarize(&fine),
        "fine_left_right_physical_cell_3_visibility":availability,
        "fine_excluded_unobserved_at_either_endpoint":left.len()-fine.len(),
        "coarse_rows":coarse, "fine_rows":fine,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [receipt, frozen, reset, preserved] = args.as_slice() else {
        return Err("usage: score_change PHYSICAL_TRAIN_JSONL FROZEN RESET PRESERVED".into());
    };
    let models = inventory::checkpoints([frozen, reset, preserved])?;
    for model in &models {
        if model.cfg != models[0].cfg
            || model.vocab_perm != models[0].vocab_perm
            || model.vocab_inv[QUOTE_SLOT] != QUOTE_ID
        {
            return Err("checkpoint shape or geometry identity mismatch".into());
        }
    }
    let bytes = fs::read(receipt)?;
    let mut input = BufReader::new(bytes.as_slice()).lines();
    let mut arms = Vec::with_capacity(3);
    for (identity, model) in inventory::IDENTITIES.into_iter().zip(&models) {
        let mut arm = Vec::with_capacity(TRAIN_ROWS);
        let mut seen = BTreeSet::new();
        for index in 0..TRAIN_ROWS {
            let row: Row =
                serde_json::from_str(&input.next().ok_or("incomplete physical receipt")??)?;
            row.check(model, identity, index)?;
            if !seen.insert(row.id) {
                return Err("duplicate TRAIN row in physical receipt".into());
            }
            arm.push(row);
        }
        arms.push(arm);
    }
    if input.next().is_some() {
        return Err("unexpected extra rows in physical receipt".into());
    }
    let shift = ButterflyV3Model::head_dot_shift(models[0].cfg.dim);
    let mut comparisons = BTreeMap::new();
    for (name, l, r) in [
        ("frozen_to_reset", 0, 1),
        ("frozen_to_preserved", 0, 2),
        ("reset_to_preserved", 1, 2),
    ] {
        comparisons.insert(name, compare(&models[l], &models[r], &arms[l], &arms[r])?);
    }
    serde_json::to_writer(
        io::stdout().lock(),
        &json!({
            "kind":"supplemental post-registration exact paired endpoint algebra",
            "input_train_physical_jsonl_sha256":hex::encode(Sha256::digest(&bytes)),
            "checkpoints":inventory::IDENTITIES.map(|(name, sha256, step)| json!({"name":name,"sha256":sha256,"step":step})),
            "cubin_sha256":CUBIN,"train_sha256":TRAIN,
            "unit":"signed exact raw-dot numerator with 2^(head_dot_shift+1) numerator units per unrounded Q15 score word",
            "head_dot_shift":shift, "numerator_units_per_unrounded_q15_score_word":1_u64 << (shift + 1),
            "rival_rule":"strongest admissible right-endpoint other by physical score, first slot on ties, kept fixed at left endpoint",
            "hidden_scope":"net change in the physically captured post-head-projection hidden vector, including effects from input embeddings and all preceding operations",
            "weight_scope":"net movement of the candidate and fixed rival row weights in the scored head",
            "comparisons":comparisons,
        }),
    )?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_integer_conservation_and_rounding() -> Result<()> {
        let result = decompose(
            Endpoint {
                hidden: &[3],
                candidate: &[2],
                rival: &[0],
                observed_candidate: 2,
                observed_rival: 0,
            },
            Endpoint {
                hidden: &[5],
                candidate: &[3],
                rival: &[1],
                observed_candidate: 4,
                observed_rival: 1,
            },
            2,
        )?;
        assert_eq!(result.hidden_change, 8);
        assert_eq!(result.row_movement, 0);
        assert_eq!(result.exact_dot_gap_delta, 8);
        assert_eq!(result.rounded_gap_delta, 1);
        assert_eq!(result.rounding_and_clamp_residual, 0);
        Ok(())
    }

    #[test]
    fn movement_residual_and_mismatch() -> Result<()> {
        let make = |candidate, observed| Endpoint {
            hidden: &[1],
            candidate,
            rival: &[0],
            observed_candidate: observed,
            observed_rival: 0,
        };
        let result = decompose(make(&[0], 0), make(&[1], 0), 2)?;
        assert_eq!(result.hidden_change, 0);
        assert_eq!(result.row_movement, 2);
        assert_eq!(result.rounding_and_clamp_residual, -2);
        assert!(decompose(make(&[0], 0), make(&[1], 1), 2).is_err());
        assert!(verify_round((i64::from(ACT_CLAMP) + 9) << 2, ACT_CLAMP, 2)?);
        Ok(())
    }

    #[test]
    fn eligibility_ties_and_exact_even_median() -> Result<()> {
        assert_eq!(strongest_other(&[9, 4, 7, 7, 8], 0, |slot| slot != 4)?, 2);
        assert_eq!(
            signs([-3, 0, 1, 4].into_iter()),
            json!({
                "signed_total":2,"absolute_total":8,"max_absolute":4,
                "median_times_two":1,"positive":2,"zero":1,"negative":1
            })
        );
        Ok(())
    }
}
