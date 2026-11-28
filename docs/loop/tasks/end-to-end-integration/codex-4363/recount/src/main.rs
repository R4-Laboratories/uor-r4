use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::{error::Error, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TRAIN: &str = "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a";
const CUBIN: &str = "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713";
const HISTORICAL: &str = "dcc3019546fd1fe6e272aa71a64d694db2754b946bc82638f64b19671e47d04d";
const ARMS: [(&str, &str); 3] = [
    (
        "frozen",
        "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589",
    ),
    (
        "reset",
        "3656e0575bd9e590e63305000e6f03ca010e1a663d6ac45d9061961f9a356bc2",
    ),
    (
        "preserved",
        "8e0721cc8b989f12af9c80b88dfac53c3a33701e6a3e8e7d2d26d2dec7fd1340",
    ),
];
const WIDTH: usize = 64;
const VOCAB: usize = WIDTH * WIDTH;
const N: usize = 256;
const QUOTE: usize = 1228;
const CLAMP: i64 = 1 << 22;
const ONE: i64 = 1 << 15;

fn ensure(ok: bool, why: impl FnOnce() -> String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(why().into())
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn word_digest(words: impl IntoIterator<Item = i32>) -> String {
    let mut sha = Sha256::new();
    for word in words {
        sha.update(word.to_le_bytes());
    }
    format!("{:x}", sha.finalize())
}

fn is_digest(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn input(path: &str, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    ensure(digest(&bytes) == expected, || {
        format!("SHA-256 mismatch for {path}")
    })?;
    Ok(bytes)
}

fn rows(bytes: &[u8]) -> Result<Vec<Value>> {
    std::str::from_utf8(bytes)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn number<T: TryFrom<u64>>(value: &Value, field: &str) -> Result<T> {
    value[field]
        .as_u64()
        .and_then(|n| T::try_from(n).ok())
        .ok_or_else(|| format!("expected unsigned integer in {field}").into())
}

fn words(
    value: &Value,
    field: &str,
    expected: usize,
    domain: std::ops::RangeInclusive<i64>,
) -> Result<Vec<i32>> {
    let values = value[field]
        .as_array()
        .ok_or_else(|| format!("missing {field}"))?;
    ensure(values.len() == expected, || {
        format!("wrong length for {field}")
    })?;
    values
        .iter()
        .map(|v| {
            v.as_i64()
                .filter(|n| domain.contains(n))
                .map(|n| n as i32)
                .ok_or_else(|| format!("invalid native word in {field}").into())
        })
        .collect()
}

fn scores(value: &Value, field: &str, expected: usize) -> Result<Vec<i32>> {
    words(value, field, expected, -CLAMP..=CLAMP)
}

struct Checkpoint {
    original_to_slot: Vec<usize>,
    slot_to_original: Vec<usize>,
    dimension: usize,
    weights: Vec<i16>,
}

impl Checkpoint {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure(
            bytes.len() == 30_270_020 && &bytes[..8] == b"UORBFY03",
            || "bad artifact envelope".into(),
        )?;
        let word = |i| -> Result<u32> { Ok(u32::from_le_bytes(bytes[i..i + 4].try_into()?)) };
        ensure(
            word(8)? == 1 && word(12)? as usize == VOCAB && word(36)? == 1 && word(40)? == 64,
            || "unexpected format, vocabulary or loss".into(),
        )?;
        let dimension = word(16)? as usize;
        ensure(
            dimension == 512 && word(20)? == 6 && word(28)? == 64,
            || "unexpected head or PKM dimensions".into(),
        )?;
        let permutation_end = 60 + 2 * VOCAB;
        let original_to_slot = bytes[60..permutation_end]
            .chunks_exact(2)
            .map(|b| usize::from(u16::from_le_bytes([b[0], b[1]])))
            .collect::<Vec<_>>();
        let mut slot_to_original = vec![VOCAB; VOCAB];
        for (id, &slot) in original_to_slot.iter().enumerate() {
            ensure(slot < VOCAB && slot_to_original[slot] == VOCAB, || {
                "invalid permutation".into()
            })?;
            slot_to_original[slot] = id;
        }
        let head_end = permutation_end + 2 * (VOCAB + WIDTH) * dimension;
        let weights = bytes[permutation_end..head_end]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok(Self {
            original_to_slot,
            slot_to_original,
            dimension,
            weights,
        })
    }

    fn head(&self, hidden: &[i32], start_row: usize) -> Vec<i32> {
        let shift = 15 + self.dimension.trailing_zeros().div_ceil(2);
        self.weights
            .chunks_exact(self.dimension)
            .skip(start_row)
            .take(WIDTH)
            .map(|row| {
                let dot = row
                    .iter()
                    .zip(hidden)
                    .map(|(&a, &b)| i64::from(a) * i64::from(b))
                    .sum::<i64>();
                ((dot + (1 << (shift - 1))) >> shift).clamp(-CLAMP, CLAMP) as i32
            })
            .collect()
    }

    fn eligible(&self, cell: usize, fine: usize) -> bool {
        self.slot_to_original[cell * WIDTH + fine] != 0
    }
}

fn rank(words: &[i32], valid: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut eligible: Vec<_> = (0..WIDTH).filter(|&position| valid(position)).collect();
    eligible.sort_by(|&a, &b| words[b].cmp(&words[a]).then_with(|| a.cmp(&b)));
    eligible
}

fn gap(words: &[i32], order: &[usize]) -> i64 {
    i64::from(words[order[0]]) - i64::from(words[order[1]])
}

fn target_gap(words: &[i32], order: &[usize], target: usize) -> i64 {
    let best_alternative = if order[0] == target {
        order[1]
    } else {
        order[0]
    };
    i64::from(words[target]) - i64::from(words[best_alternative])
}

fn compare_election(record: &Value, name: &str, words: &[i32], order: &[usize]) -> Result<()> {
    let winner = order[0];
    let runner = order[1];
    let maxima = order
        .iter()
        .copied()
        .take_while(|&i| words[i] == words[winner])
        .collect::<Vec<_>>();
    let expected = json!({"winner":winner,"runner_up":runner,"winner_score":words[winner],
        "runner_up_score":words[runner],"margin":gap(words,order),"equal_maxima":maxima});
    ensure(record[name] == expected, || {
        format!("{name} disagrees with independent election")
    })
}

fn local_derivative(words: &[i32], target: usize) -> (u64, Vec<i64>, bool) {
    let mut sum = 0;
    let mut saturated = false;
    let derivatives = words
        .iter()
        .enumerate()
        .map(|(i, &word)| {
            let (residual, weight) = if i == target {
                (i64::from(word) - ONE, 64)
            } else {
                (i64::from(word), 1)
            };
            sum += weight * ((residual * residual) >> 15);
            let raw = weight * residual;
            let clamped = raw.clamp(-CLAMP, CLAMP);
            saturated |= clamped != raw;
            clamped
        })
        .collect();
    ((sum / 2) as u64, derivatives, saturated)
}

#[derive(Clone, Copy)]
struct Reading {
    winner: usize,
    target: usize,
    admitted: bool,
    target_rank: usize,
    coarse_margin: i64,
    fine_margin: i64,
    coarse_target_gap: i64,
    fine_target_gap: i64,
    quote_cell_coarse_gap: i64,
    quote_gap_in_elected_cell: Option<i64>,
    quote_gap_any_observed_cell: Option<i64>,
    saturated_heads: [bool; 2],
}

fn verify(
    record: &Value,
    example: &Value,
    checkpoint: &Checkpoint,
    arm: (&str, &str),
    document: usize,
) -> Result<Reading> {
    let prefix = example["prefix"].as_array().ok_or("TRAIN prefix missing")?;
    let expected = example["expected"]
        .as_array()
        .ok_or("TRAIN targets missing")?;
    let target = expected
        .first()
        .and_then(Value::as_u64)
        .ok_or("TRAIN target missing")? as usize;
    let input_id = prefix.last().ok_or("TRAIN prefix empty")?;
    ensure(
        record["checkpoint"] == arm.0
            && record["checkpoint_sha256"] == arm.1
            && record["train_sha256"] == TRAIN
            && record["cubin_sha256"] == CUBIN
            && record["checkpoint_step"] == if arm.0 == "frozen" { 3333 } else { 3397 },
        || "capture identity mismatch".into(),
    )?;
    ensure(
        record["id"] == example["id"]
            && record["stratum"] == example["stratum"]
            && record["target_id"] == target
            && record["input_original_id"] == *input_id
            && record["target_first_label_document_offset"] == prefix.len()
            && record["historical_chunk"] == prefix.len() / 257
            && record["first_target_mask"] == 1
            && record["training_document_index"] == document
            && record["historical_update_index"] == document / 4
            && record["historical_train_step_before"] == 3333 + document / 4
            && record["historical_reset_label_chunk_offset"] == prefix.len() % 257
            && record["reference_free_transition_match"] == true,
        || "capture coordinates disagree with pinned TRAIN".into(),
    )?;

    let coarse = scores(record, "cuda_raw_coarse_q15", WIDTH)?;
    let elected = scores(record, "cuda_raw_elected_fine_q15", WIDTH)?;
    let gold = scores(record, "cuda_raw_target_fine_q15", WIDTH)?;
    let coarse_order = rank(&coarse, |cell| {
        (0..WIDTH).any(|fine| checkpoint.eligible(cell, fine))
    });
    let elected_cell = coarse_order[0];
    let gold_slot = checkpoint.original_to_slot[target];
    let gold_cell = gold_slot / WIDTH;
    let gold_fine = gold_slot % WIDTH;
    let fine_order = rank(&elected, |fine| checkpoint.eligible(elected_cell, fine));
    let gold_order = rank(&gold, |fine| checkpoint.eligible(gold_cell, fine));
    let elected_slot = elected_cell * WIDTH + fine_order[0];
    let winner = checkpoint.slot_to_original[elected_slot];
    let admitted = elected_cell == gold_cell;
    let gold_rank = 1 + gold_order
        .iter()
        .position(|&fine| fine == gold_fine)
        .ok_or("gold masked")?;
    let gold_competitor = if gold_order[0] == gold_fine {
        gold_order[1]
    } else {
        gold_order[0]
    };
    ensure(!admitted || elected == gold, || {
        "same-cell raw words disagree".into()
    })?;
    ensure(
        record["target_slot"] == gold_slot
            && record["target_cell"] == gold_cell
            && record["target_is_admitted"] == admitted
            && record["actual_first_global_slot"] == elected_slot
            && record["actual_first_id"] == winner
            && record["target_local_rank_in_true_cell"] == gold_rank
            && record["target_coarse_raw_score_q15"] == coarse[gold_cell]
            && record["target_fine_raw_score_q15"] == gold[gold_fine]
            && record["fine_runner_up_original_id"]
                == checkpoint.slot_to_original[elected_cell * WIDTH + fine_order[1]]
            && record["target_cell_competitor_original_id"]
                == checkpoint.slot_to_original[gold_cell * WIDTH + gold_competitor],
        || {
            "recorded chosen ID, admission or target rank disagrees with raw device scores and artifact".into()
        },
    )?;
    compare_election(record, "coarse", &coarse, &coarse_order)?;
    compare_election(record, "elected_fine", &elected, &fine_order)?;
    compare_election(record, "forced_target_cell", &gold, &gold_order)?;
    let fine_maxima_ids = fine_order
        .iter()
        .copied()
        .take_while(|&f| elected[f] == elected[fine_order[0]])
        .map(|f| checkpoint.slot_to_original[elected_cell * WIDTH + f])
        .collect::<Vec<_>>();
    ensure(
        record["fine_equal_maxima_original_ids"] == json!(fine_maxima_ids),
        || "fine maximal IDs disagree with checkpoint mapping".into(),
    )?;

    let (coarse_loss, coarse_dz, coarse_saturated) = local_derivative(&coarse, gold_cell);
    let (fine_loss, fine_dz, fine_saturated) = local_derivative(&gold, gold_fine);
    let fine_if_admitted = |slot| admitted.then_some(fine_dz[slot]);
    let gradients = &record["score_gradients"];
    ensure(gradients["realized_update_measured"] == false, || {
        "invalid optimizer attribution".into()
    })?;
    let fields = [
        ("coarse_loss_q15", json!(coarse_loss)),
        ("fine_loss_q15", json!(fine_loss)),
        ("target_coarse_dz_q15", json!(coarse_dz[gold_cell])),
        ("elected_coarse_dz_q15", json!(coarse_dz[elected_cell])),
        ("coarse_runner_up_dz_q15", json!(coarse_dz[coarse_order[1]])),
        ("target_fine_dz_q15", json!(fine_dz[gold_fine])),
        (
            "target_cell_competitor_dz_q15",
            json!(fine_dz[gold_competitor]),
        ),
        (
            "elected_fine_dz_q15",
            json!(fine_if_admitted(fine_order[0])),
        ),
        (
            "fine_runner_up_dz_q15",
            json!(fine_if_admitted(fine_order[1])),
        ),
        ("elected_fine_supervised_head", json!(admitted)),
    ];
    for (field, expected) in fields {
        ensure(gradients.get(field) == Some(&expected), || {
            format!("local score derivative mismatch in {field}")
        })?;
    }

    let hidden = words(
        record,
        "hidden_q15",
        checkpoint.dimension,
        i64::from(i32::MIN)..=i64::from(i32::MAX),
    )?;
    let pkm = words(record, "pkm_slots", 6 * 4, 0..=4095)?;
    ensure(
        record["hidden_sha256"] == word_digest(hidden.iter().copied())
            && coarse == checkpoint.head(&hidden, VOCAB)
            && elected == checkpoint.head(&hidden, elected_cell * WIDTH)
            && gold == checkpoint.head(&hidden, gold_cell * WIDTH),
        || "head-dot replay or PKM range mismatch against retained words and checkpoint".into(),
    )?;
    let rich_words = [elected_slot as i32, elected[fine_order[0]]]
        .into_iter()
        .chain(hidden)
        .chain(coarse.iter().copied())
        .chain(gold.iter().copied())
        .chain(elected.iter().copied())
        .chain(pkm);
    let blind = &record["blind_signature"];
    let rich = &record["rich_signature"];
    let sham = &record["sham_signature"];
    ensure(
        record["blind_sham_complete_match"] == true
            && record["physical_first_step_launches"] == 3
            && is_digest(&record["prefill_native_state_sha256"])
            && blind == sham
            && is_digest(&blind["native_transition_sha256"])
            && blind["native_transition_sha256"] == rich["native_transition_sha256"]
            && is_digest(&blind["complete_trace_i32le_sha256"])
            && rich["complete_trace_i32le_sha256"] == word_digest(rich_words)
            && is_digest(&blind["diagnostic_mass_q48_le_sha256"])
            && is_digest(&rich["diagnostic_mass_q48_le_sha256"]),
        || "independent signature and rich raw trace reconciliation failed".into(),
    )?;
    let quote_slot = checkpoint.original_to_slot[QUOTE];
    ensure(quote_slot == 3 * WIDTH, || {
        "unexpected quote vocabulary slot".into()
    })?;
    Ok(Reading {
        winner,
        target,
        admitted,
        target_rank: gold_rank,
        coarse_margin: gap(&coarse, &coarse_order),
        fine_margin: gap(&elected, &fine_order),
        coarse_target_gap: target_gap(&coarse, &coarse_order, gold_cell),
        fine_target_gap: target_gap(&gold, &gold_order, gold_fine),
        quote_cell_coarse_gap: target_gap(&coarse, &coarse_order, quote_slot / WIDTH),
        quote_gap_in_elected_cell: (elected_cell == quote_slot / WIDTH)
            .then(|| target_gap(&elected, &fine_order, quote_slot % WIDTH)),
        quote_gap_any_observed_cell: (elected_cell == quote_slot / WIDTH)
            .then(|| target_gap(&elected, &fine_order, quote_slot % WIDTH))
            .or_else(|| {
                (gold_cell == quote_slot / WIDTH)
                    .then(|| target_gap(&gold, &gold_order, quote_slot % WIDTH))
            }),
        saturated_heads: [coarse_saturated, fine_saturated],
    })
}

fn distribution(mut values: Vec<i64>) -> Value {
    values.sort_unstable();
    let n = values.len();
    if n == 0 {
        return Value::Null;
    }
    json!({"min":values[0],"median_lower":values[(n-1)/2],"median_upper":values[n/2],"max":values[n-1]})
}

fn describe(records: &[Reading]) -> Value {
    let margins = |fine| {
        let values = records
            .iter()
            .map(|r| if fine { r.fine_margin } else { r.coarse_margin })
            .collect::<Vec<_>>();
        json!({"tie":values.iter().filter(|&&x| x==0).count(),
            "strict_one_word":values.iter().filter(|&&x| x==1).count(),
            "strict_larger":values.iter().filter(|&&x| x>1).count(),"distribution":distribution(values)})
    };
    let mut first_ids = BTreeMap::new();
    for r in records {
        *first_ids.entry(r.winner).or_insert(0usize) += 1;
    }
    json!({"rows":records.len(),"first_id_counts":first_ids,
        "first_id_correct":records.iter().filter(|r| r.winner==r.target).count(),
        "gold_cell_admissions":records.iter().filter(|r| r.admitted).count(),
        "conditional_gold_rank_one":records.iter().filter(|r| r.target_rank==1).count(),
        "elected_fine_supervised":records.iter().filter(|r| r.admitted).count(),
        "coarse_margin":margins(false),"fine_margin":margins(true),
        "coarse_target_gap":distribution(records.iter().map(|r|r.coarse_target_gap).collect()),
        "fine_target_gap":distribution(records.iter().map(|r|r.fine_target_gap).collect()),
        "elected_cell3":records.iter().filter(|r|r.quote_gap_in_elected_cell.is_some()).count(),
        "cell3_coarse_gap":distribution(records.iter().map(|r|r.quote_cell_coarse_gap).collect()),
        "coarse_local_dz_saturation_rows":records.iter().filter(|r|r.saturated_heads[0]).count(),
        "gold_fine_local_dz_saturation_rows":records.iter().filter(|r|r.saturated_heads[1]).count(),
        "raw_head_dot_replayed_rows":records.len()})
}

fn paired(frozen: &[Reading], trained: &[Reading]) -> Value {
    let pairs: Vec<_> = frozen.iter().zip(trained).collect();
    let matched_quotes = |gap: fn(&Reading) -> Option<i64>| {
        let joint: Vec<_> = pairs
            .iter()
            .filter_map(|(a, b)| Some((gap(a)?, gap(b)?)))
            .collect();
        json!({"rows":joint.len(),
            "frozen":distribution(joint.iter().map(|&(a,_)|a).collect()),
            "trained":distribution(joint.iter().map(|&(_,b)|b).collect()),
            "delta":distribution(joint.iter().map(|&(a,b)|b-a).collect()),
            "gained":joint.iter().filter(|&&(a,b)|b>a).count(),
            "unchanged":joint.iter().filter(|&&(a,b)|b==a).count(),
            "lost":joint.iter().filter(|&&(a,b)|b<a).count()})
    };
    let mut quote_origin = BTreeMap::new();
    let mut trained_quote_matched_frozen_gap = Vec::new();
    for (before, after) in &pairs {
        if after.winner != QUOTE {
            continue;
        }
        let origin = match before.quote_gap_in_elected_cell {
            None => "frozen_other_cell",
            Some(gap) if gap < 0 => "frozen_quote_lost_on_score",
            Some(0) if before.winner == QUOTE => "frozen_quote_won_tie",
            Some(0) => "frozen_quote_lost_tie",
            Some(_) => "frozen_quote_strict",
        };
        *quote_origin.entry(origin).or_insert(0usize) += 1;
        if let Some(gap) = before.quote_gap_in_elected_cell {
            trained_quote_matched_frozen_gap.push(after.fine_margin - gap);
        }
    }
    let gains_losses = |f: fn(&Reading) -> bool| {
        json!({"gains":pairs.iter().filter(|(a,b)|!f(a)&&f(b)).count(),
               "losses":pairs.iter().filter(|(a,b)|f(a)&&!f(b)).count()})
    };
    json!({"changed_first_ids":pairs.iter().filter(|(a,b)|a.winner!=b.winner).count(),
        "first_id_correct":gains_losses(|r|r.winner==r.target),
        "gold_cell_admission":gains_losses(|r|r.admitted),
        "matched_id1228_gap_when_both_elected_cell3":matched_quotes(|r|r.quote_gap_in_elected_cell),
        "matched_id1228_gap_when_both_traces_record_cell3":matched_quotes(|r|r.quote_gap_any_observed_cell),
        "trained_quote_frozen_origin":quote_origin,
        "trained_quote_gap_delta_when_frozen_admitted_quote_cell":distribution(trained_quote_matched_frozen_gap),
        "coarse_gold_gap_delta":distribution(pairs.iter().map(|(a,b)|b.coarse_target_gap-a.coarse_target_gap).collect()),
        "fine_gold_gap_delta":distribution(pairs.iter().map(|(a,b)|b.fine_target_gap-a.fine_target_gap).collect()),
        "cell3_coarse_admission":gains_losses(|r|r.quote_gap_in_elected_cell.is_some()),
        "cell3_coarse_gap_delta":distribution(pairs.iter().map(|(a,b)|b.quote_cell_coarse_gap-a.quote_cell_coarse_gap).collect())})
}

fn verify_finite(path: &str) -> Result<String> {
    let bytes = fs::read(path)?;
    let receipt = rows(&bytes)?;
    let [calibration, complete] = receipt.as_slice() else {
        return Err("expected calibration and COMPLETE receipt lines".into());
    };
    let historical = &calibration["historical"];
    let physical = &calibration["physical"];
    let host = &calibration["host"];
    ensure(
        complete["verdict"] == "COMPLETE"
            && complete["rows"] == 3 * N
            && historical["historical"] == "PASS"
            && historical["context_len"] == 2048
            && historical["greedy_ids_checked"] == 128
            && historical["batched_greedy_ids_checked"] == 128
            && historical["q48_masses_checked"] == 128
            && historical["artifact_sha256"] == ARMS[0].1
            && historical["cubin_sha256"] == CUBIN
            && historical["fixture_sha256"] == HISTORICAL
            && historical["gpu_cc"] == json!([8, 9])
            && historical["gpu"] == "NVIDIA GeForce RTX 4090"
            && physical["finite_device"] == "PASS"
            && physical["cubin_sha256"] == CUBIN
            && physical["gpu_cc"] == json!([8, 9])
            && physical["gpu"] == historical["gpu"]
            && host["finite_host"] == "PASS",
        || "physical, historical, host or completion receipt mismatch".into(),
    )?;
    let physical_cases = physical["physical_score_fixtures"]
        .as_array()
        .ok_or("missing physical cases")?;
    let host_cases = host["fixtures"].as_array().ok_or("missing host cases")?;
    let fixtures = [
        ("exact_tie", [0, 0, -1], false, 0, 1, vec![0, 1]),
        ("slot_zero_ahead_by_one", [1, 0, -1], false, 0, 1, vec![0]),
        ("slot_zero_behind_by_one", [0, 1, -1], false, 1, 0, vec![1]),
        ("strict_winner_masked", [2, 1, 0], true, 1, 2, vec![1]),
    ];
    ensure(
        physical_cases.len() == fixtures.len() && host_cases.len() == fixtures.len(),
        || "finite fixture count mismatch".into(),
    )?;
    for ((physical_case, host_case), (name, top, masked, winner, runner, maxima)) in
        physical_cases.iter().zip(host_cases).zip(fixtures)
    {
        let mut expected = [-1; WIDTH];
        expected[..3].copy_from_slice(&top);
        let ordered = rank(&expected, |slot| !masked || slot != 0);
        ensure(
            ordered[0] == winner
                && ordered[1] == runner
                && ordered
                    .iter()
                    .copied()
                    .take_while(|&i| expected[i] == expected[winner])
                    .collect::<Vec<_>>()
                    == maxima
                && physical_case["fixture"] == name
                && host_case["fixture"] == name
                && is_digest(&physical_case["artifact_sha256"])
                && physical_case["reference_free_transition_match"] == true
                && physical_case["rust_native_transition_match"] == true
                && physical_case["slot_zero_admitted_coarse"] == !masked
                && physical_case["slot_zero_admitted_fine"] == !masked
                && scores(physical_case, "physical_coarse_raw_q15", WIDTH)? == expected
                && scores(physical_case, "physical_fine_raw_q15", WIDTH)? == expected,
            || format!("physical fixture mismatch in {name}"),
        )?;
        compare_election(physical_case, "coarse", &expected, &ordered)?;
        compare_election(physical_case, "fine", &expected, &ordered)?;
        compare_election(host_case, "election", &expected, &ordered)?;
    }
    let mut three_way = [-1; WIDTH];
    for i in [0, 1, 9] {
        three_way[i] = 17;
    }
    compare_election(
        host,
        "three_way_tie",
        &three_way,
        &rank(&three_way, |_| true),
    )?;
    Ok(digest(&bytes))
}

fn entry() -> Result<Value> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [tape, cubin, historical, frozen, reset, preserved, capture, instrument] = args.as_slice()
    else {
        return Err(
            "usage: recount TRAIN CUBIN HISTORICAL FROZEN RESET PRESERVED PHYSICAL_ROWS INSTRUMENT"
                .into(),
        );
    };
    let _ = input(cubin, CUBIN)?;
    let _ = input(historical, HISTORICAL)?;
    let calibration_sha256 = verify_finite(instrument)?;
    let examples = rows(&input(tape, TRAIN)?)?;
    let ids: BTreeSet<_> = examples
        .iter()
        .map(|row| number::<u64>(row, "id"))
        .collect::<Result<_>>()?;
    ensure(examples.len() == N && ids.len() == N, || {
        "bad TRAIN coverage".into()
    })?;
    let mut label_count = 0;
    let mut positive_quote_labels = 0;
    let mut first_labels = BTreeSet::new();
    for row in &examples {
        let labels = row["expected"].as_array().ok_or("missing TRAIN labels")?;
        let first = labels
            .first()
            .and_then(Value::as_u64)
            .ok_or("missing first TRAIN label")?;
        ensure(
            (2..VOCAB as u64).contains(&first) && labels.last() == Some(&json!(1)),
            || "wrong TRAIN label coordinates".into(),
        )?;
        label_count += labels.len();
        positive_quote_labels += labels
            .iter()
            .filter(|label| label.as_u64() == Some(QUOTE as u64))
            .count();
        first_labels.insert(first);
    }
    ensure(label_count == 2151 && positive_quote_labels == 0, || {
        "TRAIN target census mismatch".into()
    })?;
    let capture_bytes = fs::read(Path::new(capture))?;
    let captured = rows(&capture_bytes)?;
    ensure(captured.len() == 3 * N, || {
        format!("expected 768 rows, got {}", captured.len())
    })?;
    let mut audited = Vec::new();
    for (arm_index, (arm, artifact)) in ARMS.into_iter().zip([frozen, reset, preserved]).enumerate()
    {
        let checkpoint = Checkpoint::from_bytes(&input(artifact, arm.1)?)?;
        let readings = captured[arm_index * N..(arm_index + 1) * N]
            .iter()
            .zip(&examples)
            .enumerate()
            .map(|(document, (record, example))| {
                verify(record, example, &checkpoint, arm, document)
                    .map_err(|error| format!("{} row {}: {error}", arm.0, example["id"]).into())
            })
            .collect::<Result<Vec<_>>>()?;
        audited.push(readings);
    }
    let mut outcomes = BTreeMap::new();
    for (i, ((arm, _), readings)) in ARMS.into_iter().zip(&audited).enumerate() {
        let quote = readings
            .iter()
            .copied()
            .filter(|r| r.winner == QUOTE)
            .collect::<Vec<_>>();
        outcomes.insert(
            arm,
            json!({"all":describe(readings),
            "quote":describe(&quote),
            "matched_against_frozen":(i>0).then(||paired(&audited[0],readings))}),
        );
    }
    Ok(
        json!({"verdict":"PASS","capture_sha256":digest(&capture_bytes),"train_sha256":TRAIN,
        "cubin_sha256":CUBIN,"unique_articles":N,"checkpoint_rows":captured.len(),
        "checkpoint_sha256":BTreeMap::from(ARMS),"scope":"mechanical reconciliation of physical rows; local dz only",
        "historical_fixture_sha256":HISTORICAL,"physical_instrument_sha256":calibration_sha256,
        "supervised_label_count":label_count,"supervised_id1228_count":positive_quote_labels,
        "distinct_supervised_first_ids":first_labels.len(),
        "checkpoint_head_dot_recomputed_rows":captured.len(),"checkpoint_head_dot_recomputed_scores":3*WIDTH*captured.len(),
        "arms":outcomes}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&entry()?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_dot_ties_and_minimum_margin() {
        let mut checkpoint = Checkpoint {
            original_to_slot: (0..VOCAB).collect(),
            slot_to_original: (0..VOCAB).collect(),
            dimension: 512,
            weights: vec![0; 512 * WIDTH],
        };
        let hidden = vec![ONE as i32; 512];
        checkpoint.weights[0] = 16;
        checkpoint.weights[512] = 15;
        let words = checkpoint.head(&hidden, 0);
        let all = rank(&words, |_| true);
        assert_eq!(&all[..3], &[0, 1, 2]);
        assert_eq!(gap(&words, &all), 1);
        let masked = rank(&words, |f| checkpoint.eligible(0, f));
        assert_eq!(&masked[..3], &[1, 2, 3]);
        assert_eq!(gap(&words, &masked), 0);
        assert_eq!(target_gap(&words, &all, 1), -1);
        checkpoint.weights[0] = -16;
        checkpoint.weights[512] = -17;
        assert_eq!(checkpoint.head(&hidden, 0)[..2], [0, -1]);
    }

    #[test]
    fn exact_clamp_boundary_and_dz() {
        let mut words = [0; WIDTH];
        words[0] = -32_768;
        let (loss, dz, saturated) = local_derivative(&words, 0);
        assert_eq!(loss, 4_194_304);
        assert_eq!(dz[0], -CLAMP);
        assert!(!saturated);
        words[0] -= 1;
        assert!(local_derivative(&words, 0).2);
    }
}
