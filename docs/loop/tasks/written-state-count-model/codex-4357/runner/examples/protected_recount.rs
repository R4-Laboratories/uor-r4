//! Independently recount protected proofs from the raw GPU address tape.
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CERTIFICATES: &str = "a84ca2250df04cee9f3915377af1675cd4bffbdb2d0a38573c8078dbcf27b407";
const OBSERVATIONS: &str = "1a553f635caa4b3920fc076ddb3e8cc21510da8be490058e967a7aef09ca24cb";
const STEP: i128 = 1 << 32;
const ARMS: [&str; 5] = ["snapped", "transported", "raw", "swapped", "relabeled"];

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Identity {
    ordinal: u32,
    gold_id: u16,
    rival_id: u16,
}

#[derive(Deserialize)]
struct Certificate {
    #[serde(flatten)]
    identity: Identity,
    group: u64,
    gold_prior: u64,
    rival_prior: u64,
    gold_slots: [u32; 4],
    rival_slots: [u32; 4],
    required_difference: i64,
}

#[derive(Deserialize)]
struct Request {
    #[serde(flatten)]
    certificate: Frozen,
    swap_ordinal: u32,
}

#[derive(Deserialize)]
struct Frozen {
    #[serde(flatten)]
    identity: Identity,
    group: u64,
    gold_prior: u64,
    rival_prior: u64,
}

#[derive(Deserialize)]
struct Physical {
    carries: [u32; 4],
    addresses: [[u32; 2]; 4],
    priors: [u64; 2],
}

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned().into())
    }
}

fn read_lines<T: DeserializeOwned>(path: &Path, hash: Option<&str>) -> Result<Vec<T>> {
    let bytes = fs::read(path)?;
    if let Some(hash) = hash {
        check(
            hex::encode(Sha256::digest(&bytes)) == hash,
            "witness digest changed",
        )?;
    }
    std::str::from_utf8(&bytes)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn digits(mut source: u32) -> Result<[u32; 4]> {
    let mut slots = [0; 4];
    for (bank, slot) in slots.iter_mut().enumerate() {
        *slot = 64 * bank as u32 + source % 64;
        source /= 64;
    }
    check(source == 0, "noncanonical packed slots")?;
    Ok(slots)
}

fn vector(addresses: [u32; 2], relabeled: bool) -> Result<[i8; 256]> {
    let mut coefficients = [0; 256];
    for (packed, sign) in addresses.into_iter().zip([1, -1]) {
        for (bank, slot) in digits(packed)?.into_iter().enumerate() {
            let index = if relabeled {
                (slot as usize * (2 * bank + 3) + 13 + 4 * bank) % 64 + bank * 64
            } else {
                slot as usize
            };
            coefficients[index] += sign;
        }
    }
    Ok(coefficients)
}

#[derive(Clone, Copy, Default)]
struct Counts {
    carry: usize,
    contrast: usize,
    opposite: usize,
    pairs: usize,
    aliases: usize,
    impossible: usize,
    errors: usize,
}

fn recount(witness: &Path, output: &Path) -> Result<Value> {
    let certs =
        read_lines::<[Certificate; 2]>(&witness.join("certificates.jsonl"), Some(CERTIFICATES))?;
    let _: Vec<Value> = read_lines(&witness.join("observations.jsonl"), Some(OBSERVATIONS))?;
    let raw = read_lines::<(Request, Physical)>(&output.join("protected-addresses.jsonl"), None)?;
    let details: Vec<Value> = read_lines(&output.join("protected-pairs.jsonl"), None)?;
    check(
        certs.len() == 1334 && raw.len() == 2668 && details.len() == 1334,
        "incomplete proof coverage",
    )?;
    let mut donors = BTreeMap::new();
    for (req, physical) in &raw {
        check(
            donors
                .insert(
                    req.certificate.identity.ordinal,
                    (req.certificate.group, physical.carries[1]),
                )
                .is_none(),
            "duplicate raw ordinal",
        )?;
    }
    let mut rotated: Vec<_> = donors
        .iter()
        .map(|(&ordinal, &(group, _))| (group, ordinal))
        .collect();
    rotated.sort_unstable();
    let width = rotated
        .chunk_by(|a, b| a.0 == b.0)
        .map(<[_]>::len)
        .max()
        .ok_or("empty donor map")?;
    let expected_donor: BTreeMap<_, _> = rotated
        .iter()
        .zip(rotated.iter().cycle().skip(width))
        .map(|(recipient, donor)| (recipient.1, donor.1))
        .collect();
    let mut consumed_donors = BTreeMap::new();
    let mut counts = [Counts::default(); 5];
    for (index, ((original, pair), detail)) in certs
        .iter()
        .zip(raw.chunks_exact(2))
        .zip(details)
        .enumerate()
    {
        check(detail["pair_index"] == index, "pair receipt index changed")?;
        check(
            original[0].group != original[1].group,
            "same-group protected pair",
        )?;
        let mut thresholds = [0; 2];
        for (side, (req, physical)) in pair.iter().enumerate() {
            let cert = &original[side];
            let priors = [cert.gold_prior, cert.rival_prior];
            check(
                req.certificate.identity == cert.identity
                    && req.certificate.group == cert.group
                    && [req.certificate.gold_prior, req.certificate.rival_prior] == priors
                    && physical.priors == priors
                    && digits(physical.addresses[0][0])? == cert.gold_slots
                    && digits(physical.addresses[0][1])? == cert.rival_slots,
                "frozen baseline identity, prior or physical slots changed",
            )?;
            let &(group, carry) = donors.get(&req.swap_ordinal).ok_or("missing donor")?;
            check(
                consumed_donors.insert(req.swap_ordinal, ()).is_none()
                    && expected_donor.get(&cert.identity.ordinal) == Some(&req.swap_ordinal)
                    && group != cert.group
                    && physical.carries[3] == carry,
                "swapped carry lacks its distinct-group permutation donor",
            )?;
            let prior_delta = i128::from(priors[0]) - i128::from(priors[1]);
            let threshold = i128::from(cert.required_difference);
            check(
                prior_delta + STEP * threshold > 0 && prior_delta + STEP * (threshold - 1) <= 0,
                "frozen threshold is not exact",
            )?;
            thresholds[side] = threshold;
        }
        let base = [
            vector(pair[0].1.addresses[0], false)?,
            vector(pair[1].1.addresses[0], false)?,
        ];
        for (arm, count) in counts.iter_mut().enumerate() {
            let arms = [
                vector(pair[0].1.addresses[arm % 4], arm == 4)?,
                vector(pair[1].1.addresses[arm % 4], arm == 4)?,
            ];
            let mut impossible = 0;
            let mut aliases = 0;
            for side in 0..2 {
                let physical = &pair[side].1;
                count.carry += usize::from(physical.carries[arm % 4] != physical.carries[0]);
                let contribution = &arms[side];
                count.contrast += usize::from(*contribution != base[side]);
                let max: i128 = contribution
                    .iter()
                    .map(|&sign| match sign {
                        -1 => -i128::from(i16::MIN),
                        1 => i128::from(i16::MAX),
                        _ => 0,
                    })
                    .sum();
                if max < thresholds[side] {
                    impossible += 1;
                    aliases += usize::from(max == 0);
                }
            }
            let opposite = arms[0] != [0; 256]
                && arms[0]
                    .iter()
                    .zip(arms[1])
                    .all(|(&left, right)| left == -right)
                && thresholds[0] + thresholds[1] > 0;
            let forced = impossible.max(usize::from(opposite));
            check(
                detail["comparisons"][ARMS[arm]]["proof"]
                    == json!({
                        "opposite_signed": opposite,
                        "required_sum": i64::try_from(thresholds[0] + thresholds[1])?,
                        "individual_impossibilities": impossible,
                        "strict_impossible_aliases": aliases,
                        "forced_errors": forced,
                    }),
                "recount disagrees with a protected pair proof",
            )?;
            count.opposite += usize::from(opposite);
            count.impossible += impossible;
            count.aliases += aliases;
            count.pairs += usize::from(forced > 0);
            count.errors += forced;
        }
    }
    let mut arms = BTreeMap::new();
    for (name, count) in ARMS.into_iter().zip(counts) {
        let upper = 3886 - count.errors;
        arms.insert(
            name,
            json!({
                "carry_changed_rows": count.carry,
                "contrast_changed_rows": count.contrast,
                "preserved_opposite_proofs": count.opposite,
                "preserved_pairs": count.pairs,
                "cleared_pairs": 1334 - count.pairs,
                "strict_impossible_aliases": count.aliases,
                "individual_impossibilities": count.impossible,
                "unique_forced_errors": count.errors,
                "strict_win_upper_bound": upper,
                "eligible_for_full_census": count.pairs <= 777 && upper >= 3109
            }),
        );
    }
    let result = json!(arms);
    let published: Value = serde_json::from_slice(&fs::read(output.join("summary.json"))?)?;
    check(
        published["arms"] == result
            && result["snapped"]["preserved_opposite_proofs"] == 1334
            && result["relabeled"]["preserved_opposite_proofs"] == 1334,
        "independent recount differs from reported arms",
    )?;
    Ok(result)
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let [witness, output] = arguments.as_slice() else {
        return Err("expected WITNESS_DIR OUTPUT_DIR".into());
    };
    let result = recount(Path::new(witness), Path::new(output))?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
