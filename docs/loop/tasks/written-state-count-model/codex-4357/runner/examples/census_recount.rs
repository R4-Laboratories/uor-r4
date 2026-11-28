//! Independently verify the selected all-rival certificates from GPU rereads.
#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/input.rs"]
mod input;

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const DOMAIN: usize = 3886;
const IDS: usize = 4095;
const BAR: usize = 3109;
const UNIT: i128 = 1 << 32;
const FROZEN: &str = "c9989459b37b3d2c6a9f337db199ad61f36d48bc186007227b9e518e7e58dd21";
const CERTS: &str = "a84ca2250df04cee9f3915377af1675cd4bffbdb2d0a38573c8078dbcf27b407";
const OBSERVATIONS: &str = "1a553f635caa4b3920fc076ddb3e8cc21510da8be490058e967a7aef09ca24cb";
const MIDPOINT_MODULE: &str = "feca07ab5f82cd97b9cfad2c93af4dbd8789f7e51d0bb53ef744dd94532f443a";
const PRIOR_MODULE: &str = "7b12da42688cdd704ca042d1a0d928683ca21178eca124e011b2f58afeffaa74";
const ARMS: [&str; 4] = ["snapped", "transported", "raw", "swapped"];

fn fail(message: impl Into<String>) -> Box<dyn std::error::Error> {
    message.into().into()
}

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(fail(message))
    }
}

fn load(path: &Path, expected: Option<&str>) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if let Some(expected) = expected {
        check(
            hex::encode(Sha256::digest(&bytes)) == expected,
            &format!("SHA-256 differs for {}", path.display()),
        )?;
    }
    Ok(bytes)
}

fn lines<T: DeserializeOwned>(path: &Path, expected: Option<&str>) -> Result<Vec<T>> {
    std::str::from_utf8(&load(path, expected)?)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line)
                .map_err(|error| fail(format!("{}:{}: {error}", path.display(), i + 1)))
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Row {
    group: u64,
    gold: u16,
}

#[derive(Deserialize)]
struct Frozen {
    seed: u32,
    addressing: String,
    ordinal: usize,
    group: u64,
    gold: u16,
    common: bool,
}

fn cohort(root: &Path, frozen: &Path) -> Result<BTreeMap<usize, Row>> {
    let train = input::load(root)?;
    let mut seen = BTreeMap::<usize, [Option<bool>; 4]>::new();
    for row in lines::<Frozen>(frozen, Some(FROZEN))? {
        let seed = match row.seed {
            435300 => 0,
            435301 => 2,
            _ => return Err(fail("unknown frozen seed")),
        };
        let addressing = match row.addressing.as_str() {
            "geometry" => 0,
            "ordered-matched" => 1,
            _ => return Err(fail("unknown frozen addressing")),
        };
        check(
            train.ordinals.get(row.ordinal).map(|n| *n as usize) == Some(row.ordinal)
                && train.groups.get(row.ordinal) == Some(&row.group)
                && train.targets.get(row.ordinal) == Some(&row.gold)
                && (1..=IDS as u16).contains(&row.gold),
            "frozen row disagrees with the authenticated TRAIN admission",
        )?;
        let copies = seen.entry(row.ordinal).or_default();
        check(
            copies[seed + addressing].replace(row.common).is_none(),
            "duplicate frozen row/arm/seed",
        )?;
    }
    check(seen.len() == 3891, "missing frozen TRAIN admission")?;
    let mut cohort = BTreeMap::new();
    for (i, copies) in seen {
        match copies {
            [Some(true), Some(true), Some(true), Some(true)] => {
                cohort.insert(
                    i,
                    Row {
                        group: train.groups[i],
                        gold: train.targets[i],
                    },
                );
            }
            [Some(false), Some(false), Some(false), Some(false)] => {}
            _ => return Err(fail("frozen common membership differs across arms/seeds")),
        }
    }
    check(cohort.len() == DOMAIN, "wrong frozen common denominator")?;
    Ok(cohort)
}

#[derive(Deserialize)]
struct OldSide {
    ordinal: usize,
    group: u64,
    gold_id: u16,
    rival_id: u16,
    gold_prior: u64,
    rival_prior: u64,
    gold_slots: [u32; 4],
    rival_slots: [u32; 4],
    required_difference: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
struct Observation {
    ordinal: usize,
    donor: usize,
    group: u64,
    gold_id: u16,
    rival_id: u16,
    carry: u32,
    packed_slots: [u32; 2],
    q48_priors: [u64; 2],
    q48_row_sha256: String,
}

#[derive(Deserialize)]
struct Proof {
    kind: String,
    first: Vec<Observation>,
    reread: Vec<Observation>,
}

fn slots(mut address: u32) -> Result<[u32; 4]> {
    let result = std::array::from_fn(|bank| {
        let slot = address % 64;
        address /= 64;
        64 * bank as u32 + slot
    });
    check(address == 0, "packed lexical addresses exceed four banks")?;
    Ok(result)
}

fn coefficients(obs: &Observation) -> Result<[i8; 256]> {
    let mut vector = [0; 256];
    for (address, sign) in obs.packed_slots.into_iter().zip([1, -1]) {
        for slot in slots(address)? {
            vector[slot as usize] += sign;
        }
    }
    Ok(vector)
}

fn strict_requirement(prior: [u64; 2]) -> Result<i128> {
    check(
        prior.into_iter().all(|mass| (1..=1 << 48).contains(&mass)),
        "noncanonical Q48 mass",
    )?;
    let delta = i128::from(prior[1]) - i128::from(prior[0]);
    // Integer truncation differs from floor for negative nonintegral quotients.
    let floor = delta / UNIT - i128::from(delta % UNIT < 0);
    Ok(floor + 1)
}

fn verify_kind(proof: &Proof) -> Result<bool> {
    check(proof.first == proof.reread, "physical reread changed")?;
    match (proof.kind.as_str(), proof.reread.as_slice()) {
        ("alias", [obs]) => {
            check(
                coefficients(obs)? == [0; 256] && strict_requirement(obs.q48_priors)? > 0,
                "selected alias can win",
            )?;
            Ok(true)
        }
        ("opposing", [left, right]) => {
            let l = coefficients(left)?;
            let r = coefficients(right)?;
            check(
                left.ordinal != right.ordinal
                    && left.group != right.group
                    && l != [0; 256]
                    && l.into_iter().zip(r).all(|(a, b)| a + b == 0)
                    && strict_requirement(left.q48_priors)? + strict_requirement(right.q48_priors)?
                        > 0,
                "selected pair is not a strict opposite contradiction",
            )?;
            Ok(false)
        }
        _ => Err(fail("unexpected proof kind or arity")),
    }
}

fn permutation(
    cohort: &BTreeMap<usize, Row>,
    protected: &BTreeSet<usize>,
) -> Result<(BTreeMap<usize, usize>, usize)> {
    let mut donors = BTreeMap::new();
    let mut complement_width = 0;
    for is_protected in [true, false] {
        let mut ordered: Vec<_> = cohort
            .iter()
            .filter(|(ordinal, _)| protected.contains(ordinal) == is_protected)
            .map(|(&ordinal, row)| (row.group, ordinal))
            .collect();
        ordered.sort_unstable();
        let width = ordered
            .chunk_by(|a, b| a.0 == b.0)
            .map(<[_]>::len)
            .max()
            .ok_or("empty donor partition")?;
        check(
            width <= ordered.len() / 2,
            "donor partition cannot derange groups",
        )?;
        if !is_protected {
            complement_width = width;
        }
        for (recipient, donor) in ordered.iter().zip(ordered.iter().cycle().skip(width)) {
            check(recipient.0 != donor.0, "assigned donor shares a group")?;
            donors.insert(recipient.1, donor.1);
        }
    }
    Ok((donors, complement_width))
}

fn num(value: &Value, field: &str) -> Result<usize> {
    Ok(usize::try_from(
        value[field]
            .as_u64()
            .ok_or_else(|| fail(format!("missing count {field}")))?,
    )?)
}

fn digest32(digest: &str) -> Result<[u8; 32]> {
    check(
        digest.len() == 64
            && digest
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "noncanonical SHA-256",
    )?;
    let mut bytes = [0; 32];
    hex::decode_to_slice(digest, &mut bytes)?;
    Ok(bytes)
}

fn prior_ledger(
    output: &Path,
    cohort: &BTreeMap<usize, Row>,
    summary: &Value,
) -> Result<BTreeMap<usize, String>> {
    let rows: Vec<(usize, String)> = lines(&output.join("prior-rows.jsonl"), None)?;
    check(
        rows.iter().map(|(i, _)| i).eq(cohort.keys()),
        "prior ledger differs from the ordered frozen cohort",
    )?;
    let mut hash = Sha256::new();
    for (_, digest) in &rows {
        hash.update(digest32(digest)?);
    }
    check(
        summary["prior_row_hashes_sha256"] == hex::encode(hash.finalize()),
        "prior row hash ledger digest differs from the capture receipt",
    )?;
    // The bulk mass bytes are not serialized, so only the digest's format and
    // the independently reread selected row hashes can be checked here.
    digest32(
        summary["prior_q48_sha256"]
            .as_str()
            .ok_or("missing bulk Q48 digest")?,
    )?;
    Ok(rows.into_iter().collect())
}

fn verify_old(obs: &Observation, old: &OldSide) -> Result<()> {
    check(
        (obs.ordinal, obs.group, obs.gold_id, obs.rival_id)
            == (old.ordinal, old.group, old.gold_id, old.rival_id)
            && obs.q48_priors == [old.gold_prior, old.rival_prior]
            && slots(obs.packed_slots[0])? == old.gold_slots
            && slots(obs.packed_slots[1])? == old.rival_slots
            && strict_requirement(obs.q48_priors)? == i128::from(old.required_difference),
        "snapped proof differs from the #4356 certificate",
    )
}

fn verify(
    train: &Path,
    witness: &Path,
    frozen: &Path,
    eligible: &Path,
    output: &Path,
) -> Result<Value> {
    let cohort = cohort(train, frozen)?;
    let old: Vec<[OldSide; 2]> = lines(&witness.join("certificates.jsonl"), Some(CERTS))?;
    let original_reads = load(&witness.join("observations.jsonl"), Some(OBSERVATIONS))?;
    check(
        old.len() == 1334,
        "wrong number of protected snapped proofs",
    )?;
    let protected: BTreeSet<_> = old.iter().flatten().map(|side| side.ordinal).collect();
    check(
        protected.len() == 2668 && protected.iter().all(|i| cohort.contains_key(i)),
        "protected snapped proof membership changed",
    )?;
    let mut old_masses = BTreeMap::new();
    for side in old.iter().flatten() {
        old_masses.insert((side.ordinal, side.gold_id), side.gold_prior);
        old_masses.insert((side.ordinal, side.rival_id), side.rival_prior);
    }
    let (donors, complement_width) = permutation(&cohort, &protected)?;
    let reported_donors: Vec<(usize, usize)> =
        serde_json::from_slice(&load(&output.join("donors.json"), None)?)?;
    check(
        reported_donors
            .into_iter()
            .eq(donors.iter().map(|(&i, &donor)| (i, donor))),
        "donor tape differs from the two locked group-disjoint permutations",
    )?;
    let gate_bytes = load(eligible, None)?;
    let gate: Value = serde_json::from_slice(&gate_bytes)?;
    let summary: Value = serde_json::from_slice(&load(&output.join("summary.json"), None)?)?;
    check(
        gate["source_certificates_sha256"] == CERTS
            && gate["source_observations_sha256"] == hex::encode(Sha256::digest(original_reads))
            && gate["arms"]["transported"]["eligible_for_full_census"] == true
            && num(&gate["arms"]["transported"], "cleared_pairs")? >= 557
            && num(&gate["arms"]["transported"], "strict_win_upper_bound")? >= BAR
            && summary["eligibility_sha256"] == hex::encode(Sha256::digest(gate_bytes))
            && summary["historical_common_sha256"] == FROZEN
            && summary["artifact"] == input::ARTIFACT
            && summary["state"] == input::STATE
            && summary["midpoint_cubin_sha256"] == MIDPOINT_MODULE
            && summary["p4_prior_cubin_sha256"] == PRIOR_MODULE
            && summary["retention_bar"] == BAR
            && summary["protected_donors"] == protected.len()
            && summary["complement_donors"] == DOMAIN - protected.len()
            && summary["complement_rotation"] == complement_width,
        "summary/donor/gate frozen identities differ",
    )?;
    let prior_rows = prior_ledger(output, &cohort, &summary)?;

    let mut arms = BTreeMap::new();
    let mut selected_masses = old_masses;
    let mut selected_transports = BTreeMap::new();
    for arm in ARMS {
        let name = format!("{arm}-proofs.jsonl");
        let proofs: Vec<Proof> = lines(&output.join(&name), None)?;
        let reported = &summary["arms"][arm];
        let mut used = BTreeSet::new();
        let mut aliases = 0;
        let mut pairs = 0;
        for (index, proof) in proofs.iter().enumerate() {
            for obs in &proof.reread {
                let row = cohort
                    .get(&obs.ordinal)
                    .ok_or("selected proof outside the common cohort")?;
                let donor = if arm == "swapped" {
                    donors[&obs.ordinal]
                } else {
                    obs.ordinal
                };
                check(
                    used.insert(obs.ordinal)
                        && obs.group == row.group
                        && obs.gold_id == row.gold
                        && (1..=IDS as u16).contains(&obs.rival_id)
                        && obs.gold_id != obs.rival_id
                        && obs.donor == donor
                        && prior_rows.get(&obs.ordinal) == Some(&obs.q48_row_sha256),
                    "proof reuses a row or changes the frozen recipient identity",
                )?;
                for (id, mass) in [obs.gold_id, obs.rival_id].into_iter().zip(obs.q48_priors) {
                    let original = selected_masses.entry((obs.ordinal, id)).or_insert(mass);
                    check(
                        *original == mass,
                        "recipient Q48 mass changes across address arms or history",
                    )?;
                }
                if matches!(arm, "transported" | "swapped") {
                    let original = selected_transports.entry(donor).or_insert(obs.carry);
                    check(
                        *original == obs.carry,
                        "transport carry changed for the same donor",
                    )?;
                }
            }
            if verify_kind(proof)? {
                aliases += 1;
            } else {
                pairs += 1;
            }
            if arm == "snapped" {
                let historical = old.get(index).ok_or("extra snapped certificate")?;
                check(
                    proof.reread.len() == 2,
                    "snapped arm selected an unexpected alias",
                )?;
                for (obs, side) in proof.reread.iter().zip(historical) {
                    verify_old(obs, side)?;
                }
            }
        }
        let forced = aliases + pairs;
        let upper = DOMAIN
            .checked_sub(forced)
            .ok_or("impossible forced-error count")?;
        let blocking = num(reported, "blocking_aliases")?;
        let nonalias = num(reported, "nonalias_inequalities")?;
        let weaker = num(reported, "weaker_full_aliases")?;
        check(
            reported["rows"] == DOMAIN
                && reported["original_rivals_per_row"] == IDS - 1
                && reported["rows_with_blocking_alias"] == aliases
                && reported["disjoint_certified_pairs"] == pairs
                && reported["forced_errors"] == forced
                && reported["strict_win_upper_bound"] == upper
                && reported["excludes_retention"] == (upper < BAR)
                && blocking >= aliases
                && nonalias
                    .checked_add(blocking)
                    .and_then(|sum| sum.checked_add(weaker))
                    == Some(DOMAIN * (IDS - 1)),
            "reported counts or retention ceiling differ from the selected proof tape",
        )?;
        if arm == "snapped" {
            check(
                pairs == 1334
                    && aliases == 0
                    && reported["opposed_rows_across_distinct_groups"] == 3686
                    && reported["nonalias_inequalities"] == 15_909_205
                    && reported["row_signature_directions"] == 15_567_316
                    && reported["weaker_full_aliases"] == 79
                    && reported["blocking_aliases"] == 0,
                "snapped census does not reproduce #4356",
            )?;
        }
        arms.insert(
            arm,
            json!({
                "certified_alias_rows": aliases,
                "certified_disjoint_pairs": pairs,
                "unique_proof_rows": used.len(),
                "forced_errors": forced,
                "strict_win_upper_bound": upper,
                "proof_sha256": hex::encode(Sha256::digest(load(&output.join(name), None)?)),
            }),
        );
    }
    Ok(json!({
        "common_rows": DOMAIN,
        "original_rivals_per_row": IDS - 1,
        "verified": "selected GPU certificates and upper bounds",
        "prior_row_hashes_sha256": summary["prior_row_hashes_sha256"],
        "reported_bulk_prior_q48_sha256": summary["prior_q48_sha256"],
        "arms": arms
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [train, witness, frozen, eligible, output] = args.as_slice() else {
        return Err(fail(
            "expected TRAIN WITNESS_DIR FROZEN_SPARSE PROTECTED_SUMMARY CENSUS_OUT",
        ));
    };
    let result = verify(
        Path::new(train),
        Path::new(witness),
        Path::new(frozen),
        Path::new(eligible),
        Path::new(output),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ordinal: usize, slots: [u32; 2], priors: [u64; 2]) -> Observation {
        Observation {
            ordinal,
            donor: ordinal,
            group: ordinal as u64,
            gold_id: 1,
            rival_id: 2,
            carry: 0,
            packed_slots: slots,
            q48_priors: priors,
            q48_row_sha256: String::new(),
        }
    }

    #[test]
    fn verifier_distinguishes_alias_ties_and_opposite_pairs() -> Result<()> {
        assert_eq!(strict_requirement([(UNIT + 2) as u64, 1])?, -1);
        assert_eq!(strict_requirement([(UNIT + 1) as u64, 1])?, 0);
        let alias = Proof {
            kind: "alias".into(),
            first: vec![sample(3, [67, 67], [5, 5])],
            reread: vec![sample(3, [67, 67], [5, 5])],
        };
        assert!(verify_kind(&alias)?);
        let mut proof = Proof {
            kind: "opposing".into(),
            first: vec![sample(4, [0, 64], [2, 2]), sample(5, [64, 0], [3, 2])],
            reread: vec![sample(4, [0, 64], [2, 2]), sample(5, [64, 0], [3, 2])],
        };
        assert!(!verify_kind(&proof)?);
        proof.reread[1].packed_slots = [1, 0];
        assert!(verify_kind(&proof).is_err());
        proof.first = proof.reread.clone();
        assert!(verify_kind(&proof).is_err());
        Ok(())
    }
}
