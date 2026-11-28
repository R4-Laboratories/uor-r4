//! Replay the frozen #4356 contradiction panel with physically captured addresses.
use crate::{fail, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

const CERTIFICATES_SHA: &str = "a84ca2250df04cee9f3915377af1675cd4bffbdb2d0a38573c8078dbcf27b407";
const OBSERVATIONS_SHA: &str = "1a553f635caa4b3920fc076ddb3e8cc21510da8be490058e967a7aef09ca24cb";
const PAIRS: usize = 1334;
const DOMAIN: usize = 3886;
const BAR: usize = 3109;
const UNIT: i64 = 1 << 32;
const VARIANTS: [&str; 5] = ["snapped", "transported", "raw", "swapped", "relabeled"];

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Request {
    pub ordinal: u32,
    pub group: u64,
    pub gold_id: u16,
    pub rival_id: u16,
    pub gold_prior: u64,
    pub rival_prior: u64,
    pub swap_ordinal: u32,
}

/// Rows follow requests() order. Variants follow snapped, transported, raw, swapped.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Capture {
    pub carries: [u32; 4],
    pub addresses: [[u32; 2]; 4],
    pub priors: [u64; 2],
}

#[derive(Clone, Copy, Deserialize)]
struct Certificate {
    ordinal: u32,
    group: u64,
    gold_id: u16,
    rival_id: u16,
    gold_prior: u64,
    rival_prior: u64,
    gold_slots: [u32; 4],
    rival_slots: [u32; 4],
    required_difference: i64,
}

#[derive(Deserialize)]
struct Observation {
    ordinal: u32,
    gold_id: u16,
    rival_id: u16,
    gold: CandidateObservation,
    rival: CandidateObservation,
}

#[derive(Deserialize)]
struct CandidateObservation {
    prior: u64,
    gold: u32,
    rival: u32,
}

pub(crate) struct Protected {
    requests: Vec<Request>,
    baseline: Vec<[u32; 2]>,
    swap_indices: Vec<usize>,
}

fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(fail(message))
    }
}

fn read_pinned<T: DeserializeOwned>(path: &Path, expected: &str) -> Result<Vec<T>> {
    let bytes = fs::read(path)?;
    require(
        hex::encode(Sha256::digest(&bytes)) == expected,
        &format!("frozen file SHA mismatch: {}", path.display()),
    )?;
    std::str::from_utf8(&bytes)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line)
                .map_err(|error| fail(format!("{}:{}: {error}", path.display(), i + 1)))
        })
        .collect()
}

fn unpack(packed: u32) -> [u32; 4] {
    std::array::from_fn(|bank| {
        let bank = bank as u32;
        64 * bank + ((packed >> (6 * bank)) & 63)
    })
}

/// Within each bank, equal addresses cancel. Unequal addresses encode +gold/-rival.
fn incidence([gold, rival]: [u32; 2]) -> u64 {
    let gold = unpack(gold);
    let rival = unpack(rival);
    let mut signature = 0;
    for bank in 0..4 {
        if gold[bank] != rival[bank] {
            let pair = ((gold[bank] % 64) << 6) | (rival[bank] % 64);
            signature |= u64::from(pair) << (12 * bank);
        }
    }
    signature
}

fn bound([gold, rival]: [u64; 2]) -> i64 {
    (rival as i64 - gold as i64).div_euclid(UNIT) + 1
}

fn relabel(packed: u32) -> u32 {
    let mut renamed = 0;
    for bank in 0..4 {
        let slot = (packed >> (6 * bank)) & 63;
        // Odd multipliers make each affine map a bijection on the 64 slots.
        let slot = ((2 * bank + 3) * slot + 13 + 4 * bank) & 63;
        renamed |= slot << (6 * bank);
    }
    renamed
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Side {
    carry: u32,
    gold_slots: [u32; 4],
    rival_slots: [u32; 4],
    signed_incidence: u64,
    required_difference: i64,
    max_difference_i16: i64,
    strict_alias_impossible: bool,
    individually_impossible: bool,
    carry_changed: bool,
    contrast_changed: bool,
}

impl Side {
    fn new(carry: u32, addresses: [u32; 2], priors: [u64; 2], baseline: &Self) -> Self {
        let mut side = Self::read(carry, addresses, priors);
        side.carry_changed = carry != baseline.carry;
        side.contrast_changed = side.signed_incidence != baseline.signed_incidence;
        side
    }

    fn read(carry: u32, addresses: [u32; 2], priors: [u64; 2]) -> Self {
        let [gold_slots, rival_slots] = addresses.map(unpack);
        let unequal_banks = gold_slots
            .iter()
            .zip(rival_slots)
            .filter(|(gold, rival)| **gold != *rival)
            .count() as i64;
        let max_difference_i16 = unequal_banks * (i64::from(i16::MAX) - i64::from(i16::MIN));
        let required_difference = bound(priors);
        Self {
            carry,
            gold_slots,
            rival_slots,
            signed_incidence: incidence(addresses),
            required_difference,
            max_difference_i16,
            strict_alias_impossible: unequal_banks == 0 && required_difference > 0,
            individually_impossible: required_difference > max_difference_i16,
            carry_changed: false,
            contrast_changed: false,
        }
    }

    fn reverse(self) -> u64 {
        let mut reverse = 0;
        for bank in 0..4 {
            let gold = self.rival_slots[bank];
            let rival = self.gold_slots[bank];
            if gold != rival {
                reverse |= u64::from(((gold % 64) << 6) | (rival % 64)) << (12 * bank);
            }
        }
        reverse
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Proof {
    opposite_signed: bool,
    required_sum: i64,
    individual_impossibilities: usize,
    strict_impossible_aliases: usize,
    forced_errors: usize,
}

fn proof([left, right]: [Side; 2]) -> Proof {
    let required_sum = left.required_difference + right.required_difference;
    let opposite_signed =
        left.signed_incidence != 0 && left.signed_incidence == right.reverse() && required_sum > 0;
    let individual_impossibilities =
        usize::from(left.individually_impossible) + usize::from(right.individually_impossible);
    Proof {
        opposite_signed,
        required_sum,
        individual_impossibilities,
        strict_impossible_aliases: usize::from(left.strict_alias_impossible)
            + usize::from(right.strict_alias_impossible),
        forced_errors: individual_impossibilities.max(usize::from(opposite_signed)),
    }
}

pub(crate) fn load(witness_dir: &Path) -> Result<Protected> {
    let certificates: Vec<[Certificate; 2]> =
        read_pinned(&witness_dir.join("certificates.jsonl"), CERTIFICATES_SHA)?;
    let observations: Vec<Observation> =
        read_pinned(&witness_dir.join("observations.jsonl"), OBSERVATIONS_SHA)?;
    require(
        certificates.len() == PAIRS && observations.len() == 2 * PAIRS,
        "unexpected frozen protected panel size",
    )?;
    let mut reads = BTreeMap::new();
    for obs in observations {
        require(
            reads
                .insert((obs.ordinal, obs.gold_id, obs.rival_id), obs)
                .is_none(),
            "duplicate frozen GPU reread",
        )?;
    }
    let mut seen = BTreeSet::new();
    let mut requests = Vec::with_capacity(2 * PAIRS);
    let mut baseline = Vec::with_capacity(2 * PAIRS);
    for pair in certificates {
        require(pair[0].group != pair[1].group, "certificate shares a group")?;
        let mut pair_sides = Vec::with_capacity(2);
        for cert in pair {
            require(
                seen.insert(cert.ordinal)
                    && (1..=4095).contains(&cert.gold_id)
                    && (1..=4095).contains(&cert.rival_id)
                    && cert.gold_id != cert.rival_id
                    && cert.gold_prior <= 1 << 48
                    && cert.rival_prior <= 1 << 48,
                "bad frozen ID, prior, or duplicate row",
            )?;
            let obs = reads
                .remove(&(cert.ordinal, cert.gold_id, cert.rival_id))
                .ok_or_else(|| fail("missing frozen GPU reread"))?;
            let addresses = [obs.gold.gold, obs.rival.rival];
            let priors = [obs.gold.prior, obs.rival.prior];
            require(
                addresses.iter().all(|&address| address < 1 << 24)
                    && obs.gold.gold == obs.gold.rival
                    && obs.gold.gold == obs.rival.gold
                    && addresses.map(unpack) == [cert.gold_slots, cert.rival_slots]
                    && priors == [cert.gold_prior, cert.rival_prior]
                    && bound(priors) == cert.required_difference,
                "frozen certificate differs from its GPU reread",
            )?;
            baseline.push(addresses);
            pair_sides.push(Side::read(0, addresses, priors));
            requests.push(Request {
                ordinal: cert.ordinal,
                group: cert.group,
                gold_id: cert.gold_id,
                rival_id: cert.rival_id,
                gold_prior: cert.gold_prior,
                rival_prior: cert.rival_prior,
                swap_ordinal: cert.ordinal,
            });
        }
        let [left, right] = pair_sides.as_slice() else {
            return Err(fail("incomplete frozen pair"));
        };
        require(
            proof([*left, *right]).opposite_signed,
            "frozen proof did not reconstruct",
        )?;
    }
    require(reads.is_empty(), "orphan frozen GPU reread")?;
    let swap_indices = donors(&requests)?;
    for (i, &donor) in swap_indices.iter().enumerate() {
        requests[i].swap_ordinal = requests[donor].ordinal;
    }
    Ok(Protected {
        requests,
        baseline,
        swap_indices,
    })
}

fn donors(requests: &[Request]) -> Result<Vec<usize>> {
    let mut order: Vec<_> = (0..requests.len()).collect();
    order.sort_unstable_by_key(|&i| (requests[i].group, requests[i].ordinal));
    let width = order
        .chunk_by(|&a, &b| requests[a].group == requests[b].group)
        .map(<[_]>::len)
        .max()
        .ok_or_else(|| fail("empty protected cohort"))?;
    // Groups are contiguous intervals on the circle. Moving by the widest
    // interval keeps each recipient outside its original group.
    require(
        width <= order.len() / 2,
        "no group-disjoint carry permutation",
    )?;
    let mut donors = vec![0; order.len()];
    for (position, &recipient) in order.iter().enumerate() {
        let donor = order[(position + width) % order.len()];
        require(
            requests[recipient].group != requests[donor].group,
            "swapped carry shares the recipient group",
        )?;
        donors[recipient] = donor;
    }
    Ok(donors)
}

#[derive(Clone, Copy, Default, Serialize)]
struct Tally {
    carry_changed_rows: usize,
    contrast_changed_rows: usize,
    preserved_opposite_proofs: usize,
    preserved_pairs: usize,
    cleared_pairs: usize,
    strict_impossible_aliases: usize,
    individual_impossibilities: usize,
    unique_forced_errors: usize,
    strict_win_upper_bound: usize,
    eligible_for_full_census: bool,
}

impl Tally {
    fn add(&mut self, sides: [Side; 2], result: Proof) {
        self.carry_changed_rows += sides.iter().filter(|side| side.carry_changed).count();
        self.contrast_changed_rows += sides.iter().filter(|side| side.contrast_changed).count();
        self.preserved_opposite_proofs += usize::from(result.opposite_signed);
        self.preserved_pairs += usize::from(result.forced_errors > 0);
        self.cleared_pairs += usize::from(result.forced_errors == 0);
        self.strict_impossible_aliases += result.strict_impossible_aliases;
        self.individual_impossibilities += result.individual_impossibilities;
        self.unique_forced_errors += result.forced_errors;
    }
}

impl Protected {
    pub(crate) fn requests(&self) -> Vec<Request> {
        self.requests.clone()
    }

    pub(crate) fn score(&self, captures: &[Capture], out: &Path) -> Result<()> {
        require(
            captures.len() == self.requests.len(),
            "incomplete protected GPU capture",
        )?;
        let mut panel = Vec::with_capacity(captures.len());
        for (i, (&request, capture)) in self.requests.iter().zip(captures).enumerate() {
            require(
                capture.priors == [request.gold_prior, request.rival_prior]
                    && capture.addresses[0] == self.baseline[i]
                    && capture
                        .addresses
                        .iter()
                        .flatten()
                        .all(|&address| address < 1 << 24),
                &format!(
                    "GPU baseline differs from protected reread at ordinal {}",
                    request.ordinal
                ),
            )?;
            require(
                capture.carries[3] == captures[self.swap_indices[i]].carries[1],
                &format!(
                    "GPU swapped carry differs from its donor at ordinal {}",
                    request.ordinal
                ),
            )?;
            let baseline = Side::read(capture.carries[0], capture.addresses[0], capture.priors);
            panel.push(std::array::from_fn::<_, 5, _>(|arm| {
                let (carry, addresses) = if arm == 4 {
                    (capture.carries[0], capture.addresses[0].map(relabel))
                } else {
                    (capture.carries[arm], capture.addresses[arm])
                };
                Side::new(carry, addresses, capture.priors, &baseline)
            }));
        }
        fs::create_dir_all(out)?;
        let mut raw = BufWriter::new(File::create_new(out.join("protected-addresses.jsonl"))?);
        for (request, capture) in self.requests.iter().zip(captures) {
            serde_json::to_writer(&mut raw, &(request, capture))?;
            raw.write_all(b"\n")?;
        }
        raw.flush()?;
        let mut rows = BufWriter::new(File::create_new(out.join("protected-pairs.jsonl"))?);
        let mut tallies = [Tally::default(); 5];
        for (index, (requests, pair)) in self
            .requests
            .chunks_exact(2)
            .zip(panel.chunks_exact(2))
            .enumerate()
        {
            let mut comparisons = BTreeMap::new();
            let baseline = proof([pair[0][0], pair[1][0]]);
            require(
                baseline.opposite_signed,
                "captured baseline lost a protected proof",
            )?;
            for (arm, name) in VARIANTS.iter().enumerate() {
                let sides = [pair[0][arm], pair[1][arm]];
                let result = proof(sides);
                require(
                    arm != 4 || result == baseline,
                    "global bank bijection changed a proof",
                )?;
                tallies[arm].add(sides, result);
                comparisons.insert(*name, serde_json::json!({"sides": sides, "proof": result}));
            }
            serde_json::to_writer(
                &mut rows,
                &serde_json::json!({
                    "pair_index": index, "requests": requests, "comparisons": comparisons
                }),
            )?;
            rows.write_all(b"\n")?;
        }
        rows.flush()?;
        let mut arms = BTreeMap::new();
        for (name, mut tally) in VARIANTS.into_iter().zip(tallies) {
            tally.strict_win_upper_bound = DOMAIN - tally.unique_forced_errors;
            tally.eligible_for_full_census =
                tally.cleared_pairs >= 557 && tally.strict_win_upper_bound >= BAR;
            arms.insert(name, tally);
        }
        let mut summary = BufWriter::new(File::create_new(out.join("summary.json"))?);
        serde_json::to_writer_pretty(
            &mut summary,
            &serde_json::json!({
                "protected_pairs": PAIRS,
                "protected_rows": self.requests.len(),
                "common_population": DOMAIN,
                "retention_bar": BAR,
                "minimum_cleared_pairs": 557,
                "maximum_forced_errors_at_bar": DOMAIN - BAR,
                "source_certificates_sha256": CERTIFICATES_SHA,
                "source_observations_sha256": OBSERVATIONS_SHA,
                "swap_policy": "sort by (group, ordinal), rotate by largest group count",
                "relabel_policy": "within bank b, ((2b+3)*slot+13+4b) mod 64",
                "arms": arms
            }),
        )?;
        summary.write_all(b"\n")?;
        summary.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bounds_and_disjoint_error_union() {
        let left = Side::read(0, [0, 1], [0, UNIT as u64]);
        let right = Side::read(0, [1, 0], [0, 0]);
        let original = proof([left, right]);
        assert!(original.opposite_signed);
        assert_eq!(original.required_sum, 3);
        assert_eq!(original.forced_errors, 1);
        assert_eq!(
            original,
            proof([
                Side::read(0, [0, 1].map(relabel), [0, UNIT as u64]),
                Side::read(0, [1, 0].map(relabel), [0, 0]),
            ])
        );

        let alias = Side::read(0, [0, 0], [0, 0]);
        let weaker_alias = Side::read(0, [0, 0], [1, 0]);
        assert_eq!(proof([alias, alias]).forced_errors, 2);
        assert_eq!(proof([alias, weaker_alias]).strict_impossible_aliases, 1);
        assert_eq!(proof([left, weaker_alias]).forced_errors, 0);
        let saturated = Side::read(0, [1, 0], [0, 1 << 48]);
        assert!(saturated.individually_impossible);
        assert!(!saturated.strict_alias_impossible);
        assert_eq!(proof([left, saturated]).forced_errors, 1);
        assert_eq!(bound([UNIT as u64 + 1, 0]), -1);
    }

    #[test]
    fn donor_rotation_is_bijective_and_group_disjoint() -> Result<()> {
        let requests: Vec<_> = [0, 1, 2, 3, 1, 2]
            .into_iter()
            .enumerate()
            .map(|(ordinal, group)| Request {
                ordinal: ordinal as u32,
                group,
                gold_id: 1,
                rival_id: 2,
                gold_prior: 0,
                rival_prior: 0,
                swap_ordinal: 0,
            })
            .collect();
        let swapped = donors(&requests)?;
        assert_eq!(
            swapped.iter().copied().collect::<BTreeSet<_>>().len(),
            requests.len()
        );
        assert!(swapped
            .iter()
            .enumerate()
            .all(|(recipient, &donor)| requests[recipient].group != requests[donor].group));
        assert!(donors(&requests[1..=1]).is_err());
        Ok(())
    }
}
