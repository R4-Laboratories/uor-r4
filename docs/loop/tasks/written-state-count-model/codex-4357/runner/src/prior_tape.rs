//! Independently recapture the authenticated census inputs without rerunning its matching.
use super::{
    candidate_index, device, fail, input, load_common, load_engines, old_proof_slots, priors,
    require, row_digest, update_priors, Observation, Result, BATCH, DOMAIN, FROZEN_SHA, IDS,
    MIDPOINT_SHA, P4_SHA,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

const TRUNK_SHA: &str = "b75fb885ec23ef43c7f248b1fda89b9d75841d8ff5e269129f5d565df302ecf3";
const PROOFS_SHA: &str = "a67b663e7271bd60b12688a24e4d8e25b4f63fffe700e29a69230d939b4621f1";
const HISTORIC_BULK: &str = "2a41c2bcdc8260f1fbf1cfee67b33b1c9fb99f153b99b0aa544da7c1fde0f2a7";
const HISTORIC_LEDGER: &str = "d4137e3c64a6f83feff9e6dbea67a23ee76f78a27d1e98ac5007e69089576d3e";
const VOCAB: usize = IDS + 1;
const ROOT_OFFSET: usize = 70 + VOCAB;
const PROTECTED_PAIRS: usize = 1334;

#[derive(Deserialize)]
struct Proof {
    kind: String,
    first: [Observation; 2],
    reread: [Observation; 2],
}

fn load_ledger(path: &Path, common: &[usize]) -> Result<Vec<String>> {
    let bytes = fs::read(path)?;
    let mut entries = bytes
        .split_inclusive(|&byte| byte == b'\n')
        .map(serde_json::from_slice::<(usize, String)>);
    let mut expected = Vec::with_capacity(DOMAIN);
    let mut digest = Sha256::new();
    for &ordinal in common {
        let (seen, hash) = entries.next().ok_or("missing historic prior row")??;
        let mut raw = [0u8; 32];
        hex::decode_to_slice(&hash, &mut raw)?;
        require(
            seen == ordinal && hash == hex::encode(raw),
            "historic prior ledger ordinals or hash encoding changed",
        )?;
        digest.update(raw);
        expected.push(hash);
    }
    require(
        entries.next().is_none() && hex::encode(digest.finalize()) == HISTORIC_LEDGER,
        "historic prior ledger coverage or aggregate digest changed",
    )?;
    Ok(expected)
}

fn load_trunk(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    require(
        bytes.len() == (ROOT_OFFSET + VOCAB) * 4
            && hex::encode(Sha256::digest(&bytes)) == TRUNK_SHA,
        "frozen lexical trunk identity changed",
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("exact trunk word")))
        .collect())
}

pub(super) fn run(
    p4_path: &Path,
    input_path: &Path,
    witness: &Path,
    frozen: &Path,
    ledger_path: &Path,
    proof_path: &Path,
    trunk_path: &Path,
    out: &Path,
) -> Result<()> {
    require(
        device::digest() == MIDPOINT_SHA
            && hex::encode(Sha256::digest(include_bytes!(env!("P4_QUAL_CUBIN")))) == P4_SHA,
        "capture requires both historic sm_89 modules",
    )?;
    require(
        !out.try_exists()?,
        "capture requires a fresh output directory",
    )?;
    let train = input::load(input_path)?;
    let common = load_common(frozen, FROZEN_SHA, &train)?;
    let expected = load_ledger(ledger_path, &common)?;
    let trunk = load_trunk(trunk_path)?;
    let roots: Vec<u16> = trunk[ROOT_OFFSET..]
        .iter()
        .map(|&word| u16::try_from(word))
        .collect::<std::result::Result<_, _>>()?;
    let groups: Vec<u32> = common
        .iter()
        .map(|&ordinal| u32::try_from(train.groups[ordinal]))
        .collect::<std::result::Result<_, _>>()?;

    let witness_requests = super::protected::load(witness)?.requests();
    let witness_slots = old_proof_slots(witness)?;
    let proof_bytes = fs::read(proof_path)?;
    require(
        hex::encode(Sha256::digest(&proof_bytes)) == PROOFS_SHA,
        "historic snapped proof tape identity changed",
    )?;
    let proofs: Vec<Proof> = proof_bytes
        .split_inclusive(|&byte| byte == b'\n')
        .map(serde_json::from_slice)
        .collect::<std::result::Result<_, _>>()?;
    require(
        proofs.len() == PROTECTED_PAIRS && witness_requests.len() == 2 * PROTECTED_PAIRS,
        "historic protected proof coverage changed",
    )?;
    let mut seen = vec![false; DOMAIN];
    let mut protected = Vec::with_capacity(2 * PROTECTED_PAIRS);
    for (pair, proof) in proofs.iter().enumerate() {
        require(
            proof.kind == "opposing" && proof.first == proof.reread,
            "historic snapped first and second physical reads differ",
        )?;
        for (side, observation) in proof.reread.iter().enumerate() {
            let index = 2 * pair + side;
            let request = witness_requests[index];
            let row = common.binary_search(&observation.ordinal).map_err(|_| {
                fail(format!(
                    "protected ordinal {} outside common rows",
                    observation.ordinal
                ))
            })?;
            require(
                !std::mem::replace(&mut seen[row], true)
                    && observation.ordinal == request.ordinal as usize
                    && observation.donor == observation.ordinal
                    && observation.group == request.group
                    && observation.group == train.groups[observation.ordinal]
                    && observation.gold_id == train.targets[observation.ordinal]
                    && observation.gold_id == request.gold_id
                    && observation.rival_id == request.rival_id
                    && observation.q48_priors == [request.gold_prior, request.rival_prior]
                    && observation.packed_slots == witness_slots[index]
                    && observation.q48_row_sha256 == expected[row],
                "historic snapped proof side differs from the authenticated witness",
            )?;
            protected.push((row, observation));
        }
    }

    let (midpoint, p4) = load_engines(p4_path, Some(&trunk))?;
    let prompts: Vec<_> = common.iter().map(|&i| train.prefixes[i].clone()).collect();
    let features = midpoint.capture(&prompts)?;
    require(features.len() == DOMAIN, "incomplete snapped carry capture")?;
    let mut rows = Vec::with_capacity(DOMAIN * IDS);
    let mut digest = Sha256::new();
    for (batch_number, batch) in prompts.chunks(BATCH).enumerate() {
        let masses = priors(&p4, batch)?;
        for (i, masses) in masses.chunks_exact(IDS).enumerate() {
            let row = batch_number * BATCH + i;
            require(
                row_digest(masses) == expected[row],
                &format!(
                    "physical prior row digest changed at ordinal {}",
                    common[row]
                ),
            )?;
            update_priors(&mut digest, masses);
        }
        rows.extend(masses);
    }
    require(
        rows.len() == DOMAIN * IDS && hex::encode(digest.finalize()) == HISTORIC_BULK,
        "physical prior coverage or bulk digest changed",
    )?;

    let mut sites = Vec::with_capacity(2 * protected.len());
    for &(row, observation) in &protected {
        let carry = features[row].snapped;
        let priors = [observation.gold_id, observation.rival_id]
            .map(|id| rows[row * IDS + candidate_index(id)]);
        require(
            carry == observation.carry && priors == observation.q48_priors,
            &format!(
                "physical protected carry or prior changed at ordinal {}",
                observation.ordinal
            ),
        )?;
        sites.extend([observation.gold_id, observation.rival_id].map(|id| [carry, u32::from(id)]));
    }
    let addresses = midpoint.addresses(&sites)?;
    require(
        addresses.len() == sites.len()
            && addresses
                .chunks_exact(2)
                .zip(&protected)
                .all(|(sites, (_, side))| sites == side.packed_slots),
        "physical protected address differs from the two historic readings",
    )?;

    fs::create_dir(out)?;
    let mut roots_file = BufWriter::new(File::create_new(out.join("roots.bin"))?);
    for root in roots {
        roots_file.write_all(&root.to_le_bytes())?;
    }
    roots_file.flush()?;
    let mut rows_file = BufWriter::new(File::create_new(out.join("rows.bin"))?);
    for (i, &ordinal) in common.iter().enumerate() {
        rows_file.write_all(&train.ordinals[ordinal].to_le_bytes())?;
        rows_file.write_all(&groups[i].to_le_bytes())?;
        rows_file.write_all(&train.targets[ordinal].to_le_bytes())?;
        rows_file.write_all(&u16::try_from(features[i].snapped)?.to_le_bytes())?;
        rows_file.write_all(&0u64.to_le_bytes())?;
        for &mass in &rows[i * IDS..(i + 1) * IDS] {
            rows_file.write_all(&mass.to_le_bytes())?;
        }
    }
    rows_file.flush()?;
    println!(
        "captured {DOMAIN} prior rows, {} snapped protected sides, {VOCAB} candidate roots, bulk {HISTORIC_BULK}",
        protected.len()
    );
    Ok(())
}
