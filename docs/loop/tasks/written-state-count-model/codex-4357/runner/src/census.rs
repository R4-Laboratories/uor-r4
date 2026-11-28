//! Conditional all-original-ID structural census on the frozen common rows.
#[allow(dead_code)]
mod device;
#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/input.rs"]
mod input;
#[allow(dead_code)]
#[path = "../../../antigravity-17/p4-qual/src/cuda.rs"]
mod p4_cuda;
#[allow(dead_code)]
mod protected;

use device::Gpu;
use p4_cuda::P4GpuEngine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cmp::Reverse,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, ClientArtifactFile, DiscreteCountModel,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const IDS: usize = 4095;
const DOMAIN: usize = 3886;
const BAR: usize = 3109;
const UNIT: i64 = 1 << 32;
const BATCH: usize = 64;
const ARMS: [&str; 4] = ["snapped", "transported", "raw", "swapped"];

fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}

fn require(ok: bool, message: &str) -> Result<()> {
    if !ok {
        return Err(fail(message));
    }
    Ok(())
}

fn save_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

#[derive(Deserialize)]
struct HistoricRow {
    seed: u32,
    addressing: String,
    ordinal: usize,
    group: u64,
    gold: u16,
    common: bool,
}

fn load_common(path: &Path, digest: &str, train: &input::Train) -> Result<Vec<usize>> {
    let bytes = fs::read(path)?;
    require(
        hex::encode(Sha256::digest(&bytes)) == digest,
        "frozen-sparse digest mismatch",
    )?;
    let mut copies = vec![[None; 4]; train.ordinals.len()];
    for line in bytes.split_inclusive(|&byte| byte == b'\n') {
        let row: HistoricRow = serde_json::from_slice(line)?;
        let seed = input::SEEDS
            .iter()
            .position(|&seed| seed == row.seed)
            .ok_or("unknown frozen seed")?;
        let variant = match row.addressing.as_str() {
            "geometry" => 0,
            "ordered-matched" => 1,
            _ => return Err(fail("unknown frozen arm")),
        };
        let i = row.ordinal;
        require(
            train.ordinals.get(i).copied() == u32::try_from(i).ok()
                && train.groups.get(i).copied() == Some(row.group)
                && train.targets.get(i).copied() == Some(row.gold),
            "frozen identity differs from the authenticated TRAIN admission",
        )?;
        require(
            copies[i][seed * 2 + variant].replace(row.common).is_none(),
            "duplicate frozen row",
        )?;
    }
    let mut common = Vec::new();
    for (i, &flags) in copies.iter().enumerate() {
        let flag = flags[0].ok_or("missing frozen admission")?;
        require(
            flags == [Some(flag); 4],
            "inconsistent frozen common membership",
        )?;
        if flag {
            common.push(i);
        }
    }
    require(
        common.len() == DOMAIN,
        "unexpected frozen common denominator",
    )?;
    Ok(common)
}

fn donor_ordinals(
    train: &input::Train,
    common: &[usize],
    requests: &[protected::Request],
) -> Result<(Vec<usize>, usize)> {
    let mut donors = vec![None; train.ordinals.len()];
    let mut is_common = vec![false; train.ordinals.len()];
    for &i in common {
        is_common[i] = true;
    }
    for req in requests {
        let i = req.ordinal as usize;
        let donor = req.swap_ordinal as usize;
        require(
            is_common.get(i).copied() == Some(true)
                && is_common.get(donor).copied() == Some(true)
                && train.groups[i] == req.group
                && train.targets[i] == req.gold_id
                && train.groups[i] != train.groups[donor]
                && donors[i].replace(donor).is_none(),
            "protected donor disagrees with the authentic cohort",
        )?;
    }
    require(
        requests
            .iter()
            .all(|req| donors[req.swap_ordinal as usize].is_some()),
        "protected donor leaves its partition",
    )?;
    let mut complement: Vec<_> = common
        .iter()
        .copied()
        .filter(|&i| donors[i].is_none())
        .collect();
    complement.sort_unstable_by_key(|&i| (train.groups[i], i));
    let width = complement
        .chunk_by(|&a, &b| train.groups[a] == train.groups[b])
        .map(<[_]>::len)
        .max()
        .ok_or("empty complement")?;
    require(
        width <= complement.len() / 2,
        "no group-disjoint complement rotation",
    )?;
    for (i, &ordinal) in complement.iter().enumerate() {
        donors[ordinal] = Some(complement[(i + width) % complement.len()]);
    }
    let donors: Vec<_> = common
        .iter()
        .map(|&i| donors[i].ok_or("missing donor"))
        .collect::<std::result::Result<_, _>>()?;
    let mut sorted = donors.clone();
    sorted.sort_unstable();
    require(sorted == common, "swapped arm is not a bijection")?;
    require(
        common
            .iter()
            .zip(&donors)
            .all(|(&i, &donor)| train.groups[i] != train.groups[donor]),
        "a donor shares its recipient's group",
    )?;
    Ok((donors, width))
}

fn load_engines(path: &Path) -> Result<(Gpu, P4GpuEngine)> {
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(path, false)?
    else {
        return Err(fail("requires the frozen embedded P4"));
    };
    require(
        identity.artifact_sha256 == input::ARTIFACT
            && expected_state_sha256 == input::STATE
            && hex::encode(Sha256::digest(&state_bytes)) == input::STATE,
        "P4 artifact or state identity differs",
    )?;
    let model =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha256, identity)?;
    let inner = &model.p4.as_ref().ok_or("missing P4")?.inner;
    require(
        inner.state_sha256().map(hex::encode).as_deref() == Some(input::STATE),
        "P4 bound state changed",
    )?;
    let raw = p4_cuda::split_uorp4a01_sections(&state_bytes)?;
    Ok((
        Gpu::new(&model.model.p4_lexical_trunk_words()?)?,
        P4GpuEngine::new(
            raw.counts_v3,
            raw.s3_compact,
            raw.s2_skip_compact,
            raw.s4_table,
            &inner.s6.weights_q32,
            &inner.codes,
        )?,
    ))
}

fn priors(p4: &P4GpuEngine, prompts: &[Vec<u16>]) -> Result<Vec<u64>> {
    let (steps, masses) =
        p4.run_parity_batch(prompts, &vec![1; prompts.len()], &vec![true; prompts.len()])?;
    require(
        steps.len() == prompts.len()
            && masses.len() == prompts.len() * IDS
            && steps.iter().zip(prompts).all(|(step, prompt)| {
                step.status == 0 && step.context_tokens as usize == prompt.len()
            })
            && masses.iter().all(|&mass| (1..=1 << 48).contains(&mass)),
        "frozen P4 scored a different prior domain or causal window",
    )?;
    Ok(masses)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Constraint {
    key: u64,
    row: u16,
    rival: u16,
    bound: i32,
}

fn inequality(row: usize, rival: u16, addresses: [u32; 2], masses: [u64; 2]) -> Constraint {
    let [gold, other] = addresses;
    let mut direct = 0;
    let mut reverse = 0;
    for bank in 0..4 {
        let g = u64::from((gold >> (6 * bank)) & 63);
        let r = u64::from((other >> (6 * bank)) & 63);
        if g != r {
            direct |= ((g << 6) | r) << (12 * bank);
            reverse |= ((r << 6) | g) << (12 * bank);
        }
    }
    Constraint {
        key: (direct.min(reverse) << 1) | u64::from(direct > reverse),
        row: row as u16,
        rival,
        bound: ((masses[1] as i64 - masses[0] as i64).div_euclid(UNIT) + 1) as i32,
    }
}

fn opposite(left: Constraint, right: Constraint, groups: &[u64]) -> bool {
    left.key ^ 1 == right.key
        && left.row != right.row
        && groups[usize::from(left.row)] != groups[usize::from(right.row)]
        && left.bound + right.bound > 0
}

fn cover(side: &[Constraint], other: &[Constraint], groups: &[u64], covered: &mut [bool]) {
    let Some(best) = other.iter().max_by_key(|c| c.bound) else {
        return;
    };
    let next = other
        .iter()
        .filter(|c| groups[usize::from(c.row)] != groups[usize::from(best.row)])
        .max_by_key(|c| c.bound);
    for &left in side {
        if opposite(left, *best, groups) || next.is_some_and(|&right| opposite(left, right, groups))
        {
            covered[usize::from(left.row)] = true;
        }
    }
}

fn match_pairs(
    edges: &mut Vec<Constraint>,
    killed: &[Option<Constraint>],
    groups: &[u64],
) -> (Vec<[Constraint; 2]>, usize) {
    edges.retain(|site| killed[usize::from(site.row)].is_none());
    edges.sort_unstable_by_key(|x| (x.key, x.row, Reverse(x.bound), x.rival));
    edges.dedup_by_key(|x| (x.key, x.row));
    let mut pairs = Vec::new();
    let mut used = vec![false; groups.len()];
    let mut covered = used.clone();
    for bucket in edges.chunk_by(|a, b| a.key >> 1 == b.key >> 1) {
        let (left, right) = bucket.split_at(bucket.partition_point(|c| c.key & 1 == 0));
        if left.is_empty() || right.is_empty() {
            continue;
        }
        cover(left, right, groups, &mut covered);
        cover(right, left, groups, &mut covered);
        let mut available: Vec<_> = right
            .iter()
            .copied()
            .filter(|v| !used[usize::from(v.row)])
            .collect();
        for &x in left {
            if used[usize::from(x.row)] {
                continue;
            }
            if let Some(j) = available.iter().position(|&y| opposite(x, y, groups)) {
                let y = available.swap_remove(j);
                used[usize::from(x.row)] = true;
                used[usize::from(y.row)] = true;
                pairs.push([x, y]);
            }
        }
    }
    (pairs, covered.into_iter().filter(|&v| v).count())
}

const FROZEN_SHA: &str = "c9989459b37b3d2c6a9f337db199ad61f36d48bc186007227b9e518e7e58dd21";
const CERTIFICATES_SHA: &str = "a84ca2250df04cee9f3915377af1675cd4bffbdb2d0a38573c8078dbcf27b407";
const MIDPOINT_SHA: &str = "feca07ab5f82cd97b9cfad2c93af4dbd8789f7e51d0bb53ef744dd94532f443a";
const P4_SHA: &str = "7b12da42688cdd704ca042d1a0d928683ca21178eca124e011b2f58afeffaa74";
const GATE_SHA: &str = "90eadf5bfc148f3c90f4955b6aa60a496b18c5875a823a9cfd4531eef523e3db";

fn arm_carry(arm: &str, feature: device::Feature) -> u32 {
    match arm {
        "snapped" => feature.snapped,
        "raw" => feature.raw,
        "transported" | "swapped" => feature.transported,
        _ => unreachable!("known census arm"),
    }
}

fn old_proof_slots(witness: &Path) -> Result<Vec<[u32; 2]>> {
    #[derive(Deserialize)]
    struct Slots {
        gold_slots: [u32; 4],
        rival_slots: [u32; 4],
    }
    let pack = |slots: [u32; 4]| {
        slots
            .into_iter()
            .enumerate()
            .fold(0, |packed, (bank, slot)| {
                packed | ((slot & 63) << (6 * bank))
            })
    };
    let bytes = fs::read(witness.join("certificates.jsonl"))?;
    require(
        hex::encode(Sha256::digest(&bytes)) == CERTIFICATES_SHA,
        "frozen certificate digest mismatch",
    )?;
    let mut slots = Vec::new();
    for line in bytes.split_inclusive(|&byte| byte == b'\n') {
        let pair: [Slots; 2] = serde_json::from_slice(line)?;
        slots.extend(pair.map(|side| [pack(side.gold_slots), pack(side.rival_slots)]));
    }
    require(
        slots.len() == 2668,
        "unexpected frozen certificate slot count",
    )?;
    Ok(slots)
}

fn all_addresses(gpu: &Gpu, carries: &[u32]) -> Result<Vec<u32>> {
    let sites: Vec<_> = carries
        .iter()
        .flat_map(|&carry| (1..=IDS as u32).map(move |candidate| [carry, candidate]))
        .collect();
    let results = gpu.addresses(&sites)?;
    require(
        results.len() == sites.len() && results.iter().all(|&slots| slots < 1 << 24),
        "incomplete all-ID address capture",
    )?;
    Ok(results)
}

fn candidate_index(id: u16) -> usize {
    usize::from(id) - 1
}

fn update_priors(hasher: &mut Sha256, row: &[u64]) {
    for &mass in row {
        hasher.update(mass.to_le_bytes());
    }
}

fn row_digest(row: &[u64]) -> String {
    let mut hasher = Sha256::new();
    update_priors(&mut hasher, row);
    hex::encode(hasher.finalize())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
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

fn observe(identity: Observation, row_addresses: &[u32], row_priors: &[u64]) -> Observation {
    let index = [identity.gold_id, identity.rival_id].map(candidate_index);
    Observation {
        packed_slots: index.map(|i| row_addresses[i]),
        q48_priors: index.map(|i| row_priors[i]),
        ..identity
    }
}

#[derive(Serialize)]
struct Summary {
    rows: usize,
    original_rivals_per_row: usize,
    nonalias_inequalities: usize,
    weaker_full_aliases: usize,
    blocking_aliases: usize,
    rows_with_blocking_alias: usize,
    row_signature_directions: usize,
    opposed_rows_across_distinct_groups: usize,
    disjoint_certified_pairs: usize,
    forced_errors: usize,
    strict_win_upper_bound: usize,
    excludes_retention: bool,
}

struct Domain<'a> {
    common: &'a [usize],
    groups: &'a [u64],
    labels: &'a [u16],
    prompts: &'a [Vec<u16>],
    masses: &'a [u64],
    prior_digests: &'a [String],
    requests: &'a [protected::Request],
    old_proof_slots: &'a [[u32; 2]],
}

impl Domain<'_> {
    fn capture(
        &self,
        gpu: &Gpu,
        p4: &P4GpuEngine,
        arm: &str,
        carries: &[u32],
        donors: &[usize],
        out: &Path,
    ) -> Result<Summary> {
        let mut aliases = vec![None; DOMAIN];
        let mut packed = Vec::with_capacity(DOMAIN * IDS);
        let mut constraints = Vec::with_capacity(DOMAIN * (IDS - 1));
        let mut weaker = 0;
        let mut blocking = 0;
        for (chunk, batch) in carries.chunks(BATCH).enumerate() {
            let addresses = all_addresses(gpu, batch)?;
            for (offset, row) in addresses.chunks_exact(IDS).enumerate() {
                let i = BATCH * chunk + offset;
                let gold = self.labels[i];
                let g = candidate_index(gold);
                let masses = &self.masses[i * IDS..(i + 1) * IDS];
                for (r, &rival_address) in row.iter().enumerate() {
                    if g == r {
                        continue;
                    }
                    let rival = (r + 1) as u16;
                    let c = inequality(i, rival, [row[g], rival_address], [masses[g], masses[r]]);
                    if c.key == 0 {
                        if c.bound > 0 {
                            blocking += 1;
                            if aliases[i].is_none() {
                                aliases[i] = Some(c);
                            }
                        } else {
                            weaker += 1;
                        }
                    } else {
                        constraints.push(c);
                    }
                }
            }
            packed.extend(addresses);
        }
        require(
            packed.len() == DOMAIN * IDS
                && constraints.len() + weaker + blocking == DOMAIN * (IDS - 1),
            "incomplete all-rival census",
        )?;
        let nonalias = constraints.len();
        let (pairs, covered) = match_pairs(&mut constraints, &aliases, self.groups);
        let distinct = constraints.len();
        drop(constraints);
        let rows_with_alias = aliases.iter().flatten().count();
        if arm == "snapped" {
            require(
                rows_with_alias == 0
                    && blocking == 0
                    && weaker == 79
                    && nonalias == 15_909_205
                    && distinct == 15_567_316
                    && covered == 3686
                    && pairs.len() == 1334,
                "snapped census did not reconstruct the #4356 summary",
            )?;
        }
        let forced_errors = rows_with_alias + pairs.len();
        let mut proofs = Vec::with_capacity(forced_errors);
        let mut occupied = vec![false; DOMAIN];
        for certificate in aliases
            .into_iter()
            .flatten()
            .map(|c| vec![c])
            .chain(pairs.iter().map(|pair| pair.to_vec()))
        {
            let mut selected = Vec::with_capacity(certificate.len());
            for c in certificate {
                let i = usize::from(c.row);
                require(
                    !std::mem::replace(&mut occupied[i], true),
                    "row reused in combined bound",
                )?;
                let identity = Observation {
                    ordinal: self.common[i],
                    donor: donors[i],
                    group: self.groups[i],
                    gold_id: self.labels[i],
                    rival_id: c.rival,
                    carry: carries[i],
                    packed_slots: [0; 2],
                    q48_priors: [0; 2],
                    q48_row_sha256: self.prior_digests[i].clone(),
                };
                let range = i * IDS..(i + 1) * IDS;
                selected.push(observe(
                    identity,
                    &packed[range.clone()],
                    &self.masses[range],
                ));
            }
            proofs.push(selected);
        }
        drop(packed);
        if arm == "snapped" {
            let same = proofs
                .iter()
                .flatten()
                .zip(self.requests.iter().zip(self.old_proof_slots))
                .all(|(obs, (req, slots))| {
                    obs.ordinal == req.ordinal as usize
                        && obs.group == req.group
                        && obs.gold_id == req.gold_id
                        && obs.rival_id == req.rival_id
                        && obs.q48_priors == [req.gold_prior, req.rival_prior]
                        && obs.packed_slots == *slots
                });
            require(
                same && proofs.iter().map(Vec::len).sum::<usize>() == self.requests.len(),
                "snapped greedy proof membership or priors changed",
            )?;
        }
        self.reread(
            gpu,
            p4,
            arm,
            &proofs,
            &out.join(format!("{arm}-proofs.jsonl")),
        )?;
        let upper = DOMAIN - forced_errors;
        Ok(Summary {
            rows: DOMAIN,
            original_rivals_per_row: IDS - 1,
            nonalias_inequalities: nonalias,
            weaker_full_aliases: weaker,
            blocking_aliases: blocking,
            rows_with_blocking_alias: rows_with_alias,
            row_signature_directions: distinct,
            opposed_rows_across_distinct_groups: covered,
            disjoint_certified_pairs: pairs.len(),
            forced_errors,
            strict_win_upper_bound: upper,
            excludes_retention: upper < BAR,
        })
    }

    fn reread(
        &self,
        gpu: &Gpu,
        p4: &P4GpuEngine,
        arm: &str,
        proofs: &[Vec<Observation>],
        path: &Path,
    ) -> Result<()> {
        let mut writer = BufWriter::new(File::create_new(path)?);
        for batch in proofs.chunks(BATCH / 2) {
            let slots: Vec<_> = batch.iter().flatten().collect();
            let prompts: Vec<_> = slots
                .iter()
                .map(|obs| {
                    let i = self
                        .common
                        .binary_search(&obs.ordinal)
                        .expect("selected ordinal");
                    self.prompts[i].clone()
                })
                .collect();
            let masses = priors(p4, &prompts)?;
            let donor_prompts: Vec<_> = slots
                .iter()
                .map(|obs| {
                    let i = self
                        .common
                        .binary_search(&obs.donor)
                        .expect("selected donor");
                    self.prompts[i].clone()
                })
                .collect();
            let carries: Vec<_> = gpu
                .capture(&donor_prompts)?
                .into_iter()
                .map(|feature| arm_carry(arm, feature))
                .collect();
            require(
                carries.len() == slots.len(),
                "incomplete donor carry reread",
            )?;
            let sites: Vec<_> = slots
                .iter()
                .zip(&carries)
                .flat_map(|(obs, &carry)| {
                    [
                        [carry, u32::from(obs.gold_id)],
                        [carry, u32::from(obs.rival_id)],
                    ]
                })
                .collect();
            let addresses = gpu.addresses(&sites)?;
            require(addresses.len() == sites.len(), "incomplete proof reread")?;
            let mut candidate = 0;
            for first in batch {
                let mut reread = Vec::with_capacity(first.len());
                for original in first {
                    let index = [original.gold_id, original.rival_id].map(candidate_index);
                    let prior_row = &masses[candidate * IDS..(candidate + 1) * IDS];
                    let obs = Observation {
                        carry: carries[candidate],
                        packed_slots: addresses[2 * candidate..2 * candidate + 2].try_into()?,
                        q48_priors: index.map(|id| prior_row[id]),
                        q48_row_sha256: row_digest(prior_row),
                        ..original.clone()
                    };
                    require(
                        obs == *original,
                        "proof address or Q48 mass changed on GPU reread",
                    )?;
                    candidate += 1;
                    reread.push(obs);
                }
                let oriented = |x: &Observation| {
                    inequality(
                        self.common
                            .binary_search(&x.ordinal)
                            .expect("selected ordinal"),
                        x.rival_id,
                        x.packed_slots,
                        x.q48_priors,
                    )
                };
                let kind = match reread.as_slice() {
                    [single] if oriented(single).key == 0 && oriented(single).bound > 0 => "alias",
                    [left, right] if opposite(oriented(left), oriented(right), self.groups) => {
                        "opposing"
                    }
                    _ => return Err(fail("proof inequality failed on independent GPU reread")),
                };
                serde_json::to_writer(
                    &mut writer,
                    &json!({"kind":kind,"first":first,"reread":reread}),
                )?;
                writer.write_all(b"\n")?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}

fn run(
    p4: &Path,
    train_path: &Path,
    witness: &Path,
    frozen: &Path,
    eligibility: &Path,
    out: &Path,
) -> Result<()> {
    require(
        device::digest() == MIDPOINT_SHA
            && hex::encode(Sha256::digest(include_bytes!(env!("P4_QUAL_CUBIN")))) == P4_SHA,
        "census requires the pinned sm_89 modules",
    )?;
    let eligible_bytes = fs::read(eligibility)?;
    require(
        hex::encode(Sha256::digest(&eligible_bytes)) == GATE_SHA,
        "protected summary digest differs from the authorized gate",
    )?;
    let gate: Value = serde_json::from_slice(&eligible_bytes)?;
    let transported = &gate["arms"]["transported"];
    require(
        gate["protected_rows"].as_u64() == Some(2668)
            && gate["source_certificates_sha256"].as_str() == Some(CERTIFICATES_SHA)
            && transported["eligible_for_full_census"].as_bool() == Some(true)
            && transported["cleared_pairs"]
                .as_u64()
                .is_some_and(|n| n >= 557)
            && transported["strict_win_upper_bound"]
                .as_u64()
                .is_some_and(|n| n >= BAR as u64),
        "protected transported arm did not unlock the census",
    )?;
    let train = input::load(train_path)?;
    let common = load_common(frozen, FROZEN_SHA, &train)?;
    let requests = protected::load(witness)?.requests();
    let old_slots = old_proof_slots(witness)?;
    let (donors, width) = donor_ordinals(&train, &common, &requests)?;
    let (midpoint, p4) = load_engines(p4)?;
    let features = midpoint.capture(&train.prefixes)?;
    require(
        features.len() == train.ordinals.len(),
        "missing TRAIN carry",
    )?;
    let prompts: Vec<_> = common.iter().map(|&i| train.prefixes[i].clone()).collect();
    let groups: Vec<_> = common.iter().map(|&i| train.groups[i]).collect();
    let labels: Vec<_> = common.iter().map(|&i| train.targets[i]).collect();
    let mut masses = Vec::with_capacity(DOMAIN * IDS);
    for batch in prompts.chunks(BATCH) {
        masses.extend(priors(&p4, batch)?);
    }
    let mut digest = Sha256::new();
    let mut prior_digests = Vec::with_capacity(DOMAIN);
    for row in masses.chunks_exact(IDS) {
        update_priors(&mut digest, row);
        prior_digests.push(row_digest(row));
    }
    fs::create_dir(out)?;
    save_json(
        &out.join("donors.json"),
        &common.iter().zip(&donors).collect::<Vec<_>>(),
    )?;
    let mut ledger = BufWriter::new(File::create_new(out.join("prior-rows.jsonl"))?);
    let mut digest_of_row_digests = Sha256::new();
    for (&ordinal, row_sha256) in common.iter().zip(&prior_digests) {
        digest_of_row_digests.update(hex::decode(row_sha256)?);
        serde_json::to_writer(&mut ledger, &(ordinal, row_sha256))?;
        ledger.write_all(b"\n")?;
    }
    ledger.flush()?;
    let domain = Domain {
        common: &common,
        groups: &groups,
        labels: &labels,
        prompts: &prompts,
        masses: &masses,
        prior_digests: &prior_digests,
        requests: &requests,
        old_proof_slots: &old_slots,
    };
    let mut arms = serde_json::Map::new();
    for (arm_index, arm) in ARMS.into_iter().enumerate() {
        let source = if arm_index == 3 { &donors } else { &common };
        let carries: Vec<_> = source
            .iter()
            .map(|&i| arm_carry(arm, features[i]))
            .collect();
        let summary = domain.capture(&midpoint, &p4, arm, &carries, source, out)?;
        arms.insert(arm.to_owned(), serde_json::to_value(summary)?);
    }
    save_json(
        &out.join("summary.json"),
        &json!({
            "artifact": input::ARTIFACT, "state": input::STATE,
            "historical_common_sha256": FROZEN_SHA,
        "prior_q48_sha256": hex::encode(digest.finalize()),
        "prior_row_hashes_sha256": hex::encode(digest_of_row_digests.finalize()),
            "eligibility_sha256": hex::encode(Sha256::digest(eligible_bytes)),
            "midpoint_cubin_sha256": device::digest(), "p4_prior_cubin_sha256": p4.cubin_sha256,
            "retention_bar": BAR, "protected_donors": requests.len(),
            "complement_donors": DOMAIN - requests.len(), "complement_rotation": width,
            "matching": "deterministic greedy on rows without a blocking alias",
            "arms": arms,
        }),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "finite" {
        finite_device()?;
        println!("finite census PASS");
        return Ok(());
    }
    let [p4, train, witness, frozen, eligible, out] = args.as_slice() else {
        return Err(fail(
            "expected finite or P4 TRAIN WITNESS_DIR FROZEN_SPARSE PROTECTED_SUMMARY FRESH_OUT",
        ));
    };
    run(
        Path::new(p4),
        Path::new(train),
        Path::new(witness),
        Path::new(frozen),
        Path::new(eligible),
        Path::new(out),
    )
}

fn verify_finite_union(slots: [u32; 3]) -> Result<()> {
    let [a, b, alias] = slots;
    let tie = [UNIT as u64; 2];
    let left = inequality(0, 8, [a, b], tie);
    let right = inequality(1, 7, [b, a], tie);
    let blocked = inequality(2, 9, [b, alias], tie);
    let reusable = inequality(2, 7, [b, a], [UNIT as u64, 3 * UNIT as u64]);
    let groups = [10, 20, 30];
    require(
        a != b
            && b == alias
            && left.bound == 1
            && right.bound == 1
            && blocked.key == 0
            && blocked.bound == 1
            && reusable.bound == 3,
        "finite signed-address or strict-alias fixture differs",
    )?;
    let (pairs, covered) = match_pairs(
        &mut vec![left, right, reusable],
        &[None, None, Some(blocked)],
        &groups,
    );
    require(
        pairs.len() == 1 && pairs[0].iter().map(|site| site.row).sum::<u16>() == 1 && covered == 2,
        "finite combined bound reused the blocking-alias row",
    )?;
    require(
        !opposite(left, right, &[10, 10, 30])
            && !opposite(left, Constraint { bound: -1, ..right }, &groups)
            && inequality(0, 8, [a, b], [2 * UNIT as u64, UNIT as u64]).bound == 0,
        "finite group exclusion or strict threshold differs",
    )
}

fn finite_device() -> Result<()> {
    const ROOT_OFFSET: usize = 70 + IDS + 1;
    let mut trunk = vec![0; ROOT_OFFSET + IDS + 1];
    trunk[0] = 1;
    trunk[ROOT_OFFSET + 8] = 1;
    trunk[ROOT_OFFSET + 9] = 1;
    let gpu = Gpu::new(&trunk)?;
    let first = all_addresses(&gpu, &[0])?;
    let slots = [first[6], first[7], first[8]];
    require(
        gpu.addresses(&[[0, 7], [0, 8], [0, 9]])?.as_slice() == slots,
        "finite all-rival scan and selected address reread differ",
    )?;
    verify_finite_union(slots)
}

#[cfg(test)]
#[test]
fn finite_disjoint_union_arithmetic() -> Result<()> {
    verify_finite_union([0x000204, 0x000207, 0x000207])
}
