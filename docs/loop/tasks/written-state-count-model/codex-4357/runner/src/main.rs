use serde_json::json;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Write, path::Path};

use uor_r4_integer::discrete_d11::{
    count_deployment::{load_client_artifact, ClientArtifactFile},
    native_occurrence::lexical_commit::LexicalHead,
    P4LexicalCarry,
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail(msg: impl Into<String>) -> Box<dyn Error> {
    msg.into().into()
}

#[cfg(feature = "cuda")]
mod device;
#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/input.rs"]
mod input;
#[cfg(any(test, feature = "cuda"))]
mod protected;

const ARMS: [(&str, P4LexicalCarry); 3] = [
    ("snapped", P4LexicalCarry::Snapped),
    ("transported", P4LexicalCarry::TransportedMidpoint),
    ("raw", P4LexicalCarry::RawMidpoint),
];

fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::File::create_new(path)?.write_all(bytes)?;
    Ok(())
}

// Pure artifact extraction. No prompt is executed or fitted in this entry.
fn extract(p4: &Path, head: &Path, out: &Path) -> Result<()> {
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(p4, false)?
    else {
        return Err(fail("requires the embedded frozen P4"));
    };
    if identity.artifact_sha256 != input::ARTIFACT
        || expected_state_sha256 != input::STATE
        || hex::encode(Sha256::digest(&state_bytes)) != input::STATE
    {
        return Err(fail("P4 identity differs"));
    }
    drop(state_bytes);
    let state: [u8; 32] = hex::decode(input::STATE)?
        .try_into()
        .map_err(|_| "bad state digest")?;
    let old = fs::read(head)?;
    // #4356 uncentered full geometric seed 435300 is the byte-exact #4355 attachment.
    if hex::encode(Sha256::digest(&old))
        != "e5516223485f2332f1e3c4df5a2887daa7d591bb61149fe677bdc6fa5c2dc809"
        || old.len() != 584
    {
        return Err(fail("unexpected original weights"));
    }
    let words: [i16; 256] =
        std::array::from_fn(|i| i16::from_le_bytes([old[40 + 2 * i], old[41 + 2 * i]]));
    let trunk: Vec<u8> = model
        .p4_lexical_trunk_words()?
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    fs::create_dir(out)?;
    save(&out.join("trunk.bin"), &trunk)?;
    for (name, op) in ARMS {
        let head = LexicalHead::encode_geometric_bound(state, op, &words);
        if name == "snapped" && old.as_slice() != head {
            return Err(fail("snapped attachment failed round trip"));
        }
        save(&out.join(format!("{name}.lex")), &head)?;
    }
    println!(
        "{}",
        json!({"artifact":input::ARTIFACT,"state":input::STATE,
        "trunk_bytes":trunk.len(),"trunk_sha256":hex::encode(Sha256::digest(&trunk)),
        "original_lex_sha256":hex::encode(Sha256::digest(old))})
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn suffix_summary(train: &input::Train, features: &[device::Feature]) -> Result<serde_json::Value> {
    use std::collections::BTreeMap;
    let mut classes: BTreeMap<&[u16], Vec<(usize, &[u16])>> = BTreeMap::new();
    for (ordinal, prefix) in train.prefixes.iter().enumerate() {
        let start = prefix.iter().rposition(|&id| id == 0).unwrap_or(0);
        let active = &prefix[start..];
        let mid = (active.len().checked_sub(1).ok_or("empty active prefix")?) / 2;
        classes
            .entry(&active[mid..])
            .or_default()
            .push((ordinal, &active[..mid]));
    }
    let mut differing_prefix_pairs = 0u64;
    let mut differing_target_pairs = 0u64;
    let mut differing_snapped_pairs = 0u64;
    for rows in classes.values() {
        for (j, &(a, left)) in rows.iter().enumerate() {
            for &(b, right) in &rows[..j] {
                if train.groups[a] == train.groups[b] || left == right {
                    continue;
                }
                if features[a].transported != features[b].transported
                    || features[a].raw != features[b].raw
                {
                    return Err(fail(
                        "identical real suffix produced different midpoint roots",
                    ));
                }
                differing_prefix_pairs += 1;
                differing_target_pairs += u64::from(train.targets[a] != train.targets[b]);
                differing_snapped_pairs += u64::from(features[a].snapped != features[b].snapped);
            }
        }
    }
    Ok(
        json!({"cross_group_pairs_with_distinct_earlier_text_and_exact_active_midpoint_suffix":differing_prefix_pairs,
        "different_targets":differing_target_pairs,"different_snapped_carry":differing_snapped_pairs,
        "classification":"original-ID suffix equality; semantic question labels unavailable"}),
    )
}

#[cfg(feature = "cuda")]
fn run(
    trunk_path: &Path,
    expected_hash: &str,
    input_dir: &Path,
    proofs: &Path,
    out: &Path,
) -> Result<()> {
    let bytes = fs::read(trunk_path)?;
    if expected_hash.len() != 64
        || hex::encode(Sha256::digest(&bytes)) != expected_hash
        || bytes.len() % 4 != 0
    {
        return Err(fail("trunk digest/word shape mismatch"));
    }
    let trunk = bytes
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    let gpu = device::Gpu::new(&trunk)?;
    let train = input::load(input_dir)?;
    let protected = protected::load(proofs)?;
    let requests = protected.requests();
    let features = gpu.capture(&train.prefixes)?;
    if features.len() != train.ordinals.len() || features.iter().any(|f| f.status != 0) {
        return Err(fail("inconsistent device capture"));
    }
    let mut sites = Vec::with_capacity(requests.len() * 8);
    let mut carries = Vec::with_capacity(requests.len());
    for req in &requests {
        let i = usize::try_from(req.ordinal)?;
        let f = features.get(i).ok_or("unknown protected row")?;
        if train.groups[i] != req.group
            || train.targets[i] != req.gold_id
            || train.ordinals[i] != req.ordinal
        {
            return Err(fail("protected row disagrees with authentic admission"));
        }
        let donor = features
            .get(req.swap_ordinal as usize)
            .ok_or("missing shuffled donor")?;
        if train.groups[req.swap_ordinal as usize] == req.group {
            return Err(fail("within-group shuffled donor"));
        }
        let choices = [f.snapped, f.transported, f.raw, donor.transported];
        for carry in choices {
            for id in [req.gold_id, req.rival_id] {
                sites.push([carry, id.into()]);
            }
        }
        carries.push(choices);
    }
    let addresses = gpu.addresses(&sites)?;
    if addresses.len() != sites.len() {
        return Err(fail("incomplete protected device reads"));
    }
    let captures: Vec<_> = requests
        .iter()
        .zip(carries)
        .zip(addresses.chunks_exact(8))
        .map(|((req, carries), x)| protected::Capture {
            carries,
            addresses: std::array::from_fn(|i| [x[2 * i], x[2 * i + 1]]),
            priors: [req.gold_prior, req.rival_prior],
        })
        .collect();
    fs::create_dir(out)?;
    save(
        &out.join("suffix-sensitivity.json"),
        &serde_json::to_vec(&suffix_summary(&train, &features)?)?,
    )?;
    protected.score(&captures, out)?;
    let f = &features[0];
    let probe = [f.snapped, f.transported, f.raw];
    let all_sites: Vec<_> = probe
        .iter()
        .flat_map(|&c| (1..=4095).map(move |id| [c, id]))
        .collect();
    let all = gpu.addresses(&all_sites)?;
    if all.len() != all_sites.len() {
        return Err(fail("incomplete ordinary sentinel sites"));
    }
    save(
        &out.join("sentinel.json"),
        &serde_json::to_vec(
            &json!({"ordinal":0,"carries":probe,"original_ids":"1 through 4095 in order",
        "packed_addresses":all.chunks_exact(4095).collect::<Vec<_>>()}),
        )?,
    )?;
    save(
        &out.join("capture.json"),
        &serde_json::to_vec(
            &json!({"admissions":features.len(),"protected_sides":requests.len(),"trunk_sha256":expected_hash,
        "module_sha256":device::digest(),"artifact":input::ARTIFACT,"state":input::STATE}),
        )?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    match a.as_slice() {
        [cmd, p4, head, out] if cmd == "extract" => {
            extract(p4.as_ref(), head.as_ref(), out.as_ref())
        }
        #[cfg(feature = "cuda")]
        [cmd] if cmd == "finite" => {
            device::finite()?;
            println!("finite PASS");
            Ok(())
        }
        #[cfg(feature = "cuda")]
        [cmd, trunk, hash, input, proofs, out] if cmd == "protected" => run(
            trunk.as_ref(),
            hash,
            input.as_ref(),
            proofs.as_ref(),
            out.as_ref(),
        ),
        _ => Err(fail(
            "expected extract P4 HEAD NEW_OUT, finite, or protected TRUNK SHA INPUT PROOFS NEW_OUT",
        )),
    }
}
