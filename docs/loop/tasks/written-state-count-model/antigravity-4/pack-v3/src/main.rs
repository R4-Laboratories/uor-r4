//! UORKNC03 uor-r4.lut-discrete-count/1 deployment packer & verifier.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, error::Error, fs, path::Path};
use uor_r4_count_runtime::{FrozenPickMixture, PickClassRule};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, write_deployment, ClientArtifactFile, DiscreteCountModel,
};

type Failure = Box<dyn Error>;
type Result<T> = std::result::Result<T, Failure>;

const C_SHA: &str = "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
const R_SHA: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
const TRUNK_SHA: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
const COMPACT_CAP_BYTES: u64 = 3_912_936_331;
const MAX_DEPLOY_BYTES: u64 = 4_000_000_000;

fn fail(msg: impl Into<String>) -> Failure {
    msg.into().into()
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

fn pin(hex_str: &str) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut out)?;
    Ok(out)
}

fn build_p1_state(
    manifest_bytes: &[u8],
    weights_bytes: &[u8],
    compact_bytes: &[u8],
) -> Result<Vec<u8>> {
    if compact_bytes.len() as u64 > COMPACT_CAP_BYTES {
        return Err(fail("compact bytes exceed COMPACT_CAP_BYTES"));
    }
    let manifest_len = u32::try_from(manifest_bytes.len())?;
    let weights_len = u32::try_from(weights_bytes.len())?;
    let mut state = Vec::with_capacity(
        48 + manifest_bytes.len() + weights_bytes.len() + compact_bytes.len(),
    );
    state.extend_from_slice(b"UORP1A01");
    state.extend_from_slice(&manifest_len.to_le_bytes());
    state.extend_from_slice(&weights_len.to_le_bytes());
    state.extend_from_slice(&Sha256::digest(weights_bytes));
    state.extend_from_slice(manifest_bytes);
    state.extend_from_slice(weights_bytes);
    state.extend_from_slice(compact_bytes);
    Ok(state)
}

fn pack_deploy(
    trunk_path: &Path,
    compact_path: &Path,
    compact_sha: &str,
    fit_dir: &Path,
    arm: &str,
    out_d11: &Path,
) -> Result<()> {
    if out_d11.exists() {
        return Err(fail(format!(
            "refusing to overwrite {}",
            out_d11.display()
        )));
    }
    let rule = match arm {
        "a1" => PickClassRule::Three,
        "a1s" => PickClassRule::Five,
        _ => return Err(fail("pack-v3 arm must be a1 or a1s")),
    };
    pin(compact_sha)?;
    let compact_len = fs::metadata(compact_path)?.len();
    if compact_len == 0 || compact_len > COMPACT_CAP_BYTES {
        return Err(fail("compact artifact outside 1..=COMPACT_CAP_BYTES"));
    }
    if file_sha(compact_path)? != compact_sha {
        return Err(fail("compact artifact SHA-256 mismatch"));
    }
    if file_sha(trunk_path)? != TRUNK_SHA {
        return Err(fail("trunk SHA-256 mismatch"));
    }
    let manifest_bytes = fs::read(fit_dir.join(format!("{arm}.manifest.json")))?;
    let manifest_sha = sha(&manifest_bytes);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let prepared = &manifest["prepared"];
    if manifest["schema"] != "uor-r4.d37.r-fit/1"
        || manifest["arm"] != arm
        || prepared["format"] != "uor-r4.d37.r-preparation/1"
        || prepared["pick_mode"] != "identity"
        || prepared["c_sha256"] != C_SHA
        || prepared["r_sha256"] != R_SHA
        || prepared["trunk_sha256"] != TRUNK_SHA
        || prepared["compact_sha256"] != compact_sha
    {
        return Err(fail("fit manifest identity mismatch"));
    }
    let weights_bytes = fs::read(fit_dir.join(format!("{arm}.weights")))?;
    let frozen = FrozenPickMixture::from_bytes(&weights_bytes, pin(&manifest_sha)?)?;
    if frozen.rule() != rule {
        return Err(fail("Pick class rule mismatch"));
    }

    let trunk = fs::read(trunk_path)?;
    let compact_bytes = fs::read(compact_path)?;
    let state = build_p1_state(&manifest_bytes, &weights_bytes, &compact_bytes)?;
    drop(compact_bytes);
    let state_sha = sha(&state);
    write_deployment(&trunk, &state, out_d11)?;
    drop(trunk);
    drop(state);

    let deploy_len = fs::metadata(out_d11)?.len();
    if deploy_len > MAX_DEPLOY_BYTES {
        return Err(fail(format!(
            "packed deployment {deploy_len} B exceeds 4.000 GB decimal limit"
        )));
    }
    match load_client_artifact(out_d11, false)? {
        ClientArtifactFile::Embedded {
            model,
            state_bytes,
            expected_state_sha256,
            identity,
        } => {
            if expected_state_sha256 != state_sha || model.artifact_sha256() != TRUNK_SHA {
                return Err(fail("streaming readback identity mismatch"));
            }
            let bound = DiscreteCountModel::bind_embedded(
                model,
                &state_bytes,
                &expected_state_sha256,
                identity,
            )?;
            let p1 = bound.state.ok_or_else(|| fail("missing bound P1 state"))?;
            if p1.component_digests()["count_sha256"] != compact_sha {
                return Err(fail("count_sha256 readback mismatch"));
            }
            println!(
                "{}",
                json!({
                    "schema": "uor-r4.b1a.pack-deploy/1",
                    "arm": arm,
                    "deploy_bytes": deploy_len,
                    "deploy_sha256": bound.identity.as_ref().map(|i| i.artifact_sha256.clone()),
                    "trunk_sha256": TRUNK_SHA,
                    "compact_bytes": compact_len,
                    "compact_sha256": compact_sha,
                    "manifest_sha256": manifest_sha,
                    "weights_sha256": sha(&weights_bytes),
                    "state_sha256": state_sha,
                    "under_4gb_decimal": deploy_len <= MAX_DEPLOY_BYTES,
                })
            );
        }
        ClientArtifactFile::Legacy(_) => return Err(fail("expected embedded deployment")),
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(fail(
            "usage: pack-v3 TRUNK COMPACT COMPACT_SHA FIT_DIR (a1|a1s) OUT_D11",
        ));
    }
    pack_deploy(
        Path::new(&args[0]),
        Path::new(&args[1]),
        &args[2],
        Path::new(&args[3]),
        &args[4],
        Path::new(&args[5]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_p1_state_formats_header_and_refuses_oversize() -> Result<()> {
        let state = build_p1_state(b"{}", b"weights", b"UORKNC03")?;
        assert_eq!(&state[..8], b"UORP1A01");
        assert_eq!(u32::from_le_bytes(state[8..12].try_into()?), 2);
        assert_eq!(u32::from_le_bytes(state[12..16].try_into()?), 7);
        assert_eq!(&state[16..48], Sha256::digest(b"weights").as_slice());
        assert!(build_p1_state(b"{}", b"w", &vec![0u8; COMPACT_CAP_BYTES as usize + 1]).is_err());
        Ok(())
    }
}
