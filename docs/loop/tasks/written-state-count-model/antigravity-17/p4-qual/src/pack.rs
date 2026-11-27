use crate::{fail, verify_file_sha256, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use uor_r4_count_runtime::p4_serving::{
    encode_compact_s2_skip_from_uors2s01, encode_compact_s3_from_uors3t02,
};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, write_p4_deployment, ClientArtifactFile, DiscreteCountModel,
};

pub const PIN_TRUNK: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const PIN_TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
pub const PIN_MANIFEST: &str = "26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec";
pub const PIN_A1S_WEIGHTS: &str =
    "e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293";
pub const PIN_COUNTS_V3: &str = "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const PIN_S3_T02: &str = "e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c";
pub const PIN_S3_ERR_P: &str = "0884af50694b119220f15fc7401e41365e038130fe5304039a6dc065bb114c6e";
pub const PIN_S3_ERR_B: &str = "6b1dcaa559850d51d3aa08a4678a9a85c9dfa80fade1c9cfa51003050c4c6fcd";
pub const PIN_S2_S01: &str = "85307b88f6bbc43d57fe8fe896789f9c8ea97e8e25bc24b074a3932e9332c777";
pub const PIN_S2_ERRS: &str = "8c97d916e7fef4a54b547389a61caf8f68124fb0a2f7cafa44823725fc032a10";
pub const PIN_S4: &str = "cd0e43859bcd72acfa08342d7ff08a0827c582c4cd8e1592218e6ca4c43f8771";
pub const PIN_S6: &str = "5783ba3b8b03cb4e9c35ed335b89c8ff36797175e9e1b8d9db40ddfd64f46a2d";
pub const PIN_C_SHA: &str = "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";

pub const MAX_DEPLOYED_BYTES: u64 = 4_000_000_000;

pub struct PackPaths<'a> {
    pub trunk: &'a Path,
    pub tokenizer: &'a Path,
    pub manifest: &'a Path,
    pub a1s_weights: &'a Path,
    pub counts_v3: &'a Path,
    pub s3_t02: &'a Path,
    pub s3_err_p: &'a Path,
    pub s3_err_b: &'a Path,
    pub s2_s01: &'a Path,
    pub s2_errs: &'a Path,
    pub s4: &'a Path,
    pub s6: &'a Path,
    pub out: &'a Path,
    pub report: &'a Path,
}

fn bytes_to_i8(raw: Vec<u8>) -> Vec<i8> {
    raw.into_iter().map(|b| b as i8).collect()
}

pub fn run_pack(p: PackPaths<'_>) -> Result<()> {
    let trunk_bytes = verify_file_sha256(p.trunk, PIN_TRUNK, "seed1.d11")?;
    let tok_bytes = verify_file_sha256(p.tokenizer, PIN_TOKENIZER, "tokenizer.json")?;
    let manifest_bytes = verify_file_sha256(p.manifest, PIN_MANIFEST, "a1s.manifest.json")?;
    let a1s_bytes = verify_file_sha256(p.a1s_weights, PIN_A1S_WEIGHTS, "a1s.weights")?;
    let counts_bytes = verify_file_sha256(p.counts_v3, PIN_COUNTS_V3, "full-c.v3")?;
    let s3_t02_bytes = verify_file_sha256(p.s3_t02, PIN_S3_T02, "golden_shell.uors3t02")?;
    let s3_ep_bytes = verify_file_sha256(p.s3_err_p, PIN_S3_ERR_P, "p5b_s3_err_p.i8")?;
    let s3_eb_bytes = verify_file_sha256(p.s3_err_b, PIN_S3_ERR_B, "p5b_s3_err_b.i8")?;
    let s2_s01_bytes = verify_file_sha256(p.s2_s01, PIN_S2_S01, "s2_skip.uors2s01")?;
    let s2_e_bytes = verify_file_sha256(p.s2_errs, PIN_S2_ERRS, "p5b_s2_skip_errs.i8")?;
    let s4_bytes = verify_file_sha256(p.s4, PIN_S4, "c_ppmi_s4.uors4p01")?;
    let s6_bytes = verify_file_sha256(p.s6, PIN_S6, "p4_full_errwrite.uors6b01")?;

    // Extract c_sha256 from manifest and check consistency.
    let manifest_val: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    let c_sha_hex = manifest_val["prepared"]["c_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing prepared.c_sha256 in manifest"))?;
    if c_sha_hex != PIN_C_SHA {
        return Err(fail(format!(
            "manifest c_sha256 {c_sha_hex} != pinned {PIN_C_SHA}"
        )));
    }
    let mut c_sha = [0u8; 32];
    hex::decode_to_slice(c_sha_hex, &mut c_sha)?;

    let s3_ep_i8 = bytes_to_i8(s3_ep_bytes);
    let s3_eb_i8 = bytes_to_i8(s3_eb_bytes);
    let s3_compact =
        encode_compact_s3_from_uors3t02(&s3_t02_bytes, Some(&s3_ep_i8), Some(&s3_eb_i8), c_sha)
            .map_err(|e| fail(format!("encode_compact_s3_from_uors3t02 failed: {e}")))?;
    drop(s3_t02_bytes);
    drop(s3_ep_i8);
    drop(s3_eb_i8);

    let s2_e_i8 = bytes_to_i8(s2_e_bytes);
    let s2_compact = encode_compact_s2_skip_from_uors2s01(&s2_s01_bytes, Some(&s2_e_i8), c_sha)
        .map_err(|e| fail(format!("encode_compact_s2_skip_from_uors2s01 failed: {e}")))?;
    drop(s2_s01_bytes);
    drop(s2_e_i8);

    let s3_compact_sha = hex::encode(Sha256::digest(&s3_compact));
    let s2_compact_sha = hex::encode(Sha256::digest(&s2_compact));

    if p.out.exists() {
        return Err(fail(format!(
            "output {} already exists (L-002 refuse overwrite)",
            p.out.display()
        )));
    }
    write_p4_deployment(
        &trunk_bytes,
        &manifest_bytes,
        &a1s_bytes,
        &counts_bytes,
        &s3_compact,
        &s2_compact,
        &s4_bytes,
        &s6_bytes,
        p.out,
    )
    .map_err(|e| fail(format!("write_p4_deployment failed: {e}")))?;

    let p4_file_bytes = fs::metadata(p.out)?.len();
    let tok_file_bytes = tok_bytes.len() as u64;
    let total_deployed_bytes = p4_file_bytes + tok_file_bytes;
    let within_4gb_cap = total_deployed_bytes <= MAX_DEPLOYED_BYTES;

    // Cold reload verification via load_client_artifact + bind_embedded
    let reloaded = load_client_artifact(p.out, false)
        .map_err(|e| fail(format!("cold load_client_artifact failed: {e}")))?;
    let (artifact_sha256, state_sha256, component_digests) = match reloaded {
        ClientArtifactFile::Embedded {
            model,
            state_bytes,
            expected_state_sha256,
            identity,
        } => {
            let bound = DiscreteCountModel::bind_embedded(
                model,
                &state_bytes,
                &expected_state_sha256,
                identity,
            )
            .map_err(|e| fail(format!("cold bind_embedded failed: {e}")))?;
            let p4_ref = bound
                .p4
                .as_ref()
                .ok_or_else(|| fail("cold reload did not bind P4"))?;
            let id_ref = bound
                .identity
                .as_ref()
                .ok_or_else(|| fail("missing deployment identity"))?;
            (
                id_ref.artifact_sha256.clone(),
                p4_ref.artifact_sha256.clone(),
                p4_ref.component_digests.clone(),
            )
        }
        ClientArtifactFile::Legacy(_) => {
            return Err(fail(
                "cold reload parsed p4.d11 as Legacy instead of Embedded",
            ));
        }
    };

    let report = json!({
        "schema": "uor-r4.p4-pack-manifest/1",
        "c_sha256": c_sha_hex,
        "p4_path": p.out.display().to_string(),
        "p4_bytes": p4_file_bytes,
        "tokenizer_bytes": tok_file_bytes,
        "total_deployed_bytes": total_deployed_bytes,
        "cap_bytes": MAX_DEPLOYED_BYTES,
        "within_4gb_cap": within_4gb_cap,
        "artifact_sha256": artifact_sha256,
        "state_sha256": state_sha256,
        "section_bytes": {
            "trunk": trunk_bytes.len(),
            "fit_manifest": manifest_bytes.len(),
            "a1s_weights": a1s_bytes.len(),
            "counts_v3": counts_bytes.len(),
            "s3_compact": s3_compact.len(),
            "s2_skip_compact": s2_compact.len(),
            "s4_table": s4_bytes.len(),
            "s6_weights": s6_bytes.len(),
        },
        "s3_compact_sha256": s3_compact_sha,
        "s2_skip_compact_sha256": s2_compact_sha,
        "cold_reload_component_digests": component_digests,
    });
    fs::write(p.report, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !within_4gb_cap {
        return Err(fail(format!(
            "total deployed bytes {total_deployed_bytes} exceeds 4,000,000,000 B cap"
        )));
    }
    Ok(())
}
