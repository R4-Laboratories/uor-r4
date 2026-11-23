//! Three-host M7 bit-identity and fresh-process `UR4SESS1` snapshot parity suite
//! across the Rust library (`IntegerStackModel` / `IntegerStackSession`), the
//! `uor-chat` CLI (`--stack`), and the Browser WASM / Studio Web Worker
//! (`m7-parity-library-cli-browser`, Issue #2867, Milestone #2035).
//!
//! One panel runner serves two inputs:
//! - the pinned seed-5 synthetic fixture (`mk-synthetic-stack`), kept as the
//!   regression that proves the harness itself works; and
//! - a candidate `UORLUT01` artifact with an embedded tokenizer SHA-256 and
//!   its matching `tokenizer.json`, named by
//!   `UOR_M7_CANDIDATE_ARTIFACT` and `UOR_M7_CANDIDATE_TOKENIZER` (optional
//!   SHA-256 pins `UOR_M7_CANDIDATE_ARTIFACT_SHA256` and
//!   `UOR_M7_CANDIDATE_TOKENIZER_SHA256`), run with `--ignored`.
//!
//! Every leg's receipt must name the candidate's measured artifact and tokenizer
//! digests, and every library `UR4SESS1` snapshot must carry the candidate's
//! artifact digest, so no host can substitute the fixture unnoticed. A pass on
//! the fixture or on any synthetic candidate validates the harness; the
//! `m7-parity-prompts-v1` bar counts only on the accepted M4 artifact.

#![cfg(feature = "serving-wasm")]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use uor_r4_integer::session_codec;
use uor_r4_integer::stack::{
    stack_argmax, FbsExportProvenance, IntegerStackModel, StackContractProvenance,
    STACK_FBS_BOUND_SCHEMA,
};

const SYNTHETIC_SEED: u64 = 5;
const EXPECTED_ARTIFACT_BYTES: usize = 469_252;
const EXPECTED_ARTIFACT_SHA256: &str =
    "d60f88d6e08ac60763b35484307f9425f702b9fead4615dd8cbe5fdbbced92df";
const EXPECTED_WEIGHTS_PER_TOKEN: u64 = 15_968;

/// Tokens per prompt in the frozen protocol, and its fresh-process split.
const PANEL_TOKENS: usize = 48;
const SPLIT_FIRST: usize = 32;
const SPLIT_REST: usize = PANEL_TOKENS - SPLIT_FIRST;
const PANEL_ID: &str = "m7-parity-prompts-v1";

const ENV_CANDIDATE_ARTIFACT: &str = "UOR_M7_CANDIDATE_ARTIFACT";
const ENV_CANDIDATE_TOKENIZER: &str = "UOR_M7_CANDIDATE_TOKENIZER";
const ENV_CANDIDATE_ARTIFACT_SHA256: &str = "UOR_M7_CANDIDATE_ARTIFACT_SHA256";
const ENV_CANDIDATE_TOKENIZER_SHA256: &str = "UOR_M7_CANDIDATE_TOKENIZER_SHA256";

struct PromptCase {
    id: &'static str,
    prompt_ids: &'static [u32],
}

const M7_PARITY_PROMPTS: &[PromptCase] = &[
    PromptCase {
        id: "P0_bos",
        prompt_ids: &[0],
    },
    PromptCase {
        id: "P1_quad",
        prompt_ids: &[0, 7, 19, 31],
    },
    PromptCase {
        id: "P2_boundary",
        prompt_ids: &[0, 1, 38, 39],
    },
    PromptCase {
        id: "P3_fibonacci",
        prompt_ids: &[0, 1, 2, 3, 5, 8, 13, 21],
    },
    PromptCase {
        id: "P4_primes",
        prompt_ids: &[0, 2, 3, 5, 7, 11, 13, 17],
    },
    PromptCase {
        id: "P5_alternating",
        prompt_ids: &[0, 39, 1, 38, 2, 37, 3, 36],
    },
    PromptCase {
        id: "P6_repeated",
        prompt_ids: &[0, 17, 17, 17, 25, 25],
    },
    PromptCase {
        id: "P7_octave",
        prompt_ids: &[0, 4, 8, 12, 16, 20, 24, 28],
    },
];

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Deterministic 40-token byte-level BPE `tokenizer.json` matching the `vocab = 40`
/// synthetic stack (`mk-synthetic-stack`) and binding `uor-r4.literal-role-dialogue/1`.
fn build_synthetic_40_tokenizer_json() -> Vec<u8> {
    let pieces: [&str; 40] = [
        "<|bos|>", "<|eos|>", "<|unk|>", ":", "Ġ", "Ċ", "U", "s", "e", "r", "A", "i", "t", "a",
        "n", "S", "y", "m", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "b", "c", "d", "f",
        "g", "h", "k", "l", "o", "p", "q", "v",
    ];
    let mut vocab = serde_json::Map::with_capacity(pieces.len());
    for (idx, piece) in pieces.iter().enumerate() {
        vocab.insert((*piece).to_string(), json!(idx as u32));
    }
    let doc = json!({
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": []
        },
        "added_tokens": [
            { "id": 0, "content": "<|bos|>" },
            { "id": 1, "content": "<|eos|>" },
            { "id": 2, "content": "<|unk|>" }
        ],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false
        }
    });
    serde_json::to_vec_pretty(&doc).expect("serialize 40-token tokenizer.json")
}

/// The artifact and tokenizer one panel run is bound to, with digests measured
/// from the bytes on disk (never taken from a host's own report).
struct Candidate {
    artifact_path: PathBuf,
    tokenizer_path: PathBuf,
    artifact_bytes: Vec<u8>,
    artifact_sha256: String,
    tokenizer_sha256: String,
}

/// Read both files, measure their digests and check any supplied pins.
fn bind_candidate(
    artifact_path: &Path,
    tokenizer_path: &Path,
    pinned_artifact_sha256: Option<&str>,
    pinned_tokenizer_sha256: Option<&str>,
) -> Result<Candidate, String> {
    let artifact_bytes = fs::read(artifact_path)
        .map_err(|e| format!("cannot read artifact {}: {e}", artifact_path.display()))?;
    let tokenizer_bytes = fs::read(tokenizer_path)
        .map_err(|e| format!("cannot read tokenizer {}: {e}", tokenizer_path.display()))?;
    let artifact_sha256 = sha256_hex(&artifact_bytes);
    let tokenizer_sha256 = sha256_hex(&tokenizer_bytes);
    if let Some(pin) = pinned_artifact_sha256 {
        if !pin.eq_ignore_ascii_case(&artifact_sha256) {
            return Err(format!(
                "artifact sha256 {artifact_sha256} does not match the pin {pin}"
            ));
        }
    }
    if let Some(pin) = pinned_tokenizer_sha256 {
        if !pin.eq_ignore_ascii_case(&tokenizer_sha256) {
            return Err(format!(
                "tokenizer sha256 {tokenizer_sha256} does not match the pin {pin}"
            ));
        }
    }
    Ok(Candidate {
        artifact_path: artifact_path.to_path_buf(),
        tokenizer_path: tokenizer_path.to_path_buf(),
        artifact_bytes,
        artifact_sha256,
        tokenizer_sha256,
    })
}

/// A candidate run must never be the seed-5 fixture under another name.
fn refuse_fixture_substitution(candidate_artifact_sha256: &str) -> Result<(), String> {
    if candidate_artifact_sha256 == EXPECTED_ARTIFACT_SHA256 {
        return Err(format!(
            "candidate artifact is the seed-{SYNTHETIC_SEED} synthetic fixture \
             ({EXPECTED_ARTIFACT_SHA256}); a candidate run must name another artifact"
        ));
    }
    Ok(())
}

fn require_candidate_tokenizer_binding(
    model: &IntegerStackModel,
    tokenizer_sha256: &str,
) -> Result<(), String> {
    if model.tokenizer_sha256().is_none() {
        return Err("candidate artifact has no embedded tokenizer SHA-256 binding".to_owned());
    }
    model
        .require_tokenizer_sha256(tokenizer_sha256)
        .map_err(|error| error.to_string())
}

/// Pure identity check for one host's receipt: it must report exactly the
/// candidate's measured digests. A missing field is a failure, not a pass.
fn check_leg_identity(
    leg: &str,
    candidate: (&str, &str),
    observed_artifact_sha256: Option<&str>,
    observed_tokenizer_sha256: Option<&str>,
) -> Result<(), String> {
    let (expected_artifact, expected_tokenizer) = candidate;
    match observed_artifact_sha256 {
        Some(observed) if observed == expected_artifact => {}
        observed => {
            return Err(format!(
                "{leg}: artifact sha256 {observed:?} != candidate {expected_artifact}"
            ));
        }
    }
    match observed_tokenizer_sha256 {
        Some(observed) if observed == expected_tokenizer => {}
        observed => {
            return Err(format!(
                "{leg}: tokenizer sha256 {observed:?} != candidate {expected_tokenizer}"
            ));
        }
    }
    Ok(())
}

struct LibraryRunOut {
    generated_ids: Vec<u32>,
    last_token: u32,
    session_bytes: Vec<u8>,
    session_sha256: String,
}

fn run_library_sequence(
    model: &IntegerStackModel,
    restore_bytes: Option<&[u8]>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<LibraryRunOut, String> {
    let mut session = model.session();
    if let Some(raw) = restore_bytes {
        let parsed = session_codec::decode(raw)?;
        session.restore_state(&parsed).map_err(|e| format!("{e}"))?;
    }
    let mut tok = prompt_ids[0];
    for &next_prompt_tok in &prompt_ids[1..] {
        session.step(tok).map_err(|e| format!("{e}"))?;
        tok = next_prompt_tok;
    }
    let mut generated_ids = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let logits = session.step(tok).map_err(|e| format!("{e}"))?;
        tok = stack_argmax(logits) as u32;
        generated_ids.push(tok);
    }
    let session_bytes = session_codec::encode(&session.save_state());
    let session_sha256 = sha256_hex(&session_bytes);
    Ok(LibraryRunOut {
        generated_ids,
        last_token: tok,
        session_bytes,
        session_sha256,
    })
}

/// The artifact digest a `UR4SESS1` snapshot is bound to.
fn snapshot_artifact_sha256(session_bytes: &[u8]) -> Result<String, String> {
    Ok(session_codec::decode(session_bytes)?.artifact_sha256)
}

struct CliRunOut {
    generated_ids: Vec<u32>,
    last_token: u32,
    session_bytes: Vec<u8>,
    session_sha256: String,
    weights_per_token: u64,
    artifact_sha256: Option<String>,
    tokenizer_sha256: Option<String>,
}

fn run_cli_sequence(
    artifact_path: &Path,
    tokenizer_path: &Path,
    restore_path: Option<&Path>,
    save_path: &Path,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<CliRunOut, String> {
    let uor_chat_bin = env!("CARGO_BIN_EXE_uor-chat");
    let csv = prompt_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut cmd = Command::new(uor_chat_bin);
    cmd.arg("--stack")
        .arg(artifact_path)
        .arg("--tokenizer")
        .arg(tokenizer_path)
        .arg("--prompt-ids")
        .arg(&csv)
        .arg("--max-tokens")
        .arg(max_new_tokens.to_string())
        .arg("--save-session")
        .arg(save_path);
    if let Some(restore) = restore_path {
        cmd.arg("--restore-session").arg(restore);
    }
    let output = cmd
        .output()
        .map_err(|e| format!("failed to spawn uor-chat: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "uor-chat exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let parsed: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("invalid uor-chat JSON output: {e}"))?;
    if parsed["schema"].as_str() != Some("uor-r4.uor-chat-stack-parity/1") {
        return Err(format!("unexpected uor-chat schema: {}", parsed["schema"]));
    }
    let generated_ids = parsed["generated_ids"]
        .as_array()
        .ok_or_else(|| "missing generated_ids array".to_string())?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| "non-u32 generated id".to_string())
        })
        .collect::<Result<Vec<u32>, String>>()?;
    let last_token = parsed["last_token"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| "missing last_token".to_string())?;
    let weights_per_token = parsed["weights_per_token"]
        .as_u64()
        .ok_or_else(|| "missing weights_per_token".to_string())?;
    let session_bytes =
        fs::read(save_path).map_err(|e| format!("failed to read saved CLI session: {e}"))?;
    let session_sha256 = sha256_hex(&session_bytes);
    if parsed["session_sha256"].as_str() != Some(session_sha256.as_str()) {
        return Err(format!(
            "CLI JSON session_sha256 ({:?}) != saved file sha256 ({session_sha256})",
            parsed["session_sha256"].as_str()
        ));
    }
    Ok(CliRunOut {
        generated_ids,
        last_token,
        session_bytes,
        session_sha256,
        weights_per_token,
        artifact_sha256: parsed["artifact_sha256"].as_str().map(str::to_owned),
        tokenizer_sha256: parsed["tokenizer_sha256"].as_str().map(str::to_owned),
    })
}

/// Pure comparison helper for three-leg token and snapshot parity (exercised directly
/// by both the positive 8-prompt panel test and the planted-fault tamper controls).
#[allow(clippy::too_many_arguments)]
fn check_three_way_row_parity(
    prompt_id: &str,
    lib_tokens_48: &[u32],
    cli_tokens_48: &[u32],
    browser_tokens_48: &[u32],
    lib_sha32: &str,
    cli_sha32: &str,
    browser_sha32: &str,
    lib_sha48: &str,
    cli_sha48: &str,
    browser_sha48: &str,
    lib_restored_sha48: &str,
    cli_restored_sha48: &str,
    browser_restored_sha48: &str,
) -> Result<(), String> {
    if lib_tokens_48 != cli_tokens_48 {
        return Err(format!(
            "{prompt_id}: library vs uor-chat CLI 48-token trajectory mismatch"
        ));
    }
    if lib_tokens_48 != browser_tokens_48 {
        return Err(format!(
            "{prompt_id}: library vs browser WASM 48-token trajectory mismatch"
        ));
    }
    if lib_sha32 != cli_sha32 || lib_sha32 != browser_sha32 {
        return Err(format!(
            "{prompt_id}: 32-step UR4SESS1 snapshot sha256 mismatch (lib={lib_sha32}, cli={cli_sha32}, browser={browser_sha32})"
        ));
    }
    if lib_sha48 != cli_sha48 || lib_sha48 != browser_sha48 {
        return Err(format!(
            "{prompt_id}: 48-step UR4SESS1 snapshot sha256 mismatch (lib={lib_sha48}, cli={cli_sha48}, browser={browser_sha48})"
        ));
    }
    if lib_restored_sha48 != lib_sha48
        || cli_restored_sha48 != lib_sha48
        || browser_restored_sha48 != lib_sha48
    {
        return Err(format!(
            "{prompt_id}: 32+16 restored UR4SESS1 snapshot sha256 mismatch against uninterrupted 48-step snapshot"
        ));
    }
    Ok(())
}

fn make_temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "uor-m7-parity-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

/// Run the browser leg (`web/studio/verify_m7_parity.mjs`) on named inputs.
fn run_browser_leg(
    artifact_path: &Path,
    tokenizer_path: &Path,
) -> std::io::Result<std::process::Output> {
    // The harness may build the default WASM package on the first invocation.
    static PACKAGE: Mutex<()> = Mutex::new(());
    let _package = PACKAGE.lock().expect("browser package lock");
    Command::new("node")
        .arg(repo_root().join("web/studio/verify_m7_parity.mjs"))
        .env("UOR_STACK_PATH", artifact_path)
        .env("UOR_TOKENIZER_PATH", tokenizer_path)
        .output()
}

fn generate_synthetic_stack(seed: u64, out: &Path) {
    let mk_out = Command::new(env!("CARGO_BIN_EXE_mk-synthetic-stack"))
        .arg(seed.to_string())
        .arg(out)
        .output()
        .expect("run mk-synthetic-stack");
    assert!(
        mk_out.status.success(),
        "mk-synthetic-stack failed: {}",
        String::from_utf8_lossy(&mk_out.stderr)
    );
}

/// Reissue a generated test fixture with the real /10 provenance shape, leaving
/// the entire aligned payload unchanged. The parsed model validates the result.
fn bind_synthetic_fixture(artifact: &[u8], tokenizer_sha256: &str) -> Vec<u8> {
    const ALIGN: usize = 64;
    let header_len = u64::from_le_bytes(artifact[8..16].try_into().expect("fixture header length"));
    let header_end = 16 + usize::try_from(header_len).expect("fixture header fits usize");
    let mut header: Value =
        serde_json::from_slice(&artifact[16..header_end]).expect("fixture header");
    assert_eq!(header["source"]["synthetic"].as_bool(), Some(true));
    let provenance = FbsExportProvenance {
        stack_contract: StackContractProvenance {
            source: "synthetic M7 binding control".to_owned(),
            // This fixture exercises the provenance shape and tokenizer binding.
            // It does not identify a research or release contract.
            sha256: "0".repeat(64),
            semantic_blake3: "0".repeat(64),
        },
        tokenizer_sha256: tokenizer_sha256.to_owned(),
    };
    header["schema"] = json!(STACK_FBS_BOUND_SCHEMA);
    header["source"]["stack_contract"] = json!(provenance.stack_contract);
    header["source"]["tokenizer_sha256"] = json!(provenance.tokenizer_sha256);
    let new_header = serde_json::to_vec(&header).expect("serialize bound fixture header");
    let mut bytes = artifact[..8].to_vec();
    bytes.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&new_header);
    bytes.resize(bytes.len().div_ceil(ALIGN) * ALIGN, 0);
    bytes.extend_from_slice(&artifact[header_end.div_ceil(ALIGN) * ALIGN..]);
    bytes
}

/// What one full panel run compared, for the receipt.
struct PanelReceipt {
    prompts_passed: usize,
    tokens_compared: usize,
    snapshots_compared: usize,
    weights_per_token: u64,
    vocab: usize,
    wasm_sha256: String,
    wasm_glue_sha256: String,
    worker_js_sha256: String,
}

fn measured_sha256<'a>(report: &'a Value, field: &str) -> &'a str {
    let digest = report[field]
        .as_str()
        .expect("missing browser build digest");
    assert!(
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "browser {field} is not a SHA-256 digest"
    );
    digest
}

fn browser_token_ids(row: &Value, field: &str) -> Vec<u32> {
    row[field]
        .as_array()
        .expect("browser token-id array")
        .iter()
        .map(|id| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .expect("browser token ID fits u32")
        })
        .collect()
}

enum PanelInput {
    Seed5Fixture,
    Candidate,
}

/// Run the frozen `m7-parity-prompts-v1` protocol on one bound candidate.
/// Panics on the first divergence, naming the prompt and the leg.
fn run_m7_panel(candidate: &Candidate, temp_dir: &Path, input: PanelInput) -> PanelReceipt {
    let identity = (
        candidate.artifact_sha256.as_str(),
        candidate.tokenizer_sha256.as_str(),
    );

    // The legacy fixture has no embedded binding. Candidate admission requires
    // the parsed artifact to name the measured tokenizer digest.
    let model_a = IntegerStackModel::parse(&candidate.artifact_bytes).expect("parse model_a");
    assert_eq!(
        model_a.artifact_sha256(),
        candidate.artifact_sha256,
        "library: parsed model names another artifact"
    );
    let weights_per_token = model_a.weights_per_token();
    match input {
        PanelInput::Seed5Fixture => {
            assert_eq!(weights_per_token, EXPECTED_WEIGHTS_PER_TOKEN);
        }
        PanelInput::Candidate => {
            refuse_fixture_substitution(&candidate.artifact_sha256).expect("candidate identity");
            require_candidate_tokenizer_binding(&model_a, &candidate.tokenizer_sha256)
                .expect("library candidate tokenizer binding");
        }
    }
    assert_eq!(model_a.shape().read, "lorentz", "registered M7 read mode");
    let vocab = model_a.shape().vocab;
    for prompt in M7_PARITY_PROMPTS {
        assert!(
            prompt.prompt_ids.iter().all(|&id| (id as usize) < vocab),
            "{}: prompt ids exceed the candidate vocabulary ({vocab})",
            prompt.id
        );
    }
    let model_b = IntegerStackModel::parse(&candidate.artifact_bytes).expect("parse model_b");
    // Browser leg: one report covers all eight prompts.
    let browser_out = run_browser_leg(&candidate.artifact_path, &candidate.tokenizer_path)
        .expect("run node web/studio/verify_m7_parity.mjs");
    assert!(
        browser_out.status.success(),
        "verify_m7_parity.mjs failed: {}",
        String::from_utf8_lossy(&browser_out.stderr)
    );
    let browser_report: Value =
        serde_json::from_slice(&browser_out.stdout).expect("parse browser parity JSON");
    assert_eq!(
        browser_report["schema"].as_str(),
        Some("uor-r4.m7-browser-parity/1")
    );
    assert_eq!(browser_report["artifact_source"].as_str(), Some("named"));
    assert_eq!(browser_report["panel_id"].as_str(), Some(PANEL_ID));
    assert_eq!(
        browser_report["tokenizer_use"].as_str(),
        Some("token-ids-only")
    );
    let wasm_sha256 = measured_sha256(&browser_report, "wasm_sha256");
    let wasm_glue_sha256 = measured_sha256(&browser_report, "wasm_glue_sha256");
    let worker_js_sha256 = measured_sha256(&browser_report, "worker_js_sha256");
    check_leg_identity(
        "browser",
        identity,
        browser_report["artifact_sha256"].as_str(),
        browser_report["tokenizer_sha256"].as_str(),
    )
    .expect("browser receipt identity");
    assert_eq!(
        browser_report["worker_artifact_sha256"].as_str(),
        Some(candidate.artifact_sha256.as_str()),
        "browser: the Studio worker loaded another artifact"
    );
    assert_eq!(
        browser_report["weights_per_token"].as_u64(),
        Some(weights_per_token)
    );
    assert_eq!(browser_report["remote_requests"].as_u64(), Some(0));
    let browser_rows = browser_report["rows"]
        .as_array()
        .expect("browser rows array");
    assert_eq!(browser_rows.len(), M7_PARITY_PROMPTS.len());

    let mut prompts_passed = 0;
    for (idx, prompt) in M7_PARITY_PROMPTS.iter().enumerate() {
        // Leg 1: Rust library uninterrupted 48 tokens + 32+16 fresh-instance restore.
        let lib_full_48 = run_library_sequence(&model_a, None, prompt.prompt_ids, PANEL_TOKENS)
            .expect("lib uninterrupted 48");
        let lib_first_32 = run_library_sequence(&model_a, None, prompt.prompt_ids, SPLIT_FIRST)
            .expect("lib first 32");
        let lib_cont_16 = run_library_sequence(
            &model_b,
            Some(&lib_first_32.session_bytes),
            &[lib_first_32.last_token],
            SPLIT_REST,
        )
        .expect("lib cont 16");
        for (label, bytes) in [
            ("32-step", &lib_first_32.session_bytes),
            ("48-step", &lib_full_48.session_bytes),
        ] {
            assert_eq!(
                snapshot_artifact_sha256(bytes).expect("decode library snapshot"),
                candidate.artifact_sha256,
                "{}: library {label} UR4SESS1 snapshot is bound to another artifact",
                prompt.id
            );
        }
        let mut lib_combined_48 = lib_first_32.generated_ids.clone();
        lib_combined_48.extend_from_slice(&lib_cont_16.generated_ids);
        assert_eq!(
            lib_combined_48, lib_full_48.generated_ids,
            "{}: library 32+16 tokens != uninterrupted 48",
            prompt.id
        );
        assert_eq!(
            lib_cont_16.session_bytes, lib_full_48.session_bytes,
            "{}: library 32+16 UR4SESS1 bytes != uninterrupted 48 bytes",
            prompt.id
        );

        // Leg 2: `uor-chat` CLI subprocess uninterrupted 48 tokens + 32+16 fresh-process restore.
        let cli_48_path = temp_dir.join(format!("{}_cli_48.sess", prompt.id));
        let cli_32_path = temp_dir.join(format!("{}_cli_32.sess", prompt.id));
        let cli_restored_48_path = temp_dir.join(format!("{}_cli_restored_48.sess", prompt.id));

        let cli_full_48 = run_cli_sequence(
            &candidate.artifact_path,
            &candidate.tokenizer_path,
            None,
            &cli_48_path,
            prompt.prompt_ids,
            PANEL_TOKENS,
        )
        .expect("cli uninterrupted 48");
        let cli_first_32 = run_cli_sequence(
            &candidate.artifact_path,
            &candidate.tokenizer_path,
            None,
            &cli_32_path,
            prompt.prompt_ids,
            SPLIT_FIRST,
        )
        .expect("cli first 32");
        let cli_cont_16 = run_cli_sequence(
            &candidate.artifact_path,
            &candidate.tokenizer_path,
            Some(&cli_32_path),
            &cli_restored_48_path,
            &[cli_first_32.last_token],
            SPLIT_REST,
        )
        .expect("cli cont 16");
        for (label, run) in [
            ("48", &cli_full_48),
            ("32", &cli_first_32),
            ("16", &cli_cont_16),
        ] {
            check_leg_identity(
                &format!("{}: uor-chat CLI {label}", prompt.id),
                identity,
                run.artifact_sha256.as_deref(),
                run.tokenizer_sha256.as_deref(),
            )
            .expect("CLI receipt identity");
            assert_eq!(run.weights_per_token, weights_per_token);
        }
        let mut cli_combined_48 = cli_first_32.generated_ids.clone();
        cli_combined_48.extend_from_slice(&cli_cont_16.generated_ids);
        assert_eq!(
            cli_combined_48, cli_full_48.generated_ids,
            "{}: CLI 32+16 tokens != uninterrupted 48",
            prompt.id
        );
        assert_eq!(
            cli_first_32.session_bytes, lib_first_32.session_bytes,
            "{}: CLI 32-step UR4SESS1 bytes != library bytes",
            prompt.id
        );
        assert_eq!(
            cli_full_48.session_bytes, lib_full_48.session_bytes,
            "{}: CLI 48-step UR4SESS1 bytes != library bytes",
            prompt.id
        );
        assert_eq!(
            cli_cont_16.session_bytes, lib_full_48.session_bytes,
            "{}: CLI restored 48-step UR4SESS1 bytes != library bytes",
            prompt.id
        );

        // Cross-host restore check: restore CLI process A's `cli_32.sess` into the Rust library
        // and verify its 16-token continuation and final 48-step snapshot are byte-identical.
        let cross_cont_16 = run_library_sequence(
            &model_b,
            Some(&cli_first_32.session_bytes),
            &[cli_first_32.last_token],
            SPLIT_REST,
        )
        .expect("cross-host CLI->library cont 16");
        assert_eq!(cross_cont_16.generated_ids, lib_cont_16.generated_ids);
        assert_eq!(cross_cont_16.session_bytes, lib_full_48.session_bytes);

        // Leg 3: Browser WASM + Studio Web Worker row comparison.
        let brow = &browser_rows[idx];
        assert_eq!(brow["id"].as_str(), Some(prompt.id));
        assert_eq!(browser_token_ids(brow, "prompt_ids"), prompt.prompt_ids);
        let browser_tokens_48 = browser_token_ids(brow, "generated_48");
        assert_eq!(
            browser_token_ids(brow, "first_32"),
            lib_first_32.generated_ids
        );
        assert_eq!(
            browser_token_ids(brow, "cont_16"),
            lib_cont_16.generated_ids
        );
        let browser_sha32 = brow["session_32_sha256"].as_str().expect("browser_sha32");
        let browser_sha48 = brow["session_48_sha256"].as_str().expect("browser_sha48");
        let browser_restored_sha48 = brow["session_restored_48_sha256"]
            .as_str()
            .expect("browser_restored_sha48");
        assert_eq!(brow["worker_matches_wasm"].as_bool(), Some(true));

        check_three_way_row_parity(
            prompt.id,
            &lib_full_48.generated_ids,
            &cli_full_48.generated_ids,
            &browser_tokens_48,
            &lib_first_32.session_sha256,
            &cli_first_32.session_sha256,
            browser_sha32,
            &lib_full_48.session_sha256,
            &cli_full_48.session_sha256,
            browser_sha48,
            &lib_cont_16.session_sha256,
            &cli_cont_16.session_sha256,
            browser_restored_sha48,
        )
        .expect("three-way row parity must hold");
        prompts_passed += 1;
    }

    PanelReceipt {
        prompts_passed,
        tokens_compared: prompts_passed * PANEL_TOKENS,
        // A 32-step and a 48-step snapshot per prompt, matched across all three legs.
        snapshots_compared: prompts_passed * 2,
        weights_per_token,
        vocab,
        wasm_sha256: wasm_sha256.to_owned(),
        wasm_glue_sha256: wasm_glue_sha256.to_owned(),
        worker_js_sha256: worker_js_sha256.to_owned(),
    }
}

#[test]
fn library_cli_and_browser_produce_bit_identical_tokens_and_snapshots_across_frozen_panel() {
    let temp_dir = make_temp_dir("panel");
    let artifact_path = temp_dir.join("synthetic-stack.lut");
    let tokenizer_path = temp_dir.join("tokenizer-40.json");

    // The seed-5 synthetic UORLUT01 fixture: the harness regression.
    generate_synthetic_stack(SYNTHETIC_SEED, &artifact_path);
    fs::write(&tokenizer_path, build_synthetic_40_tokenizer_json())
        .expect("write tokenizer-40.json");
    let candidate = bind_candidate(
        &artifact_path,
        &tokenizer_path,
        Some(EXPECTED_ARTIFACT_SHA256),
        None,
    )
    .expect("bind the seed-5 fixture");
    assert_eq!(candidate.artifact_bytes.len(), EXPECTED_ARTIFACT_BYTES);

    let receipt = run_m7_panel(&candidate, &temp_dir, PanelInput::Seed5Fixture);
    assert_eq!(receipt.prompts_passed, M7_PARITY_PROMPTS.len());
    assert_eq!(receipt.tokens_compared, 384);

    let _ = fs::remove_dir_all(&temp_dir);
}

/// The frozen panel on a candidate named by `UOR_M7_CANDIDATE_ARTIFACT` and
/// `UOR_M7_CANDIDATE_TOKENIZER`. Prints a JSON receipt on success. A pass on
/// anything but the accepted M4 artifact validates the harness only.
#[test]
#[ignore = "requires UOR_M7_CANDIDATE_ARTIFACT and UOR_M7_CANDIDATE_TOKENIZER; run with --ignored"]
fn candidate_artifact_passes_frozen_panel_on_all_three_hosts() {
    let artifact = std::env::var_os(ENV_CANDIDATE_ARTIFACT)
        .unwrap_or_else(|| panic!("{ENV_CANDIDATE_ARTIFACT} is not set"));
    let tokenizer = std::env::var_os(ENV_CANDIDATE_TOKENIZER)
        .unwrap_or_else(|| panic!("{ENV_CANDIDATE_TOKENIZER} is not set"));
    let read_pin = |name| {
        std::env::var_os(name).map(|value| {
            value
                .into_string()
                .unwrap_or_else(|_| panic!("{name} is not a UTF-8 SHA-256 pin"))
        })
    };
    let artifact_pin = read_pin(ENV_CANDIDATE_ARTIFACT_SHA256);
    let tokenizer_pin = read_pin(ENV_CANDIDATE_TOKENIZER_SHA256);
    let candidate = bind_candidate(
        Path::new(&artifact),
        Path::new(&tokenizer),
        artifact_pin.as_deref(),
        tokenizer_pin.as_deref(),
    )
    .expect("bind the candidate");

    let temp_dir = make_temp_dir("candidate");
    let receipt = run_m7_panel(&candidate, &temp_dir, PanelInput::Candidate);
    assert_eq!(receipt.prompts_passed, M7_PARITY_PROMPTS.len());
    let record = json!({
        "schema": "uor-r4.m7-candidate-parity-receipt/1",
        "panel_id": PANEL_ID,
        "artifact_sha256": candidate.artifact_sha256,
        "artifact_bytes_len": candidate.artifact_bytes.len(),
        "artifact_sha256_pinned": artifact_pin.is_some(),
        "tokenizer_sha256": candidate.tokenizer_sha256,
        "tokenizer_artifact_binding_sha256": candidate.tokenizer_sha256,
        "tokenizer_sha256_pinned": tokenizer_pin.is_some(),
        "browser_tokenizer_use": "token-ids-only",
        "read_mode": "lorentz",
        "vocab": receipt.vocab,
        "weights_per_token": receipt.weights_per_token,
        "wasm_sha256": receipt.wasm_sha256,
        "wasm_glue_sha256": receipt.wasm_glue_sha256,
        "worker_js_sha256": receipt.worker_js_sha256,
        "prompts_passed": receipt.prompts_passed,
        "prompts": M7_PARITY_PROMPTS.len(),
        "tokens_compared": receipt.tokens_compared,
        "snapshots_compared_per_leg": receipt.snapshots_compared,
        "hosts": ["library", "uor-chat-cli", "browser-wasm-worker"],
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&record).expect("receipt JSON")
    );
    let _ = fs::remove_dir_all(&temp_dir);
}

/// Substitution controls: a receipt naming the fixture, another tokenizer or no
/// digest at all is refused, and so is a missing or mis-pinned candidate.
#[test]
fn substitution_controls_refuse_fixture_and_unbound_receipts() {
    let temp_dir = make_temp_dir("substitution");
    let unbound_path = temp_dir.join("unbound-seed6.lut");
    let candidate_path = temp_dir.join("bound-seed6.lut");
    let tokenizer_path = temp_dir.join("tokenizer-40.json");
    let other_tokenizer_path = temp_dir.join("tokenizer-40-other.json");
    generate_synthetic_stack(SYNTHETIC_SEED + 1, &unbound_path);
    let tokenizer_bytes = build_synthetic_40_tokenizer_json();
    fs::write(&tokenizer_path, &tokenizer_bytes).expect("write tokenizer");
    // Same pieces, one trailing newline: different bytes, different digest.
    let mut other = tokenizer_bytes.clone();
    other.push(b'\n');
    fs::write(&other_tokenizer_path, &other).expect("write other tokenizer");

    let tokenizer_sha256 = sha256_hex(&tokenizer_bytes);
    let unbound_bytes = fs::read(&unbound_path).expect("read seed-6 fixture");
    let unbound_model = IntegerStackModel::parse(&unbound_bytes).expect("parse unbound fixture");
    assert!(require_candidate_tokenizer_binding(&unbound_model, &tokenizer_sha256).is_err());
    fs::write(
        &candidate_path,
        bind_synthetic_fixture(&unbound_bytes, &tokenizer_sha256),
    )
    .expect("write tokenizer-bound seed-6 fixture");

    let candidate = bind_candidate(&candidate_path, &tokenizer_path, None, None)
        .expect("bind the seed-6 candidate");
    assert_ne!(candidate.artifact_sha256, EXPECTED_ARTIFACT_SHA256);
    refuse_fixture_substitution(&candidate.artifact_sha256).expect("seed 6 is not the fixture");
    assert!(
        refuse_fixture_substitution(EXPECTED_ARTIFACT_SHA256).is_err(),
        "a candidate equal to the seed-5 fixture must be refused"
    );
    let identity = (
        candidate.artifact_sha256.as_str(),
        candidate.tokenizer_sha256.as_str(),
    );

    // 1. A receipt naming the fixture in place of the candidate.
    assert!(check_leg_identity(
        "browser",
        identity,
        Some(EXPECTED_ARTIFACT_SHA256),
        Some(&candidate.tokenizer_sha256),
    )
    .is_err());
    // 2. A receipt with no digest at all.
    assert!(
        check_leg_identity("browser", identity, None, Some(&candidate.tokenizer_sha256)).is_err()
    );
    assert!(
        check_leg_identity("browser", identity, Some(&candidate.artifact_sha256), None).is_err()
    );
    // 3. The artifact's binding rejects a different tokenizer in both library and CLI.
    let bound_model =
        IntegerStackModel::parse(&candidate.artifact_bytes).expect("parse bound fixture");
    require_candidate_tokenizer_binding(&bound_model, &tokenizer_sha256).expect("right tokenizer");
    assert!(require_candidate_tokenizer_binding(&bound_model, &sha256_hex(&other)).is_err());
    let out_path = temp_dir.join("other.sess");
    let other_error = run_cli_sequence(
        &candidate_path,
        &other_tokenizer_path,
        None,
        &out_path,
        &[0, 7, 19, 31],
        4,
    )
    .err()
    .expect("the CLI must refuse the wrong tokenizer");
    assert!(
        other_error.contains("bound artifact names tokenizer")
            && other_error.contains(&tokenizer_sha256)
            && other_error.contains(&sha256_hex(&other)),
        "{other_error}"
    );
    assert!(!out_path.exists());
    // 4. Mis-pinned candidate digests are refused before any host runs.
    assert!(bind_candidate(
        &candidate_path,
        &tokenizer_path,
        Some(EXPECTED_ARTIFACT_SHA256),
        None
    )
    .is_err());
    assert!(bind_candidate(
        &candidate_path,
        &tokenizer_path,
        None,
        Some(&sha256_hex(&other))
    )
    .is_err());
    // 5. The browser leg refuses a missing named artifact or tokenizer instead of
    //    rebuilding the seed-5 fixture in its place.
    let missing = temp_dir.join("missing.lut");
    for (artifact, tokenizer) in [
        (missing.as_path(), tokenizer_path.as_path()),
        (candidate_path.as_path(), missing.as_path()),
    ] {
        let out = run_browser_leg(artifact, tokenizer).expect("run node");
        assert!(
            !out.status.success(),
            "browser leg must refuse a missing named input"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("refusing to substitute"),
            "unexpected browser refusal: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert!(
        !missing.exists(),
        "the browser leg must not create the missing artifact"
    );

    // The positive rehearsal uses the same strict admission and panel as a candidate.
    let receipt = run_m7_panel(&candidate, &temp_dir, PanelInput::Candidate);
    assert_eq!(receipt.prompts_passed, M7_PARITY_PROMPTS.len());
    assert_eq!(receipt.tokens_compared, 384);
    assert_eq!(receipt.snapshots_compared, 16);
    println!(
        "bound seed-6 synthetic rehearsal: artifact sha256 {}, tokenizer sha256 {}, WASM sha256 {}, prompts {}, tokens {}, snapshots {}",
        candidate.artifact_sha256,
        candidate.tokenizer_sha256,
        receipt.wasm_sha256,
        receipt.prompts_passed,
        receipt.tokens_compared,
        receipt.snapshots_compared,
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn planted_fault_controls_detect_token_and_snapshot_tampering() {
    let temp_dir = make_temp_dir("faults");
    let artifact_path = temp_dir.join("synthetic-stack.lut");
    let tokenizer_path = temp_dir.join("tokenizer-40.json");

    generate_synthetic_stack(SYNTHETIC_SEED, &artifact_path);
    let artifact_bytes = fs::read(&artifact_path).expect("read artifact");
    fs::write(&tokenizer_path, build_synthetic_40_tokenizer_json()).expect("write tokenizer");

    let model = IntegerStackModel::parse(&artifact_bytes).expect("parse model");
    let baseline =
        run_library_sequence(&model, None, &[0, 7, 19, 31], 32).expect("baseline 32 steps");

    // 1. Planted fault: tamper with a single token in the CLI or browser trajectory.
    let mut tampered_tokens = baseline.generated_ids.clone();
    tampered_tokens[15] ^= 1;
    assert!(
        check_three_way_row_parity(
            "P1_quad",
            &baseline.generated_ids,
            &tampered_tokens,
            &baseline.generated_ids,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
        )
        .is_err(),
        "planted token fault in CLI leg must be rejected"
    );
    assert!(
        check_three_way_row_parity(
            "P1_quad",
            &baseline.generated_ids,
            &baseline.generated_ids,
            &tampered_tokens,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
            &baseline.session_sha256,
        )
        .is_err(),
        "planted token fault in browser leg must be rejected"
    );

    // 2. Planted fault: flip one hex character of `artifact_sha256` inside a UR4SESS1 snapshot.
    let mut state = session_codec::decode(&baseline.session_bytes).expect("decode valid snapshot");
    let mut bad_sha = state.artifact_sha256.clone();
    let replacement = if bad_sha.starts_with('0') { "1" } else { "0" };
    bad_sha.replace_range(0..1, replacement);
    state.artifact_sha256 = bad_sha;
    let bad_sha_bytes = session_codec::encode(&state);
    let mut fresh_session = model.session();
    assert!(
        fresh_session
            .restore_state(&session_codec::decode(&bad_sha_bytes).expect("decode"))
            .is_err(),
        "library restore_state must reject UR4SESS1 snapshot with mismatched artifact_sha256"
    );

    let bad_sha_path = temp_dir.join("bad_sha.sess");
    let out_path = temp_dir.join("out.sess");
    fs::write(&bad_sha_path, &bad_sha_bytes).expect("write bad_sha.sess");
    assert!(
        run_cli_sequence(
            &artifact_path,
            &tokenizer_path,
            Some(&bad_sha_path),
            &out_path,
            &[baseline.last_token],
            4,
        )
        .is_err(),
        "uor-chat --restore-session must reject UR4SESS1 snapshot with mismatched artifact_sha256"
    );

    // 3. Planted fault: truncate a UR4SESS1 snapshot by 1 byte.
    let truncated = &baseline.session_bytes[..baseline.session_bytes.len() - 1];
    assert!(
        session_codec::decode(truncated).is_err(),
        "session_codec::decode must reject 1-byte truncated UR4SESS1 stream"
    );
    let trunc_path = temp_dir.join("truncated.sess");
    fs::write(&trunc_path, truncated).expect("write truncated.sess");
    assert!(
        run_cli_sequence(
            &artifact_path,
            &tokenizer_path,
            Some(&trunc_path),
            &out_path,
            &[baseline.last_token],
            4,
        )
        .is_err(),
        "uor-chat --restore-session must reject 1-byte truncated UR4SESS1 stream"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}
