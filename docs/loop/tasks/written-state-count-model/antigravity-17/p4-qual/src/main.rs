#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

mod bar_a;
mod cuda;
#[allow(dead_code, clippy::identity_op)]
#[path = "../../../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
pub mod lawful_likelihood;
mod pack;
mod parity;

use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn fail(msg: impl Into<String>) -> Box<dyn Error> {
    msg.into().into()
}

pub fn verify_file_sha256(path: &Path, expected_hex: &str, label: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)
        .map_err(|e| fail(format!("failed opening {label} at {}: {e}", path.display())))?;
    let len = file.metadata()?.len();
    let mut buf = Vec::with_capacity(usize::try_from(len)?);
    file.read_to_end(&mut buf)?;
    let actual = hex::encode(Sha256::digest(&buf));
    if actual != expected_hex {
        return Err(fail(format!(
            "{label} SHA-256 mismatch at {}: actual={actual} expected={expected_hex}",
            path.display()
        )));
    }
    Ok(buf)
}

pub fn load_uort_tokens(bytes: &[u8]) -> Result<Vec<u16>> {
    if bytes.len() < 64 || &bytes[..4] != b"UORT" {
        return Err(fail("invalid UORT header"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into()?);
    let count = u64::from_le_bytes(bytes[8..16].try_into()?);
    let vocab = u32::from_le_bytes(bytes[16..20].try_into()?);
    if version != 1 || vocab != 4096 {
        return Err(fail(format!(
            "unsupported UORT version={version} vocab={vocab}"
        )));
    }
    let payload = &bytes[64..];
    if count.checked_mul(2) != Some(payload.len() as u64) {
        return Err(fail("UORT payload byte count mismatch"));
    }
    let mut tokens = Vec::with_capacity(usize::try_from(count)?);
    for pair in payload.chunks_exact(2) {
        let id = u16::from_le_bytes([pair[0], pair[1]]);
        if id >= 4096 {
            return Err(fail(format!("UORT token id {id} >= 4096")));
        }
        tokens.push(id);
    }
    Ok(tokens)
}

fn flag_val(args: &[String], name: &str) -> Result<String> {
    for i in 0..args.len() {
        if args[i] == name {
            return args
                .get(i + 1)
                .cloned()
                .ok_or_else(|| fail(format!("missing value for {name}")));
        }
    }
    Err(fail(format!("missing required flag {name}")))
}

fn opt_flag_val(args: &[String], name: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == name {
            return args.get(i + 1).cloned();
        }
    }
    None
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let Some(cmd) = args.get(1).map(String::as_str) else {
        return Err(fail(
            "usage: p4-qual <finite|pack|parity-fixture|bar-a|verify-m1> [flags]",
        ));
    };
    let rest = &args[2..];
    match cmd {
        "finite" => {
            let out = PathBuf::from(flag_val(rest, "--out")?);
            parity::run_finite(&out)
        }
        "pack" => {
            let trunk = PathBuf::from(flag_val(rest, "--trunk")?);
            let tokenizer = PathBuf::from(flag_val(rest, "--tokenizer")?);
            let manifest = PathBuf::from(flag_val(rest, "--manifest")?);
            let a1s_weights = PathBuf::from(flag_val(rest, "--a1s-weights")?);
            let counts_v3 = PathBuf::from(flag_val(rest, "--counts-v3")?);
            let s3_t02 = PathBuf::from(flag_val(rest, "--s3-t02")?);
            let s3_err_p = PathBuf::from(flag_val(rest, "--s3-err-p")?);
            let s3_err_b = PathBuf::from(flag_val(rest, "--s3-err-b")?);
            let s2_s01 = PathBuf::from(flag_val(rest, "--s2-s01")?);
            let s2_errs = PathBuf::from(flag_val(rest, "--s2-errs")?);
            let s4 = PathBuf::from(flag_val(rest, "--s4")?);
            let s6 = PathBuf::from(flag_val(rest, "--s6")?);
            let out = PathBuf::from(flag_val(rest, "--out")?);
            let report = PathBuf::from(flag_val(rest, "--report")?);
            pack::run_pack(pack::PackPaths {
                trunk: &trunk,
                tokenizer: &tokenizer,
                manifest: &manifest,
                a1s_weights: &a1s_weights,
                counts_v3: &counts_v3,
                s3_t02: &s3_t02,
                s3_err_p: &s3_err_p,
                s3_err_b: &s3_err_b,
                s2_s01: &s2_s01,
                s2_errs: &s2_errs,
                s4: &s4,
                s6: &s6,
                out: &out,
                report: &report,
            })
        }
        "parity-fixture" => {
            let p4 = PathBuf::from(flag_val(rest, "--p4")?);
            let heldout = PathBuf::from(flag_val(rest, "--heldout")?);
            let out = PathBuf::from(flag_val(rest, "--out")?);
            let summary = PathBuf::from(flag_val(rest, "--summary")?);
            parity::run_parity_fixture(&p4, &heldout, &out, &summary)
        }
        "bar-a" => {
            let p4 = PathBuf::from(flag_val(rest, "--p4")?);
            let heldout = PathBuf::from(flag_val(rest, "--heldout")?);
            let tokenizer = PathBuf::from(flag_val(rest, "--tokenizer")?);
            let out = PathBuf::from(flag_val(rest, "--out")?);
            bar_a::run_bar_a(&p4, &heldout, &tokenizer, &out)
        }
        "verify-m1" => {
            let p4 = PathBuf::from(flag_val(rest, "--p4")?);
            let fixture = PathBuf::from(flag_val(rest, "--fixture")?);
            let uor_chat = PathBuf::from(flag_val(rest, "--uor-chat")?);
            let cli_count = opt_flag_val(rest, "--cli-count")
                .map(|s| s.parse::<usize>())
                .transpose()?
                .unwrap_or(8);
            let out = PathBuf::from(flag_val(rest, "--out")?);
            parity::run_verify_m1(&p4, &fixture, &uor_chat, cli_count, &out)
        }
        other => Err(fail(format!("unknown subcommand {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use crate::bar_a::{
        archived_se_micro_bpb, cpu_eval_bar_a_step, empirical_se_micro_bpb, interval_verdict,
        isqrt_u128, stream_basis_fraction,
    };
    use crate::cuda::{
        parse_uorknc03_descriptors, parse_uors2c01_slices, parse_uors3c01_slices,
        parse_uors4p01_arrays, ParityStepOutHost,
    };
    use crate::parity::{build_synthetic_p4_bundle, make_synth_p4_state};
    use uor_r4_count_runtime::p4_serving::{CausalSessionCache, P4Session, P4_VOCAB};

    #[test]
    fn parity_step_out_abi_is_128_bytes() {
        assert_eq!(std::mem::size_of::<ParityStepOutHost>(), 128);
    }

    #[test]
    fn synthetic_bundle_parses_all_device_descriptors_and_cpu_oracles() {
        let bundle = build_synthetic_p4_bundle().expect("synth bundle");
        let (order_count, desc) = parse_uorknc03_descriptors(&bundle.counts_v3).expect("knc03");
        assert_eq!(order_count, 3);
        assert_eq!(desc.len(), 3 * 18);

        let s3 = parse_uors3c01_slices(&bundle.s3_compact).expect("s3");
        assert_eq!(s3.token_h4.len(), P4_VOCAB);
        assert_eq!(s3.p_ctx_len, 1);
        assert!(s3.b_ctx_len >= 2);

        let s2 = parse_uors2c01_slices(&bundle.s2_compact).expect("s2");
        assert_eq!(s2.ctx_len, [1, 1, 1]);

        let s4 = parse_uors4p01_arrays(&bundle.s4_bytes).expect("s4");
        assert_eq!(s4.ctx_len, 2);

        let state = make_synth_p4_state(&bundle).expect("state");
        let mut sess = P4Session::new();
        for (name, prompt, steps, read_en) in &bundle.prompts {
            sess.sync_prompt(prompt).expect(name);
            for s in 0..(*steps as usize) {
                let (tok, tr) = sess.step(&state, *read_en, true).expect(name);
                assert!(tr.is_some());
                // Cross-check cpu_eval_bar_a_step against P4Session when read_enabled == true
                if *read_en {
                    let mut cache = CausalSessionCache::new();
                    cache.reset_from_tokens(sess.history());
                    let (raw_m, prll_m, bq) =
                        cpu_eval_bar_a_step(&state, None, sess.history(), &cache, tok).expect(name);
                    assert_eq!(raw_m, sess.last_scores_q48()[usize::from(tok)]);
                    assert!(prll_m >= 1);
                    assert!(bq <= 48 * 4096);
                }
                if tok == 1 {
                    break;
                }
                if s + 1 < (*steps as usize) {
                    sess.observe_generated(tok).expect(name);
                }
            }
        }
    }

    #[test]
    fn bar_a_integer_math_matches_protocol_pins() {
        // Uniform floor: 12 bits/tok = 49152 Q12/tok on stream basis 17,576,697 / 6,194,589 = 2.837427 B/tok
        // 12 / 2.837427 = 4.229184 BPB -> 4,229,184 micro-BPB
        let (_, _, u_mbpb) = stream_basis_fraction(49_152u128 * 196_608u128, 196_608);
        assert_eq!(u_mbpb, 4_229_183);
        assert_eq!(archived_se_micro_bpb(512), 14_525);
        assert_eq!(archived_se_micro_bpb(1024), 10_270);
        assert_eq!(archived_se_micro_bpb(2048), 7_262);
        assert_eq!(isqrt_u128(144), 12);
        assert_eq!(interval_verdict(850_000, 14_525).2, "PASS");
        assert_eq!(interval_verdict(950_000, 14_525).2, "FAIL");
        assert_eq!(interval_verdict(895_000, 14_525).2, "INCONCLUSIVE");
        assert_eq!(empirical_se_micro_bpb(&[900_000, 900_000, 900_000]), 0);
        assert!(empirical_se_micro_bpb(&[800_000, 1_000_000]) > 0);
    }
}
