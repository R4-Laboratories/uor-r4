#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]

mod brier_eval;
mod cuda;
mod reply_eval;
mod s3_decode;

use brier_eval::run_brier_eval;
use reply_eval::run_reply_eval;
use s3_decode::{GoldenShellStateTable, S2Order2ErrorTable, S3_VOCAB};
use serde_json::json;
use std::{
    env,
    path::{Path, PathBuf},
};
use uor_r4_count_runtime::p3_state::{
    brier_mix_q48, integer_brier_target_q32, BrierSample, S4ConjunctionTable, S4ControlMode,
    S4PointerEntry, P3_Q32_ONE, P3_Q48_ONE,
};
use uor_r4_count_runtime::s3_state::{S3BackoffContext, S3PrimaryContext, S3RowEntry};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Failure>;

pub fn fail(msg: impl Into<String>) -> Failure {
    msg.into().into()
}

fn run_finite(out_json: &Path) -> Result<()> {
    let mut weights = vec![0u64; 4 * 15];
    for b in 0..4usize {
        let base = P3_Q32_ONE / 15;
        let rem = P3_Q32_ONE % 15;
        for e in 0..15usize {
            weights[b * 15 + e] = base + if e == 0 { rem } else { 0 };
        }
    }
    let mut samples = Vec::with_capacity(256);
    for i in 0..256usize {
        let mut masses = Vec::with_capacity(15);
        for e in 0..15usize {
            let m = (((i as u64 + 1) * (e as u64 + 3) * 1_000_000_000_003u64) % P3_Q48_ONE).max(1);
            masses.push(m);
        }
        samples.push(BrierSample {
            bucket: i % 4,
            target_masses_q48: masses,
            rival_sq_sum_q32: (i as u64) * 17,
        });
    }

    let gs = GoldenShellStateTable {
        token_h4: [0u8; S3_VOCAB],
        primary_ctxs: vec![S3PrimaryContext {
            key: (100u32 << 16) | 5,
            total: 10,
            row_start: 0,
            row_len: 2,
        }],
        primary_rows: vec![
            S3RowEntry {
                next_token: 10,
                count: 7,
                err_i8: 12,
            },
            S3RowEntry {
                next_token: 20,
                count: 3,
                err_i8: -5,
            },
        ],
        backoff_ctxs: vec![
            S3BackoffContext {
                product_key: 200,
                total: 5,
                row_start: 0,
                row_len: 1,
            },
            S3BackoffContext {
                product_key: 300,
                total: 4,
                row_start: 1,
                row_len: 1,
            },
        ],
        backoff_rows: vec![
            S3RowEntry {
                next_token: 30,
                count: 5,
                err_i8: 0,
            },
            S3RowEntry {
                next_token: 40,
                count: 4,
                err_i8: 9,
            },
        ],
    };
    let gs_pkeys = [
        (100u32 << 16) | 5,
        (100u32 << 16) | 5,
        (200u32 << 16) | 99,
        (299u32 << 16) | 99,
        (6500u32 << 16) | 99,
    ];
    let gs_targets = [10u16, 15u16, 30u16, 40u16, 50u16];
    let gs_cpu_masses: Vec<u64> = gs_pkeys
        .iter()
        .zip(&gs_targets)
        .map(|(&k, &t)| gs.lookup_modulated_q48(k, t).0)
        .collect();

    let s2 = S2Order2ErrorTable::build_from_r_stream(&[0, 5, 9, 10, 5, 9, 11, 0, 7, 8, 9], 1024)?;
    let s2_keys = [
        (5u32 << 12) | 9,
        (5u32 << 12) | 9,
        (7u32 << 12) | 8,
        (99u32 << 12) | 99,
    ];
    let s2_targets = [10u16, 12u16, 9u16, 10u16];
    let s2_cpu_masses: Vec<u64> = s2_keys
        .iter()
        .zip(&s2_targets)
        .map(|(&k, &t)| s2.lookup_modulated_q48(k, t).0)
        .collect();

    let ents = [
        S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 8,
            err_i8: 12,
        },
        S4PointerEntry {
            role_class: 2,
            span_offset: 0,
            count: 2,
            err_i8: -4,
        },
    ];
    let ents2 = [
        S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 3,
            err_i8: 0,
        },
        S4PointerEntry {
            role_class: 2,
            span_offset: 0,
            count: 9,
            err_i8: 6,
        },
    ];
    let s4 = S4ConjunctionTable::from_sorted_rows(
        [0u8; 32],
        &[(100u32, &ents[..]), (200u32, &ents2[..])],
    )?;
    let s4_queries = [
        vec![100u32, 200u32],
        vec![100u32],
        vec![200u32],
        vec![999u32],
    ];
    let s4_cands = vec![vec![(1, 0), (2, 0)]; 4];

    #[cfg(feature = "cuda")]
    let (b_chk, b_div, s3_chk, s3_div, s2_chk, s2_div, s4_chk, s4_div, cubin_sha) = {
        let eng = cuda::P3BrierGpuEngine::new()?;
        let rep = cuda::verify_finite_gpu_vs_cpu(
            &eng,
            4,
            15,
            &samples,
            &weights,
            &gs,
            &gs_pkeys,
            &gs_targets,
            &gs_cpu_masses,
            &s2,
            &s2_keys,
            &s2_targets,
            &s2_cpu_masses,
            &s4,
            &s4_queries,
            &s4_cands,
        )?;
        (
            rep.brier_checked,
            rep.brier_divergent,
            rep.s3_checked,
            rep.s3_divergent,
            rep.s2_checked,
            rep.s2_divergent,
            rep.s4_checked,
            rep.s4_divergent,
            eng.cubin_sha256().to_string(),
        )
    };
    #[cfg(not(feature = "cuda"))]
    let (b_chk, b_div, s3_chk, s3_div, s2_chk, s2_div, s4_chk, s4_div, cubin_sha) = {
        let mut divs = 0usize;
        for s in &samples {
            let w = &weights[s.bucket * 15..(s.bucket + 1) * 15];
            let m = brier_mix_q48(w, &s.target_masses_q48)?;
            let b = integer_brier_target_q32(m, s.rival_sq_sum_q32);
            if m == 0 || b == u64::MAX {
                divs += 1;
            }
        }
        let s3_divs = gs_cpu_masses
            .iter()
            .filter(|&&m| m == 0 || m > P3_Q48_ONE)
            .count();
        let s2_divs = s2_cpu_masses
            .iter()
            .filter(|&&m| m == 0 || m > P3_Q48_ONE)
            .count();
        let s4_divs = s4_queries
            .iter()
            .zip(&s4_cands)
            .filter(|(ks, cs)| s4.vote_candidates(ks, cs, S4ControlMode::Active).len() != 2)
            .count();
        (
            samples.len(),
            divs,
            gs_pkeys.len(),
            s3_divs,
            s2_keys.len(),
            s2_divs,
            s4_queries.len(),
            s4_divs,
            "cpu_only".to_string(),
        )
    };

    let divergent = b_div + s3_div + s2_div + s4_div;
    if divergent != 0 {
        return Err(fail(format!(
            "finite GPU vs CPU divergence: brier={b_div} s3={s3_div} s2={s2_div} s4={s4_div}"
        )));
    }
    std::fs::write(
        out_json,
        serde_json::to_vec_pretty(&json!({
            "rows_checked": b_chk + s3_chk + s2_chk + s4_chk,
            "brier_checked": b_chk,
            "brier_divergent": b_div,
            "s3_checked": s3_chk,
            "s3_divergent": s3_div,
            "s2_checked": s2_chk,
            "s2_divergent": s2_div,
            "s4_checked": s4_chk,
            "s4_divergent": s4_div,
            "divergent": divergent,
            "cubin_sha256": cubin_sha,
        }))?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let cmd = args
        .get(1)
        .ok_or_else(|| fail("usage: p3-eval <finite|brier|reply> ..."))?
        .as_str();
    match cmd {
        "finite" => {
            if args.len() != 3 {
                return Err(fail("usage: p3-eval finite <out.json>"));
            }
            run_finite(&PathBuf::from(&args[2]))
        }
        "brier" => {
            if args.len() != 14 {
                return Err(fail("usage: p3-eval brier <tune_dir> <r_uort> <trunk> <r_tape> <r_pick> <tune_tape> <tune_pick> <a1s_w> <a1s_seal> <golden_shell> <out_dir> <out.json>"));
            }
            run_brier_eval(
                &PathBuf::from(&args[2]),
                &PathBuf::from(&args[3]),
                &PathBuf::from(&args[4]),
                &PathBuf::from(&args[5]),
                &PathBuf::from(&args[6]),
                &PathBuf::from(&args[7]),
                &PathBuf::from(&args[8]),
                &PathBuf::from(&args[9]),
                &PathBuf::from(&args[10]),
                &PathBuf::from(&args[11]),
                &PathBuf::from(&args[12]),
                &PathBuf::from(&args[13]),
            )?;
            Ok(())
        }
        "reply" => {
            if args.len() != 11 {
                return Err(fail("usage: p3-eval reply <compact_v3> <a1s_w> <a1s_seal> <trunk> <tokenizer> <fresh256_dir> <r_uort> <out_dir> <out.json>"));
            }
            run_reply_eval(
                &PathBuf::from(&args[2]),
                &PathBuf::from(&args[3]),
                &PathBuf::from(&args[4]),
                &PathBuf::from(&args[5]),
                &PathBuf::from(&args[6]),
                &PathBuf::from(&args[7]),
                &PathBuf::from(&args[8]),
                &PathBuf::from(&args[9]),
                &PathBuf::from(&args[10]),
            )?;
            Ok(())
        }
        _ => Err(fail(format!("unknown subcommand '{cmd}'"))),
    }
}
