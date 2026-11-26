#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]

mod c_ppmi;
mod cuda;
mod reply_eval;

use reply_eval::run_reply_eval;
use serde_json::json;
use std::{
    env,
    path::{Path, PathBuf},
};
#[cfg(not(feature = "cuda"))]
use uor_r4_count_runtime::p3_state::S4ControlMode;
use uor_r4_count_runtime::p3_state::{S4ConjunctionTable, S4PointerEntry};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Failure>;

pub fn fail(msg: impl Into<String>) -> Failure {
    msg.into().into()
}

fn run_finite(out_json: &Path) -> Result<()> {
    let train_sha = [0x5Au8; 32];
    let rows = [
        (
            (10u32 << 12) | 20,
            vec![
                S4PointerEntry {
                    role_class: 1,
                    span_offset: 0,
                    count: 15,
                    err_i8: 12,
                },
                S4PointerEntry {
                    role_class: 2,
                    span_offset: 0,
                    count: 3,
                    err_i8: -5,
                },
            ],
        ),
        (
            (30u32 << 12) | 40,
            vec![
                S4PointerEntry {
                    role_class: 1,
                    span_offset: 0,
                    count: 4,
                    err_i8: -19,
                },
                S4PointerEntry {
                    role_class: 2,
                    span_offset: 0,
                    count: 27,
                    err_i8: 25,
                },
                S4PointerEntry {
                    role_class: 3,
                    span_offset: 0,
                    count: 9,
                    err_i8: 0,
                },
            ],
        ),
        (
            (100u32 << 12) | 250,
            vec![S4PointerEntry {
                role_class: 1,
                span_offset: 0,
                count: 65_535,
                err_i8: 127,
            }],
        ),
        (
            (200u32 << 12) | 350,
            vec![S4PointerEntry {
                role_class: 2,
                span_offset: 0,
                count: 1,
                err_i8: -127,
            }],
        ),
    ];
    let refs: Vec<(u32, &[S4PointerEntry])> =
        rows.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let s4 = S4ConjunctionTable::from_sorted_rows(train_sha, &refs)?;

    let mut s4_queries = Vec::with_capacity(64);
    let mut s4_cands = Vec::with_capacity(64);
    for q in 0..64usize {
        let mut ks = Vec::new();
        if q % 2 == 0 {
            ks.push((10u32 << 12) | 20);
        }
        if q % 3 == 0 {
            ks.push((30u32 << 12) | 40);
        }
        if q % 5 == 0 {
            ks.push((100u32 << 12) | 250);
        }
        if q % 7 == 0 {
            ks.push((200u32 << 12) | 350);
        }
        if q % 11 == 0 {
            ks.push((777u32 << 12) | 888); // intentionally missing key
        }
        s4_queries.push(ks);
        if q % 4 == 3 {
            s4_cands.push(vec![(2u8, 0u16), (1u8, 0u16)]);
        } else {
            s4_cands.push(vec![(1u8, 0u16), (2u8, 0u16)]);
        }
    }

    let (cubin_sha, checked, divergent) = {
        #[cfg(feature = "cuda")]
        {
            let engine = cuda::P5S4GpuEngine::new()?;
            let rep = cuda::verify_finite_gpu_vs_cpu(&engine, &s4, &s4_queries, &s4_cands)?;
            (engine.cubin_sha256, rep.s4_checked, rep.s4_divergent)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let mut divs = 0usize;
            for (ks, cs) in s4_queries.iter().zip(s4_cands.iter()) {
                let s = s4.vote_candidates(ks, cs, S4ControlMode::Active);
                if s.len() != 2 {
                    divs += 1;
                }
            }
            ("cpu_only".to_owned(), s4_queries.len(), divs)
        }
    };

    let report = json!({
        "cubin_sha256": cubin_sha,
        "s4_checked": checked,
        "s4_divergent": divergent,
        "divergent": divergent,
    });
    std::fs::write(out_json, serde_json::to_vec_pretty(&report)?)?;
    if divergent != 0 {
        return Err(fail(format!("finite GPU vs CPU diverged: {divergent}")));
    }
    Ok(())
}

fn arg_path(args: &[String], idx: usize, name: &str) -> Result<PathBuf> {
    args.get(idx)
        .map(PathBuf::from)
        .ok_or_else(|| fail(format!("missing argument {name} at position {idx}")))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let sub = args
        .get(1)
        .map(String::as_str)
        .ok_or_else(|| fail("usage: p5-eval <finite|reply> ..."))?;
    match sub {
        "finite" => {
            let out_json = arg_path(&args, 2, "out_json")?;
            run_finite(&out_json)
        }
        "reply" => {
            let c_uort = arg_path(&args, 2, "c_uort")?;
            let compact_v3 = arg_path(&args, 3, "compact_v3")?;
            let a1s_weights = arg_path(&args, 4, "a1s_weights")?;
            let a1s_seal = arg_path(&args, 5, "a1s_seal")?;
            let trunk = arg_path(&args, 6, "trunk")?;
            let tokenizer = arg_path(&args, 7, "tokenizer")?;
            let fresh256_dir = arg_path(&args, 8, "fresh256_dir")?;
            let r_uort = arg_path(&args, 9, "r_uort")?;
            let out_dir = arg_path(&args, 10, "out_dir")?;
            let out_json = arg_path(&args, 11, "out_json")?;
            run_reply_eval(
                &c_uort,
                &compact_v3,
                &a1s_weights,
                &a1s_seal,
                &trunk,
                &tokenizer,
                &fresh256_dir,
                &r_uort,
                &out_dir,
                &out_json,
            )?;
            Ok(())
        }
        other => Err(fail(format!("unknown subcommand {other}"))),
    }
}
