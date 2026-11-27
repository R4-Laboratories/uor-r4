#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

mod cuda;
mod driver_check;
mod finite;
#[allow(dead_code, clippy::identity_op)]
#[path = "../../../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
pub mod lawful_likelihood;
mod train_eval;

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
            "usage: p5-train <finite|train|eval-devtest|pack-if-pass> [flags]",
        ));
    };
    let rest = &args[2..];
    match cmd {
        "finite" => {
            let out = PathBuf::from(flag_val(rest, "--out")?);
            finite::run_finite(&out)
        }
        "train" => {
            let p4_path = PathBuf::from(flag_val(rest, "--p4")?);
            let seed1_path = PathBuf::from(flag_val(rest, "--seed1")?);
            let r_path = PathBuf::from(flag_val(rest, "--r-uort")?);
            let devtune_path = PathBuf::from(flag_val(rest, "--dev-tune")?);
            let out_dir = PathBuf::from(flag_val(rest, "--ckpt-dir")?);
            let report_path = PathBuf::from(flag_val(rest, "--report")?);
            let min_epochs = opt_flag_val(rest, "--min-epochs")
                .map(|s| s.parse::<usize>())
                .transpose()?
                .unwrap_or(3);
            let max_epochs = opt_flag_val(rest, "--max-epochs")
                .map(|s| s.parse::<usize>())
                .transpose()?
                .unwrap_or(6);
            let k_block = opt_flag_val(rest, "--k")
                .map(|s| s.parse::<usize>())
                .transpose()?
                .unwrap_or(64);
            train_eval::run_train(train_eval::TrainArgs {
                p4_path: &p4_path,
                seed1_path: &seed1_path,
                r_path: &r_path,
                devtune_path: &devtune_path,
                out_dir: &out_dir,
                report_path: &report_path,
                min_epochs,
                max_epochs,
                k_block,
            })
        }
        "eval-devtest" => {
            let p4_path = PathBuf::from(flag_val(rest, "--p4")?);
            let ckpt_dir = PathBuf::from(flag_val(rest, "--ckpt-dir")?);
            let train_report_path = PathBuf::from(flag_val(rest, "--train-report")?);
            let devtest_dir = PathBuf::from(flag_val(rest, "--devtest-dir")?);
            let rstar_path = PathBuf::from(flag_val(rest, "--rstar")?);
            let out_report = PathBuf::from(flag_val(rest, "--out")?);
            train_eval::run_eval_devtest(train_eval::EvalDevTestArgs {
                p4_path: &p4_path,
                ckpt_dir: &ckpt_dir,
                train_report_path: &train_report_path,
                devtest_dir: &devtest_dir,
                rstar_path: &rstar_path,
                out_report: &out_report,
            })
        }
        "pack-if-pass" => {
            let devtest_report = PathBuf::from(flag_val(rest, "--devtest-report")?);
            let p4_path = PathBuf::from(flag_val(rest, "--p4")?);
            let seed1_path = PathBuf::from(flag_val(rest, "--seed1")?);
            let tokenizer_path = PathBuf::from(flag_val(rest, "--tokenizer")?);
            let s1_ckpt_path = PathBuf::from(flag_val(rest, "--s1-ckpt")?);
            let out_p5_path = PathBuf::from(flag_val(rest, "--out-p5")?);
            let out_report_path = PathBuf::from(flag_val(rest, "--report")?);
            train_eval::run_pack_if_pass(
                &devtest_report,
                &p4_path,
                &seed1_path,
                &tokenizer_path,
                &s1_ckpt_path,
                &out_p5_path,
                &out_report_path,
            )
        }
        other => Err(fail(format!("unknown subcommand {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda::{
        build_eval_tiles, parse_uorknc03_descriptors, parse_uors2c01_slices, parse_uors3c01_slices,
        parse_uors6b01_weights, precompute_uorknc03_o0, EvalArmRecordOutHost, EvalSpotOutHost,
        EvalTileCmdHost, RowStepCmdHost,
    };
    use crate::driver_check::run_driver_4096_check;
    use crate::finite::{
        build_synthetic_p5_bundle, cpu_eval_p4_window_brier_and_bits, make_p4_state_with_s1,
    };
    use crate::train_eval::{
        bvgk_decide_and_apply, evaluate_p5_gate, mean_brier_micro, micro_bpb, paired_record_ci,
        rel_change_ppm, split_records_on_bos, RowBatchScheduler,
    };
    use uor_r4_discrete::p5_brier_core::{
        batch_brier_q32, p5_bvgk_step_cpu, p5_score_block_cpu, P5BatchBrier, P5RowBrier, P5S1State,
        P5_BLOCKS, P5_VOCAB,
    };

    #[test]
    fn cuda_abi_struct_sizes_match_kernel_static_asserts() {
        assert_eq!(std::mem::size_of::<RowStepCmdHost>(), 16);
        assert_eq!(std::mem::size_of::<EvalTileCmdHost>(), 24);
        assert_eq!(std::mem::size_of::<EvalArmRecordOutHost>(), 64);
        assert_eq!(std::mem::size_of::<EvalSpotOutHost>(), 56);
    }

    #[test]
    fn row_scheduler_and_eval_tiles_visit_every_causal_target_exactly_once() {
        let stream: Vec<u16> = vec![0, 5, 10, 42, 7, 0, 11, 12, 0, 20, 21, 22, 23, 24, 25, 0, 99];
        let recs = split_records_on_bos(&stream).expect("split");
        assert_eq!(recs.len(), 4);
        let expected_targets: usize = recs.iter().map(|&(_, l)| l - 1).sum();
        assert_eq!(expected_targets, 13);

        let mut sched = RowBatchScheduler::new(&stream, &recs, 5).expect("sched");
        let mut seen = Vec::new();
        let mut batch_sizes = Vec::new();
        while let Some(cmds) = sched.next_batch() {
            batch_sizes.push(cmds.len());
            for cmd in cmds {
                let prev = stream[cmd.rec_off as usize + cmd.pos as usize - 1];
                assert_eq!(stream[cmd.rec_off as usize + cmd.pos as usize], cmd.target);
                seen.push((prev, cmd.target));
            }
        }
        assert_eq!(batch_sizes, vec![5, 5, 3]);
        assert_eq!(seen.len(), expected_targets);
        let mut expected_pairs = Vec::new();
        for &(ro, rl) in &recs {
            for pos in 1..rl {
                expected_pairs.push((stream[ro + pos - 1], stream[ro + pos]));
            }
        }
        assert_eq!(seen, expected_pairs);

        // Verify eval tiles cover every target and exact spot positions across tile boundaries.
        let long_recs = vec![(0usize, 600usize), (600, 10)];
        let tiles = build_eval_tiles(&long_recs, 197, 2);
        let mut tile_tgts = 0usize;
        let mut spot_positions = Vec::new();
        for t in &tiles {
            tile_tgts += (t.pos_end - t.pos_start) as usize;
            let mut sp = t.first_spot_pos;
            while sp != 0 && sp < t.pos_end {
                spot_positions.push((t.rec_local, sp));
                sp += 197;
            }
        }
        assert_eq!(tile_tgts, 599 + 9);
        assert_eq!(spot_positions, vec![(0, 197), (0, 394), (0, 591)]);
    }

    #[test]
    fn precompute_uorknc03_o0_matches_compact_count_lookup_exactly() {
        use sha2::{Digest, Sha256};
        use uor_r4_count_runtime::CompactCountLookup;

        let b = build_synthetic_p5_bundle().expect("synth bundle");
        let (o0_raw_tot, o0_kn, o0_ml) = precompute_uorknc03_o0(&b.counts_v3).expect("o0");
        let counts_sha: [u8; 32] = Sha256::digest(&b.counts_v3).into();
        let lookup =
            CompactCountLookup::from_bytes(&b.counts_v3, b.c_sha, counts_sha).expect("lookup");

        let mut hits = vec![(0u64, 0u32); 4096];
        let (_b880, _ho, totals, default_kn) = lookup
            .visit_p4_counts(&[0], 0, |id, kn, cnts| {
                hits[usize::from(id)] = (kn, cnts[0]);
            })
            .expect("visit o0");
        assert_eq!(o0_raw_tot, totals[0]);
        for v in 1..4096usize {
            let (exp_kn, raw0) = if hits[v].0 > 0 {
                hits[v]
            } else {
                (default_kn, 0)
            };
            let exp_ml = if raw0 > 0 {
                ((u128::from(raw0) << 48) / u128::from(totals[0])) as u64
            } else {
                0
            };
            assert_eq!(o0_kn[v], exp_kn, "o0 kn mismatch at token {v}");
            assert_eq!(o0_ml[v], exp_ml, "o0 ml mismatch at token {v}");
        }
    }

    #[test]
    fn bvgk_decide_and_apply_matches_p5_bvgk_step_cpu_exactly() {
        let mut s_a = P5S1State::reseeded(32, 4, 1234, [0x22; 32]).expect("s1");
        let mut s_b = s_a.clone();
        // Build deterministic synthetic rows
        let mut rows = Vec::new();
        for i in 0..16u64 {
            let win = vec![0u16, 2, 3, 4, 9, 6, 8, 3, 7];
            let mut pick_b = vec![1_000_000_000u64; 4 * (win.len() - 2)];
            // Reward picking pos=3 (succ=9 == target) especially at pc=1 (d=0)
            pick_b[2] = 100_000_000 + i * 10;
            pick_b[(win.len() - 2) + 2] = 300_000_000 + i * 10;
            pick_b[2 * (win.len() - 2) + 2] = 500_000_000 + i * 10;
            rows.push(P5RowBrier {
                window: win,
                target: 9,
                no_pick_brier_q32: 1_200_000_000,
                pick_brier_q32: pick_b,
            });
        }
        let batch = P5BatchBrier::new(rows).expect("batch");
        let coords: Vec<u32> = vec![16, 17, 18, 19, 28, 29, 30, 31];
        let scores = p5_score_block_cpu(&s_a, &batch, &coords).expect("score");
        let epoch0 = s_a.epoch;
        let parent_sha = s_a.parent_trunk_sha256;
        let rec_a = bvgk_decide_and_apply(&mut s_a, &coords, &scores, |scratch_roots| {
            let tmp = P5S1State::from_roots(32, 4, epoch0, parent_sha, scratch_roots.to_vec())?;
            Ok(batch_brier_q32(&tmp, &batch)?)
        })
        .expect("decide_apply");
        let rec_b = p5_bvgk_step_cpu(&mut s_b, &batch, &coords).expect("cpu_step");
        assert_eq!(rec_a, rec_b);
        assert_eq!(s_a, s_b);
    }

    #[test]
    fn non_extrapolating_p5_gate_rules_are_exact() {
        let base = 1_000_000_000u128;
        // Sub-day run with only 0.5% drop MUST NOT pass via fake linear extrapolation!
        let (pass_small, v_small, _) = evaluate_p5_gate(
            base,
            995_000_000,
            999_000_000,
            999_000_000,
            1_001_000_000,
            500_000,
            499_000,
            [-6_000_000, -4_000_000],
            [-5_000_000, -3_000_000],
            1_500, // 25 min -> 0.5% is NOT >= 1% actual
            0,
            0,
        );
        assert!(!pass_small);
        assert_eq!(v_small, "P5_S1_FROZEN_AT_DEFAULT");

        // Actual 1.2% drop in <= 1 GPU-day beating both controls without raising BPB PASSES.
        let (pass_ok, v_ok, _) = evaluate_p5_gate(
            base,
            988_000_000,
            995_000_000,
            996_000_000,
            1_002_000_000,
            500_000,
            500_000,
            [-14_000_000, -10_000_000],
            [-10_000_000, -6_000_000],
            1_500,
            0,
            0,
        );
        assert!(pass_ok);
        assert_eq!(v_ok, "P5_QUALIFIED");

        // Rising BPB fails even when Brier drops 2%.
        let (pass_bpb, v_bpb, _) = evaluate_p5_gate(
            base,
            980_000_000,
            995_000_000,
            996_000_000,
            1_002_000_000,
            500_000,
            500_001,
            [-22_000_000, -18_000_000],
            [-18_000_000, -14_000_000],
            1_500,
            0,
            0,
        );
        assert!(!pass_bpb);
        assert_eq!(v_bpb, "P5_S1_FROZEN_AT_DEFAULT");

        // Instrument failure overrides everything.
        let (pass_inst, v_inst, _) = evaluate_p5_gate(
            base,
            980_000_000,
            995_000_000,
            996_000_000,
            1_002_000_000,
            500_000,
            490_000,
            [-22_000_000, -18_000_000],
            [-18_000_000, -14_000_000],
            1_500,
            1,
            0,
        );
        assert!(!pass_inst);
        assert_eq!(v_inst, "INSTRUMENT_FAILED");
    }

    #[test]
    fn synthetic_bundle_parsers_driver_check_and_bootstrap_ci_work_on_host() {
        let b = build_synthetic_p5_bundle().expect("synth bundle");
        let (oc, desc) = parse_uorknc03_descriptors(&b.counts_v3).expect("knc");
        assert_eq!(oc, 3);
        assert_eq!(desc.len(), 3 * 18);
        assert_eq!(
            parse_uors3c01_slices(&b.s3_compact).expect("s3").p_ctx_len,
            1
        );
        assert_eq!(
            parse_uors2c01_slices(&b.s2_compact).expect("s2").ctx_len,
            [1, 1, 1]
        );
        assert_eq!(
            parse_uors6b01_weights(&b.s6_bytes).expect("s6").len(),
            2640 * 22
        );

        let s1 = P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 4390, [0x11; 32]).expect("s1");
        let p4s = make_p4_state_with_s1(&b, &s1).expect("p4s");
        let (br, bq, _) =
            cpu_eval_p4_window_brier_and_bits(&p4s, &[0, 5, 10, 42, 10, 5], 42).expect("cpu eval");
        assert!(br > 0 && bq > 0);

        let dcheck = run_driver_4096_check().expect("driver check");
        assert_eq!(dcheck["pass"], true);

        let xs = vec![90u128; 100];
        let ys = vec![100u128; 100];
        let ci = paired_record_ci(&xs, &ys, 1000, 4393).expect("ci");
        assert_eq!(ci, [-1000, -1000]);
        assert_eq!(mean_brier_micro(1u128 << 32, 1), 1_000_000);
        assert_eq!(rel_change_ppm(990, 1000), -10_000);
        assert_eq!(micro_bpb(4096 * 100, 100), 1_000_000);
    }

    #[test]
    fn rebind_and_pack_p5_deployment_round_trips_modified_s1_end_to_end() {
        use crate::train_eval::rebind_and_pack_p5_deployment;
        use serde_json::json;
        use sha2::{Digest, Sha256};
        use uor_r4_count_runtime::{FrozenPickMixture, PickClassRule};
        use uor_r4_discrete::d11_export::DiscreteLutHeader;
        use uor_r4_discrete::metadata::BinderPlan;
        use uor_r4_discrete::normalization_mechanism::NormalizationMode;
        use uor_r4_discrete::snapshot::Snapshot;
        use uor_r4_discrete::trunk::{
            build_trunk_norm_frames, build_trunk_step_table, export_trunk_d11, TrunkTables,
            TrunkTrainRow, KEPT_CELLS_DEFAULT,
        };
        use uor_r4_discrete::{ObjectivePolicy, ReadInitialization, ReadSpec, Spec};
        use uor_r4_integer::discrete_d11::count_deployment::{
            load_client_artifact, write_p4_deployment, ClientArtifactFile,
        };

        let bundle = build_synthetic_p5_bundle().expect("synth bundle");
        let s1_orig = P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 1001, [0x11; 32]).expect("s1_orig");
        let s1_trained =
            P5S1State::reseeded(P5_VOCAB, P5_BLOCKS, 2002, [0x11; 32]).expect("s1_trained");
        assert_ne!(s1_orig.roots, s1_trained.roots);

        let spec = Spec {
            vocabulary: 4096,
            blocks: 4,
            context: 192,
            kept_cells: KEPT_CELLS_DEFAULT as u32,
        };
        let stages = BinderPlan::new(spec.context as usize)
            .expect("plan")
            .stages();
        let read_spec = ReadSpec::Relation { kept: 8, sweeps: 1 };
        let gate_count = read_spec
            .gate_count(spec.blocks, stages as u32)
            .expect("gates") as usize;
        let snap = Snapshot {
            schema: uor_r4_discrete::SCHEMA.to_owned(),
            spec,
            read_spec,
            objective_policy: ObjectivePolicy::NearestAddressMarginV1,
            read_initialization: ReadInitialization::ActiveLeafV1,
            normalization_mode: NormalizationMode::ScaleMatchedSnapV1,
            position: None,
            address_metric: Default::default(),
            seed: 1,
            epoch: 0,
            table: s1_orig.to_discrete_table().expect("table"),
            routes: vec![0u32; 4 * stages],
            gates: vec![0i32; gate_count],
            operator_kappa:
                "blake3:a0ea3a10dd10c5f4856b0d881d33d94c3293bad0e68bd07a47918e2bb82c819b".to_owned(),
            composition_kappa:
                "blake3:f2986c8e68dcb30a9cb511a42547179a87b66ef212a87b728765d03e70e640b0".to_owned(),
            kernel_sha256: "a".repeat(64),
            source_sha256: "b".repeat(64),
            geometry_sha256: "c".repeat(64),
        };
        let step_table = build_trunk_step_table(spec.blocks, 0).expect("step_table");
        let norm_frames = build_trunk_norm_frames(spec.blocks, 0).expect("norm_frames");
        let tables = TrunkTables::build(&[
            TrunkTrainRow {
                read_code: 100,
                rec_code: 7,
                rec_shuf_code: 7,
                rec_unit_code: 7,
                query_token: 9,
                target_token: 2777,
            },
            TrunkTrainRow {
                read_code: 200,
                rec_code: 20,
                rec_shuf_code: 20,
                rec_unit_code: 20,
                query_token: 9,
                target_token: 50,
            },
        ])
        .expect("trunk tables");
        let seed_d11 =
            export_trunk_d11(&snap, &step_table, &norm_frames, &tables).expect("export trunk d11");
        let _hdr: DiscreteLutHeader = {
            let hlen = u64::from_le_bytes(seed_d11[8..16].try_into().unwrap()) as usize;
            serde_json::from_slice(&seed_d11[16..16 + hlen]).expect("hdr")
        };
        let seed_sha = hex::encode(Sha256::digest(&seed_d11));
        let counts_sha = hex::encode(Sha256::digest(&bundle.counts_v3));

        // Build a valid P4 fit manifest & bound A1s weights for seed_d11.
        let fit_json = serde_json::to_vec(&json!({
            "schema": "uor-r4.d37.r-fit/1",
            "arm": "a1s",
            "pick_classes": 5,
            "weight_format": "UORS5W02",
            "q32_weight_words": (3 * 16) * 5 * 6,
            "prepared": {
                "format": "uor-r4.d37.r-preparation/1",
                "pick_mode": "identity",
                "trunk_sha256": seed_sha,
                "c_sha256": hex::encode(bundle.c_sha),
                "r_sha256": hex::encode(bundle.r_sha),
                "compact_sha256": counts_sha,
            }
        }))
        .expect("fit_json");
        let fit_sha: [u8; 32] = Sha256::digest(&fit_json).into();
        let mut a1s_w = vec![0u64; (3 * 16) * 5 * 6];
        for b in 0..(48 * 5) {
            a1s_w[b * 6] = 1u64 << 32;
        }
        let a1s_bytes = FrozenPickMixture::with_rule(2, fit_sha, a1s_w, PickClassRule::Five)
            .expect("a1s")
            .to_bytes();

        let tmp_dir =
            std::env::temp_dir().join(format!("p5-train-pack-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).expect("mkdir");
        let p4_path = tmp_dir.join("p4.d11");
        let p5_path = tmp_dir.join("p5.d11");

        write_p4_deployment(
            &seed_d11,
            &fit_json,
            &a1s_bytes,
            &bundle.counts_v3,
            &bundle.s3_compact,
            &bundle.s2_compact,
            &bundle.s4_bytes,
            &bundle.s6_bytes,
            &p4_path,
        )
        .expect("write synth p4.d11");

        let p4_state_bytes = match load_client_artifact(&p4_path, false).expect("load p4") {
            ClientArtifactFile::Embedded { state_bytes, .. } => state_bytes,
            ClientArtifactFile::Legacy(_) => panic!("expected Embedded"),
        };

        let (p5_trunk_sha, artifact_sha, p5_bytes) =
            rebind_and_pack_p5_deployment(&p4_state_bytes, &seed_d11, &s1_trained, &p5_path)
                .expect("rebind and pack p5.d11");
        assert_ne!(p5_trunk_sha, seed_sha);
        assert_eq!(artifact_sha.len(), 64);
        assert!(p5_bytes > 0);
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
