#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]

mod builder;
mod cuda;
mod eval;
mod transport;

#[cfg(feature = "cuda")]
use builder::build_cpu_partitioned;
use builder::check_uors3t01_reload;
use eval::{
    compute_root_denoms, eval_mixed_masses, fit_bucketed_em, paired_record_ci, pick_mass_q48,
    score_dev_tune, DevTuneData, DEV_TUNE_UORT_SHA256, R_TARGETS, R_UORT_SHA256,
    SEED1_TRUNK_SHA256, U_COUNT_DEVTUNE_BPB_MICRO, U_COUNT_DEVTUNE_Q12,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use std::io::{BufWriter, Write};
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use transport::{split_docs, transport_stream_cpu, ArmSpec, TrunkTables};
use uor_r4_count_runtime::s3_state::{S3AddressArm, S3StateTable, S3_Q48_ONE};
use uor_r4_discrete::ngram_count::read_uort_tokens;

pub type Failure = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Failure>;

pub fn fail(msg: impl Into<String>) -> Failure {
    msg.into().into()
}

fn pin(hex_str: &str) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut out)?;
    Ok(out)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

fn verify_uors3f01_probes(
    arm_spec: ArmSpec,
    trunk: &TrunkTables,
    table: &S3StateTable,
    tokens: &[u16],
    expected_packed_sha: Option<&str>,
    gpu_rows: &[(u64, u64)],
) -> Result<(usize, usize, String)> {
    let (cpu_packed, cpu_blocks) = transport_stream_cpu(arm_spec.arm, trunk, tokens, true)?;
    if cpu_packed.len() != gpu_rows.len() {
        return Err(fail(format!(
            "probe length mismatch {} vs {}",
            cpu_packed.len(),
            gpu_rows.len()
        )));
    }
    let mut pk_hasher = Sha256::new();
    for &p in &cpu_packed {
        pk_hasher.update(p.to_le_bytes());
    }
    let cpu_packed_sha = hex::encode(pk_hasher.finalize());
    if let Some(exp) = expected_packed_sha {
        if cpu_packed_sha != exp {
            return Err(fail(format!(
                "packed query SHA mismatch on {}: gpu={exp} cpu={cpu_packed_sha}",
                arm_spec.name
            )));
        }
    }

    let total = gpu_rows.len();
    let mut checked = 0usize;
    let mut divergent = 0usize;

    // Inspect 0..256, seams around 65521 / 65536, mod 839 == 197, and final 16.
    let splits = split_docs(tokens)?;
    let mut doc_id = 0usize;
    let mut ord = 0usize;
    while doc_id < splits.starts.len() {
        let ds = splits.starts[doc_id] as usize;
        let dl = splits.lens[doc_id] as usize;
        let doc = &tokens[ds..ds + dl];
        for t in 1..dl {
            let seam = [65521usize, 65536usize]
                .iter()
                .any(|&b| (1..=6).any(|k| ord.abs_diff(k * b) <= 1));
            if ord < 256 || seam || (ord % 839 == 197) || ord >= total.saturating_sub(16) {
                checked += 1;
                let ba: [i32; 8] = cpu_blocks[ord][..8].try_into()?;
                let bb: [i32; 8] = cpu_blocks[ord][8..].try_into()?;
                let expected = table.lookup_q48(&ba, &bb, &doc[..t], doc[t])?;
                let (gpu_mass, gpu_meta) = gpu_rows[ord];
                let gpu_hk = (gpu_meta & 0xFF) as u8;
                let gpu_pr = ((gpu_meta >> 8) & 0xFF) as u8;
                let gpu_pk = ((gpu_meta >> 16) & 0xFFFF) as u16;
                let gpu_pq16 = (gpu_meta >> 32) as u32;
                let exp_hk = match expected.hit_kind {
                    uor_r4_count_runtime::s3_state::S3HitKind::ExactPrimary => 0u8,
                    uor_r4_count_runtime::s3_state::S3HitKind::ExactBackoff => 1u8,
                    uor_r4_count_runtime::s3_state::S3HitKind::StarlingMiss => 2u8,
                    uor_r4_count_runtime::s3_state::S3HitKind::UniformFallback => 3u8,
                };
                if (gpu_mass, gpu_hk, gpu_pr, gpu_pk, gpu_pq16)
                    != (
                        expected.mass_q48,
                        exp_hk,
                        expected.predicted_root,
                        expected.product_key,
                        expected.output_pick_q16,
                    )
                {
                    divergent += 1;
                }
            }
            ord += 1;
        }
        doc_id += 1;
    }
    Ok((checked, divergent, cpu_packed_sha))
}

#[cfg(feature = "cuda")]
fn write_tape(path: &Path, rows: &[(u64, u64)]) -> Result<String> {
    let mut w = BufWriter::with_capacity(1 << 20, fs::File::create(path)?);
    let mut hasher = Sha256::new();
    let mut emit = |b: &[u8]| -> Result<()> {
        w.write_all(b)?;
        hasher.update(b);
        Ok(())
    };
    emit(b"UORS3F01")?;
    emit(&(rows.len() as u64).to_le_bytes())?;
    for &(m, meta) in rows {
        emit(&m.to_le_bytes())?;
        emit(&meta.to_le_bytes())?;
    }
    w.flush()?;
    Ok(hex::encode(hasher.finalize()))
}

fn read_tape(path: &Path) -> Result<(Vec<(u64, u64)>, String)> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 || &bytes[..8] != b"UORS3F01" {
        return Err(fail("invalid UORS3F01 header"));
    }
    let count = u64::from_le_bytes(bytes[8..16].try_into()?) as usize;
    if bytes.len() != 16 + count * 16 {
        return Err(fail("UORS3F01 byte length mismatch"));
    }
    let tape_sha = hex::encode(Sha256::digest(&bytes));
    let mut out = Vec::with_capacity(count);
    for ch in bytes[16..].chunks_exact(16) {
        out.push((
            u64::from_le_bytes(ch[..8].try_into()?),
            u64::from_le_bytes(ch[8..16].try_into()?),
        ));
    }
    Ok((out, tape_sha))
}

#[cfg(feature = "cuda")]
fn cmd_finite(c128_uort: &Path, out_json: &Path) -> Result<()> {
    use builder::build_partitioned_uors3t01;
    use cuda::S3GpuEngine;

    let raw = fs::read(c128_uort)?;
    let train_sha: [u8; 32] = Sha256::digest(&raw).into();
    let tokens = read_uort_tokens(&raw)?;
    let trunk = TrunkTables::from_synthetic_seed(20_261_125)?;
    let gpu = S3GpuEngine::new(&trunk)?;
    let parent = out_json
        .parent()
        .ok_or_else(|| fail("out_json has no parent directory"))?;
    let tmp = parent.join("tmp-finite");

    let mut reports = Vec::new();
    for spec in ArmSpec::all_eleven() {
        let budget = 8192usize;
        let cpu_table = build_cpu_partitioned(spec, train_sha, &trunk, &tokens, &tmp, budget)?;
        let gpu_table = build_partitioned_uors3t01(
            spec,
            train_sha,
            &trunk,
            &tokens,
            &tmp,
            budget,
            &mut |chunk| Ok(gpu.transport_chunk(spec, chunk, false)?.0),
        )?;
        if cpu_table != gpu_table {
            return Err(fail(format!("C128 builder mismatch on arm {}", spec.name)));
        }
        let t_sha: [u8; 32] = Sha256::digest(&gpu_table).into();
        let loaded = check_uors3t01_reload(&gpu_table, train_sha, t_sha)?;
        let (packed, blks) = gpu.transport_chunk(spec, &tokens, true)?;
        let mut pk_h = Sha256::new();
        for &p in &packed {
            pk_h.update(p.to_le_bytes());
        }
        let gpu_pk_sha = hex::encode(pk_h.finalize());
        let blk_opt = matches!(spec.arm, S3AddressArm::E8Product).then_some(blks.as_slice());
        let rows_a = gpu.lookup_batched(&gpu_table, &packed, blk_opt, 128)?;
        let rows_b = gpu.lookup_batched(&gpu_table, &packed, blk_opt, 97)?;
        if rows_a != rows_b {
            return Err(fail(format!("C128 batch parity mismatch on {}", spec.name)));
        }
        let (checked, divergent, _) =
            verify_uors3f01_probes(spec, &trunk, &loaded, &tokens, Some(&gpu_pk_sha), &rows_a)?;
        if divergent != 0 {
            return Err(fail(format!(
                "C128 CUDA vs CPU diverge ({divergent}) on {}",
                spec.name
            )));
        }
        reports.push(json!({
            "arm": spec.name,
            "bytes": gpu_table.len(),
            "probes_checked": checked,
            "divergent": divergent,
        }));
    }
    let _ = fs::remove_dir_all(&tmp);
    fs::write(
        out_json,
        serde_json::to_vec_pretty(&json!({
            "cubin_sha256": gpu.cubin_sha256(),
            "arms": reports,
        }))?,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cmd_build(
    arm_name: &str,
    c_uort: &Path,
    c_sha_hex: &str,
    trunk_path: &Path,
    tmp_dir: &Path,
    max_bytes: usize,
    out_bin: &Path,
    out_meta: &Path,
) -> Result<()> {
    use builder::build_partitioned_uors3t01;
    use cuda::S3GpuEngine;

    let actual_c = file_sha256(c_uort)?;
    if actual_c != c_sha_hex {
        return Err(fail(format!("C SHA mismatch: {actual_c} != {c_sha_hex}")));
    }
    if file_sha256(trunk_path)? != SEED1_TRUNK_SHA256 {
        return Err(fail("trunk SHA mismatch"));
    }
    let spec = ArmSpec::from_name(arm_name)?;
    let trunk = TrunkTables::load_seed1(trunk_path)?;
    let gpu = S3GpuEngine::new(&trunk)?;
    let raw = fs::read(c_uort)?;
    let tokens = read_uort_tokens(&raw)?;
    drop(raw);
    let train_sha = pin(c_sha_hex)?;

    let bytes = build_partitioned_uors3t01(
        spec,
        train_sha,
        &trunk,
        &tokens,
        tmp_dir,
        max_bytes,
        &mut |chunk| Ok(gpu.transport_chunk(spec, chunk, false)?.0),
    )?;
    let bsha: [u8; 32] = Sha256::digest(&bytes).into();
    let tab = check_uors3t01_reload(&bytes, train_sha, bsha)?;
    fs::write(out_bin, &bytes)?;
    fs::write(
        out_meta,
        serde_json::to_vec_pretty(&json!({
            "arm": spec.name,
            "bytes": bytes.len(),
            "sha256": hex::encode(bsha),
            "primary_ctxs": tab.primary_context_count(),
            "primary_rows": tab.primary_row_count(),
            "backoff_ctxs": tab.backoff_context_count(),
            "backoff_rows": tab.backoff_row_count(),
        }))?,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cmd_capture(
    arm_name: &str,
    table_bin: &Path,
    c_sha_hex: &str,
    uort_path: &Path,
    uort_sha_hex: &str,
    trunk_path: &Path,
    out_65536: &Path,
    out_65521: &Path,
    out_cap_meta: &Path,
) -> Result<()> {
    use cuda::S3GpuEngine;

    if file_sha256(uort_path)? != uort_sha_hex {
        return Err(fail("split UORT SHA mismatch"));
    }
    if uort_sha_hex == eval::SEALED_DEV_TEST_UORT_SHA {
        return Err(fail("DEV-test forbidden in Stage B2b"));
    }
    if file_sha256(trunk_path)? != SEED1_TRUNK_SHA256 {
        return Err(fail("trunk SHA mismatch"));
    }
    let spec = ArmSpec::from_name(arm_name)?;
    let trunk = TrunkTables::load_seed1(trunk_path)?;
    let table_bytes = fs::read(table_bin)?;
    let train_sha = pin(c_sha_hex)?;
    let table_sha: [u8; 32] = Sha256::digest(&table_bytes).into();
    let _table = check_uors3t01_reload(&table_bytes, train_sha, table_sha)?;

    let raw = fs::read(uort_path)?;
    let tokens = read_uort_tokens(&raw)?;
    drop(raw);

    let gpu = S3GpuEngine::new(&trunk)?;
    let need_blks = matches!(spec.arm, S3AddressArm::E8Product);
    let (packed, blks) = gpu.transport_chunk(spec, &tokens, need_blks)?;
    let mut pk_h = Sha256::new();
    for &p in &packed {
        pk_h.update(p.to_le_bytes());
    }
    let packed_sha = hex::encode(pk_h.finalize());
    let blk_opt = need_blks.then_some(blks.as_slice());

    let rows_a = gpu.lookup_batched(&table_bytes, &packed, blk_opt, 65_536)?;
    let rows_b = gpu.lookup_batched(&table_bytes, &packed, blk_opt, 65_521)?;
    if rows_a != rows_b {
        return Err(fail(format!(
            "shifted-batch divergence on arm {} split {}",
            spec.name, uort_sha_hex
        )));
    }
    let tape_sha = write_tape(out_65536, &rows_a)?;
    let _ = write_tape(out_65521, &rows_b)?;
    fs::write(
        out_cap_meta,
        serde_json::to_vec_pretty(&json!({
            "arm": spec.name,
            "rows": rows_a.len(),
            "packed_sha256": packed_sha,
            "tape_sha256": tape_sha,
            "cubin_sha256": gpu.cubin_sha256(),
        }))?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_verify_tape(
    arm_name: &str,
    table_bin: &Path,
    c_sha_hex: &str,
    uort_path: &Path,
    uort_sha_hex: &str,
    trunk_path: &Path,
    tape_65536: &Path,
    cap_meta_json: &Path,
    out_verify: &Path,
) -> Result<()> {
    if file_sha256(uort_path)? != uort_sha_hex {
        return Err(fail("split UORT SHA mismatch"));
    }
    if uort_sha_hex == eval::SEALED_DEV_TEST_UORT_SHA {
        return Err(fail("DEV-test forbidden in Stage B2b"));
    }
    if file_sha256(trunk_path)? != SEED1_TRUNK_SHA256 {
        return Err(fail("trunk SHA mismatch"));
    }
    let spec = ArmSpec::from_name(arm_name)?;
    let trunk = TrunkTables::load_seed1(trunk_path)?;
    let table_bytes = fs::read(table_bin)?;
    let train_sha = pin(c_sha_hex)?;
    let table_sha: [u8; 32] = Sha256::digest(&table_bytes).into();
    let table = check_uors3t01_reload(&table_bytes, train_sha, table_sha)?;

    let cap: Value = serde_json::from_slice(&fs::read(cap_meta_json)?)?;
    let expected_packed_sha = cap["packed_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing packed_sha256 in capture meta"))?;
    let expected_tape_sha = cap["tape_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing tape_sha256 in capture meta"))?;
    let cubin_sha = cap["cubin_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing cubin_sha256 in capture meta"))?;

    let (rows_a, tape_sha) = read_tape(tape_65536)?;
    if tape_sha != expected_tape_sha {
        return Err(fail(format!(
            "tape SHA mismatch on {}: {tape_sha} != {expected_tape_sha}",
            spec.name
        )));
    }
    let raw = fs::read(uort_path)?;
    let tokens = read_uort_tokens(&raw)?;
    drop(raw);

    let (checked, divergent, cpu_packed_sha) = verify_uors3f01_probes(
        spec,
        &trunk,
        &table,
        &tokens,
        Some(expected_packed_sha),
        &rows_a,
    )?;
    if checked < 12_500 || divergent != 0 {
        return Err(fail(format!(
            "verify failed on {}: checked={checked} divergent={divergent}",
            spec.name
        )));
    }
    fs::write(
        out_verify,
        serde_json::to_vec_pretty(&json!({
            "arm": spec.name,
            "rows": rows_a.len(),
            "probes_checked": checked,
            "divergent_rows": divergent,
            "packed_sha256": cpu_packed_sha,
            "tape_sha256": tape_sha,
            "cubin_sha256": cubin_sha,
        }))?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_eval(
    trunk_path: &Path,
    r_tape_dir: &Path,
    tune_dir: &Path,
    tune_tape_dir: &Path,
    b1a_r_tape: &Path,
    b1a_r_pick: &Path,
    b1a_tune_tape: &Path,
    b1a_tune_pick: &Path,
    b1a_weights: &Path,
    b1a_seal_json: &Path,
    out_json: &Path,
) -> Result<()> {
    if file_sha256(trunk_path)? != SEED1_TRUNK_SHA256 {
        return Err(fail("trunk SHA mismatch"));
    }
    let trunk = TrunkTables::load_seed1(trunk_path)?;
    let denoms = compute_root_denoms(&trunk.token_e8_roots)?;
    let tune = DevTuneData::load_checked(tune_dir)?;
    let r_rows = R_TARGETS;
    let tune_rows = eval::DEV_TUNE_TARGETS;

    let seal_val: Value = serde_json::from_slice(&fs::read(b1a_seal_json)?)?;
    let binding_hex = seal_val["manifest_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing manifest_sha256 in a1s.seal.json"))?;
    let expected_weight_sha = seal_val["weight_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing weight_sha256 in a1s.seal.json"))?;
    if file_sha256(b1a_weights)? != expected_weight_sha {
        return Err(fail("a1s.weights SHA-256 mismatch against a1s.seal.json"));
    }
    let binding = pin(binding_hex)?;
    let r_sha = pin(R_UORT_SHA256)?;
    let tune_sha = pin(DEV_TUNE_UORT_SHA256)?;
    let trunk_sha = pin(SEED1_TRUNK_SHA256)?;

    let p1_r = eval::compute_p1_a1s_masses_from_pick_tape(
        b1a_r_tape,
        b1a_r_pick,
        b1a_weights,
        binding,
        r_sha,
        trunk_sha,
        r_rows,
    )?;
    let p1_tune = eval::compute_p1_a1s_masses_from_pick_tape(
        b1a_tune_tape,
        b1a_tune_pick,
        b1a_weights,
        binding,
        tune_sha,
        trunk_sha,
        tune_rows,
    )?;
    let p1 = score_dev_tune(&tune, &p1_tune)?;
    // Verify exact parity with Stage B1a selection.json oracle (`a1s.all.bits_q12 == 125_750_699_315`).
    if p1.all.bits_q12 != 125_750_699_315u128 {
        return Err(fail(format!(
            "P1(A1s) DEV-tune oracle mismatch: got {} expected 125750699315",
            p1.all.bits_q12
        )));
    }

    let unif_q48 = S3_Q48_ONE / 4096u64;
    let mut arm_results = Vec::new();
    let mut e8_standalone_rec_all: Option<Vec<u128>> = None;
    let mut e8_standalone_rec_novel: Option<Vec<u128>> = None;
    let mut e8_combined_rec_all: Option<Vec<u128>> = None;
    let mut e8_combined_rec_novel: Option<Vec<u128>> = None;
    let mut e8_u_geo_bits_q12: Option<u128> = None;
    let mut best_ctrl_standalone: Option<(String, u128, Vec<u128>, Vec<u128>)> = None;
    let mut best_ctrl_combined: Option<(String, u128, Vec<u128>, Vec<u128>)> = None;

    for spec in ArmSpec::all_eleven() {
        let (r_rows_vec, _) = read_tape(&r_tape_dir.join(format!("{}.65536.features", spec.name)))?;
        let (t_rows_vec, _) =
            read_tape(&tune_tape_dir.join(format!("{}.65536.features", spec.name)))?;
        if r_rows_vec.len() != r_rows || t_rows_vec.len() != tune_rows {
            return Err(fail(format!("feature tape size mismatch on {}", spec.name)));
        }
        let is_e8 = matches!(spec.arm, S3AddressArm::E8Product);
        let n_exp_standalone = if is_e8 { 3usize } else { 2usize };
        let n_exp_combined = if is_e8 { 3usize } else { 2usize };

        let build_standalone_inputs = |vec: &[(u64, u64)]| -> Result<(Vec<u8>, Vec<u64>)> {
            let mut bkts = Vec::with_capacity(vec.len());
            let mut exps = Vec::with_capacity(vec.len() * n_exp_standalone);
            for &(m, meta) in vec {
                let hk = (meta & 0xFF) as u8;
                let pr = ((meta >> 8) & 0xFF) as u8;
                let pq16 = (meta >> 32) as u32;
                bkts.push(hk.min(3));
                exps.push(m);
                if is_e8 {
                    exps.push(pick_mass_q48(pq16, pr, &denoms)?);
                }
                exps.push(unif_q48);
            }
            Ok((bkts, exps))
        };

        let (r_bkts, r_exps) = build_standalone_inputs(&r_rows_vec)?;
        let (w_standalone, _) = fit_bucketed_em(&r_bkts, &r_exps, 4, n_exp_standalone, 64, false)?;
        let (t_bkts, t_exps) = build_standalone_inputs(&t_rows_vec)?;
        let t_standalone_m = eval_mixed_masses(&t_bkts, &t_exps, n_exp_standalone, &w_standalone)?;
        let st_sc = score_dev_tune(&tune, &t_standalone_m)?;

        let build_combined_inputs =
            |vec: &[(u64, u64)], p1_m: &[u64]| -> Result<(Vec<u8>, Vec<u64>)> {
                let mut bkts = Vec::with_capacity(vec.len());
                let mut exps = Vec::with_capacity(vec.len() * n_exp_combined);
                for (i, &(m, meta)) in vec.iter().enumerate() {
                    let hk = (meta & 0xFF) as u8;
                    let pr = ((meta >> 8) & 0xFF) as u8;
                    let pq16 = (meta >> 32) as u32;
                    bkts.push(hk.min(3));
                    exps.push(p1_m[i]);
                    exps.push(m);
                    if is_e8 {
                        exps.push(pick_mass_q48(pq16, pr, &denoms)?);
                    }
                }
                Ok((bkts, exps))
            };

        let (rc_bkts, rc_exps) = build_combined_inputs(&r_rows_vec, &p1_r)?;
        let (w_combined, _) = fit_bucketed_em(&rc_bkts, &rc_exps, 4, n_exp_combined, 64, false)?;
        let (tc_bkts, tc_exps) = build_combined_inputs(&t_rows_vec, &p1_tune)?;
        let t_comb_m = eval_mixed_masses(&tc_bkts, &tc_exps, n_exp_combined, &w_combined)?;
        let cb_sc = score_dev_tune(&tune, &t_comb_m)?;

        // Also oracle U_geo directly on DEV-tune for `e8`.
        let u_geo_json = if is_e8 {
            let (_, ug_hist) = fit_bucketed_em(&tc_bkts, &tc_exps, 4, n_exp_combined, 64, true)?;
            let ug_bits = *ug_hist
                .last()
                .ok_or_else(|| fail("empty U_geo EM history"))?;
            e8_u_geo_bits_q12 = Some(ug_bits);
            let micro = ug_bits * 1_000_000 / (4096 * u128::from(cb_sc.all.bytes));
            Some(json!({ "bits_q12": ug_bits.to_string(), "bpb_micro": micro.to_string() }))
        } else {
            None
        };

        if is_e8 {
            e8_standalone_rec_all = Some(st_sc.rec_all);
            e8_standalone_rec_novel = Some(st_sc.rec_novel);
            e8_combined_rec_all = Some(cb_sc.rec_all);
            e8_combined_rec_novel = Some(cb_sc.rec_novel);
        } else {
            if best_ctrl_standalone
                .as_ref()
                .is_none_or(|b| st_sc.all.bits_q12 < b.1)
            {
                best_ctrl_standalone = Some((
                    spec.name.to_string(),
                    st_sc.all.bits_q12,
                    st_sc.rec_all,
                    st_sc.rec_novel,
                ));
            }
            if best_ctrl_combined
                .as_ref()
                .is_none_or(|b| cb_sc.all.bits_q12 < b.1)
            {
                best_ctrl_combined = Some((
                    spec.name.to_string(),
                    cb_sc.all.bits_q12,
                    cb_sc.rec_all,
                    cb_sc.rec_novel,
                ));
            }
        }

        arm_results.push(json!({
            "arm": spec.name,
            "standalone_100mb": {
                "all": st_sc.all.json(),
                "novel": st_sc.novel.json(),
                "response": st_sc.response.json(),
                "novel_response": st_sc.novel_response.json(),
            },
            "combined_with_p1_a1s": {
                "all": cb_sc.all.json(),
                "novel": cb_sc.novel.json(),
                "response": cb_sc.response.json(),
                "novel_response": cb_sc.novel_response.json(),
            },
            "oracle_u_geo_subset": u_geo_json,
        }));
    }

    let bc_s = best_ctrl_standalone.ok_or_else(|| fail("missing standalone control"))?;
    let bc_c = best_ctrl_combined.ok_or_else(|| fail("missing combined control"))?;
    let sa = e8_standalone_rec_all.ok_or_else(|| fail("missing e8 standalone all"))?;
    let sn = e8_standalone_rec_novel.ok_or_else(|| fail("missing e8 standalone novel"))?;
    let ca = e8_combined_rec_all.ok_or_else(|| fail("missing e8 combined all"))?;
    let cn = e8_combined_rec_novel.ok_or_else(|| fail("missing e8 combined novel"))?;
    let u_geo_q12 = e8_u_geo_bits_q12.ok_or_else(|| fail("missing e8 U_geo bits"))?;

    let ci_standalone_all = paired_record_ci(&sa, &bc_s.2, 1000, 4390)?;
    let ci_standalone_novel = paired_record_ci(&sn, &bc_s.3, 1000, 4390)?;
    let ci_combined_vs_ctrl_all = paired_record_ci(&ca, &bc_c.2, 1000, 4390)?;
    let ci_combined_vs_ctrl_novel = paired_record_ci(&cn, &bc_c.3, 1000, 4390)?;
    let ci_combined_vs_p1_all = paired_record_ci(&ca, &p1.rec_all, 1000, 4390)?;
    let ci_combined_vs_p1_novel = paired_record_ci(&cn, &p1.rec_novel, 1000, 4390)?;

    let e8_standalone_q12: u128 = sa.iter().sum();
    let e8_combined_q12: u128 = ca.iter().sum();
    let gate_p2_standalone_1pct =
        100 * e8_standalone_q12 <= 99 * bc_s.1 && ci_standalone_all[1] < 0;
    let gate_p2_combined_1pct =
        100 * e8_combined_q12 <= 99 * bc_c.1 && ci_combined_vs_ctrl_all[1] < 0;

    let u_geo_minus_u_count_q12 = (u_geo_q12 as i128) - (U_COUNT_DEVTUNE_Q12 as i128);
    let u_geo_bpb_micro = u_geo_q12 * 1_000_000 / (4096 * u128::from(p1.all.bytes));
    let u_geo_over_u_count_milli = (u_geo_q12 * 1000) / U_COUNT_DEVTUNE_Q12;

    fs::write(
        out_json,
        serde_json::to_vec_pretty(&json!({
            "p1_a1s_baseline": {
                "all": p1.all.json(),
                "novel": p1.novel.json(),
                "response": p1.response.json(),
                "novel_response": p1.novel_response.json(),
            },
            "u_count_devtune": {
                "source": "claude-438 (docs/research/notebook/m4-serving.md:38)",
                "bits_q12": U_COUNT_DEVTUNE_Q12.to_string(),
                "bpb_micro": U_COUNT_DEVTUNE_BPB_MICRO.to_string(),
            },
            "u_geo_vs_u_count": {
                "u_geo_bits_q12": u_geo_q12.to_string(),
                "u_geo_bpb_micro": u_geo_bpb_micro.to_string(),
                "delta_bits_q12": u_geo_minus_u_count_q12.to_string(),
                "ratio_milli": u_geo_over_u_count_milli.to_string(),
            },
            "arms": arm_results,
            "comparisons": {
                "best_standalone_control": bc_s.0,
                "best_combined_control": bc_c.0,
                "ci_standalone_e8_minus_best_ctrl_all_q12": [ci_standalone_all[0].to_string(), ci_standalone_all[1].to_string()],
                "ci_standalone_e8_minus_best_ctrl_novel_q12": [ci_standalone_novel[0].to_string(), ci_standalone_novel[1].to_string()],
                "ci_combined_e8_minus_best_ctrl_all_q12": [ci_combined_vs_ctrl_all[0].to_string(), ci_combined_vs_ctrl_all[1].to_string()],
                "ci_combined_e8_minus_best_ctrl_novel_q12": [ci_combined_vs_ctrl_novel[0].to_string(), ci_combined_vs_ctrl_novel[1].to_string()],
                "ci_combined_e8_minus_p1_all_q12": [ci_combined_vs_p1_all[0].to_string(), ci_combined_vs_p1_all[1].to_string()],
                "ci_combined_e8_minus_p1_novel_q12": [ci_combined_vs_p1_novel[0].to_string(), ci_combined_vs_p1_novel[1].to_string()],
                "gate_p2_standalone_1pct": gate_p2_standalone_1pct,
                "gate_p2_combined_1pct": gate_p2_combined_1pct,
            }
        }))?,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_devtune_pick_tape(trunk_path: &Path, tune_dir: &Path, out_pick_rows: &Path) -> Result<()> {
    let trunk_bytes = fs::read(trunk_path)?;
    let trunk_sha: [u8; 32] = Sha256::digest(&trunk_bytes).into();
    let tune = DevTuneData::load_checked(tune_dir)?;
    let tune_raw = fs::read(tune_dir.join("dev-tune.tokens.u16"))?;
    let tune_sha: [u8; 32] = Sha256::digest(&tune_raw).into();

    // Exact PickDevice setup & native probe schedule mirroring `eval-v3/src/tune.rs`.
    let model = uor_r4_integer::discrete_d11::IntegerDiscreteModel::load(trunk_path)?;
    if model.artifact_sha256() != SEED1_TRUNK_SHA256
        || model.shape().blocks != 4
        || model.shape().context != 192
    {
        return Err(fail("unexpected seed1.d11 trunk SHA or shape"));
    }
    let dev = uor_r4_discrete::device::Device::new_with_position(
        uor_r4_discrete::Spec {
            vocabulary: model.shape().vocabulary,
            blocks: model.shape().blocks,
            context: model.shape().context,
            kept_cells: model.shape().kept_cells,
        },
        uor_r4_discrete::ReadSpec::Pick,
        uor_r4_discrete::ObjectivePolicy::NearestAddressMarginV1,
        uor_r4_discrete::ReadInitialization::ActiveLeafV1,
        uor_r4_discrete::normalization_mechanism::NormalizationMode::ScaleMatchedSnapV1,
        Some(uor_r4_discrete::position_mechanism::PositionConfig::new(
            uor_r4_discrete::position_mechanism::PositionMode::IdentityControlV1,
            uor_r4_discrete::position_mechanism::PositionStepTable::identity(model.shape().blocks)?,
        )),
        model.shape().seed,
        0,
    )?;
    let mut snap = dev.snapshot()?;
    drop(dev);
    let (table, routes, gates) = model.identity_pick_inventory()?;
    let expected_table = table.clone();
    let expected_routes = routes.clone();
    let expected_gates = gates.clone();
    snap.table = table;
    snap.routes = routes;
    snap.gates = gates;
    snap.epoch = model.shape().epoch;
    let dev = uor_r4_discrete::device::Device::from_snapshot(snap, 0)?;
    let readback = dev.snapshot()?;
    if readback.table != expected_table
        || readback.routes != expected_routes
        || readback.gates != expected_gates
        || readback.position.as_ref().map(|c| c.mode.device_word()) != Some(3)
    {
        return Err(fail("CUDA Pick state readback differs from snapshot"));
    }
    let stride = uor_r4_discrete::device::FLOCK_TRACE_HEADER_WORDS
        + (model.shape().stages as usize) * uor_r4_discrete::device::FLOCK_TRACE_LEVEL_WORDS;
    let blocks = model.shape().blocks as usize;

    let mut pick_words: Vec<(u32, u32, u32, bool)> = Vec::with_capacity(eval::DEV_TUNE_TARGETS);
    let mut win_start = 0usize;
    let mut batch_wins: Vec<(usize, &[u16])> = Vec::with_capacity(4096);
    let mut frame = 0usize;
    let mut ordinal = 0usize;
    let mut native_probes = 0usize;

    let mut flush_batch = |batch: &[(usize, &[u16])]| -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let ids: Vec<Vec<_>> = batch
            .iter()
            .enumerate()
            .map(|(row, &(_, w))| {
                w.iter()
                    .enumerate()
                    .map(|(pos, &tok)| uor_r4_discrete::Identity {
                        token: u32::from(tok),
                        occurrence: pos as u64,
                        record: row as u64,
                        version: 0,
                    })
                    .collect()
            })
            .collect();
        let placeholder: Vec<u32> = batch
            .iter()
            .map(|&(_, w)| u32::from(w[w.len() - 1]))
            .collect();
        let carried = dev.carry_windows(&ids)?;
        let forward = dev.forward_traced(&carried, &placeholder)?;
        let trace = forward
            .flock_trace()?
            .ok_or_else(|| fail("missing DEV-tune Pick trace"))?;
        if trace.len() != batch.len() * blocks * stride {
            return Err(fail("unexpected DEV-tune Pick trace shape"));
        }
        for (row, &(t, w)) in batch.iter().enumerate() {
            let mut b0_words = [u32::MAX; 5];
            for b in 0..blocks {
                let rec = &trace[(row * blocks + b) * stride..][..stride];
                if rec[5..].iter().any(|&word| word != u32::MAX) {
                    return Err(fail("unexpected nonempty Pick trace remainder"));
                }
                if b == 0 {
                    b0_words.copy_from_slice(&rec[..5]);
                }
            }
            let (pick_w, dist_w, tie_w) = (b0_words[0], b0_words[1], b0_words[2]);
            let hit = if pick_w == u32::MAX {
                if dist_w != u32::MAX {
                    return Err(fail("absent pick with non-MAX distance"));
                }
                false
            } else {
                if dist_w == u32::MAX {
                    return Err(fail("present pick with MAX distance"));
                }
                let next_idx = usize::try_from(pick_w)? + 1;
                let succ = *w
                    .get(next_idx)
                    .ok_or_else(|| fail("pick successor outside window"))?;
                succ == tune.ids[t]
            };

            // Verify selected probes against CPU native `observe_identity_pick`.
            if w.len() == 1 {
                frame += 1;
            }
            let seam = [4096usize, 8192usize, 65536usize].iter().any(|&width| {
                let off = ordinal % width;
                off == 0 || off + 1 == width
            });
            let check_native = (frame <= 64 && (w.len() <= 8 || tune.ids[t] == 1))
                || seam
                || (ordinal % 4093 == 197)
                || (ordinal + 2 >= eval::DEV_TUNE_TARGETS);
            if check_native {
                native_probes += 1;
                let prefix_u32: Vec<u32> = w.iter().map(|&tok| u32::from(tok)).collect();
                let native = model.observe_identity_pick(&prefix_u32)?;
                let n0 = native
                    .block_words
                    .first()
                    .ok_or_else(|| fail("missing native block_words"))?;
                if *n0 != b0_words {
                    return Err(fail(format!(
                        "native Pick read/tie diverged at DEV-tune ordinal {ordinal}"
                    )));
                }
            }

            pick_words.push((pick_w, dist_w, tie_w, hit));
            ordinal += 1;
        }
        Ok(())
    };

    for t in 1..tune.ids.len() {
        if tune.ids[t - 1] == 0 {
            win_start = t - 1;
        }
        if tune.ids[t] != 0 {
            let start = win_start.max(t.saturating_sub(192));
            batch_wins.push((t, &tune.ids[start..t]));
            if batch_wins.len() == 4096 {
                flush_batch(&batch_wins)?;
                batch_wins.clear();
            }
        }
    }
    flush_batch(&batch_wins)?;
    if native_probes < 8_000 || pick_words.len() != eval::DEV_TUNE_TARGETS {
        return Err(fail(format!(
            "unexpected DEV-tune Pick counts: probes={native_probes}, rows={}",
            pick_words.len()
        )));
    }
    eval::write_pick_rows_from_slice(out_pick_rows, tune_sha, trunk_sha, &pick_words)
}

fn sarg(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| fail("invalid UTF-8 argument"))
}

fn main() -> Result<()> {
    let args: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    match args.as_slice() {
        #[cfg(feature = "cuda")]
        [m, c128, out] if m == Path::new("finite") => cmd_finite(c128, out),
        #[cfg(feature = "cuda")]
        [m, arm, cuort, csha, trunk, tmp, maxb, outb, outm] if m == Path::new("build") => {
            cmd_build(
                sarg(arm)?,
                cuort,
                sarg(csha)?,
                trunk,
                tmp,
                sarg(maxb)?.parse()?,
                outb,
                outm,
            )
        }
        #[cfg(feature = "cuda")]
        [m, arm, tab, csha, uort, usha, trunk, outa, outb, outm] if m == Path::new("capture") => {
            cmd_capture(
                sarg(arm)?,
                tab,
                sarg(csha)?,
                uort,
                sarg(usha)?,
                trunk,
                outa,
                outb,
                outm,
            )
        }
        [m, arm, tab, csha, uort, usha, trunk, tape, capm, outv]
            if m == Path::new("verify-tape") =>
        {
            cmd_verify_tape(
                sarg(arm)?,
                tab,
                sarg(csha)?,
                uort,
                sarg(usha)?,
                trunk,
                tape,
                capm,
                outv,
            )
        }
        #[cfg(feature = "cuda")]
        [m, trunk, tune_dir, out_pk] if m == Path::new("devtune-pick") => {
            cmd_devtune_pick_tape(trunk, tune_dir, out_pk)
        }
        [m, trunk, rt_dir, t_dir, tt_dir, brt, brp, btt, btp, bw, bs, out]
            if m == Path::new("eval") =>
        {
            cmd_eval(
                trunk, rt_dir, t_dir, tt_dir, brt, brp, btt, btp, bw, bs, out,
            )
        }
        _ => Err(fail(
            "usage: p2-s3 (finite | build | capture | verify-tape | devtune-pick | eval) ...",
        )),
    }
}
