#![cfg(feature = "p1-count-serving")]
#![forbid(unsafe_code)]

use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::process::Command;
use uor_r4_count_runtime::{
    code_pick,
    kn_score::KN_BASE_MASS,
    p3_state::{
        brier_mix_q48, s4_pair_key, BrierS6Weights, LawfulStopConfig, S4ConjunctionTable,
        S4PointerEntry, P3_Q32_ONE, P3_Q48_ONE,
    },
    p4_serving::{
        encode_compact_s2_skip_from_uors2s01, encode_compact_s3_from_uors3t02, golden_shell_pkey,
        skip_keys_for_prefix, starling_7_neighbours, CausalSessionCache, CompactS2SkipView,
        CompactS3View, P4Session, S4SourceGate, S4SourceMode, H4_ROOTS, H4_SHELL, P4_EXPERTS,
        P4_MAX_CONTEXT, P4_NORM_BUDGET, P4_Q48_ONE, P4_TOTAL_BUCKETS, P4_UNIFORM_MASS, P4_VOCAB,
    },
    PickClassRule,
};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, write_p4_deployment, ClientArtifactFile, DiscreteCountModel, SCHEMA_P4,
};
use uor_r4_integer::discrete_d11::{DISCRETE_TRUNK_SCHEMA, MAGIC, SNAPSHOT_SCHEMA};

fn h4_root_coeffs_from_p4_formula() -> [[[i8; 2]; 4]; 120] {
    const EVEN_PERMS: [[usize; 4]; 12] = [
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
    ];
    let mut roots = [[[0i8; 2]; 4]; 120];
    let mut idx = 0usize;
    for axis in [0usize, 1, 2, 3] {
        roots[idx][axis] = [-2, 0];
        roots[idx + 1][axis] = [2, 0];
        idx += 2;
    }
    for signs in 0..16 {
        for (axis, slot) in roots[idx].iter_mut().enumerate() {
            *slot = [if signs & (1 << axis) == 0 { -1 } else { 1 }, 0];
        }
        idx += 1;
    }
    for p in EVEN_PERMS {
        for signs in 0..8 {
            let s1 = if signs & 4 == 0 { -1 } else { 1 };
            let s2 = if signs & 2 == 0 { -1 } else { 1 };
            let s3 = if signs & 1 == 0 { -1 } else { 1 };
            let signed = [[0, 0], [s1, 0], [0, s2], [-s3, s3]];
            for axis in 0..4 {
                roots[idx][axis] = signed[p[axis]];
            }
            idx += 1;
        }
    }
    roots
}

fn build_synthetic_trunk_4096(context: u32) -> Vec<u8> {
    let vocab = P4_VOCAB;
    let mut stages = 1usize;
    let mut span = 3u32;
    while span <= context {
        stages += 1;
        span = span.saturating_mul(3);
    }
    let roots: [[i8; 8]; 16] = [
        [2, 2, 0, 0, 0, 0, 0, 0],
        [-2, -2, 0, 0, 0, 0, 0, 0],
        [-2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, -2, 0, -2, 0, 0, 0],
        [0, 0, -2, 0, 2, 0, 0, 0],
        [0, 0, 0, -2, 0, 0, 0, -2],
        [0, 0, 0, -2, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, -2, -2, 0],
        [0, 0, 0, 0, 0, -2, 2, 0],
        [0, 0, 0, 0, 0, 2, -2, 0],
        [0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 2, 0, 0, 0, -2],
        [0, 0, 0, 2, 0, 0, 0, 2],
        [0, 0, 2, 0, -2, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0],
        [2, -2, 0, 0, 0, 0, 0, 0],
    ];
    let mut table = Vec::with_capacity(vocab * 32);
    let mut cells = Vec::with_capacity(vocab);
    for tok in 0..vocab {
        let r = (tok & 15) as u8;
        for block in 0..4 {
            // Give 16 near aliases one distinct inner block while keeping their outer cell.
            let root = if (16..32).contains(&tok) && block == 1 {
                (r + 1) % 16
            } else {
                r
            };
            for d in roots[usize::from(root)] {
                table.push(i32::from(d));
            }
        }
        cells.push((u32::from(r) << 4) + u32::from(r));
    }
    let routes = vec![0u32; 4 * stages];
    let gates = vec![0i32; 9 * stages];
    let mut blob = Vec::new();
    let push_sec = |blob_buf: &mut Vec<u8>, bytes: &[u8]| {
        let pad = (64 - blob_buf.len() % 64) % 64;
        blob_buf.resize(blob_buf.len() + pad, 0);
        let off = blob_buf.len() as u64;
        blob_buf.extend_from_slice(bytes);
        json!({"offset": off, "bytes": bytes.len()})
    };
    let push_u32_sec = |blob_buf: &mut Vec<u8>, name: &str, words: &[u32]| {
        let b: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let span = push_sec(blob_buf, &b);
        json!({"name": name, "kind": "u32", "len": words.len(), "span": span})
    };
    let t_b: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
    let g_b: Vec<u8> = gates.iter().flat_map(|x| x.to_le_bytes()).collect();
    let t_span = push_sec(&mut blob, &t_b);
    let c_ent = push_u32_sec(&mut blob, "discrete.cells", &cells);
    let r_ent = push_u32_sec(&mut blob, "discrete.routes", &routes);
    let g_span = push_sec(&mut blob, &g_b);
    let mut tables = vec![
        json!({"name": "discrete.table", "kind": "i32", "len": table.len(), "span": t_span}),
        c_ent,
        r_ent,
        json!({"name": "discrete.gates", "kind": "i32", "len": gates.len(), "span": g_span}),
    ];
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let mut unigram = vec![1u32; 1 + vocab];
    unigram[0] = 0;
    tables.push(push_u32_sec(
        &mut blob,
        "discrete.trunk.step_candidates",
        &steps,
    ));
    tables.push(push_u32_sec(
        &mut blob,
        "discrete.trunk.norm_candidates",
        &norms,
    ));
    tables.push(push_u32_sec(&mut blob, "discrete.trunk.unigram", &unigram));
    for name in [
        "discrete.trunk.read_table",
        "discrete.trunk.rec_table",
        "discrete.trunk.read_rec_table",
        "discrete.trunk.rec_shuf_table",
        "discrete.trunk.read_rec_shuf_table",
        "discrete.trunk.rec_unit_table",
        "discrete.trunk.read_rec_unit_table",
        "discrete.trunk.query_table",
        "discrete.trunk.query_read_table",
        "discrete.trunk.query_rec_table",
        "discrete.trunk.query_rec_shuf_table",
        "discrete.trunk.query_rec_unit_table",
    ] {
        tables.push(push_u32_sec(&mut blob, name, &[]));
    }
    let trunk_hdr = serde_json::to_vec(&json!({
        "schema": DISCRETE_TRUNK_SCHEMA,
        "shape": {
            "vocabulary": vocab, "blocks": 4, "context": context, "kept_cells": 3, "stages": stages,
            "read_mode": "relation", "read_kept": 3, "read_sweeps": 1, "read_position": 0,
            "objective_policy": "nearest_address_margin_v1", "read_initialization": "zero_gates_v1",
            "seed": 7, "epoch": 1,
        },
        "matrices": [],
        "tables": tables,
        "source": {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "snapshot_sha256": "00".repeat(32),
            "operator_kappa": "blake3:a0ea3a10dd10c5f4856b0d881d33d94c3293bad0e68bd07a47918e2bb82c819b",
            "composition_kappa": "blake3:f2986c8e68dcb30a9cb511a42547179a87b66ef212a87b728765d03e70e640b0",
            "kernel_sha256": "11".repeat(32),
            "source_sha256": "22".repeat(32),
            "geometry_sha256": "33".repeat(32),
        }
    }))
    .unwrap();
    let mut trunk = Vec::new();
    trunk.extend_from_slice(MAGIC);
    trunk.extend_from_slice(&(trunk_hdr.len() as u64).to_le_bytes());
    trunk.extend_from_slice(&trunk_hdr);
    let pad = (64 - trunk.len() % 64) % 64;
    trunk.resize(trunk.len() + pad, 0);
    trunk.extend_from_slice(&blob);
    trunk
}

type SyntheticRows<'a> = &'a [(u16, u32, i8)];

fn build_synthetic_uors3t02(
    token_h4: &[u8; P4_VOCAB],
    train_sha: [u8; 32],
    primary: &[(u32, SyntheticRows<'_>)],
    backoff: &[(u16, SyntheticRows<'_>)],
) -> Vec<u8> {
    let p_ctx_len = primary.len() as u32;
    let b_ctx_len = backoff.len() as u32;
    let mut p_rows = Vec::new();
    let mut p_ctxs = Vec::new();
    for &(k, slice) in primary {
        let rs = p_rows.len() as u32;
        let rl = slice.len() as u16;
        let tot: u64 = slice.iter().map(|x| u64::from(x.1)).sum();
        p_ctxs.push((k, tot, rs, rl));
        p_rows.extend_from_slice(slice);
    }
    let mut b_rows = Vec::new();
    let mut b_ctxs = Vec::new();
    for &(pk, slice) in backoff {
        let rs = b_rows.len() as u32;
        let rl = slice.len() as u16;
        let tot: u64 = slice.iter().map(|x| u64::from(x.1)).sum();
        b_ctxs.push((pk, tot, rs, rl));
        b_rows.extend_from_slice(slice);
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"UORS3T02");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.push(7); // GoldenShell
    out.push(0);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&train_sha);
    out.extend_from_slice(&p_ctx_len.to_le_bytes());
    out.extend_from_slice(&(p_rows.len() as u32).to_le_bytes());
    out.extend_from_slice(&b_ctx_len.to_le_bytes());
    out.extend_from_slice(&(b_rows.len() as u32).to_le_bytes());
    out.extend_from_slice(token_h4);
    for (k, tot, rs, rl) in p_ctxs {
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&tot.to_le_bytes());
        out.extend_from_slice(&rs.to_le_bytes());
        out.extend_from_slice(&rl.to_le_bytes());
    }
    for (nt, cnt, err) in p_rows {
        out.extend_from_slice(&nt.to_le_bytes());
        out.extend_from_slice(&cnt.to_le_bytes());
        out.push(err as u8);
    }
    for (pk, tot, rs, rl) in b_ctxs {
        out.extend_from_slice(&pk.to_le_bytes());
        out.extend_from_slice(&tot.to_le_bytes());
        out.extend_from_slice(&rs.to_le_bytes());
        out.extend_from_slice(&rl.to_le_bytes());
    }
    for (nt, cnt, err) in b_rows {
        out.extend_from_slice(&nt.to_le_bytes());
        out.extend_from_slice(&cnt.to_le_bytes());
        out.push(err as u8);
    }
    out
}

fn build_synthetic_uors2s01(
    c_sha: [u8; 32],
    families: [&[(u64, SyntheticRows<'_>)]; 3],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"UORS2S01");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&c_sha);
    for (fid, fam) in families.iter().enumerate() {
        out.push(fid as u8);
        let mut rows = Vec::new();
        let mut ctxs = Vec::new();
        for &(k, slice) in *fam {
            let rs = rows.len() as u32;
            let rl = slice.len() as u16;
            let tot: u64 = slice.iter().map(|x| u64::from(x.1)).sum();
            ctxs.push((k, rs, rl, tot));
            rows.extend_from_slice(slice);
        }
        out.extend_from_slice(&(ctxs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for (k, rs, rl, tot) in ctxs {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&rs.to_le_bytes());
            out.extend_from_slice(&rl.to_le_bytes());
            out.extend_from_slice(&tot.to_le_bytes());
        }
        for (nt, cnt, err) in rows {
            out.extend_from_slice(&nt.to_le_bytes());
            out.extend_from_slice(&cnt.to_le_bytes());
            out.push(err as u8);
        }
    }
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    out
}

fn ref_uors3t02_lookup(
    raw_uors3t02: &[u8],
    p_errs: &[i8],
    b_errs: &[i8],
    pkey: u32,
    target: u16,
) -> u64 {
    let p_ctx_len = u32::from_le_bytes(raw_uors3t02[56..60].try_into().unwrap()) as usize;
    let p_row_len = u32::from_le_bytes(raw_uors3t02[60..64].try_into().unwrap()) as usize;
    let b_ctx_len = u32::from_le_bytes(raw_uors3t02[64..68].try_into().unwrap()) as usize;
    let p_ctx_off = 72 + P4_VOCAB;
    let p_row_off = p_ctx_off + p_ctx_len * 18;
    let b_ctx_off = p_row_off + p_row_len * 7;
    let b_row_off = b_ctx_off + b_ctx_len * 16;

    let read_row = |off: usize, err_ov: i8| -> (u16, u64) {
        let nt = u16::from_le_bytes([raw_uors3t02[off], raw_uors3t02[off + 1]]);
        let cnt = u32::from_le_bytes(raw_uors3t02[off + 2..off + 6].try_into().unwrap());
        let scale = (128i64 + i64::from(err_ov)).max(1) as u64;
        (nt, u64::from(cnt) * scale)
    };

    for i in 0..p_ctx_len {
        let co = p_ctx_off + i * 18;
        let k = u32::from_le_bytes(raw_uors3t02[co..co + 4].try_into().unwrap());
        if k == pkey {
            let rs =
                u32::from_le_bytes(raw_uors3t02[co + 12..co + 16].try_into().unwrap()) as usize;
            let rl =
                u16::from_le_bytes(raw_uors3t02[co + 16..co + 18].try_into().unwrap()) as usize;
            let mut eff_tot = 0u64;
            let mut eff_hit = 0u64;
            for (r_idx, &err_ov) in p_errs[rs..rs + rl].iter().enumerate() {
                let (nt, w) = read_row(p_row_off + (rs + r_idx) * 7, err_ov);
                eff_tot += w;
                if nt == target {
                    eff_hit = w;
                }
            }
            if eff_hit == 0 {
                return 1;
            }
            return 1
                + ((u128::from(eff_hit) * u128::from(P4_NORM_BUDGET)) / u128::from(eff_tot.max(1)))
                    as u64;
        }
    }
    let bk = (pkey >> 16) as u16;
    for i in 0..b_ctx_len {
        let co = b_ctx_off + i * 16;
        let pk = u16::from_le_bytes([raw_uors3t02[co], raw_uors3t02[co + 1]]);
        if pk == bk {
            let rs =
                u32::from_le_bytes(raw_uors3t02[co + 10..co + 14].try_into().unwrap()) as usize;
            let rl =
                u16::from_le_bytes(raw_uors3t02[co + 14..co + 16].try_into().unwrap()) as usize;
            let mut eff_tot = 0u64;
            let mut eff_hit = 0u64;
            for (r_idx, &err_ov) in b_errs[rs..rs + rl].iter().enumerate() {
                let (nt, w) = read_row(b_row_off + (rs + r_idx) * 7, err_ov);
                eff_tot += w;
                if nt == target {
                    eff_hit = w;
                }
            }
            if eff_hit == 0 {
                return 1;
            }
            return 1
                + ((u128::from(eff_hit) * u128::from(P4_NORM_BUDGET)) / u128::from(eff_tot.max(1)))
                    as u64;
        }
    }
    let (nbrs, nn) = starling_7_neighbours(bk % (9 * 9 * 9 * 9));
    let mut sum_tot = 0u64;
    let mut sum_hit = 0u64;
    for &nbk in &nbrs[..nn] {
        for i in 0..b_ctx_len {
            let co = b_ctx_off + i * 16;
            let pk = u16::from_le_bytes([raw_uors3t02[co], raw_uors3t02[co + 1]]);
            if pk == nbk {
                let rs =
                    u32::from_le_bytes(raw_uors3t02[co + 10..co + 14].try_into().unwrap()) as usize;
                let rl =
                    u16::from_le_bytes(raw_uors3t02[co + 14..co + 16].try_into().unwrap()) as usize;
                for (r_idx, &err_ov) in b_errs[rs..rs + rl].iter().enumerate() {
                    let (nt, w) = read_row(b_row_off + (rs + r_idx) * 7, err_ov);
                    sum_tot += w;
                    if nt == target {
                        sum_hit += w;
                    }
                }
            }
        }
    }
    if sum_tot > 0 {
        if sum_hit == 0 {
            return 1;
        }
        return 1
            + ((u128::from(sum_hit) * u128::from(P4_NORM_BUDGET)) / u128::from(sum_tot)) as u64;
    }
    P4_UNIFORM_MASS
}

#[test]
fn h4_shell_matches_canonical_600cell_root_coefficients() {
    let coeffs = h4_root_coeffs_from_p4_formula();
    for u in 0..H4_ROOTS {
        for v in 0..H4_ROOTS {
            let mut sa = 0i32;
            let mut sb = 0i32;
            for ([a1, b1], [a2, b2]) in coeffs[u].iter().zip(&coeffs[v]) {
                let a1 = i32::from(*a1);
                let b1 = i32::from(*b1);
                let a2 = i32::from(*a2);
                let b2 = i32::from(*b2);
                sa += a1 * a2 + b1 * b2;
                sb += a1 * b2 + b1 * a2 + b1 * b2;
            }
            let expected = match (sa, sb) {
                (4, 0) => 0u8,
                (0, 2) => 1u8,
                (2, 0) => 2u8,
                (-2, 2) => 3u8,
                (0, 0) => 4u8,
                (2, -2) => 5u8,
                (-2, 0) => 6u8,
                (0, -2) => 7u8,
                (-4, 0) => 8u8,
                _ => panic!("unexpected inner product"),
            };
            assert_eq!(H4_SHELL[u][v], expected, "mismatch at ({u}, {v})");
        }
    }
}

#[test]
fn uors3c01_lossless_codec_matches_uors3t02_across_all_4095_tokens_and_refuses_tamper() {
    let mut token_h4 = [0u8; P4_VOCAB];
    for (i, slot) in token_h4.iter_mut().enumerate() {
        *slot = (i % H4_ROOTS) as u8;
    }
    let train_sha = [0x42u8; 32];

    let p_pref = [0u16, 10, 20, 30];
    let b_pref = [0u16, 11, 21, 31];
    let pkey_hit = golden_shell_pkey(&token_h4, &p_pref);
    let pkey_bk = golden_shell_pkey(&token_h4, &b_pref);
    let mut bk_val = (pkey_bk >> 16) as u16;
    if bk_val == 6560 {
        bk_val = 100;
    }
    let pk_starling_src = bk_val + 1; // guaranteed neighbour in dim 3!
    let pk_miss_only = 3280u16; // center shell (4,4,4,4); pick a prefix whose bk isn't hit directly

    // Build sorted strictly increasing primary and backoff contexts covering all 4 varint tiers.
    let p_rows_data: [(u16, u32, i8); 4] = [
        (5, 3, -40),        // tier 0 (1..=13)
        (9, 200, 75),       // tier 13 (14..=255)
        (17, 12_345, -120), // tier 14 (256..=65535)
        (250, 99_999, 127), // tier 15 (>= 65536)
    ];
    let b_rows_a: [(u16, u32, i8); 2] = [(12, 13, 15), (99, 500, -10)];
    let b_rows_b: [(u16, u32, i8); 2] = [(12, 70_000, -5), (400, 2, 120)];

    let mut prim_vec = vec![(pkey_hit, &p_rows_data[..])];
    prim_vec.sort_by_key(|x| x.0);
    let mut back_vec = vec![(bk_val, &b_rows_a[..]), (pk_starling_src, &b_rows_b[..])];
    back_vec.sort_by_key(|x| x.0);

    let raw = build_synthetic_uors3t02(&token_h4, train_sha, &prim_vec, &back_vec);
    let p_errs = [0i8; 4]; // non-zero embedded bytes preserved when overridden -> test explicit errs!
    let p_errs_ovr = [-40i8, 75, -120, 127];
    let b_errs_ovr = [15i8, -10, -5, 120];
    let compact =
        encode_compact_s3_from_uors3t02(&raw, Some(&p_errs_ovr), Some(&b_errs_ovr), train_sha)
            .unwrap();
    let _ = p_errs;
    let view = CompactS3View::from_bytes(&compact).unwrap();
    let mut sparse = Vec::new();

    // Test 1) Primary hit, 2) Backoff hit, 3) Starling 7-neighbour hit, 4) Uniform miss
    for (pref, test_pkey) in [(&p_pref[..], pkey_hit), (&b_pref[..], pkey_bk)] {
        let (meta, def_m) = view.fill_sparse_masses_q48(pref, &mut sparse).unwrap();
        assert_eq!(meta.pkey, test_pkey);
        let mut dense = vec![def_m; P4_VOCAB];
        for &(t, m) in &sparse {
            dense[usize::from(t)] = m;
        }
        for target in 1..P4_VOCAB as u16 {
            let exp = ref_uors3t02_lookup(&raw, &p_errs_ovr, &b_errs_ovr, test_pkey, target);
            assert_eq!(
                dense[usize::from(target)],
                exp,
                "S3 sparse mismatch target={target}"
            );
            assert_eq!(view.lookup_modulated_q48(test_pkey, target).unwrap().0, exp);
        }
    }

    // Starling hit & Uniform miss tests via custom prefix finding
    let mut star_pref = None;
    let mut unif_pref = None;
    for a in 1..120u16 {
        for b in 1..120u16 {
            let cand = [0u16, a, b];
            let pk = golden_shell_pkey(&token_h4, &cand);
            let bk = (pk >> 16) as u16;
            if bk == bk_val - 1 && pk != pkey_hit && bk != bk_val && bk != pk_starling_src {
                star_pref = Some(cand);
            }
            let (nbrs, nn) = starling_7_neighbours(bk % 6561);
            if pk != pkey_hit
                && bk != bk_val
                && bk != pk_starling_src
                && !nbrs[..nn].contains(&bk_val)
                && !nbrs[..nn].contains(&pk_starling_src)
                && bk == pk_miss_only
            {
                unif_pref = Some(cand);
            }
            if star_pref.is_some() && unif_pref.is_some() {
                break;
            }
        }
    }
    let _ = star_pref;
    // Always test Starling directly using any prefix whose bk neighbours bk_val or pk_starling_src:
    'outer: for a in 1..120u16 {
        for b in 1..120u16 {
            let cand = [0u16, a, b];
            let pk = golden_shell_pkey(&token_h4, &cand);
            let bk = (pk >> 16) as u16;
            let (nbrs, nn) = starling_7_neighbours(bk % 6561);
            if pk != pkey_hit
                && bk != bk_val
                && bk != pk_starling_src
                && (nbrs[..nn].contains(&bk_val) || nbrs[..nn].contains(&pk_starling_src))
            {
                let (meta, def_m) = view.fill_sparse_masses_q48(&cand, &mut sparse).unwrap();
                let mut dense = vec![def_m; P4_VOCAB];
                for &(t, m) in &sparse {
                    dense[usize::from(t)] = m;
                }
                for target in [1u16, 5, 12, 99, 400, 1024, 4095] {
                    let exp = ref_uors3t02_lookup(&raw, &p_errs_ovr, &b_errs_ovr, pk, target);
                    assert_eq!(dense[usize::from(target)], exp);
                    assert_eq!(view.lookup_modulated_q48(pk, target).unwrap().0, exp);
                }
                assert_eq!(meta.hit_kind.as_str(), "starling");
                break 'outer;
            }
        }
    }

    // Tamper check
    let mut bad = compact.clone();
    bad[100] ^= 0x55;
    assert!(CompactS3View::from_bytes(&bad).is_err());
}

#[test]
fn uors2c01_lossless_codec_and_cache_oracle_parity() {
    let c_sha = [0x77u8; 32];
    let seq = [0u16, 10, 20, 30, 40, 10, 20, 30, 50];
    let skeys = skip_keys_for_prefix(&seq);
    let r0: [(u16, u32, i8); 3] = [(5, 7, 10), (8, 222, -33), (19, 80_000, 127)];
    let r1: [(u16, u32, i8); 2] = [(9, 1_500, -120), (14, 1, 0)];
    let r2: [(u16, u32, i8); 2] = [(3, 2, 50), (4095, 13, -50)];
    let f0 = [(skeys[0], &r0[..])];
    let f1 = [(skeys[1], &r1[..])];
    let f2 = [(skeys[2], &r2[..])];
    let raw_s2 = build_synthetic_uors2s01(c_sha, [&f0, &f1, &f2]);
    let errs_ovr = [10i8, -33, 127, -120, 0, 50, -50];
    let compact_s2 = encode_compact_s2_skip_from_uors2s01(&raw_s2, Some(&errs_ovr), c_sha).unwrap();
    let view = CompactS2SkipView::from_bytes(&compact_s2).unwrap();

    let mut sparse = Vec::new();
    for (fam, &k) in skeys.iter().enumerate() {
        let (hit, def_m) = view.fill_sparse_family_q48(fam, k, &mut sparse).unwrap();
        assert!(hit);
        let mut dense = vec![def_m; P4_VOCAB];
        for &(t, m) in &sparse {
            dense[usize::from(t)] = m;
        }
        for target in 1..P4_VOCAB as u16 {
            let exp = view.lookup_modulated_q48(fam, k, target).unwrap();
            assert_eq!(dense[usize::from(target)], exp, "fam {fam} target {target}");
        }
        // Unseen key yields uniform mass
        let miss_k = (u64::from(fam as u8) << 48) | 0x00FF_FFFF;
        let (miss_hit, miss_def) = view
            .fill_sparse_family_q48(fam, miss_k, &mut sparse)
            .unwrap();
        assert!(!miss_hit);
        assert_eq!(miss_def, P4_UNIFORM_MASS);
        assert!(sparse.is_empty());
    }

    let mut bad = compact_s2.clone();
    *bad.last_mut().unwrap() ^= 0xFF;
    assert!(CompactS2SkipView::from_bytes(&bad).is_err());

    // Incremental CausalSessionCache bit-identity across 2048-token record + BOS reset
    let mut cache_inc = CausalSessionCache::new();
    let mut full_seq = vec![0u16; 2050];
    for (i, slot) in full_seq.iter_mut().enumerate().skip(1) {
        *slot = ((i * 17 + (i % 7) * 5) % 400 + 2) as u16;
    }
    for &t in &full_seq {
        cache_inc.observe_token(t);
    }
    let mut sp_cache: [Vec<(u16, u64)>; 4] = Default::default();
    let (hs, defs) = cache_inc.eval_sparse_all(&mut sp_cache);
    let mut dense = [defs[0]; P4_VOCAB];
    let mut dense1 = [defs[1]; P4_VOCAB];
    let mut dense2 = [defs[2]; P4_VOCAB];
    let mut dense3 = [defs[3]; P4_VOCAB];
    for &(t, m) in &sp_cache[0] {
        dense[usize::from(t)] = m;
    }
    for &(t, m) in &sp_cache[1] {
        dense1[usize::from(t)] = m;
    }
    for &(t, m) in &sp_cache[2] {
        dense2[usize::from(t)] = m;
    }
    for &(t, m) in &sp_cache[3] {
        dense3[usize::from(t)] = m;
    }
    for target in 1..P4_VOCAB as u16 {
        let (ehs, ems) = cache_inc.peek_score_readonly(target);
        assert_eq!(hs, ehs);
        assert_eq!(dense[usize::from(target)], ems[0]);
        assert_eq!(dense1[usize::from(target)], ems[1]);
        assert_eq!(dense2[usize::from(target)], ems[2]);
        assert_eq!(dense3[usize::from(target)], ems[3]);
    }
    // BOS reset restores fresh empty-record uniform distribution
    cache_inc.observe_token(0);
    assert_eq!(cache_inc.peek_score_readonly(5), (0, [P4_UNIFORM_MASS; 4]));
}

fn build_synthetic_uorknc03_v2_converted(c_sha: [u8; 32]) -> Vec<u8> {
    let highest = 1u32;
    let mut rspec0 = [0u64; 4];
    let mut aspec0 = [0u64; 4];
    rspec0[0] = 1;
    rspec0[1] = 1;
    aspec0[0] = 2;
    let mut rspec1 = [0u64; 4];
    rspec1[1] = 1;

    let mut v3 = Vec::new();
    v3.extend_from_slice(b"UORKNC03");
    v3.extend_from_slice(&highest.to_le_bytes());
    v3.extend_from_slice(&c_sha);
    // Order 0 section: 1 context, 2 followers [(1, r=1, a=1), (2, r=2, a=1)]
    v3.extend_from_slice(&2u32.to_le_bytes());
    v3.extend_from_slice(&1u32.to_le_bytes());
    for w in rspec0.into_iter().chain(aspec0) {
        v3.extend_from_slice(&w.to_le_bytes());
    }
    for _ in 0..4 {
        v3.extend_from_slice(&0u32.to_le_bytes());
    }
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&2u32.to_le_bytes());
    v3.push(1);
    v3.extend_from_slice(&(1u16 << 4).to_le_bytes());
    v3.extend_from_slice(&((2u16 << 4) | 1u16).to_le_bytes());
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&0u32.to_le_bytes());
    // Order 1 section (shared): context [5] => next=2, count=2
    v3.extend_from_slice(&1u32.to_le_bytes());
    v3.extend_from_slice(&1u32.to_le_bytes());
    for w in rspec1.into_iter().chain(rspec1) {
        v3.extend_from_slice(&w.to_le_bytes());
    }
    for _ in 0..4 {
        v3.extend_from_slice(&0u32.to_le_bytes());
    }
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&1u32.to_le_bytes());
    v3.push(1);
    v3.extend_from_slice(&[0x05, 0x00]);
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&1u32.to_le_bytes());
    v3.push(0);
    v3.extend_from_slice(&((2u16 << 4) | 1u16).to_le_bytes());
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3
}

fn check_lexical_client(
    dir: &std::path::Path,
    p4_path: &std::path::Path,
    trunk: &uor_r4_integer::discrete_d11::IntegerDiscreteModel,
    p4: &uor_r4_count_runtime::p4_serving::P4State<'_>,
    prior_winner: u16,
    prefix: &[u16],
) {
    use uor_r4_integer::discrete_d11::native_occurrence::lexical::{Carry, WEIGHT_WORDS};
    let path = dir.join("synthetic.uorlex");
    let mut session = P4Session::new();
    session.sync_prompt(prefix).unwrap();
    session.step(p4, true, false).unwrap();
    let priors = session.last_scores_q48();
    let recurrence = trunk
        .evaluate_prefix_traced(&prefix.iter().map(|&x| u32::from(x)).collect::<Vec<_>>())
        .unwrap()
        .trunk_pointer
        .unwrap()
        .recurrence_code;
    // All four blocks use identity rotations. For [0,2,5] the sum is
    // roots[0] + 2*roots[2] + roots[5], which snaps to root 2 in each block.
    assert_eq!(prefix, [0, 2, 5]);
    assert_eq!(recurrence, 0x2222);
    let carry = Carry::from_packed(recurrence);
    let codebook = trunk.packed_codebook_roots();
    let addresses = |id: u16| {
        let code = codebook[usize::from(id)];
        carry
            .addresses(std::array::from_fn(|b| ((code >> (4 * b)) & 15) as u8))
            .unwrap()
    };
    assert!(prior_winner < 16);
    let target = prior_winner + 16;
    assert!(!prefix.contains(&target));
    let target_addresses = addresses(target);
    let prior_addresses = addresses(prior_winner);
    assert_eq!(target_addresses[0], prior_addresses[0]);
    assert_eq!(target_addresses[3], prior_addresses[3]);
    assert_ne!(target_addresses[1], prior_addresses[1]);
    assert!(priors[usize::from(prior_winner)] > priors[usize::from(target)]);
    let write = |favor: Option<[u16; 4]>, state: [u8; 32]| {
        let mut words = [0i16; WEIGHT_WORDS];
        if let Some(favor) = favor {
            words.fill(i16::MIN);
            for addr in favor {
                words[usize::from(addr)] = i16::MAX;
            }
        }
        let mut payload = state.to_vec();
        payload.extend_from_slice(b"UORLEX01");
        for word in words {
            payload.extend_from_slice(&word.to_le_bytes());
        }
        payload.extend_from_slice(&Sha256::digest(&payload));
        fs::write(&path, payload).unwrap();
    };
    let cli = |ids: &str, steps: &str| {
        Command::new(env!("CARGO_BIN_EXE_uor-chat"))
            .args([
                "--discrete",
                p4_path.to_str().unwrap(),
                "--prompt-ids",
                ids,
                "--max-tokens",
                steps,
                "--p1-trace",
                "--p4-lexical",
                path.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let read = |ids: &str, steps: &str| {
        let out = cli(ids, steps);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice::<serde_json::Value>(&out.stdout).unwrap()
    };
    let ids = prefix
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let state = p4.state_sha256().unwrap();
    for (label, favor, winner) in [
        ("zero", None, prior_winner),
        ("favor absent word", Some(target_addresses), target),
        ("favor prior", Some(prior_addresses), prior_winner),
    ] {
        write(favor, state);
        let output = read(&ids, "2");
        let committed = &output["generated_ids"];
        let trace = &output["p4_steps"][0];
        assert_eq!(trace["lexical"]["carry_code"], recurrence);
        assert_eq!(trace["prior_winner"]["token"], prior_winner);
        assert_eq!(trace["token"], winner);
        assert_eq!(committed[0], winner);
        let expected = priors[usize::from(winner)] as i64
            + if favor.is_some() {
                4 * i64::from(i16::MAX) * (1i64 << 32)
            } else {
                0
            };
        assert_eq!(trace["lexical"]["score_q48"], expected);
        assert_eq!(trace["lexical"]["addresses"], json!(addresses(winner)));
        assert_eq!(read(&format!("12,11,{ids}"), "1")["p4_steps"][0], *trace);
        let after = read(&format!("{ids},{winner}"), "1");
        assert_eq!(output["p4_steps"][1], after["p4_steps"][0]);
        assert_eq!(committed[1], after["generated_ids"][0]);
        println!("lexical {label}: carry={recurrence} prior={prior_winner} target={target} target_prior={} committed={committed} score={expected} addresses={:?}",
            priors[usize::from(target)], addresses(winner));
    }
    let repeated = vec!["2"; P4_MAX_CONTEXT];
    let long = read(&repeated.join(","), "2");
    let first = long["generated_ids"][0].as_u64().unwrap().to_string();
    let mut slid = repeated[1..].to_vec();
    slid.push(&first);
    assert_eq!(long["generated_ids"].as_array().unwrap().len(), 2);
    assert_eq!(
        long["p4_steps"][1],
        read(&slid.join(","), "1")["p4_steps"][0]
    );
    write(None, [0xa5; 32]);
    assert!(!cli(&ids, "1").status.success());
    write(None, state);
    let mut corrupt = fs::read(&path).unwrap();
    corrupt[41] ^= 1;
    fs::write(&path, corrupt).unwrap();
    assert!(!cli(&ids, "1").status.success());
}

#[test]
fn p4_deployment_end_to_end_bit_exact_and_uor_chat_cli() {
    let trunk = build_synthetic_trunk_4096(192);
    let trunk_sha = hex::encode(Sha256::digest(&trunk));
    let c_sha = [0x55u8; 32];
    let r_sha = [0x11u8; 32];
    let counts_bytes = build_synthetic_uorknc03_v2_converted(c_sha);
    let compact_sha = hex::encode(Sha256::digest(&counts_bytes));

    // highest=1, Five classes: 5 * (2*16) buckets * (1+4) experts = 800 words
    let a1s_len = 5 * (2 * 16) * 5;
    let fit_json = serde_json::to_vec(&json!({
        "schema": "uor-r4.d37.r-fit/1",
        "arm": "a1s",
        "pick_classes": 5,
        "weight_format": "UORS5W02",
        "q32_weight_words": a1s_len,
        "prepared": {
            "format": "uor-r4.d37.r-preparation/1",
            "c_sha256": hex::encode(c_sha),
            "r_sha256": hex::encode(r_sha),
            "pick_mode": "identity",
            "trunk_sha256": trunk_sha,
            "compact_sha256": compact_sha,
        }
    }))
    .unwrap();
    let fit_digest: [u8; 32] = Sha256::digest(&fit_json).into();
    let mut a1s_words = vec![0u64; a1s_len];
    for chunk in a1s_words.chunks_exact_mut(5) {
        chunk[0] = 1u64 << 32;
    }
    let a1s_fit = uor_r4_count_runtime::FrozenPickMixture::with_rule(
        1,
        fit_digest,
        a1s_words,
        PickClassRule::Five,
    )
    .unwrap();
    let a1s_bytes = a1s_fit.to_bytes();

    let mut token_h4 = [0u8; P4_VOCAB];
    for (i, slot) in token_h4.iter_mut().enumerate() {
        *slot = (i % H4_ROOTS) as u8;
    }
    let pref = [0u16, 2, 5];
    let pk_hit = golden_shell_pkey(&token_h4, &pref);
    let s3_rows = [
        (3u16, 5u32, 30i8),
        (7u16, 120u32, 100i8),
        (42u16, 4u32, -20i8),
    ];
    let raw_s3 = build_synthetic_uors3t02(&token_h4, c_sha, &[(pk_hit, &s3_rows)], &[]);
    let compact_s3 = encode_compact_s3_from_uors3t02(&raw_s3, None, None, c_sha).unwrap();

    let skeys = skip_keys_for_prefix(&pref);
    let r_sk0 = [(7u16, 30u32, 40i8)];
    let raw_s2 = build_synthetic_uors2s01(c_sha, [&[(skeys[0], &r_sk0)], &[], &[]]);
    let compact_s2 = encode_compact_s2_skip_from_uors2s01(&raw_s2, None, c_sha).unwrap();

    // Non-trivial S4 conjunction table linking pair (2, 5) to role_class=2
    let pk_s4 = s4_pair_key(2, 5).unwrap();
    let s4_entries = [S4PointerEntry {
        role_class: 2,
        span_offset: 0,
        count: 12,
        err_i8: 45,
    }];
    let s4_tbl = S4ConjunctionTable::from_sorted_rows(c_sha, &[(pk_s4, &s4_entries[..])]).unwrap();
    let s4_bytes = s4_tbl.encode_bytes().unwrap();

    // Non-uniform 22-expert S6 weights putting meaningful mass on KN, ML_0, ML_1, Pick, S3, S2_0, Cache_0..3!
    let mut s6 = BrierS6Weights::uniform(P4_TOTAL_BUCKETS, P4_EXPERTS, r_sha).unwrap();
    for b in 0..P4_TOTAL_BUCKETS {
        let off = b * P4_EXPERTS;
        let q16 = 1u64 << 28; // 1/16 of 2^32
        s6.weights_q32[off..off + P4_EXPERTS].fill(1);
        s6.weights_q32[off] = 4 * q16; // KN
        s6.weights_q32[off + 1] = q16; // ML_0
        s6.weights_q32[off + 2] = q16; // ML_1
        s6.weights_q32[off + 12] = q16; // Uniform
        s6.weights_q32[off + 13] = 2 * q16; // Pick
        s6.weights_q32[off + 14] = 3 * q16; // S3
        s6.weights_q32[off + 15] = 2 * q16; // S2_0
        s6.weights_q32[off + 18] = q16; // Cache_uni
        s6.weights_q32[off + 19] = P3_Q32_ONE - 15 * q16 - 13; // Cache_bi (sum = 2^32!)
    }
    let s6_bytes = s6.encode_bytes().unwrap();

    let dir = std::env::temp_dir().join(format!("uor-p4-deploy-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let p4_path = dir.join("p4-test.d11");
    write_p4_deployment(
        &trunk,
        &fit_json,
        &a1s_bytes,
        &counts_bytes,
        &compact_s3,
        &compact_s2,
        &s4_bytes,
        &s6_bytes,
        &p4_path,
    )
    .unwrap();

    // Verify load_client_artifact streams and decodes P4 cleanly, and refuses sidecar conflict
    assert!(load_client_artifact(&p4_path, true).is_err());
    let loaded_raw = fs::read(&p4_path).unwrap();
    let (model, state_bytes, expected_sha, identity) =
        match load_client_artifact(&p4_path, false).unwrap() {
            ClientArtifactFile::Embedded {
                model,
                state_bytes,
                expected_state_sha256,
                identity,
            } => (model, state_bytes, expected_state_sha256, identity),
            ClientArtifactFile::Legacy(_) => panic!("expected P4 embedded deployment"),
        };
    let dc =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_sha, identity).unwrap();
    assert!(dc.state.is_none());
    let p4 = dc.p4.as_ref().expect("P4 engine loaded");

    // Verify exhaustive 4,095-token bit-exact parity vs naive brier_mix_q48 on prompt `pref`!
    let mut sess = P4Session::new();
    sess.sync_prompt(&pref).unwrap();
    let (top_id, trace_opt) = sess.step(&p4.inner, true, true).unwrap();
    let trace = trace_opt.expect("trace populated");
    assert_eq!(trace.token_id, top_id);
    assert_eq!(trace.s4_role_choice, Some(2));
    assert_eq!(trace.s3_mode, "primary");
    assert_eq!(trace.s2_skip_hit_mask & 1, 1);

    let scores = sess.last_scores_q48();
    let w_bucket = p4.inner.s6.bucket_weights(trace.bucket).unwrap();
    let pick_opt = code_pick(&p4.inner.codes, &pref);
    let pick_succ = pick_opt.map(|(p, _, _)| pref[p + 1]);
    let mut brute_cache = CausalSessionCache::new();
    brute_cache.reset_from_tokens(&pref);
    let count_width = 2usize;
    let mut brute_counts = [0u32; 11];
    let mut brute_totals = [0u64; 11];
    let mut hasher = Sha256::new();
    let mut lookup_buf = pref.to_vec();
    lookup_buf.push(0);
    for target in 1..P4_VOCAB as u16 {
        *lookup_buf.last_mut().unwrap() = target;
        let kn = p4
            .inner
            .counts
            .lookup(
                &lookup_buf,
                pref.len(),
                &mut brute_counts[..count_width],
                &mut brute_totals[..count_width],
            )
            .unwrap();
        let mut col = [0u64; P4_EXPERTS];
        col[0] = kn;
        for j in 0..11 {
            col[1 + j] = if brute_totals[j] == 0 {
                0
            } else {
                ((u128::from(brute_counts[j]) << 48) / u128::from(brute_totals[j])) as u64
            };
        }
        col[12] = KN_BASE_MASS;
        col[13] = if pick_succ == Some(target) {
            P4_Q48_ONE
        } else {
            0
        };
        col[14] = p4.inner.s3.lookup_modulated_q48(pk_hit, target).unwrap().0;
        for fam in 0..3 {
            col[15 + fam] = p4
                .inner
                .s2_skip
                .lookup_modulated_q48(fam, skeys[fam], target)
                .unwrap();
        }
        let (_chs, cm) = brute_cache.peek_score_readonly(target);
        col[18..22].copy_from_slice(&cm);
        let exp_q48 = brier_mix_q48(w_bucket, &col).unwrap().clamp(1, P3_Q48_ONE);
        assert_eq!(
            scores[usize::from(target)],
            exp_q48,
            "P4 bit mismatch at token {target}"
        );
        hasher.update(exp_q48.to_le_bytes());
    }
    assert_eq!(hex::encode(hasher.finalize()), trace.mass_sha256);

    // Verify P4 handles a 2,048-token prompt (far above legacy trunk 192!) cleanly!
    let mut long_prompt = vec![0u16; P4_MAX_CONTEXT];
    for (i, slot) in long_prompt.iter_mut().enumerate().skip(1) {
        *slot = ((i % 7) + 2) as u16;
    }
    long_prompt[P4_MAX_CONTEXT - 2] = 2;
    long_prompt[P4_MAX_CONTEXT - 1] = 5;
    sess.sync_prompt(&long_prompt).unwrap();
    let (_id_2k, tr_2k) = sess.step(&p4.inner, true, true).unwrap();
    assert_eq!(tr_2k.unwrap().context_tokens, P4_MAX_CONTEXT);

    // Corruption refusal checks: header schema/magic spoofing, SHA-256 tamper & truncation
    let mut spoof = loaded_raw.clone();
    *spoof.last_mut().unwrap() ^= 0x01;
    let bad_path = dir.join("corrupt.d11");
    fs::write(&bad_path, &spoof).unwrap();
    let bad_loaded = load_client_artifact(&bad_path, false);
    if let Ok(ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    }) = bad_loaded
    {
        assert!(DiscreteCountModel::bind_embedded(
            model,
            &state_bytes,
            &expected_state_sha256,
            identity,
        )
        .is_err());
    } else if bad_loaded.is_ok() {
        panic!("expected tamper refusal");
    }

    let trunc_path = dir.join("trunc.d11");
    fs::write(&trunc_path, &loaded_raw[..loaded_raw.len() - 4]).unwrap();
    assert!(load_client_artifact(&trunc_path, false).is_err());

    let mut wrong_schema = loaded_raw.clone();
    let schema_pos = wrong_schema
        .windows(SCHEMA_P4.len())
        .position(|w| w == SCHEMA_P4.as_bytes())
        .unwrap();
    wrong_schema[schema_pos..schema_pos + SCHEMA_P4.len()]
        .copy_from_slice(b"uor-r4.lut-discrete-p3/1");
    let wrong_schema_path = dir.join("wrong_schema.d11");
    fs::write(&wrong_schema_path, &wrong_schema).unwrap();
    assert!(load_client_artifact(&wrong_schema_path, false).is_err());

    // End-to-end uor-chat CLI test on the packed p4.d11 file!
    let chat_out = Command::new(env!("CARGO_BIN_EXE_uor-chat"))
        .env("UOR_P1_TIMING", "1")
        .args([
            "--discrete",
            p4_path.to_str().unwrap(),
            "--prompt-ids",
            "0,2,5",
            "--max-tokens",
            "4",
            "--p1-trace",
        ])
        .output()
        .unwrap();
    assert!(
        chat_out.status.success(),
        "uor-chat failed: {}",
        String::from_utf8_lossy(&chat_out.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&chat_out.stdout).unwrap();
    assert_eq!(parsed["model_schema"], SCHEMA_P4);
    assert_eq!(
        parsed["generated_ids"][0].as_u64().unwrap(),
        u64::from(top_id)
    );
    assert_eq!(parsed["p4_steps"][0]["mass_sha256"], trace.mass_sha256);
    assert!(parsed["p4_steps"][0].get("s4_source").is_none());
    assert_eq!(parsed["p4_steps"].as_array().unwrap().len(), 4);
    let stderr = String::from_utf8_lossy(&chat_out.stderr);
    assert!(
        stderr.contains("uor-r4.p1-client-timing/1"),
        "missing timing: {stderr}"
    );

    check_lexical_client(&dir, &p4_path, &dc.model, &p4.inner, top_id, &pref);

    let gate_path = dir.join("source-gate.uors4g01");
    let digest = p4.inner.state_sha256().unwrap();
    assert_eq!(hex::encode(digest), p4.artifact_sha256);
    let gate = S4SourceGate::new(digest, [44; 32], [64; 20]).unwrap();
    fs::write(&gate_path, gate.encode_bytes()).unwrap();
    sess.sync_prompt(&pref).unwrap();
    let (disabled, disabled_trace) = sess
        .step_with_source(&p4.inner, true, true, Some((&gate, S4SourceMode::Disabled)))
        .unwrap();
    assert_eq!(disabled, top_id);
    assert_eq!(disabled_trace.unwrap().mass_sha256, trace.mass_sha256);
    let base = sess.last_scores_q48().to_vec();
    let (active, active_trace) = sess
        .step_with_source(&p4.inner, true, true, Some((&gate, S4SourceMode::Active)))
        .unwrap();
    let active_trace = active_trace.unwrap();
    let source = active_trace.s4_source.unwrap();
    assert_eq!(source.source_pos, Some(2));
    assert_eq!(source.source_token, Some(5));
    assert!(source.transfer_q48 > 0);
    let mut expected = base;
    expected[5] += source.transfer_q48;
    expected[usize::from(source.rival_token.unwrap())] -= source.transfer_q48;
    assert_eq!(sess.last_scores_q48(), expected);

    let gated_cli = |mode: &str| {
        Command::new(env!("CARGO_BIN_EXE_uor-chat"))
            .args([
                "--discrete",
                p4_path.to_str().unwrap(),
                "--prompt-ids",
                "0,2,5",
                "--max-tokens",
                "1",
                "--p1-trace",
                "--p4-source-gate",
                gate_path.to_str().unwrap(),
                "--p4-source-mode",
                mode,
            ])
            .output()
            .unwrap()
    };
    let chat = gated_cli("active");
    assert!(
        chat.status.success(),
        "{}",
        String::from_utf8_lossy(&chat.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&chat.stdout).unwrap();
    assert_eq!(parsed["generated_ids"][0], active);
    assert_eq!(
        parsed["p4_steps"][0]["mass_sha256"],
        active_trace.mass_sha256
    );
    assert_eq!(parsed["p4_steps"][0]["s4_source"]["source_pos"], 2);
    assert_eq!(
        parsed["p4_source_gate"]["p4_state_sha256"],
        p4.artifact_sha256
    );

    let wrong_gate = S4SourceGate::new([45; 32], [44; 32], [64; 20]).unwrap();
    fs::write(&gate_path, wrong_gate.encode_bytes()).unwrap();
    assert!(sess
        .step_with_source(
            &p4.inner,
            true,
            false,
            Some((&wrong_gate, S4SourceMode::Active))
        )
        .is_err());
    let chat = gated_cli("active");
    assert!(!chat.status.success());
    assert!(String::from_utf8_lossy(&chat.stderr).contains("P4 state SHA-256 mismatch"));

    // Refuse overwrite without leaving temp files behind
    assert!(write_p4_deployment(
        &trunk,
        &fit_json,
        &a1s_bytes,
        &counts_bytes,
        &compact_s3,
        &compact_s2,
        &s4_bytes,
        &s6_bytes,
        &p4_path,
    )
    .is_err());

    // Preserve previous assistant turns while opening the next response.
    let dialogue = [0, 5, 7, 1, 9, 7];
    sess.sync_prompt(&dialogue).unwrap();
    assert_eq!(sess.history(), &dialogue);
    assert!(sess.sync_prompt(&[0, 5, 7, 1]).is_err());
    assert!(sess.observe_generated(1).is_err());

    // BOS-less prompt and >2048 sliding window must match cold sync_prompt(history) bit-for-bit
    sess.sync_prompt(&[2, 5, 7]).unwrap();
    let (id_bosless, tr_bosless) = sess.step(&p4.inner, true, true).unwrap();
    let mut cold_bosless = P4Session::new();
    cold_bosless.sync_prompt(sess.history()).unwrap();
    let (id_cold_bosless, tr_cold_bosless) = cold_bosless.step(&p4.inner, true, true).unwrap();
    assert_eq!(id_bosless, id_cold_bosless);
    assert_eq!(
        tr_bosless.unwrap().mass_sha256,
        tr_cold_bosless.unwrap().mass_sha256
    );
    // EOS=1 rejected in observe_generated
    assert!(sess.observe_generated(1).is_err());

    sess.sync_prompt(&long_prompt).unwrap();
    sess.observe_generated(7).unwrap();
    sess.observe_generated(3).unwrap();
    let (id_slid, tr_slid) = sess.step(&p4.inner, true, true).unwrap();
    let mut cold_slid = P4Session::new();
    cold_slid.sync_prompt(sess.history()).unwrap();
    let (id_cold_slid, tr_cold_slid) = cold_slid.step(&p4.inner, true, true).unwrap();
    assert_eq!(id_slid, id_cold_slid);
    assert_eq!(
        tr_slid.unwrap().mass_sha256,
        tr_cold_slid.unwrap().mass_sha256
    );

    // Lawful stop/rep guard integration keeps PRLL floor >= 1 and promotes EOS when span exhausted
    let (stop_tok, _stop_tr, guard_tr) = sess
        .step_with_stop_guard(
            &p4.inner,
            true,
            false,
            Some((&[2, 5, 7], true, LawfulStopConfig::default())),
        )
        .unwrap();
    let gt = guard_tr.expect("stop guard trace");
    assert!(gt.promoted_eos);
    assert_eq!(stop_tok, 1);
    assert!(sess.last_scores_q48()[1..].iter().all(|&m| m >= 1));

    // Reject PickClassRule::Three (arm = "a1") in P4 container
    let a1_len = 3 * (2 * 16) * 5;
    let a1_fit_json = serde_json::to_vec(&json!({
        "schema": "uor-r4.d37.r-fit/1",
        "arm": "a1",
        "pick_classes": 3,
        "weight_format": "UORS5W01",
        "q32_weight_words": a1_len,
        "prepared": {
            "format": "uor-r4.d37.r-preparation/1",
            "c_sha256": hex::encode(c_sha),
            "r_sha256": hex::encode(r_sha),
            "pick_mode": "identity",
            "trunk_sha256": trunk_sha,
            "compact_sha256": compact_sha,
        }
    }))
    .unwrap();
    let a1_digest: [u8; 32] = Sha256::digest(&a1_fit_json).into();
    let mut a1_words = vec![0u64; a1_len];
    for chunk in a1_words.chunks_exact_mut(5) {
        chunk[0] = 1u64 << 32;
    }
    let a1_bytes = uor_r4_count_runtime::FrozenPickMixture::with_rule(
        1,
        a1_digest,
        a1_words,
        PickClassRule::Three,
    )
    .unwrap()
    .to_bytes();
    assert!(write_p4_deployment(
        &trunk,
        &a1_fit_json,
        &a1_bytes,
        &counts_bytes,
        &compact_s3,
        &compact_s2,
        &s4_bytes,
        &s6_bytes,
        &dir.join("reject_a1.d11"),
    )
    .is_err());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p4_adversarial_bounds_zero_count_oob_and_s4_pick_pointer_parity() {
    let c_sha = [0x66u8; 32];
    let token_h4 = [0u8; P4_VOCAB];

    // Out-of-vocab tokens never panic in public helpers
    let _ = golden_shell_pkey(&token_h4, &[9999, 5000]);
    let mut cache = CausalSessionCache::new();
    cache.observe_token(9999);
    assert_eq!(cache.peek_score_readonly(9999), (0, [P4_UNIFORM_MASS; 4]));

    // Zero count row rejected in both UORS3T02 -> UORS3C01 and UORS2S01 -> UORS2C01
    let zero_row = [(3u16, 0u32, 0i8)];
    let raw_s3_zero = build_synthetic_uors3t02(&token_h4, c_sha, &[(1, &zero_row)], &[]);
    assert!(encode_compact_s3_from_uors3t02(&raw_s3_zero, None, None, c_sha).is_err());

    let sk = skip_keys_for_prefix(&[0, 2, 5]);
    let raw_s2_zero = build_synthetic_uors2s01(c_sha, [&[(sk[0], &zero_row)], &[], &[]]);
    assert!(encode_compact_s2_skip_from_uors2s01(&raw_s2_zero, None, c_sha).is_err());

    // Crafted out-of-bounds rs + rl in UORS3T02 rejected cleanly (no panic)
    let ok_row = [(3u16, 5u32, 0i8)];
    let mut oob_s3 = build_synthetic_uors3t02(&token_h4, c_sha, &[(1, &ok_row)], &[]);
    let rs_off = 72 + P4_VOCAB + 12; // primary ctx[0].row_start
    oob_s3[rs_off..rs_off + 4].copy_from_slice(&99u32.to_le_bytes());
    assert!(encode_compact_s3_from_uors3t02(&oob_s3, None, None, c_sha).is_err());

    // Crafted out-of-bounds rs + rl in UORS2S01 (with recomputed trailing SHA) rejected cleanly
    let mut oob_s2 = build_synthetic_uors2s01(c_sha, [&[(sk[0], &ok_row)], &[], &[]]);
    let s2_rs_off = 8 + 4 + 32 + 1 + 4 + 4 + 8; // fam0 ctx[0].row_start
    oob_s2[s2_rs_off..s2_rs_off + 4].copy_from_slice(&99u32.to_le_bytes());
    let plen = oob_s2.len() - 32;
    let new_sha: [u8; 32] = Sha256::digest(&oob_s2[..plen]).into();
    oob_s2[plen..].copy_from_slice(&new_sha);
    assert!(encode_compact_s2_skip_from_uors2s01(&oob_s2, None, c_sha).is_err());

    // S4 negative-score abstention and lowest-index positive tie-break via pick_pointer
    let pk = s4_pair_key(2, 5).unwrap();
    let neg_entries = [
        S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 1,
            err_i8: -120,
        },
        S4PointerEntry {
            role_class: 2,
            span_offset: 0,
            count: 1,
            err_i8: -120,
        },
        S4PointerEntry {
            role_class: 3,
            span_offset: 0,
            count: 1,
            err_i8: -120,
        },
    ];
    let s4_neg = S4ConjunctionTable::from_sorted_rows(c_sha, &[(pk, &neg_entries[..])]).unwrap();
    assert_eq!(
        s4_neg.pick_pointer(
            &[pk],
            &[(1, 0), (2, 0), (3, 0)],
            uor_r4_count_runtime::p3_state::S4ControlMode::Active
        ),
        None
    );

    let tie_entries = [
        S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 10,
            err_i8: 20,
        },
        S4PointerEntry {
            role_class: 2,
            span_offset: 0,
            count: 10,
            err_i8: 20,
        },
    ];
    let s4_tie = S4ConjunctionTable::from_sorted_rows(c_sha, &[(pk, &tie_entries[..])]).unwrap();
    assert_eq!(
        s4_tie
            .pick_pointer(
                &[pk],
                &[(1, 0), (2, 0), (3, 0)],
                uor_r4_count_runtime::p3_state::S4ControlMode::Active
            )
            .map(|(idx, _)| idx),
        Some(0)
    );
}
