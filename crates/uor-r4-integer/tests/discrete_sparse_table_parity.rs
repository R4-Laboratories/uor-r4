use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use uor_r4_integer::discrete_d11::{
    IntegerDiscreteModel, DISCRETE_S2_EXPERT_SCHEMA, DISCRETE_S2_SCHEMA, DISCRETE_SCHEMA,
    DISCRETE_TRUNK_SCHEMA, MAGIC, S2_SECTION, SNAPSHOT_SCHEMA,
};

const ALIGN: usize = 64;
const SPARSE_TABLE_NAMES: [&str; 12] = [
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
];

const ROOTS: [[i8; 8]; 16] = [
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

const CODEX36_TRIPLES: usize = 7_164_583;
const CODEX36_DISTINCT_KEYS: usize = 4_297_201;

fn expected_stages(context: usize) -> usize {
    let mut stages = 1usize;
    let mut span = 3usize;
    while span <= context {
        stages += 1;
        span *= 3;
    }
    stages
}

fn token_4block_roots(tok: usize) -> [u8; 4] {
    let r0 = (tok & 15) as u8;
    let r1 = ((tok >> 4) & 15) as u8;
    let r2 = ((tok >> 8) & 15) as u8;
    let r3 = if tok < 16 { r0 } else { r0 ^ r1 ^ r2 };
    [r0, r1, r2, r3]
}

fn pack_4block_roots(roots: [u8; 4]) -> u16 {
    u16::from(roots[0] & 0xF)
        | (u16::from(roots[1] & 0xF) << 4)
        | (u16::from(roots[2] & 0xF) << 8)
        | (u16::from(roots[3] & 0xF) << 12)
}

fn push_section_bytes(blob: &mut Vec<u8>, bytes: &[u8]) -> Value {
    let pad = (ALIGN - blob.len() % ALIGN) % ALIGN;
    blob.resize(blob.len() + pad, 0);
    let offset = blob.len() as u64;
    blob.extend_from_slice(bytes);
    json!({"offset": offset, "bytes": bytes.len()})
}

fn push_u32_section(blob: &mut Vec<u8>, name: &str, words: &[u32]) -> Value {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let span = push_section_bytes(blob, &bytes);
    json!({"name": name, "kind": "u32", "len": words.len(), "span": span})
}

fn build_trunk_uorlut01(
    vocab: usize,
    context: usize,
    step_candidates: &[u32; 64],
    norm_candidates: &[u32; 4],
    unigram_words: &[u32],
    sparse_sections: [&[u32]; 12],
) -> Vec<u8> {
    assert_eq!(unigram_words.len(), 1 + vocab);
    let stages = expected_stages(context);
    let mut table = Vec::with_capacity(vocab * 32);
    let mut cells = Vec::with_capacity(vocab);
    for tok in 0..vocab {
        let roots = token_4block_roots(tok);
        for &r in &roots {
            for d in ROOTS[usize::from(r)] {
                table.push(i32::from(d));
            }
        }
        cells.push((u32::from(roots[0]) << 4) + u32::from(roots[3]));
    }
    let routes = vec![0u32; 4 * stages];
    let gates = vec![0i32; 4 * stages];

    let mut blob = Vec::new();
    let mut tables = Vec::new();
    let table_bytes: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
    let table_span = push_section_bytes(&mut blob, &table_bytes);
    tables.push(
        json!({"name": "discrete.table", "kind": "i32", "len": table.len(), "span": table_span}),
    );
    tables.push(push_u32_section(&mut blob, "discrete.cells", &cells));
    tables.push(push_u32_section(&mut blob, "discrete.routes", &routes));
    let gates_bytes: Vec<u8> = gates.iter().flat_map(|x| x.to_le_bytes()).collect();
    let gates_span = push_section_bytes(&mut blob, &gates_bytes);
    tables.push(
        json!({"name": "discrete.gates", "kind": "i32", "len": gates.len(), "span": gates_span}),
    );

    tables.push(push_u32_section(
        &mut blob,
        "discrete.trunk.step_candidates",
        step_candidates,
    ));
    tables.push(push_u32_section(
        &mut blob,
        "discrete.trunk.norm_candidates",
        norm_candidates,
    ));
    tables.push(push_u32_section(
        &mut blob,
        "discrete.trunk.unigram",
        unigram_words,
    ));

    for (idx, name) in SPARSE_TABLE_NAMES.iter().enumerate() {
        tables.push(push_u32_section(&mut blob, name, sparse_sections[idx]));
    }

    build_container_from_header_and_blob(vocab, context, stages, tables, &blob)
}

fn build_container_from_header_and_blob(
    vocab: usize,
    context: usize,
    stages: usize,
    tables: Vec<Value>,
    blob: &[u8],
) -> Vec<u8> {
    let header = json!({
        "schema": DISCRETE_TRUNK_SCHEMA,
        "shape": {
            "vocabulary": vocab,
            "blocks": 4,
            "context": context,
            "kept_cells": 3,
            "stages": stages,
            "read_mode": "pick",
            "read_kept": 0,
            "read_sweeps": 0,
            "read_position": 3,
            "objective_policy": "nearest_address_margin_v1",
            "read_initialization": "zero_gates_v1",
            "seed": 7,
            "epoch": 1,
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
    });
    let json = serde_json::to_vec(&header).unwrap();
    let mut out = Vec::with_capacity(16 + json.len() + ALIGN + blob.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    let pad = (ALIGN - out.len() % ALIGN) % ALIGN;
    out.resize(out.len() + pad, 0);
    out.extend_from_slice(blob);
    out
}

#[derive(Clone, Debug, Default)]
struct BTreeSparseOracle {
    entries: BTreeMap<u32, BTreeMap<u16, u32>>,
}

impl BTreeSparseOracle {
    fn from_u32_words(words: &[u32], vocab: usize) -> Self {
        assert_eq!(words.len() % 3, 0);
        let mut entries: BTreeMap<u32, BTreeMap<u16, u32>> = BTreeMap::new();
        for chunk in words.chunks_exact(3) {
            let key = chunk[0];
            let tok = u16::try_from(chunk[1]).unwrap();
            let cnt = chunk[2];
            assert!(usize::from(tok) < vocab && cnt > 0);
            let prev = entries.entry(key).or_default().insert(tok, cnt);
            assert!(prev.is_none());
        }
        Self { entries }
    }

    fn get(&self, key: u32, tok: u16) -> u32 {
        self.entries
            .get(&key)
            .and_then(|m| m.get(&tok).copied())
            .unwrap_or(0)
    }

    fn keys_for(&self, key: u32) -> Vec<u16> {
        self.entries
            .get(&key)
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default()
    }
}

struct TrunkBTreeOracle {
    unigram_best_global: u16,
    unigram_counts: Vec<u32>,
    tables: [BTreeSparseOracle; 12],
}

impl TrunkBTreeOracle {
    fn from_sections(unigram_words: &[u32], sparse_sections: [&[u32]; 12]) -> Self {
        let vocab = unigram_words.len() - 1;
        let tables =
            std::array::from_fn(|i| BTreeSparseOracle::from_u32_words(sparse_sections[i], vocab));
        Self {
            unigram_best_global: unigram_words[0] as u16,
            unigram_counts: unigram_words[1..].to_vec(),
            tables,
        }
    }

    fn level3_score(
        &self,
        query_token: u16,
        read_code: u16,
        rec_code: u16,
        pointer_succ: Option<u16>,
        tok: u16,
    ) -> u64 {
        let q_key = u32::from(query_token);
        let r_key = u32::from(read_code);
        let qr_key = (q_key << 16) | r_key;
        let qc_key = (q_key << 16) | u32::from(rec_code);
        let rc_key = (r_key << 16) | u32::from(rec_code);

        let c_r = u64::from(self.tables[0].get(r_key, tok)); // read_table
        let c_rr = u64::from(self.tables[2].get(rc_key, tok)); // read_rec_table
        let c_q = u64::from(self.tables[7].get(q_key, tok)); // query_table
        let c_qr = u64::from(self.tables[8].get(qr_key, tok)); // query_read_table
        let c_qc = u64::from(self.tables[9].get(qc_key, tok)); // query_rec_table
        let ptr_boost = if pointer_succ == Some(tok) {
            1u64 << 8
        } else {
            0
        };
        (c_qr << 6) + (c_qc << 5) + (c_rr << 3) + (c_r << 1) + c_q + ptr_boost
    }

    fn select_token(
        &self,
        query_token: u16,
        read_code: u16,
        rec_code: u16,
        pointer_succ: Option<u16>,
    ) -> u16 {
        let q_key = u32::from(query_token);
        let r_key = u32::from(read_code);
        let c_key = u32::from(rec_code);
        let mut best_tok = self.unigram_best_global;
        let mut best_s = 0u64;
        let mut best_u = 0u32;

        let mut consider = |tok: u16| {
            let s = self.level3_score(query_token, read_code, rec_code, pointer_succ, tok);
            let u = self.unigram_counts[usize::from(tok)];
            if s > best_s
                || (s == best_s && s > 0 && u > best_u)
                || (s == best_s && s > 0 && u == best_u && tok < best_tok)
            {
                best_s = s;
                best_u = u;
                best_tok = tok;
            }
        };

        for tok in self.tables[7].keys_for(q_key) {
            consider(tok);
        }
        for tok in self.tables[0].keys_for(r_key) {
            consider(tok);
        }
        for tok in self.tables[1].keys_for(c_key) {
            consider(tok);
        }
        if let Some(ptr_tok) = pointer_succ {
            consider(ptr_tok);
        }
        best_tok
    }
}

fn oracle_pointer_succ(prefix: &[u32]) -> Option<u16> {
    if prefix.len() < 2 {
        return None;
    }
    let query_roots = token_4block_roots(prefix[prefix.len() - 1] as usize);
    let mut matched = None;
    for p in 0..(prefix.len() - 1) {
        if token_4block_roots(prefix[p] as usize) == query_roots {
            matched = Some(prefix[p + 1] as u16);
        }
    }
    matched
}

/// Computes `read_code` and `rec_code` for the identity-route/identity-step fixture
/// where `routes = 0`, `gates = 0`, `step_candidates = 0 (IDENTITY)`, `norm_candidates = 0`.
fn identity_fixture_codes(prefix: &[u32]) -> (u16, u16) {
    let query_roots = token_4block_roots(prefix[prefix.len() - 1] as usize);
    let read_code = pack_4block_roots(query_roots);
    // With step = 0 (IDENTITY) and norm = 0 (IDENTITY), prev/mid/start blocks add linearly,
    // wait: use snap against ROOTS on the 4-block integer sum!
    let last_pos = prefix.len() - 1;
    let prev_pos = last_pos.saturating_sub(1);
    let mid_pos = last_pos / 2;
    let mut out_roots = [0u8; 4];
    for (b, dst) in out_roots.iter_mut().enumerate() {
        let q = ROOTS[usize::from(token_4block_roots(prefix[last_pos] as usize)[b])];
        let p = ROOTS[usize::from(token_4block_roots(prefix[prev_pos] as usize)[b])];
        let m = ROOTS[usize::from(token_4block_roots(prefix[mid_pos] as usize)[b])];
        let s = ROOTS[usize::from(token_4block_roots(prefix[0] as usize)[b])];
        let mut best_r = 0u8;
        let mut best_d = u32::MAX;
        let mut best_h = u32::MAX;
        for (r, root) in ROOTS.iter().enumerate() {
            let mut d = 0u32;
            let mut h = 0u32;
            for dim in 0..8 {
                let sum =
                    i32::from(q[dim]) + i32::from(p[dim]) + i32::from(m[dim]) + i32::from(s[dim]);
                let target = 4 * i32::from(root[dim]);
                d += (sum - target).unsigned_abs();
                h += u32::from(sum != target);
            }
            if d < best_d || (d == best_d && h < best_h) {
                best_d = d;
                best_h = h;
                best_r = r as u8;
            }
        }
        *dst = best_r;
    }
    (read_code, pack_4block_roots(out_roots))
}

fn encode_triples(triples: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut out = Vec::with_capacity(triples.len() * 3);
    for &(k, t, c) in triples {
        out.push(k);
        out.push(t);
        out.push(c);
    }
    out
}

fn fixture_sparse_data() -> (Vec<u32>, [Vec<u32>; 12]) {
    let vocab = 16usize;
    let mut unigram = vec![14u32]; // best_global = 14 on missing keys
    for t in 0..vocab {
        unigram.push(100 - (t as u32) * 3);
    }
    // Cause explicit tie on unigram counts for tok 6 and tok 9 to exercise tok_asc tie-break:
    unigram[1 + 6] = 55;
    unigram[1 + 9] = 55;
    // Higher unigram for tok 11 than tok 5 to exercise unigram tie-break on equal score:
    unigram[1 + 5] = 40;
    unigram[1 + 11] = 75;

    let read0 = u32::from(pack_4block_roots(token_4block_roots(0)));
    let rec0 = u32::from(identity_fixture_codes(&[0]).1);
    let q0_r0 = read0;
    let q1_r1 = (1u32 << 16) | u32::from(pack_4block_roots(token_4block_roots(1)));
    let q2_c2 = (2u32 << 16) | u32::from(identity_fixture_codes(&[2]).1);
    let r3_c3 = (u32::from(pack_4block_roots(token_4block_roots(3))) << 16)
        | u32::from(identity_fixture_codes(&[3]).1);

    let read_table = encode_triples(&[
        (u32::from(pack_4block_roots(token_4block_roots(0))), 4, 2),
        (u32::from(pack_4block_roots(token_4block_roots(3))), 5, 3),
        (u32::from(pack_4block_roots(token_4block_roots(5))), 6, 4),
    ]);
    let rec_table = encode_triples(&[(rec0, 7, 1)]);
    let read_rec_table = encode_triples(&[(r3_c3, 12, 8)]);
    let query_table = encode_triples(&[
        (0, 2, 5),
        (1, 3, 2),
        (2, 8, 1),
        (4, 5, 10),
        (4, 11, 10),
        (5, 9, 8),
        (6, 13, 3),
        (10, 1, 7),
    ]);
    let query_read_table = encode_triples(&[(q0_r0, 4, 3), (q1_r1, 10, 5)]);
    let query_rec_table = encode_triples(&[(q2_c2, 15, 6)]);

    // Also populate the remaining 6 auxiliary sparse tables to exercise all 12 tables end-to-end.
    let aux = encode_triples(&[(1, 2, 3), (4, 5, 6)]);
    let sections = [
        read_table,
        rec_table,
        read_rec_table,
        aux.clone(),
        aux.clone(),
        aux.clone(),
        aux.clone(),
        query_table,
        query_read_table,
        query_rec_table,
        aux.clone(),
        aux,
    ];
    (unigram, sections)
}

#[test]
fn trunk_sparse_loader_matches_btreemap_oracle_on_sorted_and_shuffled_triples() {
    let vocab = 16usize;
    let context = 4usize;
    let steps = [0u32; 64]; // IDENTITY step_candidates
    let norms = [0u32; 4]; // IDENTITY norm_candidates
    let (unigram, sorted_sections) = fixture_sparse_data();

    let sorted_refs: [&[u32]; 12] = std::array::from_fn(|i| sorted_sections[i].as_slice());
    let oracle = TrunkBTreeOracle::from_sections(&unigram, sorted_refs);
    let bytes_sorted = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, sorted_refs);

    // Shuffle triple order inside each section while keeping identical multisets of triples.
    let mut shuffled_sections = sorted_sections.clone();
    for (idx, sec) in shuffled_sections.iter_mut().enumerate() {
        let count = sec.len() / 3;
        if count >= 2 {
            let mut triples: Vec<[u32; 3]> =
                sec.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            triples.reverse();
            if count >= 3 {
                triples.swap(0, idx % count);
            }
            sec.clear();
            for [k, t, c] in triples {
                sec.extend_from_slice(&[k, t, c]);
            }
        }
    }
    let shuffled_refs: [&[u32]; 12] = std::array::from_fn(|i| shuffled_sections[i].as_slice());
    let bytes_shuffled =
        build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, shuffled_refs);

    let model_sorted = IntegerDiscreteModel::from_bytes(&bytes_sorted).unwrap();
    let model_shuffled = IntegerDiscreteModel::from_bytes(&bytes_shuffled).unwrap();

    let probes: Vec<&[u32]> = vec![
        &[0],
        &[1],
        &[2],
        &[3],
        &[4], // unigram tie-break between tok 5 (u=40) and tok 11 (u=75)
        &[5], // equal unigram & score tie-break between tok 6 and tok 9 (6 < 9)
        &[6],
        &[7],  // completely missing from all sparse tables -> fallback unigram_best_global = 14
        &[10], // pointer absent
        &[0, 1, 2, 0], // triggers pointer_succ == Some(1) on repeat of tok 0
        &[2, 9, 2], // triggers pointer_succ == Some(9) not in sparse candidate sets
        &[3, 7, 1, 3], // read_rec + pointer interaction
        &[4, 8, 4], // rec tie + pointer boost
        &[5, 9, 5], // tok tie + pointer boost
    ];

    for prefix in probes {
        let (read_code, rec_code) = identity_fixture_codes(prefix);
        let ptr_succ = oracle_pointer_succ(prefix);
        let q_tok = *prefix.last().unwrap() as u16;
        let expected_tok = u32::from(oracle.select_token(q_tok, read_code, rec_code, ptr_succ));
        let expected_vec: Vec<i32> = token_4block_roots(expected_tok as usize)
            .iter()
            .flat_map(|&r| ROOTS[usize::from(r)].iter().map(|&v| i32::from(v)))
            .collect();

        let eval_s = model_sorted.evaluate_prefix(prefix).unwrap();
        let eval_u = model_shuffled.evaluate_prefix(prefix).unwrap();
        assert_eq!(eval_s, eval_u, "sorted vs shuffled mismatch on {prefix:?}");
        assert_eq!(
            eval_s.prediction, expected_tok,
            "prediction mismatch on {prefix:?}"
        );
        assert_eq!(
            eval_s.read_vector, expected_vec,
            "read_vector mismatch on {prefix:?}"
        );

        let traced_s = model_sorted.evaluate_prefix_traced(prefix).unwrap();
        let traced_u = model_shuffled.evaluate_prefix_traced(prefix).unwrap();
        assert_eq!(traced_s, traced_u, "traced replay mismatch on {prefix:?}");
    }

    // Non-vacuous guards verifying active unigram tie-break, token-ID overwrite, and fallback.
    assert_eq!(model_sorted.evaluate_prefix(&[4]).unwrap().prediction, 11);
    assert_eq!(model_sorted.evaluate_prefix(&[5]).unwrap().prediction, 6);
    assert_eq!(model_sorted.evaluate_prefix(&[7]).unwrap().prediction, 14);

    // Multi-step 4-token autoregressive rollout parity against BTreeMap oracle.
    let mut seq_model = vec![0u32];
    let mut seq_oracle = vec![0u32];
    for _ in 0..3 {
        let m_next = model_sorted.evaluate_prefix(&seq_model).unwrap().prediction;
        let (rc, rec) = identity_fixture_codes(&seq_oracle);
        let o_next = u32::from(oracle.select_token(
            *seq_oracle.last().unwrap() as u16,
            rc,
            rec,
            oracle_pointer_succ(&seq_oracle),
        ));
        assert_eq!(m_next, o_next);
        seq_model.push(m_next);
        seq_oracle.push(o_next);
    }
    assert_eq!(seq_model, seq_oracle);
}

#[test]
fn trunk_sparse_loader_rejects_planted_faults_across_all_12_sections() {
    let vocab = 16usize;
    let context = 4usize;
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let (unigram, sorted_sections) = fixture_sparse_data();

    for target_idx in 0..12 {
        // 1) zero count fault
        let mut zero_cnt = sorted_sections.clone();
        zero_cnt[target_idx] = vec![1, 2, 0];
        let refs: [&[u32]; 12] = std::array::from_fn(|i| zero_cnt[i].as_slice());
        let bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, refs);
        assert!(
            IntegerDiscreteModel::from_bytes(&bytes).is_err(),
            "section {target_idx} accepted zero count"
        );

        // 2) token >= vocabulary fault
        let mut bad_tok = sorted_sections.clone();
        bad_tok[target_idx] = vec![1, 16, 2];
        let refs: [&[u32]; 12] = std::array::from_fn(|i| bad_tok[i].as_slice());
        let bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, refs);
        assert!(
            IntegerDiscreteModel::from_bytes(&bytes).is_err(),
            "section {target_idx} accepted tok >= vocab"
        );

        // 3) adjacent duplicate (key, token)
        let mut adj_dup = sorted_sections.clone();
        adj_dup[target_idx] = vec![5, 3, 1, 5, 3, 2];
        let refs: [&[u32]; 12] = std::array::from_fn(|i| adj_dup[i].as_slice());
        let bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, refs);
        assert!(
            IntegerDiscreteModel::from_bytes(&bytes).is_err(),
            "section {target_idx} accepted adjacent duplicate"
        );

        // 4) non-adjacent duplicate (key, token)
        let mut nonadj_dup = sorted_sections.clone();
        nonadj_dup[target_idx] = vec![9, 2, 1, 1, 0, 1, 9, 2, 4];
        let refs: [&[u32]; 12] = std::array::from_fn(|i| nonadj_dup[i].as_slice());
        let bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, refs);
        assert!(
            IntegerDiscreteModel::from_bytes(&bytes).is_err(),
            "section {target_idx} accepted non-adjacent duplicate"
        );

        // 5) non-multiple-of-3 word count
        let mut bad_len = sorted_sections.clone();
        bad_len[target_idx] = vec![1, 2, 3, 4];
        let refs: [&[u32]; 12] = std::array::from_fn(|i| bad_len[i].as_slice());
        let bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, refs);
        assert!(
            IntegerDiscreteModel::from_bytes(&bytes).is_err(),
            "section {target_idx} accepted word len % 3 != 0"
        );
    }
}

#[test]
fn trunk_sparse_loader_planted_weight_flip_changes_emitted_token_deterministically() {
    let vocab = 16usize;
    let context = 4usize;
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let (unigram, mut sections) = fixture_sparse_data();

    let base_refs: [&[u32]; 12] = std::array::from_fn(|i| sections[i].as_slice());
    let base_bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, base_refs);
    let base_model = IntegerDiscreteModel::from_bytes(&base_bytes).unwrap();
    let before = base_model.evaluate_prefix(&[0]).unwrap();
    assert_eq!(before.prediction, 4); // query_read_table (4 with c_qr=3 => 194) beats query_table

    // Planted count bump on tok 2 in query_table (`5 -> 250`): score 250 > 194 flips tok 4 -> 2.
    sections[7][2] = 250;
    let flipped_refs: [&[u32]; 12] = std::array::from_fn(|i| sections[i].as_slice());
    let oracle = TrunkBTreeOracle::from_sections(&unigram, flipped_refs);
    let flipped_bytes =
        build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, flipped_refs);
    let flipped_model = IntegerDiscreteModel::from_bytes(&flipped_bytes).unwrap();
    let after = flipped_model.evaluate_prefix(&[0]).unwrap();

    let (rc, rec) = identity_fixture_codes(&[0]);
    let oracle_after = u32::from(oracle.select_token(0, rc, rec, None));
    assert_ne!(after.prediction, before.prediction);
    assert_eq!(after.prediction, 2);
    assert_eq!(after.prediction, oracle_after);
}

#[derive(Deserialize)]
struct HeaderOnly {
    tables: Vec<HeaderTable>,
}

#[derive(Deserialize)]
struct HeaderTable {
    name: String,
    span: HeaderSpan,
}

#[derive(Deserialize)]
struct HeaderSpan {
    offset: u64,
    bytes: u64,
}

fn generate_codex36_scale_fixture_file(path: &Path) -> (usize, usize, usize, String) {
    let vocab = 4096usize;
    let context = 192usize;
    let stages = expected_stages(context);
    let mut table = Vec::with_capacity(vocab * 32);
    let mut cells = Vec::with_capacity(vocab);
    for tok in 0..vocab {
        let roots = token_4block_roots(tok);
        for &r in &roots {
            for d in ROOTS[usize::from(r)] {
                table.push(i32::from(d));
            }
        }
        cells.push((u32::from(roots[0]) << 4) + u32::from(roots[3]));
    }
    let routes = vec![0u32; 4 * stages];
    let gates = vec![0i32; 4 * stages];
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let mut unigram = Vec::with_capacity(1 + vocab);
    unigram.push(0u32);
    for t in 0..vocab {
        unigram.push((vocab - t) as u32);
    }

    let extra_triples = CODEX36_TRIPLES - CODEX36_DISTINCT_KEYS;
    let base_keys = CODEX36_DISTINCT_KEYS / 12;
    let rem_keys = CODEX36_DISTINCT_KEYS % 12;
    let base_extra = extra_triples / 12;
    let rem_extra = extra_triples % 12;

    let mut total_keys = 0usize;
    let mut total_triples = 0usize;
    let mut sparse_section_specs = Vec::with_capacity(12);
    let mut blob = Vec::new();
    let mut tables = Vec::new();
    let table_bytes: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
    let table_span = push_section_bytes(&mut blob, &table_bytes);
    tables.push(
        json!({"name": "discrete.table", "kind": "i32", "len": table.len(), "span": table_span}),
    );
    tables.push(push_u32_section(&mut blob, "discrete.cells", &cells));
    tables.push(push_u32_section(&mut blob, "discrete.routes", &routes));
    let gates_bytes: Vec<u8> = gates.iter().flat_map(|x| x.to_le_bytes()).collect();
    let gates_span = push_section_bytes(&mut blob, &gates_bytes);
    tables.push(
        json!({"name": "discrete.gates", "kind": "i32", "len": gates.len(), "span": gates_span}),
    );
    tables.push(push_u32_section(
        &mut blob,
        "discrete.trunk.step_candidates",
        &steps,
    ));
    tables.push(push_u32_section(
        &mut blob,
        "discrete.trunk.norm_candidates",
        &norms,
    ));
    tables.push(push_u32_section(
        &mut blob,
        "discrete.trunk.unigram",
        &unigram,
    ));

    // Pre-compute section sizes and write blob sequentially via buffered file I/O to avoid double-copy.
    let pre_blob_len = blob.len();
    let mut cursor = pre_blob_len;
    for (idx, name) in SPARSE_TABLE_NAMES.iter().enumerate() {
        let keys_s = base_keys + usize::from(idx < rem_keys);
        let extra_s = base_extra + usize::from(idx < rem_extra);
        let triples_s = keys_s + extra_s;
        total_keys += keys_s;
        total_triples += triples_s;
        let pad = (ALIGN - cursor % ALIGN) % ALIGN;
        cursor += pad;
        let bytes_len = triples_s * 12;
        tables.push(json!({
            "name": name,
            "kind": "u32",
            "len": triples_s * 3,
            "span": {"offset": cursor as u64, "bytes": bytes_len as u64},
        }));
        sparse_section_specs.push((pad, keys_s, extra_s));
        cursor += bytes_len;
    }
    assert_eq!(total_keys, CODEX36_DISTINCT_KEYS);
    assert_eq!(total_triples, CODEX36_TRIPLES);

    let header_bytes_only =
        build_container_from_header_and_blob(vocab, context, stages, tables, &blob);
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::with_capacity(1 << 20, file);
    writer.write_all(&header_bytes_only).unwrap();
    let zeros = [0u8; ALIGN];
    for (idx, &(pad, keys_s, extra_s)) in sparse_section_specs.iter().enumerate() {
        if pad > 0 {
            writer.write_all(&zeros[..pad]).unwrap();
        }
        for k in 0..keys_s {
            let key = k as u32;
            let tok0 = ((k + idx * 17) % (vocab - 1)) as u32;
            let cnt0 = 1 + ((k as u32) % 31);
            writer.write_all(&key.to_le_bytes()).unwrap();
            writer.write_all(&tok0.to_le_bytes()).unwrap();
            writer.write_all(&cnt0.to_le_bytes()).unwrap();
            if k < extra_s {
                let tok1 = tok0 + 1;
                let cnt1 = 1 + (((k + idx) as u32) % 19);
                writer.write_all(&key.to_le_bytes()).unwrap();
                writer.write_all(&tok1.to_le_bytes()).unwrap();
                writer.write_all(&cnt1.to_le_bytes()).unwrap();
            }
        }
    }
    writer.flush().unwrap();
    let file_bytes = fs::read(path).unwrap();
    let digest = format!("{:x}", Sha256::digest(&file_bytes));
    (file_bytes.len(), total_triples, total_keys, digest)
}

#[test]
#[ignore]
fn bench_generate_codex36_scale_fixture() {
    let path = PathBuf::from(
        std::env::var("UOR_BENCH_TRUNK_PATH")
            .expect("set UOR_BENCH_TRUNK_PATH to write the synthetic codex-36 fixture"),
    );
    let (file_bytes, triples, keys, sha) = generate_codex36_scale_fixture_file(&path);
    println!(
        "GENERATED path={} file_bytes={} triples={} distinct_keys={} sparse_bytes={} sha256={}",
        path.display(),
        file_bytes,
        triples,
        keys,
        triples * 12,
        sha
    );
}

fn bench_probe_prefixes() -> Vec<[u32; 4]> {
    (0..10_000u32)
        .map(|i| {
            let a = (i * 13) % 4096;
            let b = (i * 37 + 7) % 4096;
            let c = (i * 73 + 19) % 4096;
            let d = (i * 131 + 29) % 4096;
            [a, b, c, d]
        })
        .collect()
}

#[test]
#[ignore]
fn bench_codex36_scale_contiguous_trunk_loader() {
    let path = PathBuf::from(
        std::env::var("UOR_BENCH_TRUNK_PATH")
            .expect("set UOR_BENCH_TRUNK_PATH to benchmark contiguous loader"),
    );
    let load_start = Instant::now();
    let model = IntegerDiscreteModel::load(&path).unwrap();
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    assert!(model.is_trunk_d11());
    let probes = bench_probe_prefixes();
    // Warmup
    for p in &probes[..256] {
        let _ = model.evaluate_prefix(p).unwrap();
    }
    let eval_start = Instant::now();
    let mut hash_fold = 0u64;
    for (idx, p) in probes.iter().enumerate() {
        let eval = model.evaluate_prefix(p).unwrap();
        hash_fold ^= u64::from(eval.prediction).wrapping_mul((idx as u64) + 1);
    }
    let eval_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
    let toks_per_sec = (probes.len() as f64) / (eval_ms / 1000.0);
    println!(
        "CONTIGUOUS_BENCH artifact_sha256={} load_ms={:.3} eval_10k_ms={:.3} toks_per_sec={:.2} probe_fold={:#018x}",
        model.artifact_sha256(),
        load_ms,
        eval_ms,
        toks_per_sec,
        hash_fold
    );
}

#[test]
#[ignore]
fn bench_codex36_scale_btreemap_baseline() {
    let path = PathBuf::from(
        std::env::var("UOR_BENCH_TRUNK_PATH")
            .expect("set UOR_BENCH_TRUNK_PATH to benchmark legacy BTreeMap loader"),
    );
    let load_start = Instant::now();
    let bytes = fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let header_end = 16 + header_len;
    let header: HeaderOnly = serde_json::from_slice(&bytes[16..header_end]).unwrap();
    let pad = (ALIGN - header_end % ALIGN) % ALIGN;
    let data = &bytes[header_end + pad..];

    let read_u32 = |name: &str| -> Vec<u32> {
        let entry = header.tables.iter().find(|t| t.name == name).unwrap();
        let start = entry.span.offset as usize;
        let end = start + entry.span.bytes as usize;
        data[start..end]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };

    let unigram_words = read_u32("discrete.trunk.unigram");
    let unigram_best = unigram_words[0] as u16;
    let unigram_counts = unigram_words[1..].to_vec();
    let vocab = unigram_counts.len();

    let mut tables: Vec<BTreeSparseOracle> = Vec::with_capacity(12);
    for name in SPARSE_TABLE_NAMES {
        let words = read_u32(name);
        tables.push(BTreeSparseOracle::from_u32_words(&words, vocab));
    }
    drop(bytes);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let oracle = TrunkBTreeOracle {
        unigram_best_global: unigram_best,
        unigram_counts,
        tables: tables.try_into().unwrap(),
    };
    let probes = bench_probe_prefixes();
    let eval_start = Instant::now();
    let mut hash_fold = 0u64;
    for (idx, p) in probes.iter().enumerate() {
        let (read_code, rec_code) = identity_fixture_codes(p);
        let ptr_succ = oracle_pointer_succ(p);
        let pred = oracle.select_token(p[3] as u16, read_code, rec_code, ptr_succ);
        hash_fold ^= u64::from(pred).wrapping_mul((idx as u64) + 1);
    }
    let eval_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "BTREEMAP_BASELINE load_ms={:.3} eval_10k_ms={:.3} probe_fold={:#018x}",
        load_ms, eval_ms, hash_fold
    );
}

fn assert_models_equivalent(a: &IntegerDiscreteModel, b: &IntegerDiscreteModel, vocab: usize) {
    assert_eq!(a.artifact_schema(), b.artifact_schema());
    assert_eq!(a.artifact_sha256(), b.artifact_sha256());
    assert_eq!(a.tokenizer_sha256(), b.tokenizer_sha256());
    assert_eq!(a.shape(), b.shape());
    assert_eq!(
        a.frozen_s2().map(|s| s.triples().collect::<Vec<_>>()),
        b.frozen_s2().map(|s| s.triples().collect::<Vec<_>>()),
    );
    for t in 0..vocab.min(8) {
        let t = t as u32;
        assert_eq!(
            a.evaluate_prefix(&[t]).unwrap(),
            b.evaluate_prefix(&[t]).unwrap()
        );
        assert_eq!(
            a.evaluate_prefix(&[0, t]).unwrap(),
            b.evaluate_prefix(&[0, t]).unwrap()
        );
    }
}

#[test]
fn streaming_from_reader_and_load_match_from_bytes_across_container_variants() {
    let vocab = 16usize;
    let context = 4usize;
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let (unigram, sorted_sections) = fixture_sparse_data();
    let sorted_refs: [&[u32]; 12] = std::array::from_fn(|i| sorted_sections[i].as_slice());

    // 1) Sorted trunk UORLUT01
    let trunk_bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, sorted_refs);
    let m_bytes = IntegerDiscreteModel::from_bytes(&trunk_bytes).unwrap();
    let m_reader = IntegerDiscreteModel::from_reader(std::io::Cursor::new(&trunk_bytes)).unwrap();
    assert_models_equivalent(&m_reader, &m_bytes, vocab);

    let dir = std::env::temp_dir().join(format!("uor-stream-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let trunk_path = dir.join("trunk.d11");
    fs::write(&trunk_path, &trunk_bytes).unwrap();
    let m_disk = IntegerDiscreteModel::load(&trunk_path).unwrap();
    assert_models_equivalent(&m_disk, &m_bytes, vocab);

    // 2) Shuffled trunk UORLUT01
    let mut shuffled_sections = sorted_sections.clone();
    for (idx, sec) in shuffled_sections.iter_mut().enumerate() {
        let count = sec.len() / 3;
        if count >= 2 {
            let mut triples: Vec<[u32; 3]> =
                sec.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            triples.reverse();
            if count >= 3 {
                triples.swap(0, idx % count);
            }
            sec.clear();
            for [k, t, c] in triples {
                sec.extend_from_slice(&[k, t, c]);
            }
        }
    }
    let shuf_refs: [&[u32]; 12] = std::array::from_fn(|i| shuffled_sections[i].as_slice());
    let shuf_bytes = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, shuf_refs);
    let shuf_from_bytes = IntegerDiscreteModel::from_bytes(&shuf_bytes).unwrap();
    let shuf_from_reader =
        IntegerDiscreteModel::from_reader(std::io::Cursor::new(&shuf_bytes)).unwrap();
    assert_models_equivalent(&shuf_from_reader, &shuf_from_bytes, vocab);

    // 3) Plain uor-r4.lut-discrete/1 container (fallback path in from_reader)
    let stages = expected_stages(context);
    let mut plain_table = Vec::with_capacity(vocab * 32);
    let mut plain_cells = Vec::with_capacity(vocab);
    for tok in 0..vocab {
        let roots = token_4block_roots(tok);
        for &r in &roots {
            for d in ROOTS[usize::from(r)] {
                plain_table.push(i32::from(d));
            }
        }
        plain_cells.push((u32::from(roots[0]) << 4) + u32::from(roots[3]));
    }
    let plain_routes = vec![0u32; 4 * stages];
    let plain_gates = vec![0i32; 4 * stages];
    let mut plain_blob = Vec::new();
    let mut plain_tables = Vec::new();
    let t_bytes: Vec<u8> = plain_table.iter().flat_map(|x| x.to_le_bytes()).collect();
    let t_span = push_section_bytes(&mut plain_blob, &t_bytes);
    plain_tables.push(
        json!({"name": "discrete.table", "kind": "i32", "len": plain_table.len(), "span": t_span}),
    );
    plain_tables.push(push_u32_section(
        &mut plain_blob,
        "discrete.cells",
        &plain_cells,
    ));
    plain_tables.push(push_u32_section(
        &mut plain_blob,
        "discrete.routes",
        &plain_routes,
    ));
    let g_bytes: Vec<u8> = plain_gates.iter().flat_map(|x| x.to_le_bytes()).collect();
    let g_span = push_section_bytes(&mut plain_blob, &g_bytes);
    plain_tables.push(
        json!({"name": "discrete.gates", "kind": "i32", "len": plain_gates.len(), "span": g_span}),
    );
    let plain_header = json!({
        "schema": DISCRETE_SCHEMA,
        "shape": {
            "vocabulary": vocab, "blocks": 4, "context": context, "kept_cells": 3, "stages": stages,
            "read_mode": "pick", "read_kept": 0, "read_sweeps": 0, "read_position": 3,
            "objective_policy": "nearest_address_margin_v1", "read_initialization": "zero_gates_v1",
            "seed": 7, "epoch": 1,
        },
        "matrices": [],
        "tables": plain_tables,
        "source": {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "snapshot_sha256": "00".repeat(32),
            "operator_kappa": "blake3:a0ea3a10dd10c5f4856b0d881d33d94c3293bad0e68bd07a47918e2bb82c819b",
            "composition_kappa": "blake3:f2986c8e68dcb30a9cb511a42547179a87b66ef212a87b728765d03e70e640b0",
            "kernel_sha256": "11".repeat(32),
            "source_sha256": "22".repeat(32),
            "geometry_sha256": "33".repeat(32),
        }
    });
    let plain_hdr = serde_json::to_vec(&plain_header).unwrap();
    let mut plain_bytes = Vec::new();
    plain_bytes.extend_from_slice(MAGIC);
    plain_bytes.extend_from_slice(&(plain_hdr.len() as u64).to_le_bytes());
    plain_bytes.extend_from_slice(&plain_hdr);
    let plain_pad = (ALIGN - plain_bytes.len() % ALIGN) % ALIGN;
    plain_bytes.resize(plain_bytes.len() + plain_pad, 0);
    plain_bytes.extend_from_slice(&plain_blob);
    assert_models_equivalent(
        &IntegerDiscreteModel::from_reader(std::io::Cursor::new(&plain_bytes)).unwrap(),
        &IntegerDiscreteModel::from_bytes(&plain_bytes).unwrap(),
        vocab,
    );

    // 4) S2 passive (uor-r4.lut-discrete-trunk-s2/1) streaming parity
    let mut s2_blob = Vec::new();
    let mut s2_tables = Vec::new();
    let s2_t_span = push_section_bytes(&mut s2_blob, &t_bytes);
    s2_tables.push(
        json!({"name": "discrete.table", "kind": "i32", "len": plain_table.len(), "span": s2_t_span}),
    );
    s2_tables.push(push_u32_section(
        &mut s2_blob,
        "discrete.cells",
        &plain_cells,
    ));
    s2_tables.push(push_u32_section(
        &mut s2_blob,
        "discrete.routes",
        &plain_routes,
    ));
    let s2_g_span = push_section_bytes(&mut s2_blob, &g_bytes);
    s2_tables.push(
        json!({"name": "discrete.gates", "kind": "i32", "len": plain_gates.len(), "span": s2_g_span}),
    );
    s2_tables.push(push_u32_section(
        &mut s2_blob,
        "discrete.trunk.step_candidates",
        &steps,
    ));
    s2_tables.push(push_u32_section(
        &mut s2_blob,
        "discrete.trunk.norm_candidates",
        &norms,
    ));
    s2_tables.push(push_u32_section(
        &mut s2_blob,
        "discrete.trunk.unigram",
        &unigram,
    ));
    for (idx, name) in SPARSE_TABLE_NAMES.iter().enumerate() {
        s2_tables.push(push_u32_section(&mut s2_blob, name, sorted_refs[idx]));
    }
    let mut s2_payload = vec![0xAAu8; 96];
    let mut row = [0u8; 20];
    row[..16].copy_from_slice(&(1u128 << 120).to_le_bytes()); // order=1, route=0
    row[16..18].copy_from_slice(&2u16.to_le_bytes());
    row[18] = 3; // rank
    s2_payload.extend_from_slice(&row);
    let s2_span = push_section_bytes(&mut s2_blob, &s2_payload);
    s2_tables.push(
        json!({"name": S2_SECTION, "kind": "u32", "len": s2_payload.len() / 4, "span": s2_span}),
    );
    let s2_header = json!({
        "schema": DISCRETE_S2_SCHEMA,
        "shape": {
            "vocabulary": vocab, "blocks": 4, "context": context, "kept_cells": 3, "stages": stages,
            "read_mode": "pick", "read_kept": 0, "read_sweeps": 0, "read_position": 3,
            "objective_policy": "nearest_address_margin_v1", "read_initialization": "zero_gates_v1",
            "seed": 7, "epoch": 1,
        },
        "matrices": [],
        "tables": s2_tables,
        "source": {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "snapshot_sha256": "00".repeat(32),
            "operator_kappa": "blake3:a0ea3a10dd10c5f4856b0d881d33d94c3293bad0e68bd07a47918e2bb82c819b",
            "composition_kappa": "blake3:f2986c8e68dcb30a9cb511a42547179a87b66ef212a87b728765d03e70e640b0",
            "kernel_sha256": "11".repeat(32),
            "source_sha256": "22".repeat(32),
            "geometry_sha256": "33".repeat(32),
        }
    });
    let s2_hdr = serde_json::to_vec(&s2_header).unwrap();
    let mut s2_bytes = Vec::new();
    s2_bytes.extend_from_slice(MAGIC);
    s2_bytes.extend_from_slice(&(s2_hdr.len() as u64).to_le_bytes());
    s2_bytes.extend_from_slice(&s2_hdr);
    let s2_pad = (ALIGN - s2_bytes.len() % ALIGN) % ALIGN;
    s2_bytes.resize(s2_bytes.len() + s2_pad, 0);
    s2_bytes.extend_from_slice(&s2_blob);
    let m_s2_bytes = IntegerDiscreteModel::from_bytes(&s2_bytes).unwrap();
    let m_s2_reader = IntegerDiscreteModel::from_reader(std::io::Cursor::new(&s2_bytes)).unwrap();
    assert_models_equivalent(&m_s2_reader, &m_s2_bytes, vocab);
    // Active S2 requires V=4096; verify rejection parity between from_reader and from_bytes.
    let expert_bytes: Vec<u8> = {
        let mut h = s2_header.clone();
        h["schema"] = json!(DISCRETE_S2_EXPERT_SCHEMA);
        let h_vec = serde_json::to_vec(&h).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(h_vec.len() as u64).to_le_bytes());
        out.extend_from_slice(&h_vec);
        let pad = (ALIGN - out.len() % ALIGN) % ALIGN;
        out.resize(out.len() + pad, 0);
        out.extend_from_slice(&s2_blob);
        out
    };
    assert!(IntegerDiscreteModel::from_bytes(&expert_bytes).is_err());
    assert!(IntegerDiscreteModel::from_reader(std::io::Cursor::new(&expert_bytes)).is_err());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn streaming_from_reader_rejects_planted_truncation_padding_and_trailing_faults() {
    let vocab = 16usize;
    let context = 4usize;
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let (unigram, sorted_sections) = fixture_sparse_data();
    let sorted_refs: [&[u32]; 12] = std::array::from_fn(|i| sorted_sections[i].as_slice());
    let valid = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, sorted_refs);

    // 1) Truncated preamble / header / body
    assert!(IntegerDiscreteModel::from_reader(std::io::Cursor::new(&valid[..10])).is_err());
    assert!(IntegerDiscreteModel::from_reader(std::io::Cursor::new(&valid[..100])).is_err());
    assert!(
        IntegerDiscreteModel::from_reader(std::io::Cursor::new(&valid[..valid.len() - 4])).is_err()
    );

    // 2) Trailing byte after final section
    let mut with_tail = valid.clone();
    with_tail.push(0);
    assert!(IntegerDiscreteModel::from_reader(std::io::Cursor::new(&with_tail)).is_err());

    // 3) Non-zero header padding byte (if padding exists)
    let hdr_len = u64::from_le_bytes(valid[8..16].try_into().unwrap()) as usize;
    let unaligned = 16 + hdr_len;
    let data_start = unaligned + (ALIGN - unaligned % ALIGN) % ALIGN;
    if data_start > unaligned {
        let mut bad_hdr_pad = valid.clone();
        bad_hdr_pad[unaligned] = 1;
        assert!(IntegerDiscreteModel::from_reader(std::io::Cursor::new(&bad_hdr_pad)).is_err());
    }

    // 4) Non-zero inter-section alignment padding (routes table is 20 u32s = 80 B, padded +48 B to 128 B)
    let hdr: HeaderOnly = serde_json::from_slice(&valid[16..16 + hdr_len]).unwrap();
    let routes = hdr
        .tables
        .iter()
        .find(|t| t.name == "discrete.routes")
        .unwrap();
    let routes_end = data_start + (routes.span.offset + routes.span.bytes) as usize;
    let gates = hdr
        .tables
        .iter()
        .find(|t| t.name == "discrete.gates")
        .unwrap();
    let gates_start = data_start + gates.span.offset as usize;
    assert!(gates_start > routes_end);
    let mut bad_gap = valid.clone();
    bad_gap[routes_end] = 0x7f;
    assert!(IntegerDiscreteModel::from_reader(std::io::Cursor::new(&bad_gap)).is_err());

    // 5) Disk load() rejects file whose actual size differs from declared layout
    let dir = std::env::temp_dir().join(format!("uor-stream-fault-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let bad_path = dir.join("truncated.d11");
    fs::write(&bad_path, &valid[..valid.len() - 4]).unwrap();
    assert!(IntegerDiscreteModel::load(&bad_path).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "p1-count-serving")]
#[test]
fn embedded_uorknc03_deployment_streaming_roundtrip_parity_and_refusal() {
    use uor_r4_count_runtime::{CompactCountLookup, FrozenPickMixture, PickClassRule};
    use uor_r4_integer::discrete_d11::count_deployment::{
        load_client_artifact, ClientArtifactFile, SCHEMA as DEPLOYMENT_SCHEMA,
    };

    let vocab = 16usize;
    let context = 4usize;
    let steps = [0u32; 64];
    let norms = [0u32; 4];
    let (unigram, sorted_sections) = fixture_sparse_data();
    let sorted_refs: [&[u32]; 12] = std::array::from_fn(|i| sorted_sections[i].as_slice());
    let trunk = build_trunk_uorlut01(vocab, context, &steps, &norms, &unigram, sorted_refs);
    let trunk_sha = hex::encode(Sha256::digest(&trunk));

    let train_sha = [0x33u8; 32];
    let highest = 1u32;
    let mut rspec0 = [0u64; 4];
    let mut aspec0 = [0u64; 4];
    rspec0[0] = 1;
    rspec0[1] = 1;
    aspec0[0] = 2;

    // Order 0: key=[], followers (next=1, r=1, a=1) and (next=2, r=2, a=1).
    let mut v2 = Vec::new();
    v2.extend_from_slice(b"UORKNC02");
    v2.extend_from_slice(&highest.to_le_bytes());
    v2.extend_from_slice(&train_sha);
    v2.extend_from_slice(&2u32.to_le_bytes());
    v2.extend_from_slice(&1u32.to_le_bytes());
    for w in rspec0.into_iter().chain(aspec0) {
        v2.extend_from_slice(&w.to_le_bytes());
    }
    v2.extend_from_slice(&0u32.to_le_bytes());
    for (next, r, a) in [(1u16, 1u32, 1u32), (2u16, 2u32, 1u32)] {
        v2.extend_from_slice(&next.to_le_bytes());
        v2.extend_from_slice(&r.to_le_bytes());
        v2.extend_from_slice(&a.to_le_bytes());
    }
    // Order 1 (highest => shared): context [5] => next=2, r=2, a=2.
    let mut rspec1 = [0u64; 4];
    rspec1[1] = 1;
    v2.extend_from_slice(&1u32.to_le_bytes());
    v2.extend_from_slice(&1u32.to_le_bytes());
    for w in rspec1.into_iter().chain(rspec1) {
        v2.extend_from_slice(&w.to_le_bytes());
    }
    v2.extend_from_slice(&[0x00, 0x05]); // big-endian 12-bit key for token 5 in order 1
    v2.extend_from_slice(&0u32.to_le_bytes());
    v2.extend_from_slice(&2u16.to_le_bytes());
    v2.extend_from_slice(&2u32.to_le_bytes());

    // Exact matching UORKNC03 encoding.
    let mut v3 = Vec::new();
    v3.extend_from_slice(b"UORKNC03");
    v3.extend_from_slice(&highest.to_le_bytes());
    v3.extend_from_slice(&train_sha);
    // Order 0 section:
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
    v3.push(1); // fol_deg = 2 - 1
    v3.extend_from_slice(&((1u16 << 4) | 0u16).to_le_bytes()); // (1, 1) => code 0
    v3.extend_from_slice(&((2u16 << 4) | 1u16).to_le_bytes()); // (2, 1) => code 1
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&0u32.to_le_bytes());
    // Order 1 section (shared):
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
    v3.push(1); // parent_deg[0] = 1
    v3.extend_from_slice(&[0x05, 0x00]); // 12-bit child_tok = 5 (padded nibble = 0)
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&1u32.to_le_bytes());
    v3.push(0); // fol_deg[0] = 1 - 1
    v3.extend_from_slice(&((2u16 << 4) | 1u16).to_le_bytes()); // shared count=2 => code 1
    v3.extend_from_slice(&0u32.to_le_bytes());
    v3.extend_from_slice(&0u32.to_le_bytes());

    let v2_sha: [u8; 32] = Sha256::digest(&v2).into();
    let v3_sha: [u8; 32] = Sha256::digest(&v3).into();
    let l2 = CompactCountLookup::from_bytes(&v2, train_sha, v2_sha).unwrap();
    let l3 = CompactCountLookup::from_bytes(&v3, train_sha, v3_sha).unwrap();

    let mut weights_vec = vec![0u64; 176 * 5 * 14];
    for row in weights_vec.chunks_exact_mut(14) {
        row[0] = 1 << 32;
    }
    let frozen =
        FrozenPickMixture::with_rule(10, [0x55; 32], weights_vec, PickClassRule::Five).unwrap();
    let weights_bytes = frozen.to_bytes();
    let fit_manifest = serde_json::to_vec(&json!({
        "schema": "uor-r4.d37.r-fit/1",
        "arm": "a1s",
        "weight_format": "UORS5W02",
        "pick_classes": 5,
        "q32_weight_words": 176 * 5 * 14,
        "prepared": {
            "format": "uor-r4.d37.r-preparation/1",
            "c_sha256": hex::encode(train_sha),
            "r_sha256": "11".repeat(32),
            "pick_mode": "identity",
            "trunk_sha256": trunk_sha,
            "compact_sha256": hex::encode(v3_sha),
        }
    }))
    .unwrap();
    let mut state_bytes = Vec::new();
    state_bytes.extend_from_slice(b"UORP1A01");
    state_bytes.extend_from_slice(&(fit_manifest.len() as u32).to_le_bytes());
    state_bytes.extend_from_slice(&(weights_bytes.len() as u32).to_le_bytes());
    state_bytes.extend_from_slice(&Sha256::digest(&weights_bytes));
    state_bytes.extend_from_slice(&fit_manifest);
    state_bytes.extend_from_slice(&weights_bytes);
    state_bytes.extend_from_slice(&v3);
    let state_sha = hex::encode(Sha256::digest(&state_bytes));

    let emb_hdr = serde_json::to_vec(&json!({
        "schema": DEPLOYMENT_SCHEMA,
        "trunk_bytes": trunk.len(),
        "state_bytes": state_bytes.len(),
        "trunk_sha256": trunk_sha,
        "state_sha256": state_sha,
    }))
    .unwrap();
    let mut deploy = Vec::new();
    deploy.extend_from_slice(MAGIC);
    deploy.extend_from_slice(&(emb_hdr.len() as u64).to_le_bytes());
    deploy.extend_from_slice(&emb_hdr);
    deploy.extend_from_slice(&trunk);
    deploy.extend_from_slice(&state_bytes);

    let dir = std::env::temp_dir().join(format!("uor-v3-deploy-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("v3-embed.d11");
    fs::write(&path, &deploy).unwrap();

    let (loaded_model, loaded_state) = match load_client_artifact(&path, false).unwrap() {
        ClientArtifactFile::Embedded {
            model,
            state_bytes: sb,
            expected_state_sha256,
            identity,
        } => {
            assert_eq!(expected_state_sha256, state_sha);
            assert_eq!(
                identity.artifact_sha256,
                hex::encode(Sha256::digest(&deploy))
            );
            (model, sb)
        }
        ClientArtifactFile::Legacy(_) => panic!("expected embedded deployment"),
    };
    let want_model = IntegerDiscreteModel::from_bytes(&trunk).unwrap();
    assert_eq!(loaded_model.artifact_sha256(), want_model.artifact_sha256());

    let fit_len = u32::from_le_bytes(loaded_state[8..12].try_into().unwrap()) as usize;
    let wt_len = u32::from_le_bytes(loaded_state[12..16].try_into().unwrap()) as usize;
    let decoded_v3 =
        CompactCountLookup::from_bytes(&loaded_state[48 + fit_len + wt_len..], train_sha, v3_sha)
            .unwrap();
    for ctx_tok in [0u16, 4u16, 5u16] {
        for tgt in [1u16, 2u16, 3u16] {
            let seq = [ctx_tok, tgt];
            let (mut c2, mut t2) = ([0u32; 2], [0u64; 2]);
            let (mut c3, mut t3) = ([0u32; 2], [0u64; 2]);
            let (mut cd, mut td) = ([0u32; 2], [0u64; 2]);
            let m2 = l2.lookup(&seq, 1, &mut c2, &mut t2).unwrap();
            let m3 = l3.lookup(&seq, 1, &mut c3, &mut t3).unwrap();
            let md = decoded_v3.lookup(&seq, 1, &mut cd, &mut td).unwrap();
            assert_eq!((m2, c2, t2), (m3, c3, t3));
            assert_eq!((m2, c2, t2), (md, cd, td));
        }
    }

    // Tampered trailing byte must be rejected by streaming artifact loader.
    let mut bad = deploy.clone();
    bad.push(0);
    let bad_path = dir.join("v3-embed-bad.d11");
    fs::write(&bad_path, &bad).unwrap();
    assert!(load_client_artifact(&bad_path, false).is_err());

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "p1-count-serving")]
#[test]
fn uors3t01_e8_parity_starling_miss_equal_byte_arms_and_refusals() {
    use sha2::{Digest, Sha256};
    use uor_r4_count_runtime::s3_state::{
        build_equal_byte_s3_table, e8_root_coords_2x, e8_root_neighbours_56, e8_snap_240,
        encode_uors3t01, encoded_s3_bytes, output_pick_score_q16, random_transport_blocks,
        s3_primary_key, starling_k_neighbours, S3AddressArm, S3HitKind, S3StateTable, S3_Q16_ONE,
        S3_Q48_ONE, S3_VOCABULARY,
    };
    use uor_r4_integer::fbs_address::{e8_cell, ADDRESS_SHARED_BLOCKS};

    // 1. Bit-exact Conway-Sloane E8 parity against uor_r4_integer::fbs_address::e8_cell
    // across all 12 shared boundary blocks, extremes, and 10,000 deterministic blocks.
    for block in &ADDRESS_SHARED_BLOCKS {
        assert_eq!(u32::from(e8_snap_240(block)), e8_cell(block));
    }
    for extreme in [
        [i32::MIN; 8],
        [i32::MAX; 8],
        [i32::MIN, i32::MAX, 0, -1, 1, i32::MIN / 2, i32::MAX / 2, 0],
    ] {
        assert_eq!(u32::from(e8_snap_240(&extreme)), e8_cell(&extreme));
    }
    let mut rng = 0xD85_5300_2026_1125u64;
    for round in 0..10_000 {
        let range = match round % 3 {
            0 => i64::from(u16::MAX),
            1 => 1i64 << 20,
            _ => 9i64,
        };
        let mut block = [0i32; 8];
        for dst in &mut block {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let v = z ^ (z >> 31);
            *dst = ((v % (range as u64 + 1)) as i64 - range / 2) as i32;
        }
        assert_eq!(u32::from(e8_snap_240(&block)), e8_cell(&block));
    }

    // 2. Linear Q16 output Pick scores across all 5 E8 dot_2x shells {-8, -4, 0, 4, 8}.
    let nb_r0 = e8_root_neighbours_56(0).unwrap()[0];
    let c0 = e8_root_coords_2x(0).unwrap();
    let opp0 = (0u8..240u8)
        .find(|&r| {
            let cr = e8_root_coords_2x(r).unwrap();
            cr.iter().zip(c0.iter()).all(|(&a, &b)| a == -b)
        })
        .unwrap();
    let orth0 = (0u8..240u8)
        .find(|&r| {
            let cr = e8_root_coords_2x(r).unwrap();
            let d: i32 = cr
                .iter()
                .zip(c0.iter())
                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                .sum();
            d == 0
        })
        .unwrap();
    let neg_adj0 = (0u8..240u8)
        .find(|&r| {
            let cr = e8_root_coords_2x(r).unwrap();
            let d: i32 = cr
                .iter()
                .zip(c0.iter())
                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                .sum();
            d == -4
        })
        .unwrap();
    assert_eq!(output_pick_score_q16(0, opp0).unwrap(), 16_384);
    assert_eq!(output_pick_score_q16(0, neg_adj0).unwrap(), 28_672);
    assert_eq!(output_pick_score_q16(0, orth0).unwrap(), 40_960);
    assert_eq!(output_pick_score_q16(0, nb_r0).unwrap(), 53_248);
    assert_eq!(output_pick_score_q16(0, 0).unwrap(), S3_Q16_ONE);

    // 3. UORS3T01 round-trip, Starling miss, and fair equal-byte arms (including random_transport_blocks).
    let mut token_roots = [0u8; S3_VOCABULARY];
    for (i, slot) in token_roots.iter_mut().enumerate() {
        *slot = (i * 7 % 240) as u8;
    }
    let train_sha = [0xA5u8; 32];
    let blk_a = ADDRESS_SHARED_BLOCKS[1];
    let blk_b = ADDRESS_SHARED_BLOCKS[2];
    let ra = e8_snap_240(&blk_a);
    let nb = e8_root_neighbours_56(ra).unwrap()[0];
    let nb_coords = e8_root_coords_2x(nb).unwrap();
    let mut blk_a_miss = [0i32; 8];
    for (dst, (&orig, &alt)) in blk_a_miss
        .iter_mut()
        .zip(blk_a.iter().zip(nb_coords.iter()))
    {
        *dst = orig / 8 + i32::from(alt) * 32_768;
    }
    assert!(
        starling_k_neighbours(&blk_a_miss, e8_snap_240(&blk_a_miss), 7)
            .unwrap()
            .contains(&ra)
    );

    let p0 = &[0u16, 10][..];
    let p1 = &[0u16, 11][..];
    let p2 = &[0u16, 12][..];
    let obs = [
        (&blk_a, &blk_b, p0, 50u16),
        (&blk_a, &blk_b, p0, 50u16),
        (&blk_a, &blk_b, p1, 50u16),
        (&blk_a_miss, &blk_b, p2, 60u16),
    ];
    let rt0 = random_transport_blocks(42, p0).unwrap();
    let rt1 = random_transport_blocks(42, p1).unwrap();
    let rt2 = random_transport_blocks(42, p2).unwrap();
    assert_ne!(
        S3AddressArm::E8Product
            .context_keys(&blk_a, &blk_b, p0)
            .unwrap(),
        S3AddressArm::RandomTransport { seed: 42 }
            .context_keys(&rt0.0, &rt0.1, p0)
            .unwrap()
    );
    let rt_obs = [
        (&rt0.0, &rt0.1, p0, 50u16),
        (&rt0.0, &rt0.1, p0, 50u16),
        (&rt1.0, &rt1.1, p1, 50u16),
        (&rt2.0, &rt2.1, p2, 60u16),
    ];

    let budget = 72 + 4096 + 128;
    for (arm, bytes) in [
        (
            S3AddressArm::E8Product,
            build_equal_byte_s3_table(
                S3AddressArm::E8Product,
                train_sha,
                Some(token_roots),
                &obs,
                budget,
            )
            .unwrap(),
        ),
        (
            S3AddressArm::TokenContext { k: 6 },
            build_equal_byte_s3_table(
                S3AddressArm::TokenContext { k: 6 },
                train_sha,
                None,
                &obs,
                budget,
            )
            .unwrap(),
        ),
        (
            S3AddressArm::RandomTransport { seed: 42 },
            build_equal_byte_s3_table(
                S3AddressArm::RandomTransport { seed: 42 },
                train_sha,
                None,
                &rt_obs,
                budget,
            )
            .unwrap(),
        ),
        (
            S3AddressArm::SplitMixPrefix { seed: 42 },
            build_equal_byte_s3_table(
                S3AddressArm::SplitMixPrefix { seed: 42 },
                train_sha,
                None,
                &obs,
                budget,
            )
            .unwrap(),
        ),
    ] {
        assert!(bytes.len() <= budget);
        let decoded =
            S3StateTable::from_bytes(&bytes, train_sha, Sha256::digest(&bytes).into()).unwrap();
        assert_eq!(decoded.arm(), arm);
    }

    // RandomTransport rejects Some(token_roots) and exercises ExactPrimary / ExactBackoff / UniformFallback without StarlingMiss.
    assert!(build_equal_byte_s3_table(
        S3AddressArm::RandomTransport { seed: 42 },
        train_sha,
        Some(token_roots),
        &rt_obs,
        budget,
    )
    .is_err());
    let full_rt = build_equal_byte_s3_table(
        S3AddressArm::RandomTransport { seed: 42 },
        train_sha,
        None,
        &rt_obs[..3],
        4096,
    )
    .unwrap();
    let rt_tab =
        S3StateTable::from_bytes(&full_rt, train_sha, Sha256::digest(&full_rt).into()).unwrap();
    assert!(!rt_tab.has_token_roots());
    let rt_prim = rt_tab.lookup_q48(&rt0.0, &rt0.1, p0, 50).unwrap();
    assert_eq!(rt_prim.hit_kind, S3HitKind::ExactPrimary);
    assert_eq!(rt_prim.predicted_root, 0);
    assert_eq!(rt_prim.output_pick_q16, S3_Q16_ONE);
    let rt_back = rt_tab.lookup_q48(&rt0.0, &rt0.1, &[0, 99], 50).unwrap();
    assert_eq!(rt_back.hit_kind, S3HitKind::ExactBackoff);
    assert_eq!(rt_back.output_pick_q16, S3_Q16_ONE);
    let rt0_nb_coords =
        e8_root_coords_2x(e8_root_neighbours_56(e8_snap_240(&rt0.0)).unwrap()[0]).unwrap();
    let mut rt0_miss_blk = [0i32; 8];
    for (dst, (&orig, &alt)) in rt0_miss_blk
        .iter_mut()
        .zip(rt0.0.iter().zip(rt0_nb_coords.iter()))
    {
        *dst = orig / 8 + i32::from(alt) * 32_768;
    }
    let rt_unif = rt_tab
        .lookup_q48(&rt0_miss_blk, &rt0.1, &[0, 99], 50)
        .unwrap();
    assert_eq!(rt_unif.hit_kind, S3HitKind::UniformFallback);
    assert_eq!(rt_unif.predicted_root, 0);
    assert_eq!(rt_unif.output_pick_q16, S3_Q16_ONE);

    // SplitMixPrefix position invariance across distinct prefixes sharing 16 trailing tokens + lookup_q48 checks.
    let suffix16: Vec<u16> = (100..116).collect();
    let mut padded = vec![5u16, 7u16, 9u16];
    padded.extend_from_slice(&suffix16);
    assert_eq!(
        S3AddressArm::SplitMixPrefix { seed: 99 }
            .context_keys(&blk_a, &blk_b, &suffix16)
            .unwrap(),
        S3AddressArm::SplitMixPrefix { seed: 99 }
            .context_keys(&blk_a, &blk_b, &padded)
            .unwrap()
    );
    let full_sm = build_equal_byte_s3_table(
        S3AddressArm::SplitMixPrefix { seed: 42 },
        train_sha,
        None,
        &obs[..3],
        4096,
    )
    .unwrap();
    let sm_tab =
        S3StateTable::from_bytes(&full_sm, train_sha, Sha256::digest(&full_sm).into()).unwrap();
    let sm_prim = sm_tab.lookup_q48(&blk_a, &blk_b, p0, 50).unwrap();
    assert_eq!(sm_prim.hit_kind, S3HitKind::ExactPrimary);
    assert_eq!(sm_prim.output_pick_q16, S3_Q16_ONE);
    let sm_unif = sm_tab.lookup_q48(&blk_a, &blk_b, &[999, 888], 50).unwrap();
    assert_eq!(sm_unif.hit_kind, S3HitKind::UniformFallback);
    assert_eq!(sm_unif.output_pick_q16, S3_Q16_ONE);

    // TokenContext k=1, k=2 (including &[42] vs &[0, 42] sentinel 4096), k=6 collision-free & zero token_e8_roots tax.
    let tc1 = S3AddressArm::TokenContext { k: 1 };
    let tc2 = S3AddressArm::TokenContext { k: 2 };
    let k1_keys = tc1.context_keys(&blk_a, &blk_b, &[9, 42]).unwrap();
    assert_eq!(k1_keys.0, s3_primary_key(0, 42));
    assert_eq!(k1_keys.1, None);
    let k2_keys = tc2.context_keys(&blk_a, &blk_b, &[9, 42]).unwrap();
    assert_eq!(k2_keys.0, s3_primary_key(9, 42));
    assert_eq!(k2_keys.1, Some(42));
    let k2_short = tc2.context_keys(&blk_a, &blk_b, &[42]).unwrap();
    let k2_bos = tc2.context_keys(&blk_a, &blk_b, &[0, 42]).unwrap();
    assert_ne!(k2_short.0, k2_bos.0);
    assert_eq!(k2_short.0, s3_primary_key(S3_VOCABULARY as u16, 42));
    assert_eq!(k2_bos.0, s3_primary_key(0, 42));

    let full_tc2 = build_equal_byte_s3_table(tc2, train_sha, None, &obs[..3], 4096).unwrap();
    let tc2_tab =
        S3StateTable::from_bytes(&full_tc2, train_sha, Sha256::digest(&full_tc2).into()).unwrap();
    let tc2_prim = tc2_tab.lookup_q48(&blk_a, &blk_b, p0, 50).unwrap();
    assert_eq!(tc2_prim.hit_kind, S3HitKind::ExactPrimary);
    assert_eq!(tc2_prim.output_pick_q16, S3_Q16_ONE);
    let tc2_back = tc2_tab.lookup_q48(&blk_a, &blk_b, &[55, 10], 50).unwrap();
    assert_eq!(tc2_back.hit_kind, S3HitKind::ExactBackoff);
    let tc2_unif = tc2_tab.lookup_q48(&blk_a, &blk_b, &[55, 777], 50).unwrap();
    assert_eq!(tc2_unif.hit_kind, S3HitKind::UniformFallback);

    let full_tc6 = build_equal_byte_s3_table(
        S3AddressArm::TokenContext { k: 6 },
        train_sha,
        None,
        &obs[..3],
        4096,
    )
    .unwrap();
    let tc6_tab =
        S3StateTable::from_bytes(&full_tc6, train_sha, Sha256::digest(&full_tc6).into()).unwrap();
    assert_eq!(
        tc6_tab.lookup_q48(&blk_a, &blk_b, p0, 50).unwrap().hit_kind,
        S3HitKind::ExactPrimary
    );
    assert_eq!(
        tc6_tab
            .lookup_q48(&blk_a, &blk_b, &[88, 10], 50)
            .unwrap()
            .hit_kind,
        S3HitKind::ExactBackoff
    );

    // snap_i64_accum high-count scaling (count = u32::MAX forces right-shift > 0 without i32 overflow).
    let (big_pk, _, _, _) = S3AddressArm::E8Product
        .context_keys(&blk_a, &blk_b, &[0, 10])
        .unwrap();
    let big_bytes = encode_uors3t01(
        S3AddressArm::E8Product,
        train_sha,
        Some(token_roots),
        &[(big_pk, &[(50, u32::MAX, 0)])],
        &[],
    )
    .unwrap();
    let big_tab =
        S3StateTable::from_bytes(&big_bytes, train_sha, Sha256::digest(&big_bytes).into()).unwrap();
    let big_res = big_tab.lookup_q48(&blk_a, &blk_b, &[0, 10], 50).unwrap();
    assert_eq!(big_res.predicted_root, token_roots[50]);
    assert_eq!(big_res.output_pick_q16, S3_Q16_ONE);

    let full_e8 = build_equal_byte_s3_table(
        S3AddressArm::E8Product,
        train_sha,
        Some(token_roots),
        &obs[..3],
        8192,
    )
    .unwrap();
    let table =
        S3StateTable::from_bytes(&full_e8, train_sha, Sha256::digest(&full_e8).into()).unwrap();
    let exact = table.lookup_q48(&blk_a, &blk_b, &[0, 10], 50).unwrap();
    assert_eq!(exact.hit_kind, S3HitKind::ExactPrimary);
    assert_eq!(exact.mass_q48, S3_Q48_ONE);
    assert_eq!(
        exact.output_pick_q16,
        output_pick_score_q16(exact.predicted_root, token_roots[50]).unwrap()
    );
    let hit_other_target = table.lookup_q48(&blk_a, &blk_b, &[0, 10], 77).unwrap();
    assert_eq!(hit_other_target.hit_kind, S3HitKind::ExactPrimary);
    assert_eq!(hit_other_target.mass_q48, 1);
    let starling = table.lookup_q48(&blk_a_miss, &blk_b, &[0, 99], 50).unwrap();
    assert_eq!(starling.hit_kind, S3HitKind::StarlingMiss);
    assert_eq!(starling.mass_q48, S3_Q48_ONE);
    let far = [99_999, -99_999, 99_999, -99_999, 1, 2, 3, 4];
    let uniform = table.lookup_q48(&far, &far, &[0, 99], 50).unwrap();
    assert_eq!(uniform.hit_kind, S3HitKind::UniformFallback);
    assert_eq!(uniform.mass_q48, S3_Q48_ONE / (S3_VOCABULARY as u64));

    // 4. Planted fail-closed & overflow-safety refusals.
    assert!(encoded_s3_bytes(true, usize::MAX / 18 + 1, 0, 0, 0).is_err());
    assert!(S3StateTable::from_bytes(&full_e8, train_sha, [0u8; 32]).is_err());
    assert!(
        S3StateTable::from_bytes(&full_e8, [0u8; 32], Sha256::digest(&full_e8).into()).is_err()
    );
    let mut bad_magic = full_e8.clone();
    bad_magic[0] ^= 1;
    assert!(
        S3StateTable::from_bytes(&bad_magic, train_sha, Sha256::digest(&bad_magic).into()).is_err()
    );
    let mut bad_ver = full_e8.clone();
    bad_ver[8] = 2;
    assert!(
        S3StateTable::from_bytes(&bad_ver, train_sha, Sha256::digest(&bad_ver).into()).is_err()
    );
    let mut bad_flags = full_e8.clone();
    bad_flags[14] = 2;
    assert!(
        S3StateTable::from_bytes(&bad_flags, train_sha, Sha256::digest(&bad_flags).into()).is_err()
    );
    let mut bad_arm_flag = full_e8.clone();
    bad_arm_flag[14] = 0; // E8Product requires flags & 1 != 0
    assert!(S3StateTable::from_bytes(
        &bad_arm_flag,
        train_sha,
        Sha256::digest(&bad_arm_flag).into()
    )
    .is_err());
    let mut bad_arm = full_e8.clone();
    bad_arm[12] = 9;
    assert!(
        S3StateTable::from_bytes(&bad_arm, train_sha, Sha256::digest(&bad_arm).into()).is_err()
    );
    let mut trailing = full_e8.clone();
    trailing.push(0);
    assert!(
        S3StateTable::from_bytes(&trailing, train_sha, Sha256::digest(&trailing).into()).is_err()
    );
    let mut bad_root = full_e8.clone();
    bad_root[72] = 240;
    assert!(
        S3StateTable::from_bytes(&bad_root, train_sha, Sha256::digest(&bad_root).into()).is_err()
    );
    let mut trunc = full_e8.clone();
    trunc.pop();
    assert!(S3StateTable::from_bytes(&trunc, train_sha, Sha256::digest(&trunc).into()).is_err());

    // Crafted 32-bit multiplication-wrap overflow payload must fail closed cleanly.
    let mut ovf = vec![0u8; 72];
    ovf[..8].copy_from_slice(b"UORS3T01");
    ovf[8..12].copy_from_slice(&1u32.to_le_bytes());
    ovf[12] = 1; // TokenContext
    ovf[13] = 1; // k=1
    ovf[24..56].copy_from_slice(&train_sha);
    ovf[56..60].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    assert!(S3StateTable::from_bytes(&ovf, train_sha, Sha256::digest(&ovf).into()).is_err());

    // Corrupted primary last_token >= 4096, row_len == 0, total != sum, count == 0, next_token >= 4096, unsorted.
    let valid_tc = S3StateTable::encode_bytes(
        S3AddressArm::TokenContext { k: 2 },
        train_sha,
        None,
        &[
            (s3_primary_key(1, 10), &[(5, 3, 0), (6, 2, 0)]),
            (s3_primary_key(2, 10), &[(7, 1, 0)]),
        ],
        &[(10, &[(5, 6, 0)])],
    )
    .unwrap();
    let mut bad_lt = valid_tc.clone();
    bad_lt[72] = 0;
    bad_lt[73] = 0x10; // last_tok = 4096
    assert!(S3StateTable::from_bytes(&bad_lt, train_sha, Sha256::digest(&bad_lt).into()).is_err());

    let mut bad_tot = valid_tc.clone();
    bad_tot[72 + 4] ^= 1; // corrupt context total
    assert!(
        S3StateTable::from_bytes(&bad_tot, train_sha, Sha256::digest(&bad_tot).into()).is_err()
    );

    let mut bad_rlen = valid_tc.clone();
    bad_rlen[72 + 16] = 0;
    bad_rlen[72 + 17] = 0; // row_len = 0
    assert!(
        S3StateTable::from_bytes(&bad_rlen, train_sha, Sha256::digest(&bad_rlen).into()).is_err()
    );

    // First primary row starts at 72 + 2*18 = 108.
    let mut bad_cnt0 = valid_tc.clone();
    bad_cnt0[108 + 2..108 + 6].copy_from_slice(&0u32.to_le_bytes());
    assert!(
        S3StateTable::from_bytes(&bad_cnt0, train_sha, Sha256::digest(&bad_cnt0).into()).is_err()
    );

    let mut bad_ntok = valid_tc.clone();
    bad_ntok[108..110].copy_from_slice(&4096u16.to_le_bytes());
    assert!(
        S3StateTable::from_bytes(&bad_ntok, train_sha, Sha256::digest(&bad_ntok).into()).is_err()
    );

    let mut unsorted_rows = valid_tc.clone();
    unsorted_rows[108..110].copy_from_slice(&8u16.to_le_bytes()); // row0 next_token 8 > row1 6
    assert!(S3StateTable::from_bytes(
        &unsorted_rows,
        train_sha,
        Sha256::digest(&unsorted_rows).into()
    )
    .is_err());

    let mut unsorted_ctx = valid_tc.clone();
    unsorted_ctx[72..76].copy_from_slice(&s3_primary_key(2, 10).to_le_bytes());
    assert!(S3StateTable::from_bytes(
        &unsorted_ctx,
        train_sha,
        Sha256::digest(&unsorted_ctx).into()
    )
    .is_err());
}

#[cfg(feature = "p1-count-serving")]
#[test]
fn uors4p01_uors6b01_p3_parity_controls_and_refusals() {
    use sha2::{Digest, Sha256};
    use uor_r4_count_runtime::p3_state::{
        apply_lawful_stop_and_rep_guard, apply_s3_s4_err_write, brier_mix_q48,
        extract_turn_pair_keys, isqrt_ceil_u64, isqrt_u64, p3_token_primes, paired_band_compare,
        s4_pair_key, s4_prime_product_u64, unpack_s4_pair_key, BrierS6Weights, BrierSample,
        LawfulStopConfig, S4ConjunctionTable, S4ControlMode, S4HingeExample, S4PointerEntry,
        P3_EOS, P3_Q16_ONE, P3_Q32_ONE, P3_Q48_ONE, P3_VOCABULARY,
    };
    use uor_r4_integer::fbs_pair_rows::token_primes;

    // 1. Cross-crate prime bijection between p3_state and fbs_pair_rows across all 4,096 tokens.
    let primes_p3 = p3_token_primes();
    let primes_fbs = token_primes(P3_VOCABULARY).expect("4096 primes fit < 2^31");
    assert_eq!(primes_p3, primes_fbs);
    assert_eq!(
        s4_prime_product_u64(101, 4095, &primes_p3).unwrap(),
        u64::from(primes_fbs[101]) * u64::from(primes_fbs[4095])
    );

    // 2. Order-free S4 conjunction pointers, hinge updates, controls & lossless UORS4P01 codec.
    let train_sha = [0x5Cu8; 32];
    let turn_keys = extract_turn_pair_keys(&[200, 10, 300, 10], 0).unwrap();
    assert_eq!(turn_keys.len(), 3);
    let k0 = s4_pair_key(10, 200).unwrap();
    let k1 = s4_pair_key(10, 300).unwrap();
    let k2 = s4_pair_key(200, 300).unwrap();
    let k3 = s4_pair_key(400, 500).unwrap();
    let e_gold_a = S4PointerEntry {
        role_class: 1,
        span_offset: 2,
        count: 3,
        err_i8: 0,
    };
    let e_rival_a = S4PointerEntry {
        role_class: 2,
        span_offset: 3,
        count: 4,
        err_i8: 0,
    };
    let e_gold_b = S4PointerEntry {
        role_class: 1,
        span_offset: 4,
        count: 5,
        err_i8: 0,
    };
    let mut s4_rows = vec![
        (k0, vec![e_gold_a, e_rival_a]),
        (k1, vec![e_gold_a, e_rival_a]),
        (k2, vec![e_gold_a, e_rival_a]),
        (k3, vec![e_gold_b, e_rival_a]),
    ];
    s4_rows.sort_by_key(|(k, _)| *k);
    let s4_refs: Vec<(u32, &[S4PointerEntry])> =
        s4_rows.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let mut s4 = S4ConjunctionTable::from_sorted_rows(train_sha, &s4_refs).unwrap();

    let candidates = vec![(1u8, 2u16), (2u8, 3u16), (1u8, 4u16)];
    let mut examples = Vec::new();
    for i in 0..80usize {
        if i % 2 == 0 {
            examples.push(S4HingeExample {
                pair_keys: turn_keys.clone(),
                candidates: candidates.clone(),
                gold_index: 0,
            });
        } else {
            examples.push(S4HingeExample {
                pair_keys: vec![k3],
                candidates: candidates.clone(),
                gold_index: 2,
            });
        }
    }
    let h1 = s4
        .s4_hinge_step(&examples, (P3_Q16_ONE as i64) / 2, 2)
        .unwrap();
    let h2 = s4
        .s4_hinge_step(&examples, (P3_Q16_ONE as i64) / 2, 2)
        .unwrap();
    assert_eq!(h2.exact_top1, 80);
    assert!(h2.total_hinge_loss_q16 <= h1.total_hinge_loss_q16);

    let s4_bytes = s4.encode_bytes().unwrap();
    let s4_sha: [u8; 32] = Sha256::digest(&s4_bytes).into();
    let s4_reloaded = S4ConjunctionTable::from_bytes(&s4_bytes, train_sha, s4_sha).unwrap();
    assert_eq!(s4_reloaded, s4);

    let mut act_hits = Vec::with_capacity(80);
    let mut no_hits = Vec::with_capacity(80);
    let mut shuf_hits = Vec::with_capacity(80);
    for (idx, ex) in examples.iter().enumerate() {
        act_hits.push(
            s4_reloaded
                .pick_pointer(&ex.pair_keys, &ex.candidates, S4ControlMode::Active)
                .map(|(i, _)| i)
                == Some(ex.gold_index),
        );
        no_hits.push(
            s4_reloaded
                .pick_pointer(&ex.pair_keys, &ex.candidates, S4ControlMode::NoMemory)
                .map(|(i, _)| i)
                == Some(ex.gold_index),
        );
        shuf_hits.push(
            s4_reloaded
                .pick_pointer(
                    &ex.pair_keys,
                    &ex.candidates,
                    S4ControlMode::ShuffledRole {
                        seed: 900 + idx as u64,
                    },
                )
                .map(|(i, _)| i)
                == Some(ex.gold_index),
        );
    }
    let pb_nomem = paired_band_compare(&act_hits, &no_hits).unwrap();
    let pb_shuf = paired_band_compare(&act_hits, &shuf_hits).unwrap();
    assert!(pb_nomem.beats_control_by_band);
    assert!(pb_shuf.beats_control_by_band);

    // 3. Three-epoch integer Brier error-write, UORS6B01 round-trip & signed S3/S4 cell writes.
    let mut s6 = BrierS6Weights::uniform(2, 4, train_sha).unwrap();
    let mut samples = Vec::new();
    for i in 0..32usize {
        let bucket = i % 2;
        let masses = if bucket == 0 {
            vec![
                P3_Q48_ONE / 16,
                P3_Q48_ONE / 16,
                P3_Q48_ONE / 8,
                (P3_Q48_ONE / 8) * 7,
            ]
        } else {
            vec![
                (P3_Q48_ONE / 8) * 7,
                P3_Q48_ONE / 16,
                P3_Q48_ONE / 16,
                P3_Q48_ONE / 8,
            ]
        };
        samples.push(BrierSample {
            bucket,
            target_masses_q48: masses,
            rival_sq_sum_q32: 0,
        });
    }
    let mut prev_brier = s6.mean_brier_q32(&samples).unwrap();
    for _ in 0..3 {
        let ep = s6.brier_error_write_epoch(&samples, 1, 1).unwrap();
        assert!(ep.mean_brier_after_q32 < prev_brier);
        prev_brier = ep.mean_brier_after_q32;
    }
    let s6_bytes = s6.encode_bytes().unwrap();
    let s6_sha: [u8; 32] = Sha256::digest(&s6_bytes).into();
    let s6_reloaded = BrierS6Weights::from_bytes(&s6_bytes, train_sha, s6_sha).unwrap();
    assert_eq!(s6_reloaded, s6);
    assert_eq!(
        s6_reloaded.bucket_weights(0).unwrap().iter().sum::<u64>(),
        P3_Q32_ONE
    );
    let m_test = brier_mix_q48(
        s6_reloaded.bucket_weights(0).unwrap(),
        &samples[0].target_masses_q48,
    )
    .unwrap();
    assert!(m_test >= 1 && m_test <= P3_Q48_ONE);
    assert!(apply_s3_s4_err_write(0, true, P3_Q48_ONE / 8, 10) > 0);

    // 4. Lawful stop & repetition guard keeps positive masses and enforces EOS stop + refusals.
    let cfg = LawfulStopConfig::default();
    let mut scores = vec![16u64; P3_VOCABULARY];
    scores[42] = 65_536;
    scores[99] = 16_384;
    let trace =
        apply_lawful_stop_and_rep_guard(&[10, 20, 30, 42, 10, 20, 30], &mut scores, false, cfg)
            .unwrap();
    assert_eq!(trace.greedy_token, 99);
    assert!(scores.iter().all(|&s| s >= 1));
    let trace_stop = apply_lawful_stop_and_rep_guard(&[10, 20], &mut scores, true, cfg).unwrap();
    assert!(trace_stop.promoted_eos && trace_stop.greedy_token == P3_EOS);

    // Exact square root & fail-closed S4/S6 wire refusals.
    assert_eq!(isqrt_u64(200), 14);
    assert_eq!(isqrt_ceil_u64(200), 15);
    assert!(unpack_s4_pair_key(0x1000_0001).is_err());
    let mut bad_s4 = s4_bytes;
    bad_s4[56..60].copy_from_slice(&0x1000_0001u32.to_le_bytes());
    assert!(
        S4ConjunctionTable::from_bytes(&bad_s4, train_sha, Sha256::digest(&bad_s4).into()).is_err()
    );
    let mut bad_s6 = s6_bytes;
    bad_s6[0] ^= 0xFF;
    assert!(
        BrierS6Weights::from_bytes(&bad_s6, train_sha, Sha256::digest(&bad_s6).into()).is_err()
    );
}

#[cfg(not(feature = "serving-wasm"))]
#[test]
fn p3_prll_positive_floor_mass_invariants() {
    use uor_r4_integer::lawful_likelihood::{bits_q12, mle_masses_from_counts, PROBABILITY_ONE};

    // Any 4,096-entry positive score table emitted by `apply_lawful_stop_and_rep_guard`
    // (`s >= 1`, `sum >= 4096`) yields exact Q48 PRLL masses with finite `bits_q12` everywhere.
    let mut scores = vec![16u64; 4096];
    scores[42] = 16; // suppressed 4-gram cycle target retains >= 1 floor
    scores[99] = 16_384;
    let masses = mle_masses_from_counts(&scores).unwrap();
    assert_eq!(masses.iter().sum::<u64>(), PROBABILITY_ONE);
    assert!(masses.iter().all(|&m| m >= 1));
    assert!(bits_q12(masses[99]).unwrap() < bits_q12(masses[42]).unwrap());

    let mut stop_scores = vec![1u64; 4096];
    stop_scores[1] = PROBABILITY_ONE;
    let stop_masses = mle_masses_from_counts(&stop_scores).unwrap();
    assert_eq!(stop_masses.iter().sum::<u64>(), PROBABILITY_ONE);
    assert!(stop_masses.iter().all(|&m| m >= 1));
    assert!(bits_q12(stop_masses[1]).unwrap() < bits_q12(stop_masses[99]).unwrap());
}
