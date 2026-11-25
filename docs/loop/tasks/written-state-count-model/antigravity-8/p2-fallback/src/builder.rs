#![forbid(unsafe_code)]

use crate::fallback::{encoded_s3t02_bytes, FallbackAddressArm, S3FallbackTable, S3_VOCABULARY};
use crate::transport::{split_docs, transport_stream_cpu, unpack_query, ArmSpec, TrunkTables};
use crate::{fail, Result};
use rayon::prelude::*;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

pub const PRIM_PARTITIONS: usize = 32;
pub const BACK_PARTITIONS: usize = 8;
const PASS1_PAR_BATCH: usize = 4;
const TARGET_CHUNK_DOCS_TOKENS: usize = 16_000_000;

#[inline]
fn primary_part_id(arm: FallbackAddressArm, pkey: u32) -> usize {
    let mod_n = arm.primary_hi_modulus();
    match arm {
        FallbackAddressArm::TokenContext { k: 1 } => {
            ((pkey as usize) * PRIM_PARTITIONS / mod_n).min(31)
        }
        _ => (((pkey >> 16) as usize) * PRIM_PARTITIONS / mod_n).min(31),
    }
}

#[inline]
fn backoff_part_id(arm: FallbackAddressArm, bk: u16) -> usize {
    let mod_n = arm.backoff_modulus();
    ((usize::from(bk)) * BACK_PARTITIONS / mod_n).min(7)
}

fn append_chunk_to_partitions(
    arm: FallbackAddressArm,
    packed: &[u64],
    prim_writers: &mut [BufWriter<File>],
    back_writers: &mut [BufWriter<File>],
) -> Result<()> {
    let mut sorted = packed.to_vec();
    sorted.par_sort_unstable();

    let has_bk = arm.has_backoff();
    let mut back_raw: Vec<u64> = if has_bk {
        Vec::with_capacity(sorted.len() / 2)
    } else {
        Vec::new()
    };

    let mut idx = 0usize;
    while idx < sorted.len() {
        let key = sorted[idx];
        let mut end = idx + 1;
        while end < sorted.len() && sorted[end] == key {
            end += 1;
        }
        let cnt = u32::try_from(end - idx)?;
        let (pkey, tok) = unpack_query(key);
        let p_id = primary_part_id(arm, pkey);
        prim_writers[p_id].write_all(&key.to_le_bytes())?;
        prim_writers[p_id].write_all(&cnt.to_le_bytes())?;
        if let Some(bk) = arm.derive_bk(pkey) {
            let b_key = (u32::from(bk) << 12) | u32::from(tok);
            back_raw.push((u64::from(b_key) << 32) | u64::from(cnt));
        }
        idx = end;
    }
    drop(sorted);

    if has_bk && !back_raw.is_empty() {
        back_raw.par_sort_unstable_by_key(|&x| x >> 32);
        let mut bi = 0usize;
        while bi < back_raw.len() {
            let b_key = (back_raw[bi] >> 32) as u32;
            let mut sum = (back_raw[bi] & 0xFFFF_FFFF) as u32;
            let mut bj = bi + 1;
            while bj < back_raw.len() && ((back_raw[bj] >> 32) as u32) == b_key {
                sum = sum.saturating_add((back_raw[bj] & 0xFFFF_FFFF) as u32);
                bj += 1;
            }
            let bk = (b_key >> 12) as u16;
            let b_id = backoff_part_id(arm, bk);
            back_writers[b_id].write_all(&b_key.to_le_bytes())?;
            back_writers[b_id].write_all(&sum.to_le_bytes())?;
            bi = bj;
        }
    }
    Ok(())
}

#[derive(Default)]
struct HistSummary {
    row_hist: BTreeMap<u32, usize>,
    ctx_max_hist: BTreeMap<u32, usize>,
}

fn coalesce_primary_partition(path: &Path) -> Result<HistSummary> {
    let meta_len = usize::try_from(fs::metadata(path)?.len())?;
    if meta_len == 0 {
        return Ok(HistSummary::default());
    }
    if meta_len % 12 != 0 {
        return Err(fail("truncated primary partition file"));
    }
    let n_recs = meta_len / 12;
    let mut pairs = Vec::with_capacity(n_recs);
    {
        let mut rd = BufReader::with_capacity(256 << 10, File::open(path)?);
        let mut rec = [0u8; 12];
        for _ in 0..n_recs {
            rd.read_exact(&mut rec)?;
            pairs.push((
                u64::from_le_bytes(rec[..8].try_into()?),
                u32::from_le_bytes(rec[8..12].try_into()?),
            ));
        }
    }
    pairs.sort_unstable_by_key(|&(k, _)| k);

    let mut summary = HistSummary::default();
    let mut out = BufWriter::with_capacity(256 << 10, File::create(path)?);
    let mut last_pk: Option<u32> = None;
    let mut cur_max = 0u32;
    let mut i = 0usize;
    while i < pairs.len() {
        let (k, first_c) = pairs[i];
        let mut sum = first_c;
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == k {
            sum = sum.saturating_add(pairs[j].1);
            j += 1;
        }
        out.write_all(&k.to_le_bytes())?;
        out.write_all(&sum.to_le_bytes())?;
        *summary.row_hist.entry(sum).or_insert(0) += 1;
        let (pk, _) = unpack_query(k);
        if last_pk == Some(pk) {
            if sum > cur_max {
                cur_max = sum;
            }
        } else {
            if last_pk.is_some() {
                *summary.ctx_max_hist.entry(cur_max).or_insert(0) += 1;
            }
            last_pk = Some(pk);
            cur_max = sum;
        }
        i = j;
    }
    if last_pk.is_some() {
        *summary.ctx_max_hist.entry(cur_max).or_insert(0) += 1;
    }
    out.flush()?;
    Ok(summary)
}

fn coalesce_backoff_partition(path: &Path) -> Result<HistSummary> {
    let meta_len = usize::try_from(fs::metadata(path)?.len())?;
    if meta_len == 0 {
        return Ok(HistSummary::default());
    }
    if meta_len % 8 != 0 {
        return Err(fail("truncated backoff partition file"));
    }
    let n_recs = meta_len / 8;
    let mut pairs = Vec::with_capacity(n_recs);
    {
        let mut rd = BufReader::with_capacity(256 << 10, File::open(path)?);
        let mut rec = [0u8; 8];
        for _ in 0..n_recs {
            rd.read_exact(&mut rec)?;
            pairs.push((
                u32::from_le_bytes(rec[..4].try_into()?),
                u32::from_le_bytes(rec[4..8].try_into()?),
            ));
        }
    }
    pairs.sort_unstable_by_key(|&(k, _)| k);

    let mut summary = HistSummary::default();
    let mut out = BufWriter::with_capacity(256 << 10, File::create(path)?);
    let mut last_bk: Option<u16> = None;
    let mut cur_max = 0u32;
    let mut i = 0usize;
    while i < pairs.len() {
        let (k, first_c) = pairs[i];
        let mut sum = first_c;
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == k {
            sum = sum.saturating_add(pairs[j].1);
            j += 1;
        }
        out.write_all(&k.to_le_bytes())?;
        out.write_all(&sum.to_le_bytes())?;
        *summary.row_hist.entry(sum).or_insert(0) += 1;
        let bk = (k >> 12) as u16;
        if last_bk == Some(bk) {
            if sum > cur_max {
                cur_max = sum;
            }
        } else {
            if last_bk.is_some() {
                *summary.ctx_max_hist.entry(cur_max).or_insert(0) += 1;
            }
            last_bk = Some(bk);
            cur_max = sum;
        }
        i = j;
    }
    if last_bk.is_some() {
        *summary.ctx_max_hist.entry(cur_max).or_insert(0) += 1;
    }
    out.flush()?;
    Ok(summary)
}

fn merge_histograms(parts: Vec<HistSummary>) -> (BTreeMap<u32, usize>, BTreeMap<u32, usize>) {
    let mut rows = BTreeMap::new();
    let mut ctxs = BTreeMap::new();
    for part in parts {
        for (k, v) in part.row_hist {
            *rows.entry(k).or_insert(0) += v;
        }
        for (k, v) in part.ctx_max_hist {
            *ctxs.entry(k).or_insert(0) += v;
        }
    }
    (rows, ctxs)
}

struct PruneState {
    has_token_aux: bool,
    max_bytes: usize,
    min_keep: u32,
    boundary_cnt: Option<u32>,
    p_ctxs: usize,
    p_rows: usize,
    b_ctxs: usize,
    b_rows: usize,
    cur_bytes: usize,
}

impl PruneState {
    fn recompute_bytes(&mut self) -> Result<()> {
        self.cur_bytes = encoded_s3t02_bytes(
            self.has_token_aux,
            self.p_ctxs,
            self.p_rows,
            self.b_ctxs,
            self.b_rows,
        )?;
        Ok(())
    }
}

type RowTuple = (u16, u32, i8);
type GroupedRows<K> = Vec<(K, Vec<RowTuple>)>;
type GroupedRefs<'a, K> = Vec<(K, &'a [RowTuple])>;

fn flush_ctx_rows<K: Copy + Ord>(
    key: K,
    ctx_rows: &mut Vec<(u16, u32)>,
    is_primary: bool,
    st: &mut PruneState,
    out: &mut GroupedRows<K>,
) -> Result<()> {
    if ctx_rows.is_empty() {
        return Ok(());
    }
    let mut kept = Vec::with_capacity(ctx_rows.len());
    let mut remaining = ctx_rows.len();
    for &(tok, cnt) in ctx_rows.iter() {
        if st.boundary_cnt == Some(cnt) && st.cur_bytes > st.max_bytes {
            remaining -= 1;
            if is_primary {
                st.p_rows -= 1;
                if remaining == 0 {
                    st.p_ctxs -= 1;
                }
            } else {
                st.b_rows -= 1;
                if remaining == 0 {
                    st.b_ctxs -= 1;
                }
            }
            st.recompute_bytes()?;
        } else {
            kept.push((tok, cnt, 0i8));
        }
    }
    ctx_rows.clear();
    if !kept.is_empty() {
        out.push((key, kept));
    }
    Ok(())
}

pub fn build_partitioned_uors3t02(
    arm_spec: ArmSpec,
    train_sha: [u8; 32],
    trunk: &TrunkTables,
    tokens: &[u16],
    tmp_dir: &Path,
    max_bytes: usize,
    produce_chunk_packed: &mut dyn FnMut(&[u16]) -> Result<Vec<u64>>,
) -> Result<Vec<u8>> {
    fs::create_dir_all(tmp_dir)?;
    let prim_paths: Vec<PathBuf> = (0..PRIM_PARTITIONS)
        .map(|p| tmp_dir.join(format!("{}.prim.{:02}.bin", arm_spec.name, p)))
        .collect();
    let back_paths: Vec<PathBuf> = (0..BACK_PARTITIONS)
        .map(|p| tmp_dir.join(format!("{}.back.{:02}.bin", arm_spec.name, p)))
        .collect();

    let mut prim_writers: Vec<BufWriter<File>> = prim_paths
        .iter()
        .map(|p| -> Result<_> { Ok(BufWriter::with_capacity(256 << 10, File::create(p)?)) })
        .collect::<Result<Vec<_>>>()?;
    let mut back_writers: Vec<BufWriter<File>> = back_paths
        .iter()
        .map(|p| -> Result<_> { Ok(BufWriter::with_capacity(256 << 10, File::create(p)?)) })
        .collect::<Result<Vec<_>>>()?;

    let splits = split_docs(tokens)?;
    let mut doc_idx = 0usize;
    while doc_idx < splits.starts.len() {
        let batch_tok_start = splits.starts[doc_idx] as usize;
        let mut batch_tok_end = batch_tok_start + splits.lens[doc_idx] as usize;
        doc_idx += 1;
        while doc_idx < splits.starts.len() {
            let next_end = splits.starts[doc_idx] as usize + splits.lens[doc_idx] as usize;
            if next_end - batch_tok_start > TARGET_CHUNK_DOCS_TOKENS {
                break;
            }
            batch_tok_end = next_end;
            doc_idx += 1;
        }
        let slice = &tokens[batch_tok_start..batch_tok_end];
        let packed = produce_chunk_packed(slice)?;
        append_chunk_to_partitions(arm_spec.arm, &packed, &mut prim_writers, &mut back_writers)?;
    }
    for w in &mut prim_writers {
        w.flush()?;
    }
    for w in &mut back_writers {
        w.flush()?;
    }
    drop(prim_writers);
    drop(back_writers);

    let mut p_summaries = Vec::with_capacity(PRIM_PARTITIONS);
    for chunk in prim_paths.chunks(PASS1_PAR_BATCH) {
        let res = chunk
            .par_iter()
            .map(|p| coalesce_primary_partition(p))
            .collect::<Result<Vec<_>>>()?;
        p_summaries.extend(res);
    }
    let mut b_summaries = Vec::with_capacity(BACK_PARTITIONS);
    for chunk in back_paths.chunks(PASS1_PAR_BATCH) {
        let res = chunk
            .par_iter()
            .map(|p| coalesce_backoff_partition(p))
            .collect::<Result<Vec<_>>>()?;
        b_summaries.extend(res);
    }

    let (p_row_hist, p_ctx_max_hist) = merge_histograms(p_summaries);
    let (b_row_hist, b_ctx_max_hist) = merge_histograms(b_summaries);

    let token_aux = arm_spec.token_aux_table(trunk);
    let has_token_aux = token_aux.is_some();

    let mut all_counts: Vec<u32> = p_row_hist
        .keys()
        .chain(b_row_hist.keys())
        .copied()
        .collect();
    all_counts.sort_unstable();
    all_counts.dedup();

    let mut pr = p_row_hist.values().sum::<usize>();
    let mut pc = p_ctx_max_hist.values().sum::<usize>();
    let mut br = b_row_hist.values().sum::<usize>();
    let mut bc = b_ctx_max_hist.values().sum::<usize>();
    let mut keep_ge = 1u32;
    let mut boundary_cnt: Option<u32> = None;
    for &c in &all_counts {
        let bytes = encoded_s3t02_bytes(has_token_aux, pc, pr, bc, br)?;
        if bytes <= max_bytes {
            keep_ge = c;
            break;
        }
        boundary_cnt = Some(c);
        pr -= p_row_hist.get(&c).copied().unwrap_or(0);
        pc -= p_ctx_max_hist.get(&c).copied().unwrap_or(0);
        br -= b_row_hist.get(&c).copied().unwrap_or(0);
        bc -= b_ctx_max_hist.get(&c).copied().unwrap_or(0);
        keep_ge = c + 1;
    }
    if encoded_s3t02_bytes(has_token_aux, pc, pr, bc, br)? > max_bytes {
        return Err(fail(format!(
            "even empty UORS3T02 exceeds max_bytes {max_bytes}"
        )));
    }

    let min_keep = boundary_cnt.unwrap_or(keep_ge);
    let sum_ge =
        |hist: &BTreeMap<u32, usize>| -> usize { hist.range(min_keep..).map(|(_, &v)| v).sum() };
    let mut st = PruneState {
        has_token_aux,
        max_bytes,
        min_keep,
        boundary_cnt,
        p_ctxs: sum_ge(&p_ctx_max_hist),
        p_rows: sum_ge(&p_row_hist),
        b_ctxs: sum_ge(&b_ctx_max_hist),
        b_rows: sum_ge(&b_row_hist),
        cur_bytes: 0,
    };
    st.recompute_bytes()?;

    let mut prim_grouped: GroupedRows<u32> = Vec::with_capacity(st.p_ctxs.min(2_000_000));
    let mut cur_pk: Option<u32> = None;
    let mut cur_rows: Vec<(u16, u32)> = Vec::new();
    for path in &prim_paths {
        let len = usize::try_from(fs::metadata(path)?.len())?;
        if len % 12 != 0 {
            return Err(fail("truncated primary partition in pass 2"));
        }
        if len > 0 {
            let mut rd = BufReader::with_capacity(256 << 10, File::open(path)?);
            let mut rec = [0u8; 12];
            for _ in 0..(len / 12) {
                rd.read_exact(&mut rec)?;
                let k = u64::from_le_bytes(rec[..8].try_into()?);
                let cnt = u32::from_le_bytes(rec[8..12].try_into()?);
                if cnt < st.min_keep {
                    continue;
                }
                let (pk, tok) = unpack_query(k);
                if cur_pk != Some(pk) {
                    if let Some(prev) = cur_pk {
                        flush_ctx_rows(prev, &mut cur_rows, true, &mut st, &mut prim_grouped)?;
                    }
                    cur_pk = Some(pk);
                }
                cur_rows.push((tok, cnt));
            }
        }
        let _ = fs::remove_file(path);
    }
    if let Some(prev) = cur_pk {
        flush_ctx_rows(prev, &mut cur_rows, true, &mut st, &mut prim_grouped)?;
    }

    let mut back_grouped: GroupedRows<u16> = Vec::with_capacity(st.b_ctxs.min(57_600));
    let mut cur_bk: Option<u16> = None;
    for path in &back_paths {
        let len = usize::try_from(fs::metadata(path)?.len())?;
        if len % 8 != 0 {
            return Err(fail("truncated backoff partition in pass 2"));
        }
        if len > 0 {
            let mut rd = BufReader::with_capacity(256 << 10, File::open(path)?);
            let mut rec = [0u8; 8];
            for _ in 0..(len / 8) {
                rd.read_exact(&mut rec)?;
                let b_key = u32::from_le_bytes(rec[..4].try_into()?);
                let cnt = u32::from_le_bytes(rec[4..8].try_into()?);
                if cnt < st.min_keep {
                    continue;
                }
                let bk = (b_key >> 12) as u16;
                let tok = (b_key & 0xFFF) as u16;
                if cur_bk != Some(bk) {
                    if let Some(prev) = cur_bk {
                        flush_ctx_rows(prev, &mut cur_rows, false, &mut st, &mut back_grouped)?;
                    }
                    cur_bk = Some(bk);
                }
                cur_rows.push((tok, cnt));
            }
        }
        let _ = fs::remove_file(path);
    }
    if let Some(prev) = cur_bk {
        flush_ctx_rows(prev, &mut cur_rows, false, &mut st, &mut back_grouped)?;
    }

    if st.cur_bytes > max_bytes {
        return Err(fail(format!(
            "internal boundary pruning left {} > {max_bytes}",
            st.cur_bytes
        )));
    }

    let prim_refs: GroupedRefs<'_, u32> = prim_grouped
        .iter()
        .map(|(k, r)| (*k, r.as_slice()))
        .collect();
    let back_refs: GroupedRefs<'_, u16> = back_grouped
        .iter()
        .map(|(k, r)| (*k, r.as_slice()))
        .collect();
    S3FallbackTable::encode_bytes(arm_spec.arm, train_sha, token_aux, &prim_refs, &back_refs)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn build_cpu_partitioned(
    arm_spec: ArmSpec,
    train_sha: [u8; 32],
    trunk: &TrunkTables,
    tokens: &[u16],
    tmp_dir: &Path,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    build_partitioned_uors3t02(
        arm_spec,
        train_sha,
        trunk,
        tokens,
        tmp_dir,
        max_bytes,
        &mut |chunk| Ok(transport_stream_cpu(arm_spec.arm, trunk, chunk, false)?.0),
    )
}

pub fn check_uors3t02_reload(
    bytes: &[u8],
    expected_train_sha: [u8; 32],
    expected_bytes_sha: [u8; 32],
) -> Result<S3FallbackTable> {
    if bytes.len() < 72 || bytes.len() > 100_000_000 {
        return Err(fail(format!(
            "UORS3T02 byte length {} out of <= 100 MB bound",
            bytes.len()
        )));
    }
    let _ = S3_VOCABULARY;
    S3FallbackTable::from_bytes(bytes, expected_train_sha, expected_bytes_sha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{compute_h4_denoms, compute_hopf_denoms};
    use sha2::{Digest, Sha256};

    #[test]
    fn fallback_arms_roundtrip_and_lookup_determinism() {
        let trunk = TrunkTables::from_synthetic_seed(77).unwrap();
        let train_sha = [0x99u8; 32];
        let stream = vec![
            0u16, 5, 6, 7, 5, 6, 7, 1, 0, 5, 6, 8, 9, 10, 1, 0, 5, 6, 7, 12, 13, 1,
        ];
        let tmp = std::env::temp_dir().join("p2-fallback-builder-test");
        let _ = fs::remove_dir_all(&tmp);
        let denoms_hopf = compute_hopf_denoms(&trunk.token_hopf);
        let denoms_h4 = compute_h4_denoms(&trunk.token_h4);

        for spec in ArmSpec::all_fourteen() {
            let full =
                build_cpu_partitioned(spec, train_sha, &trunk, &stream, &tmp, 1 << 20).unwrap();
            let tightened =
                build_cpu_partitioned(spec, train_sha, &trunk, &stream, &tmp, 4096 + 512).unwrap();
            assert!(tightened.len() <= 4096 + 512);
            let sha_f: [u8; 32] = Sha256::digest(&full).into();
            let sha_t: [u8; 32] = Sha256::digest(&tightened).into();
            let tab_f = check_uors3t02_reload(&full, train_sha, sha_f).unwrap();
            let tab_t = check_uors3t02_reload(&tightened, train_sha, sha_t).unwrap();
            let (packed, blks) = transport_stream_cpu(spec.arm, &trunk, &stream, true).unwrap();
            let _ = packed;
            // Verify lookup determinism & miss paths across both full and pruned tables.
            let ba: [i32; 8] = blks[0][..8].try_into().unwrap();
            let bb: [i32; 8] = blks[0][8..].try_into().unwrap();
            let r1 = tab_f
                .lookup_q48(&ba, &bb, &stream[..1], stream[1], &denoms_hopf, &denoms_h4)
                .unwrap();
            let r2 = tab_f
                .lookup_q48(&ba, &bb, &stream[..1], stream[1], &denoms_hopf, &denoms_h4)
                .unwrap();
            assert_eq!(r1, r2);
            let _ = tab_t
                .lookup_q48(&ba, &bb, &[0, 99, 100, 101], 777, &denoms_hopf, &denoms_h4)
                .unwrap();

            // Blocker 1 regression guard: primary/backoff buckets span all partitions evenly.
            let hi_max = spec.arm.primary_hi_modulus();
            let mut seen_p = [false; PRIM_PARTITIONS];
            for hi in 0..hi_max {
                let pkey = match spec.arm {
                    FallbackAddressArm::TokenContext { k: 1 } => hi as u32,
                    _ => (hi as u32) << 16,
                };
                seen_p[primary_part_id(spec.arm, pkey)] = true;
            }
            assert!(
                seen_p.iter().all(|&b| b),
                "arm {} primary partitions did not span all 32 buckets",
                spec.name
            );
            if spec.arm.has_backoff() {
                let bk_max = spec.arm.backoff_modulus();
                let mut seen_b = [false; BACK_PARTITIONS];
                for bk in 0..bk_max {
                    seen_b[backoff_part_id(spec.arm, bk as u16)] = true;
                }
                assert!(
                    seen_b.iter().all(|&b| b),
                    "arm {} backoff partitions did not span all 8 buckets",
                    spec.name
                );
            }
        }
        let _ = fs::remove_dir_all(&tmp);
    }
}
