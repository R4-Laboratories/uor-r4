#![forbid(unsafe_code)]
#![allow(dead_code)]

use crate::{fail, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use uor_r4_count_runtime::p3_state::P3_Q16_ONE;
use uor_r4_count_runtime::s3_state::{S3BackoffContext, S3PrimaryContext, S3RowEntry};

#[inline]
pub fn brier_err_delta_u64(mixed_q48: u64, lr_shift: u32) -> u64 {
    let pq = (mixed_q48 >> 32).min(P3_Q16_ONE);
    (P3_Q16_ONE - pq) >> lr_shift.min(16)
}

pub const S3_VOCAB: usize = 4_096;
pub const S3_Q48_ONE: u64 = 1u64 << 48;
pub const S3_NORM_BUDGET: u64 = S3_Q48_ONE - (S3_VOCAB as u64);
pub const S3_UNIFORM_MASS: u64 = S3_Q48_ONE / (S3_VOCAB as u64);
pub const H4_ROOTS: usize = 120;
const GOLDEN_PRODUCT_KEYS: usize = 9 * 9 * 9 * 9;
const MAGIC_S3_FB: &[u8; 8] = b"UORS3T02";
const S3_HDR_BYTES: usize = 72;
const MAGIC_S2_SKIP: &[u8; 8] = b"UORS2S01";

use uor_r4_integer::h4_classifier::H4_ROOT_COEFFICIENTS;

const fn build_h4_shell_table() -> [[u8; H4_ROOTS]; H4_ROOTS] {
    let mut shells = [[0u8; H4_ROOTS]; H4_ROOTS];
    let mut u = 0usize;
    while u < H4_ROOTS {
        let mut v = 0usize;
        while v < H4_ROOTS {
            let mut sum_a = 0i32;
            let mut sum_b = 0i32;
            let mut d = 0usize;
            while d < 4 {
                let a1 = H4_ROOT_COEFFICIENTS[u][d][0] as i32;
                let b1 = H4_ROOT_COEFFICIENTS[u][d][1] as i32;
                let a2 = H4_ROOT_COEFFICIENTS[v][d][0] as i32;
                let b2 = H4_ROOT_COEFFICIENTS[v][d][1] as i32;
                sum_a += a1 * a2 + b1 * b2;
                sum_b += a1 * b2 + b1 * a2 + b1 * b2;
                d += 1;
            }
            shells[u][v] = match (sum_a, sum_b) {
                (4, 0) => 0u8,
                (0, 2) => 1u8,
                (2, 0) => 2u8,
                (-2, 2) => 3u8,
                (0, 0) => 4u8,
                (2, -2) => 5u8,
                (-2, 0) => 6u8,
                (0, -2) => 7u8,
                (-4, 0) => 8u8,
                _ => panic!("invalid algebraic inner product on 600-cell H4 roots"),
            };
            v += 1;
        }
        u += 1;
    }
    shells
}

pub const H4_SHELL: [[u8; H4_ROOTS]; H4_ROOTS] = build_h4_shell_table();

fn rd_u16(b: &[u8], off: &mut usize) -> Result<u16> {
    let s = b.get(*off..*off + 2).ok_or_else(|| fail("truncated u16"))?;
    *off += 2;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn rd_u32(b: &[u8], off: &mut usize) -> Result<u32> {
    let s = b.get(*off..*off + 4).ok_or_else(|| fail("truncated u32"))?;
    *off += 4;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_u64(b: &[u8], off: &mut usize) -> Result<u64> {
    let s = b.get(*off..*off + 8).ok_or_else(|| fail("truncated u64"))?;
    *off += 8;
    Ok(u64::from_le_bytes(s.try_into()?))
}

#[inline]
pub fn golden_shell_pkey(token_h4: &[u8; S3_VOCAB], prefix: &[u16]) -> u32 {
    if prefix.is_empty() {
        return 0;
    }
    let len = prefix.len();
    let last_tok = prefix[len - 1];
    let clast = usize::from(token_h4[usize::from(last_tok)]);
    let mut s = [0u32; 4];
    for (idx, slot) in s.iter_mut().enumerate() {
        let lag = idx + 1;
        if len > lag {
            let prev = prefix[len - 1 - lag];
            let cp = usize::from(token_h4[usize::from(prev)]);
            *slot = u32::from(H4_SHELL[cp][clast]);
        }
    }
    let pk = ((s[0] * 9 + s[1]) * 9 + s[2]) * 9 + s[3];
    (pk << 16) | u32::from(last_tok)
}

#[inline]
fn starling_7_neighbours(pk: u16) -> ([u16; 7], usize) {
    let mut s = [0i32; 4];
    s[3] = i32::from(pk % 9);
    let mut rem = pk / 9;
    s[2] = i32::from(rem % 9);
    rem /= 9;
    s[1] = i32::from(rem % 9);
    s[0] = i32::from(rem / 9);
    let mut out = [0u16; 7];
    let mut n = 0usize;
    for dim in 0..4 {
        for delta in [-1i32, 1i32] {
            if n >= 7 {
                break;
            }
            let nv = s[dim] + delta;
            if (0..=8).contains(&nv) {
                let mut t = s;
                t[dim] = nv;
                out[n] = (((t[0] * 9 + t[1]) * 9 + t[2]) * 9 + t[3]) as u16;
                n += 1;
            }
        }
    }
    (out, n)
}

pub struct GoldenShellStateTable {
    pub token_h4: [u8; S3_VOCAB],
    pub primary_ctxs: Vec<S3PrimaryContext>,
    pub primary_rows: Vec<S3RowEntry>,
    pub backoff_ctxs: Vec<S3BackoffContext>,
    pub backoff_rows: Vec<S3RowEntry>,
}

impl GoldenShellStateTable {
    pub fn load_checked(
        bytes: &[u8],
        expected_train_sha: [u8; 32],
        expected_bytes_sha: [u8; 32],
    ) -> Result<Self> {
        let actual: [u8; 32] = Sha256::digest(bytes).into();
        if actual != expected_bytes_sha {
            return Err(fail("golden_shell.uors3t02 SHA-256 mismatch"));
        }
        if bytes.len() < S3_HDR_BYTES || &bytes[..8] != MAGIC_S3_FB {
            return Err(fail("invalid UORS3T02 magic"));
        }
        let mut off = 8usize;
        if rd_u32(bytes, &mut off)? != 1 {
            return Err(fail("unsupported UORS3T02 version"));
        }
        let code = bytes[off];
        let param = bytes[off + 1];
        off += 2;
        let flags = rd_u16(bytes, &mut off)?;
        let seed = rd_u64(bytes, &mut off)?;
        if code != 7 || param != 0 || flags != 1 || seed != 0 {
            return Err(fail("expected GoldenShell UORS3T02 arm (7, 0, aux=1, 0)"));
        }
        if bytes[off..off + 32] != expected_train_sha {
            return Err(fail("UORS3T02 train_sha mismatch"));
        }
        off += 32;
        let p_ctx_len = rd_u32(bytes, &mut off)? as usize;
        let p_row_len = rd_u32(bytes, &mut off)? as usize;
        let b_ctx_len = rd_u32(bytes, &mut off)? as usize;
        let b_row_len = rd_u32(bytes, &mut off)? as usize;
        let expected_len = S3_HDR_BYTES
            + S3_VOCAB
            + p_ctx_len * 18
            + p_row_len * 7
            + b_ctx_len * 16
            + b_row_len * 7;
        if bytes.len() != expected_len {
            return Err(fail("golden_shell.uors3t02 file length mismatch"));
        }
        let mut token_h4 = [0u8; S3_VOCAB];
        token_h4.copy_from_slice(&bytes[off..off + S3_VOCAB]);
        off += S3_VOCAB;
        if token_h4.iter().any(|&v| usize::from(v) >= H4_ROOTS) {
            return Err(fail("golden_shell token_h4 out of range"));
        }
        let mut primary_ctxs = Vec::with_capacity(p_ctx_len);
        for _ in 0..p_ctx_len {
            primary_ctxs.push(S3PrimaryContext {
                key: rd_u32(bytes, &mut off)?,
                total: rd_u64(bytes, &mut off)?,
                row_start: rd_u32(bytes, &mut off)?,
                row_len: rd_u16(bytes, &mut off)?,
            });
        }
        let mut primary_rows = Vec::with_capacity(p_row_len);
        for _ in 0..p_row_len {
            let next_token = rd_u16(bytes, &mut off)?;
            let count = rd_u32(bytes, &mut off)?;
            let err_i8 = bytes[off] as i8;
            off += 1;
            primary_rows.push(S3RowEntry {
                next_token,
                count,
                err_i8,
            });
        }
        let mut backoff_ctxs = Vec::with_capacity(b_ctx_len);
        for _ in 0..b_ctx_len {
            backoff_ctxs.push(S3BackoffContext {
                product_key: rd_u16(bytes, &mut off)?,
                total: rd_u64(bytes, &mut off)?,
                row_start: rd_u32(bytes, &mut off)?,
                row_len: rd_u16(bytes, &mut off)?,
            });
        }
        let mut backoff_rows = Vec::with_capacity(b_row_len);
        for _ in 0..b_row_len {
            let next_token = rd_u16(bytes, &mut off)?;
            let count = rd_u32(bytes, &mut off)?;
            let err_i8 = bytes[off] as i8;
            off += 1;
            backoff_rows.push(S3RowEntry {
                next_token,
                count,
                err_i8,
            });
        }
        Ok(Self {
            token_h4,
            primary_ctxs,
            primary_rows,
            backoff_ctxs,
            backoff_rows,
        })
    }

    pub fn lookup_modulated_q48(&self, pkey: u32, target: u16) -> (u64, Option<(bool, usize)>) {
        if let Ok(ci) = self.primary_ctxs.binary_search_by_key(&pkey, |c| c.key) {
            let ctx = &self.primary_ctxs[ci];
            let s = ctx.row_start as usize;
            let e = s + usize::from(ctx.row_len);
            let rows = &self.primary_rows[s..e];
            if let Ok(ri) = rows.binary_search_by_key(&target, |r| r.next_token) {
                let mut eff_total = 0u64;
                let mut eff_hit = 0u64;
                for (idx, r) in rows.iter().enumerate() {
                    let scale = (128i64 + i64::from(r.err_i8)).max(1) as u64;
                    let w = u64::from(r.count) * scale;
                    eff_total += w;
                    if idx == ri {
                        eff_hit = w;
                    }
                }
                let mass = 1u64
                    + ((u128::from(eff_hit) * u128::from(S3_NORM_BUDGET))
                        / u128::from(eff_total.max(1))) as u64;
                return (mass, Some((true, s + ri)));
            }
            return (1u64, None);
        }
        let bk = (pkey >> 16) as u16;
        if let Ok(bi) = self
            .backoff_ctxs
            .binary_search_by_key(&bk, |c| c.product_key)
        {
            let ctx = &self.backoff_ctxs[bi];
            let s = ctx.row_start as usize;
            let e = s + usize::from(ctx.row_len);
            let rows = &self.backoff_rows[s..e];
            if let Ok(ri) = rows.binary_search_by_key(&target, |r| r.next_token) {
                let mut eff_total = 0u64;
                let mut eff_hit = 0u64;
                for (idx, r) in rows.iter().enumerate() {
                    let scale = (128i64 + i64::from(r.err_i8)).max(1) as u64;
                    let w = u64::from(r.count) * scale;
                    eff_total += w;
                    if idx == ri {
                        eff_hit = w;
                    }
                }
                let mass = 1u64
                    + ((u128::from(eff_hit) * u128::from(S3_NORM_BUDGET))
                        / u128::from(eff_total.max(1))) as u64;
                return (mass, Some((false, s + ri)));
            }
            return (1u64, None);
        }
        let (nbrs, nn) = starling_7_neighbours(bk % (GOLDEN_PRODUCT_KEYS as u16));
        let mut sum_eff_total = 0u64;
        let mut sum_eff_hit = 0u64;
        for &nbk in &nbrs[..nn] {
            if let Ok(bi) = self
                .backoff_ctxs
                .binary_search_by_key(&nbk, |c| c.product_key)
            {
                let ctx = &self.backoff_ctxs[bi];
                let s = ctx.row_start as usize;
                let e = s + usize::from(ctx.row_len);
                let rows = &self.backoff_rows[s..e];
                let ri_opt = rows.binary_search_by_key(&target, |r| r.next_token).ok();
                for (idx, r) in rows.iter().enumerate() {
                    let scale = (128i64 + i64::from(r.err_i8)).max(1) as u64;
                    let w = u64::from(r.count) * scale;
                    sum_eff_total += w;
                    if ri_opt == Some(idx) {
                        sum_eff_hit += w;
                    }
                }
            }
        }
        if sum_eff_total > 0 {
            let mass = 1u64
                + ((u128::from(sum_eff_hit) * u128::from(S3_NORM_BUDGET))
                    / u128::from(sum_eff_total)) as u64;
            return (mass, None);
        }
        (S3_UNIFORM_MASS, None)
    }

    pub fn apply_zero_centered_deltas(
        &mut self,
        p_delta: &[u64],
        p_hits: &[u64],
        b_delta: &[u64],
        b_hits: &[u64],
    ) -> usize {
        let mut updated = 0usize;
        for ctx in &self.primary_ctxs {
            let s = ctx.row_start as usize;
            let e = s + usize::from(ctx.row_len);
            let ctx_hits: u64 = p_hits[s..e].iter().sum();
            let ctx_delta_sum: i128 = p_delta[s..e].iter().copied().map(i128::from).sum();
            if ctx_hits == 0 || ctx.total == 0 {
                continue;
            }
            let ch = i128::from(ctx_hits);
            let tot = i128::from(ctx.total);
            for (r, &pd) in self.primary_rows[s..e].iter_mut().zip(&p_delta[s..e]) {
                let target_push = i128::from(pd) / ch;
                let rival_push = (ctx_delta_sum * i128::from(r.count)) / (tot * ch);
                let net = (target_push - rival_push).clamp(-127, 127) as i16;
                let nxt = (i16::from(r.err_i8) + net).clamp(-127, 127) as i8;
                if nxt != r.err_i8 {
                    r.err_i8 = nxt;
                    updated += 1;
                }
            }
        }
        for ctx in &self.backoff_ctxs {
            let s = ctx.row_start as usize;
            let e = s + usize::from(ctx.row_len);
            let ctx_hits: u64 = b_hits[s..e].iter().sum();
            let ctx_delta_sum: i128 = b_delta[s..e].iter().copied().map(i128::from).sum();
            if ctx_hits == 0 || ctx.total == 0 {
                continue;
            }
            let ch = i128::from(ctx_hits);
            let tot = i128::from(ctx.total);
            for (r, &bd) in self.backoff_rows[s..e].iter_mut().zip(&b_delta[s..e]) {
                let target_push = i128::from(bd) / ch;
                let rival_push = (ctx_delta_sum * i128::from(r.count)) / (tot * ch);
                let net = (target_push - rival_push).clamp(-127, 127) as i16;
                let nxt = (i16::from(r.err_i8) + net).clamp(-127, 127) as i8;
                if nxt != r.err_i8 {
                    r.err_i8 = nxt;
                    updated += 1;
                }
            }
        }
        updated
    }

    pub fn p_errs(&self) -> Vec<i8> {
        self.primary_rows.iter().map(|r| r.err_i8).collect()
    }

    pub fn b_errs(&self) -> Vec<i8> {
        self.backoff_rows.iter().map(|r| r.err_i8).collect()
    }

    pub fn set_errs(&mut self, p_errs: &[i8], b_errs: &[i8]) -> Result<()> {
        if p_errs.len() != self.primary_rows.len() || b_errs.len() != self.backoff_rows.len() {
            return Err(fail("GoldenShellStateTable set_errs len mismatch"));
        }
        for (r, &e) in self.primary_rows.iter_mut().zip(p_errs) {
            r.err_i8 = e;
        }
        for (r, &e) in self.backoff_rows.iter_mut().zip(b_errs) {
            r.err_i8 = e;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn upload_gpu(
        &self,
        gpu: &crate::cuda::P4BrierGpuEngine,
    ) -> Result<crate::cuda::GpuS3Table> {
        let p_keys: Vec<u32> = self.primary_ctxs.iter().map(|c| c.key).collect();
        let p_rs: Vec<u32> = self.primary_ctxs.iter().map(|c| c.row_start).collect();
        let p_rl: Vec<u16> = self.primary_ctxs.iter().map(|c| c.row_len).collect();
        let p_tok: Vec<u16> = self.primary_rows.iter().map(|r| r.next_token).collect();
        let p_cnt: Vec<u32> = self.primary_rows.iter().map(|r| r.count).collect();
        let p_err = self.p_errs();

        let b_keys: Vec<u16> = self.backoff_ctxs.iter().map(|c| c.product_key).collect();
        let b_rs: Vec<u32> = self.backoff_ctxs.iter().map(|c| c.row_start).collect();
        let b_rl: Vec<u16> = self.backoff_ctxs.iter().map(|c| c.row_len).collect();
        let b_tok: Vec<u16> = self.backoff_rows.iter().map(|r| r.next_token).collect();
        let b_cnt: Vec<u32> = self.backoff_rows.iter().map(|r| r.count).collect();
        let b_err = self.b_errs();

        gpu.upload_s3(
            &p_keys, &p_rs, &p_rl, &p_tok, &p_cnt, &p_err, &b_keys, &b_rs, &b_rl, &b_tok, &b_cnt,
            &b_err,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub fn apply_epoch_brier_error_writes(
        &mut self,
        pkeys: &[u32],
        targets: &[u16],
        mixed_q48: &[u64],
        lr_shift: u32,
    ) -> usize {
        let mut p_delta = vec![0u64; self.primary_rows.len()];
        let mut b_delta = vec![0u64; self.backoff_rows.len()];
        let mut p_hits = vec![0u64; self.primary_rows.len()];
        let mut b_hits = vec![0u64; self.backoff_rows.len()];
        for i in 0..targets.len() {
            let (_m, cell) = self.lookup_modulated_q48(pkeys[i], targets[i]);
            if let Some((is_prim, ridx)) = cell {
                let d = brier_err_delta_u64(mixed_q48[i], lr_shift);
                if is_prim {
                    p_delta[ridx] += d;
                    p_hits[ridx] += 1;
                } else {
                    b_delta[ridx] += d;
                    b_hits[ridx] += 1;
                }
            }
        }
        self.apply_zero_centered_deltas(&p_delta, &p_hits, &b_delta, &b_hits)
    }
}

/// Injective 64-bit `S2` key (kept for `finite` test parity).
#[inline]
pub fn pack_s2_key(order: u8, prev1: u16, prev2: u16, prev3: u16, prev4: u16) -> u64 {
    let o = u64::from(order) << 48;
    match order {
        1 => o | u64::from(prev1 & 0x0fff),
        2 => o | (u64::from(prev2 & 0x0fff) << 12) | u64::from(prev1 & 0x0fff),
        3 => {
            o | (u64::from(prev3 & 0x0fff) << 24)
                | (u64::from(prev2 & 0x0fff) << 12)
                | u64::from(prev1 & 0x0fff)
        }
        4 => {
            o | (u64::from(prev4 & 0x0fff) << 36)
                | (u64::from(prev3 & 0x0fff) << 24)
                | (u64::from(prev2 & 0x0fff) << 12)
                | u64::from(prev1 & 0x0fff)
        }
        _ => o,
    }
}

/// Injective 64-bit `S2-skip` key for non-contiguous skip-gram conjunction families:
/// `0`: `skip_lag2` `(t_{n-2} -> t_n)`
/// `1`: `skip_3_1`  `((t_{n-3}, t_{n-1}) -> t_n)`
/// `2`: `skip_4_1`  `((t_{n-4}, t_{n-1}) -> t_n)`
#[inline]
pub fn pack_skip_key(family: u8, prev1: u16, prev2: u16, prev3: u16, prev4: u16) -> u64 {
    let f = u64::from(family) << 48;
    match family {
        0 => f | u64::from(prev2 & 0x0fff),
        1 => f | (u64::from(prev3 & 0x0fff) << 12) | u64::from(prev1 & 0x0fff),
        2 => f | (u64::from(prev4 & 0x0fff) << 12) | u64::from(prev1 & 0x0fff),
        _ => f,
    }
}

type CtxFollowerAgg = (u64, u64, Vec<(u16, u32)>);

pub struct S2OrderErrorTable {
    order: u8,
    ctxs: Vec<(u64, u32, u16, u64)>,
    rows: Vec<S3RowEntry>,
}

fn select_bounded_contexts(
    mut aggs: Vec<CtxFollowerAgg>,
    order: u8,
    max_bytes: usize,
) -> Result<S2OrderErrorTable> {
    aggs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut selected: Vec<CtxFollowerAgg> = Vec::new();
    let mut used_bytes = 0usize;
    for (key, total, mut items) in aggs {
        if items.is_empty() || used_bytes + 21 > max_bytes {
            if used_bytes + 21 > max_bytes {
                break;
            }
            continue;
        }
        let max_keep = (max_bytes - used_bytes - 14) / 7;
        if max_keep == 0 {
            break;
        }
        if items.len() > max_keep {
            items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            items.truncate(max_keep);
            items.sort_by_key(|x| x.0);
        }
        used_bytes += 14 + items.len() * 7;
        selected.push((key, total, items));
    }

    selected.sort_by_key(|c| c.0);
    let mut ctxs = Vec::with_capacity(selected.len());
    let mut rows = Vec::new();
    for (key, tot, items) in selected {
        let rs = u32::try_from(rows.len())?;
        let rl = u16::try_from(items.len())?;
        for (tok, cnt) in items {
            rows.push(S3RowEntry {
                next_token: tok,
                count: cnt,
                err_i8: 0,
            });
        }
        ctxs.push((key, rs, rl, tot));
    }
    let table = S2OrderErrorTable { order, ctxs, rows };
    if table.encoded_bytes() > max_bytes {
        return Err(fail("S2OrderErrorTable exceeded max_bytes"));
    }
    Ok(table)
}

impl S2OrderErrorTable {
    pub fn build_order_from_stream(tokens: &[u16], order: u8, max_bytes: usize) -> Result<Self> {
        if !(1..=4).contains(&order) {
            return Err(fail(format!("unsupported S2 order {order}")));
        }
        let mut events: Vec<(u64, u16)> = Vec::with_capacity(tokens.len());
        let mut p1 = 0u16;
        let mut p2 = 0u16;
        let mut p3 = 0u16;
        let mut p4 = 0u16;
        let mut at_record_head = false;
        for (idx, &tok) in tokens.iter().enumerate() {
            if tok == 0 {
                p1 = 0;
                p2 = 0;
                p3 = 0;
                p4 = 0;
                at_record_head = true;
                continue;
            }
            if idx == 0 && !at_record_head {
                p1 = tok;
                continue;
            }
            events.push((pack_s2_key(order, p1, p2, p3, p4), tok));
            p4 = p3;
            p3 = p2;
            p2 = p1;
            p1 = tok;
        }
        events.sort_unstable();

        let mut aggs: Vec<CtxFollowerAgg> = Vec::new();
        let mut i = 0usize;
        while i < events.len() {
            let key = events[i].0;
            let mut items: Vec<(u16, u32)> = Vec::new();
            let mut total = 0u64;
            while i < events.len() && events[i].0 == key {
                let tok = events[i].1;
                let start_t = i;
                while i < events.len() && events[i].0 == key && events[i].1 == tok {
                    i += 1;
                }
                let cnt = u32::try_from(i - start_t).unwrap_or(u32::MAX);
                total += u64::from(cnt);
                items.push((tok, cnt));
            }
            aggs.push((key, total, items));
        }
        drop(events);
        select_bounded_contexts(aggs, order, max_bytes)
    }

    pub fn family(&self) -> u8 {
        self.order
    }

    pub fn ctx_count(&self) -> usize {
        self.ctxs.len()
    }

    pub fn entry_count(&self) -> usize {
        self.rows.len()
    }

    pub fn encoded_bytes(&self) -> usize {
        self.ctxs.len() * 14 + self.rows.len() * 7
    }

    pub fn reset_errs(&mut self) {
        for r in &mut self.rows {
            r.err_i8 = 0;
        }
    }

    pub fn errs(&self) -> Vec<i8> {
        self.rows.iter().map(|r| r.err_i8).collect()
    }

    pub fn set_errs(&mut self, errs: &[i8]) -> Result<()> {
        if errs.len() != self.rows.len() {
            return Err(fail("S2OrderErrorTable set_errs len mismatch"));
        }
        for (r, &e) in self.rows.iter_mut().zip(errs) {
            r.err_i8 = e;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn upload_gpu(
        &self,
        gpu: &crate::cuda::P4BrierGpuEngine,
    ) -> Result<crate::cuda::GpuS2Table> {
        let keys: Vec<u64> = self.ctxs.iter().map(|c| c.0).collect();
        let rs: Vec<u32> = self.ctxs.iter().map(|c| c.1).collect();
        let rl: Vec<u16> = self.ctxs.iter().map(|c| c.2).collect();
        let tok: Vec<u16> = self.rows.iter().map(|r| r.next_token).collect();
        let cnt: Vec<u32> = self.rows.iter().map(|r| r.count).collect();
        let err = self.errs();
        gpu.upload_s2(&keys, &rs, &rl, &tok, &cnt, &err)
    }

    pub fn lookup_modulated_q48(&self, key: u64, target: u16) -> (u64, Option<usize>) {
        if let Ok(ci) = self.ctxs.binary_search_by_key(&key, |c| c.0) {
            let (_, rs, rl, _) = self.ctxs[ci];
            let s = rs as usize;
            let e = s + usize::from(rl);
            let slice = &self.rows[s..e];
            if let Ok(ri) = slice.binary_search_by_key(&target, |r| r.next_token) {
                let mut eff_total = 0u64;
                let mut eff_hit = 0u64;
                for (idx, r) in slice.iter().enumerate() {
                    let scale = (128i64 + i64::from(r.err_i8)).max(1) as u64;
                    let w = u64::from(r.count) * scale;
                    eff_total += w;
                    if idx == ri {
                        eff_hit = w;
                    }
                }
                let mass = 1u64
                    + ((u128::from(eff_hit) * u128::from(S3_NORM_BUDGET))
                        / u128::from(eff_total.max(1))) as u64;
                return (mass, Some(s + ri));
            }
            return (1u64, None);
        }
        (S3_UNIFORM_MASS, None)
    }

    pub fn apply_zero_centered_deltas(&mut self, delta: &[u64], hits: &[u64]) -> usize {
        let mut updated = 0usize;
        for &(_key, rs, rl, tot) in &self.ctxs {
            let s = rs as usize;
            let e = s + usize::from(rl);
            let ctx_hits: u64 = hits[s..e].iter().sum();
            let ctx_delta_sum: i128 = delta[s..e].iter().copied().map(i128::from).sum();
            if ctx_hits == 0 || tot == 0 {
                continue;
            }
            let ch = i128::from(ctx_hits);
            let tot_i = i128::from(tot);
            for (r, &d) in self.rows[s..e].iter_mut().zip(&delta[s..e]) {
                let target_push = i128::from(d) / ch;
                let rival_push = (ctx_delta_sum * i128::from(r.count)) / (tot_i * ch);
                let net = (target_push - rival_push).clamp(-127, 127) as i16;
                let nxt = (i16::from(r.err_i8) + net).clamp(-127, 127) as i8;
                if nxt != r.err_i8 {
                    r.err_i8 = nxt;
                    updated += 1;
                }
            }
        }
        updated
    }

    #[cfg(not(feature = "cuda"))]
    pub fn apply_epoch_brier_error_writes(
        &mut self,
        keys: &[u64],
        targets: &[u16],
        mixed_q48: &[u64],
        lr_shift: u32,
    ) -> usize {
        let mut delta = vec![0u64; self.rows.len()];
        let mut hits = vec![0u64; self.rows.len()];
        for i in 0..targets.len() {
            if let (_, Some(ridx)) = self.lookup_modulated_q48(keys[i], targets[i]) {
                delta[ridx] += brier_err_delta_u64(mixed_q48[i], lr_shift);
                hits[ridx] += 1;
            }
        }
        self.apply_zero_centered_deltas(&delta, &hits)
    }
}

fn compact_followers(list: &mut Vec<(u16, u32)>) {
    if list.len() <= 1 {
        return;
    }
    list.sort_unstable_by_key(|x| x.0);
    let mut write = 0usize;
    let mut read = 0usize;
    while read < list.len() {
        let tok = list[read].0;
        let mut sum = 0u64;
        while read < list.len() && list[read].0 == tok {
            sum += u64::from(list[read].1);
            read += 1;
        }
        list[write] = (tok, sum.min(u64::from(u32::MAX)) as u32);
        write += 1;
    }
    list.truncate(write);
}

fn flush_packed_events(buf: &mut Vec<u32>, followers: &mut [Vec<(u16, u32)>]) {
    if buf.is_empty() {
        return;
    }
    buf.sort_unstable();
    let mut i = 0usize;
    let len = buf.len();
    while i < len {
        let val = buf[i];
        let start = i;
        while i < len && buf[i] == val {
            i += 1;
        }
        let slot = (val >> 12) as usize;
        let tok = (val & 0x0fff) as u16;
        let cnt = u32::try_from(i - start).unwrap_or(u32::MAX);
        let dst = &mut followers[slot];
        dst.push((tok, cnt));
        if dst.len() >= 1024 {
            compact_followers(dst);
        }
    }
    buf.clear();
}

fn read_uort_header(file: &mut File) -> Result<(u64, [u8; 64])> {
    let mut hdr = [0u8; 64];
    file.read_exact(&mut hdr)?;
    if &hdr[..4] != b"UORT" {
        return Err(fail("UORT header missing magic"));
    }
    let count = u64::from_le_bytes(hdr[8..16].try_into()?);
    let file_len = file.metadata()?.len();
    let want = 64u64
        .checked_add(
            count
                .checked_mul(2)
                .ok_or_else(|| fail("UORT token count overflow"))?,
        )
        .ok_or_else(|| fail("UORT byte count overflow"))?;
    if file_len != want {
        return Err(fail(format!(
            "UORT header count {count} expects {want} bytes, got {file_len}"
        )));
    }
    Ok((count, hdr))
}

/// Disjoint-`C` non-contiguous skip-gram conjunction tables (`skip_lag2`, `skip_3_1`, `skip_4_1`).
pub struct S2SkipTable {
    pub c_corpus_sha256: [u8; 32],
    pub families: [S2OrderErrorTable; 3],
}

impl S2SkipTable {
    /// Two-pass streaming builder over any valid `UORT` file (`c.tokens.u16`) using bounded RAM
    /// (~400 MB peak) and exact pair counts for top `K = 220,000` contexts per skip family.
    pub fn build_from_uort_file(uort_path: &Path, budgets: [usize; 3]) -> Result<Self> {
        let mut f1 = File::open(uort_path)?;
        let (tok_count, hdr1) = read_uort_header(&mut f1)?;
        let mut hasher = Sha256::new();
        hasher.update(hdr1);

        // Pass 1: exact lag2 full 4096x4096 counts + exact context totals for skip_3_1 and skip_4_1.
        let mut lag2_counts = vec![0u32; S3_VOCAB * S3_VOCAB];
        let mut tot_3_1 = vec![0u32; S3_VOCAB * S3_VOCAB];
        let mut tot_4_1 = vec![0u32; S3_VOCAB * S3_VOCAB];

        let mut reader1 = BufReader::with_capacity(4 << 20, f1);
        let mut chunk = vec![0u8; 4 << 20];
        let mut g_idx = 0u64;
        let mut p1 = 0u16;
        let mut p2 = 0u16;
        let mut p3 = 0u16;
        let mut p4 = 0u16;
        let mut at_record_head = false;
        let mut remaining_bytes = tok_count * 2;
        while remaining_bytes > 0 {
            let to_read = usize::try_from(remaining_bytes.min(chunk.len() as u64))?;
            let sl = &mut chunk[..to_read];
            reader1.read_exact(sl)?;
            hasher.update(&*sl);
            remaining_bytes -= to_read as u64;
            for pair in sl.chunks_exact(2) {
                let tok = u16::from_le_bytes([pair[0], pair[1]]);
                if tok == 0 {
                    p1 = 0;
                    p2 = 0;
                    p3 = 0;
                    p4 = 0;
                    at_record_head = true;
                    g_idx += 1;
                    continue;
                }
                if g_idx == 0 && !at_record_head {
                    p1 = tok;
                    g_idx += 1;
                    continue;
                }
                let tu = usize::from(tok & 0x0fff);
                let p1u = usize::from(p1 & 0x0fff);
                let p2u = usize::from(p2 & 0x0fff);
                let p3u = usize::from(p3 & 0x0fff);
                let p4u = usize::from(p4 & 0x0fff);
                let idx0 = (p2u << 12) | tu;
                lag2_counts[idx0] = lag2_counts[idx0].saturating_add(1);
                let idx31 = (p3u << 12) | p1u;
                tot_3_1[idx31] = tot_3_1[idx31].saturating_add(1);
                let idx41 = (p4u << 12) | p1u;
                tot_4_1[idx41] = tot_4_1[idx41].saturating_add(1);

                p4 = p3;
                p3 = p2;
                p2 = p1;
                p1 = tok;
                g_idx += 1;
            }
        }
        let c_corpus_sha256: [u8; 32] = hasher.finalize().into();

        // Build family 0 (`skip_lag2`) directly from `lag2_counts`.
        let mut aggs_lag2: Vec<CtxFollowerAgg> = Vec::with_capacity(S3_VOCAB);
        for p2u in 0..S3_VOCAB {
            let base = p2u << 12;
            let mut items = Vec::new();
            let mut tot = 0u64;
            for tu in 0..S3_VOCAB {
                let c = lag2_counts[base + tu];
                if c > 0 {
                    tot += u64::from(c);
                    items.push((tu as u16, c));
                }
            }
            if tot > 0 {
                let k = pack_skip_key(0, 0, p2u as u16, 0, 0);
                aggs_lag2.push((k, tot, items));
            }
        }
        drop(lag2_counts);
        let fam0 = select_bounded_contexts(aggs_lag2, 0, budgets[0])?;

        // Select top-K active contexts (`K <= 220,000 < 2^18`) for skip_3_1 and skip_4_1.
        const TOP_K_CTX: usize = 220_000;
        let select_top_k = |tots: Vec<u32>| -> (Vec<u32>, Vec<(u32, u64)>) {
            let mut ranked: Vec<(u32, u32)> = tots
                .iter()
                .enumerate()
                .filter_map(|(idx, &cnt)| (cnt > 0).then_some((idx as u32, cnt)))
                .collect();
            drop(tots);
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            ranked.truncate(TOP_K_CTX);
            let mut slot_map = vec![u32::MAX; S3_VOCAB * S3_VOCAB];
            let mut slot_meta = Vec::with_capacity(ranked.len());
            for (slot, &(raw_idx, tot)) in ranked.iter().enumerate() {
                slot_map[raw_idx as usize] = slot as u32;
                slot_meta.push((raw_idx, u64::from(tot)));
            }
            (slot_map, slot_meta)
        };
        let (slot_map31, slot_meta31) = select_top_k(tot_3_1);
        let (slot_map41, slot_meta41) = select_top_k(tot_4_1);

        // Pass 2: collect exact follower counts for selected top-K contexts via 64 MB chunked sort-merge.
        let mut f2 = File::open(uort_path)?;
        let (_cnt2, _hdr2) = read_uort_header(&mut f2)?;
        let mut reader2 = BufReader::with_capacity(4 << 20, f2);
        const FLUSH_CAP: usize = 16_777_216;
        let mut buf31: Vec<u32> = Vec::with_capacity(FLUSH_CAP);
        let mut buf41: Vec<u32> = Vec::with_capacity(FLUSH_CAP);
        let mut fol31: Vec<Vec<(u16, u32)>> = vec![Vec::new(); slot_meta31.len()];
        let mut fol41: Vec<Vec<(u16, u32)>> = vec![Vec::new(); slot_meta41.len()];

        let mut g2 = 0u64;
        p1 = 0;
        p2 = 0;
        p3 = 0;
        p4 = 0;
        at_record_head = false;
        let mut rem2 = tok_count * 2;
        while rem2 > 0 {
            let to_read = usize::try_from(rem2.min(chunk.len() as u64))?;
            let sl = &mut chunk[..to_read];
            reader2.read_exact(sl)?;
            rem2 -= to_read as u64;
            for pair in sl.chunks_exact(2) {
                let tok = u16::from_le_bytes([pair[0], pair[1]]);
                if tok == 0 {
                    p1 = 0;
                    p2 = 0;
                    p3 = 0;
                    p4 = 0;
                    at_record_head = true;
                    g2 += 1;
                    continue;
                }
                if g2 == 0 && !at_record_head {
                    p1 = tok;
                    g2 += 1;
                    continue;
                }
                let tu = u32::from(tok & 0x0fff);
                let p1u = usize::from(p1 & 0x0fff);
                let p3u = usize::from(p3 & 0x0fff);
                let p4u = usize::from(p4 & 0x0fff);
                let s31 = slot_map31[(p3u << 12) | p1u];
                if s31 != u32::MAX {
                    buf31.push((s31 << 12) | tu);
                    if buf31.len() == FLUSH_CAP {
                        flush_packed_events(&mut buf31, &mut fol31);
                    }
                }
                let s41 = slot_map41[(p4u << 12) | p1u];
                if s41 != u32::MAX {
                    buf41.push((s41 << 12) | tu);
                    if buf41.len() == FLUSH_CAP {
                        flush_packed_events(&mut buf41, &mut fol41);
                    }
                }
                p4 = p3;
                p3 = p2;
                p2 = p1;
                p1 = tok;
                g2 += 1;
            }
        }
        flush_packed_events(&mut buf31, &mut fol31);
        flush_packed_events(&mut buf41, &mut fol41);
        drop(slot_map31);
        drop(slot_map41);

        let finalize_family = |family: u8,
                               slot_meta: Vec<(u32, u64)>,
                               mut followers: Vec<Vec<(u16, u32)>>,
                               budget: usize|
         -> Result<S2OrderErrorTable> {
            let mut aggs = Vec::with_capacity(slot_meta.len());
            for (slot, (raw_idx, tot)) in slot_meta.into_iter().enumerate() {
                compact_followers(&mut followers[slot]);
                let p_hi = (raw_idx >> 12) as u16;
                let p_lo = (raw_idx & 0x0fff) as u16;
                let key = match family {
                    1 => pack_skip_key(1, p_lo, 0, p_hi, 0),
                    2 => pack_skip_key(2, p_lo, 0, 0, p_hi),
                    _ => return Err(fail("invalid skip family")),
                };
                aggs.push((key, tot, std::mem::take(&mut followers[slot])));
            }
            select_bounded_contexts(aggs, family, budget)
        };
        let fam1 = finalize_family(1, slot_meta31, fol31, budgets[1])?;
        let fam2 = finalize_family(2, slot_meta41, fol41, budgets[2])?;
        Ok(Self {
            c_corpus_sha256,
            families: [fam0, fam1, fam2],
        })
    }

    pub fn encode_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(64 + self.total_encoded_bytes());
        out.extend_from_slice(MAGIC_S2_SKIP);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&self.c_corpus_sha256);
        for (fid, fam) in self.families.iter().enumerate() {
            out.push(fid as u8);
            out.extend_from_slice(&u32::try_from(fam.ctxs.len())?.to_le_bytes());
            out.extend_from_slice(&u32::try_from(fam.rows.len())?.to_le_bytes());
            for &(key, rs, rl, tot) in &fam.ctxs {
                out.extend_from_slice(&key.to_le_bytes());
                out.extend_from_slice(&rs.to_le_bytes());
                out.extend_from_slice(&rl.to_le_bytes());
                out.extend_from_slice(&tot.to_le_bytes());
            }
            for r in &fam.rows {
                out.extend_from_slice(&r.next_token.to_le_bytes());
                out.extend_from_slice(&r.count.to_le_bytes());
                out.push(r.err_i8 as u8);
            }
        }
        let digest: [u8; 32] = Sha256::digest(&out).into();
        out.extend_from_slice(&digest);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8], expected_c_sha: [u8; 32]) -> Result<Self> {
        if bytes.len() < 8 + 4 + 32 + 32 || &bytes[..8] != MAGIC_S2_SKIP {
            return Err(fail("invalid UORS2S01 magic"));
        }
        let payload_len = bytes.len() - 32;
        let actual: [u8; 32] = Sha256::digest(&bytes[..payload_len]).into();
        if actual != bytes[payload_len..] {
            return Err(fail("UORS2S01 trailing SHA-256 mismatch"));
        }
        let payload = &bytes[..payload_len];
        let mut off = 8usize;
        if rd_u32(payload, &mut off)? != 1 {
            return Err(fail("unsupported UORS2S01 version"));
        }
        let mut c_corpus_sha256 = [0u8; 32];
        c_corpus_sha256.copy_from_slice(
            payload
                .get(off..off + 32)
                .ok_or_else(|| fail("UORS2S01 truncated corpus sha"))?,
        );
        off += 32;
        if c_corpus_sha256 != expected_c_sha {
            return Err(fail("UORS2S01 c_corpus_sha256 mismatch"));
        }
        let mut families_vec = Vec::with_capacity(3);
        for expected_fid in 0..3u8 {
            let fid = *payload
                .get(off)
                .ok_or_else(|| fail("UORS2S01 truncated fid"))?;
            off += 1;
            if fid != expected_fid {
                return Err(fail("UORS2S01 family ordering mismatch"));
            }
            let clen = rd_u32(payload, &mut off)? as usize;
            let rlen = rd_u32(payload, &mut off)? as usize;
            let mut ctxs = Vec::with_capacity(clen);
            for _ in 0..clen {
                let k = rd_u64(payload, &mut off)?;
                let rs = rd_u32(payload, &mut off)?;
                let rl = rd_u16(payload, &mut off)?;
                let tot = rd_u64(payload, &mut off)?;
                ctxs.push((k, rs, rl, tot));
            }
            let mut rows = Vec::with_capacity(rlen);
            for _ in 0..rlen {
                let nt = rd_u16(payload, &mut off)?;
                let cnt = rd_u32(payload, &mut off)?;
                let err_i8 = *payload
                    .get(off)
                    .ok_or_else(|| fail("UORS2S01 truncated err_i8"))?
                    as i8;
                off += 1;
                rows.push(S3RowEntry {
                    next_token: nt,
                    count: cnt,
                    err_i8,
                });
            }
            families_vec.push(S2OrderErrorTable {
                order: fid,
                ctxs,
                rows,
            });
        }
        if off != payload.len() {
            return Err(fail("UORS2S01 trailing payload length mismatch"));
        }
        Ok(Self {
            c_corpus_sha256,
            families: families_vec
                .try_into()
                .map_err(|_| fail("UORS2S01 family count mismatch"))?,
        })
    }

    pub fn total_encoded_bytes(&self) -> usize {
        self.families.iter().map(|t| t.encoded_bytes()).sum()
    }

    pub fn per_family_bytes(&self) -> [usize; 3] {
        [
            self.families[0].encoded_bytes(),
            self.families[1].encoded_bytes(),
            self.families[2].encoded_bytes(),
        ]
    }

    pub fn per_family_ctxs(&self) -> [usize; 3] {
        [
            self.families[0].ctx_count(),
            self.families[1].ctx_count(),
            self.families[2].ctx_count(),
        ]
    }

    pub fn per_family_entries(&self) -> [usize; 3] {
        [
            self.families[0].entry_count(),
            self.families[1].entry_count(),
            self.families[2].entry_count(),
        ]
    }

    pub fn reset_errs(&mut self) {
        for f in &mut self.families {
            f.reset_errs();
        }
    }
}

pub fn stream_golden_and_skip_keys(
    tokens: &[u16],
    token_h4: &[u8; S3_VOCAB],
) -> (Vec<u16>, Vec<u32>, [Vec<u64>; 3]) {
    let mut targets = Vec::with_capacity(tokens.len());
    let mut gkeys = Vec::with_capacity(tokens.len());
    let mut k_lag2 = Vec::with_capacity(tokens.len());
    let mut k_3_1 = Vec::with_capacity(tokens.len());
    let mut k_4_1 = Vec::with_capacity(tokens.len());
    let mut start = 0usize;
    for (at, &tok) in tokens.iter().enumerate() {
        if tok == 0 {
            start = at;
            continue;
        }
        if at == 0 {
            continue;
        }
        let win_start = start.max(at.saturating_sub(192));
        let prefix = &tokens[win_start..at];
        gkeys.push(golden_shell_pkey(token_h4, prefix));
        let p1 = if at > start + 1 { tokens[at - 1] } else { 0 };
        let p2 = if at > start + 2 { tokens[at - 2] } else { 0 };
        let p3 = if at > start + 3 { tokens[at - 3] } else { 0 };
        let p4 = if at > start + 4 { tokens[at - 4] } else { 0 };
        k_lag2.push(pack_skip_key(0, p1, p2, p3, p4));
        k_3_1.push(pack_skip_key(1, p1, p2, p3, p4));
        k_4_1.push(pack_skip_key(2, p1, p2, p3, p4));
        targets.push(tok);
    }
    (targets, gkeys, [k_lag2, k_3_1, k_4_1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_skip_table_roundtrip_and_full_vocab_normalization() {
        let toks = [
            0u16, 5, 9, 10, 5, 9, 11, 5, 12, 10, 5, 9, 11, 0, 7, 8, 9, 12,
        ];
        let tmp = std::env::temp_dir().join(format!("p4-skip-test-{}.u16", std::process::id()));
        let mut raw = Vec::with_capacity(64 + toks.len() * 2);
        raw.extend_from_slice(b"UORT\x01\x00\x00\x00");
        raw.extend_from_slice(&(toks.len() as u64).to_le_bytes());
        raw.resize(64, 0u8);
        for &t in &toks {
            raw.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(&tmp, &raw).unwrap();

        let mut table = S2SkipTable::build_from_uort_file(&tmp, [2048, 2048, 2048]).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let bytes = table.encode_bytes().unwrap();
        let reloaded = S2SkipTable::from_bytes(&bytes, table.c_corpus_sha256).unwrap();
        assert_eq!(reloaded.per_family_bytes(), table.per_family_bytes());

        // Check skip keys align across boundaries and sums <= 2^48 on all 4096 vocab tokens.
        // 12 targets in rec 1 (5..11), then BOS=0 at index 13, so target ord 12 is `7` (`p2=0`) and ord 15 is `12` (`p1=9, p3=7`).
        let (_tg, _gk, skeys) = stream_golden_and_skip_keys(&toks, &[0u8; S3_VOCAB]);
        assert_eq!(skeys[0][12], pack_skip_key(0, 0, 0, 0, 0));
        assert_eq!(skeys[1][15], pack_skip_key(1, 9, 8, 7, 0));

        let k_3_1 = pack_skip_key(1, 10, 0, 9, 0); // pattern [9, 5, 10] -> follower 5
        let sum_hit: u128 = (0..S3_VOCAB as u16)
            .map(|v| u128::from(table.families[1].lookup_modulated_q48(k_3_1, v).0))
            .sum();
        assert!(sum_hit <= u128::from(S3_Q48_ONE));
        table.reset_errs();
        assert_eq!(brier_err_delta_u64(S3_Q48_ONE / 8, 7), 448);
    }
}
