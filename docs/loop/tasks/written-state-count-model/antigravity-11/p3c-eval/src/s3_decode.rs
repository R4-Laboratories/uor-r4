#![forbid(unsafe_code)]

use crate::{fail, Result};
use sha2::{Digest, Sha256};
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

    /// Exact lawful Q48 lookup using signed `i8` error-cell modulation `(128 + err_i8)`
    /// normalized so `sum_{v=0..4095} p(v) <= 2^48` in every branch (commits on context hit).
    /// Returns `(mass_q48, opt_row_is_primary_and_idx)`.
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

    #[cfg(feature = "cuda")]
    pub fn p_errs(&self) -> Vec<i8> {
        self.primary_rows.iter().map(|r| r.err_i8).collect()
    }

    #[cfg(feature = "cuda")]
    pub fn b_errs(&self) -> Vec<i8> {
        self.backoff_rows.iter().map(|r| r.err_i8).collect()
    }

    #[cfg(feature = "cuda")]
    pub fn upload_gpu(
        &self,
        gpu: &crate::cuda::P3BrierGpuEngine,
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

/// Injective 64-bit `S2` context key for orders `1..=4` (12-bit tokens, zero hash collisions).
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

/// Bounded pure-integer `S2` `i8` error-cell table for one context order (`1..=4`).
/// Selected in descending `context_total` coverage under `14 * ctxs + 7 * rows <= max_bytes`.
pub struct S2OrderErrorTable {
    order: u8,
    ctxs: Vec<(u64, u32, u16, u64)>, // (injective_u64_key, row_start, row_len, total_count)
    rows: Vec<S3RowEntry>,
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

        // Aggregate per-context `(ctx_key, total_count, vec<(tok, cnt)>)`, then pick top coverage.
        struct CtxAgg {
            key: u64,
            total: u64,
            items: Vec<(u16, u32)>,
        }
        let mut aggs: Vec<CtxAgg> = Vec::new();
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
            aggs.push(CtxAgg { key, total, items });
        }
        drop(events);

        // Highest total coverage first (with deterministic ascending key tie-break).
        aggs.sort_by(|a, b| b.total.cmp(&a.total).then(a.key.cmp(&b.key)));

        type SelectedS2Ctx = (u64, u64, Vec<(u16, u32)>);
        let mut selected: Vec<SelectedS2Ctx> = Vec::new();
        let mut used_bytes = 0usize;
        for mut agg in aggs {
            if used_bytes + 21 > max_bytes {
                break;
            }
            let max_keep = (max_bytes - used_bytes - 14) / 7;
            if max_keep == 0 {
                break;
            }
            if agg.items.len() > max_keep {
                agg.items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                agg.items.truncate(max_keep);
                agg.items.sort_by_key(|x| x.0);
            }
            used_bytes += 14 + agg.items.len() * 7;
            selected.push((agg.key, agg.total, agg.items));
        }

        // Re-sort selected contexts by injective `key` ascending for binary search.
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
        let table = Self { order, ctxs, rows };
        if table.encoded_bytes() > max_bytes {
            return Err(fail("S2OrderErrorTable exceeded max_bytes"));
        }
        Ok(table)
    }

    pub fn order(&self) -> u8 {
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

    #[cfg(feature = "cuda")]
    pub fn errs(&self) -> Vec<i8> {
        self.rows.iter().map(|r| r.err_i8).collect()
    }

    #[cfg(feature = "cuda")]
    pub fn upload_gpu(
        &self,
        gpu: &crate::cuda::P3BrierGpuEngine,
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

    #[cfg(any(test, not(feature = "cuda")))]
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

/// Multi-order bounded `S2` `i8` error-written tables for context orders `1..=4` (`<= 100 MB` total).
pub struct S2MultiOrderErrorTable {
    pub orders: [S2OrderErrorTable; 4],
}

impl S2MultiOrderErrorTable {
    pub fn build_from_r_stream(tokens: &[u16], max_bytes_per_order: usize) -> Result<Self> {
        let o1 = S2OrderErrorTable::build_order_from_stream(tokens, 1, max_bytes_per_order)?;
        let o2 = S2OrderErrorTable::build_order_from_stream(tokens, 2, max_bytes_per_order)?;
        let o3 = S2OrderErrorTable::build_order_from_stream(tokens, 3, max_bytes_per_order)?;
        let o4 = S2OrderErrorTable::build_order_from_stream(tokens, 4, max_bytes_per_order)?;
        Ok(Self {
            orders: [o1, o2, o3, o4],
        })
    }

    pub fn total_encoded_bytes(&self) -> usize {
        self.orders.iter().map(|t| t.encoded_bytes()).sum()
    }

    pub fn per_order_bytes(&self) -> [usize; 4] {
        [
            self.orders[0].encoded_bytes(),
            self.orders[1].encoded_bytes(),
            self.orders[2].encoded_bytes(),
            self.orders[3].encoded_bytes(),
        ]
    }

    pub fn per_order_ctxs(&self) -> [usize; 4] {
        [
            self.orders[0].ctx_count(),
            self.orders[1].ctx_count(),
            self.orders[2].ctx_count(),
            self.orders[3].ctx_count(),
        ]
    }

    pub fn per_order_entries(&self) -> [usize; 4] {
        [
            self.orders[0].entry_count(),
            self.orders[1].entry_count(),
            self.orders[2].entry_count(),
            self.orders[3].entry_count(),
        ]
    }

    pub fn reset_errs(&mut self) {
        for t in &mut self.orders {
            t.reset_errs();
        }
    }
}

pub fn stream_golden_and_multi_s2_keys(
    tokens: &[u16],
    token_h4: &[u8; S3_VOCAB],
) -> (Vec<u16>, Vec<u32>, [Vec<u64>; 4]) {
    let mut targets = Vec::with_capacity(tokens.len());
    let mut gkeys = Vec::with_capacity(tokens.len());
    let mut k1 = Vec::with_capacity(tokens.len());
    let mut k2 = Vec::with_capacity(tokens.len());
    let mut k3 = Vec::with_capacity(tokens.len());
    let mut k4 = Vec::with_capacity(tokens.len());
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
        k1.push(pack_s2_key(1, p1, p2, p3, p4));
        k2.push(pack_s2_key(2, p1, p2, p3, p4));
        k3.push(pack_s2_key(3, p1, p2, p3, p4));
        k4.push(pack_s2_key(4, p1, p2, p3, p4));
        targets.push(tok);
    }
    (targets, gkeys, [k1, k2, k3, k4])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_4096_vocab_distribution_normalized_and_s2_cap_and_boundary_exact() {
        // Construct GoldenShellStateTable with primary, backoff, and disjoint Starling-7 neighbours.
        let p_ctx = S3PrimaryContext {
            key: (100u32 << 16) | 5,
            total: 10,
            row_start: 0,
            row_len: 2,
        };
        let p_rows = vec![
            S3RowEntry {
                next_token: 10,
                count: 7,
                err_i8: 20,
            },
            S3RowEntry {
                next_token: 20,
                count: 3,
                err_i8: -10,
            },
        ];
        let b_ctxs = vec![
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
            S3BackoffContext {
                product_key: 301,
                total: 6,
                row_start: 2,
                row_len: 1,
            },
        ];
        let b_rows = vec![
            S3RowEntry {
                next_token: 30,
                count: 5,
                err_i8: 0,
            },
            S3RowEntry {
                next_token: 40,
                count: 4,
                err_i8: 15,
            },
            S3RowEntry {
                next_token: 41,
                count: 6,
                err_i8: -15,
            },
        ];
        // Note: 300 in base-9 is (0, 3, 6, 3); one Starling neighbour along dim 3 (+1) is 301!
        // Choose query pk = 299 -> coords (0, 3, 6, 2), whose Starling neighbours include 300 (not 299 itself)!
        let gs = GoldenShellStateTable {
            token_h4: [0u8; S3_VOCAB],
            primary_ctxs: vec![p_ctx],
            primary_rows: p_rows,
            backoff_ctxs: b_ctxs,
            backoff_rows: b_rows,
        };
        let sum_vocab = |key: u32| -> u128 {
            (0..S3_VOCAB as u16)
                .map(|v| u128::from(gs.lookup_modulated_q48(key, v).0))
                .sum()
        };
        assert!(sum_vocab((100u32 << 16) | 5) <= u128::from(S3_Q48_ONE));
        assert!(sum_vocab((200u32 << 16) | 99) <= u128::from(S3_Q48_ONE));
        // 300 + 81 (dim 1? 300's neighbour 299 has 300 and 219/381/etc): sum <= 2^48!
        assert!(sum_vocab((299u32 << 16) | 99) <= u128::from(S3_Q48_ONE));
        assert_eq!(sum_vocab((6500u32 << 16) | 99), u128::from(S3_Q48_ONE));

        // Verify S2 exact byte cap <= 64 and multi-order <= 256, descending coverage & full 4096-vocab normalization.
        let tokens = vec![0u16, 5, 9, 10, 5, 9, 11, 0, 7, 8, 9, 12];
        let s2_small = S2OrderErrorTable::build_order_from_stream(&tokens, 2, 64).unwrap();
        assert!(s2_small.encoded_bytes() <= 64);

        let multi = S2MultiOrderErrorTable::build_from_r_stream(&tokens, 256).unwrap();
        assert!(multi.total_encoded_bytes() <= 1024);
        assert!(multi.per_order_bytes().iter().all(|&b| b <= 256));

        let k2_5_9 = pack_s2_key(2, 9, 5, 0, 0);
        let sum_hit: u128 = (0..S3_VOCAB as u16)
            .map(|v| u128::from(multi.orders[1].lookup_modulated_q48(k2_5_9, v).0))
            .sum();
        assert!(sum_hit <= u128::from(S3_Q48_ONE));
        let k2_miss = pack_s2_key(2, 998, 999, 0, 0);
        let sum_miss: u128 = (0..S3_VOCAB as u16)
            .map(|v| u128::from(multi.orders[1].lookup_modulated_q48(k2_miss, v).0))
            .sum();
        assert_eq!(sum_miss, u128::from(S3_Q48_ONE));

        // Verify multi-record boundary key parity across all 4 orders between stream_golden_and_multi_s2_keys and build_from_r_stream:
        let (_tg, _gk, s2keys) = stream_golden_and_multi_s2_keys(&tokens, &[0u8; S3_VOCAB]);
        // Tokens: [0(0), 5(1), 9(2), 10(3), 5(4), 9(5), 11(6), 0(7), 7(8), 8(9), 9(10), 12(11)]
        // Target index 6 is tok=7 after delimiter 0 -> p1..p4 = 0;
        // index 7 is tok=8 -> p1=7, p2..p4=0; index 8 is tok=9 -> p1=8, p2=7;
        // index 9 is tok=12 -> p1=9, p2=8, p3=7, p4=0.
        assert_eq!(s2keys[1][6], pack_s2_key(2, 0, 0, 0, 0));
        assert_eq!(s2keys[1][7], pack_s2_key(2, 7, 0, 0, 0));
        assert_eq!(s2keys[1][8], pack_s2_key(2, 8, 7, 0, 0));
        assert_eq!(s2keys[2][9], pack_s2_key(3, 9, 8, 7, 0));
        assert_eq!(s2keys[3][9], pack_s2_key(4, 9, 8, 7, 0));
        assert_eq!(brier_err_delta_u64(S3_Q48_ONE / 8, 8), 224);
        assert_eq!(brier_err_delta_u64((S3_Q48_ONE * 3) / 4, 8), 64);
    }
}
