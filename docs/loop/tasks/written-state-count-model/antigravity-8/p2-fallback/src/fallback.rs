#![forbid(unsafe_code)]

use crate::{fail, Result};
use sha2::{Digest, Sha256};
use uor_r4_count_runtime::s3_state::{
    S3BackoffContext, S3HitKind, S3LookupResult, S3PrimaryContext, S3RowEntry,
};
pub use uor_r4_count_runtime::s3_state::{S3_Q16_ONE, S3_Q48_ONE, S3_VOCABULARY};
use uor_r4_integer::h4_classifier::H4_ROOT_COEFFICIENTS;

pub const VSA_WINDOW: usize = 7;
pub const VSA_SEED: u64 = 0x5653_415F_4255_4E44; // CANONICAL_TIE_BREAKER_SEED
pub const VSA_PRODUCT_KEYS: usize = 32_768; // 15-bit Q_15 hypercube
pub const HOPF_SECTORS: usize = 96; // 6 faces * 4 quadrants * 4 fiber bins
pub const HOPF_PRODUCT_KEYS: usize = HOPF_SECTORS * HOPF_SECTORS; // 9,216
pub const H4_ROOTS: usize = 120;
pub const H4_PRODUCT_KEYS: usize = H4_ROOTS * H4_ROOTS; // 14,400
pub const GOLDEN_PRODUCT_KEYS: usize = 9 * 9 * 9 * 9; // 6,561
pub const CTRL_PRODUCT_KEYS: usize = 57_600;
pub const STARLING_NEIGHBOURS: usize = 7;
pub const PHI_Q30: i64 = 1_737_381_337; // round(phi * 2^30)

pub const MAGIC_S3_FB: &[u8; 8] = b"UORS3T02";
const S3_VERSION: u32 = 1;
const S3_HDR_BYTES: usize = 72;
const S3_PRIMARY_CTX_BYTES: usize = 18;
const S3_BACKOFF_CTX_BYTES: usize = 16;
const S3_ROW_BYTES: usize = 7;
pub const S3_FLAG_HAS_TOKEN_AUX: u16 = 1;

#[inline]
pub const fn splitmix64_step_const(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn splitmix64_step(state: &mut u64) -> u64 {
    splitmix64_step_const(state)
}

const fn build_vsa_codes_table() -> [u64; S3_VOCABULARY] {
    let mut table = [0u64; S3_VOCABULARY];
    let mut tok = 0usize;
    while tok < S3_VOCABULARY {
        let mut st = VSA_SEED ^ (tok as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        table[tok] = splitmix64_step_const(&mut st);
        tok += 1;
    }
    table
}

/// Canonical 64-bit SplitMix64 token hypervectors matching `Hypervector::<1>::from_seed(VSA_SEED, t)`.
pub static VSA_TOKEN_CODES: [u64; S3_VOCABULARY] = build_vsa_codes_table();

const fn build_vsa_denom_byte_table() -> [[u64; 256]; 8] {
    let codes = build_vsa_codes_table();
    let mut bit_ones = [0u64; 64];
    let mut tok = 0usize;
    while tok < S3_VOCABULARY {
        let code = codes[tok];
        let mut bit = 0usize;
        while bit < 64 {
            if (code >> bit) & 1 == 1 {
                bit_ones[bit] += 1;
            }
            bit += 1;
        }
        tok += 1;
    }
    let mut table = [[0u64; 256]; 8];
    let mut byte_idx = 0usize;
    while byte_idx < 8 {
        let mut val = 0usize;
        while val < 256 {
            let mut sum = if byte_idx == 0 {
                (S3_VOCABULARY as u64) * 1008
            } else {
                0u64
            };
            let mut b = 0usize;
            while b < 8 {
                let bit = byte_idx * 8 + b;
                let ones = bit_ones[bit];
                let matches = if ((val >> b) & 1) == 1 {
                    ones
                } else {
                    (S3_VOCABULARY as u64) - ones
                };
                sum += matches * 1008;
                b += 1;
            }
            table[byte_idx][val] = sum;
            val += 1;
        }
        byte_idx += 1;
    }
    table
}

/// Exact sum over all 4,096 vocab tokens of `(65 - popcnt(b_ctx ^ VSA_TOKEN_CODES[c])) * 1008`
/// decomposed across the 8 bytes of `b_ctx`.
pub const VSA_DENOM_BYTE: [[u64; 256]; 8] = build_vsa_denom_byte_table();

#[inline]
pub fn vsa_denom(b_ctx: u64) -> u64 {
    let mut sum = 0u64;
    let mut byte_idx = 0usize;
    while byte_idx < 8 {
        let b = ((b_ctx >> (byte_idx * 8)) & 0xFF) as usize;
        sum += VSA_DENOM_BYTE[byte_idx][b];
        byte_idx += 1;
    }
    sum
}

/// Exact causal role-bound 64-bit VSA hypervector over up to `VSA_WINDOW` (= 7) prefix tokens.
pub fn vsa_context_hv(prefix: &[u16]) -> u64 {
    if prefix.is_empty() {
        return 0;
    }
    let len = prefix.len();
    let last_tok = prefix[len - 1];
    let h_last = VSA_TOKEN_CODES[usize::from(last_tok)];
    let m = len.min(VSA_WINDOW);
    let mut counts = [0u32; 64];
    for lag in 0..m {
        let tok = prefix[len - 1 - lag];
        let z = VSA_TOKEN_CODES[usize::from(tok)].rotate_left((lag as u32) * 4 + 1) ^ h_last;
        for (b, slot) in counts.iter_mut().enumerate() {
            *slot += ((z >> b) & 1) as u32;
        }
    }
    let thresh = (m as u32) / 2;
    let even = m.is_multiple_of(2);
    let mut b_ctx = 0u64;
    for (b, &cnt) in counts.iter().enumerate() {
        let bit = if cnt > thresh {
            true
        } else if even && cnt == thresh {
            ((VSA_SEED >> b) & 1) == 1
        } else {
            false
        };
        if bit {
            b_ctx |= 1u64 << b;
        }
    }
    b_ctx
}

#[inline]
pub fn vsa_output_pick_q16(b_ctx: u64, target: u16) -> u32 {
    let dh = (b_ctx ^ VSA_TOKEN_CODES[usize::from(target)]).count_ones();
    (65 - dh) * 1008
}

#[inline]
fn normalize_q_i64(q: &[i32; 4]) -> [i64; 4] {
    let mut max_abs = 0u64;
    for &v in q {
        let a = i64::from(v).unsigned_abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let limit = 1u64 << 20;
    let mut shift = 0u32;
    while (max_abs >> shift) > limit {
        shift += 1;
    }
    [
        i64::from(q[0]) >> shift,
        i64::from(q[1]) >> shift,
        i64::from(q[2]) >> shift,
        i64::from(q[3]) >> shift,
    ]
}

/// Scale-invariant integer Hopf fibration sector in `0..96` (6 faces x 4 quadrants x 4 fiber bins).
pub fn hopf_sector_96(q: &[i32; 4]) -> u8 {
    let [a, b, c, d] = normalize_q_i64(q);
    let x = 2 * (a * c + b * d);
    let y = 2 * (b * c - a * d);
    let z = a * a + b * b - c * c - d * d;
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();
    let (face, u, v) = if ax >= ay && ax >= az {
        (if x >= 0 { 0u8 } else { 1u8 }, y, z)
    } else if ay >= ax && ay >= az {
        (if y >= 0 { 2u8 } else { 3u8 }, z, x)
    } else {
        (if z >= 0 { 4u8 } else { 5u8 }, x, y)
    };
    let quad = u8::from(u >= 0) | (u8::from(v >= 0) << 1);
    let fiber = if a != 0 || b != 0 {
        u8::from(a >= 0) | (u8::from(b >= 0) << 1)
    } else {
        u8::from(c >= 0) | (u8::from(d >= 0) << 1)
    };
    face * 16 + quad * 4 + fiber
}

#[inline]
pub fn hopf_pick_score_q16(pred_sector: u8, cand_sector: u8) -> u32 {
    let pf = pred_sector / 16;
    let pq = (pred_sector >> 2) & 3;
    let pfib = pred_sector & 3;
    let cf = cand_sector / 16;
    let cq = (cand_sector >> 2) & 3;
    let cfib = cand_sector & 3;
    let mut sc = 16_384u32;
    if pf == cf {
        sc += 24_576;
    }
    if pq == cq {
        sc += 12_288;
    }
    if pfib == cfib {
        sc += 12_288;
    }
    sc
}

#[inline]
pub fn hopf_starling_7(h0: u8, h1: u8) -> [u16; STARLING_NEIGHBOURS] {
    let flip_u = |s: u8| s ^ 4;
    let flip_v = |s: u8| s ^ 8;
    let fib_inc = |s: u8| (s & !3) | ((s + 1) & 3);
    let fib_dec = |s: u8| (s & !3) | ((s + 3) & 3);
    let k = |a: u8, b: u8| u16::from(a) * (HOPF_SECTORS as u16) + u16::from(b);
    [
        k(flip_u(h0), h1),
        k(flip_v(h0), h1),
        k(fib_inc(h0), h1),
        k(fib_dec(h0), h1),
        k(h0, flip_u(h1)),
        k(h0, flip_v(h1)),
        k(h0, fib_inc(h1)),
    ]
}

const fn build_h4_root_q30_table() -> [[i64; 4]; H4_ROOTS] {
    let mut out = [[0i64; 4]; H4_ROOTS];
    let mut r = 0usize;
    while r < H4_ROOTS {
        let mut d = 0usize;
        while d < 4 {
            let a = H4_ROOT_COEFFICIENTS[r][d][0] as i64;
            let b = H4_ROOT_COEFFICIENTS[r][d][1] as i64;
            out[r][d] = a * (1i64 << 30) + b * PHI_Q30;
            d += 1;
        }
        r += 1;
    }
    out
}

pub const H4_ROOTS_Q30_2X: [[i64; 4]; H4_ROOTS] = build_h4_root_q30_table();

/// Pure-integer exact 120-vertex 600-cell H4 snap using `PHI_Q30` and lowest-index tie breaking.
pub fn h4_snap_120(q: &[i32; 4]) -> u8 {
    let qs = normalize_q_i64(q);
    let mut best_idx = 0usize;
    let mut best_dot = i64::MIN;
    let mut r = 0usize;
    while r < H4_ROOTS {
        let row = &H4_ROOTS_Q30_2X[r];
        let dot = qs[0] * row[0] + qs[1] * row[1] + qs[2] * row[2] + qs[3] * row[3];
        if r == 0 || dot > best_dot {
            best_dot = dot;
            best_idx = r;
        }
        r += 1;
    }
    best_idx as u8
}

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
            let s = match (sum_a, sum_b) {
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
            shells[u][v] = s;
            v += 1;
        }
        u += 1;
    }
    shells
}

/// Exact pairwise 600-cell shell distance (`0..=8`) between any two of the 120 H4 roots.
pub const H4_SHELL: [[u8; H4_ROOTS]; H4_ROOTS] = build_h4_shell_table();

const fn build_h4_adj12_table() -> [[u8; 12]; H4_ROOTS] {
    let mut adj = [[0u8; 12]; H4_ROOTS];
    let mut u = 0usize;
    while u < H4_ROOTS {
        let mut cnt = 0usize;
        let mut v = 0usize;
        while v < H4_ROOTS {
            if H4_SHELL[u][v] == 1 {
                adj[u][cnt] = v as u8;
                cnt += 1;
            }
            v += 1;
        }
        if cnt != 12 {
            panic!("every 600-cell vertex must have degree 12");
        }
        u += 1;
    }
    adj
}

/// Precomputed 12 shell-1 neighbours per vertex of the 600-cell in ascending index order.
pub const H4_ADJ_12: [[u8; 12]; H4_ROOTS] = build_h4_adj12_table();

#[inline]
pub fn h4_pick_score_q16(pred_root: u8, cand_root: u8) -> u32 {
    let shell = u32::from(H4_SHELL[usize::from(pred_root)][usize::from(cand_root)]);
    8_192 + (8 - shell) * 7_168
}

#[inline]
pub fn golden_shell_starling_7(pk: u16) -> ([u16; STARLING_NEIGHBOURS], usize) {
    let mut s = [0i32; 4];
    s[3] = i32::from(pk % 9);
    let mut rem = pk / 9;
    s[2] = i32::from(rem % 9);
    rem /= 9;
    s[1] = i32::from(rem % 9);
    s[0] = i32::from(rem / 9);
    let mut out = [0u16; STARLING_NEIGHBOURS];
    let mut n = 0usize;
    for dim in 0..4 {
        for delta in [-1i32, 1i32] {
            if n >= STARLING_NEIGHBOURS {
                break;
            }
            let nv = s[dim] + delta;
            if (0..=8).contains(&nv) {
                let mut t = s;
                t[dim] = nv;
                let npk = (((t[0] * 9 + t[1]) * 9 + t[2]) * 9 + t[3]) as u16;
                out[n] = npk;
                n += 1;
            }
        }
    }
    (out, n)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackAddressArm {
    TokenContext { k: u8 },
    RandomTransport { seed: u64 },
    SplitMixPrefix { seed: u64 },
    VsaSplitMix,
    HopfSector,
    H4SixHundredCell,
    GoldenShell,
}

impl FallbackAddressArm {
    pub fn encode_parts(self) -> Result<(u8, u8, u64)> {
        match self {
            Self::TokenContext { k } => {
                if !(1..=12).contains(&k) {
                    return Err(fail("TokenContext arm requires k in 1..=12"));
                }
                Ok((1, k, 0))
            }
            Self::RandomTransport { seed } => Ok((2, 0, seed)),
            Self::SplitMixPrefix { seed } => Ok((3, 0, seed)),
            Self::VsaSplitMix => Ok((4, VSA_WINDOW as u8, VSA_SEED)),
            Self::HopfSector => Ok((5, 0, 0)),
            Self::H4SixHundredCell => Ok((6, 0, 0)),
            Self::GoldenShell => Ok((7, 0, 0)),
        }
    }

    pub fn from_parts(code: u8, param: u8, seed: u64) -> Result<Self> {
        match code {
            1 if (1..=12).contains(&param) && seed == 0 => Ok(Self::TokenContext { k: param }),
            2 if param == 0 => Ok(Self::RandomTransport { seed }),
            3 if param == 0 => Ok(Self::SplitMixPrefix { seed }),
            4 if param == VSA_WINDOW as u8 && seed == VSA_SEED => Ok(Self::VsaSplitMix),
            5 if param == 0 && seed == 0 => Ok(Self::HopfSector),
            6 if param == 0 && seed == 0 => Ok(Self::H4SixHundredCell),
            7 if param == 0 && seed == 0 => Ok(Self::GoldenShell),
            _ => Err(fail(format!(
                "invalid UORS3T02 arm header ({code}, {param}, {seed})"
            ))),
        }
    }

    pub fn is_fallback_candidate(self) -> bool {
        matches!(
            self,
            Self::VsaSplitMix | Self::HopfSector | Self::H4SixHundredCell | Self::GoldenShell
        )
    }

    pub fn uses_token_aux(self) -> bool {
        matches!(
            self,
            Self::HopfSector | Self::H4SixHundredCell | Self::GoldenShell
        )
    }

    pub fn has_backoff(self) -> bool {
        !matches!(self, Self::TokenContext { k: 1 })
    }

    pub fn primary_hi_modulus(self) -> usize {
        match self {
            Self::TokenContext { k: 1 } => S3_VOCABULARY,
            Self::TokenContext { k: 2 } => S3_VOCABULARY + 1,
            Self::TokenContext { .. }
            | Self::RandomTransport { .. }
            | Self::SplitMixPrefix { .. } => CTRL_PRODUCT_KEYS,
            Self::VsaSplitMix => VSA_PRODUCT_KEYS,
            Self::HopfSector => HOPF_PRODUCT_KEYS,
            Self::H4SixHundredCell => H4_PRODUCT_KEYS,
            Self::GoldenShell => GOLDEN_PRODUCT_KEYS,
        }
    }

    pub fn backoff_modulus(self) -> usize {
        match self {
            Self::TokenContext { k: 1 } => S3_VOCABULARY,
            Self::TokenContext { .. } => S3_VOCABULARY + 1,
            Self::RandomTransport { .. } | Self::SplitMixPrefix { .. } => CTRL_PRODUCT_KEYS,
            Self::VsaSplitMix => VSA_PRODUCT_KEYS,
            Self::HopfSector => HOPF_PRODUCT_KEYS,
            Self::H4SixHundredCell => H4_PRODUCT_KEYS,
            Self::GoldenShell => GOLDEN_PRODUCT_KEYS,
        }
    }

    pub fn derive_bk(self, pkey: u32) -> Option<u16> {
        match self {
            Self::TokenContext { k: 1 } => None,
            Self::TokenContext { .. } => Some((pkey & 0xFFFF) as u16),
            _ => Some((pkey >> 16) as u16),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallbackLookupResult {
    pub inner: S3LookupResult,
    pub pick_denom: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3FallbackTable {
    arm: FallbackAddressArm,
    train_sha: [u8; 32],
    token_aux: Option<[u8; S3_VOCABULARY]>,
    primary_ctxs: Vec<S3PrimaryContext>,
    primary_rows: Vec<S3RowEntry>,
    backoff_ctxs: Vec<S3BackoffContext>,
    backoff_rows: Vec<S3RowEntry>,
}

pub fn encoded_s3t02_bytes(
    has_token_aux: bool,
    primary_ctxs: usize,
    primary_rows: usize,
    backoff_ctxs: usize,
    backoff_rows: usize,
) -> Result<usize> {
    let tok_b = if has_token_aux { S3_VOCABULARY } else { 0 };
    S3_HDR_BYTES
        .checked_add(tok_b)
        .and_then(|n| n.checked_add(primary_ctxs.checked_mul(S3_PRIMARY_CTX_BYTES)?))
        .and_then(|n| n.checked_add(primary_rows.checked_mul(S3_ROW_BYTES)?))
        .and_then(|n| n.checked_add(backoff_ctxs.checked_mul(S3_BACKOFF_CTX_BYTES)?))
        .and_then(|n| n.checked_add(backoff_rows.checked_mul(S3_ROW_BYTES)?))
        .ok_or_else(|| fail("UORS3T02 byte size overflow"))
}

fn read_u16_le(bytes: &[u8], off: &mut usize) -> Result<u16> {
    let end = off
        .checked_add(2)
        .ok_or_else(|| fail("UORS3T02 offset overflow"))?;
    let s = bytes.get(*off..end).ok_or_else(|| fail("truncated u16"))?;
    *off = end;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32_le(bytes: &[u8], off: &mut usize) -> Result<u32> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| fail("UORS3T02 offset overflow"))?;
    let s = bytes.get(*off..end).ok_or_else(|| fail("truncated u32"))?;
    *off = end;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_u64_le(bytes: &[u8], off: &mut usize) -> Result<u64> {
    let end = off
        .checked_add(8)
        .ok_or_else(|| fail("UORS3T02 offset overflow"))?;
    let s = bytes.get(*off..end).ok_or_else(|| fail("truncated u64"))?;
    *off = end;
    Ok(u64::from_le_bytes(s.try_into()?))
}

#[inline]
fn mul_div_q48(count: u64, total: u64) -> u64 {
    if count == 0 {
        return 1;
    }
    if count >= total {
        return S3_Q48_ONE;
    }
    let m = (u128::from(count) << 48) / u128::from(total);
    u64::try_from(m.max(1).min(u128::from(S3_Q48_ONE))).unwrap_or(S3_Q48_ONE)
}

#[inline]
fn best_tok_in_rows(rows: &[S3RowEntry]) -> Option<u16> {
    let mut best: Option<(u32, u16)> = None;
    for r in rows {
        match best {
            None => best = Some((r.count, r.next_token)),
            Some((bc, bt)) => {
                if r.count > bc || (r.count == bc && r.next_token < bt) {
                    best = Some((r.count, r.next_token));
                }
            }
        }
    }
    best.map(|(_, t)| t)
}

pub type RowTupleSlice<'a> = &'a [(u16, u32, i8)];

impl S3FallbackTable {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn arm(&self) -> FallbackAddressArm {
        self.arm
    }
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn token_aux(&self) -> Option<&[u8; S3_VOCABULARY]> {
        self.token_aux.as_ref()
    }
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn primary_context_count(&self) -> usize {
        self.primary_ctxs.len()
    }
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn primary_row_count(&self) -> usize {
        self.primary_rows.len()
    }
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn backoff_context_count(&self) -> usize {
        self.backoff_ctxs.len()
    }
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn backoff_row_count(&self) -> usize {
        self.backoff_rows.len()
    }

    pub fn encode_bytes(
        arm: FallbackAddressArm,
        train_sha: [u8; 32],
        token_aux: Option<[u8; S3_VOCABULARY]>,
        primary: &[(u32, RowTupleSlice<'_>)],
        backoff: &[(u16, RowTupleSlice<'_>)],
    ) -> Result<Vec<u8>> {
        let (code, param, seed) = arm.encode_parts()?;
        if arm.uses_token_aux() != token_aux.is_some() {
            return Err(fail("UORS3T02 token_aux presence mismatch with arm"));
        }
        let mut p_ctxs = Vec::with_capacity(primary.len());
        let mut p_rows = Vec::new();
        for &(key, rows) in primary {
            let row_start = u32::try_from(p_rows.len())?;
            let row_len = u16::try_from(rows.len())?;
            let mut total = 0u64;
            for &(tok, cnt, _) in rows {
                total = total
                    .checked_add(u64::from(cnt))
                    .ok_or_else(|| fail("total overflow"))?;
                p_rows.push((tok, cnt, 0i8));
            }
            p_ctxs.push((key, total, row_start, row_len));
        }
        let mut b_ctxs = Vec::with_capacity(backoff.len());
        let mut b_rows = Vec::new();
        for &(pk, rows) in backoff {
            let row_start = u32::try_from(b_rows.len())?;
            let row_len = u16::try_from(rows.len())?;
            let mut total = 0u64;
            for &(tok, cnt, _) in rows {
                total = total
                    .checked_add(u64::from(cnt))
                    .ok_or_else(|| fail("total overflow"))?;
                b_rows.push((tok, cnt, 0i8));
            }
            b_ctxs.push((pk, total, row_start, row_len));
        }
        let total_bytes = encoded_s3t02_bytes(
            token_aux.is_some(),
            p_ctxs.len(),
            p_rows.len(),
            b_ctxs.len(),
            b_rows.len(),
        )?;
        let mut out = Vec::with_capacity(total_bytes);
        out.extend_from_slice(MAGIC_S3_FB);
        out.extend_from_slice(&S3_VERSION.to_le_bytes());
        out.push(code);
        out.push(param);
        let flags = if token_aux.is_some() {
            S3_FLAG_HAS_TOKEN_AUX
        } else {
            0
        };
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&seed.to_le_bytes());
        out.extend_from_slice(&train_sha);
        out.extend_from_slice(&u32::try_from(p_ctxs.len())?.to_le_bytes());
        out.extend_from_slice(&u32::try_from(p_rows.len())?.to_le_bytes());
        out.extend_from_slice(&u32::try_from(b_ctxs.len())?.to_le_bytes());
        out.extend_from_slice(&u32::try_from(b_rows.len())?.to_le_bytes());
        if let Some(aux) = &token_aux {
            out.extend_from_slice(aux);
        }
        for (key, total, rs, rl) in p_ctxs {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&total.to_le_bytes());
            out.extend_from_slice(&rs.to_le_bytes());
            out.extend_from_slice(&rl.to_le_bytes());
        }
        for (tok, cnt, e) in p_rows {
            out.extend_from_slice(&tok.to_le_bytes());
            out.extend_from_slice(&cnt.to_le_bytes());
            out.push(e as u8);
        }
        for (pk, total, rs, rl) in b_ctxs {
            out.extend_from_slice(&pk.to_le_bytes());
            out.extend_from_slice(&total.to_le_bytes());
            out.extend_from_slice(&rs.to_le_bytes());
            out.extend_from_slice(&rl.to_le_bytes());
        }
        for (tok, cnt, e) in b_rows {
            out.extend_from_slice(&tok.to_le_bytes());
            out.extend_from_slice(&cnt.to_le_bytes());
            out.push(e as u8);
        }
        if out.len() != total_bytes {
            return Err(fail("UORS3T02 internal size mismatch"));
        }
        Ok(out)
    }

    pub fn from_bytes(
        bytes: &[u8],
        expected_train_sha: [u8; 32],
        expected_bytes_sha: [u8; 32],
    ) -> Result<Self> {
        let actual_sha: [u8; 32] = Sha256::digest(bytes).into();
        if actual_sha != expected_bytes_sha {
            return Err(fail("UORS3T02 byte SHA-256 mismatch"));
        }
        if bytes.len() < S3_HDR_BYTES || &bytes[..8] != MAGIC_S3_FB {
            return Err(fail("invalid UORS3T02 header"));
        }
        let mut off = 8usize;
        let ver = read_u32_le(bytes, &mut off)?;
        if ver != S3_VERSION {
            return Err(fail(format!("unsupported UORS3T02 version {ver}")));
        }
        let code = bytes[off];
        let param = bytes[off + 1];
        off += 2;
        let flags = read_u16_le(bytes, &mut off)?;
        if (flags & !S3_FLAG_HAS_TOKEN_AUX) != 0 {
            return Err(fail("UORS3T02 unknown upper flags bits set"));
        }
        let seed = read_u64_le(bytes, &mut off)?;
        let arm = FallbackAddressArm::from_parts(code, param, seed)?;
        let has_aux = (flags & S3_FLAG_HAS_TOKEN_AUX) != 0;
        if arm.uses_token_aux() != has_aux {
            return Err(fail("UORS3T02 aux flag mismatch"));
        }
        let mut train_sha = [0u8; 32];
        train_sha.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        if train_sha != expected_train_sha {
            return Err(fail("UORS3T02 train_sha mismatch"));
        }
        let p_ctx_len = read_u32_le(bytes, &mut off)? as usize;
        let p_row_len = read_u32_le(bytes, &mut off)? as usize;
        let b_ctx_len = read_u32_le(bytes, &mut off)? as usize;
        let b_row_len = read_u32_le(bytes, &mut off)? as usize;
        let expected_len =
            encoded_s3t02_bytes(has_aux, p_ctx_len, p_row_len, b_ctx_len, b_row_len)?;
        if bytes.len() != expected_len {
            return Err(fail("UORS3T02 file length mismatch"));
        }
        let token_aux = if has_aux {
            let mut arr = [0u8; S3_VOCABULARY];
            arr.copy_from_slice(&bytes[off..off + S3_VOCABULARY]);
            off += S3_VOCABULARY;
            let bound: u8 = match arm {
                FallbackAddressArm::HopfSector => HOPF_SECTORS as u8,
                FallbackAddressArm::H4SixHundredCell | FallbackAddressArm::GoldenShell => {
                    H4_ROOTS as u8
                }
                _ => u8::MAX,
            };
            if arr.iter().any(|&v| v >= bound) {
                return Err(fail("UORS3T02 token_aux entry out of range"));
            }
            Some(arr)
        } else {
            None
        };
        let mut primary_ctxs = Vec::with_capacity(p_ctx_len);
        for _ in 0..p_ctx_len {
            primary_ctxs.push(S3PrimaryContext {
                key: read_u32_le(bytes, &mut off)?,
                total: read_u64_le(bytes, &mut off)?,
                row_start: read_u32_le(bytes, &mut off)?,
                row_len: read_u16_le(bytes, &mut off)?,
            });
        }
        let mut primary_rows = Vec::with_capacity(p_row_len);
        for _ in 0..p_row_len {
            let next_token = read_u16_le(bytes, &mut off)?;
            let count = read_u32_le(bytes, &mut off)?;
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
                product_key: read_u16_le(bytes, &mut off)?,
                total: read_u64_le(bytes, &mut off)?,
                row_start: read_u32_le(bytes, &mut off)?,
                row_len: read_u16_le(bytes, &mut off)?,
            });
        }
        let mut backoff_rows = Vec::with_capacity(b_row_len);
        for _ in 0..b_row_len {
            let next_token = read_u16_le(bytes, &mut off)?;
            let count = read_u32_le(bytes, &mut off)?;
            let err_i8 = bytes[off] as i8;
            off += 1;
            backoff_rows.push(S3RowEntry {
                next_token,
                count,
                err_i8,
            });
        }
        // Structural & ascending order invariant validation matching S3StateTable::from_bytes.
        let mut prev_pk: Option<u32> = None;
        for c in &primary_ctxs {
            if prev_pk.is_some_and(|p| c.key <= p) || c.row_len == 0 || c.total == 0 {
                return Err(fail("UORS3T02 invalid or unsorted primary context"));
            }
            prev_pk = Some(c.key);
            let rs = c.row_start as usize;
            let rl = c.row_len as usize;
            let rows = primary_rows
                .get(rs..rs + rl)
                .ok_or_else(|| fail("UORS3T02 primary row slice OOB"))?;
            let mut sum = 0u64;
            let mut pt: Option<u16> = None;
            for r in rows {
                if usize::from(r.next_token) >= S3_VOCABULARY
                    || r.count == 0
                    || pt.is_some_and(|t| r.next_token <= t)
                {
                    return Err(fail("UORS3T02 invalid or unsorted primary row"));
                }
                pt = Some(r.next_token);
                sum += u64::from(r.count);
            }
            if sum != c.total {
                return Err(fail("UORS3T02 primary context total mismatch"));
            }
        }
        let mut prev_bk: Option<u16> = None;
        for c in &backoff_ctxs {
            if prev_bk.is_some_and(|p| c.product_key <= p) || c.row_len == 0 || c.total == 0 {
                return Err(fail("UORS3T02 invalid or unsorted backoff context"));
            }
            prev_bk = Some(c.product_key);
            let rs = c.row_start as usize;
            let rl = c.row_len as usize;
            let rows = backoff_rows
                .get(rs..rs + rl)
                .ok_or_else(|| fail("UORS3T02 backoff row slice OOB"))?;
            let mut sum = 0u64;
            let mut pt: Option<u16> = None;
            for r in rows {
                if usize::from(r.next_token) >= S3_VOCABULARY
                    || r.count == 0
                    || pt.is_some_and(|t| r.next_token <= t)
                {
                    return Err(fail("UORS3T02 invalid or unsorted backoff row"));
                }
                pt = Some(r.next_token);
                sum += u64::from(r.count);
            }
            if sum != c.total {
                return Err(fail("UORS3T02 backoff context total mismatch"));
            }
        }
        Ok(Self {
            arm,
            train_sha,
            token_aux,
            primary_ctxs,
            primary_rows,
            backoff_ctxs,
            backoff_rows,
        })
    }

    pub fn lookup_q48(
        &self,
        block_a: &[i32; 8],
        block_b: &[i32; 8],
        prefix: &[u16],
        target: u16,
        denoms_hopf: &[u64; HOPF_SECTORS],
        denoms_h4: &[u64; H4_ROOTS],
    ) -> Result<FallbackLookupResult> {
        let last_tok = *prefix.last().unwrap_or(&0);
        let pkey = match self.arm {
            FallbackAddressArm::GoldenShell => {
                let aux = self.token_aux.as_ref().expect("golden token_h4");
                compute_golden_shell_pkey(aux, prefix)
            }
            _ => compute_pkey(self.arm, block_a, block_b, prefix)?,
        };
        let bk_opt = self.arm.derive_bk(pkey);
        let report_pk = bk_opt.unwrap_or((pkey >> 16) as u16);

        let score_pick = |pred: u8| -> (u32, u64) {
            match self.arm {
                FallbackAddressArm::VsaSplitMix => {
                    let b_ctx = vsa_context_hv(prefix);
                    (vsa_output_pick_q16(b_ctx, target), vsa_denom(b_ctx))
                }
                FallbackAddressArm::HopfSector => {
                    let aux = self.token_aux.as_ref().expect("hopf token_aux");
                    let cand = aux[usize::from(target)];
                    (
                        hopf_pick_score_q16(pred, cand),
                        denoms_hopf[usize::from(pred)],
                    )
                }
                FallbackAddressArm::H4SixHundredCell | FallbackAddressArm::GoldenShell => {
                    let aux = self.token_aux.as_ref().expect("h4 token_aux");
                    let cand = aux[usize::from(target)];
                    (h4_pick_score_q16(pred, cand), denoms_h4[usize::from(pred)])
                }
                _ => (
                    S3_Q16_ONE,
                    u64::from(S3_VOCABULARY as u32) * u64::from(S3_Q16_ONE),
                ),
            }
        };

        let pred_from_rows = |rows: &[S3RowEntry], fallback_pred: u8| -> u8 {
            match (&self.token_aux, best_tok_in_rows(rows)) {
                (Some(aux), Some(bt)) => aux[usize::from(bt)],
                _ => fallback_pred,
            }
        };

        let uniform_pred = || -> u8 {
            match self.arm {
                FallbackAddressArm::HopfSector => {
                    let q0: [i32; 4] = [block_a[0], block_a[1], block_a[2], block_a[3]];
                    hopf_sector_96(&q0)
                }
                FallbackAddressArm::H4SixHundredCell => {
                    let q0: [i32; 4] = [block_a[0], block_a[1], block_a[2], block_a[3]];
                    h4_snap_120(&q0)
                }
                FallbackAddressArm::GoldenShell => self
                    .token_aux
                    .as_ref()
                    .map(|a| a[usize::from(last_tok)])
                    .unwrap_or(0),
                _ => 0,
            }
        };

        // 1. Exact Primary
        if let Ok(pidx) = self.primary_ctxs.binary_search_by_key(&pkey, |c| c.key) {
            let c = &self.primary_ctxs[pidx];
            let rows =
                &self.primary_rows[c.row_start as usize..c.row_start as usize + c.row_len as usize];
            let rc = rows
                .binary_search_by_key(&target, |r| r.next_token)
                .map(|i| rows[i].count)
                .unwrap_or(0);
            let pred = match self.arm {
                FallbackAddressArm::GoldenShell => self
                    .token_aux
                    .as_ref()
                    .map(|a| a[usize::from(last_tok)])
                    .unwrap_or(0),
                _ => pred_from_rows(rows, uniform_pred()),
            };
            let (pick_q16, pick_denom) = score_pick(pred);
            return Ok(FallbackLookupResult {
                inner: S3LookupResult {
                    hit_kind: S3HitKind::ExactPrimary,
                    product_key: report_pk,
                    raw_count: u64::from(rc),
                    total_count: c.total,
                    mass_q48: mul_div_q48(u64::from(rc), c.total),
                    predicted_root: pred,
                    output_pick_q16: pick_q16,
                },
                pick_denom,
            });
        }

        // 2. Exact Backoff
        if let Some(bk) = bk_opt {
            if let Ok(bidx) = self
                .backoff_ctxs
                .binary_search_by_key(&bk, |c| c.product_key)
            {
                let c = &self.backoff_ctxs[bidx];
                let rows = &self.backoff_rows
                    [c.row_start as usize..c.row_start as usize + c.row_len as usize];
                let rc = rows
                    .binary_search_by_key(&target, |r| r.next_token)
                    .map(|i| rows[i].count)
                    .unwrap_or(0);
                let pred = match self.arm {
                    FallbackAddressArm::GoldenShell => self
                        .token_aux
                        .as_ref()
                        .map(|a| a[usize::from(last_tok)])
                        .unwrap_or(0),
                    _ => pred_from_rows(rows, uniform_pred()),
                };
                let (pick_q16, pick_denom) = score_pick(pred);
                return Ok(FallbackLookupResult {
                    inner: S3LookupResult {
                        hit_kind: S3HitKind::ExactBackoff,
                        product_key: report_pk,
                        raw_count: u64::from(rc),
                        total_count: c.total,
                        mass_q48: mul_div_q48(u64::from(rc), c.total),
                        predicted_root: pred,
                        output_pick_q16: pick_q16,
                    },
                    pick_denom,
                });
            }
        }

        // 3. Starling 7-neighbour miss (fallback candidates only)
        if self.arm.is_fallback_candidate() {
            let mut nbs = [0u16; STARLING_NEIGHBOURS];
            let nb_len = match self.arm {
                FallbackAddressArm::VsaSplitMix => {
                    let pk = (pkey >> 16) as u16;
                    for (j, slot) in nbs.iter_mut().enumerate() {
                        *slot = pk ^ (1u16 << j);
                    }
                    STARLING_NEIGHBOURS
                }
                FallbackAddressArm::HopfSector => {
                    let pk = (pkey >> 16) as u16;
                    let h0 = (pk / (HOPF_SECTORS as u16)) as u8;
                    let h1 = (pk % (HOPF_SECTORS as u16)) as u8;
                    nbs = hopf_starling_7(h0, h1);
                    STARLING_NEIGHBOURS
                }
                FallbackAddressArm::H4SixHundredCell => {
                    let pk = (pkey >> 16) as u16;
                    let c0 = (pk / (H4_ROOTS as u16)) as u8;
                    let c1 = (pk % (H4_ROOTS as u16)) as u8;
                    for (j, slot) in nbs.iter_mut().enumerate() {
                        *slot = u16::from(H4_ADJ_12[usize::from(c0)][j]) * (H4_ROOTS as u16)
                            + u16::from(c1);
                    }
                    STARLING_NEIGHBOURS
                }
                FallbackAddressArm::GoldenShell => {
                    let (arr, len) = golden_shell_starling_7((pkey >> 16) as u16);
                    nbs = arr;
                    len
                }
                _ => 0,
            };

            let mut sum_count = 0u64;
            let mut sum_total = 0u64;
            let mut best_row: Option<(u32, u16)> = None;
            for &npk in &nbs[..nb_len] {
                if let Ok(bidx) = self
                    .backoff_ctxs
                    .binary_search_by_key(&npk, |c| c.product_key)
                {
                    let c = &self.backoff_ctxs[bidx];
                    sum_total += c.total;
                    let rows = &self.backoff_rows
                        [c.row_start as usize..c.row_start as usize + c.row_len as usize];
                    if let Ok(ri) = rows.binary_search_by_key(&target, |r| r.next_token) {
                        sum_count += u64::from(rows[ri].count);
                    }
                    if self.arm.uses_token_aux()
                        && !matches!(self.arm, FallbackAddressArm::GoldenShell)
                    {
                        for r in rows {
                            match best_row {
                                None => best_row = Some((r.count, r.next_token)),
                                Some((bc, bt)) => {
                                    if r.count > bc || (r.count == bc && r.next_token < bt) {
                                        best_row = Some((r.count, r.next_token));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if sum_total > 0 {
                let pred = match self.arm {
                    FallbackAddressArm::GoldenShell => self
                        .token_aux
                        .as_ref()
                        .map(|a| a[usize::from(last_tok)])
                        .unwrap_or(0),
                    _ => match (&self.token_aux, best_row) {
                        (Some(aux), Some((_, bt))) => aux[usize::from(bt)],
                        _ => uniform_pred(),
                    },
                };
                let (pick_q16, pick_denom) = score_pick(pred);
                return Ok(FallbackLookupResult {
                    inner: S3LookupResult {
                        hit_kind: S3HitKind::StarlingMiss,
                        product_key: report_pk,
                        raw_count: sum_count,
                        total_count: sum_total,
                        mass_q48: mul_div_q48(sum_count, sum_total),
                        predicted_root: pred,
                        output_pick_q16: pick_q16,
                    },
                    pick_denom,
                });
            }
        }

        // 4. Uniform fallback
        let pred = uniform_pred();
        let (pick_q16, pick_denom) = score_pick(pred);
        Ok(FallbackLookupResult {
            inner: S3LookupResult {
                hit_kind: S3HitKind::UniformFallback,
                product_key: report_pk,
                raw_count: 0,
                total_count: S3_VOCABULARY as u64,
                mass_q48: S3_Q48_ONE / (S3_VOCABULARY as u64),
                predicted_root: pred,
                output_pick_q16: pick_q16,
            },
            pick_denom,
        })
    }
}

pub fn compute_pkey(
    arm: FallbackAddressArm,
    block_a: &[i32; 8],
    block_b: &[i32; 8],
    prefix: &[u16],
) -> Result<u32> {
    let last_tok = *prefix.last().unwrap_or(&0);
    if usize::from(last_tok) >= S3_VOCABULARY {
        return Err(fail("prefix last token >= 4096"));
    }
    match arm {
        FallbackAddressArm::RandomTransport { .. } => {
            let s3a = uor_r4_count_runtime::s3_state::S3AddressArm::RandomTransport { seed: 0 };
            Ok(s3a.context_keys(block_a, block_b, prefix)?.0)
        }
        FallbackAddressArm::TokenContext { k } => {
            let s3a = uor_r4_count_runtime::s3_state::S3AddressArm::TokenContext { k };
            Ok(s3a.context_keys(block_a, block_b, prefix)?.0)
        }
        FallbackAddressArm::SplitMixPrefix { seed } => {
            let s3a = uor_r4_count_runtime::s3_state::S3AddressArm::SplitMixPrefix { seed };
            Ok(s3a.context_keys(block_a, block_b, prefix)?.0)
        }
        FallbackAddressArm::VsaSplitMix => {
            let b_ctx = vsa_context_hv(prefix);
            let pk = (b_ctx & 0x7FFF) as u32;
            Ok((pk << 16) | u32::from(last_tok))
        }
        FallbackAddressArm::HopfSector => {
            let q0: [i32; 4] = [block_a[0], block_a[1], block_a[2], block_a[3]];
            let q1: [i32; 4] = [block_b[0], block_b[1], block_b[2], block_b[3]];
            let h0 = u32::from(hopf_sector_96(&q0));
            let h1 = u32::from(hopf_sector_96(&q1));
            let pk = h0 * (HOPF_SECTORS as u32) + h1;
            Ok((pk << 16) | u32::from(last_tok))
        }
        FallbackAddressArm::H4SixHundredCell => {
            let q0: [i32; 4] = [block_a[0], block_a[1], block_a[2], block_a[3]];
            let q1: [i32; 4] = [block_b[0], block_b[1], block_b[2], block_b[3]];
            let c0 = u32::from(h4_snap_120(&q0));
            let c1 = u32::from(h4_snap_120(&q1));
            let pk = c0 * (H4_ROOTS as u32) + c1;
            Ok((pk << 16) | u32::from(last_tok))
        }
        FallbackAddressArm::GoldenShell => {
            let _ = (block_a, block_b);
            Err(fail(
                "GoldenShell requires token_h4 table; use compute_golden_shell_pkey",
            ))
        }
    }
}

#[inline]
pub fn compute_golden_shell_pkey(token_h4: &[u8; S3_VOCABULARY], prefix: &[u16]) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h4_shell_table_exact_600cell_cardinalities() {
        let expected_counts = [1usize, 12, 20, 12, 30, 12, 20, 12, 1];
        for u in 0..H4_ROOTS {
            let mut counts = [0usize; 9];
            for (v, &s) in H4_SHELL[u].iter().enumerate() {
                assert_eq!(s, H4_SHELL[v][u], "asymmetric shell ({u}, {v})");
                counts[usize::from(s)] += 1;
            }
            assert_eq!(counts, expected_counts, "bad shell counts at vertex {u}");
            assert_eq!(H4_ADJ_12[u].len(), 12);
            for &nb in &H4_ADJ_12[u] {
                assert_eq!(H4_SHELL[u][usize::from(nb)], 1);
            }
        }
    }

    #[test]
    fn vsa_denom_byte_table_is_bit_exact() {
        for seed in [
            0u64,
            1,
            0xFFFF_FFFF_FFFF_FFFF,
            0x1234_5678_9ABC_DEF0,
            VSA_SEED,
        ] {
            let direct: u64 = VSA_TOKEN_CODES
                .iter()
                .map(|&code| u64::from((65 - (seed ^ code).count_ones()) * 1008))
                .sum();
            assert_eq!(vsa_denom(seed), direct);
        }
    }
}
