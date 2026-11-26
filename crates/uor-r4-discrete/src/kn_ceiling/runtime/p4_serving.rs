/// Shared pure-integer `P4` serving runtime (`uor-r4.lut-discrete-p4/1`):
/// - Lossless `UORS3C01` compact codec & zero-copy reader for Golden-Shell `S3`
/// - Lossless `UORS2C01` compact codec & zero-copy reader for `S2-skip`
/// - Incremental `O(1)` causal session cache (`CausalSessionCache`, `0` B disk)
/// - `P4State` / `P4Session` over the Stage B5b 22-expert `UORS6B01` mixture
///   with `2,048`-token context, `S4` conjunction vote trace, and `P4StepTrace`.
///
/// `#![forbid(unsafe_code)]`, pure integer Q32/Q48 simplex mixture, zero floats, zero matmul, zero normalized exponential.
pub const P4_VOCAB: usize = 4_096;
pub const P4_MAX_CONTEXT: usize = 2_048;
pub const P4_BASE_BUCKETS: usize = 880;
pub const P4_CACHE_STATES: usize = 3;
pub const P4_TOTAL_BUCKETS: usize = P4_BASE_BUCKETS * P4_CACHE_STATES;
pub const P4_EXPERTS: usize = 22;
pub const P4_Q32_ONE: u64 = 1u64 << 32;
pub const P4_Q48_ONE: u64 = 1u64 << 48;
pub const P4_NORM_BUDGET: u64 = P4_Q48_ONE - (P4_VOCAB as u64);
pub const P4_UNIFORM_MASS: u64 = P4_Q48_ONE / (P4_VOCAB as u64);

pub const H4_ROOTS: usize = 120;
pub const GOLDEN_PRODUCT_KEYS: usize = 9 * 9 * 9 * 9;
pub const MAGIC_UORS3C01: &[u8; 8] = b"UORS3C01";
pub const MAGIC_UORS2C01: &[u8; 8] = b"UORS2C01";
const MAGIC_UORS3T02: &[u8; 8] = b"UORS3T02";
const MAGIC_UORS2S01: &[u8; 8] = b"UORS2S01";
const UORS3T02_HDR_BYTES: usize = 72;
const UORS3C01_VERSION: u32 = 1;
const UORS2C01_VERSION: u32 = 1;

const EVEN_PERMUTATIONS: [[usize; 4]; 12] = [
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

const fn build_h4_root_coefficients() -> [[[i8; 2]; 4]; H4_ROOTS] {
    let mut roots = [[[0i8; 2]; 4]; H4_ROOTS];
    let mut index = 0usize;
    let mut axis = 0usize;
    while axis < 4 {
        roots[index][axis] = [-2, 0];
        roots[index + 1][axis] = [2, 0];
        index += 2;
        axis += 1;
    }
    let mut signs = 0usize;
    while signs < 16 {
        axis = 0;
        while axis < 4 {
            roots[index][axis] = [if signs & (1 << axis) == 0 { -1 } else { 1 }, 0];
            axis += 1;
        }
        signs += 1;
        index += 1;
    }
    let mut permutation = 0usize;
    while permutation < 12 {
        signs = 0;
        while signs < 8 {
            let s1 = if signs & 4 == 0 { -1 } else { 1 };
            let s2 = if signs & 2 == 0 { -1 } else { 1 };
            let s3 = if signs & 1 == 0 { -1 } else { 1 };
            let signed = [[0, 0], [s1, 0], [0, s2], [-s3, s3]];
            axis = 0;
            while axis < 4 {
                roots[index][axis] = signed[EVEN_PERMUTATIONS[permutation][axis]];
                axis += 1;
            }
            index += 1;
            signs += 1;
        }
        permutation += 1;
    }
    roots
}

const H4_ROOTS_COEFFS: [[[i8; 2]; 4]; H4_ROOTS] = build_h4_root_coefficients();

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
                let a1 = H4_ROOTS_COEFFS[u][d][0] as i32;
                let b1 = H4_ROOTS_COEFFS[u][d][1] as i32;
                let a2 = H4_ROOTS_COEFFS[v][d][0] as i32;
                let b2 = H4_ROOTS_COEFFS[v][d][1] as i32;
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

#[inline]
pub fn golden_shell_pkey(token_h4: &[u8; P4_VOCAB], prefix: &[u16]) -> u32 {
    if prefix.is_empty() {
        return 0;
    }
    let len = prefix.len();
    let last_tok = prefix[len - 1];
    let clast = usize::from(token_h4.get(usize::from(last_tok)).copied().unwrap_or(0));
    if clast >= H4_ROOTS {
        return u32::from(last_tok);
    }
    let mut s = [0u32; 4];
    for (idx, slot) in s.iter_mut().enumerate() {
        let lag = idx + 1;
        if len > lag {
            let prev = prefix[len - 1 - lag];
            let cp = usize::from(token_h4.get(usize::from(prev)).copied().unwrap_or(0));
            if cp < H4_ROOTS {
                *slot = u32::from(H4_SHELL[cp][clast]);
            }
        }
    }
    let pk = ((s[0] * 9 + s[1]) * 9 + s[2]) * 9 + s[3];
    (pk << 16) | u32::from(last_tok)
}

#[inline]
pub fn starling_7_neighbours(pk: u16) -> ([u16; 7], usize) {
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

#[inline]
pub fn skip_keys_for_prefix(prefix: &[u16]) -> [u64; 3] {
    let lag = |k: usize| prefix.len().checked_sub(k).map_or(0, |idx| prefix[idx]);
    let p1 = lag(1);
    let p2 = lag(2);
    let p3 = lag(3);
    let p4 = lag(4);
    [
        pack_skip_key(0, p1, p2, p3, p4),
        pack_skip_key(1, p1, p2, p3, p4),
        pack_skip_key(2, p1, p2, p3, p4),
    ]
}

#[inline]
fn rd_u16(b: &[u8], off: &mut usize) -> Result<u16> {
    let s = b
        .get(*off..*off + 2)
        .ok_or_else(|| invalid("truncated u16"))?;
    *off += 2;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

#[inline]
fn rd_u32(b: &[u8], off: &mut usize) -> Result<u32> {
    let s = b
        .get(*off..*off + 4)
        .ok_or_else(|| invalid("truncated u32"))?;
    *off += 4;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

#[inline]
fn rd_u64(b: &[u8], off: &mut usize) -> Result<u64> {
    let s = b
        .get(*off..*off + 8)
        .ok_or_else(|| invalid("truncated u64"))?;
    *off += 8;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

fn encode_compact_row(next_token: u16, count: u32, err_i8: i8, out: &mut Vec<u8>) -> Result<()> {
    if next_token == 0 || usize::from(next_token) >= P4_VOCAB {
        return Err(invalid(format!("invalid P4 row token {next_token}")));
    }
    if count == 0 {
        return Err(invalid("compact row count must be non-zero"));
    }
    let tag: u16 = if (1..=13).contains(&count) {
        (count - 1) as u16
    } else if count <= u32::from(u8::MAX) {
        13
    } else if count <= u32::from(u16::MAX) {
        14
    } else {
        15
    };
    let hdr = (next_token & 0x0FFF) | (tag << 12);
    out.extend_from_slice(&hdr.to_le_bytes());
    match tag {
        0..=12 => {}
        13 => out.push(count as u8),
        14 => out.extend_from_slice(&(count as u16).to_le_bytes()),
        _ => out.extend_from_slice(&count.to_le_bytes()),
    }
    out.push(err_i8 as u8);
    Ok(())
}

#[inline]
fn decode_compact_row(slice: &[u8], pos: &mut usize) -> Result<(u16, u64)> {
    let hdr = rd_u16(slice, pos)?;
    let tok = hdr & 0x0FFF;
    let tag = hdr >> 12;
    let count = match tag {
        0..=12 => u32::from(tag) + 1,
        13 => {
            let v = *slice
                .get(*pos)
                .ok_or_else(|| invalid("truncated compact row u8 count"))?;
            *pos += 1;
            u32::from(v)
        }
        14 => u32::from(rd_u16(slice, pos)?),
        _ => rd_u32(slice, pos)?,
    };
    if count == 0 {
        return Err(invalid("compact row count must be non-zero"));
    }
    let err_i8 = *slice
        .get(*pos)
        .ok_or_else(|| invalid("truncated compact row err_i8"))? as i8;
    *pos += 1;
    let scale = (128i64 + i64::from(err_i8)).max(1) as u64;
    let weight = u64::from(count).saturating_mul(scale);
    Ok((tok, weight))
}

fn validate_compact_ctx_index(
    idx_bytes: &[u8],
    ctx_len: usize,
    key_bytes: usize,
    blob: &[u8],
    what: &str,
) -> Result<()> {
    let stride = key_bytes + 4;
    let need = ctx_len
        .checked_mul(stride)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| invalid(format!("{what} index size overflow")))?;
    if idx_bytes.len() != need {
        return Err(invalid(format!("{what} index length mismatch")));
    }
    let read_key = |b: &[u8], off: usize| -> u32 {
        match key_bytes {
            2 => u32::from(u16::from_le_bytes([b[off], b[off + 1]])),
            3 => u32::from(b[off]) | (u32::from(b[off + 1]) << 8) | (u32::from(b[off + 2]) << 16),
            4 => u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]),
            _ => 0,
        }
    };
    let read_off = |b: &[u8], off: usize| -> usize {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as usize
    };
    if ctx_len == 0 {
        if read_off(idx_bytes, 0) != blob.len() {
            return Err(invalid(format!("{what} empty sentinel mismatch")));
        }
        return Ok(());
    }
    let first_start = read_off(idx_bytes, key_bytes);
    if first_start != 0 {
        return Err(invalid(format!("{what} first byte_start must be 0")));
    }
    let mut prev_key = None;
    let mut prev_start = 0usize;
    for i in 0..ctx_len {
        let base = i * stride;
        let k = read_key(idx_bytes, base);
        let start = read_off(idx_bytes, base + key_bytes);
        if prev_key.is_some_and(|pk| pk >= k) || start < prev_start {
            return Err(invalid(format!("{what} non-monotonic keys or offsets")));
        }
        let next_start = if i + 1 < ctx_len {
            read_off(idx_bytes, (i + 1) * stride + key_bytes)
        } else {
            read_off(idx_bytes, ctx_len * stride)
        };
        if next_start <= start || next_start > blob.len() {
            return Err(invalid(format!(
                "{what} empty or out-of-range context slice"
            )));
        }
        let slice = &blob[start..next_start];
        let mut pos = 0usize;
        let mut prev_tok = 0u16;
        while pos < slice.len() {
            let (tok, _) = decode_compact_row(slice, &mut pos)?;
            if tok == 0 || usize::from(tok) >= P4_VOCAB || tok <= prev_tok {
                return Err(invalid(format!("{what} unsorted or invalid row token")));
            }
            prev_tok = tok;
        }
        prev_key = Some(k);
        prev_start = start;
    }
    let sentinel = read_off(idx_bytes, ctx_len * stride);
    if sentinel != blob.len() {
        return Err(invalid(format!("{what} trailing sentinel != blob length")));
    }
    Ok(())
}

#[inline]
fn bsearch_compact_slice<'a>(
    idx_bytes: &[u8],
    ctx_len: usize,
    key_bytes: usize,
    blob: &'a [u8],
    target_key: u32,
) -> Option<&'a [u8]> {
    if ctx_len == 0 {
        return None;
    }
    let stride = key_bytes + 4;
    let key_at = |i: usize| -> u32 {
        let off = i * stride;
        match key_bytes {
            2 => u32::from(u16::from_le_bytes([idx_bytes[off], idx_bytes[off + 1]])),
            3 => {
                u32::from(idx_bytes[off])
                    | (u32::from(idx_bytes[off + 1]) << 8)
                    | (u32::from(idx_bytes[off + 2]) << 16)
            }
            4 => u32::from_le_bytes([
                idx_bytes[off],
                idx_bytes[off + 1],
                idx_bytes[off + 2],
                idx_bytes[off + 3],
            ]),
            _ => 0,
        }
    };
    let off_at = |i: usize| -> usize {
        let base = if i < ctx_len {
            i * stride + key_bytes
        } else {
            ctx_len * stride
        };
        u32::from_le_bytes([
            idx_bytes[base],
            idx_bytes[base + 1],
            idx_bytes[base + 2],
            idx_bytes[base + 3],
        ]) as usize
    };
    let mut lo = 0usize;
    let mut hi = ctx_len;
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        let mk = key_at(mid);
        if mk == target_key {
            return Some(&blob[off_at(mid)..off_at(mid + 1)]);
        } else if mk < target_key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    None
}

#[inline]
fn push_modulated_slice(slice: &[u8], out: &mut Vec<(u16, u64)>) -> Result<()> {
    out.clear();
    let mut pos = 0usize;
    let mut eff_total = 0u64;
    while pos < slice.len() {
        let (tok, w) = decode_compact_row(slice, &mut pos)?;
        eff_total += w;
        out.push((tok, w));
    }
    let denom = u128::from(eff_total.max(1));
    let budget = u128::from(P4_NORM_BUDGET);
    for entry in out.iter_mut() {
        entry.1 = 1u64 + ((u128::from(entry.1) * budget) / denom) as u64;
    }
    Ok(())
}

#[inline]
fn eval_modulated_target(slice: &[u8], target: u16) -> Result<u64> {
    let mut pos = 0usize;
    let mut eff_total = 0u64;
    let mut eff_hit = 0u64;
    while pos < slice.len() {
        let (tok, w) = decode_compact_row(slice, &mut pos)?;
        eff_total += w;
        if tok == target {
            eff_hit = w;
        }
    }
    if eff_hit == 0 {
        return Ok(1u64);
    }
    Ok(1u64
        + ((u128::from(eff_hit) * u128::from(P4_NORM_BUDGET)) / u128::from(eff_total.max(1)))
            as u64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3HitKind {
    Primary,
    Backoff,
    Starling,
    UniformMiss,
}

impl S3HitKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Backoff => "backoff",
            Self::Starling => "starling",
            Self::UniformMiss => "uniform_miss",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S3LookupMeta {
    pub pkey: u32,
    pub backoff_key: u16,
    pub hit_kind: S3HitKind,
}

/// Zero-copy view over a lossless compact `UORS3C01` Golden-Shell table.
pub struct CompactS3View<'a> {
    pub train_sha256: [u8; 32],
    pub token_h4: &'a [u8; P4_VOCAB],
    p_ctx_len: usize,
    p_idx: &'a [u8],
    p_blob: &'a [u8],
    b_ctx_len: usize,
    b_idx: &'a [u8],
    b_blob: &'a [u8],
    pub bytes_sha256: [u8; 32],
}

impl<'a> CompactS3View<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        // 8 magic + 4 ver + 32 sha + 4096 h4 + 16 lens + 32 trailing sha256
        let min_len = 8 + 4 + 32 + P4_VOCAB + 16 + 32;
        if bytes.len() < min_len || &bytes[..8] != MAGIC_UORS3C01 {
            return Err(invalid("invalid UORS3C01 header or magic"));
        }
        let body_end = bytes.len() - 32;
        let expected_sha: [u8; 32] = Sha256::digest(&bytes[..body_end]).into();
        if expected_sha != bytes[body_end..] {
            return Err(invalid("UORS3C01 trailing SHA-256 mismatch"));
        }
        let mut off = 8usize;
        if rd_u32(bytes, &mut off)? != UORS3C01_VERSION {
            return Err(invalid("unsupported UORS3C01 version"));
        }
        let mut train_sha256 = [0u8; 32];
        train_sha256.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let token_h4: &'a [u8; P4_VOCAB] = bytes[off..off + P4_VOCAB]
            .try_into()
            .map_err(|_| invalid("UORS3C01 token_h4 slice error"))?;
        off += P4_VOCAB;
        if token_h4.iter().any(|&v| usize::from(v) >= H4_ROOTS) {
            return Err(invalid("UORS3C01 token_h4 root out of range"));
        }
        let p_ctx_len = rd_u32(bytes, &mut off)? as usize;
        let p_blob_len = rd_u32(bytes, &mut off)? as usize;
        let b_ctx_len = rd_u32(bytes, &mut off)? as usize;
        let b_blob_len = rd_u32(bytes, &mut off)? as usize;
        let p_idx_len = p_ctx_len
            .checked_mul(8)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| invalid("UORS3C01 p_idx size overflow"))?;
        let b_idx_len = b_ctx_len
            .checked_mul(6)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| invalid("UORS3C01 b_idx size overflow"))?;
        let need_end = off
            .checked_add(p_idx_len)
            .and_then(|n| n.checked_add(p_blob_len))
            .and_then(|n| n.checked_add(b_idx_len))
            .and_then(|n| n.checked_add(b_blob_len))
            .ok_or_else(|| invalid("UORS3C01 section size overflow"))?;
        if need_end != body_end {
            return Err(invalid("UORS3C01 total byte length mismatch"));
        }
        let p_idx = &bytes[off..off + p_idx_len];
        off += p_idx_len;
        let p_blob = &bytes[off..off + p_blob_len];
        off += p_blob_len;
        let b_idx = &bytes[off..off + b_idx_len];
        off += b_idx_len;
        let b_blob = &bytes[off..off + b_blob_len];
        validate_compact_ctx_index(p_idx, p_ctx_len, 4, p_blob, "UORS3C01 primary")?;
        validate_compact_ctx_index(b_idx, b_ctx_len, 2, b_blob, "UORS3C01 backoff")?;
        Ok(Self {
            train_sha256,
            token_h4,
            p_ctx_len,
            p_idx,
            p_blob,
            b_ctx_len,
            b_idx,
            b_blob,
            bytes_sha256: Sha256::digest(bytes).into(),
        })
    }

    pub fn pkey_for_prefix(&self, prefix: &[u16]) -> u32 {
        golden_shell_pkey(self.token_h4, prefix)
    }

    /// Evaluate all active tokens for `prefix` in a single pass, writing `(tok, mass_q48)` into
    /// `sparse_out` and returning `(meta, default_mass_for_unlisted_tokens)`.
    pub fn fill_sparse_masses_q48(
        &self,
        prefix: &[u16],
        sparse_out: &mut Vec<(u16, u64)>,
    ) -> Result<(S3LookupMeta, u64)> {
        sparse_out.clear();
        let pkey = golden_shell_pkey(self.token_h4, prefix);
        let bk = (pkey >> 16) as u16;
        if let Some(slice) = bsearch_compact_slice(self.p_idx, self.p_ctx_len, 4, self.p_blob, pkey)
        {
            push_modulated_slice(slice, sparse_out)?;
            return Ok((
                S3LookupMeta {
                    pkey,
                    backoff_key: bk,
                    hit_kind: S3HitKind::Primary,
                },
                1u64,
            ));
        }
        if let Some(slice) =
            bsearch_compact_slice(self.b_idx, self.b_ctx_len, 2, self.b_blob, u32::from(bk))
        {
            push_modulated_slice(slice, sparse_out)?;
            return Ok((
                S3LookupMeta {
                    pkey,
                    backoff_key: bk,
                    hit_kind: S3HitKind::Backoff,
                },
                1u64,
            ));
        }
        let (nbrs, nn) = starling_7_neighbours(bk % (GOLDEN_PRODUCT_KEYS as u16));
        let mut sum_eff_total = 0u64;
        for &nbk in &nbrs[..nn] {
            if let Some(slice) =
                bsearch_compact_slice(self.b_idx, self.b_ctx_len, 2, self.b_blob, u32::from(nbk))
            {
                let mut pos = 0usize;
                while pos < slice.len() {
                    let (tok, w) = decode_compact_row(slice, &mut pos)?;
                    sum_eff_total += w;
                    sparse_out.push((tok, w));
                }
            }
        }
        if sum_eff_total > 0 {
            sparse_out.sort_unstable_by_key(|x| x.0);
            let mut write = 0usize;
            let mut read = 0usize;
            let budget = u128::from(P4_NORM_BUDGET);
            let denom = u128::from(sum_eff_total);
            while read < sparse_out.len() {
                let (tok, _) = sparse_out[read];
                let mut w_sum = 0u64;
                while let Some(&(t, w)) = sparse_out.get(read) {
                    if t != tok {
                        break;
                    }
                    w_sum += w;
                    read += 1;
                }
                sparse_out[write] = (tok, 1u64 + ((u128::from(w_sum) * budget) / denom) as u64);
                write += 1;
            }
            sparse_out.truncate(write);
            return Ok((
                S3LookupMeta {
                    pkey,
                    backoff_key: bk,
                    hit_kind: S3HitKind::Starling,
                },
                1u64,
            ));
        }
        Ok((
            S3LookupMeta {
                pkey,
                backoff_key: bk,
                hit_kind: S3HitKind::UniformMiss,
            },
            P4_UNIFORM_MASS,
        ))
    }

    pub fn lookup_modulated_q48(&self, pkey: u32, target: u16) -> Result<(u64, S3HitKind)> {
        if let Some(slice) = bsearch_compact_slice(self.p_idx, self.p_ctx_len, 4, self.p_blob, pkey)
        {
            return Ok((eval_modulated_target(slice, target)?, S3HitKind::Primary));
        }
        let bk = (pkey >> 16) as u16;
        if let Some(slice) =
            bsearch_compact_slice(self.b_idx, self.b_ctx_len, 2, self.b_blob, u32::from(bk))
        {
            return Ok((eval_modulated_target(slice, target)?, S3HitKind::Backoff));
        }
        let (nbrs, nn) = starling_7_neighbours(bk % (GOLDEN_PRODUCT_KEYS as u16));
        let mut sum_eff_total = 0u64;
        let mut sum_eff_hit = 0u64;
        for &nbk in &nbrs[..nn] {
            if let Some(slice) =
                bsearch_compact_slice(self.b_idx, self.b_ctx_len, 2, self.b_blob, u32::from(nbk))
            {
                let mut pos = 0usize;
                while pos < slice.len() {
                    let (tok, w) = decode_compact_row(slice, &mut pos)?;
                    sum_eff_total += w;
                    if tok == target {
                        sum_eff_hit += w;
                    }
                }
            }
        }
        if sum_eff_total > 0 {
            let mass = if sum_eff_hit == 0 {
                1u64
            } else {
                1u64 + ((u128::from(sum_eff_hit) * u128::from(P4_NORM_BUDGET))
                    / u128::from(sum_eff_total)) as u64
            };
            return Ok((mass, S3HitKind::Starling));
        }
        Ok((P4_UNIFORM_MASS, S3HitKind::UniformMiss))
    }
}

/// Losslessly encode a frozen `UORS3T02` Golden-Shell table plus optional external `err_i8`
/// slices into compact `UORS3C01` wire format (`~54.4 MB` for full `golden_shell.uors3t02`).
pub fn encode_compact_s3_from_uors3t02(
    uors3t02_bytes: &[u8],
    p_errs: Option<&[i8]>,
    b_errs: Option<&[i8]>,
    expected_train_sha: [u8; 32],
) -> Result<Vec<u8>> {
    if uors3t02_bytes.len() < UORS3T02_HDR_BYTES || &uors3t02_bytes[..8] != MAGIC_UORS3T02 {
        return Err(invalid("invalid UORS3T02 magic"));
    }
    let mut off = 8usize;
    if rd_u32(uors3t02_bytes, &mut off)? != 1 {
        return Err(invalid("unsupported UORS3T02 version"));
    }
    let code = uors3t02_bytes[off];
    let param = uors3t02_bytes[off + 1];
    off += 2;
    let flags = rd_u16(uors3t02_bytes, &mut off)?;
    let seed = rd_u64(uors3t02_bytes, &mut off)?;
    if code != 7 || param != 0 || flags != 1 || seed != 0 {
        return Err(invalid(
            "expected GoldenShell UORS3T02 arm (7, 0, aux=1, 0)",
        ));
    }
    let train_sha = &uors3t02_bytes[off..off + 32];
    if train_sha != expected_train_sha {
        return Err(invalid("UORS3T02 train_sha256 mismatch"));
    }
    off += 32;
    let p_ctx_len = rd_u32(uors3t02_bytes, &mut off)? as usize;
    let p_row_len = rd_u32(uors3t02_bytes, &mut off)? as usize;
    let b_ctx_len = rd_u32(uors3t02_bytes, &mut off)? as usize;
    let b_row_len = rd_u32(uors3t02_bytes, &mut off)? as usize;
    let p_ctx_bytes = p_ctx_len
        .checked_mul(18)
        .ok_or_else(|| invalid("UORS3T02 p_ctx byte length overflow"))?;
    let p_row_bytes = p_row_len
        .checked_mul(7)
        .ok_or_else(|| invalid("UORS3T02 p_row byte length overflow"))?;
    let b_ctx_bytes = b_ctx_len
        .checked_mul(16)
        .ok_or_else(|| invalid("UORS3T02 b_ctx byte length overflow"))?;
    let b_row_bytes = b_row_len
        .checked_mul(7)
        .ok_or_else(|| invalid("UORS3T02 b_row byte length overflow"))?;
    let expected_len = UORS3T02_HDR_BYTES
        .checked_add(P4_VOCAB)
        .and_then(|n| n.checked_add(p_ctx_bytes))
        .and_then(|n| n.checked_add(p_row_bytes))
        .and_then(|n| n.checked_add(b_ctx_bytes))
        .and_then(|n| n.checked_add(b_row_bytes))
        .ok_or_else(|| invalid("UORS3T02 total length overflow"))?;
    if uors3t02_bytes.len() != expected_len {
        return Err(invalid("UORS3T02 file length mismatch"));
    }
    if p_errs.is_some_and(|e| e.len() != p_row_len) || b_errs.is_some_and(|e| e.len() != b_row_len)
    {
        return Err(invalid("UORS3T02 external err_i8 slice length mismatch"));
    }
    let token_h4 = &uors3t02_bytes[off..off + P4_VOCAB];
    off += P4_VOCAB;
    if token_h4.iter().any(|&v| usize::from(v) >= H4_ROOTS) {
        return Err(invalid("UORS3T02 token_h4 out of range"));
    }

    let p_ctxs_raw = &uors3t02_bytes[off..off + p_ctx_bytes];
    off += p_ctx_bytes;
    let p_rows_raw = &uors3t02_bytes[off..off + p_row_bytes];
    off += p_row_bytes;
    let b_ctxs_raw = &uors3t02_bytes[off..off + b_ctx_bytes];
    off += b_ctx_bytes;
    let b_rows_raw = &uors3t02_bytes[off..off + b_row_bytes];

    let mut p_idx = Vec::with_capacity(p_ctx_len.saturating_mul(8).saturating_add(4));
    let mut p_blob = Vec::with_capacity(p_row_len.saturating_mul(4));
    for i in 0..p_ctx_len {
        let mut co = i * 18;
        let key = rd_u32(p_ctxs_raw, &mut co)?;
        let _tot = rd_u64(p_ctxs_raw, &mut co)?;
        let rs = rd_u32(p_ctxs_raw, &mut co)? as usize;
        let rl = rd_u16(p_ctxs_raw, &mut co)? as usize;
        let end = rs
            .checked_add(rl)
            .ok_or_else(|| invalid("UORS3T02 primary row span overflow"))?;
        if end > p_row_len {
            return Err(invalid("UORS3T02 primary row span out of bounds"));
        }
        let byte_start =
            u32::try_from(p_blob.len()).map_err(|_| invalid("UORS3C01 p_blob exceeded u32"))?;
        p_idx.extend_from_slice(&key.to_le_bytes());
        p_idx.extend_from_slice(&byte_start.to_le_bytes());
        for r in rs..end {
            let mut ro = r * 7;
            let nt = rd_u16(p_rows_raw, &mut ro)?;
            let cnt = rd_u32(p_rows_raw, &mut ro)?;
            let err = match p_errs {
                Some(pe) => pe[r],
                None => p_rows_raw[ro] as i8,
            };
            encode_compact_row(nt, cnt, err, &mut p_blob)?;
        }
    }
    let p_blob_len =
        u32::try_from(p_blob.len()).map_err(|_| invalid("UORS3C01 p_blob exceeded u32"))?;
    p_idx.extend_from_slice(&p_blob_len.to_le_bytes());

    let mut b_idx = Vec::with_capacity(b_ctx_len.saturating_mul(6).saturating_add(4));
    let mut b_blob = Vec::with_capacity(b_row_len.saturating_mul(4));
    for i in 0..b_ctx_len {
        let mut co = i * 16;
        let pk = rd_u16(b_ctxs_raw, &mut co)?;
        let _tot = rd_u64(b_ctxs_raw, &mut co)?;
        let rs = rd_u32(b_ctxs_raw, &mut co)? as usize;
        let rl = rd_u16(b_ctxs_raw, &mut co)? as usize;
        let end = rs
            .checked_add(rl)
            .ok_or_else(|| invalid("UORS3T02 backoff row span overflow"))?;
        if end > b_row_len {
            return Err(invalid("UORS3T02 backoff row span out of bounds"));
        }
        let byte_start =
            u32::try_from(b_blob.len()).map_err(|_| invalid("UORS3C01 b_blob exceeded u32"))?;
        b_idx.extend_from_slice(&pk.to_le_bytes());
        b_idx.extend_from_slice(&byte_start.to_le_bytes());
        for r in rs..end {
            let mut ro = r * 7;
            let nt = rd_u16(b_rows_raw, &mut ro)?;
            let cnt = rd_u32(b_rows_raw, &mut ro)?;
            let err = match b_errs {
                Some(be) => be[r],
                None => b_rows_raw[ro] as i8,
            };
            encode_compact_row(nt, cnt, err, &mut b_blob)?;
        }
    }
    let b_blob_len =
        u32::try_from(b_blob.len()).map_err(|_| invalid("UORS3C01 b_blob exceeded u32"))?;
    b_idx.extend_from_slice(&b_blob_len.to_le_bytes());

    let mut out = Vec::with_capacity(
        8 + 4 + 32 + P4_VOCAB + 16 + p_idx.len() + p_blob.len() + b_idx.len() + b_blob.len() + 32,
    );
    out.extend_from_slice(MAGIC_UORS3C01);
    out.extend_from_slice(&UORS3C01_VERSION.to_le_bytes());
    out.extend_from_slice(&expected_train_sha);
    out.extend_from_slice(token_h4);
    out.extend_from_slice(&(p_ctx_len as u32).to_le_bytes());
    out.extend_from_slice(&p_blob_len.to_le_bytes());
    out.extend_from_slice(&(b_ctx_len as u32).to_le_bytes());
    out.extend_from_slice(&b_blob_len.to_le_bytes());
    out.extend_from_slice(&p_idx);
    out.extend_from_slice(&p_blob);
    out.extend_from_slice(&b_idx);
    out.extend_from_slice(&b_blob);
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    let _ = CompactS3View::from_bytes(&out)?;
    Ok(out)
}

struct CompactS2SkipFamily<'a> {
    ctx_len: usize,
    idx: &'a [u8],
    blob: &'a [u8],
}

/// Zero-copy view over a lossless compact `UORS2C01` `S2-skip` table.
pub struct CompactS2SkipView<'a> {
    pub c_corpus_sha256: [u8; 32],
    families: [CompactS2SkipFamily<'a>; 3],
    pub bytes_sha256: [u8; 32],
}

impl<'a> CompactS2SkipView<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let min_len = 8 + 4 + 32 + 3 * (1 + 8 + 4) + 32;
        if bytes.len() < min_len || &bytes[..8] != MAGIC_UORS2C01 {
            return Err(invalid("invalid UORS2C01 header or magic"));
        }
        let body_end = bytes.len() - 32;
        let expected_sha: [u8; 32] = Sha256::digest(&bytes[..body_end]).into();
        if expected_sha != bytes[body_end..] {
            return Err(invalid("UORS2C01 trailing SHA-256 mismatch"));
        }
        let mut off = 8usize;
        if rd_u32(bytes, &mut off)? != UORS2C01_VERSION {
            return Err(invalid("unsupported UORS2C01 version"));
        }
        let mut c_corpus_sha256 = [0u8; 32];
        c_corpus_sha256.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let mut fams = Vec::with_capacity(3);
        for expected_fid in 0..3u8 {
            let fid = *bytes
                .get(off)
                .ok_or_else(|| invalid("UORS2C01 truncated fid"))?;
            off += 1;
            if fid != expected_fid {
                return Err(invalid("UORS2C01 family id order mismatch"));
            }
            let ctx_len = rd_u32(bytes, &mut off)? as usize;
            let blob_len = rd_u32(bytes, &mut off)? as usize;
            let idx_len = ctx_len
                .checked_mul(7)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| invalid("UORS2C01 idx overflow"))?;
            let idx_end = off
                .checked_add(idx_len)
                .ok_or_else(|| invalid("UORS2C01 idx end overflow"))?;
            let idx = bytes
                .get(off..idx_end)
                .ok_or_else(|| invalid("UORS2C01 truncated idx"))?;
            off = idx_end;
            let blob_end = off
                .checked_add(blob_len)
                .ok_or_else(|| invalid("UORS2C01 blob end overflow"))?;
            let blob = bytes
                .get(off..blob_end)
                .ok_or_else(|| invalid("UORS2C01 truncated blob"))?;
            off = blob_end;
            validate_compact_ctx_index(idx, ctx_len, 3, blob, &format!("UORS2C01 fam {fid}"))?;
            fams.push(CompactS2SkipFamily { ctx_len, idx, blob });
        }
        if off != body_end {
            return Err(invalid("UORS2C01 trailing payload length mismatch"));
        }
        let [f0, f1, f2]: [CompactS2SkipFamily<'a>; 3] = fams
            .try_into()
            .map_err(|_| invalid("UORS2C01 family count mismatch"))?;
        Ok(Self {
            c_corpus_sha256,
            families: [f0, f1, f2],
            bytes_sha256: Sha256::digest(bytes).into(),
        })
    }

    pub fn fill_sparse_family_q48(
        &self,
        family: usize,
        key: u64,
        sparse_out: &mut Vec<(u16, u64)>,
    ) -> Result<(bool, u64)> {
        sparse_out.clear();
        let fam = self
            .families
            .get(family)
            .ok_or_else(|| invalid("invalid S2-skip family index"))?;
        if (key >> 48) != family as u64 || ((key >> 24) & 0xFF_FFFF) != 0 {
            return Ok((false, P4_UNIFORM_MASS));
        }
        let lo24 = (key & 0xFF_FFFF) as u32;
        if let Some(slice) = bsearch_compact_slice(fam.idx, fam.ctx_len, 3, fam.blob, lo24) {
            push_modulated_slice(slice, sparse_out)?;
            return Ok((true, 1u64));
        }
        Ok((false, P4_UNIFORM_MASS))
    }

    pub fn lookup_modulated_q48(&self, family: usize, key: u64, target: u16) -> Result<u64> {
        let fam = self
            .families
            .get(family)
            .ok_or_else(|| invalid("invalid S2-skip family index"))?;
        if (key >> 48) != family as u64 || ((key >> 24) & 0xFF_FFFF) != 0 {
            return Ok(P4_UNIFORM_MASS);
        }
        let lo24 = (key & 0xFF_FFFF) as u32;
        if let Some(slice) = bsearch_compact_slice(fam.idx, fam.ctx_len, 3, fam.blob, lo24) {
            return eval_modulated_target(slice, target);
        }
        Ok(P4_UNIFORM_MASS)
    }
}

/// Losslessly encode a frozen `UORS2S01` `S2-skip` table plus optional concatenated `err_i8`
/// bytes across all 3 families (`p5b_s2_skip_errs.i8`) into compact `UORS2C01` (`~38.9 MB`).
pub fn encode_compact_s2_skip_from_uors2s01(
    uors2s01_bytes: &[u8],
    all_errs: Option<&[i8]>,
    expected_c_sha: [u8; 32],
) -> Result<Vec<u8>> {
    if uors2s01_bytes.len() < 8 + 4 + 32 + 32 || &uors2s01_bytes[..8] != MAGIC_UORS2S01 {
        return Err(invalid("invalid UORS2S01 magic"));
    }
    let payload_len = uors2s01_bytes.len() - 32;
    let actual: [u8; 32] = Sha256::digest(&uors2s01_bytes[..payload_len]).into();
    if actual != uors2s01_bytes[payload_len..] {
        return Err(invalid("UORS2S01 trailing SHA-256 mismatch"));
    }
    let payload = &uors2s01_bytes[..payload_len];
    let mut off = 8usize;
    if rd_u32(payload, &mut off)? != 1 {
        return Err(invalid("unsupported UORS2S01 version"));
    }
    if payload.get(off..off + 32) != Some(expected_c_sha.as_slice()) {
        return Err(invalid("UORS2S01 c_corpus_sha256 mismatch"));
    }
    off += 32;

    let mut out = Vec::with_capacity(payload.len() / 2 + 64);
    out.extend_from_slice(MAGIC_UORS2C01);
    out.extend_from_slice(&UORS2C01_VERSION.to_le_bytes());
    out.extend_from_slice(&expected_c_sha);

    let mut err_cursor = 0usize;
    for expected_fid in 0..3u8 {
        let fid = *payload
            .get(off)
            .ok_or_else(|| invalid("UORS2S01 truncated fid"))?;
        off += 1;
        if fid != expected_fid {
            return Err(invalid("UORS2S01 family ordering mismatch"));
        }
        let clen = rd_u32(payload, &mut off)? as usize;
        let rlen = rd_u32(payload, &mut off)? as usize;
        let err_end = err_cursor
            .checked_add(rlen)
            .ok_or_else(|| invalid("UORS2S01 err_i8 cursor overflow"))?;
        if all_errs.is_some_and(|e| err_end > e.len()) {
            return Err(invalid("UORS2S01 external err_i8 slice too short"));
        }
        let ctx_bytes = clen
            .checked_mul(22)
            .ok_or_else(|| invalid("UORS2S01 ctx byte length overflow"))?;
        let ctx_end = off
            .checked_add(ctx_bytes)
            .ok_or_else(|| invalid("UORS2S01 ctx slice overflow"))?;
        let ctxs_raw = payload
            .get(off..ctx_end)
            .ok_or_else(|| invalid("UORS2S01 truncated ctxs"))?;
        off = ctx_end;
        let row_bytes = rlen
            .checked_mul(7)
            .ok_or_else(|| invalid("UORS2S01 row byte length overflow"))?;
        let row_end = off
            .checked_add(row_bytes)
            .ok_or_else(|| invalid("UORS2S01 row slice overflow"))?;
        let rows_raw = payload
            .get(off..row_end)
            .ok_or_else(|| invalid("UORS2S01 truncated rows"))?;
        off = row_end;

        let fam_errs = all_errs.map(|e| &e[err_cursor..err_end]);
        err_cursor = err_end;

        let mut idx = Vec::with_capacity(clen.saturating_mul(7).saturating_add(4));
        let mut blob = Vec::with_capacity(rlen.saturating_mul(4));
        for i in 0..clen {
            let mut co = i * 22;
            let key = rd_u64(ctxs_raw, &mut co)?;
            let rs = rd_u32(ctxs_raw, &mut co)? as usize;
            let rl = rd_u16(ctxs_raw, &mut co)? as usize;
            let _tot = rd_u64(ctxs_raw, &mut co)?;
            let span_end = rs
                .checked_add(rl)
                .ok_or_else(|| invalid("UORS2S01 row span overflow"))?;
            if span_end > rlen {
                return Err(invalid("UORS2S01 row span out of bounds"));
            }
            if (key >> 48) != u64::from(fid) || ((key >> 24) & 0xFF_FFFF) != 0 {
                return Err(invalid(format!("UORS2S01 key {key:#x} exceeds lo24")));
            }
            let lo24 = (key & 0xFF_FFFF) as u32;
            let byte_start =
                u32::try_from(blob.len()).map_err(|_| invalid("UORS2C01 blob exceeded u32"))?;
            idx.extend_from_slice(&lo24.to_le_bytes()[..3]);
            idx.extend_from_slice(&byte_start.to_le_bytes());
            for r in rs..span_end {
                let mut ro = r * 7;
                let nt = rd_u16(rows_raw, &mut ro)?;
                let cnt = rd_u32(rows_raw, &mut ro)?;
                let err = match fam_errs {
                    Some(fe) => fe[r],
                    None => rows_raw[ro] as i8,
                };
                encode_compact_row(nt, cnt, err, &mut blob)?;
            }
        }
        let blob_len =
            u32::try_from(blob.len()).map_err(|_| invalid("UORS2C01 blob exceeded u32"))?;
        idx.extend_from_slice(&blob_len.to_le_bytes());
        out.push(fid);
        out.extend_from_slice(&(clen as u32).to_le_bytes());
        out.extend_from_slice(&blob_len.to_le_bytes());
        out.extend_from_slice(&idx);
        out.extend_from_slice(&blob);
    }
    if off != payload.len() || all_errs.is_some_and(|e| err_cursor != e.len()) {
        return Err(invalid("UORS2S01 length or err_i8 count mismatch"));
    }
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    let _ = CompactS2SkipView::from_bytes(&out)?;
    Ok(out)
}

// ---------- Incremental CausalSessionCache (exact Stage B4/B5b semantics) ----------

const TRI_TABLE_CAP: usize = 16_384;
const TRI_MASK: usize = TRI_TABLE_CAP - 1;
const LCP_TABLE_CAP: usize = 16_384;
const LCP_MASK: usize = LCP_TABLE_CAP - 1;

#[inline]
fn cache_lawful_mass(hit_w: u64, total_w: u64) -> u64 {
    if total_w == 0 {
        return P4_UNIFORM_MASS;
    }
    if hit_w == 0 {
        return 1;
    }
    1u64 + ((u128::from(hit_w) * u128::from(P4_NORM_BUDGET)) / u128::from(total_w)) as u64
}

#[inline]
fn hash_u32(k: u32) -> usize {
    (k.wrapping_mul(0x9E37_79B1)) as usize
}

#[inline]
fn hash_u64(k: u64) -> usize {
    ((k.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 48) as usize
}

pub struct CausalSessionCache {
    uni_w: [u32; P4_VOCAB],
    uni_touched: Vec<u16>,
    uni_total: u64,
    rec_tokens: Vec<u16>,

    bi_slot: [u32; P4_VOCAB],
    bi_touched: Vec<u16>,
    bi_lists: Vec<Vec<(u16, u16)>>,

    tri_keys: Vec<u32>,
    tri_slots: Vec<u32>,
    tri_touched: Vec<usize>,
    tri_lists: Vec<Vec<(u16, u16)>>,

    lcp_keys: Vec<u64>,
    lcp_slots: Vec<u32>,
    lcp_touched: Vec<usize>,
    lcp_lists: Vec<[usize; 9]>,
}

impl Default for CausalSessionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalSessionCache {
    pub fn new() -> Self {
        Self {
            uni_w: [0u32; P4_VOCAB],
            uni_touched: Vec::with_capacity(512),
            uni_total: 0,
            rec_tokens: Vec::with_capacity(P4_MAX_CONTEXT),
            bi_slot: [u32::MAX; P4_VOCAB],
            bi_touched: Vec::with_capacity(512),
            bi_lists: Vec::with_capacity(512),
            tri_keys: vec![u32::MAX; TRI_TABLE_CAP],
            tri_slots: vec![u32::MAX; TRI_TABLE_CAP],
            tri_touched: Vec::with_capacity(2048),
            tri_lists: Vec::with_capacity(2048),
            lcp_keys: vec![u64::MAX; LCP_TABLE_CAP],
            lcp_slots: vec![u32::MAX; LCP_TABLE_CAP],
            lcp_touched: Vec::with_capacity(2048),
            lcp_lists: Vec::with_capacity(2048),
        }
    }

    pub fn reset_record(&mut self) {
        for &t in &self.uni_touched {
            self.uni_w[usize::from(t)] = 0;
        }
        self.uni_touched.clear();
        self.uni_total = 0;
        self.rec_tokens.clear();

        for &t in &self.bi_touched {
            self.bi_slot[usize::from(t)] = u32::MAX;
        }
        self.bi_touched.clear();
        self.bi_lists.clear();

        for &idx in &self.tri_touched {
            self.tri_keys[idx] = u32::MAX;
            self.tri_slots[idx] = u32::MAX;
        }
        self.tri_touched.clear();
        self.tri_lists.clear();

        for &idx in &self.lcp_touched {
            self.lcp_keys[idx] = u64::MAX;
            self.lcp_slots[idx] = u32::MAX;
        }
        self.lcp_touched.clear();
        self.lcp_lists.clear();
    }

    #[inline]
    fn find_tri_slot(&self, key: u32) -> Option<usize> {
        let mut idx = hash_u32(key) & TRI_MASK;
        loop {
            let k = self.tri_keys[idx];
            if k == key {
                return Some(self.tri_slots[idx] as usize);
            }
            if k == u32::MAX {
                return None;
            }
            idx = (idx + 1) & TRI_MASK;
        }
    }

    #[inline]
    fn get_or_insert_tri_slot(&mut self, key: u32) -> usize {
        if self.tri_touched.len() * 4 >= TRI_TABLE_CAP * 3 {
            if let Some(s) = self.find_tri_slot(key) {
                return s;
            }
        }
        let mut idx = hash_u32(key) & TRI_MASK;
        loop {
            let k = self.tri_keys[idx];
            if k == key {
                return self.tri_slots[idx] as usize;
            }
            if k == u32::MAX {
                let slot = self.tri_lists.len();
                self.tri_lists.push(Vec::with_capacity(4));
                self.tri_keys[idx] = key;
                self.tri_slots[idx] = slot as u32;
                self.tri_touched.push(idx);
                return slot;
            }
            idx = (idx + 1) & TRI_MASK;
        }
    }

    #[inline]
    fn find_lcp_slot(&self, key: u64) -> Option<usize> {
        let mut idx = hash_u64(key) & LCP_MASK;
        loop {
            let k = self.lcp_keys[idx];
            if k == key {
                return Some(self.lcp_slots[idx] as usize);
            }
            if k == u64::MAX {
                return None;
            }
            idx = (idx + 1) & LCP_MASK;
        }
    }

    #[inline]
    fn get_or_insert_lcp_slot(&mut self, key: u64) -> usize {
        if self.lcp_touched.len() * 4 >= LCP_TABLE_CAP * 3 {
            if let Some(s) = self.find_lcp_slot(key) {
                return s;
            }
        }
        let mut idx = hash_u64(key) & LCP_MASK;
        loop {
            let k = self.lcp_keys[idx];
            if k == key {
                return self.lcp_slots[idx] as usize;
            }
            if k == u64::MAX {
                let slot = self.lcp_lists.len();
                self.lcp_lists.push([0usize; 9]);
                self.lcp_keys[idx] = key;
                self.lcp_slots[idx] = slot as u32;
                self.lcp_touched.push(idx);
                return slot;
            }
            idx = (idx + 1) & LCP_MASK;
        }
    }

    /// Replay a complete causal record token slice (resetting at any `BOS == 0`).
    pub fn reset_from_tokens(&mut self, tokens: &[u16]) {
        self.reset_record();
        for &tok in tokens {
            self.observe_token(tok);
        }
    }

    /// Advance cache state on a single non-BOS token `tok`, or reset on `tok == 0`.
    pub fn observe_token(&mut self, tok: u16) {
        if tok == 0 {
            self.reset_record();
        } else if usize::from(tok) < P4_VOCAB {
            self.observe_non_bos(tok);
        }
    }

    fn observe_non_bos(&mut self, tok: u16) {
        if usize::from(tok) >= P4_VOCAB {
            return;
        }
        let pos = self.rec_tokens.len();
        if pos > 0 && (pos & 63 == 0) {
            let mut new_tot = 0u64;
            for &t in &self.uni_touched {
                let slot = &mut self.uni_w[usize::from(t)];
                *slot >>= 1;
                new_tot += u64::from(*slot);
            }
            self.uni_total = new_tot;
        }
        let u_idx = usize::from(tok);
        if self.uni_w[u_idx] == 0 && !self.uni_touched.contains(&tok) {
            self.uni_touched.push(tok);
        }
        self.uni_w[u_idx] = self.uni_w[u_idx].saturating_add(64);
        self.uni_total += 64;

        if pos >= 1 {
            let p1 = self.rec_tokens[pos - 1];
            let sl = if self.bi_slot[usize::from(p1)] == u32::MAX {
                let s = self.bi_lists.len();
                self.bi_lists.push(Vec::with_capacity(4));
                self.bi_slot[usize::from(p1)] = s as u32;
                self.bi_touched.push(p1);
                s
            } else {
                self.bi_slot[usize::from(p1)] as usize
            };
            let list = &mut self.bi_lists[sl];
            if let Some(pair) = list.iter_mut().find(|(nt, _)| *nt == tok) {
                pair.1 = pair.1.saturating_add(1);
            } else {
                list.push((tok, 1));
            }
        }

        if pos >= 2 {
            let p2 = self.rec_tokens[pos - 2];
            let p1 = self.rec_tokens[pos - 1];
            let tkey = (u32::from(p2) << 12) | u32::from(p1);
            if self.tri_touched.len() * 4 < TRI_TABLE_CAP * 3 {
                let sl = self.get_or_insert_tri_slot(tkey);
                let list = &mut self.tri_lists[sl];
                if let Some(pair) = list.iter_mut().find(|(nt, _)| *nt == tok) {
                    pair.1 = pair.1.saturating_add(1);
                } else {
                    list.push((tok, 1));
                }
            }
        }

        if pos >= 3 {
            let p3 = self.rec_tokens[pos - 3];
            let p2 = self.rec_tokens[pos - 2];
            let p1 = self.rec_tokens[pos - 1];
            let lkey = (u64::from(p3) << 24) | (u64::from(p2) << 12) | u64::from(p1);
            if self.lcp_touched.len() * 4 < LCP_TABLE_CAP * 3 {
                let sl = self.get_or_insert_lcp_slot(lkey);
                let arr = &mut self.lcp_lists[sl];
                let cnt = arr[0];
                if cnt < 8 {
                    arr[0] = cnt + 1;
                    arr[1 + cnt] = pos;
                } else {
                    for k in 1..8 {
                        arr[k] = arr[k + 1];
                    }
                    arr[8] = pos;
                }
            }
        }

        self.rec_tokens.push(tok);
    }

    /// Evaluate a single target token `tok` read-only (oracle helper).
    pub fn peek_score_readonly(&self, tok: u16) -> (u8, [u64; 4]) {
        if usize::from(tok) >= P4_VOCAB {
            return (0, [P4_UNIFORM_MASS; 4]);
        }
        let pos = self.rec_tokens.len();
        let uni_m = if self.uni_total == 0 {
            P4_UNIFORM_MASS
        } else {
            cache_lawful_mass(u64::from(self.uni_w[usize::from(tok)]), self.uni_total)
        };
        let mut hit_bi = false;
        let bi_m = if pos >= 1 {
            let sl = self.bi_slot[usize::from(self.rec_tokens[pos - 1])];
            if sl == u32::MAX {
                P4_UNIFORM_MASS
            } else {
                hit_bi = true;
                let mut tot = 0u64;
                let mut hit = 0u64;
                for &(nt, cnt) in &self.bi_lists[sl as usize] {
                    let c = u64::from(cnt);
                    tot += c;
                    if nt == tok {
                        hit = c;
                    }
                }
                cache_lawful_mass(hit, tot)
            }
        } else {
            P4_UNIFORM_MASS
        };
        let mut hit_tri = false;
        let tri_m = if pos >= 2 {
            let tkey =
                (u32::from(self.rec_tokens[pos - 2]) << 12) | u32::from(self.rec_tokens[pos - 1]);
            if let Some(sl) = self.find_tri_slot(tkey) {
                hit_tri = true;
                let mut tot = 0u64;
                let mut hit = 0u64;
                for &(nt, cnt) in &self.tri_lists[sl] {
                    let c = u64::from(cnt);
                    tot += c;
                    if nt == tok {
                        hit = c;
                    }
                }
                cache_lawful_mass(hit, tot)
            } else {
                P4_UNIFORM_MASS
            }
        } else {
            P4_UNIFORM_MASS
        };
        let mut hit_lcp = false;
        let lcp_m = if pos >= 3 {
            let lkey = (u64::from(self.rec_tokens[pos - 3]) << 24)
                | (u64::from(self.rec_tokens[pos - 2]) << 12)
                | u64::from(self.rec_tokens[pos - 1]);
            if let Some(sl) = self.find_lcp_slot(lkey) {
                let entry = self.lcp_lists[sl];
                let cnt = entry[0];
                if cnt > 0 {
                    hit_lcp = true;
                    let mut tot_w = 0u64;
                    let mut hit_w = 0u64;
                    for &prev_follow_pos in &entry[1..=cnt] {
                        let max_l = 16usize.min(prev_follow_pos).min(pos);
                        let mut match_len = 3usize;
                        while match_len < max_l {
                            if self.rec_tokens[prev_follow_pos - 1 - match_len]
                                != self.rec_tokens[pos - 1 - match_len]
                            {
                                break;
                            }
                            match_len += 1;
                        }
                        let w = (match_len * match_len) as u64;
                        tot_w += w;
                        if self.rec_tokens[prev_follow_pos] == tok {
                            hit_w += w;
                        }
                    }
                    cache_lawful_mass(hit_w, tot_w)
                } else {
                    P4_UNIFORM_MASS
                }
            } else {
                P4_UNIFORM_MASS
            }
        } else {
            P4_UNIFORM_MASS
        };
        let hit_state = if hit_tri || hit_lcp {
            2
        } else if hit_bi {
            1
        } else {
            0
        };
        (hit_state, [uni_m, bi_m, tri_m, lcp_m])
    }

    /// Evaluate all 4 cache experts sparsely (`default_masses`, `sparse_out`) for all legal tokens.
    pub fn eval_sparse_all(&self, sparse_out: &mut [Vec<(u16, u64)>; 4]) -> (u8, [u64; 4]) {
        for s in sparse_out.iter_mut() {
            s.clear();
        }
        let pos = self.rec_tokens.len();

        let uni_default = if self.uni_total == 0 {
            P4_UNIFORM_MASS
        } else {
            for &t in &self.uni_touched {
                let hw = u64::from(self.uni_w[usize::from(t)]);
                if hw > 0 {
                    sparse_out[0].push((t, cache_lawful_mass(hw, self.uni_total)));
                }
            }
            1u64
        };

        let mut hit_bi = false;
        let bi_default = if pos >= 1 {
            let sl = self.bi_slot[usize::from(self.rec_tokens[pos - 1])];
            if sl == u32::MAX {
                P4_UNIFORM_MASS
            } else {
                hit_bi = true;
                let items = &self.bi_lists[sl as usize];
                let tot: u64 = items.iter().map(|&(_, cnt)| u64::from(cnt)).sum();
                for &(nt, cnt) in items {
                    sparse_out[1].push((nt, cache_lawful_mass(u64::from(cnt), tot)));
                }
                1u64
            }
        } else {
            P4_UNIFORM_MASS
        };

        let mut hit_tri = false;
        let tri_default = if pos >= 2 {
            let tkey =
                (u32::from(self.rec_tokens[pos - 2]) << 12) | u32::from(self.rec_tokens[pos - 1]);
            if let Some(sl) = self.find_tri_slot(tkey) {
                hit_tri = true;
                let items = &self.tri_lists[sl];
                let tot: u64 = items.iter().map(|&(_, cnt)| u64::from(cnt)).sum();
                for &(nt, cnt) in items {
                    sparse_out[2].push((nt, cache_lawful_mass(u64::from(cnt), tot)));
                }
                1u64
            } else {
                P4_UNIFORM_MASS
            }
        } else {
            P4_UNIFORM_MASS
        };

        let mut hit_lcp = false;
        let lcp_default = if pos >= 3 {
            let lkey = (u64::from(self.rec_tokens[pos - 3]) << 24)
                | (u64::from(self.rec_tokens[pos - 2]) << 12)
                | u64::from(self.rec_tokens[pos - 1]);
            if let Some(sl) = self.find_lcp_slot(lkey) {
                let entry = self.lcp_lists[sl];
                let cnt = entry[0];
                if cnt > 0 {
                    hit_lcp = true;
                    let mut tot_w = 0u64;
                    let mut raw_hits: [(u16, u64); 8] = [(0, 0); 8];
                    let mut unique = 0usize;
                    for &prev_follow_pos in &entry[1..=cnt] {
                        let max_l = 16usize.min(prev_follow_pos).min(pos);
                        let mut match_len = 3usize;
                        while match_len < max_l {
                            if self.rec_tokens[prev_follow_pos - 1 - match_len]
                                != self.rec_tokens[pos - 1 - match_len]
                            {
                                break;
                            }
                            match_len += 1;
                        }
                        let w = (match_len * match_len) as u64;
                        tot_w += w;
                        let nt = self.rec_tokens[prev_follow_pos];
                        if let Some(slot) = raw_hits[..unique].iter_mut().find(|x| x.0 == nt) {
                            slot.1 += w;
                        } else {
                            raw_hits[unique] = (nt, w);
                            unique += 1;
                        }
                    }
                    for &(nt, hw) in &raw_hits[..unique] {
                        sparse_out[3].push((nt, cache_lawful_mass(hw, tot_w)));
                    }
                    1u64
                } else {
                    P4_UNIFORM_MASS
                }
            } else {
                P4_UNIFORM_MASS
            }
        } else {
            P4_UNIFORM_MASS
        };

        let hit_state = if hit_tri || hit_lcp {
            2u8
        } else if hit_bi {
            1u8
        } else {
            0u8
        };
        (
            hit_state,
            [uni_default, bi_default, tri_default, lcp_default],
        )
    }
}

// ---------- P4State, P4Session & Raw-Word P4StepTrace ----------

#[derive(Debug, Clone)]
pub struct P4StepTrace {
    pub context_tokens: usize,
    pub prefix_sha256: String,
    pub highest_order: usize,
    pub pick_pos: Option<u32>,
    pub pick_distance: Option<u32>,
    pub pick_ties: u32,
    pub pick_successor: Option<u16>,
    pub pick_class: u8,
    pub s3_pkey: u32,
    pub s3_backoff_key: u16,
    pub s3_mode: String,
    pub s2_skip_keys: [u64; 3],
    pub s2_skip_hit_mask: u8,
    pub cache_hit_state: u8,
    pub bucket: usize,
    pub s4_role_scores: [i64; 3],
    pub s4_role_choice: Option<u8>,
    pub token_id: u16,
    pub mass_q48: u64,
    pub top_ties: u32,
    pub mass_sha256: String,
}

pub struct P4State<'a> {
    pub counts: CompactCountLookup<'a>,
    pub a1s_weights: FrozenPickMixture,
    pub s3: CompactS3View<'a>,
    pub s2_skip: CompactS2SkipView<'a>,
    pub s4: S4ConjunctionTable,
    pub s6: BrierS6Weights,
    pub codes: Vec<[u16; 4]>,
    pub counts_sha256: [u8; 32],
    pub a1s_weights_sha256: [u8; 32],
    pub fit_manifest_sha256: [u8; 32],
    pub s4_sha256: [u8; 32],
    pub s6_sha256: [u8; 32],
}

impl<'a> P4State<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        counts: CompactCountLookup<'a>,
        counts_sha256: [u8; 32],
        a1s_weights: FrozenPickMixture,
        a1s_weights_sha256: [u8; 32],
        fit_manifest_sha256: [u8; 32],
        s3: CompactS3View<'a>,
        s2_skip: CompactS2SkipView<'a>,
        s4: S4ConjunctionTable,
        s4_sha256: [u8; 32],
        s6: BrierS6Weights,
        s6_sha256: [u8; 32],
        codes: Vec<[u16; 4]>,
    ) -> Result<Self> {
        if codes.len() != P4_VOCAB {
            return Err(invalid("P4State expects 4096 identity pick code rows"));
        }
        if a1s_weights.rule() != PickClassRule::Five {
            return Err(invalid("P4State requires 5-class A1s Pick mixture"));
        }
        if s6.buckets != P4_TOTAL_BUCKETS || s6.experts != P4_EXPERTS {
            return Err(invalid(format!(
                "P4State expects {}x{} S6 weights, got {}x{}",
                P4_TOTAL_BUCKETS, P4_EXPERTS, s6.buckets, s6.experts
            )));
        }
        for b in 0..s6.buckets {
            let w = s6.bucket_weights(b)?;
            let sum: u64 = w.iter().copied().sum();
            if sum != P4_Q32_ONE {
                return Err(invalid(format!(
                    "P4 S6 bucket {b} simplex weight sum {sum} != {}",
                    P4_Q32_ONE
                )));
            }
        }
        Ok(Self {
            counts,
            a1s_weights,
            s3,
            s2_skip,
            s4,
            s6,
            codes,
            counts_sha256,
            a1s_weights_sha256,
            fit_manifest_sha256,
            s4_sha256,
            s6_sha256,
        })
    }
}

pub struct P4Session {
    cache: CausalSessionCache,
    history: Vec<u16>,
    kn_hits_buf: Vec<(u16, u64, [u32; 11])>,
    s3_sparse: Vec<(u16, u64)>,
    s2_sparse: [Vec<(u16, u64)>; 3],
    cache_sparse: [Vec<(u16, u64)>; 4],
    acc_buf: Vec<u128>,
    scores_buf: Vec<u64>,
    pair_keys_buf: Vec<u32>,
}

impl Default for P4Session {
    fn default() -> Self {
        Self::new()
    }
}

impl P4Session {
    pub fn new() -> Self {
        Self {
            cache: CausalSessionCache::new(),
            history: Vec::with_capacity(P4_MAX_CONTEXT),
            kn_hits_buf: Vec::with_capacity(P4_VOCAB),
            s3_sparse: Vec::with_capacity(512),
            s2_sparse: [
                Vec::with_capacity(512),
                Vec::with_capacity(512),
                Vec::with_capacity(512),
            ],
            cache_sparse: [
                Vec::with_capacity(256),
                Vec::with_capacity(64),
                Vec::with_capacity(64),
                Vec::with_capacity(16),
            ],
            acc_buf: vec![0u128; P4_VOCAB],
            scores_buf: vec![1u64; P4_VOCAB],
            pair_keys_buf: Vec::with_capacity(256),
        }
    }

    pub fn sync_prompt(&mut self, context: &[u16]) -> Result<()> {
        let bos_start = context.iter().rposition(|&id| id == 0).unwrap_or(0);
        let rec = &context[bos_start..];
        if rec.is_empty() || rec.len() > P4_MAX_CONTEXT || rec.contains(&1) {
            return Err(invalid(
                "P4 decoding needs an open causal record <= 2048 tokens",
            ));
        }
        if rec.iter().any(|&id| usize::from(id) >= P4_VOCAB) {
            return Err(invalid("P4 prompt contains invalid original token ID"));
        }
        self.history.clear();
        self.history.extend_from_slice(rec);
        self.cache.reset_from_tokens(rec);
        Ok(())
    }

    pub fn observe_generated(&mut self, token: u16) -> Result<()> {
        if usize::from(token) >= P4_VOCAB || token == 1 {
            return Err(invalid(
                "P4 generated token out of vocabulary or closed with EOS",
            ));
        }
        if token == 0 {
            self.history.clear();
            self.history.push(0);
            self.cache.reset_record();
            return Ok(());
        }
        if self.history.len() == P4_MAX_CONTEXT {
            self.history.remove(0);
            self.history.push(token);
            // Rebuild causal session cache over the exact slid window so live decode stays
            // bit-identical to cold `sync_prompt(session.history())`.
            self.cache.reset_from_tokens(&self.history);
            return Ok(());
        }
        self.history.push(token);
        self.cache.observe_token(token);
        Ok(())
    }

    pub fn history(&self) -> &[u16] {
        &self.history
    }

    /// Q48 mixed probabilities for tokens `[0..4096)` computed on the most recent `step` call.
    pub fn last_scores_q48(&self) -> &[u64] {
        &self.scores_buf
    }

    fn evaluate_s4_roles(&mut self, state: &P4State<'_>) -> ([i64; 3], Option<u8>) {
        self.pair_keys_buf.clear();
        let tail_start = self.history.len().saturating_sub(24);
        let tail = &self.history[tail_start..];
        for i in 0..tail.len() {
            for j in (i + 1)..tail.len() {
                if let Ok(pk) = s4_pair_key(tail[i], tail[j]) {
                    self.pair_keys_buf.push(pk);
                }
            }
        }
        self.pair_keys_buf.sort_unstable();
        self.pair_keys_buf.dedup();
        let cands = [(1u8, 0u16), (2u8, 0u16), (3u8, 0u16)];
        let scores = state
            .s4
            .vote_candidates(&self.pair_keys_buf, &cands, S4ControlMode::Active);
        let arr = [scores[0], scores[1], scores[2]];
        let best_role = state
            .s4
            .pick_pointer(&self.pair_keys_buf, &cands, S4ControlMode::Active)
            .map(|(idx, _score)| (idx + 1) as u8);
        (arr, best_role)
    }

    /// Score all legal non-BOS tokens `1..4096` into `self.scores_buf`, returning
    /// `(argmax_id, trace_opt)`.
    pub fn step(
        &mut self,
        state: &P4State<'_>,
        read_enabled: bool,
        trace: bool,
    ) -> Result<(u16, Option<P4StepTrace>)> {
        let (tok, trace_opt, _guard) =
            self.step_with_stop_guard(state, read_enabled, trace, None)?;
        Ok((tok, trace_opt))
    }

    /// Score all legal non-BOS tokens `1..4096` into `self.scores_buf`, optionally applying the
    /// pure-integer `p3_state::apply_lawful_stop_and_rep_guard` cycle-breaking and EOS rule while
    /// keeping every legal token mass `>= 1` (PRLL floor compliant).
    pub fn step_with_stop_guard(
        &mut self,
        state: &P4State<'_>,
        read_enabled: bool,
        trace: bool,
        stop_guard: Option<(&[u16], bool, LawfulStopConfig)>,
    ) -> Result<(u16, Option<P4StepTrace>, Option<StopGuardTrace>)> {
        if self.history.is_empty() || self.history.len() > P4_MAX_CONTEXT {
            return Err(invalid("P4Session has invalid active history length"));
        }
        let pick = read_enabled
            .then(|| code_pick(&state.codes, &self.history))
            .flatten();
        let pick_successor = pick.map(|(pos, _, _)| self.history[pos + 1]);
        let pick_dist = pick.map(|(_, d, _)| d);
        let pick_class = state.a1s_weights.rule().classify(pick_dist);

        let (s3_meta, s3_def) = state
            .s3
            .fill_sparse_masses_q48(&self.history, &mut self.s3_sparse)?;
        let skip_keys = skip_keys_for_prefix(&self.history);
        let mut s2_def = [0u64; 3];
        let mut s2_hit_mask = 0u8;
        for (fam, &skey) in skip_keys.iter().enumerate() {
            let (hit, def_m) =
                state
                    .s2_skip
                    .fill_sparse_family_q48(fam, skey, &mut self.s2_sparse[fam])?;
            s2_def[fam] = def_m;
            if hit {
                s2_hit_mask |= 1u8 << fam;
            }
        }
        let (cache_hs, cache_def) = self.cache.eval_sparse_all(&mut self.cache_sparse);

        self.kn_hits_buf.clear();
        let kn_hits = &mut self.kn_hits_buf;
        let (bucket_880, highest_order, totals, default_kn) =
            state
                .counts
                .visit_p4_counts(&self.history, pick_class, |id, kn, counts| {
                    kn_hits.push((id, kn, *counts));
                })?;

        let bucket_2640 = usize::from(cache_hs) * P4_BASE_BUCKETS + usize::from(bucket_880);
        let w = state.s6.bucket_weights(bucket_2640)?;

        let base_acc = u128::from(w[0]) * u128::from(default_kn)
            + u128::from(w[12]) * u128::from(KN_BASE_MASS)
            + u128::from(w[14]) * u128::from(s3_def)
            + u128::from(w[15]) * u128::from(s2_def[0])
            + u128::from(w[16]) * u128::from(s2_def[1])
            + u128::from(w[17]) * u128::from(s2_def[2])
            + u128::from(w[18]) * u128::from(cache_def[0])
            + u128::from(w[19]) * u128::from(cache_def[1])
            + u128::from(w[20]) * u128::from(cache_def[2])
            + u128::from(w[21]) * u128::from(cache_def[3]);
        self.acc_buf[1..P4_VOCAB].fill(base_acc);

        if let Some(ps) = pick_successor {
            if ps > 0 && usize::from(ps) < P4_VOCAB {
                self.acc_buf[usize::from(ps)] += u128::from(w[13]) * u128::from(P4_Q48_ONE);
            }
        }

        let w0 = u128::from(w[0]);
        let d0 = w0 * u128::from(default_kn);
        for &(id, kn, ref counts) in &self.kn_hits_buf {
            let idx = usize::from(id);
            let mut a = self.acc_buf[idx] - d0 + w0 * u128::from(kn);
            for (j, (&cnt, &tot)) in counts.iter().zip(&totals).enumerate() {
                if tot > 0 && cnt > 0 && w[1 + j] > 0 {
                    let m = ((u128::from(cnt) << 48) / u128::from(tot)) as u64;
                    a += u128::from(w[1 + j]) * u128::from(m);
                }
            }
            self.acc_buf[idx] = a;
        }

        if w[14] > 0 {
            let w14 = u128::from(w[14]);
            let d14 = w14 * u128::from(s3_def);
            for &(id, m) in &self.s3_sparse {
                let idx = usize::from(id);
                self.acc_buf[idx] = self.acc_buf[idx] - d14 + w14 * u128::from(m);
            }
        }

        for fam in 0..3 {
            let wf = w[15 + fam];
            if wf > 0 {
                let wfu = u128::from(wf);
                let df = wfu * u128::from(s2_def[fam]);
                for &(id, m) in &self.s2_sparse[fam] {
                    let idx = usize::from(id);
                    self.acc_buf[idx] = self.acc_buf[idx] - df + wfu * u128::from(m);
                }
            }
        }

        for ci in 0..4 {
            let wc = w[18 + ci];
            if wc > 0 {
                let wcu = u128::from(wc);
                let dc = wcu * u128::from(cache_def[ci]);
                for &(id, m) in &self.cache_sparse[ci] {
                    let idx = usize::from(id);
                    self.acc_buf[idx] = self.acc_buf[idx] - dc + wcu * u128::from(m);
                }
            }
        }

        self.scores_buf[0] = 0;
        for id in 1..P4_VOCAB {
            self.scores_buf[id] = ((self.acc_buf[id] >> 32) as u64).clamp(1, P4_Q48_ONE);
        }

        let guard_trace = if let Some((generated, span_exhausted, cfg)) = stop_guard {
            let gt = apply_lawful_stop_and_rep_guard(
                generated,
                &mut self.scores_buf,
                span_exhausted,
                cfg,
            )?;
            Some(gt)
        } else {
            None
        };

        let mut best_id = 0u16;
        let mut best_mass = 0u64;
        let mut top_ties = 0u32;
        let mut mass_hasher = trace.then(Sha256::new);
        for id in 1..P4_VOCAB {
            let q48 = self.scores_buf[id];
            if let Some(h) = &mut mass_hasher {
                h.update(q48.to_le_bytes());
            }
            if q48 > best_mass {
                best_id = id as u16;
                best_mass = q48;
                top_ties = 1;
            } else if q48 == best_mass {
                top_ties += 1;
            }
        }
        if best_id == 0 {
            return Err(invalid("P4 read found no positive legal mass"));
        }

        let trace_opt = if let Some(h) = mass_hasher {
            let mut ph = Sha256::new();
            for &tok in &self.history {
                ph.update(tok.to_le_bytes());
            }
            let (s4_role_scores, s4_role_choice) = self.evaluate_s4_roles(state);
            Some(P4StepTrace {
                context_tokens: self.history.len(),
                prefix_sha256: hex_lower(&ph.finalize()),
                highest_order,
                pick_pos: pick.map(|(p, _, _)| p as u32),
                pick_distance: pick_dist,
                pick_ties: pick.map_or(0, |(_, _, t)| t),
                pick_successor,
                pick_class,
                s3_pkey: s3_meta.pkey,
                s3_backoff_key: s3_meta.backoff_key,
                s3_mode: s3_meta.hit_kind.as_str().to_owned(),
                s2_skip_keys: skip_keys,
                s2_skip_hit_mask: s2_hit_mask,
                cache_hit_state: cache_hs,
                bucket: bucket_2640,
                s4_role_scores,
                s4_role_choice,
                token_id: best_id,
                mass_q48: best_mass,
                top_ties,
                mass_sha256: hex_lower(&h.finalize()),
            })
        } else {
            None
        };
        Ok((best_id, trace_opt, guard_trace))
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}
