/// Shared pure-integer D85 S3 geometric state-row runtime (`UORS3T01`):
/// Conway-Sloane E8 240-root snapping, Starling k-neighbour miss reads,
/// E8 linear output Pick alignment, and fair P2 equal-byte control arms.
pub const E8_BLOCK_LEN: usize = 8;
pub const E8_ROOTS: usize = 240;
const E8_INTEGER_ROOTS: usize = 112;
pub const E8_ADJACENT_COUNT: usize = 56;
pub const PRODUCT_KEYS: usize = E8_ROOTS * E8_ROOTS;
pub const STARLING_NEIGHBOURS: usize = 7;
pub const S3_VOCABULARY: usize = 4096;
pub const S3_Q48_ONE: u64 = 1u64 << 48;
pub const S3_Q16_ONE: u32 = 1u32 << 16;
pub const MAGIC_S3: &[u8; 8] = b"UORS3T01";
const S3_VERSION: u32 = 1;
pub const S3_FLAG_HAS_TOKEN_ROOTS: u16 = 1;
const S3_HEADER_BYTES: usize = 72;
const S3_PRIMARY_CTX_BYTES: usize = 18;
const S3_BACKOFF_CTX_BYTES: usize = 16;
const S3_ROW_BYTES: usize = 7;

const fn compute_e8_root_coords_2x(c: usize) -> [i8; E8_BLOCK_LEN] {
    let mut r = [0i8; E8_BLOCK_LEN];
    if c < E8_INTEGER_ROOTS {
        let pair = c / 4;
        let signs = c % 4;
        let mut k = 0usize;
        let mut lo = 0usize;
        while lo < E8_BLOCK_LEN {
            let mut hi = lo + 1;
            while hi < E8_BLOCK_LEN {
                if k == pair {
                    r[lo] = if signs & 2 != 0 { -2 } else { 2 };
                    r[hi] = if signs & 1 != 0 { -2 } else { 2 };
                    return r;
                }
                k += 1;
                hi += 1;
            }
            lo += 1;
        }
        return r;
    }
    let code = c - E8_INTEGER_ROOTS;
    let mut minus = 0usize;
    let mut i = 0usize;
    while i < E8_BLOCK_LEN - 1 {
        if (code >> i) & 1 == 1 {
            r[i] = -1;
            minus += 1;
        } else {
            r[i] = 1;
        }
        i += 1;
    }
    r[7] = if minus % 2 == 1 { -1 } else { 1 };
    r
}

const fn build_e8_root_coords_table() -> [[i8; E8_BLOCK_LEN]; E8_ROOTS] {
    let mut table = [[0i8; E8_BLOCK_LEN]; E8_ROOTS];
    let mut cell = 0usize;
    while cell < E8_ROOTS {
        table[cell] = compute_e8_root_coords_2x(cell);
        cell += 1;
    }
    table
}

/// Precomputed 2x-scaled integer coordinates (`norm^2 = 8`) for all 240 E8 roots.
pub const E8_ROOT_COORDS_2X: [[i8; E8_BLOCK_LEN]; E8_ROOTS] = build_e8_root_coords_table();

const fn dot_2x_coords(a: &[i8; E8_BLOCK_LEN], b: &[i8; E8_BLOCK_LEN]) -> i32 {
    let mut acc = 0i32;
    let mut i = 0usize;
    while i < E8_BLOCK_LEN {
        acc += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    acc
}

#[allow(long_running_const_eval)]
const fn build_e8_neighbours_table() -> [[u8; E8_ADJACENT_COUNT]; E8_ROOTS] {
    let mut table = [[0u8; E8_ADJACENT_COUNT]; E8_ROOTS];
    let mut cell = 0usize;
    while cell < E8_ROOTS {
        let mut count = 0usize;
        let mut cand = 0usize;
        while cand < E8_ROOTS {
            if dot_2x_coords(&E8_ROOT_COORDS_2X[cell], &E8_ROOT_COORDS_2X[cand]) == 4 {
                table[cell][count] = cand as u8;
                count += 1;
            }
            cand += 1;
        }
        cell += 1;
    }
    table
}

/// Precomputed 56-neighbour E8 adjacency (`dot_2x == 4`) in ascending root index order.
pub const E8_NEIGHBOURS_56: [[u8; E8_ADJACENT_COUNT]; E8_ROOTS] = build_e8_neighbours_table();

/// Exact Conway-Sloane nearest E8 root (`0..240`) to an integer 8-block,
/// matching `uor_r4_integer::fbs_address::e8_cell` bit-for-bit.
pub fn e8_snap_240(block: &[i32; E8_BLOCK_LEN]) -> u8 {
    let mut magnitude = [0i64; E8_BLOCK_LEN];
    let mut negative = [false; E8_BLOCK_LEN];
    for i in 0..E8_BLOCK_LEN {
        let v = block[i];
        magnitude[i] = i64::from(v).abs();
        negative[i] = v < 0;
    }
    let (mut first, mut second, mut eighth) = (0usize, 0usize, 0usize);
    for i in 0..E8_BLOCK_LEN {
        let mut place = 0usize;
        for j in 0..E8_BLOCK_LEN {
            if magnitude[j] > magnitude[i] || (j < i && magnitude[j] == magnitude[i]) {
                place += 1;
            }
        }
        match place {
            0 => first = i,
            1 => second = i,
            7 => eighth = i,
            _ => {}
        }
    }
    let mut total = 0i64;
    let mut neg_count = 0usize;
    for i in 0..E8_BLOCK_LEN {
        total += magnitude[i];
        if negative[i] {
            neg_count += 1;
        }
    }
    let odd = neg_count % 2 == 1;
    let two_entries = 2 * (magnitude[first] + magnitude[second]);
    let halves = total - if odd { 2 * magnitude[eighth] } else { 0 };
    if two_entries >= halves {
        let (lo, hi) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        let pair = lo * (15 - lo) / 2 + hi - lo - 1;
        return (4 * pair + 2 * usize::from(negative[lo]) + usize::from(negative[hi])) as u8;
    }
    let mut cell = E8_INTEGER_ROOTS;
    for (i, n) in negative.iter().enumerate().take(E8_BLOCK_LEN - 1) {
        if *n {
            cell += 1 << i;
        }
    }
    if odd && eighth + 1 < E8_BLOCK_LEN {
        if negative[eighth] {
            cell -= 1 << eighth;
        } else {
            cell += 1 << eighth;
        }
    }
    cell as u8
}

/// Exact 2x-scaled integer E8 root coordinates (`norm^2 = 8`) for `cell < 240`.
pub fn e8_root_coords_2x(cell: u8) -> Result<[i8; E8_BLOCK_LEN]> {
    E8_ROOT_COORDS_2X
        .get(usize::from(cell))
        .copied()
        .ok_or_else(|| invalid(format!("E8 root {cell} out of range 0..240")))
}

/// Exact 2x integer inner product between E8 roots `a` and `b` in `{-8, -4, 0, 4, 8}`.
pub fn e8_root_dot_2x(a: u8, b: u8) -> Result<i32> {
    let ca = e8_root_coords_2x(a)?;
    let cb = e8_root_coords_2x(b)?;
    Ok(dot_2x_coords(&ca, &cb))
}

/// Exact 56 nearest root neighbours (`dot_2x == 4`) in ascending root index order.
pub fn e8_root_neighbours_56(cell: u8) -> Result<[u8; E8_ADJACENT_COUNT]> {
    E8_NEIGHBOURS_56
        .get(usize::from(cell))
        .copied()
        .ok_or_else(|| invalid(format!("E8 root {cell} out of range 0..240")))
}

/// Starling k-neighbour hard ranking over the 56 nearest roots of `root`,
/// ordered by descending integer block inner product and ascending root index tie-break.
pub fn starling_k_neighbours(block: &[i32; E8_BLOCK_LEN], root: u8, k: usize) -> Result<Vec<u8>> {
    let neighbours = e8_root_neighbours_56(root)?;
    let mut scored = [(0i64, 0u8); E8_ADJACENT_COUNT];
    for (idx, &n) in neighbours.iter().enumerate() {
        let coords = e8_root_coords_2x(n)?;
        let mut score = 0i64;
        for j in 0..E8_BLOCK_LEN {
            score += i64::from(block[j]) * i64::from(coords[j]);
        }
        scored[idx] = (score, n);
    }
    scored.sort_unstable_by(|&(sa, ra), &(sb, rb)| sb.cmp(&sa).then_with(|| ra.cmp(&rb)));
    Ok(scored
        .iter()
        .take(k.min(E8_ADJACENT_COUNT))
        .map(|&(_, r)| r)
        .collect())
}

/// Pure affine-linear E8 output Pick alignment in Q16 (`16384..=65536`),
/// mapping `dot_2x` in `{-8, -4, 0, 4, 8}` linearly to `{16384, 28672, 40960, 53248, 65536}`.
pub fn output_pick_score_q16(predicted_root: u8, candidate_root: u8) -> Result<u32> {
    let dot = e8_root_dot_2x(predicted_root, candidate_root)?;
    match dot {
        -8 | -4 | 0 | 4 | 8 => Ok(16_384 + ((dot + 8) as u32) * 3_072),
        _ => Err(invalid(format!("invalid E8 dot product {dot}"))),
    }
}

fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn signed_perm8(seed: u64) -> ([usize; E8_BLOCK_LEN], [i32; E8_BLOCK_LEN]) {
    let mut perm = [0usize; E8_BLOCK_LEN];
    for i in 0..E8_BLOCK_LEN {
        perm[i] = i;
    }
    let mut rng = seed;
    for i in (1..E8_BLOCK_LEN).rev() {
        let j = (splitmix64_step(&mut rng) % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let bits = splitmix64_step(&mut rng);
    let mut signs = [1i32; E8_BLOCK_LEN];
    for i in 0..E8_BLOCK_LEN {
        if (bits >> i) & 1 == 1 {
            signs[i] = -1;
        }
    }
    (perm, signs)
}

fn step_random_block(
    prev: &[i32; E8_BLOCK_LEN],
    perm: &[usize; E8_BLOCK_LEN],
    signs: &[i32; E8_BLOCK_LEN],
    token_seed: u64,
) -> [i32; E8_BLOCK_LEN] {
    let mut next = [0i32; E8_BLOCK_LEN];
    let mut rng = token_seed;
    for i in 0..E8_BLOCK_LEN {
        let v = prev[perm[i]];
        let decayed = v - (v >> 2);
        let delta = ((splitmix64_step(&mut rng) % 8193) as i32) - 4096;
        next[i] = decayed.saturating_mul(signs[i]).saturating_add(delta);
    }
    next
}

/// Pure-integer path-dependent random hyperoctahedral transport over `prefix`,
/// preserving path recurrence while removing E8 trunk geometry (`DECISION.md` P2 control ii).
pub fn random_transport_blocks(
    seed: u64,
    prefix: &[u16],
) -> Result<([i32; E8_BLOCK_LEN], [i32; E8_BLOCK_LEN])> {
    let (perm_a, signs_a) = signed_perm8(seed ^ 0xA5A5_5A5A_1357_2468);
    let (perm_b, signs_b) = signed_perm8(seed ^ 0x5A5A_A5A5_2468_1357);
    let mut block_a = [0i32; E8_BLOCK_LEN];
    let mut block_b = [0i32; E8_BLOCK_LEN];
    for &tok in prefix {
        if usize::from(tok) >= S3_VOCABULARY {
            return Err(invalid(format!("S3 prefix token {tok} outside 0..4096")));
        }
        let step_a_seed = seed ^ u64::from(tok).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let step_b_seed = seed ^ u64::from(tok).wrapping_mul(0xD1B5_4A32_D192_ED03);
        block_a = step_random_block(&block_a, &perm_a, &signs_a, step_a_seed);
        block_b = step_random_block(&block_b, &perm_b, &signs_b, step_b_seed);
    }
    Ok((block_a, block_b))
}

/// Treatment vs registered equal-byte control arms from DECISION §4 P2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3AddressArm {
    E8Product,
    TokenContext {
        k: u8,
    },
    /// Uses caller-supplied `(block_a, block_b)` generated by a `seed`-re-seeded transport
    /// (e.g. `random_transport_blocks` or Stage B2b re-seeded CUDA transport).
    RandomTransport {
        seed: u64,
    },
    SplitMixPrefix {
        seed: u64,
    },
}

impl S3AddressArm {
    pub fn encode_parts(self) -> Result<(u8, u8, u64)> {
        match self {
            Self::E8Product => Ok((0, 0, 0)),
            Self::TokenContext { k } => {
                if !(1..=12).contains(&k) {
                    return Err(invalid("TokenContext arm requires k in 1..=12"));
                }
                Ok((1, k, 0))
            }
            Self::RandomTransport { seed } => Ok((2, 0, seed)),
            Self::SplitMixPrefix { seed } => Ok((3, 0, seed)),
        }
    }

    pub fn from_parts(code: u8, param: u8, seed: u64) -> Result<Self> {
        match code {
            0 if param == 0 && seed == 0 => Ok(Self::E8Product),
            1 if (1..=12).contains(&param) && seed == 0 => Ok(Self::TokenContext { k: param }),
            2 if param == 0 => Ok(Self::RandomTransport { seed }),
            3 if param == 0 => Ok(Self::SplitMixPrefix { seed }),
            _ => Err(invalid(format!(
                "invalid S3AddressArm header ({code}, {param}, {seed})"
            ))),
        }
    }

    pub fn allows_token_roots_flag(self, has_token_roots: bool) -> bool {
        matches!(self, Self::E8Product) == has_token_roots
    }

    /// Derive `(primary_key, backoff_key, ra, rb)` for each arm without E8 geometry leakage
    /// or `prefix.len()` positional leakage in non-geometric controls.
    pub fn context_keys(
        self,
        block_a: &[i32; E8_BLOCK_LEN],
        block_b: &[i32; E8_BLOCK_LEN],
        prefix: &[u16],
    ) -> Result<(u32, Option<u16>, u8, u8)> {
        let last_tok = match prefix.last() {
            Some(&t) => {
                if usize::from(t) >= S3_VOCABULARY {
                    return Err(invalid(format!("S3 prefix token {t} outside 0..4096")));
                }
                t
            }
            None => 0,
        };
        match self {
            Self::E8Product | Self::RandomTransport { .. } => {
                let ra = e8_snap_240(block_a);
                let rb = e8_snap_240(block_b);
                let pk = u16::from(ra) * (E8_ROOTS as u16) + u16::from(rb);
                Ok((s3_primary_key(pk, last_tok), Some(pk), ra, rb))
            }
            Self::TokenContext { k } => {
                if !(1..=12).contains(&k) {
                    return Err(invalid("TokenContext arm requires k in 1..=12"));
                }
                if k == 1 {
                    return Ok((s3_primary_key(0, last_tok), None, 0, 0));
                }
                if k == 2 {
                    let prev_tok = if prefix.len() >= 2 {
                        let t = prefix[prefix.len() - 2];
                        if usize::from(t) >= S3_VOCABULARY {
                            return Err(invalid(format!("S3 prefix token {t} outside 0..4096")));
                        }
                        t
                    } else {
                        S3_VOCABULARY as u16
                    };
                    return Ok((s3_primary_key(prev_tok, last_tok), Some(last_tok), 0, 0));
                }
                let start = prefix.len().saturating_sub(usize::from(k));
                let last_idx = prefix.len().saturating_sub(1);
                let tail_prev = &prefix[start..last_idx];
                let mut h = 0x517C_C1B7_2722_0A95u64 ^ u64::from(k) ^ (tail_prev.len() as u64);
                for &tok in tail_prev {
                    if usize::from(tok) >= S3_VOCABULARY {
                        return Err(invalid(format!("S3 prefix token {tok} outside 0..4096")));
                    }
                    h ^= u64::from(tok).wrapping_add(1);
                    h = splitmix64_step(&mut h);
                }
                let hi = (h % (PRODUCT_KEYS as u64)) as u16;
                Ok((s3_primary_key(hi, last_tok), Some(last_tok), 0, 0))
            }
            Self::SplitMixPrefix { seed } => {
                let start = prefix.len().saturating_sub(16);
                let tail = &prefix[start..];
                let mut h = seed ^ (tail.len() as u64);
                for &tok in tail {
                    if usize::from(tok) >= S3_VOCABULARY {
                        return Err(invalid(format!("S3 prefix token {tok} outside 0..4096")));
                    }
                    h ^= u64::from(tok).wrapping_add(1);
                    h = splitmix64_step(&mut h);
                }
                let pk = (h % (PRODUCT_KEYS as u64)) as u16;
                Ok((s3_primary_key(pk, last_tok), Some(pk), 0, 0))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3HitKind {
    ExactPrimary,
    ExactBackoff,
    StarlingMiss,
    UniformFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S3LookupResult {
    pub hit_kind: S3HitKind,
    pub product_key: u16,
    pub raw_count: u64,
    pub total_count: u64,
    pub mass_q48: u64,
    pub predicted_root: u8,
    pub output_pick_q16: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S3RowEntry {
    pub next_token: u16,
    pub count: u32,
    pub err_i8: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S3PrimaryContext {
    pub key: u32,
    pub total: u64,
    pub row_start: u32,
    pub row_len: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S3BackoffContext {
    pub product_key: u16,
    pub total: u64,
    pub row_start: u32,
    pub row_len: u16,
}

pub fn s3_primary_key(product_key: u16, last_token: u16) -> u32 {
    (u32::from(product_key) << 16) | u32::from(last_token)
}

/// Overflow-safe serialized size in bytes of a `UORS3T01` table (safe on 32-bit `wasm32`).
pub fn encoded_s3_bytes(
    has_token_roots: bool,
    primary_ctxs: usize,
    primary_rows: usize,
    backoff_ctxs: usize,
    backoff_rows: usize,
) -> Result<usize> {
    let token_bytes = if has_token_roots { S3_VOCABULARY } else { 0 };
    let p_ctx_b = primary_ctxs
        .checked_mul(S3_PRIMARY_CTX_BYTES)
        .ok_or_else(|| invalid("UORS3T01 primary context byte size overflow"))?;
    let p_row_b = primary_rows
        .checked_mul(S3_ROW_BYTES)
        .ok_or_else(|| invalid("UORS3T01 primary row byte size overflow"))?;
    let b_ctx_b = backoff_ctxs
        .checked_mul(S3_BACKOFF_CTX_BYTES)
        .ok_or_else(|| invalid("UORS3T01 backoff context byte size overflow"))?;
    let b_row_b = backoff_rows
        .checked_mul(S3_ROW_BYTES)
        .ok_or_else(|| invalid("UORS3T01 backoff row byte size overflow"))?;

    S3_HEADER_BYTES
        .checked_add(token_bytes)
        .and_then(|n| n.checked_add(p_ctx_b))
        .and_then(|n| n.checked_add(p_row_b))
        .and_then(|n| n.checked_add(b_ctx_b))
        .and_then(|n| n.checked_add(b_row_b))
        .ok_or_else(|| invalid("UORS3T01 total byte size overflow"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3StateTable {
    arm: S3AddressArm,
    train_sha: [u8; 32],
    token_e8_roots: Option<[u8; S3_VOCABULARY]>,
    primary_ctxs: Vec<S3PrimaryContext>,
    primary_rows: Vec<S3RowEntry>,
    backoff_ctxs: Vec<S3BackoffContext>,
    backoff_rows: Vec<S3RowEntry>,
}

fn read_u16_le(bytes: &[u8], off: &mut usize) -> Result<u16> {
    let end = off
        .checked_add(2)
        .ok_or_else(|| invalid("UORS3T01 offset overflow"))?;
    let slice = bytes
        .get(*off..end)
        .ok_or_else(|| invalid("truncated UORS3T01 u16"))?;
    *off = end;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(bytes: &[u8], off: &mut usize) -> Result<u32> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| invalid("UORS3T01 offset overflow"))?;
    let slice = bytes
        .get(*off..end)
        .ok_or_else(|| invalid("truncated UORS3T01 u32"))?;
    *off = end;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_le(bytes: &[u8], off: &mut usize) -> Result<u64> {
    let end = off
        .checked_add(8)
        .ok_or_else(|| invalid("UORS3T01 offset overflow"))?;
    let slice = bytes
        .get(*off..end)
        .ok_or_else(|| invalid("truncated UORS3T01 u64"))?;
    *off = end;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn snap_i64_accum(acc: &[i64; E8_BLOCK_LEN]) -> u8 {
    let mut max_abs = 0u64;
    for &v in acc {
        let a = v.unsigned_abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let limit = (i32::MAX as u64) / 2;
    let mut shift = 0u32;
    while (max_abs >> shift) > limit {
        shift += 1;
    }
    let mut scaled = [0i32; E8_BLOCK_LEN];
    for i in 0..E8_BLOCK_LEN {
        scaled[i] = (acc[i] >> shift) as i32;
    }
    e8_snap_240(&scaled)
}

impl S3StateTable {
    pub fn arm(&self) -> S3AddressArm {
        self.arm
    }

    pub fn has_token_roots(&self) -> bool {
        self.token_e8_roots.is_some()
    }

    pub fn train_sha(&self) -> [u8; 32] {
        self.train_sha
    }

    pub fn primary_context_count(&self) -> usize {
        self.primary_ctxs.len()
    }

    pub fn primary_row_count(&self) -> usize {
        self.primary_rows.len()
    }

    pub fn backoff_context_count(&self) -> usize {
        self.backoff_ctxs.len()
    }

    pub fn backoff_row_count(&self) -> usize {
        self.backoff_rows.len()
    }

    pub fn token_e8_root(&self, token: u16) -> Result<Option<u8>> {
        let idx = usize::from(token);
        if idx >= S3_VOCABULARY {
            return Err(invalid(format!(
                "S3 token {token} outside vocabulary 0..4096"
            )));
        }
        Ok(self.token_e8_roots.as_ref().map(|roots| roots[idx]))
    }

    /// Encode sorted primary and backoff context/row tables into lossless `UORS3T01` wire format.
    pub fn encode_bytes(
        arm: S3AddressArm,
        train_sha: [u8; 32],
        token_e8_roots: Option<[u8; S3_VOCABULARY]>,
        primary: &[(u32, &[(u16, u32, i8)])],
        backoff: &[(u16, &[(u16, u32, i8)])],
    ) -> Result<Vec<u8>> {
        let (arm_code, arm_param, seed) = arm.encode_parts()?;
        let has_token_roots = token_e8_roots.is_some();
        if !arm.allows_token_roots_flag(has_token_roots) {
            return Err(invalid(
                "S3AddressArm disagrees with token_e8_roots presence flag",
            ));
        }
        if let Some(roots) = &token_e8_roots {
            for &r in roots {
                if usize::from(r) >= E8_ROOTS {
                    return Err(invalid(format!("token E8 root {r} >= {E8_ROOTS}")));
                }
            }
        }
        let mut primary_row_count = 0usize;
        for &(_, rows) in primary {
            primary_row_count = primary_row_count
                .checked_add(rows.len())
                .ok_or_else(|| invalid("primary row count overflow"))?;
        }
        let mut backoff_row_count = 0usize;
        for &(_, rows) in backoff {
            backoff_row_count = backoff_row_count
                .checked_add(rows.len())
                .ok_or_else(|| invalid("backoff row count overflow"))?;
        }
        let p_ctx_u32 =
            u32::try_from(primary.len()).map_err(|_| invalid("primary ctx len > u32::MAX"))?;
        let p_row_u32 =
            u32::try_from(primary_row_count).map_err(|_| invalid("primary row len > u32::MAX"))?;
        let b_ctx_u32 =
            u32::try_from(backoff.len()).map_err(|_| invalid("backoff ctx len > u32::MAX"))?;
        let b_row_u32 =
            u32::try_from(backoff_row_count).map_err(|_| invalid("backoff row len > u32::MAX"))?;

        let total_bytes = encoded_s3_bytes(
            has_token_roots,
            primary.len(),
            primary_row_count,
            backoff.len(),
            backoff_row_count,
        )?;
        let flags = if has_token_roots {
            S3_FLAG_HAS_TOKEN_ROOTS
        } else {
            0
        };
        let mut out = Vec::with_capacity(total_bytes);
        out.extend_from_slice(MAGIC_S3);
        out.extend_from_slice(&S3_VERSION.to_le_bytes());
        out.push(arm_code);
        out.push(arm_param);
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&seed.to_le_bytes());
        out.extend_from_slice(&train_sha);
        out.extend_from_slice(&p_ctx_u32.to_le_bytes());
        out.extend_from_slice(&p_row_u32.to_le_bytes());
        out.extend_from_slice(&b_ctx_u32.to_le_bytes());
        out.extend_from_slice(&b_row_u32.to_le_bytes());
        if let Some(roots) = &token_e8_roots {
            out.extend_from_slice(roots);
        }

        let mut p_off = 0u32;
        let mut prev_pk: Option<u32> = None;
        for &(key, rows) in primary {
            let pk = (key >> 16) as usize;
            let last_tok = (key & 0xFFFF) as usize;
            if pk >= PRODUCT_KEYS || last_tok >= S3_VOCABULARY {
                return Err(invalid("primary S3 context key outside valid bounds"));
            }
            if prev_pk.is_some_and(|prev| key <= prev) {
                return Err(invalid("primary S3 contexts not strictly ascending"));
            }
            if rows.is_empty() || rows.len() > usize::from(u16::MAX) {
                return Err(invalid("primary S3 context has invalid row count"));
            }
            prev_pk = Some(key);
            let mut total = 0u64;
            for &(_, count, _) in rows {
                total = total
                    .checked_add(u64::from(count))
                    .ok_or_else(|| invalid("primary S3 context total overflow"))?;
            }
            if total == 0 {
                return Err(invalid("primary S3 context has zero total"));
            }
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&total.to_le_bytes());
            out.extend_from_slice(&p_off.to_le_bytes());
            out.extend_from_slice(&(rows.len() as u16).to_le_bytes());
            let delta = u32::try_from(rows.len()).map_err(|_| invalid("row len > u32::MAX"))?;
            p_off = p_off
                .checked_add(delta)
                .ok_or_else(|| invalid("primary S3 row offset overflow"))?;
        }
        for &(_, rows) in primary {
            let mut prev_tok: Option<u16> = None;
            for &(next_token, count, err_i8) in rows {
                if usize::from(next_token) >= S3_VOCABULARY || count == 0 {
                    return Err(invalid("invalid primary S3 follower token or zero count"));
                }
                if prev_tok.is_some_and(|prev| next_token <= prev) {
                    return Err(invalid("primary S3 follower tokens not strictly ascending"));
                }
                prev_tok = Some(next_token);
                out.extend_from_slice(&next_token.to_le_bytes());
                out.extend_from_slice(&count.to_le_bytes());
                out.push(err_i8 as u8);
            }
        }

        let mut b_off = 0u32;
        let mut prev_bk: Option<u16> = None;
        for &(pk, rows) in backoff {
            if usize::from(pk) >= PRODUCT_KEYS {
                return Err(invalid("backoff S3 product_key >= 57600"));
            }
            if prev_bk.is_some_and(|prev| pk <= prev) {
                return Err(invalid("backoff S3 contexts not strictly ascending"));
            }
            if rows.is_empty() || rows.len() > usize::from(u16::MAX) {
                return Err(invalid("backoff S3 context has invalid row count"));
            }
            prev_bk = Some(pk);
            let mut total = 0u64;
            for &(_, count, _) in rows {
                total = total
                    .checked_add(u64::from(count))
                    .ok_or_else(|| invalid("backoff S3 context total overflow"))?;
            }
            if total == 0 {
                return Err(invalid("backoff S3 context has zero total"));
            }
            out.extend_from_slice(&pk.to_le_bytes());
            out.extend_from_slice(&total.to_le_bytes());
            out.extend_from_slice(&b_off.to_le_bytes());
            out.extend_from_slice(&(rows.len() as u16).to_le_bytes());
            let delta = u32::try_from(rows.len()).map_err(|_| invalid("row len > u32::MAX"))?;
            b_off = b_off
                .checked_add(delta)
                .ok_or_else(|| invalid("backoff S3 row offset overflow"))?;
        }
        for &(_, rows) in backoff {
            let mut prev_tok: Option<u16> = None;
            for &(next_token, count, err_i8) in rows {
                if usize::from(next_token) >= S3_VOCABULARY || count == 0 {
                    return Err(invalid("invalid backoff S3 follower token or zero count"));
                }
                if prev_tok.is_some_and(|prev| next_token <= prev) {
                    return Err(invalid("backoff S3 follower tokens not strictly ascending"));
                }
                prev_tok = Some(next_token);
                out.extend_from_slice(&next_token.to_le_bytes());
                out.extend_from_slice(&count.to_le_bytes());
                out.push(err_i8 as u8);
            }
        }
        if out.len() != total_bytes {
            return Err(invalid("internal UORS3T01 byte length mismatch"));
        }
        Ok(out)
    }

    /// Decode and fail-closed validate `UORS3T01` bytes against pinned train/payload digests.
    pub fn from_bytes(
        bytes: &[u8],
        expected_train_sha: [u8; 32],
        expected_bytes_sha: [u8; 32],
    ) -> Result<Self> {
        let actual_sha: [u8; 32] = Sha256::digest(bytes).into();
        if actual_sha != expected_bytes_sha {
            return Err(invalid("UORS3T01 payload SHA-256 mismatch"));
        }
        if bytes.len() < S3_HEADER_BYTES || &bytes[..8] != MAGIC_S3 {
            return Err(invalid("invalid UORS3T01 header or magic"));
        }
        let mut off = 8usize;
        let version = read_u32_le(bytes, &mut off)?;
        if version != S3_VERSION {
            return Err(invalid(format!("unsupported UORS3T01 version {version}")));
        }
        let arm_code = bytes[off];
        let arm_param = bytes[off + 1];
        off += 2;
        let flags = read_u16_le(bytes, &mut off)?;
        if flags & !S3_FLAG_HAS_TOKEN_ROOTS != 0 {
            return Err(invalid(format!("unknown UORS3T01 header flags {flags:#x}")));
        }
        let has_token_roots = (flags & S3_FLAG_HAS_TOKEN_ROOTS) != 0;
        let seed = read_u64_le(bytes, &mut off)?;
        let arm = S3AddressArm::from_parts(arm_code, arm_param, seed)?;
        if !arm.allows_token_roots_flag(has_token_roots) {
            return Err(invalid(
                "UORS3T01 header flags inconsistent with S3AddressArm",
            ));
        }
        let train_sha: [u8; 32] = bytes[off..off + 32]
            .try_into()
            .map_err(|_| invalid("truncated train sha in UORS3T01"))?;
        off += 32;
        if train_sha != expected_train_sha {
            return Err(invalid("UORS3T01 train SHA-256 mismatch"));
        }
        let p_ctx_len = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("p_ctx_len exceeds platform usize"))?;
        let p_row_len = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("p_row_len exceeds platform usize"))?;
        let b_ctx_len = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("b_ctx_len exceeds platform usize"))?;
        let b_row_len = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("b_row_len exceeds platform usize"))?;

        let expected_len =
            encoded_s3_bytes(has_token_roots, p_ctx_len, p_row_len, b_ctx_len, b_row_len)?;
        if bytes.len() != expected_len {
            return Err(invalid(format!(
                "UORS3T01 byte length {} != expected {expected_len}",
                bytes.len()
            )));
        }

        let token_e8_roots = if has_token_roots {
            let end = off
                .checked_add(S3_VOCABULARY)
                .ok_or_else(|| invalid("UORS3T01 token root offset overflow"))?;
            let mut roots = [0u8; S3_VOCABULARY];
            roots.copy_from_slice(
                bytes
                    .get(off..end)
                    .ok_or_else(|| invalid("truncated token_e8_roots"))?,
            );
            off = end;
            for &r in &roots {
                if usize::from(r) >= E8_ROOTS {
                    return Err(invalid(format!("UORS3T01 token root {r} >= {E8_ROOTS}")));
                }
            }
            Some(roots)
        } else {
            None
        };

        let mut primary_ctxs = Vec::with_capacity(p_ctx_len);
        let mut cursor_p = 0u32;
        let mut prev_p_key: Option<u32> = None;
        for _ in 0..p_ctx_len {
            let key = read_u32_le(bytes, &mut off)?;
            let total = read_u64_le(bytes, &mut off)?;
            let row_start = read_u32_le(bytes, &mut off)?;
            let row_len = read_u16_le(bytes, &mut off)?;
            if (key >> 16) as usize >= PRODUCT_KEYS
                || (key & 0xFFFF) as usize >= S3_VOCABULARY
                || row_len == 0
                || total == 0
                || row_start != cursor_p
                || prev_p_key.is_some_and(|p| key <= p)
            {
                return Err(invalid("corrupt primary S3 context table"));
            }
            prev_p_key = Some(key);
            cursor_p = cursor_p
                .checked_add(u32::from(row_len))
                .ok_or_else(|| invalid("primary S3 row span overflow"))?;
            primary_ctxs.push(S3PrimaryContext {
                key,
                total,
                row_start,
                row_len,
            });
        }
        if cursor_p as usize != p_row_len {
            return Err(invalid("primary S3 row count mismatch"));
        }

        let mut primary_rows = Vec::with_capacity(p_row_len);
        for ctx in &primary_ctxs {
            let mut sum = 0u64;
            let mut prev_tok: Option<u16> = None;
            for _ in 0..ctx.row_len {
                let next_token = read_u16_le(bytes, &mut off)?;
                let count = read_u32_le(bytes, &mut off)?;
                let err_i8 = bytes[off] as i8;
                off += 1;
                if usize::from(next_token) >= S3_VOCABULARY
                    || count == 0
                    || prev_tok.is_some_and(|p| next_token <= p)
                {
                    return Err(invalid("corrupt primary S3 row entry"));
                }
                prev_tok = Some(next_token);
                sum = sum
                    .checked_add(u64::from(count))
                    .ok_or_else(|| invalid("primary S3 row count sum overflow"))?;
                primary_rows.push(S3RowEntry {
                    next_token,
                    count,
                    err_i8,
                });
            }
            if sum != ctx.total {
                return Err(invalid("primary S3 context total != sum of row counts"));
            }
        }

        let mut backoff_ctxs = Vec::with_capacity(b_ctx_len);
        let mut cursor_b = 0u32;
        let mut prev_b_key: Option<u16> = None;
        for _ in 0..b_ctx_len {
            let product_key = read_u16_le(bytes, &mut off)?;
            let total = read_u64_le(bytes, &mut off)?;
            let row_start = read_u32_le(bytes, &mut off)?;
            let row_len = read_u16_le(bytes, &mut off)?;
            if usize::from(product_key) >= PRODUCT_KEYS
                || row_len == 0
                || total == 0
                || row_start != cursor_b
                || prev_b_key.is_some_and(|p| product_key <= p)
            {
                return Err(invalid("corrupt backoff S3 context table"));
            }
            prev_b_key = Some(product_key);
            cursor_b = cursor_b
                .checked_add(u32::from(row_len))
                .ok_or_else(|| invalid("backoff S3 row span overflow"))?;
            backoff_ctxs.push(S3BackoffContext {
                product_key,
                total,
                row_start,
                row_len,
            });
        }
        if cursor_b as usize != b_row_len {
            return Err(invalid("backoff S3 row count mismatch"));
        }

        let mut backoff_rows = Vec::with_capacity(b_row_len);
        for ctx in &backoff_ctxs {
            let mut sum = 0u64;
            let mut prev_tok: Option<u16> = None;
            for _ in 0..ctx.row_len {
                let next_token = read_u16_le(bytes, &mut off)?;
                let count = read_u32_le(bytes, &mut off)?;
                let err_i8 = bytes[off] as i8;
                off += 1;
                if usize::from(next_token) >= S3_VOCABULARY
                    || count == 0
                    || prev_tok.is_some_and(|p| next_token <= p)
                {
                    return Err(invalid("corrupt backoff S3 row entry"));
                }
                prev_tok = Some(next_token);
                sum = sum
                    .checked_add(u64::from(count))
                    .ok_or_else(|| invalid("backoff S3 row count sum overflow"))?;
                backoff_rows.push(S3RowEntry {
                    next_token,
                    count,
                    err_i8,
                });
            }
            if sum != ctx.total {
                return Err(invalid("backoff S3 context total != sum of row counts"));
            }
        }
        if off != bytes.len() {
            return Err(invalid("trailing bytes in UORS3T01"));
        }

        Ok(Self {
            arm,
            train_sha,
            token_e8_roots,
            primary_ctxs,
            primary_rows,
            backoff_ctxs,
            backoff_rows,
        })
    }

    fn primary_slice(&self, key: u32) -> Option<(u64, &[S3RowEntry])> {
        let idx = self
            .primary_ctxs
            .binary_search_by_key(&key, |c| c.key)
            .ok()?;
        let ctx = self.primary_ctxs[idx];
        let start = ctx.row_start as usize;
        let end = start + usize::from(ctx.row_len);
        Some((ctx.total, &self.primary_rows[start..end]))
    }

    fn backoff_slice(&self, product_key: u16) -> Option<(u64, &[S3RowEntry])> {
        let idx = self
            .backoff_ctxs
            .binary_search_by_key(&product_key, |c| c.product_key)
            .ok()?;
        let ctx = self.backoff_ctxs[idx];
        let start = ctx.row_start as usize;
        let end = start + usize::from(ctx.row_len);
        Some((ctx.total, &self.backoff_rows[start..end]))
    }

    fn accumulate_weighted_root(
        acc: &mut [i64; E8_BLOCK_LEN],
        coords: &[i8; E8_BLOCK_LEN],
        weight: i64,
    ) -> Result<()> {
        for j in 0..E8_BLOCK_LEN {
            acc[j] = acc[j]
                .checked_add(weight * i64::from(coords[j]))
                .ok_or_else(|| invalid("S3 root accumulator overflow"))?;
        }
        Ok(())
    }

    fn predicted_root_and_pick(
        &self,
        rows: &[S3RowEntry],
        cand_root: Option<u8>,
    ) -> Result<(u8, u32)> {
        let (Some(roots), Some(cand)) = (&self.token_e8_roots, cand_root) else {
            return Ok((0, S3_Q16_ONE));
        };
        let mut acc = [0i64; E8_BLOCK_LEN];
        for row in rows {
            let coords = e8_root_coords_2x(roots[usize::from(row.next_token)])?;
            let w = i64::from(row.count);
            Self::accumulate_weighted_root(&mut acc, &coords, w)?;
        }
        let predicted_root = snap_i64_accum(&acc);
        let output_pick_q16 = output_pick_score_q16(predicted_root, cand)?;
        Ok((predicted_root, output_pick_q16))
    }

    /// Look up target in S3 state rows (primary -> exact backoff -> Starling 7-neighbour miss -> uniform fallback).
    pub fn lookup_q48(
        &self,
        block_a: &[i32; E8_BLOCK_LEN],
        block_b: &[i32; E8_BLOCK_LEN],
        prefix: &[u16],
        target: u16,
    ) -> Result<S3LookupResult> {
        let cand_root = self.token_e8_root(target)?;
        let (prim_key, backoff_key, ra, rb) = self.arm.context_keys(block_a, block_b, prefix)?;
        let report_pk = backoff_key.unwrap_or((prim_key >> 16) as u16);

        if let Some((total, rows)) = self.primary_slice(prim_key) {
            let raw_count = rows
                .binary_search_by_key(&target, |r| r.next_token)
                .ok()
                .map_or(0, |i| u64::from(rows[i].count));
            let mass_q48 = (((u128::from(raw_count) * u128::from(S3_Q48_ONE)) / u128::from(total))
                .max(1)
                .min(u128::from(S3_Q48_ONE))) as u64;
            let (predicted_root, output_pick_q16) =
                self.predicted_root_and_pick(rows, cand_root)?;
            return Ok(S3LookupResult {
                hit_kind: S3HitKind::ExactPrimary,
                product_key: report_pk,
                raw_count,
                total_count: total,
                mass_q48,
                predicted_root,
                output_pick_q16,
            });
        }

        if let Some(bk) = backoff_key {
            if let Some((total, rows)) = self.backoff_slice(bk) {
                let raw_count = rows
                    .binary_search_by_key(&target, |r| r.next_token)
                    .ok()
                    .map_or(0, |i| u64::from(rows[i].count));
                let mass_q48 = (((u128::from(raw_count) * u128::from(S3_Q48_ONE))
                    / u128::from(total))
                .max(1)
                .min(u128::from(S3_Q48_ONE))) as u64;
                let (predicted_root, output_pick_q16) =
                    self.predicted_root_and_pick(rows, cand_root)?;
                return Ok(S3LookupResult {
                    hit_kind: S3HitKind::ExactBackoff,
                    product_key: report_pk,
                    raw_count,
                    total_count: total,
                    mass_q48,
                    predicted_root,
                    output_pick_q16,
                });
            }
        }

        if matches!(self.arm, S3AddressArm::E8Product) {
            let na = starling_k_neighbours(block_a, ra, STARLING_NEIGHBOURS)?;
            let nb = starling_k_neighbours(block_b, rb, STARLING_NEIGHBOURS)?;
            let block_dot = |blk: &[i32; E8_BLOCK_LEN], coords: &[i8; E8_BLOCK_LEN]| -> i64 {
                let mut acc = 0i64;
                for j in 0..E8_BLOCK_LEN {
                    acc += i64::from(blk[j]) * i64::from(coords[j]);
                }
                acc
            };
            let base_a = e8_root_coords_2x(ra)?;
            let base_b = e8_root_coords_2x(rb)?;
            let sa_base = block_dot(block_a, &base_a);
            let sb_base = block_dot(block_b, &base_b);

            let mut probes: Vec<(i64, u16)> = Vec::with_capacity(na.len() + nb.len());
            for &ra_alt in &na {
                let c = e8_root_coords_2x(ra_alt)?;
                let pk = u16::from(ra_alt) * (E8_ROOTS as u16) + u16::from(rb);
                probes.push((block_dot(block_a, &c) + sb_base, pk));
            }
            for &rb_alt in &nb {
                let c = e8_root_coords_2x(rb_alt)?;
                let pk = u16::from(ra) * (E8_ROOTS as u16) + u16::from(rb_alt);
                probes.push((sa_base + block_dot(block_b, &c), pk));
            }
            probes
                .sort_unstable_by(|&(sa, pka), &(sb, pkb)| sb.cmp(&sa).then_with(|| pka.cmp(&pkb)));

            let mut sum_count = 0u64;
            let mut sum_total = 0u64;
            let mut root_acc = [0i64; E8_BLOCK_LEN];
            for &(_, alt_pk) in probes.iter().take(STARLING_NEIGHBOURS) {
                if let Some((tot, rows)) = self.backoff_slice(alt_pk) {
                    sum_total += tot;
                    if let Ok(idx) = rows.binary_search_by_key(&target, |r| r.next_token) {
                        sum_count += u64::from(rows[idx].count);
                    }
                    if let Some(roots) = &self.token_e8_roots {
                        for row in rows {
                            let coords = e8_root_coords_2x(roots[usize::from(row.next_token)])?;
                            let w = i64::from(row.count);
                            Self::accumulate_weighted_root(&mut root_acc, &coords, w)?;
                        }
                    }
                }
            }
            if sum_total > 0 {
                let mass_q48 = (((u128::from(sum_count) * u128::from(S3_Q48_ONE))
                    / u128::from(sum_total))
                .max(1)
                .min(u128::from(S3_Q48_ONE))) as u64;
                let (predicted_root, output_pick_q16) = if let Some(cand) = cand_root {
                    let pred = snap_i64_accum(&root_acc);
                    (pred, output_pick_score_q16(pred, cand)?)
                } else {
                    (0, S3_Q16_ONE)
                };
                return Ok(S3LookupResult {
                    hit_kind: S3HitKind::StarlingMiss,
                    product_key: report_pk,
                    raw_count: sum_count,
                    total_count: sum_total,
                    mass_q48,
                    predicted_root,
                    output_pick_q16,
                });
            }
        }

        let uniform_mass = S3_Q48_ONE / (S3_VOCABULARY as u64);
        let (predicted_root, output_pick_q16) = if let Some(cand) = cand_root {
            (ra, output_pick_score_q16(ra, cand)?)
        } else {
            (0, S3_Q16_ONE)
        };
        Ok(S3LookupResult {
            hit_kind: S3HitKind::UniformFallback,
            product_key: report_pk,
            raw_count: 0,
            total_count: 0,
            mass_q48: uniform_mass,
            predicted_root,
            output_pick_q16,
        })
    }
}

/// Deterministic equal-byte S3 builder: accumulates primary and backoff counts,
/// then prunes lowest-count entries until encoded `UORS3T01` fits inside `max_bytes`.
pub fn build_equal_byte_s3_table(
    arm: S3AddressArm,
    train_sha: [u8; 32],
    token_e8_roots: Option<[u8; S3_VOCABULARY]>,
    observations: &[(&[i32; E8_BLOCK_LEN], &[i32; E8_BLOCK_LEN], &[u16], u16)],
    max_bytes: usize,
) -> Result<Vec<u8>> {
    use std::collections::BTreeMap;

    let has_token_roots = token_e8_roots.is_some();
    if !arm.allows_token_roots_flag(has_token_roots) {
        return Err(invalid(
            "S3AddressArm disagrees with token_e8_roots presence flag",
        ));
    }

    let mut prim: BTreeMap<(u32, u16), u32> = BTreeMap::new();
    let mut back: BTreeMap<(u16, u16), u32> = BTreeMap::new();
    for &(blk_a, blk_b, prefix, next_tok) in observations {
        if usize::from(next_tok) >= S3_VOCABULARY {
            return Err(invalid("S3 observation token outside 0..4096"));
        }
        let (pkey, bk_opt, _, _) = arm.context_keys(blk_a, blk_b, prefix)?;
        let p_slot = prim.entry((pkey, next_tok)).or_insert(0);
        *p_slot = p_slot.saturating_add(1);
        if let Some(bk) = bk_opt {
            let b_slot = back.entry((bk, next_tok)).or_insert(0);
            *b_slot = b_slot.saturating_add(1);
        }
    }

    let mut p_ctx_counts: BTreeMap<u32, usize> = BTreeMap::new();
    for &(pkey, _) in prim.keys() {
        *p_ctx_counts.entry(pkey).or_insert(0) += 1;
    }
    let mut b_ctx_counts: BTreeMap<u16, usize> = BTreeMap::new();
    for &(bk, _) in back.keys() {
        *b_ctx_counts.entry(bk).or_insert(0) += 1;
    }

    // Order candidates ascending by (count, kind_order, key_u64, next_token) to prune deterministically.
    // kind_order = 0 (primary) before 1 (backoff) preserves coarse backoff support longer.
    let mut candidates: Vec<(u32, u8, u64, u16)> = Vec::with_capacity(prim.len() + back.len());
    for (&(pkey, tok), &cnt) in &prim {
        candidates.push((cnt, 0, u64::from(pkey), tok));
    }
    for (&(bk, tok), &cnt) in &back {
        candidates.push((cnt, 1, u64::from(bk), tok));
    }
    candidates.sort_unstable();

    let mut p_ctxs = p_ctx_counts.len();
    let mut p_rows = prim.len();
    let mut b_ctxs = b_ctx_counts.len();
    let mut b_rows = back.len();
    let mut current_bytes = encoded_s3_bytes(has_token_roots, p_ctxs, p_rows, b_ctxs, b_rows)?;

    for &(_, kind, key_u64, tok) in &candidates {
        if current_bytes <= max_bytes {
            break;
        }
        if kind == 0 {
            let pkey = key_u64 as u32;
            if prim.remove(&(pkey, tok)).is_some() {
                p_rows -= 1;
                if let Some(c) = p_ctx_counts.get_mut(&pkey) {
                    *c -= 1;
                    if *c == 0 {
                        p_ctx_counts.remove(&pkey);
                        p_ctxs -= 1;
                    }
                }
            }
        } else {
            let bk = key_u64 as u16;
            if back.remove(&(bk, tok)).is_some() {
                b_rows -= 1;
                if let Some(c) = b_ctx_counts.get_mut(&bk) {
                    *c -= 1;
                    if *c == 0 {
                        b_ctx_counts.remove(&bk);
                        b_ctxs -= 1;
                    }
                }
            }
        }
        current_bytes = encoded_s3_bytes(has_token_roots, p_ctxs, p_rows, b_ctxs, b_rows)?;
    }

    if current_bytes > max_bytes {
        return Err(invalid(format!(
            "max_bytes {max_bytes} below minimum UORS3T01 header+table size {current_bytes}"
        )));
    }

    let mut prim_grouped: Vec<(u32, Vec<(u16, u32, i8)>)> = Vec::new();
    for ((pkey, tok), cnt) in prim {
        match prim_grouped.last_mut() {
            Some((last_k, rows)) if *last_k == pkey => rows.push((tok, cnt, 0)),
            _ => prim_grouped.push((pkey, vec![(tok, cnt, 0)])),
        }
    }
    let mut back_grouped: Vec<(u16, Vec<(u16, u32, i8)>)> = Vec::new();
    for ((bk, tok), cnt) in back {
        match back_grouped.last_mut() {
            Some((last_k, rows)) if *last_k == bk => rows.push((tok, cnt, 0)),
            _ => back_grouped.push((bk, vec![(tok, cnt, 0)])),
        }
    }
    let prim_refs: Vec<(u32, &[(u16, u32, i8)])> = prim_grouped
        .iter()
        .map(|(k, r)| (*k, r.as_slice()))
        .collect();
    let back_refs: Vec<(u16, &[(u16, u32, i8)])> = back_grouped
        .iter()
        .map(|(k, r)| (*k, r.as_slice()))
        .collect();
    S3StateTable::encode_bytes(arm, train_sha, token_e8_roots, &prim_refs, &back_refs)
}

/// Lossless `UORS3T01` serializer wrapper delegating to `S3StateTable::encode_bytes`.
pub fn encode_uors3t01(
    arm: S3AddressArm,
    train_sha: [u8; 32],
    token_e8_roots: Option<[u8; S3_VOCABULARY]>,
    primary: &[(u32, &[(u16, u32, i8)])],
    backoff: &[(u16, &[(u16, u32, i8)])],
) -> Result<Vec<u8>> {
    S3StateTable::encode_bytes(arm, train_sha, token_e8_roots, primary, backoff)
}
