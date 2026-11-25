/// Shared pure-integer D85 `P3` runtime:
/// - `S4` order-free conjunction pointer table (`UORS4P01`) + margin hinge trainer
/// - `S6` / `S3` / `S4` integer Brier error-write optimizer (`UORS6B01`) + exact Q32 simplex projection
/// - Lawful integer 4-gram cycle & repetition guard + EOS termination
/// - Exact integer paired-band evaluation grader
///
/// All loops use explicit scalar integer arithmetic (`#![forbid(unsafe_code)]`,
/// zero floats, D23/D25/D85 integer-only paths, zero SIMD `fmov`s on ARM64, checked
/// `wasm32`-safe allocation math).
pub const P3_VOCABULARY: usize = 4_096;
pub const P3_BOS: u16 = 0;
pub const P3_EOS: u16 = 1;
pub const P3_Q16_ONE: u64 = 1u64 << 16;
pub const P3_Q32_ONE: u64 = 1u64 << 32;
pub const P3_Q48_ONE: u64 = 1u64 << 48;

pub const MAGIC_S4: &[u8; 8] = b"UORS4P01";
pub const MAGIC_S6_BRIER: &[u8; 8] = b"UORS6B01";
pub const P3_FORMAT_VERSION: u32 = 1;

const S4_HEADER_BYTES: usize = 56;
const S4_CTX_BYTES: usize = 18;
const S4_ENTRY_BYTES: usize = 8;
const S6_HEADER_BYTES: usize = 64;

fn checked_len(n: usize, stride: usize, what: &str) -> Result<usize> {
    n.checked_mul(stride)
        .ok_or_else(|| invalid(format!("{what} size overflowed usize")))
}

fn add_len(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid(format!("{what} total bytes overflowed usize")))
}

fn read_u8(bytes: &[u8], off: &mut usize) -> Result<u8> {
    let end = off
        .checked_add(1)
        .ok_or_else(|| invalid("read_u8 offset overflow"))?;
    let v = *bytes
        .get(*off)
        .ok_or_else(|| invalid("unexpected EOF reading u8"))?;
    *off = end;
    Ok(v)
}

fn read_u16_le(bytes: &[u8], off: &mut usize) -> Result<u16> {
    let end = off
        .checked_add(2)
        .ok_or_else(|| invalid("read_u16_le offset overflow"))?;
    let slice = bytes
        .get(*off..end)
        .ok_or_else(|| invalid("unexpected EOF reading u16"))?;
    *off = end;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(bytes: &[u8], off: &mut usize) -> Result<u32> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| invalid("read_u32_le offset overflow"))?;
    let slice = bytes
        .get(*off..end)
        .ok_or_else(|| invalid("unexpected EOF reading u32"))?;
    *off = end;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_le(bytes: &[u8], off: &mut usize) -> Result<u64> {
    let end = off
        .checked_add(8)
        .ok_or_else(|| invalid("read_u64_le offset overflow"))?;
    let slice = bytes
        .get(*off..end)
        .ok_or_else(|| invalid("unexpected EOF reading u64"))?;
    *off = end;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// First 4,096 primes computed cleanly on host/WASM for bijection checks against
/// `s4_pair_key`. Every product of two primes fits in `u64` (`< 2^31`).
pub fn p3_token_primes() -> Vec<u32> {
    let mut primes: Vec<u32> = Vec::with_capacity(P3_VOCABULARY);
    let mut candidate = 2u32;
    while primes.len() < P3_VOCABULARY {
        let mut is_prime = true;
        let mut i = 0usize;
        while i < primes.len() {
            let p = primes[i];
            if u64::from(p) * u64::from(p) > u64::from(candidate) {
                break;
            }
            if candidate.is_multiple_of(p) {
                is_prime = false;
                break;
            }
            i += 1;
        }
        if is_prime {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

/// Order-free squarefree prime product `prime(a) * prime(b)` (`a != b`, `< 4096`)
/// for fixture bijection checks against `s4_pair_key`.
pub fn s4_prime_product_u64(a: u16, b: u16, primes: &[u32]) -> Result<u64> {
    if primes.len() < P3_VOCABULARY {
        return Err(invalid("token primes slice shorter than 4096"));
    }
    let (lo, hi) = unpack_s4_pair_key(s4_pair_key(a, b)?)?;
    u64::from(primes[usize::from(lo)])
        .checked_mul(u64::from(primes[usize::from(hi)]))
        .ok_or_else(|| invalid("S4 prime product overflow"))
}

/// Canonical order-free squarefree `S4` pair key `(lo << 12) | hi` with `lo < hi < 4096`.
/// Bijective to the prime product `prime(a) * prime(b)` (`a != b`).
pub fn s4_pair_key(a: u16, b: u16) -> Result<u32> {
    if usize::from(a) >= P3_VOCABULARY || usize::from(b) >= P3_VOCABULARY {
        return Err(invalid("S4 pair token id >= 4096"));
    }
    if a == b {
        return Err(invalid("S4 conjunction pair requires two distinct tokens"));
    }
    let lo = if a < b { a } else { b };
    let hi = if a < b { b } else { a };
    Ok((u32::from(lo) << 12) | u32::from(hi))
}

/// Unpack a canonical `S4` pair key back into `(lo, hi)` with `lo < hi < 4096`.
pub fn unpack_s4_pair_key(key: u32) -> Result<(u16, u16)> {
    if (key >> 24) != 0 {
        return Err(invalid("invalid S4 pair key layout"));
    }
    let lo = (key >> 12) as u16;
    let hi = (key & 0x0FFF) as u16;
    if usize::from(lo) >= P3_VOCABULARY || usize::from(hi) >= P3_VOCABULARY || lo >= hi {
        return Err(invalid("invalid S4 pair key layout"));
    }
    Ok((lo, hi))
}

/// Extract sorted, deduplicated order-free `S4` pair keys from a user-turn token slice.
/// `max_distinct == 0` disables pruning; otherwise caps unique ascending turn tokens first.
pub fn extract_turn_pair_keys(turn_tokens: &[u16], max_distinct: usize) -> Result<Vec<u32>> {
    let mut uniq: Vec<u16> = Vec::with_capacity(turn_tokens.len());
    for &t in turn_tokens {
        if usize::from(t) >= P3_VOCABULARY {
            return Err(invalid("user turn contains token id >= 4096"));
        }
        if let Err(pos) = uniq.binary_search(&t) {
            uniq.insert(pos, t);
        }
    }
    if max_distinct > 0 && uniq.len() > max_distinct {
        uniq.truncate(max_distinct);
    }
    if uniq.len() < 2 {
        return Ok(Vec::new());
    }
    let pair_cap = uniq.len() * (uniq.len() - 1) / 2;
    let mut keys = Vec::with_capacity(pair_cap);
    let mut i = 0usize;
    while i < uniq.len() {
        let mut j = i + 1;
        while j < uniq.len() {
            keys.push((u32::from(uniq[i]) << 12) | u32::from(uniq[j]));
            j += 1;
        }
        i += 1;
    }
    Ok(keys)
}

/// One pointer-valued entry stored under an `S4` conjunction key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct S4PointerEntry {
    pub role_class: u8,
    pub span_offset: u16,
    pub count: u32,
    pub err_i8: i8,
}

/// Evaluation control mode required by `DECISION.md` §2 & `P3` gate (`R1`/`R2`/`R4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S4ControlMode {
    Active,
    NoMemory,
    ShuffledRole { seed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct S4ContextRow {
    pair_key: u32,
    total: u64,
    offset: usize,
    len: usize,
}

/// Pure-integer `UORS4P01` conjunction pointer table (`DECISION.md:114,307-308`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S4ConjunctionTable {
    train_sha256: [u8; 32],
    contexts: Vec<S4ContextRow>,
    entries: Vec<S4PointerEntry>,
}

/// Summary of one deterministic `S4` pointer-choice hinge pass (`DECISION.md:308`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S4HingeEpochReport {
    pub examples: usize,
    pub exact_top1: usize,
    pub total_hinge_loss_q16: u64,
    pub updates_applied: usize,
}

/// One labelled pointer-choice training example over user-turn pair keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S4HingeExample {
    pub pair_keys: Vec<u32>,
    pub candidates: Vec<(u8, u16)>,
    pub gold_index: usize,
}

pub fn encoded_s4_bytes(ctx_count: usize, entry_count: usize) -> Result<usize> {
    let c_bytes = checked_len(ctx_count, S4_CTX_BYTES, "S4 contexts")?;
    let e_bytes = checked_len(entry_count, S4_ENTRY_BYTES, "S4 entries")?;
    add_len(
        S4_HEADER_BYTES,
        add_len(c_bytes, e_bytes, "S4 body")?,
        "S4 total",
    )
}

impl S4ConjunctionTable {
    /// Construct an `S4ConjunctionTable` from strictly ascending `(pair_key, entries)` groups.
    pub fn from_sorted_rows(
        train_sha256: [u8; 32],
        rows: &[(u32, &[S4PointerEntry])],
    ) -> Result<Self> {
        let mut contexts = Vec::with_capacity(rows.len());
        let mut total_entries = 0usize;
        for &(_, list) in rows {
            total_entries = total_entries
                .checked_add(list.len())
                .ok_or_else(|| invalid("S4 total entry count overflow"))?;
        }
        let mut entries = Vec::with_capacity(total_entries);
        let mut prev_key: Option<u32> = None;
        for &(pair_key, list) in rows {
            unpack_s4_pair_key(pair_key)?;
            if prev_key.is_some_and(|p| pair_key <= p) {
                return Err(invalid("S4 pair keys must be strictly ascending"));
            }
            if list.is_empty() || list.len() > usize::from(u16::MAX) {
                return Err(invalid("S4 row entry length must be in 1..=u16::MAX"));
            }
            prev_key = Some(pair_key);
            let mut sum_count = 0u64;
            let mut prev_ptr: Option<(u8, u16)> = None;
            let offset = entries.len();
            for &entry in list {
                let ptr = (entry.role_class, entry.span_offset);
                if prev_ptr.is_some_and(|p| ptr <= p) {
                    return Err(invalid(
                        "S4 pointer entries within row must have strictly ascending (role_class, span_offset)",
                    ));
                }
                if entry.count == 0 {
                    return Err(invalid("S4 counted pointer entry cannot have count == 0"));
                }
                prev_ptr = Some(ptr);
                sum_count = sum_count
                    .checked_add(u64::from(entry.count))
                    .ok_or_else(|| invalid("S4 row count sum overflow"))?;
                entries.push(entry);
            }
            contexts.push(S4ContextRow {
                pair_key,
                total: sum_count,
                offset,
                len: list.len(),
            });
        }
        Ok(Self {
            train_sha256,
            contexts,
            entries,
        })
    }

    pub fn context_len(&self) -> usize {
        self.contexts.len()
    }

    pub fn entry_len(&self) -> usize {
        self.entries.len()
    }

    pub fn train_sha256(&self) -> [u8; 32] {
        self.train_sha256
    }

    /// Encode to canonical self-describing `UORS4P01` wire bytes.
    pub fn encode_bytes(&self) -> Result<Vec<u8>> {
        let c_len_u32 =
            u32::try_from(self.contexts.len()).map_err(|_| invalid("S4 ctx count > u32::MAX"))?;
        let e_len_u32 =
            u32::try_from(self.entries.len()).map_err(|_| invalid("S4 entry count > u32::MAX"))?;
        let total_bytes = encoded_s4_bytes(self.contexts.len(), self.entries.len())?;
        let mut out = Vec::with_capacity(total_bytes);
        out.extend_from_slice(MAGIC_S4);
        out.extend_from_slice(&P3_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.train_sha256);
        out.extend_from_slice(&c_len_u32.to_le_bytes());
        out.extend_from_slice(&e_len_u32.to_le_bytes());
        for ctx in &self.contexts {
            let off_u32 =
                u32::try_from(ctx.offset).map_err(|_| invalid("S4 entry offset > u32::MAX"))?;
            let len_u16 = u16::try_from(ctx.len).map_err(|_| invalid("S4 entry len > u16::MAX"))?;
            out.extend_from_slice(&ctx.pair_key.to_le_bytes());
            out.extend_from_slice(&off_u32.to_le_bytes());
            out.extend_from_slice(&len_u16.to_le_bytes());
            out.extend_from_slice(&ctx.total.to_le_bytes());
        }
        for e in &self.entries {
            out.push(e.role_class);
            out.extend_from_slice(&e.span_offset.to_le_bytes());
            out.extend_from_slice(&e.count.to_le_bytes());
            out.push(e.err_i8 as u8);
        }
        if out.len() != total_bytes {
            return Err(invalid("internal UORS4P01 byte length mismatch"));
        }
        Ok(out)
    }

    /// Fail-closed decode of `UORS4P01` wire bytes with digest and structural verification.
    pub fn from_bytes(
        bytes: &[u8],
        expected_train_sha: [u8; 32],
        expected_bytes_sha: [u8; 32],
    ) -> Result<Self> {
        let actual_sha: [u8; 32] = Sha256::digest(bytes).into();
        if actual_sha != expected_bytes_sha {
            return Err(invalid("UORS4P01 payload SHA-256 mismatch"));
        }
        if bytes.len() < S4_HEADER_BYTES || &bytes[..8] != MAGIC_S4 {
            return Err(invalid("invalid UORS4P01 magic or header length"));
        }
        let mut off = 8usize;
        let version = read_u32_le(bytes, &mut off)?;
        if version != P3_FORMAT_VERSION {
            return Err(invalid(format!("unsupported UORS4P01 version {version}")));
        }
        let reserved = read_u32_le(bytes, &mut off)?;
        if reserved != 0 {
            return Err(invalid("non-zero UORS4P01 reserved flags"));
        }
        let mut train_sha = [0u8; 32];
        train_sha.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        if train_sha != expected_train_sha {
            return Err(invalid("UORS4P01 train_sha256 mismatch"));
        }
        let ctx_count = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("UORS4P01 ctx_count overflows usize"))?;
        let entry_count = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("UORS4P01 entry_count overflows usize"))?;
        let expected_len = encoded_s4_bytes(ctx_count, entry_count)?;
        if bytes.len() != expected_len {
            return Err(invalid(format!(
                "UORS4P01 byte length {} != expected {}",
                bytes.len(),
                expected_len
            )));
        }

        let mut raw_ctx = Vec::with_capacity(ctx_count);
        let mut i = 0usize;
        while i < ctx_count {
            let pair_key = read_u32_le(bytes, &mut off)?;
            let offset = usize::try_from(read_u32_le(bytes, &mut off)?)
                .map_err(|_| invalid("UORS4P01 offset overflows usize"))?;
            let len = usize::from(read_u16_le(bytes, &mut off)?);
            let hdr_total = read_u64_le(bytes, &mut off)?;
            raw_ctx.push((pair_key, offset, len, hdr_total));
            i += 1;
        }
        let mut entries = Vec::with_capacity(entry_count);
        let mut j = 0usize;
        while j < entry_count {
            let role_class = read_u8(bytes, &mut off)?;
            let span_offset = read_u16_le(bytes, &mut off)?;
            let count = read_u32_le(bytes, &mut off)?;
            let err_i8 = read_u8(bytes, &mut off)? as i8;
            entries.push(S4PointerEntry {
                role_class,
                span_offset,
                count,
                err_i8,
            });
            j += 1;
        }

        let mut contexts = Vec::with_capacity(ctx_count);
        let mut prev_key: Option<u32> = None;
        let mut expected_off = 0usize;
        for (pair_key, offset, len, hdr_total) in raw_ctx {
            unpack_s4_pair_key(pair_key)?;
            if prev_key.is_some_and(|p| pair_key <= p) {
                return Err(invalid("UORS4P01 keys not strictly ascending"));
            }
            if len == 0 || offset != expected_off {
                return Err(invalid("UORS4P01 row offset or length invalid"));
            }
            let end = offset
                .checked_add(len)
                .ok_or_else(|| invalid("UORS4P01 row range overflow"))?;
            if end > entries.len() {
                return Err(invalid("UORS4P01 row extends beyond entry table"));
            }
            prev_key = Some(pair_key);
            expected_off = end;

            let mut sum_count = 0u64;
            let mut prev_ptr: Option<(u8, u16)> = None;
            let mut idx = offset;
            while idx < end {
                let e = entries[idx];
                let ptr = (e.role_class, e.span_offset);
                if prev_ptr.is_some_and(|p| ptr <= p) {
                    return Err(invalid(
                        "UORS4P01 entries within row not strictly ascending",
                    ));
                }
                if e.count == 0 {
                    return Err(invalid("UORS4P01 entry has zero count"));
                }
                prev_ptr = Some(ptr);
                sum_count = sum_count
                    .checked_add(u64::from(e.count))
                    .ok_or_else(|| invalid("UORS4P01 count sum overflow"))?;
                idx += 1;
            }
            if hdr_total != sum_count {
                return Err(invalid("UORS4P01 context total mismatch"));
            }
            contexts.push(S4ContextRow {
                pair_key,
                total: sum_count,
                offset,
                len,
            });
        }
        if expected_off != entries.len() {
            return Err(invalid("UORS4P01 trailing unreferenced entries"));
        }
        Ok(Self {
            train_sha256: train_sha,
            contexts,
            entries,
        })
    }

    fn find_ctx(&self, pair_key: u32) -> Option<usize> {
        self.contexts
            .binary_search_by_key(&pair_key, |c| c.pair_key)
            .ok()
    }

    /// Deterministic integer vote scores (`Q16` units) over candidate `(role_class, span_offset)`
    /// pointers under the specified evaluation control mode.
    pub fn vote_candidates(
        &self,
        pair_keys: &[u32],
        candidates: &[(u8, u16)],
        mode: S4ControlMode,
    ) -> Vec<i64> {
        let mut scores = vec![0i64; candidates.len()];
        if candidates.is_empty() {
            return scores;
        }
        match mode {
            S4ControlMode::NoMemory => scores,
            S4ControlMode::Active => {
                for &key in pair_keys {
                    let Some(cidx) = self.find_ctx(key) else {
                        continue;
                    };
                    let ctx = &self.contexts[cidx];
                    let end = ctx.offset + ctx.len;
                    let mut idx = ctx.offset;
                    while idx < end {
                        let e = self.entries[idx];
                        let ptr = (e.role_class, e.span_offset);
                        let vote = ((u64::from(e.count) << 16) / ctx.total) as i64
                            + i64::from(e.err_i8) * 512;
                        let mut c = 0usize;
                        while c < candidates.len() {
                            if candidates[c] == ptr {
                                scores[c] += vote;
                            }
                            c += 1;
                        }
                        idx += 1;
                    }
                }
                scores
            }
            S4ControlMode::ShuffledRole { seed } => {
                // Deterministic cyclic derangement/permutation of candidate role slots so no slot keeps
                // its gold correspondence systematically while using identical input keys and table bytes.
                let mut perm: Vec<usize> = Vec::with_capacity(candidates.len());
                let mut i = 0usize;
                while i < candidates.len() {
                    perm.push(i);
                    i += 1;
                }
                if perm.len() > 1 {
                    let mut state = splitmix64(seed);
                    let mut k = perm.len() - 1;
                    while k > 0 {
                        state = splitmix64(state);
                        let j = (state % (k as u64)) as usize; // Sattolo derangement: guarantees perm[idx] != idx
                        perm.swap(k, j);
                        k -= 1;
                    }
                }
                let mut permuted = Vec::with_capacity(candidates.len());
                let mut i = 0usize;
                while i < candidates.len() {
                    permuted.push(candidates[perm[i]]);
                    i += 1;
                }
                self.vote_candidates(pair_keys, &permuted, S4ControlMode::Active)
            }
        }
    }

    /// Return the argmax candidate index (ties break to lowest index) and its Q16 score, or
    /// `None` if all votes are `<= 0` or candidates is empty.
    pub fn pick_pointer(
        &self,
        pair_keys: &[u32],
        candidates: &[(u8, u16)],
        mode: S4ControlMode,
    ) -> Option<(usize, i64)> {
        let scores = self.vote_candidates(pair_keys, candidates, mode);
        let mut best: Option<(usize, i64)> = None;
        let mut i = 0usize;
        while i < scores.len() {
            let s = scores[i];
            if s > 0 && best.is_none_or(|(_, bs)| s > bs) {
                best = Some((i, s));
            }
            i += 1;
        }
        best
    }

    /// Run one deterministic pure-integer pointer-choice margin hinge epoch (`DECISION.md:308`).
    /// Updates `err_i8` cells in canonical key order whenever `rival_score + margin_q16 > gold_score`.
    pub fn s4_hinge_step(
        &mut self,
        examples: &[S4HingeExample],
        margin_q16: i64,
        step_i8: i8,
    ) -> Result<S4HingeEpochReport> {
        if step_i8 <= 0 {
            return Err(invalid("S4 hinge step_i8 must be positive"));
        }
        let mut exact_top1 = 0usize;
        let mut total_hinge_loss_q16 = 0u64;
        let mut updates_applied = 0usize;
        for ex in examples {
            if ex.candidates.len() < 2 || ex.gold_index >= ex.candidates.len() {
                return Err(invalid(
                    "invalid S4HingeExample candidate slice or gold_index",
                ));
            }
            let scores = self.vote_candidates(&ex.pair_keys, &ex.candidates, S4ControlMode::Active);
            let gold_ptr = ex.candidates[ex.gold_index];
            let gold_score = scores[ex.gold_index];
            let mut best_rival_idx: Option<usize> = None;
            let mut c = 0usize;
            while c < scores.len() {
                if c != ex.gold_index && best_rival_idx.is_none_or(|r| scores[c] > scores[r]) {
                    best_rival_idx = Some(c);
                }
                c += 1;
            }
            let rival_idx = best_rival_idx.ok_or_else(|| invalid("missing rival candidate"))?;
            let rival_ptr = ex.candidates[rival_idx];
            let rival_score = scores[rival_idx];
            if gold_score > rival_score {
                exact_top1 += 1;
            }
            let hinge = rival_score
                .saturating_add(margin_q16)
                .saturating_sub(gold_score);
            if hinge > 0 {
                total_hinge_loss_q16 = total_hinge_loss_q16.saturating_add(hinge as u64);
                let step = i16::from(step_i8);
                for &key in &ex.pair_keys {
                    let Some(cidx) = self.find_ctx(key) else {
                        continue;
                    };
                    let ctx = &self.contexts[cidx];
                    let end = ctx.offset + ctx.len;
                    let mut idx = ctx.offset;
                    while idx < end {
                        let ptr = (self.entries[idx].role_class, self.entries[idx].span_offset);
                        if ptr == gold_ptr {
                            let next =
                                (i16::from(self.entries[idx].err_i8) + step).clamp(-127, 127);
                            self.entries[idx].err_i8 = next as i8;
                            updates_applied += 1;
                        } else if ptr == rival_ptr {
                            let next =
                                (i16::from(self.entries[idx].err_i8) - step).clamp(-127, 127);
                            self.entries[idx].err_i8 = next as i8;
                            updates_applied += 1;
                        }
                        idx += 1;
                    }
                }
            }
        }
        Ok(S4HingeEpochReport {
            examples: examples.len(),
            exact_top1,
            total_hinge_loss_q16,
            updates_applied,
        })
    }
}

/// Mix Q48 expert masses with Q32 simplex weights using pure integer arithmetic.
pub fn brier_mix_q48(weights_q32: &[u64], expert_masses_q48: &[u64]) -> Result<u64> {
    if weights_q32.is_empty() || weights_q32.len() != expert_masses_q48.len() {
        return Err(invalid("brier_mix_q48 length mismatch or empty"));
    }
    let mut w_sum = 0u64;
    let mut i = 0usize;
    while i < weights_q32.len() {
        w_sum = w_sum
            .checked_add(weights_q32[i])
            .ok_or_else(|| invalid("brier_mix_q48 weight sum overflow"))?;
        i += 1;
    }
    if w_sum != P3_Q32_ONE {
        return Err(invalid("brier_mix_q48 weights must sum to 2^32"));
    }
    let mut acc = 0u128;
    let mut k = 0usize;
    while k < weights_q32.len() {
        acc += u128::from(weights_q32[k]) * u128::from(expert_masses_q48[k]);
        k += 1;
    }
    let mixed = (acc >> 32) as u64;
    Ok(mixed.clamp(1, P3_Q48_ONE))
}

/// Pure-integer Q32 Brier score `(1 - p_y)^2 + rival_sq_sum_q32` (`p_y` in Q16).
pub fn integer_brier_target_q32(target_mass_q48: u64, rival_sq_sum_q32: u64) -> u64 {
    let p_q16 = (target_mass_q48 >> 32).min(P3_Q16_ONE);
    let err = P3_Q16_ONE - p_q16;
    (err * err).saturating_add(rival_sq_sum_q32)
}

/// Exact integer Euclidean projection onto `{w >= floor_q32, sum(w) == 2^32}`.
pub fn project_simplex_q32(raw: &[i128], floor_q32: u64) -> Result<Vec<u64>> {
    let n = raw.len();
    if n == 0 || floor_q32 == 0 {
        return Err(invalid(
            "simplex projection requires non-empty slice and floor_q32 >= 1",
        ));
    }
    let min_sum = u64::try_from(n)
        .ok()
        .and_then(|cnt| cnt.checked_mul(floor_q32))
        .ok_or_else(|| invalid("simplex floor product overflow"))?;
    if min_sum > P3_Q32_ONE {
        return Err(invalid("simplex floor * len exceeds 2^32"));
    }
    #[derive(Clone, Copy)]
    struct ShiftedMass {
        val: i128,
        orig: usize,
    }

    let rem_budget = i128::from(P3_Q32_ONE - min_sum);
    let f = i128::from(floor_q32);
    let mut shifted: Vec<ShiftedMass> = Vec::with_capacity(n);
    let mut idx = 0usize;
    while idx < n {
        shifted.push(ShiftedMass {
            val: raw[idx].saturating_sub(f),
            orig: idx,
        });
        idx += 1;
    }
    shifted.sort_by(|a, b| b.val.cmp(&a.val).then(a.orig.cmp(&b.orig)));

    let mut prefix_sum = 0i128;
    let mut k_keep = 0usize;
    let mut k = 0usize;
    while k < n {
        prefix_sum = prefix_sum.saturating_add(shifted[k].val);
        let k1 = (k + 1) as i128;
        if prefix_sum.saturating_sub(rem_budget) < k1.saturating_mul(shifted[k].val) {
            k_keep = k + 1;
        } else {
            break;
        }
        k += 1;
    }

    let mut out = vec![floor_q32; n];
    if rem_budget == 0 || k_keep == 0 {
        return Ok(out);
    }
    let mut active_sum = 0i128;
    let mut i = 0usize;
    while i < k_keep {
        active_sum = active_sum.saturating_add(shifted[i].val);
        i += 1;
    }
    let excess = active_sum - rem_budget;
    let k1 = k_keep as i128;
    let tau_q = excess.div_euclid(k1);
    let mut tau_r = excess.rem_euclid(k1) as usize;
    let mut total = floor_q32 * ((n - k_keep) as u64);
    let mut j = 0usize;
    while j < k_keep {
        let orig = shifted[j].orig;
        let mut val = shifted[j].val - tau_q;
        if tau_r > 0 && val > 0 {
            val -= 1;
            tau_r -= 1;
        }
        if val > 0 {
            let add = u64::try_from(val).map_err(|_| invalid("simplex add overflow"))?;
            out[orig] += add;
        }
        total += out[orig];
        j += 1;
    }
    if total < P3_Q32_ONE {
        out[shifted[0].orig] += P3_Q32_ONE - total;
    } else if total > P3_Q32_ONE {
        let mut drop = total - P3_Q32_ONE;
        let mut d = k_keep;
        while d > 0 && drop > 0 {
            d -= 1;
            let orig = shifted[d].orig;
            let avail = out[orig] - floor_q32;
            let take = avail.min(drop);
            out[orig] -= take;
            drop -= take;
        }
    }
    Ok(out)
}

/// Apply one signed integer Brier error-write update `trunc_div(target - p, 2^lr_shift)` to an S3/S4 `i8` cell.
pub fn apply_s3_s4_err_write(
    err_cell: i8,
    target_hit: bool,
    mixed_mass_q48: u64,
    lr_shift: u32,
) -> i8 {
    let p_q16 = ((mixed_mass_q48 >> 32).min(P3_Q16_ONE)) as i32;
    let target_q16 = if target_hit { P3_Q16_ONE as i32 } else { 0 };
    let delta = (target_q16 - p_q16) / (1i32 << lr_shift.min(16));
    (i32::from(err_cell) + delta).clamp(-127, 127) as i8
}

/// One Brier error-write training sample over a bucket and its per-expert target Q48 masses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrierSample {
    pub bucket: usize,
    pub target_masses_q48: Vec<u64>,
    pub rival_sq_sum_q32: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrierEpochReport {
    pub epoch: u32,
    pub samples: usize,
    pub mean_brier_before_q32: u64,
    pub mean_brier_after_q32: u64,
    pub buckets_updated: usize,
}

/// Integer `UORS6B01` mixer state updated by Brier error-writing (`DECISION.md:169,309,315`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrierS6Weights {
    pub buckets: usize,
    pub experts: usize,
    pub epoch: u32,
    pub train_brier_q32_mean: u64,
    pub train_sha256: [u8; 32],
    pub weights_q32: Vec<u64>,
}

pub fn encoded_s6_brier_bytes(buckets: usize, experts: usize) -> Result<usize> {
    let slots = checked_len(buckets, experts, "S6 buckets*experts")?;
    let payload = checked_len(slots, 8, "S6 Q32 weights")?;
    add_len(S6_HEADER_BYTES, payload, "S6 total bytes")
}

impl BrierS6Weights {
    /// Initialize uniform Q32 simplex weights across `buckets x experts` (`floor >= 1`).
    pub fn uniform(buckets: usize, experts: usize, train_sha256: [u8; 32]) -> Result<Self> {
        if buckets == 0 || experts == 0 || u64::try_from(experts).is_err() {
            return Err(invalid(
                "BrierS6Weights requires non-zero buckets and experts",
            ));
        }
        let e_u64 = experts as u64;
        let q = P3_Q32_ONE / e_u64;
        let r = (P3_Q32_ONE % e_u64) as usize;
        if q == 0 {
            return Err(invalid("too many experts for Q32 simplex"));
        }
        let total_slots = checked_len(buckets, experts, "uniform S6 slots")?;
        let mut weights_q32 = Vec::with_capacity(total_slots);
        let mut b = 0usize;
        while b < buckets {
            let mut e = 0usize;
            while e < experts {
                weights_q32.push(q + if e < r { 1 } else { 0 });
                e += 1;
            }
            b += 1;
        }
        Ok(Self {
            buckets,
            experts,
            epoch: 0,
            train_brier_q32_mean: 0,
            train_sha256,
            weights_q32,
        })
    }

    pub fn bucket_weights(&self, bucket: usize) -> Result<&[u64]> {
        if bucket >= self.buckets {
            return Err(invalid("S6 bucket out of range"));
        }
        let start = bucket * self.experts;
        Ok(&self.weights_q32[start..start + self.experts])
    }

    /// Mean Q32 Brier score across a sample slice without updating weights.
    pub fn mean_brier_q32(&self, samples: &[BrierSample]) -> Result<u64> {
        if samples.is_empty() {
            return Err(invalid("mean_brier_q32 requires at least one sample"));
        }
        let mut total = 0u128;
        for s in samples {
            if s.target_masses_q48.len() != self.experts {
                return Err(invalid("BrierSample expert count mismatch"));
            }
            let w = self.bucket_weights(s.bucket)?;
            let mixed = brier_mix_q48(w, &s.target_masses_q48)?;
            total += u128::from(integer_brier_target_q32(mixed, s.rival_sq_sum_q32));
        }
        Ok((total / (samples.len() as u128)) as u64)
    }

    /// Execute one deterministic pure-integer Brier LMS error-write epoch (`DECISION.md:169,309`),
    /// projecting every updated bucket back onto the exact `2^32` integer simplex (`w >= floor_q32`).
    pub fn brier_error_write_epoch(
        &mut self,
        samples: &[BrierSample],
        lr_shift: u32,
        floor_q32: u64,
    ) -> Result<BrierEpochReport> {
        if samples.is_empty() {
            return Err(invalid(
                "brier_error_write_epoch requires at least one sample",
            ));
        }
        let grad_slots = checked_len(self.buckets, self.experts, "Brier gradient accumulators")?;
        let mut grads = vec![0i128; grad_slots];
        let mut touched = vec![0u64; self.buckets];
        let mut brier_before = 0u128;

        for s in samples {
            if s.bucket >= self.buckets || s.target_masses_q48.len() != self.experts {
                return Err(invalid("BrierSample bucket or expert mismatch"));
            }
            let start = s.bucket * self.experts;
            let w = &self.weights_q32[start..start + self.experts];
            let mixed_q48 = brier_mix_q48(w, &s.target_masses_q48)?;
            brier_before += u128::from(integer_brier_target_q32(mixed_q48, s.rival_sq_sum_q32));
            let p_q16 = (mixed_q48 >> 32).min(P3_Q16_ONE) as i128;
            let err_q16 = (P3_Q16_ONE as i128) - p_q16;
            let mut e = 0usize;
            while e < self.experts {
                let m_q16 = (s.target_masses_q48[e] >> 32) as i128;
                grads[start + e] += err_q16 * m_q16;
                e += 1;
            }
            touched[s.bucket] += 1;
        }

        let mut buckets_updated = 0usize;
        let mut b = 0usize;
        while b < self.buckets {
            if touched[b] > 0 {
                let start = b * self.experts;
                let denom = i128::from(touched[b]);
                let mut raw = Vec::with_capacity(self.experts);
                let mut e = 0usize;
                while e < self.experts {
                    let mean_grad = grads[start + e] / denom;
                    let step = mean_grad >> lr_shift.min(31);
                    raw.push(i128::from(self.weights_q32[start + e]) + step);
                    e += 1;
                }
                let projected = project_simplex_q32(&raw, floor_q32)?;
                let mut idx = 0usize;
                while idx < self.experts {
                    self.weights_q32[start + idx] = projected[idx];
                    idx += 1;
                }
                buckets_updated += 1;
            }
            b += 1;
        }

        self.epoch = self.epoch.saturating_add(1);
        let mean_before = (brier_before / (samples.len() as u128)) as u64;
        let mean_after = self.mean_brier_q32(samples)?;
        self.train_brier_q32_mean = mean_after;
        Ok(BrierEpochReport {
            epoch: self.epoch,
            samples: samples.len(),
            mean_brier_before_q32: mean_before,
            mean_brier_after_q32: mean_after,
            buckets_updated,
        })
    }

    pub fn encode_bytes(&self) -> Result<Vec<u8>> {
        if self.buckets == 0 || self.experts == 0 {
            return Err(invalid(
                "BrierS6Weights requires non-zero buckets and experts",
            ));
        }
        let expected_slots = checked_len(self.buckets, self.experts, "S6 buckets*experts")?;
        if self.weights_q32.len() != expected_slots {
            return Err(invalid("BrierS6Weights weights_q32 length mismatch"));
        }
        let mut b = 0usize;
        while b < self.buckets {
            let start = b * self.experts;
            let mut sum = 0u64;
            let mut e = 0usize;
            while e < self.experts {
                let w = self.weights_q32[start + e];
                if w == 0 {
                    return Err(invalid("BrierS6Weights bucket contains zero weight"));
                }
                sum = sum
                    .checked_add(w)
                    .ok_or_else(|| invalid("BrierS6Weights bucket sum overflow"))?;
                e += 1;
            }
            if sum != P3_Q32_ONE {
                return Err(invalid("BrierS6Weights bucket sum != 2^32"));
            }
            b += 1;
        }
        let total = encoded_s6_brier_bytes(self.buckets, self.experts)?;
        let b_u32 = u32::try_from(self.buckets).map_err(|_| invalid("buckets > u32::MAX"))?;
        let e_u32 = u32::try_from(self.experts).map_err(|_| invalid("experts > u32::MAX"))?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(MAGIC_S6_BRIER);
        out.extend_from_slice(&P3_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.train_sha256);
        out.extend_from_slice(&b_u32.to_le_bytes());
        out.extend_from_slice(&e_u32.to_le_bytes());
        out.extend_from_slice(&self.train_brier_q32_mean.to_le_bytes());
        for &w in &self.weights_q32 {
            out.extend_from_slice(&w.to_le_bytes());
        }
        if out.len() != total {
            return Err(invalid("internal UORS6B01 byte length mismatch"));
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
            return Err(invalid("UORS6B01 payload SHA-256 mismatch"));
        }
        if bytes.len() < S6_HEADER_BYTES || &bytes[..8] != MAGIC_S6_BRIER {
            return Err(invalid("invalid UORS6B01 magic or header length"));
        }
        let mut off = 8usize;
        let version = read_u32_le(bytes, &mut off)?;
        if version != P3_FORMAT_VERSION {
            return Err(invalid(format!("unsupported UORS6B01 version {version}")));
        }
        let epoch = read_u32_le(bytes, &mut off)?;
        let mut train_sha = [0u8; 32];
        train_sha.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        if train_sha != expected_train_sha {
            return Err(invalid("UORS6B01 train_sha256 mismatch"));
        }
        let buckets = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("UORS6B01 buckets overflows usize"))?;
        let experts = usize::try_from(read_u32_le(bytes, &mut off)?)
            .map_err(|_| invalid("UORS6B01 experts overflows usize"))?;
        let train_brier_q32_mean = read_u64_le(bytes, &mut off)?;
        if buckets == 0 || experts == 0 {
            return Err(invalid("UORS6B01 zero buckets or experts"));
        }
        let expected_len = encoded_s6_brier_bytes(buckets, experts)?;
        if bytes.len() != expected_len {
            return Err(invalid(format!(
                "UORS6B01 byte length {} != expected {}",
                bytes.len(),
                expected_len
            )));
        }
        let total_slots = checked_len(buckets, experts, "UORS6B01 slots")?;
        let mut weights_q32 = Vec::with_capacity(total_slots);
        let mut i = 0usize;
        while i < total_slots {
            weights_q32.push(read_u64_le(bytes, &mut off)?);
            i += 1;
        }
        let mut b = 0usize;
        while b < buckets {
            let mut sum = 0u64;
            let base = b * experts;
            let mut e = 0usize;
            while e < experts {
                let w = weights_q32[base + e];
                if w == 0 {
                    return Err(invalid(
                        "UORS6B01 zero simplex weight violates uniform floor",
                    ));
                }
                sum = sum
                    .checked_add(w)
                    .ok_or_else(|| invalid("UORS6B01 bucket sum overflow"))?;
                e += 1;
            }
            if sum != P3_Q32_ONE {
                return Err(invalid("UORS6B01 bucket weight sum != 2^32"));
            }
            b += 1;
        }
        Ok(Self {
            buckets,
            experts,
            epoch,
            train_brier_q32_mean,
            train_sha256: train_sha,
            weights_q32,
        })
    }
}

/// Configuration for the lawful integer stop & repetition rule (`DECISION.md:310`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LawfulStopConfig {
    pub cycle_n: usize,
    pub max_repeat_run: usize,
    pub rep_penalty_shift: u32,
    pub eos_boost_shift: u32,
    pub max_reply_tokens: usize,
}

impl Default for LawfulStopConfig {
    fn default() -> Self {
        Self {
            cycle_n: 4,
            max_repeat_run: 3,
            rep_penalty_shift: 4,
            eos_boost_shift: 2,
            max_reply_tokens: 256,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopGuardTrace {
    pub suppressed_cycle_tokens: usize,
    pub suppressed_run_token: Option<u16>,
    pub promoted_eos: bool,
    pub greedy_token: u16,
}

/// Apply the pure-integer cycle-breaking, repeat suppression, and EOS termination rule
/// over `scores_q48[0..4096]`. Never zeroes any score below `1` (PRLL floor compliant).
pub fn apply_lawful_stop_and_rep_guard(
    generated: &[u16],
    scores_q48: &mut [u64],
    span_exhausted: bool,
    cfg: LawfulStopConfig,
) -> Result<StopGuardTrace> {
    if scores_q48.len() != P3_VOCABULARY {
        return Err(invalid("scores_q48 must have length 4096"));
    }
    for &t in generated {
        if usize::from(t) >= P3_VOCABULARY {
            return Err(invalid("generated history contains token id >= 4096"));
        }
    }
    if cfg.cycle_n < 2 || cfg.max_repeat_run == 0 || cfg.max_reply_tokens == 0 {
        return Err(invalid("invalid LawfulStopConfig parameters"));
    }

    let shift = cfg.rep_penalty_shift.min(48);
    let mut suppressed_cycle = 0usize;
    let ctx_len = cfg.cycle_n - 1;
    if generated.len() >= cfg.cycle_n {
        let tail_start = generated.len() - ctx_len;
        let mut seen_cycle_next = [false; P3_VOCABULARY];
        let mut pos = 0usize;
        while pos < tail_start {
            let mut matched = true;
            let mut k = 0usize;
            while k < ctx_len {
                if generated[pos + k] != generated[tail_start + k] {
                    matched = false;
                    break;
                }
                k += 1;
            }
            if matched {
                let next_tok = generated[pos + ctx_len];
                seen_cycle_next[usize::from(next_tok)] = true;
            }
            pos += 1;
        }
        let mut v = 0usize;
        while v < P3_VOCABULARY {
            if seen_cycle_next[v] && v != usize::from(P3_EOS) {
                scores_q48[v] = (scores_q48[v] >> shift).max(1);
                suppressed_cycle += 1;
            }
            v += 1;
        }
    }

    let mut suppressed_run_token = None;
    if generated.len() >= cfg.max_repeat_run {
        let last = generated[generated.len() - 1];
        let mut all_same = true;
        let start = generated.len() - cfg.max_repeat_run;
        let mut idx = start;
        while idx < generated.len() {
            if generated[idx] != last {
                all_same = false;
                break;
            }
            idx += 1;
        }
        if all_same && last != P3_EOS {
            let idx = usize::from(last);
            scores_q48[idx] = (scores_q48[idx] >> shift).max(1);
            suppressed_run_token = Some(last);
        }
    }

    let promoted_eos = span_exhausted || generated.len() >= cfg.max_reply_tokens;
    let eos_idx = usize::from(P3_EOS);
    let mut greedy = 0u16;
    let mut best_score = 0u64;
    let vocab_len = scores_q48.len();
    let mut v = 0usize;
    while v < vocab_len {
        let mut s = scores_q48[v];
        if s == 0 {
            s = 1;
            scores_q48[v] = 1;
        }
        if promoted_eos && v != eos_idx && s >= P3_Q48_ONE {
            s = P3_Q48_ONE - 1;
            scores_q48[v] = s;
        }
        if s > best_score {
            best_score = s;
            greedy = v as u16;
        }
        v += 1;
    }

    if promoted_eos {
        let peak = best_score.max(1);
        let boost = peak.saturating_add((peak >> cfg.eos_boost_shift.min(16)).max(1));
        scores_q48[eos_idx] = boost.min(P3_Q48_ONE);
        greedy = P3_EOS;
    }

    Ok(StopGuardTrace {
        suppressed_cycle_tokens: suppressed_cycle,
        suppressed_run_token,
        promoted_eos,
        greedy_token: greedy,
    })
}

/// Exact integer floor square root `floor(sqrt(n))` via binary restoring search.
pub fn isqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut op = n;
    let mut res = 0u64;
    let mut one = 1u64 << 62;
    while one > op {
        one >>= 2;
    }
    while one != 0 {
        if op >= res + one {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }
    res
}

/// Exact integer ceiling square root `ceil(sqrt(n))`.
pub fn isqrt_ceil_u64(n: u64) -> u64 {
    let fl = isqrt_u64(n);
    if fl * fl == n {
        fl
    } else {
        fl + 1
    }
}

/// Paired McNemar-style band statistics (`DECISION.md:98,313`: `2 * sqrt(b + c)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairedBandStats {
    pub total: usize,
    pub active_correct: usize,
    pub control_correct: usize,
    pub wins_active: usize,
    pub wins_control: usize,
    pub net_gain: i64,
    pub band_ceil: u64,
    pub beats_control_by_band: bool,
}

pub fn paired_band_compare(active_hits: &[bool], control_hits: &[bool]) -> Result<PairedBandStats> {
    if active_hits.len() != control_hits.len() {
        return Err(invalid("paired_band_compare hit slice length mismatch"));
    }
    let mut active_correct = 0usize;
    let mut control_correct = 0usize;
    let mut wins_active = 0usize;
    let mut wins_control = 0usize;
    let mut i = 0usize;
    while i < active_hits.len() {
        let a = active_hits[i];
        let c = control_hits[i];
        if a {
            active_correct += 1;
        }
        if c {
            control_correct += 1;
        }
        if a && !c {
            wins_active += 1;
        } else if !a && c {
            wins_control += 1;
        }
        i += 1;
    }
    let net_gain = (wins_active as i64) - (wins_control as i64);
    let discordant = (wins_active + wins_control) as u64;
    let band_ceil = 2u64.saturating_mul(isqrt_ceil_u64(discordant));
    Ok(PairedBandStats {
        total: active_hits.len(),
        active_correct,
        control_correct,
        wins_active,
        wins_control,
        net_gain,
        band_ceil,
        beats_control_by_band: net_gain > (band_ceil as i64),
    })
}
