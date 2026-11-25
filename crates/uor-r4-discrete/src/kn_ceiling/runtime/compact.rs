const MAGIC: &[u8; 8] = b"UORKNC02";
const MAGIC_V3: &[u8; 8] = b"UORKNC03";
const KEY_MASK: u128 = (1 << 120) - 1;
const INLINE_PAIRS: [(u32, u32); 15] = [
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 1),
    (5, 2),
    (5, 3),
    (6, 1),
    (6, 2),
];

fn key_width(order: u32) -> usize {
    (order as usize * 12).div_ceil(8)
}

fn key_body(word: u128, order: u32) -> [u8; 16] {
    let shift = 120 - order * 12;
    ((word & KEY_MASK) >> shift).to_be_bytes()
}

fn follower_width(shared: bool) -> usize {
    if shared {
        6
    } else {
        10
    }
}

fn take<'a>(remaining: &mut &'a [u8], size: usize) -> Result<&'a [u8]> {
    let (head, tail) = remaining
        .split_at_checked(size)
        .ok_or_else(|| invalid("truncated compact count lookup"))?;
    *remaining = tail;
    Ok(head)
}

fn word<const N: usize>(remaining: &mut &[u8]) -> Result<[u8; N]> {
    take(remaining, N)?
        .try_into()
        .map_err(|_| invalid("invalid compact count word"))
}

struct CompactOrder<'a> {
    contexts: &'a [u8],
    followers: &'a [u8],
    key_width: usize,
    shared: bool,
    discounts: KnDiscounts,
}

impl CompactOrder<'_> {
    fn stride(&self) -> usize {
        self.key_width + 4
    }

    fn context_key(&self, index: usize) -> &[u8] {
        let start = index * self.stride();
        &self.contexts[start..start + self.key_width]
    }

    fn context_offset(&self, index: usize) -> usize {
        let end = (index + 1) * self.stride();
        if end > self.contexts.len() {
            return self.followers.len() / follower_width(self.shared);
        }
        let offset = &self.contexts[end - 4..end];
        u32::from_le_bytes([offset[0], offset[1], offset[2], offset[3]]) as usize
    }

    fn follower_rows(&self, context: usize) -> Option<&[u8]> {
        let width = follower_width(self.shared);
        let start = self.context_offset(context);
        let end = self.context_offset(context + 1);
        // Every valid context has at least one and at most 4,095 successors.
        if end <= start || end - start >= NGRAM_VOCABULARY as usize {
            return None;
        }
        self.followers
            .get(start.checked_mul(width)?..end.checked_mul(width)?)
    }

    fn read_follower(&self, row: &[u8]) -> (u16, u32, u32) {
        let next = u16::from_le_bytes([row[0], row[1]]);
        let raw = u32::from_le_bytes([row[2], row[3], row[4], row[5]]);
        let kn = if self.shared {
            raw
        } else {
            u32::from_le_bytes([row[6], row[7], row[8], row[9]])
        };
        (next, raw, kn)
    }

    fn validate(&self, order: u32, raw: [u64; 4], kn: [u64; 4]) -> Result<()> {
        if self.context_offset(0) != 0 {
            return Err(invalid("compact lookup does not begin at row zero"));
        }
        let mut spectra = [[0u64; 4]; 2];
        let mut previous = None;
        for i in 0..self.contexts.len() / self.stride() {
            let key = self.context_key(i);
            if (order % 2 == 1 && key[0] > 0x0f) || previous.is_some_and(|prior| prior >= key) {
                return Err(invalid("invalid or unordered compact context keys"));
            }
            previous = Some(key);
            let rows = self
                .follower_rows(i)
                .ok_or_else(|| invalid("invalid compact follower span"))?;
            let mut previous_next = NGRAM_BOS;
            for row in rows.chunks_exact(follower_width(self.shared)) {
                let (next, raw, kn) = self.read_follower(row);
                if next <= previous_next
                    || u32::from(next) >= NGRAM_VOCABULARY
                    || raw == 0
                    || kn == 0
                {
                    return Err(invalid("invalid or unordered compact follower"));
                }
                previous_next = next;
                for (spectrum, count) in spectra.iter_mut().zip([raw, kn]) {
                    if count <= 4 {
                        spectrum[count as usize - 1] += 1;
                    }
                }
            }
        }
        if spectra != [raw, kn] {
            return Err(invalid("compact count spectrum does not match followers"));
        }
        Ok(())
    }

    fn matched_rows(&self, search_key: &[u8]) -> Option<&[u8]> {
        let n = self.contexts.len() / self.stride();
        let (mut left, mut right) = (0, n);
        while left < right {
            let middle = left + (right - left) / 2;
            if self.context_key(middle) < search_key {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        if left == n || self.context_key(left) != search_key {
            return None;
        }
        self.follower_rows(left)
    }

    fn lookup(&self, search_key: &[u8], target: u16) -> Option<(u32, u64, u32, NgramContext)> {
        let mut raw_total = 0;
        let mut raw_target = 0;
        let mut kn_target = 0;
        let mut context = NgramContext {
            key: 0,
            total: 0,
            followers: 0,
            n1: 0,
            n2: 0,
            n3_plus: 0,
        };
        for row in self
            .matched_rows(search_key)?
            .chunks_exact(follower_width(self.shared))
        {
            let (next, raw, kn) = self.read_follower(row);
            raw_total += u64::from(raw);
            context.total += u64::from(kn);
            context.followers += 1;
            match kn {
                1 => context.n1 += 1,
                2 => context.n2 += 1,
                _ => context.n3_plus += 1,
            }
            if next == target {
                raw_target = raw;
                kn_target = kn;
            }
        }
        Some((raw_target, raw_total, kn_target, context))
    }
}

struct CompactV3Order<'a> {
    order: u32,
    shared: bool,
    contexts: usize,
    followers: usize,
    parent_contexts: usize,
    discounts: KnDiscounts,
    parent_sb: &'a [u8],
    parent_deg: &'a [u8],
    parent_deg_esc: &'a [u8],
    child_toks: &'a [u8],
    fol_sb: &'a [u8],
    fol_deg: &'a [u8],
    fol_deg_esc: &'a [u8],
    fol_words: &'a [u8],
    esc_sb: &'a [u8],
    esc16: &'a [u8],
    esc32: &'a [u8],
}

fn read_u16_le(slice: &[u8], idx: usize) -> u16 {
    let at = idx * 2;
    u16::from_le_bytes([slice[at], slice[at + 1]])
}

fn read_u32_le(slice: &[u8], idx: usize) -> u32 {
    let at = idx * 4;
    u32::from_le_bytes([slice[at], slice[at + 1], slice[at + 2], slice[at + 3]])
}

fn find_esc_deg(esc: &[u8], count: usize, idx: u32) -> Option<u32> {
    let mut left = 0usize;
    let mut right = count;
    while left < right {
        let mid = left + (right - left) / 2;
        let at = mid * 6;
        let k = u32::from_le_bytes([esc[at], esc[at + 1], esc[at + 2], esc[at + 3]]);
        if k < idx {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    if left < count {
        let at = left * 6;
        let k = u32::from_le_bytes([esc[at], esc[at + 1], esc[at + 2], esc[at + 3]]);
        if k == idx {
            return Some(u32::from(u16::from_le_bytes([esc[at + 4], esc[at + 5]])));
        }
    }
    None
}

fn span_from_sb(
    sb: &[u8],
    deg: &[u8],
    esc: &[u8],
    idx: usize,
    total_items: usize,
    bias: u32,
) -> Option<(usize, usize)> {
    if idx >= total_items {
        return None;
    }
    let block = idx / 64;
    let mut off = read_u32_le(sb, block) as usize;
    let esc_count = esc.len() / 6;
    let start_in_block = block * 64;
    for j in start_in_block..idx {
        let d = deg[j];
        let val = if d < 255 {
            u32::from(d) + bias
        } else {
            find_esc_deg(esc, esc_count, j as u32)?
        };
        off = off.checked_add(val as usize)?;
    }
    let d_self = deg[idx];
    let deg_self = if d_self < 255 {
        u32::from(d_self) + bias
    } else {
        find_esc_deg(esc, esc_count, idx as u32)?
    };
    let end = off.checked_add(deg_self as usize)?;
    Some((off, end))
}

impl<'a> CompactV3Order<'a> {
    fn from_wire(
        order: u32,
        shared: bool,
        parent_contexts: usize,
        remaining: &mut &'a [u8],
    ) -> Result<Self> {
        let followers = u32::from_le_bytes(word(remaining)?) as usize;
        let contexts = u32::from_le_bytes(word(remaining)?) as usize;
        if (followers == 0) != (contexts == 0) {
            return Err(invalid("invalid v3 compact order lengths"));
        }
        if order == 0 && contexts > 1 {
            return Err(invalid("v3 order 0 cannot exceed one context"));
        }
        let mut spectra = [[0u64; 4]; 2];
        for count in spectra.iter_mut().flatten() {
            *count = u64::from_le_bytes(word(remaining)?);
        }
        let parent_deg_esc_len = u32::from_le_bytes(word(remaining)?) as usize;
        let fol_deg_esc_len = u32::from_le_bytes(word(remaining)?) as usize;
        let esc16_len = u32::from_le_bytes(word(remaining)?) as usize;
        let esc32_len = u32::from_le_bytes(word(remaining)?) as usize;
        if order == 0 && parent_deg_esc_len != 0 {
            return Err(invalid("v3 order 0 cannot have parent degree escapes"));
        }
        let (discounts, _) = KnDiscounts::with_fallback(spectra[1])?;
        let mul = |count: usize, width: usize| -> Result<usize> {
            count
                .checked_mul(width)
                .ok_or_else(|| invalid("v3 compact section size overflow"))
        };
        let (parent_sb, parent_deg, parent_deg_esc, child_toks) = if order == 0 {
            (&[][..], &[][..], &[][..], &[][..])
        } else {
            let psb = take(remaining, mul(parent_contexts.div_ceil(64) + 1, 4)?)?;
            let pdeg = take(remaining, parent_contexts)?;
            let pdesc = take(remaining, mul(parent_deg_esc_len, 6)?)?;
            let cbytes = contexts
                .checked_mul(3)
                .map(|b| b.div_ceil(2))
                .ok_or_else(|| invalid("v3 child tokens overflow"))?;
            let ctoks = take(remaining, cbytes)?;
            (psb, pdeg, pdesc, ctoks)
        };
        let fol_sb = take(remaining, mul(contexts.div_ceil(64) + 1, 4)?)?;
        let fol_deg = take(remaining, contexts)?;
        let fol_deg_esc = take(remaining, mul(fol_deg_esc_len, 6)?)?;
        let fol_words = take(remaining, mul(followers, 2)?)?;
        let esc_sb = take(remaining, mul(followers.div_ceil(64) + 1, 4)?)?;
        let esc16 = take(remaining, mul(esc16_len, if shared { 2 } else { 4 })?)?;
        let esc32 = take(remaining, mul(esc32_len, if shared { 8 } else { 12 })?)?;
        let table = Self {
            order,
            shared,
            contexts,
            followers,
            parent_contexts,
            discounts,
            parent_sb,
            parent_deg,
            parent_deg_esc,
            child_toks,
            fol_sb,
            fol_deg,
            fol_deg_esc,
            fol_words,
            esc_sb,
            esc16,
            esc32,
        };
        table.validate(spectra[0], spectra[1])?;
        Ok(table)
    }

    fn child_tok(&self, child_idx: usize) -> u16 {
        let base = (child_idx / 2) * 3;
        if child_idx % 2 == 0 {
            u16::from(self.child_toks[base]) | (u16::from(self.child_toks[base + 1] & 0x0f) << 8)
        } else {
            (u16::from(self.child_toks[base + 1]) >> 4)
                | (u16::from(self.child_toks[base + 2]) << 4)
        }
    }

    fn child_span(&self, parent_idx: usize) -> Option<(usize, usize)> {
        if self.order == 0 {
            return None;
        }
        span_from_sb(
            self.parent_sb,
            self.parent_deg,
            self.parent_deg_esc,
            parent_idx,
            self.parent_contexts,
            0,
        )
    }

    fn find_child(&self, parent_idx: usize, tok: u16) -> Option<usize> {
        let (start, end) = self.child_span(parent_idx)?;
        let (mut left, mut right) = (start, end);
        while left < right {
            let mid = left + (right - left) / 2;
            if self.child_tok(mid) < tok {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left < end && self.child_tok(left) == tok {
            Some(left)
        } else {
            None
        }
    }

    fn follower_span(&self, ctx_idx: usize) -> Option<(usize, usize)> {
        span_from_sb(
            self.fol_sb,
            self.fol_deg,
            self.fol_deg_esc,
            ctx_idx,
            self.contexts,
            1,
        )
    }

    fn esc_rank_at(&self, follower_idx: usize) -> usize {
        let block = follower_idx / 64;
        let mut rank = read_u32_le(self.esc_sb, block) as usize;
        for j in (block * 64)..follower_idx {
            if (read_u16_le(self.fol_words, j) & 0x0f) == 15 {
                rank += 1;
            }
        }
        rank
    }

    fn find_esc32(&self, follower_idx: u32) -> Option<(u32, u32)> {
        let stride = if self.shared { 8 } else { 12 };
        let count = self.esc32.len() / stride;
        let mut left = 0usize;
        let mut right = count;
        while left < right {
            let mid = left + (right - left) / 2;
            let k = read_u32_le(self.esc32, mid * (stride / 4));
            if k < follower_idx {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left < count {
            let w = left * (stride / 4);
            if read_u32_le(self.esc32, w) == follower_idx {
                let raw = read_u32_le(self.esc32, w + 1);
                let kn = if self.shared {
                    raw
                } else {
                    read_u32_le(self.esc32, w + 2)
                };
                return Some((raw, kn));
            }
        }
        None
    }

    fn read_follower_counts(
        &self,
        follower_idx: usize,
        esc_rank: &mut usize,
    ) -> Option<(u16, u32, u32)> {
        let word = read_u16_le(self.fol_words, follower_idx);
        let next = word >> 4;
        let code = (word & 0x0f) as usize;
        if code < 15 {
            let (raw, kn) = if self.shared {
                let c = code as u32 + 1;
                (c, c)
            } else {
                INLINE_PAIRS[code]
            };
            return Some((next, raw, kn));
        }
        let rank = *esc_rank;
        *esc_rank += 1;
        let (raw, kn) = if self.shared {
            if rank * 2 + 2 > self.esc16.len() {
                return None;
            }
            let r16 = read_u16_le(self.esc16, rank);
            if r16 != 0 {
                (u32::from(r16), u32::from(r16))
            } else {
                self.find_esc32(follower_idx as u32)?
            }
        } else {
            if rank * 4 + 4 > self.esc16.len() {
                return None;
            }
            let r16 = read_u16_le(self.esc16, rank * 2);
            let k16 = read_u16_le(self.esc16, rank * 2 + 1);
            if r16 != 0 {
                if k16 == 0 {
                    return None;
                }
                (u32::from(r16), u32::from(k16))
            } else {
                if k16 != 0 {
                    return None;
                }
                self.find_esc32(follower_idx as u32)?
            }
        };
        if raw == 0 || kn == 0 {
            return None;
        }
        Some((next, raw, kn))
    }

    fn validate(&self, expected_raw: [u64; 4], expected_kn: [u64; 4]) -> Result<()> {
        if self.order >= 1 {
            if read_u32_le(self.parent_sb, 0) != 0
                || read_u32_le(self.parent_sb, self.parent_contexts.div_ceil(64)) as usize
                    != self.contexts
            {
                return Err(invalid("v3 parent superblock bounds mismatch"));
            }
            if self.contexts % 2 == 1 && (self.child_toks.last().copied().unwrap_or(0) >> 4) != 0 {
                return Err(invalid("v3 odd child_toks has nonzero padding nibble"));
            }
            let p_esc_len = self.parent_deg_esc.len() / 6;
            let mut p_esc_cur = 0usize;
            let mut prev_esc_parent: Option<u32> = None;
            let mut child_off = 0usize;
            for b in 0..self.parent_contexts.div_ceil(64) {
                if read_u32_le(self.parent_sb, b) as usize != child_off {
                    return Err(invalid("v3 parent superblock offset mismatch"));
                }
                let end_p = ((b + 1) * 64).min(self.parent_contexts);
                for p in (b * 64)..end_p {
                    let d = self.parent_deg[p];
                    let deg = if d < 255 {
                        usize::from(d)
                    } else {
                        if p_esc_cur >= p_esc_len {
                            return Err(invalid("v3 missing parent degree escape"));
                        }
                        let at = p_esc_cur * 6;
                        let key = u32::from_le_bytes([
                            self.parent_deg_esc[at],
                            self.parent_deg_esc[at + 1],
                            self.parent_deg_esc[at + 2],
                            self.parent_deg_esc[at + 3],
                        ]);
                        let val = u16::from_le_bytes([
                            self.parent_deg_esc[at + 4],
                            self.parent_deg_esc[at + 5],
                        ]);
                        if prev_esc_parent.is_some_and(|prior| prior >= key)
                            || key as usize != p
                            || !(255..=NGRAM_VOCABULARY as usize).contains(&usize::from(val))
                        {
                            return Err(invalid("v3 invalid parent degree escape"));
                        }
                        prev_esc_parent = Some(key);
                        p_esc_cur += 1;
                        usize::from(val)
                    };
                    if deg > NGRAM_VOCABULARY as usize || child_off + deg > self.contexts {
                        return Err(invalid("v3 parent child degree out of range"));
                    }
                    let mut prev_tok: Option<u16> = None;
                    for c in child_off..(child_off + deg) {
                        let tok = self.child_tok(c);
                        if prev_tok.is_some_and(|prior| prior >= tok) {
                            return Err(invalid("v3 unsorted child token in parent span"));
                        }
                        prev_tok = Some(tok);
                    }
                    child_off += deg;
                }
            }
            if child_off != self.contexts || p_esc_cur != p_esc_len {
                return Err(invalid("v3 unconsumed parent contexts or escapes"));
            }
        }

        if read_u32_le(self.fol_sb, 0) != 0
            || read_u32_le(self.fol_sb, self.contexts.div_ceil(64)) as usize != self.followers
            || read_u32_le(self.esc_sb, 0) != 0
            || read_u32_le(self.esc_sb, self.followers.div_ceil(64)) as usize
                != self.esc16.len() / if self.shared { 2 } else { 4 }
        {
            return Err(invalid("v3 follower or escape superblock bounds mismatch"));
        }

        let f_esc_len = self.fol_deg_esc.len() / 6;
        let esc16_len = self.esc16.len() / if self.shared { 2 } else { 4 };
        let esc32_stride = if self.shared { 8 } else { 12 };
        let esc32_len = self.esc32.len() / esc32_stride;
        let mut f_esc_cur = 0usize;
        let mut prev_esc_ctx: Option<u32> = None;
        let mut esc16_cur = 0usize;
        let mut esc32_cur = 0usize;
        let mut fol_off = 0usize;
        let mut spectra = [[0u64; 4]; 2];

        for b in 0..self.contexts.div_ceil(64) {
            if read_u32_le(self.fol_sb, b) as usize != fol_off {
                return Err(invalid("v3 follower superblock offset mismatch"));
            }
            let end_c = ((b + 1) * 64).min(self.contexts);
            for ctx in (b * 64)..end_c {
                let d = self.fol_deg[ctx];
                let deg = if d < 255 {
                    usize::from(d) + 1
                } else {
                    if f_esc_cur >= f_esc_len {
                        return Err(invalid("v3 missing follower degree escape"));
                    }
                    let at = f_esc_cur * 6;
                    let key = u32::from_le_bytes([
                        self.fol_deg_esc[at],
                        self.fol_deg_esc[at + 1],
                        self.fol_deg_esc[at + 2],
                        self.fol_deg_esc[at + 3],
                    ]);
                    let val =
                        u16::from_le_bytes([self.fol_deg_esc[at + 4], self.fol_deg_esc[at + 5]]);
                    if prev_esc_ctx.is_some_and(|prior| prior >= key)
                        || key as usize != ctx
                        || !(256..NGRAM_VOCABULARY as usize).contains(&usize::from(val))
                    {
                        return Err(invalid("v3 invalid follower degree escape"));
                    }
                    prev_esc_ctx = Some(key);
                    f_esc_cur += 1;
                    usize::from(val)
                };
                if fol_off + deg > self.followers {
                    return Err(invalid("v3 follower span exceeds total followers"));
                }
                let mut prev_next = NGRAM_BOS;
                for f in fol_off..(fol_off + deg) {
                    if f % 64 == 0 && read_u32_le(self.esc_sb, f / 64) as usize != esc16_cur {
                        return Err(invalid("v3 esc_sb offset mismatch"));
                    }
                    let w = read_u16_le(self.fol_words, f);
                    let code = (w & 0x0f) as usize;
                    let is_esc32 = if code == 15 {
                        if self.shared {
                            esc16_cur < esc16_len && read_u16_le(self.esc16, esc16_cur) == 0
                        } else {
                            esc16_cur < esc16_len && read_u16_le(self.esc16, esc16_cur * 2) == 0
                        }
                    } else {
                        false
                    };
                    let (next, raw, kn) = self
                        .read_follower_counts(f, &mut esc16_cur)
                        .ok_or_else(|| invalid("v3 invalid compact follower"))?;
                    if is_esc32 {
                        if esc32_cur >= esc32_len {
                            return Err(invalid("v3 esc32 underflow"));
                        }
                        let w32 = esc32_cur * (esc32_stride / 4);
                        if read_u32_le(self.esc32, w32) as usize != f {
                            return Err(invalid("v3 esc32 order/index mismatch"));
                        }
                        if raw <= 65535 && kn <= 65535 {
                            return Err(invalid("v3 non-minimal esc32 entry"));
                        }
                        esc32_cur += 1;
                    }
                    if next <= prev_next || u32::from(next) >= NGRAM_VOCABULARY {
                        return Err(invalid("v3 unsorted or invalid compact follower token"));
                    }
                    prev_next = next;
                    for (spectrum, count) in spectra.iter_mut().zip([raw, kn]) {
                        if count <= 4 {
                            spectrum[count as usize - 1] += 1;
                        }
                    }
                }
                fol_off += deg;
            }
        }
        if fol_off != self.followers
            || f_esc_cur != f_esc_len
            || esc16_cur != esc16_len
            || esc32_cur != esc32_len
            || spectra != [expected_raw, expected_kn]
        {
            return Err(invalid("v3 compact spectrum or escape cursor mismatch"));
        }
        Ok(())
    }

    fn lookup_ctx(&self, ctx_idx: usize, target: u16) -> Option<(u32, u64, u32, NgramContext)> {
        let (start, end) = self.follower_span(ctx_idx)?;
        let mut esc_rank = self.esc_rank_at(start);
        let mut raw_total = 0u64;
        let mut raw_target = 0u32;
        let mut kn_target = 0u32;
        let mut context = NgramContext {
            key: 0,
            total: 0,
            followers: 0,
            n1: 0,
            n2: 0,
            n3_plus: 0,
        };
        for f in start..end {
            let (next, raw, kn) = self.read_follower_counts(f, &mut esc_rank)?;
            raw_total += u64::from(raw);
            context.total += u64::from(kn);
            context.followers += 1;
            match kn {
                1 => context.n1 += 1,
                2 => context.n2 += 1,
                _ => context.n3_plus += 1,
            }
            if next == target {
                raw_target = raw;
                kn_target = kn;
            }
        }
        Some((raw_target, raw_total, kn_target, context))
    }
}

enum LookupOrders<'a> {
    V2(Vec<CompactOrder<'a>>),
    V3(Vec<CompactV3Order<'a>>),
}

/// Borrow a validated complete count index in the serialized representation.
/// Construction validates every row; each lookup uses stack scalars only.
pub struct CompactCountLookup<'a> {
    orders: LookupOrders<'a>,
}

impl<'a> CompactCountLookup<'a> {
    pub fn from_bytes(
        bytes: &'a [u8],
        expected_training: [u8; 32],
        expected_artifact: [u8; 32],
    ) -> Result<Self> {
        if Sha256::digest(bytes)[..] != expected_artifact {
            return Err(invalid("compact count artifact digest mismatch"));
        }
        let mut remaining = bytes;
        let magic = word::<8>(&mut remaining)?;
        let is_v3 = match &magic {
            m if m == MAGIC => false,
            m if m == MAGIC_V3 => true,
            _ => return Err(invalid("wrong compact count lookup format")),
        };
        let highest = u32::from_le_bytes(word(&mut remaining)?);
        if highest > NGRAM_MAX_ORDER || word::<32>(&mut remaining)? != expected_training {
            return Err(invalid("invalid order or compact count training binding"));
        }
        let orders = if !is_v3 {
            let mut v2 = Vec::with_capacity(highest as usize + 1);
            for order in 0..=highest {
                let followers = u32::from_le_bytes(word(&mut remaining)?) as usize;
                let contexts = u32::from_le_bytes(word(&mut remaining)?) as usize;
                let mut spectra = [[0u64; 4]; 2];
                for count in spectra.iter_mut().flatten() {
                    *count = u64::from_le_bytes(word(&mut remaining)?);
                }
                let key_width = key_width(order);
                let shared = order == highest;
                let size = |count: usize, width| {
                    count
                        .checked_mul(width)
                        .ok_or_else(|| invalid("compact count table length overflow"))
                };
                let (discounts, _) = KnDiscounts::with_fallback(spectra[1])?;
                let table = CompactOrder {
                    contexts: take(&mut remaining, size(contexts, key_width + 4)?)?,
                    followers: take(&mut remaining, size(followers, follower_width(shared))?)?,
                    key_width,
                    shared,
                    discounts,
                };
                table.validate(order, spectra[0], spectra[1])?;
                v2.push(table);
            }
            LookupOrders::V2(v2)
        } else {
            let mut v3 = Vec::with_capacity(highest as usize + 1);
            let mut parent_contexts = 0usize;
            for order in 0..=highest {
                let shared = order == highest;
                let table =
                    CompactV3Order::from_wire(order, shared, parent_contexts, &mut remaining)?;
                parent_contexts = table.contexts;
                v3.push(table);
            }
            LookupOrders::V3(v3)
        };
        if !remaining.is_empty() {
            return Err(invalid("trailing bytes in compact count lookup"));
        }
        Ok(Self { orders })
    }

    /// Visit the raw and adjusted count words in canonical (order, key, next)
    /// order, using the same validated tables and row decoder as prediction.
    pub fn visit_counts(&self, mut visit: impl FnMut(u32, u128, u16, u32, u32)) -> Result<()> {
        match &self.orders {
            LookupOrders::V2(orders) => {
                for (k, order) in orders.iter().enumerate() {
                    let k = k as u32;
                    let width = follower_width(order.shared);
                    for context in 0..order.contexts.len() / order.stride() {
                        let mut body = [0u8; 16];
                        body[16 - order.key_width..].copy_from_slice(order.context_key(context));
                        let key =
                            (u128::from(k) << 120) | (u128::from_be_bytes(body) << (120 - 12 * k));
                        let followers = order.follower_rows(context).ok_or_else(|| {
                            invalid("invalid compact follower span during readback")
                        })?;
                        for row in followers.chunks_exact(width) {
                            let (next, raw, adjusted) = order.read_follower(row);
                            visit(k, key, next, raw, adjusted);
                        }
                    }
                }
            }
            LookupOrders::V3(orders) => {
                let mut prev_keys: Vec<u128> = Vec::new();
                for (k, order) in orders.iter().enumerate() {
                    let mut next_keys = Vec::with_capacity(order.contexts);
                    let mut esc_rank = 0usize;
                    if k == 0 {
                        for ctx in 0..order.contexts {
                            next_keys.push(0);
                            let (f_start, f_end) = order
                                .follower_span(ctx)
                                .ok_or_else(|| invalid("invalid v3 follower span"))?;
                            for f in f_start..f_end {
                                let (next, raw, adjusted) = order
                                    .read_follower_counts(f, &mut esc_rank)
                                    .ok_or_else(|| invalid("invalid v3 follower"))?;
                                visit(0, 0, next, raw, adjusted);
                            }
                        }
                    } else {
                        let ku = k as u32;
                        for (p_idx, &p_key) in prev_keys.iter().enumerate() {
                            let (c_start, c_end) = order
                                .child_span(p_idx)
                                .ok_or_else(|| invalid("invalid v3 child span"))?;
                            for ctx in c_start..c_end {
                                let tok = order.child_tok(ctx);
                                let key = (p_key & KEY_MASK)
                                    | (u128::from(ku) << 120)
                                    | (u128::from(tok) << (120 - 12 * ku));
                                next_keys.push(key);
                                let (f_start, f_end) = order
                                    .follower_span(ctx)
                                    .ok_or_else(|| invalid("invalid v3 follower span"))?;
                                for f in f_start..f_end {
                                    let (next, raw, adjusted) = order
                                        .read_follower_counts(f, &mut esc_rank)
                                        .ok_or_else(|| invalid("invalid v3 follower"))?;
                                    visit(ku, key, next, raw, adjusted);
                                }
                            }
                        }
                    }
                    prev_keys = next_keys;
                }
            }
        }
        Ok(())
    }

    /// Number of stored order sections, including order zero.
    pub fn order_count(&self) -> usize {
        match &self.orders {
            LookupOrders::V2(o) => o.len(),
            LookupOrders::V3(o) => o.len(),
        }
    }

    /// Fill per-order raw count and total words and return the original KN Q48
    /// mass, using the caller's causal history and target ID at `at`.
    pub fn lookup(
        &self,
        tokens: &[u16],
        at: usize,
        counts: &mut [u32],
        totals: &mut [u64],
    ) -> Result<u64> {
        let target = *tokens
            .get(at)
            .ok_or_else(|| invalid("missing compact lookup target"))?;
        if target == NGRAM_BOS || u32::from(target) >= NGRAM_VOCABULARY {
            return Err(invalid(
                "compact lookup target is not a valid non-BOS token",
            ));
        }
        if counts.len() != self.order_count() || totals.len() != self.order_count() {
            return Err(invalid("wrong compact lookup output width"));
        }
        counts.fill(0);
        totals.fill(0);
        let mut mass = u128::from(KN_BASE_MASS);
        match &self.orders {
            LookupOrders::V2(orders) => {
                for (k, order) in orders.iter().take(at + 1).enumerate() {
                    let key = PrimeRouteKey::from_context(&tokens[at - k..at])?;
                    let body = key_body(key.word(), k as u32);
                    if let Some((count, total, adjusted, context)) =
                        order.lookup(&body[16 - order.key_width..], target)
                    {
                        counts[k] = count;
                        totals[k] = total;
                        mass = interpolate_order(mass, adjusted, context, order.discounts);
                    }
                }
            }
            LookupOrders::V3(orders) => {
                for &tok in &tokens[at.saturating_sub(orders.len().saturating_sub(1))..at] {
                    if u32::from(tok) >= NGRAM_VOCABULARY {
                        return Err(invalid("token id is outside the 12-bit vocabulary"));
                    }
                }
                let mut ctx_idx = 0usize;
                for (k, order) in orders.iter().take(at + 1).enumerate() {
                    if k == 0 {
                        if order.contexts == 0 {
                            break;
                        }
                        ctx_idx = 0;
                    } else {
                        let child_tok = tokens[at - k];
                        let Some(next_ctx) = order.find_child(ctx_idx, child_tok) else {
                            break;
                        };
                        ctx_idx = next_ctx;
                    }
                    let (count, total, adjusted, context) = order
                        .lookup_ctx(ctx_idx, target)
                        .ok_or_else(|| invalid("invalid v3 compact follower span"))?;
                    counts[k] = count;
                    totals[k] = total;
                    mass = interpolate_order(mass, adjusted, context, order.discounts);
                }
            }
        }
        u64::try_from(mass.max(1)).map_err(|_| invalid("compact KN mass exceeded u64"))
    }
}
