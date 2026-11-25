#![forbid(unsafe_code)]
//! Pure-integer 4 KiB geometric virtual-RAM backing store for `core.wasm`.
//!
//! Adapted from `afflom/vRAM` (`device/src/*`, MIT OR Apache-2.0) without external
//! dependencies, floating-point arithmetic, or `vTensor` imports:
//! - Uniform `Fill`
//! - Periodically folded & shifted `Repeated`
//! - Base + exception `Sparse`
//! - 64x64 shared binary column mask `Selector` & `Corrected`
//! - Top-16 4-bit nibble palette `Packed`
//! - Content-addressed `Shared` literal pool & `Residual` deltas with weak reclamation
//!
//! Every candidate representation is decoded and compared byte-for-byte against the
//! submitted 4 KiB page before admission.

use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

use serde_json::{json, Value as JsonValue};

pub const PAGE: usize = 4096;
const SIDE: usize = 64;
const GROUP: usize = 64;
const FANOUT: usize = 256;
const LEAF: usize = GROUP * FANOUT;

pub const MAX_STORES: usize = 16;
pub const DEFAULT_MAX_PAGES: usize = 65_536; // 256 MiB logical
pub const HARD_MAX_PAGES: usize = 1_048_576; // 4 GiB logical sparse
pub const DEFAULT_MAX_BACKING_BYTES: usize = 64 * 1024 * 1024;
pub const HARD_MAX_BACKING_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_TRANSFER_BYTES: usize = 16 * 1024 * 1024;

pub const TAG_FILL: u8 = 0;
pub const TAG_REPEATED: u8 = 1;
pub const TAG_SPARSE: u8 = 2;
pub const TAG_SELECTOR: u8 = 3;
pub const TAG_CORRECTED: u8 = 4;
pub const TAG_PACKED: u8 = 5;
pub const TAG_SHARED: u8 = 6;
pub const TAG_RESIDUAL: u8 = 7;

pub const ERR_INVALID_HANDLE: i32 = -1;
pub const ERR_OUT_OF_RANGE: i32 = -2;
pub const ERR_INVALID_ARG: i32 = -3;
pub const ERR_CAPACITY: i32 = -4;
pub const ERR_INTERNAL: i32 = -5;

const FILL_CHARGE: usize = 1;
const REPEATED_FIXED_CHARGE: usize = 2;
const SPARSE_FIXED_CHARGE: usize = 2;
const ENTRY_CHARGE: usize = 3; // (u16, u8)
const SELECTOR_CHARGE: usize = SIDE * 2 + 8; // 136 bytes
const PACKED_FIXED_CHARGE: usize = 16 + PAGE / 2; // 2064 bytes
const RESIDUAL_FIXED_CHARGE: usize = 8;
const MAX_RESIDUAL_EDITS: usize = (PAGE - RESIDUAL_FIXED_CHARGE - 1) / ENTRY_CHARGE; // 1362
const SPARSE_LIMIT: usize = (SELECTOR_CHARGE - SPARSE_FIXED_CHARGE - 1) / ENTRY_CHARGE; // 44

static MASK_TABLE: OnceLock<[u64; 256]> = OnceLock::new();
static NIBBLE_TABLE: OnceLock<[[u8; 2]; 256]> = OnceLock::new();
static REGISTRY: OnceLock<Mutex<VramRegistry>> = OnceLock::new();

fn mask_table() -> &'static [u64; 256] {
    MASK_TABLE.get_or_init(|| {
        let mut table = [0u64; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut bytes = [0u8; 8];
            for (j, byte) in bytes.iter_mut().enumerate() {
                *byte = if (i >> j) & 1 != 0 { 0xFF } else { 0x00 };
            }
            *slot = u64::from_le_bytes(bytes);
        }
        table
    })
}

fn nibble_table() -> &'static [[u8; 2]; 256] {
    NIBBLE_TABLE.get_or_init(|| {
        let mut lut = [[0u8; 2]; 256];
        for (word, pair) in lut.iter_mut().enumerate() {
            pair[0] = (word & 0x0f) as u8;
            pair[1] = ((word >> 4) & 0x0f) as u8;
        }
        lut
    })
}

fn registry() -> &'static Mutex<VramRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(VramRegistry::default()))
}

#[inline]
fn blend_word(left: u64, right: u64, mask: u64) -> u64 {
    (left & !mask) | (right & mask)
}

#[inline]
fn fnv1a_page_sample(data: &[u8; PAGE]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for slice in [&data[..8], &data[PAGE / 2..PAGE / 2 + 8], &data[PAGE - 8..]] {
        for &b in slice {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

#[derive(Clone, PartialEq, Eq)]
struct Repeated {
    block: Box<[u8]>,
}

impl Repeated {
    fn factor(page: &[u8; PAGE]) -> Option<&[u8]> {
        let mut block = &page[..];
        while block.len() > 1 {
            let (first, second) = block.split_at(block.len() / 2);
            if first != second {
                break;
            }
            block = first;
        }
        (block.len() < PAGE).then_some(block)
    }

    fn shifted(page: &[u8; PAGE], incumbent: usize) -> Option<&[u8]> {
        let max_disp = incumbent
            .saturating_sub(REPEATED_FIXED_CHARGE + 1)
            .min(PAGE / 2);
        if max_disp == 0 {
            return None;
        }
        let prefix = &page[..8];
        let end = (max_disp + 8).min(PAGE);
        let mut start = 1usize;
        while start + 8 <= end {
            if let Some(rel) = page[start..end].windows(8).position(|w| w == prefix) {
                let disp = start + rel;
                if disp <= max_disp && page[disp..] == page[..PAGE - disp] {
                    return Some(&page[..disp]);
                }
                start = disp + 1;
            } else {
                break;
            }
        }
        None
    }

    fn cost(block_len: usize) -> usize {
        REPEATED_FIXED_CHARGE + block_len
    }

    fn charge(&self) -> usize {
        Self::cost(self.block.len())
    }

    fn decode(&self, out: &mut [u8; PAGE]) {
        let mut n = self.block.len();
        out[..n].copy_from_slice(&self.block);
        while n < PAGE {
            let take = n.min(PAGE - n);
            out.copy_within(..take, n);
            n += take;
        }
    }
}

#[derive(Clone)]
struct Exceptions {
    base: u8,
    entries: Box<[(u16, u8)]>,
}

impl Exceptions {
    fn cost(entry_count: usize) -> usize {
        SPARSE_FIXED_CHARGE + entry_count * ENTRY_CHARGE
    }

    fn fit_bounded(bytes: &[u8; PAGE], max_entries: usize) -> Option<Self> {
        let cap = max_entries.min(SPARSE_LIMIT);
        let block_size = (cap + 1).next_power_of_two().min(PAGE);
        let blocks: Vec<&[u8]> = bytes.chunks_exact(block_size).collect();
        if blocks.len() <= cap {
            return None;
        }
        let witness = blocks[..=cap]
            .iter()
            .find(|b| !b.is_empty() && b[1..] == b[..b.len() - 1])?;
        let base = witness[0];
        let mut entries = Vec::new();
        for (i, &b) in bytes.iter().enumerate() {
            if b != base {
                if entries.len() >= cap {
                    return None;
                }
                entries.push((i as u16, b));
            }
        }
        Some(Self {
            base,
            entries: entries.into_boxed_slice(),
        })
    }

    fn decode(&self, out: &mut [u8; PAGE]) {
        out.fill(self.base);
        for &(idx, byte) in self.entries.iter() {
            out[usize::from(idx)] = byte;
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Selector {
    values: [[u8; 2]; SIDE],
    columns: u64,
}

impl Selector {
    fn fit_exact(bytes: &[u8; PAGE]) -> Option<Box<Self>> {
        let mut columns = 0u64;
        let mut pivot = 0usize;
        if let Some(row) = bytes
            .chunks_exact(SIDE)
            .find(|row| row.iter().any(|&b| b != row[0]))
        {
            pivot = row.iter().position(|&b| b != row[0])?;
            let second = row[pivot];
            for (j, &b) in row.iter().enumerate() {
                if b == second {
                    columns |= 1u64 << j;
                } else if b != row[0] {
                    return None;
                }
            }
        }
        let mut values = [[0u8; 2]; SIDE];
        for (v, row) in values.iter_mut().zip(bytes.chunks_exact(SIDE)) {
            *v = [row[0], row[pivot]];
        }
        let model = Self { values, columns };
        let mut check = [0u8; PAGE];
        model.decode(&mut check);
        (check == *bytes).then(|| Box::new(model))
    }

    fn fit_dominant(bytes: &[u8; PAGE]) -> Box<Self> {
        let mut col_diff = [0usize; SIDE];
        for row in bytes.chunks_exact(SIDE) {
            let anchor = row[0];
            for (j, &b) in row.iter().enumerate() {
                if b != anchor {
                    col_diff[j] += 1;
                }
            }
        }
        let mut columns = 0u64;
        for (j, &count) in col_diff.iter().enumerate() {
            if count * 2 >= SIDE {
                columns |= 1u64 << j;
            }
        }
        // Fallback if columns ended up all zero on a row-varying page: use first non-matching col.
        if columns == 0 {
            if let Some(pos) = col_diff.iter().position(|&c| c > 0) {
                columns = 1u64 << pos;
            }
        }
        let mut values = [[0u8; 2]; SIDE];
        for (r, row) in bytes.chunks_exact(SIDE).enumerate() {
            let mut left_counts = [0u16; 256];
            let mut right_counts = [0u16; 256];
            let mut best_left = row[0];
            let mut best_left_freq = 0u16;
            let mut best_right = row[0];
            let mut best_right_freq = 0u16;
            for (j, &b) in row.iter().enumerate() {
                if (columns >> j) & 1 == 0 {
                    let c = &mut left_counts[usize::from(b)];
                    *c += 1;
                    if *c > best_left_freq {
                        best_left_freq = *c;
                        best_left = b;
                    }
                } else {
                    let c = &mut right_counts[usize::from(b)];
                    *c += 1;
                    if *c > best_right_freq {
                        best_right_freq = *c;
                        best_right = b;
                    }
                }
            }
            values[r] = [best_left, best_right];
        }
        Box::new(Self { values, columns })
    }

    fn decode(&self, page: &mut [u8; PAGE]) {
        let lut = mask_table();
        let masks = self.columns.to_le_bytes().map(|b| lut[usize::from(b)]);
        for (row, [a, b]) in page.chunks_exact_mut(SIDE).zip(self.values) {
            let left = u64::from_le_bytes([a; 8]);
            let right = u64::from_le_bytes([b; 8]);
            for (word, mask) in row.chunks_exact_mut(8).zip(masks) {
                word.copy_from_slice(&blend_word(left, right, mask).to_le_bytes());
            }
        }
    }
}

#[derive(Clone)]
struct Corrected {
    base: Box<Selector>,
    edits: Box<[(u16, u8)]>,
}

impl Corrected {
    fn cost(edit_count: usize) -> usize {
        SELECTOR_CHARGE + edit_count * ENTRY_CHARGE
    }

    fn fit(base: &Selector, target: &[u8; PAGE], max_cost: usize) -> Option<Self> {
        let mut base_bytes = [0u8; PAGE];
        base.decode(&mut base_bytes);
        let max_edits = max_cost.saturating_sub(SELECTOR_CHARGE + 1) / ENTRY_CHARGE;
        let mut edits = Vec::new();
        for (i, (&actual, &pred)) in target.iter().zip(base_bytes.iter()).enumerate() {
            if actual != pred {
                if edits.len() >= max_edits {
                    return None;
                }
                edits.push((i as u16, actual));
            }
        }
        Some(Self {
            base: Box::new(base.clone()),
            edits: edits.into_boxed_slice(),
        })
    }

    fn decode(&self, out: &mut [u8; PAGE]) {
        self.base.decode(out);
        for &(i, b) in self.edits.iter() {
            out[usize::from(i)] = b;
        }
    }
}

#[derive(Clone)]
struct Packed {
    table: [u8; 16],
    codes: Box<[u8; PAGE / 2]>,
    exceptions: Box<[(u16, u8)]>,
}

impl Packed {
    fn cost(exception_count: usize) -> usize {
        PACKED_FIXED_CHARGE + exception_count * ENTRY_CHARGE
    }

    fn fit(bytes: &[u8; PAGE], histogram: &[usize; 256], max_cost: usize) -> Option<Self> {
        let mut symbols: [u8; 256] = std::array::from_fn(|i| i as u8);
        let rank = |&c: &u8| (std::cmp::Reverse(histogram[usize::from(c)]), c);
        symbols.select_nth_unstable_by_key(15, rank);
        let mut table: [u8; 16] = symbols[..16].try_into().ok()?;
        table.sort_unstable_by_key(rank);

        let matched: usize = table.iter().map(|&b| histogram[usize::from(b)]).sum();
        let exception_count = PAGE - matched;
        let total_cost = Self::cost(exception_count);
        if total_cost >= max_cost || total_cost >= PAGE {
            return None;
        }

        let mut rev = [0xffu8; 256];
        for (idx, &sym) in table.iter().enumerate() {
            rev[usize::from(sym)] = idx as u8;
        }

        let mut codes = Box::new([0u8; PAGE / 2]);
        let mut exceptions = Vec::with_capacity(exception_count);
        for (pair_idx, pair) in bytes.chunks_exact(2).enumerate() {
            let mut lo = rev[usize::from(pair[0])];
            if lo == 0xff {
                exceptions.push(((pair_idx * 2) as u16, pair[0]));
                lo = 0;
            }
            let mut hi = rev[usize::from(pair[1])];
            if hi == 0xff {
                exceptions.push(((pair_idx * 2 + 1) as u16, pair[1]));
                hi = 0;
            }
            codes[pair_idx] = lo | (hi << 4);
        }

        Some(Self {
            table,
            codes,
            exceptions: exceptions.into_boxed_slice(),
        })
    }

    fn decode(&self, out: &mut [u8; PAGE]) {
        let lut = nibble_table();
        for (&code, pair) in self.codes.iter().zip(out.chunks_exact_mut(2)) {
            let [a, b] = lut[usize::from(code)];
            pair[0] = self.table[usize::from(a)];
            pair[1] = self.table[usize::from(b)];
        }
        for &(i, b) in self.exceptions.iter() {
            out[usize::from(i)] = b;
        }
    }
}

struct Literal {
    bytes: Box<[u8; PAGE]>,
    live: Arc<AtomicUsize>,
}

impl Drop for Literal {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct Residual {
    base: Arc<Literal>,
    edits: Box<[(u16, u8)]>,
}

impl Residual {
    fn cost(edit_count: usize) -> usize {
        RESIDUAL_FIXED_CHARGE + edit_count * ENTRY_CHARGE
    }

    fn fit(base: &Arc<Literal>, data: &[u8; PAGE], max_edits: usize) -> Option<Self> {
        let limit = max_edits.min(MAX_RESIDUAL_EDITS);
        let mut edits = Vec::new();
        for (i, (&actual, &pred)) in data.iter().zip(base.bytes.iter()).enumerate() {
            if actual != pred {
                if edits.len() >= limit {
                    return None;
                }
                edits.push((i as u16, actual));
            }
        }
        Some(Self {
            base: Arc::clone(base),
            edits: edits.into_boxed_slice(),
        })
    }

    fn decode(&self, out: &mut [u8; PAGE]) {
        out.copy_from_slice(&*self.base.bytes);
        for &(i, b) in self.edits.iter() {
            out[usize::from(i)] = b;
        }
    }
}

type BucketEntry = (u64, Weak<Literal>);

struct LiteralPool {
    buckets: Vec<Vec<BucketEntry>>,
    live: Arc<AtomicUsize>,
    shared_hits: u64,
    residual_hits: u64,
}

impl LiteralPool {
    fn new() -> Self {
        Self {
            buckets: vec![Vec::new(); 16],
            live: Arc::new(AtomicUsize::new(0)),
            shared_hits: 0,
            residual_hits: 0,
        }
    }

    fn live_count(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    fn live_bytes(&self) -> usize {
        self.live_count() * PAGE
    }

    fn maybe_rehash(&mut self) {
        let live = self.live_count();
        let cap = self.buckets.len();
        let next_cap = if live * 2 > cap && cap < 4096 {
            cap * 2
        } else if cap > 16 && live * 8 < cap {
            cap / 2
        } else {
            return;
        };
        let mut next = vec![Vec::new(); next_cap];
        let mask = next_cap - 1;
        for bucket in &self.buckets {
            for (tag, weak) in bucket {
                if weak.strong_count() > 0 {
                    next[*tag as usize & mask].push((*tag, weak.clone()));
                }
            }
        }
        self.buckets = next;
    }

    fn find_candidate(
        &self,
        hash: u64,
        data: &[u8; PAGE],
        skip_residual_ptrs: &[usize],
    ) -> Option<Value> {
        if self.buckets.is_empty() {
            return None;
        }
        let mask = self.buckets.len() - 1;
        let bucket = &self.buckets[hash as usize & mask];
        let mut best_residual: Option<Residual> = None;

        for weak in bucket
            .iter()
            .filter(|(tag, _)| *tag == hash)
            .map(|(_, w)| w)
        {
            let Some(page) = weak.upgrade() else {
                continue;
            };
            if *page.bytes == *data {
                return Some(Value::Shared(page));
            }
            let page_ptr = Arc::as_ptr(&page) as usize;
            if skip_residual_ptrs.contains(&page_ptr) {
                continue;
            }
            let max_edits = best_residual
                .as_ref()
                .map_or(MAX_RESIDUAL_EDITS, |r| r.edits.len().saturating_sub(1));
            if let Some(res) = Residual::fit(&page, data, max_edits) {
                best_residual = Some(res);
            }
        }
        best_residual.map(|r| Value::Residual(Box::new(r)))
    }

    fn alloc_literal(&mut self, hash: u64, data: &[u8; PAGE]) -> Arc<Literal> {
        let lit = Arc::new(Literal {
            bytes: Box::new(*data),
            live: Arc::clone(&self.live),
        });
        self.live.fetch_add(1, Ordering::Relaxed);
        self.maybe_rehash();
        let mask = self.buckets.len() - 1;
        let bucket = &mut self.buckets[hash as usize & mask];
        bucket.retain(|(_, w)| w.strong_count() > 0);
        let entry = (hash, Arc::downgrade(&lit));
        if let Some(slot) = bucket.iter_mut().filter(|(tag, _)| *tag == hash).nth(3) {
            *slot = entry;
        } else {
            bucket.push(entry);
        }
        lit
    }

    fn prune_dead(&mut self) {
        for bucket in &mut self.buckets {
            bucket.retain(|(_, w)| w.strong_count() > 0);
        }
        self.maybe_rehash();
    }
}

#[derive(Clone)]
enum Value {
    Fill(u8),
    Repeated(Box<Repeated>),
    Sparse(Box<Exceptions>),
    Selector(Box<Selector>),
    Corrected(Box<Corrected>),
    Packed(Box<Packed>),
    Shared(Arc<Literal>),
    Residual(Box<Residual>),
}

impl Value {
    fn tag(&self) -> u8 {
        match self {
            Self::Fill(_) => TAG_FILL,
            Self::Repeated(_) => TAG_REPEATED,
            Self::Sparse(_) => TAG_SPARSE,
            Self::Selector(_) => TAG_SELECTOR,
            Self::Corrected(_) => TAG_CORRECTED,
            Self::Packed(_) => TAG_PACKED,
            Self::Shared(_) => TAG_SHARED,
            Self::Residual(_) => TAG_RESIDUAL,
        }
    }

    /// Fixed-width backing charge (independent of host pointer size so wasm32 & host match).
    fn non_pool_charge(&self) -> usize {
        match self {
            Self::Fill(_) => FILL_CHARGE,
            Self::Repeated(r) => r.charge(),
            Self::Sparse(s) => Exceptions::cost(s.entries.len()),
            Self::Selector(_) => SELECTOR_CHARGE,
            Self::Corrected(c) => Corrected::cost(c.edits.len()),
            Self::Packed(p) => Packed::cost(p.exceptions.len()),
            Self::Shared(_) => 0, // Charged once in LiteralPool::live_bytes()
            Self::Residual(r) => Residual::cost(r.edits.len()),
        }
    }

    fn selector_base(&self) -> Option<&Selector> {
        match self {
            Self::Selector(s) => Some(s),
            Self::Corrected(c) => Some(&c.base),
            _ => None,
        }
    }

    fn literal_base(&self) -> Option<&Arc<Literal>> {
        match self {
            Self::Shared(b) => Some(b),
            Self::Residual(r) => Some(&r.base),
            _ => None,
        }
    }

    fn decode(&self, out: &mut [u8; PAGE]) {
        match self {
            Self::Fill(b) => out.fill(*b),
            Self::Repeated(r) => r.decode(out),
            Self::Sparse(s) => s.decode(out),
            Self::Selector(m) => m.decode(out),
            Self::Corrected(c) => c.decode(out),
            Self::Packed(p) => p.decode(out),
            Self::Shared(l) => out.copy_from_slice(&*l.bytes),
            Self::Residual(r) => r.decode(out),
        }
    }
}

#[inline]
fn is_uniform(bytes: &[u8; PAGE]) -> bool {
    bytes[1..] == bytes[..PAGE - 1]
}

fn consider_candidate(best: &mut Option<(usize, Value)>, bytes: &[u8; PAGE], candidate: Value) {
    let cost = candidate.non_pool_charge();
    if cost >= PAGE {
        return;
    }
    let better = best.as_ref().is_none_or(|(b_cost, _)| cost < *b_cost);
    if better {
        let mut decoded = [0u8; PAGE];
        candidate.decode(&mut decoded);
        if decoded == *bytes {
            *best = Some((cost, candidate));
        }
    }
}

fn pick_best_structured(prior_selector: Option<&Selector>, bytes: &[u8; PAGE]) -> Option<Value> {
    if is_uniform(bytes) {
        return Some(Value::Fill(bytes[0]));
    }

    let mut best: Option<(usize, Value)> = None;

    if let Some(block) = Repeated::factor(bytes) {
        consider_candidate(
            &mut best,
            bytes,
            Value::Repeated(Box::new(Repeated {
                block: block.into(),
            })),
        );
    } else if let Some(block) = Repeated::shifted(bytes, PAGE) {
        consider_candidate(
            &mut best,
            bytes,
            Value::Repeated(Box::new(Repeated {
                block: block.into(),
            })),
        );
    }

    let ceiling = best.as_ref().map_or(PAGE, |(c, _)| *c);
    if let Some(sparse) = Exceptions::fit_bounded(bytes, ceiling / ENTRY_CHARGE) {
        consider_candidate(&mut best, bytes, Value::Sparse(Box::new(sparse)));
    }

    let ceiling = best.as_ref().map_or(PAGE, |(c, _)| *c);
    if SELECTOR_CHARGE < ceiling {
        if let Some(sel) = Selector::fit_exact(bytes) {
            consider_candidate(&mut best, bytes, Value::Selector(sel));
        }
    }

    let ceiling = best.as_ref().map_or(PAGE, |(c, _)| *c);
    if let Some(prior) = prior_selector {
        if let Some(corr) = Corrected::fit(prior, bytes, ceiling) {
            consider_candidate(&mut best, bytes, Value::Corrected(Box::new(corr)));
        }
    }
    let dom = Selector::fit_dominant(bytes);
    let ceiling = best.as_ref().map_or(PAGE, |(c, _)| *c);
    if let Some(corr) = Corrected::fit(&dom, bytes, ceiling) {
        consider_candidate(&mut best, bytes, Value::Corrected(Box::new(corr)));
    }

    let mut histogram = [0usize; 256];
    for &b in bytes {
        histogram[usize::from(b)] += 1;
    }
    let ceiling = best.as_ref().map_or(PAGE, |(c, _)| *c);
    if let Some(packed) = Packed::fit(bytes, &histogram, ceiling) {
        consider_candidate(&mut best, bytes, Value::Packed(Box::new(packed)));
    }

    best.map(|(_, val)| val)
}

#[derive(Clone)]
struct Group {
    values: [Value; GROUP],
    occupied: u64,
}

impl Group {
    fn new() -> Self {
        Self {
            values: std::array::from_fn(|_| Value::Fill(0)),
            occupied: 0,
        }
    }

    fn prior_selector(&self, slot: usize) -> Option<&Selector> {
        self.values[slot].selector_base().or_else(|| {
            self.values
                .iter()
                .enumerate()
                .find(|(idx, _)| (self.occupied >> idx) & 1 != 0)
                .and_then(|(_, v)| v.selector_base())
        })
    }
}

struct Leaf {
    groups: [Option<Box<Group>>; FANOUT],
    live: usize,
}

impl Leaf {
    fn new() -> Self {
        Self {
            groups: std::array::from_fn(|_| None),
            live: 0,
        }
    }
}

#[derive(Default)]
struct TagCounts {
    fill: u64,
    repeated: u64,
    sparse: u64,
    selector: u64,
    corrected: u64,
    packed: u64,
    shared: u64,
    residual: u64,
}

impl TagCounts {
    fn inc(&mut self, tag: u8) {
        match tag {
            TAG_FILL => self.fill += 1,
            TAG_REPEATED => self.repeated += 1,
            TAG_SPARSE => self.sparse += 1,
            TAG_SELECTOR => self.selector += 1,
            TAG_CORRECTED => self.corrected += 1,
            TAG_PACKED => self.packed += 1,
            TAG_SHARED => self.shared += 1,
            TAG_RESIDUAL => self.residual += 1,
            _ => {}
        }
    }

    fn dec(&mut self, tag: u8) {
        match tag {
            TAG_FILL => self.fill = self.fill.saturating_sub(1),
            TAG_REPEATED => self.repeated = self.repeated.saturating_sub(1),
            TAG_SPARSE => self.sparse = self.sparse.saturating_sub(1),
            TAG_SELECTOR => self.selector = self.selector.saturating_sub(1),
            TAG_CORRECTED => self.corrected = self.corrected.saturating_sub(1),
            TAG_PACKED => self.packed = self.packed.saturating_sub(1),
            TAG_SHARED => self.shared = self.shared.saturating_sub(1),
            TAG_RESIDUAL => self.residual = self.residual.saturating_sub(1),
            _ => {}
        }
    }
}

struct RollbackEntry {
    page_idx: usize,
    was_occupied: bool,
    old_value: Value,
}

pub struct VramStore {
    max_pages: usize,
    max_backing_bytes: usize,
    leaves: BTreeMap<usize, Box<Leaf>>,
    pool: LiteralPool,
    non_pool_bytes: usize,
    occupied_pages: usize,
    counts: TagCounts,
    reads: u64,
    writes: u64,
    discards: u64,
}

impl VramStore {
    pub fn new(max_pages: usize, max_backing_bytes: usize) -> Result<Self, i32> {
        let max_pages = if max_pages == 0 {
            DEFAULT_MAX_PAGES
        } else {
            max_pages
        };
        let max_backing_bytes = if max_backing_bytes == 0 {
            DEFAULT_MAX_BACKING_BYTES
        } else {
            max_backing_bytes
        };
        if max_pages > HARD_MAX_PAGES || max_backing_bytes > HARD_MAX_BACKING_BYTES {
            return Err(ERR_INVALID_ARG);
        }
        Ok(Self {
            max_pages,
            max_backing_bytes,
            leaves: BTreeMap::new(),
            pool: LiteralPool::new(),
            non_pool_bytes: 0,
            occupied_pages: 0,
            counts: TagCounts::default(),
            reads: 0,
            writes: 0,
            discards: 0,
        })
    }

    fn total_backing_bytes(&self) -> usize {
        self.non_pool_bytes.saturating_add(self.pool.live_bytes())
    }

    fn get_value(&self, page: usize) -> Option<&Value> {
        let leaf = self.leaves.get(&(page / LEAF))?;
        let group = leaf.groups[(page / GROUP) % FANOUT].as_deref()?;
        let slot = page % GROUP;
        ((group.occupied >> slot) & 1 != 0).then_some(&group.values[slot])
    }

    fn prior_selector(&self, page: usize) -> Option<&Selector> {
        let leaf = self.leaves.get(&(page / LEAF))?;
        let group = leaf.groups[(page / GROUP) % FANOUT].as_deref()?;
        group.prior_selector(page % GROUP)
    }

    pub fn read_page(&mut self, page: usize, out: &mut [u8; PAGE]) -> Result<(), i32> {
        if page >= self.max_pages {
            return Err(ERR_OUT_OF_RANGE);
        }
        self.reads += 1;
        match self.get_value(page) {
            Some(val) => val.decode(out),
            None => out.fill(0),
        }
        Ok(())
    }

    fn apply_page_value_keep_old(
        &mut self,
        page: usize,
        was_occupied: bool,
        new_value: Value,
    ) -> Value {
        let leaf = self
            .leaves
            .entry(page / LEAF)
            .or_insert_with(|| Box::new(Leaf::new()));
        let group = leaf.groups[(page / GROUP) % FANOUT].get_or_insert_with(|| {
            leaf.live += 1;
            Box::new(Group::new())
        });
        let slot = page % GROUP;
        let new_tag = new_value.tag();
        let new_non_pool = new_value.non_pool_charge();
        group.occupied |= 1u64 << slot;
        let old_value = std::mem::replace(&mut group.values[slot], new_value);
        let old_non_pool = if was_occupied {
            self.counts.dec(old_value.tag());
            old_value.non_pool_charge()
        } else {
            self.occupied_pages += 1;
            0
        };
        self.non_pool_bytes = self
            .non_pool_bytes
            .saturating_sub(old_non_pool)
            .saturating_add(new_non_pool);
        self.counts.inc(new_tag);
        old_value
    }

    fn rollback_entry(&mut self, entry: RollbackEntry) {
        let page = entry.page_idx;
        let cur_occupied = self.get_value(page).is_some();
        if entry.was_occupied {
            let replaced = self.apply_page_value_keep_old(page, cur_occupied, entry.old_value);
            drop(replaced);
            self.pool.prune_dead();
        } else if cur_occupied {
            self.clear_single_page(page);
        }
    }

    fn clear_single_page(&mut self, page: usize) {
        let Some(leaf) = self.leaves.get_mut(&(page / LEAF)) else {
            return;
        };
        let g_idx = (page / GROUP) % FANOUT;
        let Some(group) = leaf.groups[g_idx].as_mut() else {
            return;
        };
        let slot = page % GROUP;
        if (group.occupied >> slot) & 1 == 0 {
            return;
        }
        let prev = std::mem::replace(&mut group.values[slot], Value::Fill(0));
        group.occupied &= !(1u64 << slot);
        self.counts.dec(prev.tag());
        self.non_pool_bytes = self.non_pool_bytes.saturating_sub(prev.non_pool_charge());
        self.occupied_pages = self.occupied_pages.saturating_sub(1);
        drop(prev);
        if group.occupied == 0 {
            leaf.groups[g_idx] = None;
            leaf.live = leaf.live.saturating_sub(1);
            if leaf.live == 0 {
                self.leaves.remove(&(page / LEAF));
            }
        }
        self.pool.prune_dead();
    }

    pub fn write_page(&mut self, page: usize, data: &[u8; PAGE]) -> Result<u8, i32> {
        let (tag, _was_occ, old_value) = self.write_page_inner(page, data, &[])?;
        drop(old_value);
        self.pool.prune_dead();
        Ok(tag)
    }

    fn write_page_inner(
        &mut self,
        page: usize,
        data: &[u8; PAGE],
        rollbacks: &[RollbackEntry],
    ) -> Result<(u8, bool, Value), i32> {
        if page >= self.max_pages {
            return Err(ERR_OUT_OF_RANGE);
        }

        let mut rb_map: BTreeMap<usize, (&Arc<Literal>, usize)> = BTreeMap::new();
        for entry in rollbacks {
            if let Some(lit) = entry.old_value.literal_base() {
                let ptr = Arc::as_ptr(lit) as usize;
                rb_map
                    .entry(ptr)
                    .and_modify(|(_, c)| *c += 1)
                    .or_insert((lit, 1));
            }
        }
        let mut rollback_dead_bytes = 0usize;
        let mut skip_residual_ptrs: Vec<usize> = Vec::new();
        for (&ptr, (arc, rb_c)) in &rb_map {
            if Arc::strong_count(arc) == *rb_c {
                rollback_dead_bytes = rollback_dead_bytes.saturating_add(PAGE);
                skip_residual_ptrs.push(ptr);
            }
        }
        let rb_count_of = |ptr: usize| rb_map.get(&ptr).map_or(0usize, |(_, c)| *c);

        let (was_occupied, old_non_pool, incumbent_sole_ptr, prior) = {
            let cur = self.get_value(page);
            let was_occ = cur.is_some();
            let non_pool = cur.map_or(0, Value::non_pool_charge);
            let sole_ptr = cur.and_then(Value::literal_base).and_then(|lit| {
                let ptr = Arc::as_ptr(lit) as usize;
                (Arc::strong_count(lit) == 1 + rb_count_of(ptr)).then_some(ptr)
            });
            let prior = self.prior_selector(page).cloned();
            (was_occ, non_pool, sole_ptr, prior)
        };
        if let Some(sole_ptr) = incumbent_sole_ptr {
            if !skip_residual_ptrs.contains(&sole_ptr) {
                skip_residual_ptrs.push(sole_ptr);
            }
        }

        let structured = pick_best_structured(prior.as_ref(), data);
        let hash = fnv1a_page_sample(data);
        let pool_cand = self.pool.find_candidate(hash, data, &skip_residual_ptrs);
        let effective_pool_bytes = self.pool.live_bytes().saturating_sub(rollback_dead_bytes);
        let freed_on_replace = if incumbent_sole_ptr.is_some() { PAGE } else { 0 };

        let candidate_pool_cost = |p: &Value| -> usize {
            match p {
                Value::Shared(arc) => {
                    let ptr = Arc::as_ptr(arc) as usize;
                    let same_exclusive = incumbent_sole_ptr == Some(ptr);
                    let store_owners = Arc::strong_count(arc).saturating_sub(1 + rb_count_of(ptr));
                    if same_exclusive || store_owners == 0 {
                        PAGE
                    } else {
                        0
                    }
                }
                Value::Residual(r) => {
                    let ptr = Arc::as_ptr(&r.base) as usize;
                    let store_owners = Arc::strong_count(&r.base).saturating_sub(1 + rb_count_of(ptr));
                    let extra = if store_owners == 0 { PAGE } else { 0 };
                    Residual::cost(r.edits.len()).saturating_add(extra)
                }
                _ => p.non_pool_charge(),
            }
        };

        // Select between structured geometric encoding and literal pool candidate.
        let mut hit_kind: u8 = 0; // 1 = shared, 2 = residual
        let chosen: Option<Value> = match (structured, pool_cand) {
            (Some(s), Some(p)) => {
                if candidate_pool_cost(&p) <= s.non_pool_charge() {
                    hit_kind = match &p {
                        Value::Shared(_) => 1,
                        Value::Residual(_) => 2,
                        _ => 0,
                    };
                    Some(p)
                } else {
                    Some(s)
                }
            }
            (Some(s), None) => Some(s),
            (None, Some(p)) => {
                if matches!(&p, Value::Residual(_)) && candidate_pool_cost(&p) >= PAGE {
                    None
                } else {
                    hit_kind = match &p {
                        Value::Shared(_) => 1,
                        Value::Residual(_) => 2,
                        _ => 0,
                    };
                    Some(p)
                }
            }
            (None, None) => None,
        };

        let chosen = match chosen {
            Some(c) => {
                let (extra_pool, freed_pool) = match c.literal_base() {
                    Some(lit) => {
                        let ptr = Arc::as_ptr(lit) as usize;
                        let store_owners = Arc::strong_count(lit).saturating_sub(1 + rb_count_of(ptr));
                        let extra = if store_owners == 0 { PAGE } else { 0 };
                        let freed = if incumbent_sole_ptr.is_some() && incumbent_sole_ptr != Some(ptr) {
                            PAGE
                        } else {
                            0
                        };
                        (extra, freed)
                    }
                    None => (0, freed_on_replace),
                };
                let projected = self
                    .non_pool_bytes
                    .saturating_sub(old_non_pool)
                    .saturating_add(c.non_pool_charge())
                    .saturating_add(effective_pool_bytes.saturating_sub(freed_pool))
                    .saturating_add(extra_pool);
                if projected > self.max_backing_bytes {
                    return Err(ERR_CAPACITY);
                }
                let mut verify = [0u8; PAGE];
                c.decode(&mut verify);
                if verify != *data {
                    return Err(ERR_INTERNAL);
                }
                c
            }
            None => {
                let projected = self
                    .non_pool_bytes
                    .saturating_sub(old_non_pool)
                    .saturating_add(effective_pool_bytes.saturating_sub(freed_on_replace))
                    .saturating_add(PAGE);
                if projected > self.max_backing_bytes {
                    return Err(ERR_CAPACITY);
                }
                let lit = self.pool.alloc_literal(hash, data);
                Value::Shared(lit)
            }
        };

        if hit_kind == 1 {
            self.pool.shared_hits += 1;
        } else if hit_kind == 2 {
            self.pool.residual_hits += 1;
        }
        self.writes += 1;
        let tag = chosen.tag();
        let old_value = self.apply_page_value_keep_old(page, was_occupied, chosen);
        Ok((tag, was_occupied, old_value))
    }

    pub fn write_range(&mut self, byte_offset: u64, data: &[u8]) -> Result<(), i32> {
        if data.len() > MAX_TRANSFER_BYTES {
            return Err(ERR_INVALID_ARG);
        }
        let max_bytes = (self.max_pages as u64).saturating_mul(PAGE as u64);
        let end = byte_offset
            .checked_add(data.len() as u64)
            .ok_or(ERR_OUT_OF_RANGE)?;
        if end > max_bytes {
            return Err(ERR_OUT_OF_RANGE);
        }
        if data.is_empty() {
            return Ok(());
        }

        let saved_reads = self.reads;
        let saved_writes = self.writes;
        let saved_discards = self.discards;
        let saved_shared_hits = self.pool.shared_hits;
        let saved_residual_hits = self.pool.residual_hits;

        let first_page = (byte_offset / PAGE as u64) as usize;
        let last_page = ((end - 1) / PAGE as u64) as usize;
        let mut rollbacks: Vec<RollbackEntry> = Vec::with_capacity(last_page - first_page + 1);

        let mut cursor = 0usize;
        for page_idx in first_page..=last_page {
            let page_start = (page_idx as u64) * (PAGE as u64);
            let within_start = if page_idx == first_page {
                (byte_offset - page_start) as usize
            } else {
                0
            };
            let copy_len = (PAGE - within_start).min(data.len() - cursor);

            let mut buf = [0u8; PAGE];
            if copy_len < PAGE {
                if let Some(cur) = self.get_value(page_idx) {
                    cur.decode(&mut buf);
                }
            }
            buf[within_start..within_start + copy_len]
                .copy_from_slice(&data[cursor..cursor + copy_len]);
            match self.write_page_inner(page_idx, &buf, &rollbacks) {
                Ok((_tag, was_occupied, old_value)) => {
                    rollbacks.push(RollbackEntry {
                        page_idx,
                        was_occupied,
                        old_value,
                    });
                }
                Err(err) => {
                    for entry in rollbacks.into_iter().rev() {
                        self.rollback_entry(entry);
                    }
                    self.reads = saved_reads;
                    self.writes = saved_writes;
                    self.discards = saved_discards;
                    self.pool.shared_hits = saved_shared_hits;
                    self.pool.residual_hits = saved_residual_hits;
                    self.pool.prune_dead();
                    return Err(err);
                }
            }
            cursor += copy_len;
        }
        drop(rollbacks);
        self.pool.prune_dead();
        Ok(())
    }

    pub fn read_range(&mut self, byte_offset: u64, out: &mut [u8]) -> Result<(), i32> {
        if out.len() > MAX_TRANSFER_BYTES {
            return Err(ERR_INVALID_ARG);
        }
        let max_bytes = (self.max_pages as u64).saturating_mul(PAGE as u64);
        let end = byte_offset
            .checked_add(out.len() as u64)
            .ok_or(ERR_OUT_OF_RANGE)?;
        if end > max_bytes {
            return Err(ERR_OUT_OF_RANGE);
        }
        if out.is_empty() {
            return Ok(());
        }

        let first_page = (byte_offset / PAGE as u64) as usize;
        let last_page = ((end - 1) / PAGE as u64) as usize;
        let mut cursor = 0usize;
        let mut page_buf = [0u8; PAGE];
        for page_idx in first_page..=last_page {
            let page_start = (page_idx as u64) * (PAGE as u64);
            let within_start = if page_idx == first_page {
                (byte_offset - page_start) as usize
            } else {
                0
            };
            let copy_len = (PAGE - within_start).min(out.len() - cursor);
            self.read_page(page_idx, &mut page_buf)?;
            out[cursor..cursor + copy_len]
                .copy_from_slice(&page_buf[within_start..within_start + copy_len]);
            cursor += copy_len;
        }
        Ok(())
    }

    pub fn discard_pages(&mut self, start: usize, count: usize) -> Result<usize, i32> {
        let end = start.checked_add(count).ok_or(ERR_OUT_OF_RANGE)?;
        if end > self.max_pages {
            return Err(ERR_OUT_OF_RANGE);
        }
        if count == 0 {
            return Ok(0);
        }
        let mut cleared = 0usize;
        let leaf_start = start / LEAF;
        let leaf_end = (end - 1) / LEAF;
        let leaf_ids: Vec<usize> = self
            .leaves
            .range(leaf_start..=leaf_end)
            .map(|(&id, _)| id)
            .collect();
        let mut empty_leaves: Vec<usize> = Vec::new();
        for leaf_id in leaf_ids {
            let Some(leaf) = self.leaves.get_mut(&leaf_id) else {
                continue;
            };
            let base = leaf_id * LEAF;
            let lo = start.saturating_sub(base);
            let hi = (end - base).min(LEAF);
            for g_idx in lo / GROUP..hi.div_ceil(GROUP) {
                let Some(group) = leaf.groups[g_idx].as_mut() else {
                    continue;
                };
                let g_lo = lo.saturating_sub(g_idx * GROUP);
                let g_hi = (hi - g_idx * GROUP).min(GROUP);
                for slot in g_lo..g_hi {
                    if (group.occupied >> slot) & 1 != 0 {
                        let prev = std::mem::replace(&mut group.values[slot], Value::Fill(0));
                        group.occupied &= !(1u64 << slot);
                        self.counts.dec(prev.tag());
                        self.non_pool_bytes =
                            self.non_pool_bytes.saturating_sub(prev.non_pool_charge());
                        self.occupied_pages = self.occupied_pages.saturating_sub(1);
                        cleared += 1;
                    }
                }
                if group.occupied == 0 {
                    leaf.groups[g_idx] = None;
                    leaf.live = leaf.live.saturating_sub(1);
                }
            }
            if leaf.live == 0 {
                empty_leaves.push(leaf_id);
            }
        }
        for id in empty_leaves {
            self.leaves.remove(&id);
        }
        self.discards += cleared as u64;
        self.pool.prune_dead();
        Ok(cleared)
    }

    pub fn stats_json(&self) -> JsonValue {
        let logical_bytes = (self.occupied_pages as u64).saturating_mul(PAGE as u64);
        let backing_bytes = self.total_backing_bytes() as u64;
        let savings_permille: u64 = if logical_bytes == 0 {
            0
        } else if backing_bytes >= logical_bytes {
            0
        } else {
            (((logical_bytes - backing_bytes) as u128 * 1000) / (logical_bytes as u128)) as u64
        };
        json!({
            "page_bytes": PAGE,
            "max_pages": self.max_pages,
            "max_backing_bytes": self.max_backing_bytes,
            "occupied_pages": self.occupied_pages,
            "logical_bytes": logical_bytes,
            "backing_bytes": backing_bytes,
            "non_pool_bytes": self.non_pool_bytes,
            "pool_bytes": self.pool.live_bytes(),
            "unique_literals": self.pool.live_count(),
            "shared_hits": self.pool.shared_hits,
            "residual_hits": self.pool.residual_hits,
            "leaves": self.leaves.len(),
            "savings_permille": savings_permille,
            "reads": self.reads,
            "writes": self.writes,
            "discards": self.discards,
            "representations": {
                "fill": self.counts.fill,
                "repeated": self.counts.repeated,
                "sparse": self.counts.sparse,
                "selector": self.counts.selector,
                "corrected": self.counts.corrected,
                "packed": self.counts.packed,
                "shared": self.counts.shared,
                "residual": self.counts.residual,
            }
        })
    }
}

#[derive(Default)]
struct RegistrySlot {
    gen: u32,
    store: Option<VramStore>,
}

#[derive(Default)]
struct VramRegistry {
    slots: Vec<RegistrySlot>,
}

impl VramRegistry {
    fn encode_handle(idx: usize, gen: u32) -> i32 {
        (((gen & 0x00ff_ffff) as i32) << 5) | ((idx + 1) as i32)
    }

    fn decode_handle(handle: i32) -> Option<(usize, u32)> {
        if handle <= 0 {
            return None;
        }
        let raw = handle as u32;
        if (raw >> 29) != 0 {
            return None;
        }
        let low = (raw & 0x1f) as usize;
        if low == 0 || low > MAX_STORES {
            return None;
        }
        let gen = (raw >> 5) & 0x00ff_ffff;
        Some((low - 1, gen))
    }

    fn create(&mut self, max_pages: usize, max_backing_bytes: usize) -> Result<i32, i32> {
        let store = VramStore::new(max_pages, max_backing_bytes)?;
        if let Some((idx, slot)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, s)| s.store.is_none())
        {
            slot.store = Some(store);
            return Ok(Self::encode_handle(idx, slot.gen));
        }
        if self.slots.len() >= MAX_STORES {
            return Err(ERR_CAPACITY);
        }
        let idx = self.slots.len();
        self.slots.push(RegistrySlot {
            gen: 0,
            store: Some(store),
        });
        Ok(Self::encode_handle(idx, 0))
    }

    fn destroy(&mut self, handle: i32) -> Result<(), i32> {
        let (idx, gen) = Self::decode_handle(handle).ok_or(ERR_INVALID_HANDLE)?;
        let slot = self.slots.get_mut(idx).ok_or(ERR_INVALID_HANDLE)?;
        if slot.gen != gen || slot.store.is_none() {
            return Err(ERR_INVALID_HANDLE);
        }
        slot.store = None;
        slot.gen = slot.gen.wrapping_add(1) & 0x00ff_ffff;
        Ok(())
    }

    fn get_mut(&mut self, handle: i32) -> Result<&mut VramStore, i32> {
        let (idx, gen) = Self::decode_handle(handle).ok_or(ERR_INVALID_HANDLE)?;
        let slot = self.slots.get_mut(idx).ok_or(ERR_INVALID_HANDLE)?;
        if slot.gen != gen {
            return Err(ERR_INVALID_HANDLE);
        }
        slot.store.as_mut().ok_or(ERR_INVALID_HANDLE)
    }

    fn get(&self, handle: i32) -> Result<&VramStore, i32> {
        let (idx, gen) = Self::decode_handle(handle).ok_or(ERR_INVALID_HANDLE)?;
        let slot = self.slots.get(idx).ok_or(ERR_INVALID_HANDLE)?;
        if slot.gen != gen {
            return Err(ERR_INVALID_HANDLE);
        }
        slot.store.as_ref().ok_or(ERR_INVALID_HANDLE)
    }
}

pub fn ffi_create(max_pages: i32, max_backing_bytes: i32) -> i32 {
    if max_pages < 0 || max_backing_bytes < 0 {
        return ERR_INVALID_ARG;
    }
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    reg.create(max_pages as usize, max_backing_bytes as usize)
        .unwrap_or_else(|e| e)
}

pub fn ffi_destroy(handle: i32) -> i32 {
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    match reg.destroy(handle) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

pub fn ffi_write_page(handle: i32, page_idx: i32, bytes: &[u8]) -> i32 {
    if page_idx < 0 || bytes.len() != PAGE {
        return ERR_INVALID_ARG;
    }
    let Ok(fixed) = <&[u8; PAGE]>::try_from(bytes) else {
        return ERR_INVALID_ARG;
    };
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    match reg.get_mut(handle) {
        Ok(store) => match store.write_page(page_idx as usize, fixed) {
            Ok(tag) => i32::from(tag),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

pub fn ffi_read_page(handle: i32, page_idx: i32, out: &mut [u8]) -> i32 {
    if page_idx < 0 || out.len() != PAGE {
        return ERR_INVALID_ARG;
    }
    let Ok(fixed) = <&mut [u8; PAGE]>::try_from(out) else {
        return ERR_INVALID_ARG;
    };
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    match reg.get_mut(handle) {
        Ok(store) => match store.read_page(page_idx as usize, fixed) {
            Ok(()) => 0,
            Err(e) => e,
        },
        Err(e) => e,
    }
}

pub fn ffi_write_range(handle: i32, byte_offset: u64, data: &[u8]) -> i32 {
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    match reg.get_mut(handle) {
        Ok(store) => match store.write_range(byte_offset, data) {
            Ok(()) => 0,
            Err(e) => e,
        },
        Err(e) => e,
    }
}

pub fn ffi_read_range(handle: i32, byte_offset: u64, out: &mut [u8]) -> i32 {
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    match reg.get_mut(handle) {
        Ok(store) => match store.read_range(byte_offset, out) {
            Ok(()) => 0,
            Err(e) => e,
        },
        Err(e) => e,
    }
}

pub fn ffi_discard_pages(handle: i32, start_page: i32, page_count: i32) -> i32 {
    if start_page < 0 || page_count < 0 {
        return ERR_INVALID_ARG;
    }
    let Ok(mut reg) = registry().lock() else {
        return ERR_INTERNAL;
    };
    match reg.get_mut(handle) {
        Ok(store) => match store.discard_pages(start_page as usize, page_count as usize) {
            Ok(cleared) => i32::try_from(cleared).unwrap_or(i32::MAX),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

pub fn ffi_stats_json_bytes(handle: i32) -> Option<Vec<u8>> {
    let Ok(reg) = registry().lock() else {
        return None;
    };
    let store = reg.get(handle).ok()?;
    Some(store.stats_json().to_string().into_bytes())
}

fn hex_decode(hex: &str) -> Result<Vec<u8>, String> {
    let clean: Vec<u8> = hex.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !clean.len().is_multiple_of(2) {
        return Err("hex string must have even length".to_owned());
    }
    if clean.len() / 2 > MAX_TRANSFER_BYTES {
        return Err("hex payload exceeds MAX_TRANSFER_BYTES".to_owned());
    }
    let mut out = Vec::with_capacity(clean.len() / 2);
    let val = |b: u8| -> Result<u8, String> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err("invalid hex character".to_owned()),
        }
    };
    for pair in clean.chunks_exact(2) {
        out.push((val(pair[0])? << 4) | val(pair[1])?);
    }
    Ok(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}

pub fn run_self_test() -> JsonValue {
    let mut store = match VramStore::new(64, 512 * 1024) {
        Ok(s) => s,
        Err(e) => return json!({ "ok": false, "error": format!("store init failed ({e})") }),
    };

    // Exercise all representation families in a single isolated store.
    let p0 = [0x42u8; PAGE]; // Fill
    let p1: [u8; PAGE] = std::array::from_fn(|i| ((i % 16) * 17) as u8); // Repeated
    let mut p2 = [0x10u8; PAGE]; // Sparse
    for k in 0..8 {
        p2[k * 311] = (k + 1) as u8;
    }
    let p3: [u8; PAGE] = std::array::from_fn(|i| {
        let r = (i / SIDE) as u8;
        let c = i % SIDE;
        if c % 5 == 0 {
            r.wrapping_add(1)
        } else {
            200u8.wrapping_sub(r)
        }
    }); // Selector
    let mut p4 = p3;
    p4[0] ^= 0x55;
    p4[4095] ^= 0x55; // Corrected
    let mut p5: [u8; PAGE] = [0u8; PAGE]; // Packed (16-symbol non-periodic)
    let mut s5: u32 = 0x1357_9bdf;
    for slot in &mut p5 {
        s5 ^= s5 << 13;
        s5 ^= s5 >> 17;
        s5 ^= s5 << 5;
        *slot = ((s5 >> 24) & 0x0f) as u8;
    }
    let mut p6: [u8; PAGE] = [0u8; PAGE]; // Unique literal
    let mut state: u32 = 0x9e37_79b9;
    for slot in &mut p6 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *slot = (state >> 24) as u8;
    }
    let p7 = p6; // Shared
    let mut p8 = p6; // Residual (preserves fingerprint sample regions at 0, 2048, 4088)
    p8[100] ^= 0xff;
    p8[1500] ^= 0x33;

    let pages = [p0, p1, p2, p3, p4, p5, p6, p7, p8];
    let mut tags = Vec::with_capacity(pages.len());
    for (idx, page) in pages.iter().enumerate() {
        match store.write_page(idx, page) {
            Ok(t) => tags.push(t),
            Err(e) => return json!({ "ok": false, "error": format!("write page {idx}: {e}") }),
        }
    }
    let mut verify = [0u8; PAGE];
    for (idx, expected) in pages.iter().enumerate() {
        if let Err(e) = store.read_page(idx, &mut verify) {
            return json!({ "ok": false, "error": format!("read page {idx}: {e}") });
        }
        if verify != *expected {
            return json!({ "ok": false, "error": format!("roundtrip mismatch at page {idx}") });
        }
    }
    let stats = store.stats_json();
    json!({
        "ok": true,
        "tags": tags,
        "stats": stats,
        "expected_tags": [TAG_FILL, TAG_REPEATED, TAG_SPARSE, TAG_SELECTOR, TAG_CORRECTED, TAG_PACKED, TAG_SHARED, TAG_SHARED, TAG_RESIDUAL],
    })
}

fn json_i32(value: &JsonValue, key: &str) -> Result<i32, JsonValue> {
    let raw = value[key]
        .as_i64()
        .ok_or_else(|| json!({ "ok": false, "error": format!("{key} must be an integer") }))?;
    i32::try_from(raw)
        .map_err(|_| json!({ "ok": false, "error": format!("{key} out of i32 range") }))
}

fn json_optional_nonneg_i32(value: &JsonValue, key: &str) -> Result<i32, JsonValue> {
    let field = &value[key];
    if field.is_null() {
        return Ok(0);
    }
    let raw = field
        .as_i64()
        .ok_or_else(|| json!({ "ok": false, "error": format!("{key} must be a non-negative integer") }))?;
    if raw < 0 {
        return Err(json!({ "ok": false, "error": format!("{key} must be non-negative") }));
    }
    i32::try_from(raw)
        .map_err(|_| json!({ "ok": false, "error": format!("{key} out of i32 range") }))
}

pub fn run_from_json(value: &JsonValue) -> JsonValue {
    let action = value["action"].as_str().unwrap_or("");
    match action {
        "self_test" => run_self_test(),
        "create" => {
            let mp = match json_optional_nonneg_i32(value, "max_pages") {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let mb = match json_optional_nonneg_i32(value, "max_backing_bytes") {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let handle = ffi_create(mp, mb);
            if handle <= 0 {
                json!({ "ok": false, "code": handle, "error": format!("vram_create failed ({handle})") })
            } else {
                json!({ "ok": true, "handle": handle })
            }
        }
        "destroy" => {
            let handle = match json_i32(value, "handle") {
                Ok(h) => h,
                Err(resp) => return resp,
            };
            let rc = ffi_destroy(handle);
            json!({ "ok": rc == 0, "code": rc })
        }
        "stats" => {
            let handle = match json_i32(value, "handle") {
                Ok(h) => h,
                Err(resp) => return resp,
            };
            let Ok(reg) = registry().lock() else {
                return json!({ "ok": false, "error": "registry lock failed" });
            };
            match reg.get(handle) {
                Ok(store) => json!({ "ok": true, "stats": store.stats_json() }),
                Err(code) => json!({ "ok": false, "code": code, "error": "invalid vram handle" }),
            }
        }
        "write_hex" => {
            let handle = match json_i32(value, "handle") {
                Ok(h) => h,
                Err(resp) => return resp,
            };
            let Some(offset) = value["offset"].as_u64() else {
                return json!({ "ok": false, "error": "offset must be a non-negative integer" });
            };
            let Some(hex) = value["hex"].as_str() else {
                return json!({ "ok": false, "error": "hex must be a string" });
            };
            let bytes = match hex_decode(hex) {
                Ok(b) => b,
                Err(err) => return json!({ "ok": false, "error": err }),
            };
            let rc = ffi_write_range(handle, offset, &bytes);
            if rc == 0 {
                json!({ "ok": true, "bytes": bytes.len() })
            } else {
                json!({ "ok": false, "code": rc, "error": format!("vram_write_range failed ({rc})") })
            }
        }
        "read_hex" => {
            let handle = match json_i32(value, "handle") {
                Ok(h) => h,
                Err(resp) => return resp,
            };
            let Some(offset) = value["offset"].as_u64() else {
                return json!({ "ok": false, "error": "offset must be a non-negative integer" });
            };
            let Some(len_u64) = value["len"].as_u64() else {
                return json!({ "ok": false, "error": "len must be a non-negative integer" });
            };
            if len_u64 > MAX_TRANSFER_BYTES as u64 {
                return json!({ "ok": false, "error": "len exceeds MAX_TRANSFER_BYTES" });
            }
            let len = len_u64 as usize;
            let mut out = vec![0u8; len];
            let rc = ffi_read_range(handle, offset, &mut out);
            if rc == 0 {
                json!({ "ok": true, "hex": hex_encode(&out) })
            } else {
                json!({ "ok": false, "code": rc, "error": format!("vram_read_range failed ({rc})") })
            }
        }
        "discard" => {
            let handle = match json_i32(value, "handle") {
                Ok(h) => h,
                Err(resp) => return resp,
            };
            let start = match json_i32(value, "start_page") {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let count = match json_i32(value, "page_count") {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let rc = ffi_discard_pages(handle, start, count);
            if rc >= 0 {
                json!({ "ok": true, "cleared_pages": rc })
            } else {
                json!({ "ok": false, "code": rc, "error": format!("vram_discard_pages failed ({rc})") })
            }
        }
        _ => json!({ "ok": false, "error": format!("unknown vram action {action:?}") }),
    }
}
