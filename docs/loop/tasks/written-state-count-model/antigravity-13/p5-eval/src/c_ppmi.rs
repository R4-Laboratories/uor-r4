#![forbid(unsafe_code)]

use crate::{fail, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use uor_r4_count_runtime::p3_state::P3_VOCABULARY;
use uor_r4_tokenizer::ByteBpeTokenizer;

pub const C_CORPUS_SHA256: &str =
    "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";

const PASS1_HASH_SLOTS: usize = 1 << 22; // 4,194,304 open-addressed slots (~50 MB)
const PASS1_LOAD_LIMIT: usize = (PASS1_HASH_SLOTS * 3) / 4; // 3,145,728 max tracked keys
const PASS2_HASH_SLOTS: usize = 1 << 17; // 131,072 slots for <= 32,800 active words (~1.3 MB)
const RING_WINDOW: usize = 6;
const CTX_SLOTS: usize = 1024;
const TOP_PPMI_BITS: usize = 128;

#[inline]
pub fn pack_word_atom(toks: &[u16]) -> Option<u64> {
    let len = toks.len();
    if !(1..=5).contains(&len) {
        return None;
    }
    let mut k = (len as u64) << 60;
    for (i, &t) in toks.iter().enumerate() {
        if usize::from(t) >= P3_VOCABULARY {
            return None;
        }
        k |= u64::from(t) << (48 - i * 12);
    }
    Some(k)
}

#[inline]
pub fn unpack_word_atom(key: u64) -> ([u16; 5], usize) {
    let len = ((key >> 60) & 0x0f) as usize;
    let mut out = [0u16; 5];
    for (i, slot) in out.iter_mut().enumerate().take(len.min(5)) {
        *slot = ((key >> (48 - i * 12)) & 0x0fff) as u16;
    }
    (out, len.min(5))
}

#[inline]
fn hash_atom_key(mut x: u64) -> usize {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x as usize
}

#[inline]
fn popcount_1024(a: &[u64; 16], b: &[u64; 16]) -> u64 {
    let mut s = 0u64;
    for k in 0..16 {
        s += u64::from((a[k] & b[k]).count_ones());
    }
    s
}

pub struct VocabWordClass {
    pub is_word_start: [bool; P3_VOCABULARY],
    pub is_word_cont: [bool; P3_VOCABULARY],
    pub trimmed_len: [u8; P3_VOCABULARY],
}

impl VocabWordClass {
    pub fn from_tokenizer(tok: &ByteBpeTokenizer) -> Self {
        let mut is_word_start = [false; P3_VOCABULARY];
        let mut is_word_cont = [false; P3_VOCABULARY];
        let mut trimmed_len = [0u8; P3_VOCABULARY];
        for t in 0..P3_VOCABULARY {
            let s = tok.decode(&[t as u32]);
            if let Some(rest) = s.strip_prefix(' ') {
                if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_lowercase()) {
                    is_word_start[t] = true;
                    trimmed_len[t] = rest.len().min(255) as u8;
                }
            } else if !s.is_empty() && s.bytes().all(|b| b.is_ascii_lowercase()) {
                is_word_cont[t] = true;
                trimmed_len[t] = s.len().min(255) as u8;
            }
        }
        Self {
            is_word_start,
            is_word_cont,
            trimmed_len,
        }
    }

    pub fn atom_from_tokens(&self, toks: &[u16]) -> Option<u64> {
        if toks.is_empty() || toks.len() > 5 {
            return None;
        }
        let t0 = usize::from(toks[0]);
        if t0 >= P3_VOCABULARY || !self.is_word_start[t0] {
            return None;
        }
        let mut cbytes = usize::from(self.trimmed_len[t0]);
        for &tk in &toks[1..] {
            let tu = usize::from(tk);
            if tu >= P3_VOCABULARY || !self.is_word_cont[tu] {
                return None;
            }
            cbytes += usize::from(self.trimmed_len[tu]);
        }
        if cbytes < 3 {
            return None;
        }
        pack_word_atom(toks)
    }

    pub fn decode_atom(&self, key: u64, tok: &ByteBpeTokenizer) -> String {
        let (arr, len) = unpack_word_atom(key);
        let ids: Vec<u32> = arr[..len].iter().map(|&x| u32::from(x)).collect();
        tok.decode(&ids)
    }
}

struct WordStreamParser {
    buf: [u16; 5],
    len: usize,
    char_bytes: usize,
    overflow: bool,
}

impl WordStreamParser {
    fn new() -> Self {
        Self {
            buf: [0u16; 5],
            len: 0,
            char_bytes: 0,
            overflow: false,
        }
    }

    #[inline]
    fn take_valid(&mut self) -> Option<u64> {
        let out = if self.len > 0 && !self.overflow && self.char_bytes >= 3 {
            pack_word_atom(&self.buf[..self.len])
        } else {
            None
        };
        self.len = 0;
        self.char_bytes = 0;
        self.overflow = false;
        out
    }

    /// Push one token from the stream; returns `(completed_word_atom_if_any, record_break)`.
    #[inline]
    fn step(&mut self, tok: u16, cls: &VocabWordClass) -> (Option<u64>, bool) {
        if tok == 0 || tok == 1 {
            let done = self.take_valid();
            return (done, true);
        }
        let tu = usize::from(tok);
        if tu >= P3_VOCABULARY {
            let done = self.take_valid();
            return (done, false);
        }
        if cls.is_word_start[tu] {
            let done = self.take_valid();
            self.buf[0] = tok;
            self.len = 1;
            self.char_bytes = usize::from(cls.trimmed_len[tu]);
            return (done, false);
        }
        if cls.is_word_cont[tu] {
            if self.len > 0 && !self.overflow {
                if self.len < 5 {
                    self.buf[self.len] = tok;
                    self.len += 1;
                    self.char_bytes += usize::from(cls.trimmed_len[tu]);
                } else {
                    self.overflow = true;
                    self.len = 0;
                }
            }
            return (None, false);
        }
        let done = self.take_valid();
        (done, false)
    }
}

struct AtomCountMap {
    keys: Vec<u64>,
    counts: Vec<u32>,
    filled: usize,
}

impl AtomCountMap {
    fn new() -> Self {
        Self {
            keys: vec![0u64; PASS1_HASH_SLOTS],
            counts: vec![0u32; PASS1_HASH_SLOTS],
            filled: 0,
        }
    }

    #[inline]
    fn ensure_seed(&mut self, key: u64) {
        if key == 0 {
            return;
        }
        let mask = PASS1_HASH_SLOTS - 1;
        let mut idx = hash_atom_key(key) & mask;
        loop {
            let k = self.keys[idx];
            if k == key {
                return;
            }
            if k == 0 {
                self.keys[idx] = key;
                self.counts[idx] = 0;
                self.filled += 1;
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    #[inline]
    fn inc(&mut self, key: u64) {
        let mask = PASS1_HASH_SLOTS - 1;
        let mut idx = hash_atom_key(key) & mask;
        loop {
            let k = self.keys[idx];
            if k == key {
                self.counts[idx] = self.counts[idx].saturating_add(1);
                return;
            }
            if k == 0 {
                if self.filled < PASS1_LOAD_LIMIT {
                    self.keys[idx] = key;
                    self.counts[idx] = 1;
                    self.filled += 1;
                }
                return;
            }
            idx = (idx + 1) & mask;
        }
    }
}

struct WordIdLookup {
    keys: Vec<u64>,
    wids: Vec<u16>,
}

impl WordIdLookup {
    fn new(words: &[u64]) -> Self {
        let mut keys = vec![0u64; PASS2_HASH_SLOTS];
        let mut wids = vec![u16::MAX; PASS2_HASH_SLOTS];
        let mask = PASS2_HASH_SLOTS - 1;
        for (wid, &k) in words.iter().enumerate() {
            if k == 0 {
                continue;
            }
            let mut idx = hash_atom_key(k) & mask;
            while keys[idx] != 0 {
                idx = (idx + 1) & mask;
            }
            keys[idx] = k;
            wids[idx] = wid as u16;
        }
        Self { keys, wids }
    }

    #[inline]
    fn get(&self, key: u64) -> u16 {
        let mask = PASS2_HASH_SLOTS - 1;
        let mut idx = hash_atom_key(key) & mask;
        loop {
            let k = self.keys[idx];
            if k == key {
                return self.wids[idx];
            }
            if k == 0 {
                return u16::MAX;
            }
            idx = (idx + 1) & mask;
        }
    }
}

pub struct WordPpmiConfig {
    pub top_words: usize,
    pub stop_word_ranks: usize,
    pub ctx_skip_top: usize,
    pub prop_cap: usize,
    pub min_cand_freq: u64,
    pub min_h2_overlap_bits: u64,
}

impl WordPpmiConfig {
    pub fn production() -> Self {
        Self {
            top_words: 32_768,
            stop_word_ranks: 256,
            ctx_skip_top: 128,
            prop_cap: 1_024,
            min_cand_freq: 16,
            min_h2_overlap_bits: 18,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PropagatedAtom {
    pub atom_key: u64,
    pub margin: u64,
    pub h2_win_bits: u64,
    pub freq: u64,
}

pub struct CPpmiResult {
    pub corpus_tokens: u64,
    pub corpus_words_parsed: u64,
    pub unique_tracked_atoms: usize,
    pub active_words: usize,
    pub prop_job: Vec<PropagatedAtom>,
    pub prop_home: Vec<PropagatedAtom>,
}

fn read_uort_header(file: &mut File) -> Result<u64> {
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
    Ok(count)
}

pub fn build_c_ppmi_from_uort_file(
    c_uort_path: &Path,
    expected_c_sha_hex: &str,
    cls: &VocabWordClass,
    job_anchor_atoms: &[u64],
    home_anchor_atoms: &[u64],
    cfg: &WordPpmiConfig,
) -> Result<CPpmiResult> {
    let mut f1 = File::open(c_uort_path)?;
    let mut hdr = [0u8; 64];
    f1.read_exact(&mut hdr)?;
    if &hdr[..4] != b"UORT" {
        return Err(fail("C UORT header missing magic"));
    }
    let tok_count = u64::from_le_bytes(hdr[8..16].try_into()?);
    let file_len = f1.metadata()?.len();
    let want = 64u64
        .checked_add(
            tok_count
                .checked_mul(2)
                .ok_or_else(|| fail("C UORT token count overflow"))?,
        )
        .ok_or_else(|| fail("C UORT byte count overflow"))?;
    if file_len != want {
        return Err(fail(format!(
            "C UORT count {tok_count} expects {want} bytes, got {file_len}"
        )));
    }
    let mut hasher = Sha256::new();
    hasher.update(hdr);

    let mut pass1_map = AtomCountMap::new();
    for &ak in job_anchor_atoms.iter().chain(home_anchor_atoms.iter()) {
        pass1_map.ensure_seed(ak);
    }

    let mut parser = WordStreamParser::new();
    let mut words_parsed = 0u64;
    let mut chunk = vec![0u8; 4 << 20];
    let mut reader1 = BufReader::with_capacity(4 << 20, f1);
    let mut rem1 = tok_count * 2;
    while rem1 > 0 {
        let to_read = usize::try_from(rem1.min(chunk.len() as u64))?;
        let sl = &mut chunk[..to_read];
        reader1.read_exact(sl)?;
        hasher.update(&*sl);
        rem1 -= to_read as u64;
        for pair in sl.chunks_exact(2) {
            let tok = u16::from_le_bytes([pair[0], pair[1]]);
            if let (Some(atom), _) = parser.step(tok, cls) {
                pass1_map.inc(atom);
                words_parsed += 1;
            }
        }
    }
    if let Some(atom) = parser.take_valid() {
        pass1_map.inc(atom);
        words_parsed += 1;
    }
    let actual_c_sha = hex::encode(hasher.finalize());
    if actual_c_sha != expected_c_sha_hex {
        return Err(fail(format!(
            "C UORT SHA-256 mismatch: expected {expected_c_sha_hex}, got {actual_c_sha}"
        )));
    }

    let ctx_setup = prepare_pass2_tables(
        pass1_map,
        job_anchor_atoms,
        home_anchor_atoms,
        cfg,
        tok_count,
        words_parsed,
    );
    let mut f2 = File::open(c_uort_path)?;
    let cnt2 = read_uort_header(&mut f2)?;
    if cnt2 != tok_count {
        return Err(fail("C UORT Pass 2 header count changed"));
    }
    let mut reader2 = BufReader::with_capacity(4 << 20, f2);
    let mut state = Pass2AccumState::new(&ctx_setup);
    let mut p2 = WordStreamParser::new();
    let mut rem2 = tok_count * 2;
    while rem2 > 0 {
        let to_read = usize::try_from(rem2.min(chunk.len() as u64))?;
        let sl = &mut chunk[..to_read];
        reader2.read_exact(sl)?;
        rem2 -= to_read as u64;
        for pair in sl.chunks_exact(2) {
            let tok = u16::from_le_bytes([pair[0], pair[1]]);
            state.feed_token(tok, cls, &ctx_setup, &mut p2);
        }
    }
    state.finish_eof(&ctx_setup, &mut p2);
    Ok(score_ppmi_and_propagate(&ctx_setup, &state, cfg))
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn build_c_ppmi_from_slice(
    tokens: &[u16],
    cls: &VocabWordClass,
    job_anchor_atoms: &[u64],
    home_anchor_atoms: &[u64],
    cfg: &WordPpmiConfig,
) -> CPpmiResult {
    let mut pass1_map = AtomCountMap::new();
    for &ak in job_anchor_atoms.iter().chain(home_anchor_atoms.iter()) {
        pass1_map.ensure_seed(ak);
    }
    let mut parser = WordStreamParser::new();
    let mut words_parsed = 0u64;
    for &tok in tokens {
        if let (Some(atom), _) = parser.step(tok, cls) {
            pass1_map.inc(atom);
            words_parsed += 1;
        }
    }
    if let Some(atom) = parser.take_valid() {
        pass1_map.inc(atom);
        words_parsed += 1;
    }

    let ctx_setup = prepare_pass2_tables(
        pass1_map,
        job_anchor_atoms,
        home_anchor_atoms,
        cfg,
        tokens.len() as u64,
        words_parsed,
    );
    let mut state = Pass2AccumState::new(&ctx_setup);
    let mut p2 = WordStreamParser::new();
    for &tok in tokens {
        state.feed_token(tok, cls, &ctx_setup, &mut p2);
    }
    state.finish_eof(&ctx_setup, &mut p2);
    score_ppmi_and_propagate(&ctx_setup, &state, cfg)
}

struct Pass2Setup {
    corpus_tokens: u64,
    corpus_words_parsed: u64,
    unique_tracked_atoms: usize,
    words: Vec<u64>,
    freq: Vec<u64>,
    is_stop: Vec<bool>,
    ctx_slot: Vec<u16>,
    ctx_freq: Vec<u64>,
    ctx_slot_count: usize,
    anchor_role: Vec<u8>,
    anchor_weight: Vec<u64>,
    wid_lookup: WordIdLookup,
}

fn prepare_pass2_tables(
    pass1_map: AtomCountMap,
    job_anchor_atoms: &[u64],
    home_anchor_atoms: &[u64],
    cfg: &WordPpmiConfig,
    corpus_tokens: u64,
    corpus_words_parsed: u64,
) -> Pass2Setup {
    let unique_tracked_atoms = pass1_map.filled;
    let mut count_by_key: BTreeMap<u64, u64> = BTreeMap::new();
    for (&k, &c) in pass1_map.keys.iter().zip(pass1_map.counts.iter()) {
        if k != 0 {
            count_by_key.insert(k, u64::from(c));
        }
    }
    let mut ranked: Vec<(u64, u64)> = count_by_key
        .iter()
        .filter(|(_, &c)| c > 0)
        .map(|(&k, &c)| (k, c))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut selected: Vec<(u64, u64)> = ranked.iter().copied().take(cfg.top_words).collect();
    let mut in_selected: BTreeMap<u64, usize> = selected
        .iter()
        .enumerate()
        .map(|(i, &(k, _))| (k, i))
        .collect();
    for &ak in job_anchor_atoms.iter().chain(home_anchor_atoms.iter()) {
        in_selected.entry(ak).or_insert_with(|| {
            let c = count_by_key.get(&ak).copied().unwrap_or(0);
            let wid = selected.len();
            selected.push((ak, c));
            wid
        });
    }

    let n_words = selected.len();
    let words: Vec<u64> = selected.iter().map(|&(k, _)| k).collect();
    let freq: Vec<u64> = selected.iter().map(|&(_, c)| c).collect();
    let mut is_stop = vec![false; n_words];
    for &(k, _) in ranked.iter().take(cfg.stop_word_ranks) {
        if let Some(&wid) = in_selected.get(&k) {
            is_stop[wid] = true;
        }
    }

    let mut ctx_slot = vec![u16::MAX; n_words];
    let mut ctx_freq = [0u64; CTX_SLOTS];
    let mut ctx_slot_count = 0usize;
    let skip_top = if ranked.len() > cfg.ctx_skip_top {
        cfg.ctx_skip_top
    } else {
        0
    };
    for &(k, c) in ranked.iter().skip(skip_top) {
        if ctx_slot_count >= CTX_SLOTS {
            break;
        }
        if let Some(&wid) = in_selected.get(&k) {
            ctx_slot[wid] = ctx_slot_count as u16;
            ctx_freq[ctx_slot_count] = c.max(1);
            ctx_slot_count += 1;
        }
    }

    let mut anchor_role = vec![0u8; n_words];
    let mut anchor_weight = vec![0u64; n_words];
    for &ak in job_anchor_atoms {
        if let Some(&wid) = in_selected.get(&ak) {
            anchor_role[wid] = 1;
            let fw = freq[wid].min(1u64 << 20);
            anchor_weight[wid] = ((1u64 << 20) / (fw + 16_384)).max(1);
        }
    }
    for &ak in home_anchor_atoms {
        if let Some(&wid) = in_selected.get(&ak) {
            anchor_role[wid] = 2;
            let fw = freq[wid].min(1u64 << 20);
            anchor_weight[wid] = ((1u64 << 20) / (fw + 16_384)).max(1);
        }
    }

    let wid_lookup = WordIdLookup::new(&words);
    Pass2Setup {
        corpus_tokens,
        corpus_words_parsed,
        unique_tracked_atoms,
        words,
        freq,
        is_stop,
        ctx_slot,
        ctx_freq: ctx_freq.to_vec(),
        ctx_slot_count,
        anchor_role,
        anchor_weight,
        wid_lookup,
    }
}

struct Pass2AccumState {
    co_ctx: Vec<u32>,
    co_job: Vec<u64>,
    co_home: Vec<u64>,
    total_ctx_pairs: u64,
    ring: [u16; RING_WINDOW],
    rlen: usize,
}

impl Pass2AccumState {
    fn new(setup: &Pass2Setup) -> Self {
        let n_words = setup.words.len();
        Self {
            co_ctx: vec![0u32; n_words * CTX_SLOTS],
            co_job: vec![0u64; n_words],
            co_home: vec![0u64; n_words],
            total_ctx_pairs: 0,
            ring: [u16::MAX; RING_WINDOW],
            rlen: 0,
        }
    }

    #[inline]
    fn push_atom(&mut self, atom_opt: Option<u64>, rec_break: bool, setup: &Pass2Setup) {
        if let Some(atom) = atom_opt {
            let wb = setup.wid_lookup.get(atom);
            if wb != u16::MAX {
                let wbu = usize::from(wb);
                let cs_b = setup.ctx_slot[wbu];
                let ar_b = setup.anchor_role[wbu];
                let aw_b = setup.anchor_weight[wbu];
                for &wa in &self.ring[..self.rlen] {
                    if wa == u16::MAX {
                        continue;
                    }
                    let wau = usize::from(wa);
                    if cs_b != u16::MAX {
                        let idx = wau * CTX_SLOTS + usize::from(cs_b);
                        self.co_ctx[idx] = self.co_ctx[idx].saturating_add(1);
                        self.total_ctx_pairs += 1;
                    }
                    let cs_a = setup.ctx_slot[wau];
                    if cs_a != u16::MAX {
                        let idx = wbu * CTX_SLOTS + usize::from(cs_a);
                        self.co_ctx[idx] = self.co_ctx[idx].saturating_add(1);
                        self.total_ctx_pairs += 1;
                    }
                    let ar_a = setup.anchor_role[wau];
                    if ar_a == 1 {
                        self.co_job[wbu] =
                            self.co_job[wbu].saturating_add(setup.anchor_weight[wau]);
                    } else if ar_a == 2 {
                        self.co_home[wbu] =
                            self.co_home[wbu].saturating_add(setup.anchor_weight[wau]);
                    }
                    if ar_b == 1 {
                        self.co_job[wau] = self.co_job[wau].saturating_add(aw_b);
                    } else if ar_b == 2 {
                        self.co_home[wau] = self.co_home[wau].saturating_add(aw_b);
                    }
                }
            }
            if self.rlen < RING_WINDOW {
                self.ring[self.rlen] = wb;
                self.rlen += 1;
            } else {
                self.ring.copy_within(1..RING_WINDOW, 0);
                self.ring[RING_WINDOW - 1] = wb;
            }
        }
        if rec_break {
            self.rlen = 0;
        }
    }

    #[inline]
    fn feed_token(
        &mut self,
        tok: u16,
        cls: &VocabWordClass,
        setup: &Pass2Setup,
        parser: &mut WordStreamParser,
    ) {
        let (atom_opt, rec_break) = parser.step(tok, cls);
        self.push_atom(atom_opt, rec_break, setup);
    }

    fn finish_eof(&mut self, setup: &Pass2Setup, parser: &mut WordStreamParser) {
        let atom_opt = parser.take_valid();
        self.push_atom(atom_opt, true, setup);
    }
}

fn score_ppmi_and_propagate(
    setup: &Pass2Setup,
    state: &Pass2AccumState,
    cfg: &WordPpmiConfig,
) -> CPpmiResult {
    let n_words = setup.words.len();
    let tot_w = u128::from(state.total_ctx_pairs.max(1));
    let mut sigs = vec![[0u64; 16]; n_words];

    for (w, sig_slot) in sigs.iter_mut().enumerate().take(n_words) {
        if setup.freq[w] == 0 {
            continue;
        }
        let fw = u128::from(setup.freq[w]);
        let base = w * CTX_SLOTS;
        let mut ranked: Vec<(u128, usize)> = Vec::with_capacity(setup.ctx_slot_count);
        for (c, &cf) in setup.ctx_freq.iter().enumerate().take(setup.ctx_slot_count) {
            let cnt = u128::from(state.co_ctx[base + c]);
            if cnt == 0 {
                continue;
            }
            let num = (cnt * tot_w) << 16;
            let den = (fw * u128::from(cf)).max(1);
            let ratio = num / den;
            if ratio > (1u128 << 16) {
                ranked.push((ratio, c));
            }
        }
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut s = [0u64; 16];
        for &(_r, bit) in ranked.iter().take(TOP_PPMI_BITS) {
            s[bit >> 6] |= 1u64 << (bit & 63);
        }
        *sig_slot = s;
    }

    let mut job_anchors = Vec::new();
    let mut home_anchors = Vec::new();
    for (w, &role) in setup.anchor_role.iter().enumerate() {
        if role == 1 {
            job_anchors.push(w);
        } else if role == 2 {
            home_anchors.push(w);
        }
    }

    let top3_popcount = |w: usize, anchors: &[usize]| -> (u64, u64) {
        let mut ovs: Vec<u64> = anchors
            .iter()
            .map(|&a| popcount_1024(&sigs[w], &sigs[a]))
            .collect();
        ovs.sort_unstable_by(|a, b| b.cmp(a));
        let top_sum: u64 = ovs.iter().take(3).sum();
        let best_single = ovs.first().copied().unwrap_or(0);
        (top_sum << 9, best_single)
    };

    let mut prop_job: Vec<PropagatedAtom> = Vec::new();
    let mut prop_home: Vec<PropagatedAtom> = Vec::new();
    let min_h2_scaled = cfg.min_h2_overlap_bits << 9;
    for w in 0..n_words {
        if setup.is_stop[w] || setup.anchor_role[w] != 0 || setup.freq[w] < cfg.min_cand_freq {
            continue;
        }
        let (h2_j, best_j) = top3_popcount(w, &job_anchors);
        let (h2_h, best_h) = top3_popcount(w, &home_anchors);
        let h1_j = ((state.co_job[w] << 14) / (setup.freq[w] + 256)).min(65_536);
        let h1_h = ((state.co_home[w] << 14) / (setup.freq[w] + 256)).min(65_536);
        let sj = 2 * h2_j + h1_j;
        let sh = 2 * h2_h + h1_h;
        if sj * 4 >= sh * 5 && sj > sh && h2_j >= min_h2_scaled {
            prop_job.push(PropagatedAtom {
                atom_key: setup.words[w],
                margin: sj - sh,
                h2_win_bits: best_j,
                freq: setup.freq[w],
            });
        } else if sh * 4 >= sj * 5 && sh > sj && h2_h >= min_h2_scaled {
            prop_home.push(PropagatedAtom {
                atom_key: setup.words[w],
                margin: sh - sj,
                h2_win_bits: best_h,
                freq: setup.freq[w],
            });
        }
    }
    prop_job.sort_by(|a, b| b.margin.cmp(&a.margin).then(a.atom_key.cmp(&b.atom_key)));
    prop_home.sort_by(|a, b| b.margin.cmp(&a.margin).then(a.atom_key.cmp(&b.atom_key)));
    prop_job.truncate(cfg.prop_cap);
    prop_home.truncate(cfg.prop_cap);

    CPpmiResult {
        corpus_tokens: setup.corpus_tokens,
        corpus_words_parsed: setup.corpus_words_parsed,
        unique_tracked_atoms: setup.unique_tracked_atoms,
        active_words: n_words,
        prop_job,
        prop_home,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vocab_class() -> VocabWordClass {
        let mut cls = VocabWordClass {
            is_word_start: [false; P3_VOCABULARY],
            is_word_cont: [false; P3_VOCABULARY],
            trimmed_len: [0u8; P3_VOCABULARY],
        };
        // Tokens 10..200: word starts (3 bytes trimmed, so single-token valid words)
        for t in 10..200usize {
            cls.is_word_start[t] = true;
            cls.trimmed_len[t] = 3;
        }
        // Tokens 200..400: 2-byte word starts (need continuation token to hit >= 3 bytes)
        for t in 200..400usize {
            cls.is_word_start[t] = true;
            cls.trimmed_len[t] = 2;
        }
        // Tokens 400..500: continuation subwords (3 bytes each)
        for t in 400..500usize {
            cls.is_word_cont[t] = true;
            cls.trimmed_len[t] = 3;
        }
        cls
    }

    #[test]
    fn injective_word_atom_roundtrip_and_multitoken_two_hop_ppmi() {
        let cls = test_vocab_class();
        let a_job = pack_word_atom(&[210, 410]).unwrap();
        let syn_job = pack_word_atom(&[211, 411]).unwrap();
        let a_home = pack_word_atom(&[220, 420]).unwrap();
        let syn_home = pack_word_atom(&[221, 421]).unwrap();
        assert_eq!(unpack_word_atom(syn_job), ([211, 411, 0, 0, 0], 2));
        assert_ne!(a_job, syn_job);

        // Construct filler words so context basis slots 12..60 exist cleanly, plus shared 2-hop
        // context words connecting `a_job` <-> `syn_job` and `a_home` <-> `syn_home` without direct 1-hop overlap!
        let mut toks = Vec::new();
        for f in 100u16..140u16 {
            for _ in 0..100 {
                toks.extend_from_slice(&[0, f, f + 1]);
            }
        }
        for _ in 0..10 {
            for c in 20u16..34u16 {
                toks.extend_from_slice(&[0, 210, 410, c]);
                toks.extend_from_slice(&[0, 211, 411, c]);
            }
            for c in 40u16..54u16 {
                toks.extend_from_slice(&[0, 220, 420, c]);
                toks.extend_from_slice(&[0, 221, 421, c]);
            }
        }
        let cfg = WordPpmiConfig {
            top_words: 4096,
            stop_word_ranks: 4,
            ctx_skip_top: 12,
            prop_cap: 64,
            min_cand_freq: 4,
            min_h2_overlap_bits: 1,
        };
        let res = build_c_ppmi_from_slice(&toks, &cls, &[a_job], &[a_home], &cfg);
        assert!(res.prop_job.iter().any(|p| p.atom_key == syn_job));
        assert!(!res.prop_job.iter().any(|p| p.atom_key == syn_home));
        assert!(res.prop_home.iter().any(|p| p.atom_key == syn_home));
        assert!(!res.prop_home.iter().any(|p| p.atom_key == syn_job));
    }
}
