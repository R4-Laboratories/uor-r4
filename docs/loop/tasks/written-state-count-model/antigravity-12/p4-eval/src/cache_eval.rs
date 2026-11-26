#![forbid(unsafe_code)]

use crate::s3_decode::{S3_NORM_BUDGET, S3_Q48_ONE, S3_UNIFORM_MASS, S3_VOCAB};

const TRI_TABLE_CAP: usize = 16_384;
const TRI_MASK: usize = TRI_TABLE_CAP - 1;
const LCP_TABLE_CAP: usize = 16_384;
const LCP_MASK: usize = LCP_TABLE_CAP - 1;

#[inline]
fn lawful_mass(hit_w: u64, total_w: u64) -> u64 {
    if total_w == 0 {
        return S3_UNIFORM_MASS;
    }
    if hit_w == 0 {
        return 1;
    }
    1u64 + ((u128::from(hit_w) * u128::from(S3_NORM_BUDGET)) / u128::from(total_w)) as u64
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
    uni_w: [u32; S3_VOCAB],
    uni_touched: Vec<u16>,
    uni_total: u64,
    rec_tokens: Vec<u16>,

    bi_slot: [u32; S3_VOCAB],
    bi_touched: Vec<u16>,
    bi_lists: Vec<Vec<(u16, u16)>>,

    tri_keys: Vec<u32>,
    tri_slots: Vec<u32>,
    tri_touched: Vec<usize>,
    tri_lists: Vec<Vec<(u16, u16)>>,

    lcp_keys: Vec<u64>,
    lcp_slots: Vec<u32>,
    lcp_touched: Vec<usize>,
    lcp_lists: Vec<[usize; 9]>, // [len, p0..p7] ring buffer of up to 8 most recent follower indices
}

impl Default for CausalSessionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalSessionCache {
    pub fn new() -> Self {
        Self {
            uni_w: [0u32; S3_VOCAB],
            uni_touched: Vec::with_capacity(512),
            uni_total: 0,
            rec_tokens: Vec::with_capacity(4096),
            bi_slot: [u32::MAX; S3_VOCAB],
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
            // Bounded per-record safety guard: return last slot if dense record > 12k unique bigrams.
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

    /// Score target token `tok` causally from prior tokens in the current record ONLY, then update state.
    /// Returns `(cache_hit_state in 0..=2, [uni_q48, bi_q48, tri_q48, lcp_q48])`.
    pub fn score_and_advance(&mut self, tok: u16) -> (u8, [u64; 4]) {
        let pos = self.rec_tokens.len();

        // 1. Recency Unigram
        let uni_m = if self.uni_total == 0 {
            S3_UNIFORM_MASS
        } else {
            let hw = u64::from(self.uni_w[usize::from(tok)]);
            lawful_mass(hw, self.uni_total)
        };

        // 2. Local Bigram
        let mut hit_bi = false;
        let bi_m = if pos >= 1 {
            let p1 = self.rec_tokens[pos - 1];
            let sl = self.bi_slot[usize::from(p1)];
            if sl == u32::MAX {
                S3_UNIFORM_MASS
            } else {
                hit_bi = true;
                let items = &self.bi_lists[sl as usize];
                let mut tot = 0u64;
                let mut hit = 0u64;
                for &(nt, cnt) in items {
                    let c = u64::from(cnt);
                    tot += c;
                    if nt == tok {
                        hit = c;
                    }
                }
                lawful_mass(hit, tot)
            }
        } else {
            S3_UNIFORM_MASS
        };

        // 3. Local Trigram
        let mut hit_tri = false;
        let tri_m = if pos >= 2 {
            let p2 = self.rec_tokens[pos - 2];
            let p1 = self.rec_tokens[pos - 1];
            let tkey = (u32::from(p2) << 12) | u32::from(p1);
            if let Some(sl) = self.find_tri_slot(tkey) {
                hit_tri = true;
                let items = &self.tri_lists[sl];
                let mut tot = 0u64;
                let mut hit = 0u64;
                for &(nt, cnt) in items {
                    let c = u64::from(cnt);
                    tot += c;
                    if nt == tok {
                        hit = c;
                    }
                }
                lawful_mass(hit, tot)
            } else {
                S3_UNIFORM_MASS
            }
        } else {
            S3_UNIFORM_MASS
        };

        // 4. Local Bounded Suffix-Match Cache (keyed by (p3, p2, p1), up to L=16 match length)
        let mut hit_lcp = false;
        let lcp_m = if pos >= 3 {
            let p3 = self.rec_tokens[pos - 3];
            let p2 = self.rec_tokens[pos - 2];
            let p1 = self.rec_tokens[pos - 1];
            let lkey = (u64::from(p3) << 24) | (u64::from(p2) << 12) | u64::from(p1);
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
                            let a = self.rec_tokens[prev_follow_pos - 1 - match_len];
                            let b = self.rec_tokens[pos - 1 - match_len];
                            if a != b {
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
                    lawful_mass(hit_w, tot_w)
                } else {
                    S3_UNIFORM_MASS
                }
            } else {
                S3_UNIFORM_MASS
            }
        } else {
            S3_UNIFORM_MASS
        };

        let hit_state = if hit_tri || hit_lcp {
            2u8
        } else if hit_bi {
            1u8
        } else {
            0u8
        };

        // ---------- Causal State Update (Strictly AFTER scoring `tok`) ----------
        if (pos > 0) && (pos & 63 == 0) {
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
        (hit_state, [uni_m, bi_m, tri_m, lcp_m])
    }

    /// Score target `v` read-only without advancing state (for full-vocab simplex verification tests).
    #[cfg(test)]
    pub fn peek_score_readonly(&self, tok: u16) -> (u8, [u64; 4]) {
        let pos = self.rec_tokens.len();
        let uni_m = if self.uni_total == 0 {
            S3_UNIFORM_MASS
        } else {
            lawful_mass(u64::from(self.uni_w[usize::from(tok)]), self.uni_total)
        };
        let mut hit_bi = false;
        let bi_m = if pos >= 1 {
            let sl = self.bi_slot[usize::from(self.rec_tokens[pos - 1])];
            if sl == u32::MAX {
                S3_UNIFORM_MASS
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
                lawful_mass(hit, tot)
            }
        } else {
            S3_UNIFORM_MASS
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
                lawful_mass(hit, tot)
            } else {
                S3_UNIFORM_MASS
            }
        } else {
            S3_UNIFORM_MASS
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
                            let a = self.rec_tokens[prev_follow_pos - 1 - match_len];
                            let b = self.rec_tokens[pos - 1 - match_len];
                            if a != b {
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
                    lawful_mass(hit_w, tot_w)
                } else {
                    S3_UNIFORM_MASS
                }
            } else {
                S3_UNIFORM_MASS
            }
        } else {
            S3_UNIFORM_MASS
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
}

/// Compute strictly causal per-target `(cache_hit_states, [uni, bi, tri, lcp] Q48 masses)` for all non-BOS targets.
/// Resets all cache state at every `BOS = 0` delimiter (aligned with `r_targets` / `tune_targets`).
pub fn compute_causal_cache_features(tokens: &[u16]) -> (Vec<u8>, Vec<[u64; 4]>) {
    let mut cache = CausalSessionCache::new();
    let mut hit_states = Vec::with_capacity(tokens.len());
    let mut masses = Vec::with_capacity(tokens.len());
    for (at, &tok) in tokens.iter().enumerate() {
        if tok == 0 {
            cache.reset_record();
            continue;
        }
        if at == 0 {
            // Defensive alignment matching `stream_golden_and_multi_s2_keys` (`at == 0` non-BOS skipped).
            continue;
        }
        let (hs, ms) = cache.score_and_advance(tok);
        hit_states.push(hs);
        masses.push(ms);
    }
    debug_assert_eq!(S3_Q48_ONE, 1u64 << 48);
    (hit_states, masses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_experts_sum_to_lawful_q48_and_reset_at_bos() {
        let mut cache = CausalSessionCache::new();
        let rec1 = [10u16, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30];
        for &t in &rec1 {
            let (_hs, _ms) = cache.score_and_advance(t);
        }
        // Prefix is now [..., 10, 20, 30]: every single expert must have sum_{v=0..4095} m[e](v) <= 2^48 and m >= 1.
        let mut sums = [0u128; 4];
        for v in 0..S3_VOCAB as u16 {
            let (_hs, ms) = cache.peek_score_readonly(v);
            for e in 0..4 {
                assert!(ms[e] >= 1);
                assert!(ms[e] <= S3_Q48_ONE);
                sums[e] += u128::from(ms[e]);
            }
        }
        for (e, &sum_e) in sums.iter().enumerate() {
            assert!(
                sum_e <= u128::from(S3_Q48_ONE),
                "cache expert {e} exceeded 2^48: {sum_e}"
            );
        }
        let (hs_hit, ms_hit) = cache.score_and_advance(40);
        assert_eq!(hs_hit, 2);
        assert!(ms_hit[1] > S3_UNIFORM_MASS);
        assert!(ms_hit[2] > S3_UNIFORM_MASS);
        assert!(ms_hit[3] > S3_UNIFORM_MASS);

        // Record boundary BOS = 0 must completely wipe history.
        let stream = [0u16, 10, 20, 30, 40, 10, 20, 30, 40, 0, 10, 20, 30, 40];
        let (hss, mss) = compute_causal_cache_features(&stream);
        // After second BOS (target 10 at index 10 -> ord 8): uni_total == 0 -> all 4 are S3_UNIFORM_MASS, hit_state == 0!
        assert_eq!(hss.len(), 12);
        assert_eq!(hss[8], 0);
        assert_eq!(mss[8], [S3_UNIFORM_MASS; 4]);
    }
}
