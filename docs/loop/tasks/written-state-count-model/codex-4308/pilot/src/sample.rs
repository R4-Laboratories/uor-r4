use crate::{fail, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const TRAIN_SHA: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub const DEV_SHA: &str = "75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a";

/// Absolute BOS offset and positive, record-relative target position. The target is excluded from the prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    pub record: usize,
    pub target_at: usize,
}

pub struct Corpus {
    pub ids: Vec<u16>,
    starts: Vec<usize>,
    eligible_before: Vec<usize>,
}

pub struct Draw(u64);
impl Draw {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn index(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let word = z ^ (z >> 31);
        ((u128::from(word) * bound as u128) >> 64) as usize
    }
}

impl Corpus {
    pub fn load(path: &Path, expected: &str) -> Result<Self> {
        let raw = std::fs::read(path)?;
        if hex::encode(Sha256::digest(&raw)) != expected || raw.len() < 64 || &raw[..4] != b"UORT" {
            return Err(fail("incorrect pinned UORT input or header"));
        }
        let version = u32::from_le_bytes(raw[4..8].try_into()?);
        let n = usize::try_from(u64::from_le_bytes(raw[8..16].try_into()?))?;
        let vocab = u32::from_le_bytes(raw[16..20].try_into()?);
        if version != 1 || vocab != 4096 || n.checked_mul(2) != Some(raw.len() - 64) {
            return Err(fail("UORT version, vocabulary or length mismatch"));
        }
        let ids = raw[64..]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        Self::from_ids(ids)
    }

    fn from_ids(ids: Vec<u16>) -> Result<Self> {
        if ids.first() != Some(&0) || ids.iter().any(|&id| id >= 4096) {
            return Err(fail("invalid BOS or token id"));
        }
        let mut starts: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(j, &id)| (id == 0).then_some(j))
            .collect();
        starts.push(ids.len());
        let mut eligible_before = vec![0];
        for ends in starts.windows(2) {
            let count = ends[1] - ends[0] - 1;
            if count == 0 {
                return Err(fail("empty source record"));
            }
            eligible_before.push(eligible_before.last().copied().unwrap_or(0) + count);
        }
        Ok(Self {
            ids,
            starts,
            eligible_before,
        })
    }

    pub fn record_count(&self) -> usize {
        self.starts.len() - 1
    }
    pub fn target_count(&self) -> usize {
        *self.eligible_before.last().unwrap_or(&0)
    }
    pub fn prefix(&self, pos: Position) -> &[u16] {
        &self.ids[pos.record..pos.record + pos.target_at]
    }

    pub fn train_sample(
        &self,
        batches: usize,
        rows: usize,
        seed: u64,
    ) -> Result<Vec<Vec<Position>>> {
        let sample_size = batches
            .checked_mul(rows)
            .ok_or_else(|| fail("sample overflow"))?;
        let total = self.target_count();
        if sample_size == 0 || total < sample_size {
            return Err(fail("invalid training sample size"));
        }
        let mut draw = Draw::new(seed);
        let mut out = vec![Vec::with_capacity(rows); batches];
        for stratum in 0..sample_size {
            let lo = stratum * total / sample_size;
            let hi = (stratum + 1) * total / sample_size;
            let target = lo + draw.index(hi - lo);
            let r = self
                .eligible_before
                .partition_point(|&start| start <= target)
                - 1;
            out[stratum % batches].push(Position {
                record: self.starts[r],
                target_at: 1 + target - self.eligible_before[r],
            });
        }
        Ok(out)
    }

    pub fn dev_sample(&self, count: usize, seed: u64) -> Result<Vec<Position>> {
        if count == 0 || count > self.record_count() {
            return Err(fail("invalid development sample size"));
        }
        let mut draw = Draw::new(seed);
        Ok((0..count)
            .map(|i| {
                let lo = i * self.record_count() / count;
                let hi = (i + 1) * self.record_count() / count;
                let r = lo + draw.index(hi - lo);
                Position {
                    record: self.starts[r],
                    target_at: 1 + draw.index(self.starts[r + 1] - self.starts[r] - 1),
                }
            })
            .collect())
    }
}

pub fn sample_digest(corpus: &Corpus, positions: &[Position]) -> String {
    let mut h = Sha256::new();
    for p in positions {
        h.update((p.record as u64).to_le_bytes());
        h.update((p.target_at as u64).to_le_bytes());
        h.update(corpus.ids[p.record + p.target_at].to_le_bytes());
        h.update(prefix_digest(corpus, *p));
    }
    hex::encode(h.finalize())
}

pub fn prefix_digest(corpus: &Corpus, position: Position) -> [u8; 32] {
    let prefix = corpus.prefix(position);
    let active = &prefix[prefix.len().saturating_sub(2048)..];
    let words: Vec<u8> = active.iter().flat_map(|id| id.to_le_bytes()).collect();
    Sha256::digest(words).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_draw_visits_each_eligible_target_once_without_crossing_bos() {
        let corpus = Corpus::from_ids(vec![0, 10, 11, 0, 12, 13, 0, 14, 15]).unwrap();
        let mut targets = corpus
            .train_sample(2, 3, 4308)
            .unwrap()
            .into_iter()
            .flatten()
            .map(|p| {
                assert_eq!(corpus.prefix(p)[0], 0);
                assert!(!corpus.prefix(p)[1..].contains(&0));
                corpus.ids[p.record + p.target_at]
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        assert_eq!(targets, [10, 11, 12, 13, 14, 15]);
        let dev = corpus.dev_sample(3, 4309).unwrap();
        assert_eq!(dev.iter().map(|p| p.record).collect::<Vec<_>>(), [0, 3, 6]);
    }
}
