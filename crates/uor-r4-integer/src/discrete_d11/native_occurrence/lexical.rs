//! A signed lexical contribution addressed by relative native orbit transforms.

use super::super::{cl06_bfs, orbit_root_id};
use crate::{invalid, Result};

const BLOCKS: usize = 4;
const ROOTS: usize = 16;
const STATES: usize = 64;
pub const WEIGHT_WORDS: usize = BLOCKS * STATES;
pub(super) const MAGIC: &[u8; 8] = b"UORLEX01";
pub(super) const ORDERED_MAGIC: &[u8; 8] = b"UORLEX02";
pub(super) const MATCHED_ORDERED_MAGIC: &[u8; 8] = b"UORLEX03";
pub(super) const TRANSPORTED_MIDPOINT_MAGIC: &[u8; 8] = b"UORLEX04";
pub(super) const RAW_MIDPOINT_MAGIC: &[u8; 8] = b"UORLEX05";

const fn reachable_mask() -> u64 {
    let mut mask = 0;
    let mut index = 0;
    while index < super::super::BFS.len() {
        mask |= 1 << super::super::BFS[index];
        index += 1;
    }
    mask
}

/// Slots addressed by at least one transition between the 16 native orbit roots.
pub const MATCHED_SLOT_MASK: u64 = reachable_mask();
const MATCHED_COUNT: usize = MATCHED_SLOT_MASK.count_ones() as usize;
const MATCHED_SLOTS: [u16; MATCHED_COUNT] = {
    let mut slots = [0; MATCHED_COUNT];
    let mut remaining = MATCHED_SLOT_MASK;
    let mut index = 0;
    while remaining != 0 {
        slots[index] = remaining.trailing_zeros() as u16;
        remaining &= remaining - 1;
        index += 1;
    }
    slots
};

pub const STATE_BYTES: usize = MAGIC.len() + WEIGHT_WORDS * 2;
/// One signed word contributes 2^-16 on the prior's unit mass scale.
pub const SCORE_UNIT_Q48: i64 = 1 << 32;
const MAX_PRIOR_Q48: u64 = 1 << 48;
const SUFFIX_LENGTHS: [usize; BLOCKS] = [1, 4, 16, 2048];

/// One partial FNV-1a32 hash per exact suffix of the active original-ID history.
pub struct OrderedPrefix([u32; BLOCKS]);

impl OrderedPrefix {
    pub fn new(history: &[u16]) -> Result<Self> {
        if history.is_empty()
            || history.len() > SUFFIX_LENGTHS[3]
            || history.iter().any(|&id| id >= 4096)
        {
            return Err(invalid("invalid ordered lexical history"));
        }
        Ok(Self(SUFFIX_LENGTHS.map(|length| {
            history[history.len().saturating_sub(length)..]
                .iter()
                .copied()
                .fold(2_166_136_261, hash_original_id)
        })))
    }

    pub fn states(&self) -> [u32; BLOCKS] {
        self.0
    }

    pub fn addresses(&self, candidate: u16) -> Result<[u16; BLOCKS]> {
        self.address_with(candidate, |hash| (hash >> 26) as u16)
    }

    /// Ordered control using the same reachable words per bank as D11 geometry.
    pub fn matched_addresses(&self, candidate: u16) -> Result<[u16; BLOCKS]> {
        self.address_with(candidate, |hash| {
            MATCHED_SLOTS[(hash % MATCHED_COUNT as u32) as usize]
        })
    }

    fn address_with(&self, candidate: u16, slot: impl Fn(u32) -> u16) -> Result<[u16; BLOCKS]> {
        if !(1..4096).contains(&candidate) {
            return Err(invalid("invalid ordered lexical original candidate ID"));
        }
        Ok(std::array::from_fn(|bank| {
            let mut hash = hash_original_id(self.0[bank], candidate);
            hash ^= hash >> 16;
            hash = hash.wrapping_mul(0x85eb_ca6b);
            hash ^= hash >> 13;
            (bank * STATES) as u16 + slot(hash)
        }))
    }
}

fn hash_original_id(hash: u32, id: u16) -> u32 {
    id.to_le_bytes().into_iter().fold(hash, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(16_777_619)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Carry([u8; BLOCKS]);

impl Carry {
    /// Decode D11's packed recurrence roots, with block zero in the low nibble.
    pub const fn from_packed(packed: u16) -> Self {
        Self([
            (packed & 15) as u8,
            ((packed >> 4) & 15) as u8,
            ((packed >> 8) & 15) as u8,
            ((packed >> 12) & 15) as u8,
        ])
    }

    /// Recover the exact orbit roots from the ordinary writer's carried read vector.
    pub fn from_vector(vector: &[i32]) -> Result<Self> {
        let (blocks, remainder) = vector.as_chunks::<8>();
        if blocks.len() != BLOCKS || !remainder.is_empty() {
            return Err(invalid("lexical carry must contain four native blocks"));
        }
        let mut roots = [0; BLOCKS];
        for (root, block) in roots.iter_mut().zip(blocks) {
            *root = orbit_root_id(block)
                .ok_or_else(|| invalid("lexical carry block lies outside the native orbit"))?
                as u8;
        }
        Ok(Self(roots))
    }

    /// One address per block, shared by all original IDs with these relative transforms.
    pub fn addresses(&self, candidate_roots: [u8; BLOCKS]) -> Result<[u16; BLOCKS]> {
        if candidate_roots
            .iter()
            .any(|&root| usize::from(root) >= ROOTS)
        {
            return Err(invalid(
                "lexical candidate root lies outside the native orbit",
            ));
        }
        Ok(std::array::from_fn(|block| {
            let source = usize::from(self.0[block]);
            let target = usize::from(candidate_roots[block]);
            (block * STATES + cl06_bfs(source, target)) as u16
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Weights([i16; WEIGHT_WORDS]);

impl Weights {
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        Self::decode(bytes, MAGIC)
    }

    pub(super) fn decode(bytes: &[u8], magic: &[u8; 8]) -> Result<Self> {
        if bytes.len() != STATE_BYTES || !bytes.starts_with(magic) {
            return Err(invalid(
                "lexical state has the wrong version or byte length",
            ));
        }
        let mut words = [0; WEIGHT_WORDS];
        for (word, bytes) in words.iter_mut().zip(bytes[MAGIC.len()..].chunks_exact(2)) {
            *word = i16::from_le_bytes([bytes[0], bytes[1]]);
        }
        Ok(Self(words))
    }

    pub fn to_le_bytes(&self) -> [u8; STATE_BYTES] {
        Self::encode(&self.0, MAGIC)
    }

    pub(super) fn encode(words: &[i16; WEIGHT_WORDS], magic: &[u8; 8]) -> [u8; STATE_BYTES] {
        let mut bytes = [0; STATE_BYTES];
        bytes[..MAGIC.len()].copy_from_slice(magic);
        for (encoded, word) in bytes[MAGIC.len()..].chunks_exact_mut(2).zip(words) {
            encoded.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The contribution is a signed ranking adjustment on the prior's scale.
    /// The original token ID remains the caller's commit identity.
    pub fn score_q48(
        &self,
        carry: &Carry,
        candidate_roots: [u8; BLOCKS],
        prior_q48: u64,
    ) -> Result<i64> {
        self.score_addresses_q48(carry.addresses(candidate_roots)?, prior_q48)
    }

    /// The same scalar election applies to either lawful address operator.
    pub fn score_addresses_q48(&self, addresses: [u16; BLOCKS], prior_q48: u64) -> Result<i64> {
        if prior_q48 > MAX_PRIOR_Q48 {
            return Err(invalid("lexical prior is outside the Q48 mass domain"));
        }
        let mut adjustment = 0;
        for (bank, address) in addresses.into_iter().map(usize::from).enumerate() {
            if address / STATES != bank {
                return Err(invalid("lexical address lies outside its bank"));
            }
            adjustment += i64::from(self.0[address]);
        }
        // Four i16 magnitudes bound the scaled adjustment by 2^49.
        Ok(prior_q48 as i64 + adjustment * SCORE_UNIT_Q48)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{cl06_action, orbit_root_coords};
    use super::*;

    fn vector(roots: [usize; BLOCKS]) -> Vec<i32> {
        roots.into_iter().flat_map(orbit_root_coords).collect()
    }

    #[test]
    fn relative_addresses_map_each_carry_root_to_the_candidate() {
        for source in 0..ROOTS {
            let carried = vector([source; BLOCKS]);
            let carry = Carry::from_vector(&carried).unwrap();
            for target in 0..ROOTS {
                let addresses = carry.addresses([target as u8; BLOCKS]).unwrap();
                for (block, address) in addresses.into_iter().enumerate() {
                    let (bank, action) =
                        (usize::from(address) / STATES, usize::from(address) % STATES);
                    assert_eq!(bank, block);
                    let carried = carried[block * 8..(block + 1) * 8].try_into().unwrap();
                    assert_eq!(cl06_action(action, carried), orbit_root_coords(target));
                }
            }
        }
    }

    #[test]
    fn saved_signed_words_reverse_a_stronger_same_coarse_rival() {
        let carry = Carry::from_vector(&vector([2, 5, 9, 12])).unwrap();
        assert_eq!(Carry::from_packed(0xc952), carry);
        let target = [6, 7, 4, 1];
        let rival = [6, 7, 11, 1];
        let target_prior = 55 * SCORE_UNIT_Q48 as u64;
        let rival_prior = 56 * SCORE_UNIT_Q48 as u64;
        let mut bytes = Weights([0; WEIGHT_WORDS]).to_le_bytes();
        let zero = Weights::from_le_bytes(&bytes).unwrap();
        assert_eq!(
            zero.score_q48(&carry, target, target_prior).unwrap(),
            target_prior as i64
        );
        assert_eq!(
            zero.score_q48(&carry, rival, rival_prior).unwrap(),
            rival_prior as i64
        );

        let [_, _, target_address, _] = carry.addresses(target).unwrap();
        let [_, _, rival_address, _] = carry.addresses(rival).unwrap();
        assert_ne!(target_address, rival_address);
        for (address, value) in [(target_address, 2i16), (rival_address, -1i16)] {
            let offset = MAGIC.len() + 2 * usize::from(address);
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        let loaded = Weights::from_le_bytes(&bytes).unwrap();
        assert_eq!(loaded.to_le_bytes(), bytes);
        let target_score = loaded.score_q48(&carry, target, target_prior).unwrap();
        let rival_score = loaded.score_q48(&carry, rival, rival_prior).unwrap();
        assert_eq!(target_score, 57 * SCORE_UNIT_Q48);
        assert_eq!(rival_score, 55 * SCORE_UNIT_Q48);
        assert!(target_score > rival_score);
        assert_eq!(
            loaded.score_q48(&carry, target, 0).unwrap(),
            2 * SCORE_UNIT_Q48
        );
    }

    #[test]
    fn rejects_wrong_state_carry_candidate_and_prior() {
        let weights = Weights([0; WEIGHT_WORDS]);
        let mut bytes = weights.to_le_bytes();
        assert!(Weights::from_le_bytes(&bytes[..STATE_BYTES - 1]).is_err());
        bytes[7] = b'2';
        assert!(Weights::from_le_bytes(&bytes).is_err());
        assert!(Carry::from_vector(&[0; 31]).is_err());
        assert!(Carry::from_vector(&[0; 32]).is_err());
        let carry = Carry::from_vector(&vector([0; BLOCKS])).unwrap();
        assert!(weights.score_q48(&carry, [0, 0, 0, 16], 0).is_err());
        assert!(weights
            .score_q48(&carry, [0; BLOCKS], MAX_PRIOR_Q48 + 1)
            .is_err());
    }
}
