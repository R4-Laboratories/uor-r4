//! Optional original-ID election over the ordinary P4 candidate masses.

use super::super::{IntegerDiscreteModel, P4LexicalCarry};
use super::lexical::{
    AugmentedAddressing, Carry, OrderedPrefix, Weights, JOINT_MAGIC, MAGIC, MATCHED_ORDERED_MAGIC,
    ORDERED_MAGIC, RAW_MIDPOINT_MAGIC, STATE_BYTES, TRANSPORTED_MIDPOINT_MAGIC, UNARY_MAGIC,
    WEIGHT_WORDS,
};
use crate::{invalid, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// P4 state identity, versioned words, SHA-256 of both.
pub const BOUND_BYTES: usize = 32 + STATE_BYTES + 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Addressing {
    Geometry(P4LexicalCarry),
    Augmented(AugmentedAddressing),
    Ordered,
    MatchedOrdered,
}

impl Addressing {
    fn magic(self) -> &'static [u8; 8] {
        match self {
            Self::Geometry(P4LexicalCarry::Snapped) => MAGIC,
            Self::Geometry(P4LexicalCarry::TransportedMidpoint) => TRANSPORTED_MIDPOINT_MAGIC,
            Self::Geometry(P4LexicalCarry::RawMidpoint) => RAW_MIDPOINT_MAGIC,
            Self::Augmented(AugmentedAddressing::Joint) => JOINT_MAGIC,
            Self::Augmented(AugmentedAddressing::Unary) => UNARY_MAGIC,
            Self::Ordered => ORDERED_MAGIC,
            Self::MatchedOrdered => MATCHED_ORDERED_MAGIC,
        }
    }
}

pub struct LexicalHead {
    weights: Weights,
    addressing: Addressing,
}

#[derive(Debug, Serialize)]
pub struct LexicalChoice {
    pub carry_code: Option<u16>,
    pub token: u16,
    pub prior_q48: u64,
    pub score_q48: i64,
    pub top_ties: u32,
    pub addresses: [u16; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_address: Option<u16>,
}

impl LexicalHead {
    /// Bind the same signed word layout to a P4 state and explicit address operator.
    pub fn encode_bound(
        state: [u8; 32],
        ordered: bool,
        words: &[i16; WEIGHT_WORDS],
    ) -> [u8; BOUND_BYTES] {
        Self::encode(
            state,
            if ordered {
                Addressing::Ordered
            } else {
                Addressing::Geometry(P4LexicalCarry::Snapped)
            },
            words,
        )
    }

    pub fn encode_matched_bound(state: [u8; 32], words: &[i16; WEIGHT_WORDS]) -> [u8; BOUND_BYTES] {
        Self::encode(state, Addressing::MatchedOrdered, words)
    }

    pub fn encode_geometric_bound(
        state: [u8; 32],
        operator: P4LexicalCarry,
        words: &[i16; WEIGHT_WORDS],
    ) -> [u8; BOUND_BYTES] {
        Self::encode(state, Addressing::Geometry(operator), words)
    }

    /// Bind the augmented native state, requiring unused physical words to be zero.
    pub fn encode_augmented_bound(
        state: [u8; 32],
        operator: AugmentedAddressing,
        words: &[i16; WEIGHT_WORDS],
    ) -> Result<[u8; BOUND_BYTES]> {
        Weights::validate_augmented(words)?;
        Ok(Self::encode(state, Addressing::Augmented(operator), words))
    }

    fn encode(
        state: [u8; 32],
        addressing: Addressing,
        words: &[i16; WEIGHT_WORDS],
    ) -> [u8; BOUND_BYTES] {
        let magic = addressing.magic();
        let mut out = [0; BOUND_BYTES];
        let (payload, checksum) = out.split_at_mut(BOUND_BYTES - 32);
        payload[..32].copy_from_slice(&state);
        payload[32..].copy_from_slice(&Weights::encode(words, magic));
        checksum.copy_from_slice(&Sha256::digest(payload));
        out
    }

    pub fn bind(bytes: &[u8], state: &[u8; 32]) -> Result<Self> {
        if bytes.len() != BOUND_BYTES || &bytes[..32] != state {
            return Err(invalid("lexical head length or P4 state identity differs"));
        }
        let (payload, digest) = bytes.split_at(BOUND_BYTES - 32);
        if Sha256::digest(payload).as_slice() != digest {
            return Err(invalid("lexical head checksum differs"));
        }
        let addressing = match &payload[32..40] {
            magic if magic == MAGIC => Addressing::Geometry(P4LexicalCarry::Snapped),
            magic if magic == ORDERED_MAGIC => Addressing::Ordered,
            magic if magic == MATCHED_ORDERED_MAGIC => Addressing::MatchedOrdered,
            magic if magic == TRANSPORTED_MIDPOINT_MAGIC => {
                Addressing::Geometry(P4LexicalCarry::TransportedMidpoint)
            }
            magic if magic == RAW_MIDPOINT_MAGIC => {
                Addressing::Geometry(P4LexicalCarry::RawMidpoint)
            }
            magic if magic == JOINT_MAGIC => Addressing::Augmented(AugmentedAddressing::Joint),
            magic if magic == UNARY_MAGIC => Addressing::Augmented(AugmentedAddressing::Unary),
            _ => return Err(invalid("unknown lexical address operator")),
        };
        let weights = Weights::decode(&payload[32..], addressing.magic())?;
        if matches!(addressing, Addressing::Augmented(_)) {
            weights.validate_augmented_payload()?;
        }
        Ok(Self {
            weights,
            addressing,
        })
    }

    pub fn is_ordered(&self) -> bool {
        matches!(
            self.addressing,
            Addressing::Ordered | Addressing::MatchedOrdered
        )
    }

    pub fn addressing_name(&self) -> &'static str {
        match self.addressing {
            Addressing::Geometry(P4LexicalCarry::Snapped) => "geometry",
            Addressing::Geometry(P4LexicalCarry::TransportedMidpoint) => "midpoint-transported",
            Addressing::Geometry(P4LexicalCarry::RawMidpoint) => "midpoint-raw",
            Addressing::Augmented(AugmentedAddressing::Joint) => "geometry-joint",
            Addressing::Augmented(AugmentedAddressing::Unary) => "geometry-unary",
            Addressing::Ordered => "ordered",
            Addressing::MatchedOrdered => "ordered-matched",
        }
    }

    /// Preserve the ordinary trace's label for both historical ordered operators.
    pub fn ordinary_addressing_name(&self) -> &'static str {
        if self.is_ordered() {
            "ordered"
        } else {
            self.addressing_name()
        }
    }

    /// Replay the augmented lookup and signed score on supplied native operands.
    pub fn score_augmented_q48(
        &self,
        carry: &Carry,
        candidate_roots: [u8; 4],
        prior_q48: u64,
    ) -> Result<i64> {
        let Addressing::Augmented(operator) = self.addressing else {
            return Err(invalid("lexical head is not augmented"));
        };
        let (addresses, extra) = carry.augmented_addresses(candidate_roots, operator)?;
        self.weights
            .score_with_extra(addresses, Some(extra), prior_q48)
    }

    /// The caller supplies the current causal window and the masses just scored for it.
    /// The rank is signed and does not alter the model's probability masses.
    pub fn commit(
        &self,
        trunk: &IntegerDiscreteModel,
        prefix: &[u16],
        priors: &[u64],
    ) -> Result<LexicalChoice> {
        if prefix.is_empty()
            || prefix.len() > 2048
            || priors.len() != trunk.codebook_roots.len()
            || priors.len() != 4096
            || prefix.iter().any(|&id| id >= 4096)
        {
            return Err(invalid(
                "lexical election requires an aligned P4 vocabulary and causal prefix",
            ));
        }
        let (carry_code, ordered_prefix) = match self.addressing {
            Addressing::Geometry(_) | Addressing::Augmented(_) => {
                let operator = match self.addressing {
                    Addressing::Geometry(operator) => operator,
                    _ => P4LexicalCarry::Snapped,
                };
                let pointer = trunk
                    .trunk
                    .as_ref()
                    .ok_or_else(|| invalid("lexical election requires a recurrence trunk"))?;
                (
                    Some(pointer.compute_carry_code(prefix, &trunk.codebook_roots, operator)?),
                    None,
                )
            }
            Addressing::Ordered | Addressing::MatchedOrdered => {
                (None, Some(OrderedPrefix::new(prefix)?))
            }
        };
        let carry = carry_code.map(Carry::from_packed);
        let addresses = |id: usize| -> Result<([u16; 4], Option<u16>)> {
            if let Some(ordered) = &ordered_prefix {
                let slots = if self.addressing == Addressing::MatchedOrdered {
                    ordered.matched_addresses(id as u16)?
                } else {
                    ordered.addresses(id as u16)?
                };
                return Ok((slots, None));
            }
            let carry = carry.ok_or_else(|| invalid("missing native lexical carry"))?;
            let roots = trunk.codebook_roots[id];
            if let Addressing::Augmented(operator) = self.addressing {
                let (slots, extra) = carry.augmented_addresses(roots, operator)?;
                Ok((slots, Some(extra)))
            } else {
                Ok((carry.addresses(roots)?, None))
            }
        };
        let mut best = i64::MIN;
        let mut token = 0usize;
        let mut ties = 0;
        for (id, &prior) in priors.iter().enumerate().skip(1) {
            let (slots, extra) = addresses(id)?;
            let score = self.weights.score_with_extra(slots, extra, prior)?;
            if score > best {
                best = score;
                token = id;
                ties = 1;
            } else if score == best {
                ties += 1;
            }
        }
        let (addresses, extra_address) = addresses(token)?;
        Ok(LexicalChoice {
            carry_code,
            token: token as u16,
            prior_q48: priors[token],
            score_q48: best,
            top_ties: ties,
            addresses,
            extra_address,
        })
    }
}
