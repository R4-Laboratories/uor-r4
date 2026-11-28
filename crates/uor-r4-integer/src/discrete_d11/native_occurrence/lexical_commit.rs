//! Optional original-ID election over the ordinary P4 candidate masses.

use super::super::IntegerDiscreteModel;
use super::lexical::{
    Carry, OrderedPrefix, Weights, MAGIC, ORDERED_MAGIC, STATE_BYTES, WEIGHT_WORDS,
};
use crate::{invalid, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// P4 state identity, versioned words, SHA-256 of both.
pub const BOUND_BYTES: usize = 32 + STATE_BYTES + 32;

pub struct LexicalHead {
    weights: Weights,
    ordered: bool,
}

#[derive(Debug, Serialize)]
pub struct LexicalChoice {
    pub carry_code: Option<u16>,
    pub token: u16,
    pub prior_q48: u64,
    pub score_q48: i64,
    pub top_ties: u32,
    pub addresses: [u16; 4],
}

impl LexicalHead {
    /// Bind the same signed word layout to a P4 state and explicit address operator.
    pub fn encode_bound(
        state: [u8; 32],
        ordered: bool,
        words: &[i16; WEIGHT_WORDS],
    ) -> [u8; BOUND_BYTES] {
        let magic = if ordered { ORDERED_MAGIC } else { MAGIC };
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
        let ordered = payload[32..].starts_with(ORDERED_MAGIC);
        let magic = if ordered { ORDERED_MAGIC } else { MAGIC };
        Ok(Self {
            weights: Weights::decode(&payload[32..], magic)?,
            ordered,
        })
    }

    pub fn is_ordered(&self) -> bool {
        self.ordered
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
        let (carry_code, ordered_prefix) = if self.ordered {
            (None, Some(OrderedPrefix::new(prefix)?))
        } else {
            let pointer = trunk
                .trunk
                .as_ref()
                .ok_or_else(|| invalid("lexical election requires a recurrence trunk"))?;
            (
                Some(pointer.compute_rec_code(prefix, &trunk.codebook_roots)?),
                None,
            )
        };
        let carry = carry_code.map(Carry::from_packed);
        let addresses = |id: usize| match &ordered_prefix {
            Some(ordered) => ordered.addresses(id as u16),
            None => carry
                .ok_or_else(|| invalid("missing native lexical carry"))?
                .addresses(trunk.codebook_roots[id]),
        };
        let mut best = i64::MIN;
        let mut token = 0usize;
        let mut ties = 0;
        for (id, &prior) in priors.iter().enumerate().skip(1) {
            let score = self.weights.score_addresses_q48(addresses(id)?, prior)?;
            if score > best {
                best = score;
                token = id;
                ties = 1;
            } else if score == best {
                ties += 1;
            }
        }
        Ok(LexicalChoice {
            carry_code,
            token: token as u16,
            prior_q48: priors[token],
            score_q48: best,
            top_ties: ties,
            addresses: addresses(token)?,
        })
    }
}
