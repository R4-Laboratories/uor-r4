//! Optional original-ID election over the ordinary P4 candidate masses.

use super::super::IntegerDiscreteModel;
use super::lexical::{Carry, Weights, STATE_BYTES};
use crate::{invalid, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// P4 state identity, versioned words, SHA-256 of both.
pub const BOUND_BYTES: usize = 32 + STATE_BYTES + 32;

pub struct LexicalHead {
    weights: Weights,
}

#[derive(Debug, Serialize)]
pub struct LexicalChoice {
    pub carry_code: u16,
    pub token: u16,
    pub prior_q48: u64,
    pub score_q48: i64,
    pub top_ties: u32,
    pub addresses: [u16; 4],
}

impl LexicalHead {
    pub fn bind(bytes: &[u8], state: &[u8; 32]) -> Result<Self> {
        if bytes.len() != BOUND_BYTES || &bytes[..32] != state {
            return Err(invalid("lexical head length or P4 state identity differs"));
        }
        let (payload, digest) = bytes.split_at(BOUND_BYTES - 32);
        if Sha256::digest(payload).as_slice() != digest {
            return Err(invalid("lexical head checksum differs"));
        }
        Ok(Self {
            weights: Weights::from_le_bytes(&payload[32..])?,
        })
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
            || priors.len() != trunk.codebook_roots.len()
            || priors.len() != 4096
            || prefix.iter().any(|&id| id >= 4096)
        {
            return Err(invalid(
                "lexical election requires an aligned P4 vocabulary and causal prefix",
            ));
        }
        let pointer = trunk
            .trunk
            .as_ref()
            .ok_or_else(|| invalid("lexical election requires a recurrence trunk"))?;
        let carry_code = pointer.compute_rec_code(prefix, &trunk.codebook_roots)?;
        let carry = Carry::from_packed(carry_code);
        let mut best = i64::MIN;
        let mut token = 0usize;
        let mut ties = 0;
        for (id, (&roots, &prior)) in trunk.codebook_roots.iter().zip(priors).enumerate().skip(1) {
            let score = self.weights.score_q48(&carry, roots, prior)?;
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
            addresses: carry.addresses(trunk.codebook_roots[token])?,
        })
    }
}
