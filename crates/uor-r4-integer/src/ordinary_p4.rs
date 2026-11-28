//! Shared ordinary P4 text encoding, artifact binding and committed-token replay.

use crate::{discrete_d11::IntegerDiscreteModel, IntegerError, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
    ByteBpeTokenizer,
};

pub const MAX_MESSAGE_BYTES: usize = 1 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiscreteTextMessage {
    role: String,
    content: String,
}

pub fn encode_discrete_messages(
    bytes: &[u8],
    encoder: &DialogueEncoder<'_>,
    protocol: &DialogueProtocol,
) -> Result<Vec<u32>> {
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(IntegerError::Invalid("--messages exceeds 1 MiB".into()));
    }
    let messages: Vec<DiscreteTextMessage> = serde_json::from_slice(bytes).map_err(|_| {
        IntegerError::Invalid("--messages requires a JSON array of role/content objects".into())
    })?;
    if messages.len().is_multiple_of(2)
        || messages.iter().enumerate().any(|(i, message)| {
            let expected = if i % 2 == 0 { "user" } else { "assistant" };
            message.role != expected || message.content.trim().is_empty()
        })
    {
        return Err(IntegerError::Invalid(
            "--messages requires nonempty alternating user/assistant turns ending in a user turn"
                .into(),
        ));
    }
    let roles: Vec<_> = messages
        .iter()
        .map(|message| Message {
            role: &message.role,
            content: &message.content,
        })
        .collect();
    let encoded = encoder.encode_assistant_prefix(&roles);
    if encoded.emitted_turns != roles.len()
        || encoded.special_token_occurrences != 0
        || encoded
            .tokens
            .iter()
            .filter(|&&id| id == protocol.eos_id)
            .count()
            != roles.len() / 2
    {
        return Err(IntegerError::Invalid(
            "--messages contains an unencodable role or an embedded special token".into(),
        ));
    }
    Ok(encoded.tokens)
}

pub fn load_discrete_tokenizer(
    model: &IntegerDiscreteModel,
    path: &std::path::Path,
) -> Result<(ByteBpeTokenizer, String)> {
    let bytes = std::fs::read(path)?;
    let digest = hex::encode(Sha256::digest(&bytes));
    if model.tokenizer_sha256() != Some(digest.as_str()) {
        return Err(IntegerError::Invalid(
            "discrete artifact does not bind this tokenizer SHA-256".into(),
        ));
    }
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes)
        .ok_or_else(|| IntegerError::Invalid("invalid discrete tokenizer JSON".into()))?;
    if tokenizer.vocab_size() != model.shape().vocabulary as usize {
        return Err(IntegerError::Invalid(
            "discrete tokenizer vocabulary mismatch".into(),
        ));
    }
    Ok((tokenizer, digest))
}

#[cfg(any(feature = "p1-count-serving", feature = "p4-only-client"))]
mod count {
    use super::*;
    use crate::discrete_d11::native_occurrence::lexical_commit::{LexicalHead, BOUND_BYTES};
    use std::{io::Read, path::Path};
    use uor_r4_count_runtime::p4_serving::{P4Session, P4State, S4SourceGate, S4SourceMode};

    pub fn load_lexical(path: &Path, engine: &P4State<'_>) -> Result<(LexicalHead, String)> {
        let state = engine
            .state_sha256()
            .ok_or_else(|| IntegerError::Invalid("P4 state has no identity".into()))?;
        let mut bytes = Vec::with_capacity(BOUND_BYTES + 1);
        std::fs::File::open(path)?
            .take((BOUND_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        Ok((
            LexicalHead::bind(&bytes, &state)?,
            hex::encode(Sha256::digest(&bytes)),
        ))
    }

    pub fn step(
        session: &mut P4Session,
        model: &IntegerDiscreteModel,
        engine: &P4State<'_>,
        read_enabled: bool,
        trace: bool,
        source: Option<(&S4SourceGate, S4SourceMode)>,
        lexical: Option<&LexicalHead>,
    ) -> Result<(u16, Option<serde_json::Value>)> {
        let (id, observation) = session
            .step_with_source(engine, read_enabled, trace, source)
            .map_err(|e| IntegerError::Invalid(e.to_string()))?;
        let lexical_choice = lexical
            .map(|head| head.commit(model, session.history(), session.last_scores_q48()))
            .transpose()?;
        let committed = lexical_choice.as_ref().map_or(id, |choice| choice.token);
        if let Some(st) = observation {
            let mut step_record = serde_json::json!({
                "token": st.token_id,
                "bucket": st.bucket,
                "highest_order": st.highest_order,
                "mass_q48": st.mass_q48,
                "top_ties": st.top_ties,
                "mass_sha256": st.mass_sha256,
                "prefix_sha256": st.prefix_sha256,
                "context_tokens": st.context_tokens,
                "pick_pos": st.pick_pos,
                "pick_distance": st.pick_distance,
                "pick_ties": st.pick_ties,
                "pick_successor": st.pick_successor,
                "pick_class": st.pick_class,
                "s3_pkey": st.s3_pkey,
                "s3_backoff_key": st.s3_backoff_key,
                "s3_mode": st.s3_mode,
                "s2_skip_keys": st.s2_skip_keys,
                "s2_skip_hit_mask": st.s2_skip_hit_mask,
                "cache_hit_state": st.cache_hit_state,
                "s4_role_scores": st.s4_role_scores,
                "s4_role_choice": st.s4_role_choice,
            });
            if let Some(source) = st.s4_source {
                step_record["s4_source"] = serde_json::json!({
                    "resolved_role": source.resolved_role,
                    "effective_role": source.effective_role,
                    "gate_index": source.gate_index,
                    "source_pos": source.source_pos,
                    "source_token": source.source_token,
                    "rival_token": source.rival_token,
                    "weight_i8": source.weight_i8,
                    "transfer_q48": source.transfer_q48,
                    "source_mass_before_q48": source.source_mass_before_q48,
                    "rival_mass_before_q48": source.rival_mass_before_q48,
                });
            }
            if let Some(choice) = &lexical_choice {
                step_record["prior_winner"] = serde_json::json!({
                    "token": st.token_id, "mass_q48": st.mass_q48, "top_ties": st.top_ties,
                });
                step_record["token"] = committed.into();
                step_record["mass_q48"] = choice.prior_q48.into();
                step_record["top_ties"] = choice.top_ties.into();
                step_record["lexical"] = serde_json::to_value(choice)?;
            }
            return Ok((committed, Some(step_record)));
        }
        Ok((committed, None))
    }
}

#[cfg(any(feature = "p1-count-serving", feature = "p4-only-client"))]
pub use count::{load_lexical, step};
