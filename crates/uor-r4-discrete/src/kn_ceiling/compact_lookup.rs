//! Resident raw and KN lookup over a common, query-independent key index.
//! Each order has fixed-width context and follower rows so serving can borrow
//! the artifact without a second expanded copy.

use super::{
    complete_lookup::{validate_records, Order},
    invalid, FrozenPickMixture, PickTarget,
};
use crate::{
    kn_score::{interpolate_order, KnDiscounts, KN_BASE_MASS},
    ngram_count::{
        partition_from_triples, NgramAssembler, NgramContext, NgramOrderDigest, NgramTriple,
        PrimeRouteKey, NGRAM_BOS, NGRAM_MAX_ORDER, NGRAM_VOCABULARY,
    },
    Result,
};
use sha2::{Digest, Sha256};

include!("runtime/compact.rs");
include!("runtime/rank.rs");

/// Pack full canonical raw and adjusted tables from the same count stream.
/// Their key/next support is the same because the adjusted assembler refuses
/// zero-mass continuations. The highest order also shares count values.
pub fn encode_compact_lookup(
    highest: u32,
    training: [u8; 32],
    raw: &[Order],
    adjusted: &[Order],
) -> Result<Vec<u8>> {
    if highest > NGRAM_MAX_ORDER || raw.len() != highest as usize + 1 || raw.len() != adjusted.len()
    {
        return Err(invalid("invalid compact lookup order layout"));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&highest.to_le_bytes());
    bytes.extend_from_slice(&training);
    for (order, (raw_order, adjusted_order)) in (0..=highest).zip(raw.iter().zip(adjusted)) {
        let (raw_digest, raw_rows, raw_contexts) = raw_order;
        let (adjusted_digest, adjusted_rows, adjusted_contexts) = adjusted_order;
        let shared = order == highest;
        for (digest, rows, contexts) in [raw_order, adjusted_order] {
            if digest.order != order
                || digest.grams != rows.len() as u64
                || digest.contexts != contexts.len() as u64
            {
                return Err(invalid("incomplete or reordered compact count assembly"));
            }
            validate_records(rows, contexts, digest.count_of_counts)?;
        }
        if raw_contexts.len() != adjusted_contexts.len()
            || raw_rows.len() != adjusted_rows.len()
            || raw_rows.iter().zip(adjusted_rows).any(|(raw, kn)| {
                (raw.key, raw.next) != (kn.key, kn.next) || (shared && raw.count != kn.count)
            })
        {
            return Err(invalid(
                "raw and adjusted count support or shared values differ",
            ));
        }
        bytes.extend_from_slice(&as_u32(raw_rows.len())?.to_le_bytes());
        bytes.extend_from_slice(&as_u32(raw_contexts.len())?.to_le_bytes());
        for spectrum in [raw_digest.count_of_counts, adjusted_digest.count_of_counts] {
            for word in spectrum {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        let mut row_start = 0u32;
        for context in raw_contexts {
            if PrimeRouteKey::from_word(context.key)?.order() != order {
                return Err(invalid("compact context key has the wrong order"));
            }
            bytes.extend_from_slice(&key_body(context.key, order)[16 - key_width(order)..]);
            bytes.extend_from_slice(&row_start.to_le_bytes());
            row_start += context.followers;
        }
        for (raw, kn) in raw_rows.iter().zip(adjusted_rows) {
            bytes.extend_from_slice(&raw.next.to_le_bytes());
            bytes.extend_from_slice(&raw.count.to_le_bytes());
            if !shared {
                bytes.extend_from_slice(&kn.count.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

fn inline_code(shared: bool, raw: u32, kn: u32) -> Option<u8> {
    if shared {
        (1..=15).contains(&raw).then_some((raw - 1) as u8)
    } else {
        INLINE_PAIRS
            .iter()
            .position(|&(r, k)| r == raw && k == kn)
            .map(|idx| idx as u8)
    }
}

fn push_deg_esc(esc: &mut Vec<u8>, key: u32, deg: u16) {
    esc.extend_from_slice(&key.to_le_bytes());
    esc.extend_from_slice(&deg.to_le_bytes());
}

/// Pack suffix-closed canonical raw and adjusted tables into `UORKNC03`
/// reversed-trie lossless format.
pub fn encode_compact_lookup_v3(
    highest: u32,
    training: [u8; 32],
    raw: &[Order],
    adjusted: &[Order],
) -> Result<Vec<u8>> {
    if highest > NGRAM_MAX_ORDER || raw.len() != highest as usize + 1 || raw.len() != adjusted.len()
    {
        return Err(invalid("invalid compact lookup order layout"));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC_V3);
    bytes.extend_from_slice(&highest.to_le_bytes());
    bytes.extend_from_slice(&training);
    let mut prev_contexts: Vec<u128> = Vec::new();
    for (order_idx, (raw_order, adjusted_order)) in (0..=highest).zip(raw.iter().zip(adjusted)) {
        let (raw_digest, raw_rows, raw_contexts) = raw_order;
        let (adjusted_digest, adjusted_rows, adjusted_contexts) = adjusted_order;
        let shared = order_idx == highest;
        for (digest, rows, contexts) in [raw_order, adjusted_order] {
            if digest.order != order_idx
                || digest.grams != rows.len() as u64
                || digest.contexts != contexts.len() as u64
            {
                return Err(invalid("incomplete or reordered compact count assembly"));
            }
            validate_records(rows, contexts, digest.count_of_counts)?;
        }
        if raw_contexts.len() != adjusted_contexts.len()
            || raw_rows.len() != adjusted_rows.len()
            || raw_rows.iter().zip(adjusted_rows).any(|(raw, kn)| {
                (raw.key, raw.next) != (kn.key, kn.next) || (shared && raw.count != kn.count)
            })
        {
            return Err(invalid(
                "raw and adjusted count support or shared values differ",
            ));
        }

        let parent_contexts = prev_contexts.len();
        let mut parent_sb = Vec::new();
        let mut parent_deg = Vec::new();
        let mut parent_deg_esc = Vec::new();
        let mut child_toks = Vec::new();
        if order_idx == 0 {
            if raw_contexts.len() > 1 {
                return Err(invalid("order 0 cannot have more than 1 context"));
            }
            if let Some(c0) = raw_contexts.first() {
                if PrimeRouteKey::from_word(c0.key)?.order() != 0 {
                    return Err(invalid("compact context key has the wrong order"));
                }
            }
        } else {
            let mut child_idx = 0usize;
            let mut next_prev_keys = Vec::with_capacity(raw_contexts.len());
            for (p_idx, &p_key) in prev_contexts.iter().enumerate() {
                if p_idx % 64 == 0 {
                    parent_sb.extend_from_slice(&as_u32(child_idx)?.to_le_bytes());
                }
                let start = child_idx;
                while child_idx < raw_contexts.len() {
                    let parsed = PrimeRouteKey::from_word(raw_contexts[child_idx].key)?;
                    if parsed.order() != order_idx {
                        return Err(invalid("compact context key has the wrong order"));
                    }
                    let backoff = parsed
                        .back_off()
                        .ok_or_else(|| invalid("missing backoff key"))?
                        .word();
                    if backoff != p_key {
                        break;
                    }
                    next_prev_keys.push(parsed.word());
                    child_idx += 1;
                }
                let deg = child_idx - start;
                if deg < 255 {
                    parent_deg.push(deg as u8);
                } else {
                    parent_deg.push(255);
                    let deg_u16: u16 = deg
                        .try_into()
                        .map_err(|_| invalid("parent degree overflow"))?;
                    push_deg_esc(&mut parent_deg_esc, as_u32(p_idx)?, deg_u16);
                }
            }
            if child_idx != raw_contexts.len() {
                return Err(invalid(
                    "encode_compact_lookup_v3 requires suffix-closed contexts",
                ));
            }
            parent_sb.extend_from_slice(&as_u32(child_idx)?.to_le_bytes());
            child_toks.resize(
                raw_contexts
                    .len()
                    .checked_mul(3)
                    .ok_or_else(|| invalid("overflow"))?
                    .div_ceil(2),
                0,
            );
            for (idx, ctx) in raw_contexts.iter().enumerate() {
                let tok = PrimeRouteKey::from_word(ctx.key)?
                    .slot(order_idx)
                    .ok_or_else(|| invalid("missing context slot"))?;
                let base = (idx / 2) * 3;
                if idx % 2 == 0 {
                    child_toks[base] = (tok & 0xff) as u8;
                    child_toks[base + 1] =
                        (child_toks[base + 1] & 0xf0) | ((tok >> 8) as u8 & 0x0f);
                } else {
                    child_toks[base + 1] =
                        (child_toks[base + 1] & 0x0f) | (((tok & 0x0f) as u8) << 4);
                    child_toks[base + 2] = (tok >> 4) as u8;
                }
            }
            prev_contexts = next_prev_keys;
        }
        if order_idx == 0 {
            prev_contexts = raw_contexts.iter().map(|c| c.key).collect();
        }
        let _ = parent_contexts;

        let mut fol_sb = Vec::with_capacity((raw_contexts.len().div_ceil(64) + 1) * 4);
        let mut fol_deg = Vec::with_capacity(raw_contexts.len());
        let mut fol_deg_esc = Vec::new();
        let mut row_off = 0usize;
        for (c_idx, ctx) in raw_contexts.iter().enumerate() {
            if c_idx % 64 == 0 {
                fol_sb.extend_from_slice(&as_u32(row_off)?.to_le_bytes());
            }
            let deg = ctx.followers as usize;
            if deg == 0 || deg >= NGRAM_VOCABULARY as usize {
                return Err(invalid("v3 context follower count out of range"));
            }
            if deg <= 255 {
                fol_deg.push((deg - 1) as u8);
            } else {
                fol_deg.push(255);
                push_deg_esc(&mut fol_deg_esc, as_u32(c_idx)?, deg as u16);
            }
            row_off += deg;
        }
        fol_sb.extend_from_slice(&as_u32(row_off)?.to_le_bytes());

        let mut fol_words = Vec::with_capacity(raw_rows.len() * 2);
        let mut esc_sb = Vec::with_capacity((raw_rows.len().div_ceil(64) + 1) * 4);
        let mut esc16 = Vec::new();
        let mut esc32 = Vec::new();
        let mut esc16_len = 0u32;
        let mut esc32_len = 0u32;
        for (f_idx, (raw_r, kn_r)) in raw_rows.iter().zip(adjusted_rows).enumerate() {
            if f_idx % 64 == 0 {
                esc_sb.extend_from_slice(&esc16_len.to_le_bytes());
            }
            let code = if let Some(c) = inline_code(shared, raw_r.count, kn_r.count) {
                c
            } else {
                esc16_len = esc16_len
                    .checked_add(1)
                    .ok_or_else(|| invalid("v3 esc16 count overflow"))?;
                if raw_r.count <= 65535 && kn_r.count <= 65535 {
                    esc16.extend_from_slice(&(raw_r.count as u16).to_le_bytes());
                    if !shared {
                        esc16.extend_from_slice(&(kn_r.count as u16).to_le_bytes());
                    }
                } else {
                    esc16.extend_from_slice(&0u16.to_le_bytes());
                    if !shared {
                        esc16.extend_from_slice(&0u16.to_le_bytes());
                    }
                    esc32.extend_from_slice(&as_u32(f_idx)?.to_le_bytes());
                    esc32.extend_from_slice(&raw_r.count.to_le_bytes());
                    if !shared {
                        esc32.extend_from_slice(&kn_r.count.to_le_bytes());
                    }
                    esc32_len += 1;
                }
                15
            };
            let word = (raw_r.next << 4) | u16::from(code);
            fol_words.extend_from_slice(&word.to_le_bytes());
        }
        esc_sb.extend_from_slice(&esc16_len.to_le_bytes());

        bytes.extend_from_slice(&as_u32(raw_rows.len())?.to_le_bytes());
        bytes.extend_from_slice(&as_u32(raw_contexts.len())?.to_le_bytes());
        for spectrum in [raw_digest.count_of_counts, adjusted_digest.count_of_counts] {
            for w in spectrum {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&as_u32(parent_deg_esc.len() / 6)?.to_le_bytes());
        bytes.extend_from_slice(&as_u32(fol_deg_esc.len() / 6)?.to_le_bytes());
        bytes.extend_from_slice(&esc16_len.to_le_bytes());
        bytes.extend_from_slice(&esc32_len.to_le_bytes());
        if order_idx >= 1 {
            bytes.extend_from_slice(&parent_sb);
            bytes.extend_from_slice(&parent_deg);
            bytes.extend_from_slice(&parent_deg_esc);
            bytes.extend_from_slice(&child_toks);
        }
        bytes.extend_from_slice(&fol_sb);
        bytes.extend_from_slice(&fol_deg);
        bytes.extend_from_slice(&fol_deg_esc);
        bytes.extend_from_slice(&fol_words);
        bytes.extend_from_slice(&esc_sb);
        bytes.extend_from_slice(&esc16);
        bytes.extend_from_slice(&esc32);
    }
    Ok(bytes)
}

fn as_u32(len: usize) -> Result<u32> {
    len.try_into()
        .map_err(|_| invalid("compact table exceeds u32 rows"))
}

impl CompactCountLookup<'_> {
    /// Audit the reloaded payload against the original raw and adjusted
    /// canonical count streams, reconstructing complete contexts in bounded
    /// batches. This is a build-time check and is outside the lookup path.
    pub fn canonical_digests(&self) -> Result<Vec<[NgramOrderDigest; 2]>> {
        const BATCH: usize = 64 * 1024;
        let order_count = self.order_count();
        let mut assemblers = (0..order_count)
            .map(|k| {
                let order = k as u32;
                Ok((
                    NgramAssembler::new(order, false)?,
                    NgramAssembler::new_adjusted(order, false)?,
                    Vec::new(),
                    Vec::new(),
                    None::<u128>,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut flush_err: Option<crate::Error> = None;
        self.visit_counts(|order, key, next, raw_count, adjusted_count| {
            if flush_err.is_some() {
                return;
            }
            let (raw_asm, adj_asm, raw_rows, adjusted_rows, last_key) =
                &mut assemblers[order as usize];
            if last_key.is_some_and(|prior| prior != key) && raw_rows.len() >= BATCH {
                if let Err(e) = raw_asm
                    .push_owned(partition_from_triples(std::mem::take(raw_rows)))
                    .and_then(|()| {
                        adj_asm.push_owned(partition_from_triples(std::mem::take(adjusted_rows)))
                    })
                {
                    flush_err = Some(e);
                    return;
                }
            }
            *last_key = Some(key);
            raw_rows.push(NgramTriple {
                key,
                next,
                count: raw_count,
            });
            adjusted_rows.push(NgramTriple {
                key,
                next,
                count: adjusted_count,
            });
        })?;
        if let Some(e) = flush_err {
            return Err(e);
        }
        let mut digests = Vec::with_capacity(order_count);
        for (mut raw, mut adjusted, raw_rows, adjusted_rows, _) in assemblers {
            if !raw_rows.is_empty() {
                raw.push_owned(partition_from_triples(raw_rows))?;
            }
            if !adjusted_rows.is_empty() {
                adjusted.push_owned(partition_from_triples(adjusted_rows))?;
            }
            let (raw_d, _, _) = raw.finish();
            let (adj_d, _, _) = adjusted.finish();
            digests.push([raw_d, adj_d]);
        }
        Ok(digests)
    }
}

#[cfg(test)]
mod ranking_tests;

#[path = "compact_lookup/replay.rs"]
mod replay;
