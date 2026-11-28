//! Static preflight for the geometry and matched-ordered P4 lexical heads.
#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use uor_r4_integer::discrete_d11::native_occurrence::{
    lexical::{MATCHED_SLOT_MASK, SCORE_UNIT_Q48, STATE_BYTES, WEIGHT_WORDS},
    lexical_commit::{LexicalHead, BOUND_BYTES},
};

const STATE: &str = "368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778";
const BANK_SIZE: usize = u64::BITS as usize;
const BANKS: usize = WEIGHT_WORDS / BANK_SIZE;
const WORD_START: usize = 40;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Serialize)]
struct Range {
    min: i32,
    max: i32,
    lower_headroom: i32,
    upper_headroom: i32,
    minimum_headroom: i32,
}

impl Range {
    fn new(min: i32, max: i32) -> Self {
        let lower_headroom = min - i32::from(i16::MIN);
        let upper_headroom = i32::from(i16::MAX) - max;
        Self {
            min,
            max,
            lower_headroom,
            upper_headroom,
            minimum_headroom: lower_headroom.min(upper_headroom),
        }
    }
}

#[derive(Debug, Serialize)]
struct Bank {
    bank: usize,
    delta_i16_words: i32,
    before: Range,
    after: Range,
}

#[derive(Debug, Serialize)]
struct Word {
    index: usize,
    reachable: bool,
    before: i16,
    after: i16,
    delta: i32,
}

#[derive(Debug, Serialize)]
struct Report {
    state_sha256: &'static str,
    source_sha256: String,
    centered_sha256: String,
    addressing: &'static str,
    reachable_slots_per_bank: u32,
    score_shift_q48: i64,
    banks: Vec<Bank>,
    words: Vec<Word>,
}

fn reachable(index: usize) -> bool {
    MATCHED_SLOT_MASK & (1 << (index % BANK_SIZE)) != 0
}

fn decode_words(bytes: &[u8; BOUND_BYTES]) -> [i16; WEIGHT_WORDS] {
    let mut words = [0; WEIGHT_WORDS];
    for (word, pair) in words
        .iter_mut()
        .zip(bytes[WORD_START..32 + STATE_BYTES].chunks_exact(2))
    {
        *word = i16::from_le_bytes([pair[0], pair[1]]);
    }
    words
}

fn center(source: &[u8], expected_sha256: &str) -> Result<([u8; BOUND_BYTES], Report)> {
    let source: &[u8; BOUND_BYTES] = source.try_into()?;
    let source_sha256 = hex::encode(Sha256::digest(source));
    if source_sha256 != expected_sha256 {
        return Err("source SHA-256 differs".into());
    }
    let mut state = [0; 32];
    hex::decode_to_slice(STATE, &mut state)?;
    let original = LexicalHead::bind(source, &state)?;
    let encode = match &source[32..WORD_START] {
        b"UORLEX01" => {
            |state, words: &[i16; WEIGHT_WORDS]| LexicalHead::encode_bound(state, false, words)
        }
        b"UORLEX03" => LexicalHead::encode_matched_bound,
        _ => return Err("expected geometry or matched-ordered lexical head".into()),
    };
    if BOUND_BYTES != 584 || BANKS != 4 || MATCHED_SLOT_MASK.count_ones() != 26 {
        return Err("unexpected compiled lexical head layout".into());
    }
    let before_words = decode_words(source);
    if &encode(state, &before_words) != source {
        return Err("source cannot round-trip through the bound encoder".into());
    }

    let mut centered_words = before_words;
    let mut banks = Vec::with_capacity(BANKS);
    for bank in 0..BANKS {
        let indices = bank * BANK_SIZE..(bank + 1) * BANK_SIZE;
        let mut min = i32::from(i16::MAX);
        let mut max = i32::from(i16::MIN);
        for index in indices.clone().filter(|&index| reachable(index)) {
            let word = i32::from(before_words[index]);
            min = min.min(word);
            max = max.max(word);
        }
        let delta = (-1 - max - min) / 2;
        for index in indices.filter(|&index| reachable(index)) {
            centered_words[index] = i16::try_from(i32::from(before_words[index]) + delta)?;
        }
        let after = Range::new(min + delta, max + delta);
        if (-1 - after.max - after.min) / 2 != 0 {
            return Err("centered midpoint is not idempotent".into());
        }
        banks.push(Bank {
            bank,
            delta_i16_words: delta,
            before: Range::new(min, max),
            after,
        });
    }
    let centered = encode(state, &centered_words);
    let rebound = LexicalHead::bind(&centered, &state)?;
    let addressing = original.addressing_name();
    if rebound.addressing_name() != addressing || decode_words(&centered) != centered_words {
        return Err("rebound head differs from the centered state".into());
    }
    let words = verify_deltas(&before_words, &centered_words, &banks)?;
    // Both operators inspect exactly one masked slot in each bank. Every legal
    // candidate therefore receives this same additive shift, with no i64 overflow:
    // the original and centered scores both lie between -2^49 and 3*2^48.
    let shift_words: i64 = banks
        .iter()
        .map(|bank| i64::from(bank.delta_i16_words))
        .sum();
    let score_shift_q48 = shift_words
        .checked_mul(SCORE_UNIT_Q48)
        .ok_or("Q48 score shift overflow")?;
    let report = Report {
        state_sha256: STATE,
        source_sha256,
        centered_sha256: hex::encode(Sha256::digest(centered)),
        addressing,
        reachable_slots_per_bank: MATCHED_SLOT_MASK.count_ones(),
        score_shift_q48,
        banks,
        words,
    };
    Ok((centered, report))
}

fn verify_deltas(
    before: &[i16; WEIGHT_WORDS],
    after: &[i16; WEIGHT_WORDS],
    banks: &[Bank],
) -> Result<Vec<Word>> {
    let mut words = Vec::with_capacity(WEIGHT_WORDS);
    for (index, (&before, &after)) in before.iter().zip(after).enumerate() {
        let delta = i32::from(after) - i32::from(before);
        let reachable = reachable(index);
        if reachable {
            if delta != banks[index / BANK_SIZE].delta_i16_words {
                return Err(format!("nonuniform delta at reachable word {index}").into());
            }
        } else if before != 0 || after != 0 {
            return Err(format!("nonzero unreachable word {index}").into());
        }
        words.push(Word {
            index,
            reachable,
            before,
            after,
            delta,
        });
    }
    Ok(words)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [source, expected_sha256, destination] = args.as_slice() else {
        return Err(
            "usage: uor-lexical-bank-recenter SOURCE EXPECTED_SHA256 NEW_ATTACHMENT".into(),
        );
    };
    let expected_sha256 = expected_sha256.to_str().ok_or("invalid SHA-256 argument")?;
    let (centered, report) = center(&fs::read(source)?, expected_sha256)?;
    let json = serde_json::to_vec(&report)?;
    write_new(&PathBuf::from(destination), &centered)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&json)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_r4_integer::discrete_d11::native_occurrence::lexical::{Carry, OrderedPrefix, Weights};

    fn state() -> [u8; 32] {
        let mut state = [0; 32];
        hex::decode_to_slice(STATE, &mut state).unwrap();
        state
    }

    fn sha(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn geometry_weights(words: &[i16; WEIGHT_WORDS]) -> Weights {
        let encoded = LexicalHead::encode_bound(state(), false, words);
        Weights::from_le_bytes(&encoded[32..32 + STATE_BYTES]).unwrap()
    }

    #[test]
    fn both_address_operators_preserve_every_id_rank_and_tie() {
        let carry = Carry::from_packed(0xc952);
        let ordered = OrderedPrefix::new(&[55, 57, 4095]).unwrap();
        for matched in [false, true] {
            let sign = if matched { -1 } else { 1 };
            let words = std::array::from_fn(|index| {
                if reachable(index) {
                    sign * (100 + (index % BANK_SIZE) as i16)
                } else {
                    0
                }
            });
            let source = if matched {
                LexicalHead::encode_matched_bound(state(), &words)
            } else {
                LexicalHead::encode_bound(state(), false, &words)
            };
            let (output, report) = center(&source, &sha(&source)).unwrap();
            assert_eq!(report.score_shift_q48.signum(), -i64::from(sign));
            let before = geometry_weights(&words);
            let after = geometry_weights(&decode_words(&output));
            let mut ranked_before = Vec::new();
            let mut ranked_after = Vec::new();
            let mut aliases = std::collections::HashMap::new();
            let mut equal_aliases = 0;
            for id in 1..4096u16 {
                let addresses = if matched {
                    ordered.matched_addresses(id).unwrap()
                } else {
                    let root = (id % 16) as u8;
                    carry.addresses([root, root, root, root]).unwrap()
                };
                let prior = match id {
                    4094 => 1 << 48,
                    4095 => 1,
                    _ => 0,
                };
                let before = before.score_addresses_q48(addresses, prior).unwrap();
                let after = after.score_addresses_q48(addresses, prior).unwrap();
                assert_eq!(after - before, report.score_shift_q48);
                if let Some((other_id, other_score)) =
                    aliases.insert((addresses, prior), (id, before))
                {
                    assert_ne!(id, other_id);
                    assert_eq!(before, other_score);
                    equal_aliases += 1;
                }
                ranked_before.push((std::cmp::Reverse(before), id));
                ranked_after.push((std::cmp::Reverse(after), id));
            }
            assert!(equal_aliases > 0);
            ranked_before.sort();
            ranked_after.sort();
            assert!(ranked_before.windows(2).any(|pair| pair[0].0 == pair[1].0));
            for (before, after) in ranked_before.iter().zip(&ranked_after) {
                assert_eq!(before.1, after.1);
            }
        }
    }

    #[test]
    fn i16_extrema_keep_width_and_center_idempotently() {
        let mut words = std::array::from_fn(|index| {
            if !reachable(index) {
                0
            } else if index / BANK_SIZE < 2 {
                i16::MIN
            } else {
                i16::MAX
            }
        });
        let first_slot = MATCHED_SLOT_MASK.trailing_zeros() as usize;
        words[first_slot] = i16::MAX;
        let source = LexicalHead::encode_bound(state(), false, &words);
        let (output, report) = center(&source, &sha(&source)).unwrap();
        assert_eq!(report.banks[0].delta_i16_words, 0);
        assert_eq!(report.banks[1].delta_i16_words, 32_767);
        assert_eq!(report.banks[2].delta_i16_words, -32_767);
        assert_eq!(report.banks[1].after.min, -1);
        assert_eq!(report.banks[2].after.max, 0);
        for bank in &report.banks {
            assert_eq!(
                bank.before.max - bank.before.min,
                bank.after.max - bank.after.min
            );
            assert!(bank.after.minimum_headroom >= bank.before.minimum_headroom);
        }
        let (second_output, second_report) = center(&output, &sha(&output)).unwrap();
        assert_eq!(second_output, output);
        assert_eq!(second_report.score_shift_q48, 0);
        assert!(second_report.words.iter().all(|word| word.delta == 0));
    }

    #[test]
    fn rejects_broken_identity_inactive_words_and_nonuniform_delta() {
        let before = [0; WEIGHT_WORDS];
        let mut source = LexicalHead::encode_bound(state(), false, &before);
        assert!(center(&source, "bad").is_err());
        assert!(center(&source[..583], "bad").is_err());
        source[WORD_START] ^= 1;
        assert!(center(&source, &sha(&source)).is_err());
        let wrong_state = LexicalHead::encode_bound([1; 32], false, &before);
        assert!(center(&wrong_state, &sha(&wrong_state)).is_err());
        let unmatched = LexicalHead::encode_bound(state(), true, &before);
        assert!(center(&unmatched, &sha(&unmatched)).is_err());

        let inactive = (!MATCHED_SLOT_MASK).trailing_zeros() as usize;
        let mut words = before;
        words[inactive] = 1;
        let source = LexicalHead::encode_bound(state(), false, &words);
        assert!(center(&source, &sha(&source))
            .unwrap_err()
            .to_string()
            .contains("unreachable"));

        let source = LexicalHead::encode_bound(state(), false, &before);
        let (_, report) = center(&source, &sha(&source)).unwrap();
        let active = MATCHED_SLOT_MASK.trailing_zeros() as usize;
        words = before;
        words[active] = 1;
        assert!(verify_deltas(&before, &words, &report.banks)
            .unwrap_err()
            .to_string()
            .contains("nonuniform"));
    }
}
