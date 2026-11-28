use crate::{input::Corpus, require, Result, SPLITS};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

pub struct Partition {
    pub parent: Vec<usize>,
    pub splits: Vec<usize>,
    pub audit: Value,
}

fn root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn join(corpus: &Corpus, parent: &mut [usize], left: usize, right: usize) {
    let left = root(parent, left);
    let right = root(parent, right);
    if corpus.records[left].id <= corpus.records[right].id {
        parent[right] = left;
    } else {
        parent[left] = right;
    }
}

pub fn normalize(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut pending_space = false;
    let mut digit_run = false;
    for character in raw.chars().flat_map(char::to_lowercase) {
        if character.is_whitespace() {
            pending_space = !output.is_empty();
            digit_run = false;
            continue;
        }
        if pending_space {
            output.push(' ');
            pending_space = false;
        }
        if character.is_ascii_digit() {
            if !digit_run {
                // U+0000 is reserved as the digit-run marker. Escape literal NUL.
                output.push('\0');
                output.push('1');
            }
            digit_run = true;
        } else {
            if character == '\0' {
                output.push('\0');
                output.push('0');
            } else {
                output.push(character);
            }
            digit_run = false;
        }
    }
    output
}

fn split(id: &str) -> usize {
    match Sha256::digest(id.as_bytes())[0] % 10 {
        0..=6 => 0,
        7 => 1,
        _ => 2,
    }
}

pub fn assign(corpus: &Corpus) -> Result<Partition> {
    let mut parent: Vec<_> = (0..corpus.records.len()).collect();
    let mut wording = HashMap::<String, usize>::new();
    let mut original = HashMap::<&str, usize>::new();
    let mut repeats = BTreeSet::new();
    let mut duplicate_original = BTreeSet::new();
    for turn in &corpus.turns {
        if let Some(earlier) = original.insert(&turn.user, turn.record) {
            if earlier != turn.record {
                duplicate_original.insert(turn.user.as_str());
                join(corpus, &mut parent, earlier, turn.record);
            }
        }
        let family = normalize(&turn.user);
        if let Some(earlier) = wording.insert(family.clone(), turn.record) {
            if earlier != turn.record {
                repeats.insert(family);
                join(corpus, &mut parent, earlier, turn.record);
            }
        }
    }
    for index in 0..parent.len() {
        parent[index] = root(&mut parent, index);
    }
    let splits: Vec<_> = parent
        .iter()
        .map(|&index| split(&corpus.records[index].id))
        .collect();
    let mut component_sizes = vec![0; parent.len()];
    for &key in &parent {
        component_sizes[key] += 1;
    }
    let mut cross_wording = BTreeSet::new();
    let mut cross_original = BTreeSet::new();
    let mut per_split = [[0usize; 3]; SPLITS.len()];
    for (index, &which) in splits.iter().enumerate() {
        per_split[which][0] += 1;
        per_split[which][1] += usize::from(component_sizes[index] > 0);
    }
    for turn in &corpus.turns {
        let which = splits[turn.record];
        per_split[which][2] += 1;
        let family = normalize(&turn.user);
        if splits[wording[&family]] != which {
            cross_wording.insert(family);
        }
        if splits[original[turn.user.as_str()]] != which {
            cross_original.insert(turn.user.as_str());
        }
    }
    require(
        cross_wording.is_empty() && cross_original.is_empty(),
        "registered family crosses splits",
    )?;
    let audit = json!({
        "rule": "SHA256(least original record identifier UTF-8)[0] % 10",
        "normalized_full_user_wordings": wording.len(),
        "repeated_normalized_wordings_across_records": repeats.len(),
        "duplicate_exact_user_sources_across_records": duplicate_original.len(),
        "normalized_wording_families_crossing_splits": cross_wording.len(),
        "exact_user_sources_crossing_splits": cross_original.len(),
        "largest_component_records": component_sizes.iter().copied().max().unwrap_or(0),
        "splits": std::array::from_fn::<_, 3, _>(|i| json!({
            "split": SPLITS[i],
            "records": per_split[i][0],
            "components": per_split[i][1],
            "turns": per_split[i][2],
        })),
    });
    Ok(Partition {
        parent,
        splits,
        audit,
    })
}

const SPAN: usize = 50;
const HASH_BASE: u64 = 0x100000001b3;

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
struct Span {
    hash: u64,
    turn: usize,
    start: usize,
}

pub fn answer_overlap(corpus: &Corpus, partition: &Partition) -> Value {
    let mut spans = Vec::new();
    let mut eligible_turns = [0; SPLITS.len()];
    let mut eligible_records: [BTreeSet<usize>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeSet::new());
    let power = HASH_BASE.wrapping_pow((SPAN - 1) as u32);
    for (index, turn) in corpus.turns.iter().enumerate() {
        if turn.answer.len() < SPAN {
            continue;
        }
        let which = partition.splits[turn.record];
        eligible_turns[which] += 1;
        eligible_records[which].insert(turn.record);
        let term = |id| u64::from(id) + 1;
        let mut hash = turn.answer[..SPAN].iter().fold(0u64, |value, &id| {
            value.wrapping_mul(HASH_BASE).wrapping_add(term(id))
        });
        spans.push(Span {
            hash,
            turn: index,
            start: 0,
        });
        for (start, window) in turn.answer.windows(SPAN + 1).enumerate() {
            hash = hash
                .wrapping_sub(term(window[0]).wrapping_mul(power))
                .wrapping_mul(HASH_BASE)
                .wrapping_add(term(window[SPAN]));
            spans.push(Span {
                hash,
                turn: index,
                start: start + 1,
            });
        }
    }
    spans.sort_unstable();
    let mut contaminated_turns = vec![false; corpus.turns.len()];
    let mut contaminated_records: [BTreeSet<usize>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeSet::new());
    let mut cross_split_spans = 0;
    let mut occurrence_count = 0;
    let mut exact = HashMap::<&[u32], u8>::new();
    for bucket in spans.chunk_by(|left, right| left.hash == right.hash) {
        if bucket.len() < 2 {
            continue;
        }
        exact.clear();
        for span in bucket {
            let turn = &corpus.turns[span.turn];
            let which = partition.splits[turn.record];
            let token_span = &turn.answer[span.start..span.start + SPAN];
            *exact.entry(token_span).or_default() |= 1u8 << which;
        }
        cross_split_spans += exact.values().filter(|mask| mask.count_ones() > 1).count();
        for span in bucket {
            let turn = &corpus.turns[span.turn];
            let token_span = &turn.answer[span.start..span.start + SPAN];
            if exact[token_span].count_ones() > 1 {
                occurrence_count += 1;
                contaminated_turns[span.turn] = true;
                contaminated_records[partition.splits[turn.record]].insert(turn.record);
            }
        }
    }
    let mut contaminated_turn_counts = [0; SPLITS.len()];
    for (turn, found) in corpus.turns.iter().zip(contaminated_turns) {
        if found {
            contaminated_turn_counts[partition.splits[turn.record]] += 1;
        }
    }
    json!({
        "minimum_consecutive_visible_answer_ids": SPAN,
        "positions_checked": spans.len(),
        "distinct_exact_spans_present_in_multiple_splits": cross_split_spans,
        "positions_in_exact_spans_present_in_multiple_splits": occurrence_count,
        "splits": std::array::from_fn::<_, 3, _>(|i| json!({
            "split": SPLITS[i],
            "eligible_records": eligible_records[i].len(),
            "records_with_cross_split_exact_overlap": contaminated_records[i].len(),
            "eligible_turns": eligible_turns[i],
            "turns_with_cross_split_exact_overlap": contaminated_turn_counts[i],
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{RecordKey, Turn};

    #[test]
    fn keeps_transitive_families_whole_and_counts_only_exact_long_answer_reuse() -> Result<()> {
        let base: Vec<_> = (300..350).collect();
        let make_turn = |record, ordinal, user: &str, answer: Vec<u32>| Turn {
            record,
            ordinal,
            user: user.to_string(),
            opening_offset: 0,
            opening: vec![],
            answer,
            answer_text: String::new(),
            eos_supervised: true,
        };
        let corpus = Corpus {
            records: ["c", "a", "b", "outside"]
                .into_iter()
                .enumerate()
                .map(|(ordinal, id)| RecordKey {
                    id: id.to_string(),
                    ordinal,
                })
                .collect(),
            turns: vec![
                make_turn(0, 0, "  FÖO 12 \t Hi ", base.clone()),
                make_turn(1, 0, "föo 341 hi", base[..49].to_vec()),
                make_turn(1, 1, "different turn", vec![]),
                make_turn(2, 0, "different turn", vec![]),
                make_turn(3, 0, "standalone", [vec![5], base].concat()),
            ],
            visible_tokens: 0,
            token_bytes: vec![],
        };
        let mut partition = assign(&corpus)?;
        assert_eq!(partition.parent, [1, 1, 1, 3]);
        assert_eq!(partition.splits[..3], [split("a"); 3]);
        assert_eq!(
            partition.audit["repeated_normalized_wordings_across_records"],
            2
        );
        assert_eq!(
            partition.audit["duplicate_exact_user_sources_across_records"],
            1
        );
        partition.splits = vec![0, 0, 0, 2];
        let overlap = answer_overlap(&corpus, &partition);
        assert_eq!(overlap["positions_checked"], 3);
        assert_eq!(
            overlap["distinct_exact_spans_present_in_multiple_splits"],
            1
        );
        assert_eq!(
            overlap["positions_in_exact_spans_present_in_multiple_splits"],
            2
        );
        Ok(())
    }
}
