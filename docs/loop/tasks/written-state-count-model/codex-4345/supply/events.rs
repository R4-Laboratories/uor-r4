use crate::{groups::Partition, input::Corpus, require, Result, EOS, SPLITS, WINDOW};
use regex_syntax::{
    hir::{Class, ClassUnicode, HirKind},
    Parser,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Write,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum CharClass {
    SentenceEnd,
    OtherPunctuation,
    Whitespace,
    Alphanumeric,
    Other,
}

fn punctuation_class() -> Result<ClassUnicode> {
    let hir = Parser::new().parse(r"\p{Punctuation}")?;
    match hir.kind() {
        HirKind::Class(Class::Unicode(class)) => Ok(class.clone()),
        _ => Err("Unicode punctuation class unexpectedly unavailable".into()),
    }
}

fn character_class(value: char, punctuation: &ClassUnicode) -> CharClass {
    if matches!(value, '.' | '!' | '?') {
        CharClass::SentenceEnd
    } else if value.is_whitespace() {
        CharClass::Whitespace
    } else if value.is_alphanumeric() {
        CharClass::Alphanumeric
    } else {
        let ranges = punctuation.ranges();
        let index = ranges.partition_point(|range| range.end() < value);
        if ranges
            .get(index)
            .is_some_and(|range| range.start() <= value)
        {
            CharClass::OtherPunctuation
        } else {
            CharClass::Other
        }
    }
}

struct TokenShape {
    after_class: Option<CharClass>,
    completed_alnum: bool,
    split_alnum: bool,
    whitespace_only: bool,
    punctuation_or_space_only: bool,
    ends_mid_char: bool,
}

fn shapes(
    text: &str,
    ids: &[u32],
    token_bytes: &[Vec<u8>],
    punctuation: &ClassUnicode,
) -> Result<Vec<TokenShape>> {
    let mut offset = 0usize;
    let mut chars = text.char_indices().peekable();
    let mut result = Vec::with_capacity(ids.len());
    let mut after_class = None;
    for &id in ids {
        let bytes = token_bytes
            .get(id as usize)
            .ok_or("missing original token bytes")?;
        let end = offset + bytes.len();
        require(
            text.as_bytes().get(offset..end) == Some(bytes),
            "answer token bytes do not reconstruct text",
        )?;
        let mut completed_alnum = false;
        let mut split_alnum = false;
        while let Some(&(start, character)) = chars.peek() {
            if start + character.len_utf8() > end {
                break;
            }
            let class = character_class(character, punctuation);
            if class == CharClass::Alphanumeric {
                completed_alnum = true;
                split_alnum |= start < offset;
            }
            after_class = Some(class);
            chars.next();
        }
        let complete_token_text = text.get(offset..end);
        let whitespace_only = complete_token_text
            .is_some_and(|text| !text.is_empty() && text.chars().all(char::is_whitespace));
        let punctuation_or_space_only = complete_token_text.is_some_and(|text| {
            !text.is_empty()
                && text.chars().all(|character| {
                    matches!(
                        character_class(character, punctuation),
                        CharClass::SentenceEnd
                            | CharClass::OtherPunctuation
                            | CharClass::Whitespace
                    )
                })
        });
        result.push(TokenShape {
            after_class,
            completed_alnum,
            split_alnum,
            whitespace_only,
            punctuation_or_space_only,
            ends_mid_char: !text.is_char_boundary(end),
        });
        offset = end;
    }
    require(
        offset == text.len() && chars.next().is_none(),
        "answer token byte span is incomplete",
    )?;
    Ok(result)
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct Stratum {
    completed_answer_ids: usize,
    opening_length_bin: usize,
    has_history: bool,
    last_char_class: CharClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Stop,
    Continue,
}

#[derive(Clone, Copy, Debug)]
struct Event {
    turn: usize,
    stratum: Stratum,
    target: u32,
    prefix_hash: u64,
}

impl Event {
    fn action(self) -> Action {
        if self.target == EOS {
            Action::Stop
        } else {
            Action::Continue
        }
    }

    fn prefix(self, corpus: &Corpus) -> impl Iterator<Item = u32> + '_ {
        let turn = &corpus.turns[self.turn];
        turn.opening
            .iter()
            .chain(&turn.answer[..self.stratum.completed_answer_ids])
            .copied()
    }

    fn record(self, corpus: &Corpus) -> usize {
        corpus.turns[self.turn].record
    }
}

#[derive(Default, Serialize)]
struct Counts {
    turns: usize,
    turns_with_genuine_history: usize,
    opening_fits: usize,
    opening_with_next_id_fits: usize,
    visible_answer_ids: usize,
    internal_ids_with_target_room: usize,
    internal_ids_without_target_room: usize,
    internal_ids_with_no_earlier_alnum: usize,
    internal_ids_whitespace_only: usize,
    internal_ids_punctuation_or_space_only: usize,
    internal_ids_other_without_completed_alnum: usize,
    internal_non_alnum_ids_followed_by_later_alnum_within_budget: usize,
    internal_non_alnum_ids_followed_by_later_alnum_beyond_budget: usize,
    internal_non_alnum_ids_with_no_later_alnum: usize,
    internal_ids_ending_mid_character: usize,
    internal_ids_completing_split_alnum: usize,
    continuation_candidates_completing_split_alnum: usize,
    supervised_eos: usize,
    supervised_eos_with_target_room: usize,
    supervised_eos_without_target_room: usize,
    supervised_eos_with_room_but_no_earlier_alnum: usize,
    history_eos_mask_zero: usize,
    history_eos_mask_zero_with_target_room: usize,
    history_eos_mask_zero_without_target_room: usize,
    stop_candidates: usize,
    continue_candidates: usize,
    turns_with_stop_candidate: usize,
    turns_with_continue_candidate: usize,
    removed_ambiguous_stops: usize,
    removed_ambiguous_continues: usize,
    eligible_stops_after_ambiguity: usize,
    eligible_continues_after_ambiguity: usize,
    further_unique_stop_records: usize,
    further_unique_continuation_records: usize,
    matched_pairs: usize,
    matched_stop_components: usize,
    matched_continue_components: usize,
    matched_distinct_components: usize,
}

fn hash_append(state: u64, id: u32) -> u64 {
    (state ^ u64::from(id)).wrapping_mul(0x100000001b3)
}

fn population(corpus: &Corpus, partition: &Partition, counts: &mut [Counts]) -> Result<Vec<Event>> {
    let punctuation = punctuation_class()?;
    let mut events = Vec::new();
    for (index, turn) in corpus.turns.iter().enumerate() {
        let which = &mut counts[partition.splits[turn.record]];
        let opening_len = turn.opening.len();
        let has_history = turn.ordinal > 0;
        let eos_fits = opening_len + turn.answer.len() < WINDOW;
        which.turns += 1;
        which.turns_with_genuine_history += usize::from(has_history);
        which.opening_fits += usize::from(opening_len <= WINDOW);
        which.opening_with_next_id_fits += usize::from(opening_len < WINDOW);
        which.visible_answer_ids += turn.answer.len();
        if turn.eos_supervised {
            which.supervised_eos += 1;
            which.supervised_eos_with_target_room += usize::from(eos_fits);
            which.supervised_eos_without_target_room += usize::from(!eos_fits);
        } else {
            which.history_eos_mask_zero += 1;
            which.history_eos_mask_zero_with_target_room += usize::from(eos_fits);
            which.history_eos_mask_zero_without_target_room += usize::from(!eos_fits);
        }
        let room = WINDOW.saturating_sub(opening_len).min(turn.answer.len());
        which.internal_ids_with_target_room += room;
        which.internal_ids_without_target_room += turn.answer.len() - room;
        if room == 0 {
            continue;
        }
        let shapes = shapes(
            &turn.answer_text,
            &turn.answer,
            &corpus.token_bytes,
            &punctuation,
        )?;
        let mut earlier_alnum = false;
        let mut last_char_class = None;
        let mut prefix_hash = turn
            .opening
            .iter()
            .copied()
            .fold(0xcbf29ce484222325, hash_append);
        let mut admitted_continue = false;
        let last_lexical_with_room = shapes[..room]
            .iter()
            .rposition(|shape| shape.completed_alnum);
        let last_lexical = shapes.iter().rposition(|shape| shape.completed_alnum);
        for (position, (&target, shape)) in turn.answer.iter().zip(&shapes).take(room).enumerate() {
            which.internal_ids_ending_mid_character += usize::from(shape.ends_mid_char);
            which.internal_ids_completing_split_alnum += usize::from(shape.split_alnum);
            if !earlier_alnum {
                which.internal_ids_with_no_earlier_alnum += 1;
            } else if shape.completed_alnum {
                events.push(Event {
                    turn: index,
                    stratum: Stratum {
                        completed_answer_ids: position,
                        opening_length_bin: opening_len / 8,
                        has_history,
                        last_char_class: last_char_class
                            .ok_or("missing previously visible character")?,
                    },
                    target,
                    prefix_hash,
                });
                which.continue_candidates += 1;
                which.continuation_candidates_completing_split_alnum +=
                    usize::from(shape.split_alnum);
                admitted_continue = true;
            } else {
                if shape.whitespace_only {
                    which.internal_ids_whitespace_only += 1;
                } else if shape.punctuation_or_space_only {
                    which.internal_ids_punctuation_or_space_only += 1;
                } else {
                    which.internal_ids_other_without_completed_alnum += 1;
                }
                if last_lexical_with_room.is_some_and(|last| last > position) {
                    which.internal_non_alnum_ids_followed_by_later_alnum_within_budget += 1;
                } else if last_lexical.is_some_and(|last| last > position) {
                    which.internal_non_alnum_ids_followed_by_later_alnum_beyond_budget += 1;
                } else {
                    which.internal_non_alnum_ids_with_no_later_alnum += 1;
                }
            }
            earlier_alnum |= shape.completed_alnum;
            last_char_class = shape.after_class;
            prefix_hash = hash_append(prefix_hash, target);
        }
        which.turns_with_continue_candidate += usize::from(admitted_continue);
        if turn.eos_supervised && eos_fits {
            if earlier_alnum {
                events.push(Event {
                    turn: index,
                    stratum: Stratum {
                        completed_answer_ids: turn.answer.len(),
                        opening_length_bin: opening_len / 8,
                        has_history,
                        last_char_class: last_char_class
                            .ok_or("missing terminal visible character")?,
                    },
                    target: EOS,
                    prefix_hash,
                });
                which.stop_candidates += 1;
                which.turns_with_stop_candidate += 1;
            } else {
                which.supervised_eos_with_room_but_no_earlier_alnum += 1;
            }
        }
    }
    Ok(events)
}

fn unregister_conflicts(
    corpus: &Corpus,
    partition: &Partition,
    events: &mut Vec<Event>,
    counts: &mut [Counts],
) -> Value {
    events.sort_by_key(|event| event.prefix_hash);
    let mut exact = HashMap::<Vec<u32>, (u8, u8)>::new();
    let mut duplicate_prefixes = 0;
    let mut opposing_actions = 0;
    let mut cross_split_duplicates = 0;
    let mut kept = Vec::with_capacity(events.len());
    for bucket in events.chunk_by(|left, right| left.prefix_hash == right.prefix_hash) {
        if bucket.len() == 1 {
            kept.push(bucket[0]);
            continue;
        }
        exact.clear();
        for &event in bucket {
            let entry = exact.entry(event.prefix(corpus).collect()).or_default();
            entry.0 |= if event.action() == Action::Stop { 1 } else { 2 };
            entry.1 |= 1u8 << partition.splits[event.record(corpus)];
        }
        duplicate_prefixes += bucket.len() - exact.len();
        opposing_actions += exact.values().filter(|(actions, _)| *actions == 3).count();
        cross_split_duplicates += exact
            .values()
            .filter(|(_, splits)| splits.count_ones() > 1)
            .count();
        for &event in bucket {
            let prefix: Vec<_> = event.prefix(corpus).collect();
            let (actions, _) = exact[&prefix];
            if actions == 3 {
                let which = &mut counts[partition.splits[event.record(corpus)]];
                match event.action() {
                    Action::Stop => which.removed_ambiguous_stops += 1,
                    Action::Continue => which.removed_ambiguous_continues += 1,
                }
            } else {
                kept.push(event);
            }
        }
    }
    *events = kept;
    json!({
        "duplicate_exact_prefix_events_beyond_first": duplicate_prefixes,
        "distinct_exact_prefixes_with_opposing_actions_excluded": opposing_actions,
        "distinct_exact_prefixes_in_multiple_splits": cross_split_duplicates,
    })
}

#[derive(Default)]
struct Distribution {
    stops_before_conflict_exclusion: usize,
    continues_before_conflict_exclusion: usize,
    stops: usize,
    continues: usize,
    matched: usize,
}

fn index_for(corpus: &Corpus, event: Event) -> (&str, usize, usize) {
    let turn = &corpus.turns[event.turn];
    (
        &corpus.records[turn.record].id,
        turn.ordinal,
        event.stratum.completed_answer_ids,
    )
}

fn match_events(
    corpus: &Corpus,
    partition: &Partition,
    events: &[Event],
    counts: &mut [Counts],
    distributions: &mut [BTreeMap<Stratum, Distribution>],
) -> Vec<(Event, Event)> {
    let mut stop_records: [BTreeSet<usize>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeSet::new());
    let mut continue_records: [BTreeSet<usize>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeSet::new());
    let mut candidates = BTreeMap::<(usize, Stratum), Vec<Event>>::new();
    for &event in events {
        let record = event.record(corpus);
        let split = partition.splits[record];
        let distribution = distributions[split].entry(event.stratum).or_default();
        match event.action() {
            Action::Continue => {
                candidates
                    .entry((split, event.stratum))
                    .or_default()
                    .push(event);
                counts[split].eligible_continues_after_ambiguity += 1;
                distribution.continues += 1;
                continue_records[split].insert(record);
            }
            Action::Stop => {
                counts[split].eligible_stops_after_ambiguity += 1;
                distribution.stops += 1;
                stop_records[split].insert(record);
            }
        }
    }
    for split in 0..SPLITS.len() {
        counts[split].further_unique_stop_records = stop_records[split].len();
        counts[split].further_unique_continuation_records = continue_records[split].len();
    }
    let mut matched = Vec::new();
    let mut used_stop = BTreeSet::new();
    let mut used_continue = BTreeSet::new();
    let mut stop_components: [BTreeSet<usize>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeSet::new());
    let mut continue_components: [BTreeSet<usize>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeSet::new());
    for &stop in events.iter().filter(|event| event.action() == Action::Stop) {
        let record = stop.record(corpus);
        if used_stop.contains(&record) {
            continue;
        }
        let split = partition.splits[record];
        let component = partition.parent[record];
        let Some(continuation) = candidates
            .get(&(split, stop.stratum))
            .and_then(|candidates| {
                candidates.iter().copied().find(|candidate| {
                    let other = candidate.record(corpus);
                    !used_continue.contains(&other) && partition.parent[other] != component
                })
            })
        else {
            continue;
        };
        let other = continuation.record(corpus);
        if let Some(distribution) = distributions[split].get_mut(&stop.stratum) {
            distribution.matched += 1;
        }
        counts[split].matched_pairs += 1;
        stop_components[split].insert(component);
        continue_components[split].insert(partition.parent[other]);
        used_stop.insert(record);
        used_continue.insert(other);
        matched.push((stop, continuation));
    }
    for split in 0..SPLITS.len() {
        counts[split].matched_stop_components = stop_components[split].len();
        counts[split].matched_continue_components = continue_components[split].len();
        counts[split].matched_distinct_components = stop_components[split]
            .union(&continue_components[split])
            .count();
    }
    matched
}

fn event_value(corpus: &Corpus, partition: &Partition, event: Event) -> Value {
    let turn = &corpus.turns[event.turn];
    let record = &corpus.records[turn.record];
    json!({
        "record": record.id,
        "ordinal": record.ordinal,
        "turn": turn.ordinal,
        "component": corpus.records[partition.parent[turn.record]].id,
        "prefix": event.prefix(corpus).collect::<Vec<_>>(),
        "opening_len": turn.opening.len(),
        "completed_answer_ids": event.stratum.completed_answer_ids,
        "target": event.target,
        "payload_offset": turn.opening_offset + turn.opening.len() + event.stratum.completed_answer_ids,
        "last_char_class": event.stratum.last_char_class,
        "has_history": event.stratum.has_history,
    })
}

fn row(writer: &mut (impl Write + ?Sized), value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub struct Outputs<'a> {
    pub events: &'a mut dyn Write,
    pub pairs: &'a mut dyn Write,
    pub truth: &'a mut dyn Write,
    pub strata: &'a mut dyn Write,
}

fn distribution_value(key: Stratum, distribution: &Distribution) -> Value {
    json!({
        "key": key,
        "stops_before_conflict_exclusion": distribution.stops_before_conflict_exclusion,
        "continues_before_conflict_exclusion": distribution.continues_before_conflict_exclusion,
        "stops_after_conflict_exclusion": distribution.stops,
        "continues_after_conflict_exclusion": distribution.continues,
        "matched_pairs": distribution.matched,
    })
}

pub fn census(corpus: &Corpus, partition: &Partition, outputs: Outputs<'_>) -> Result<Value> {
    let mut counts: [Counts; SPLITS.len()] = std::array::from_fn(|_| Counts::default());
    let mut observed = population(corpus, partition, &mut counts)?;
    let mut distributions: [BTreeMap<Stratum, Distribution>; SPLITS.len()] =
        std::array::from_fn(|_| BTreeMap::new());
    for &event in &observed {
        let split = partition.splits[event.record(corpus)];
        let stratum = distributions[split].entry(event.stratum).or_default();
        match event.action() {
            Action::Stop => stratum.stops_before_conflict_exclusion += 1,
            Action::Continue => stratum.continues_before_conflict_exclusion += 1,
        }
    }
    let ambiguity = unregister_conflicts(corpus, partition, &mut observed, &mut counts);
    observed.sort_by_key(|&event| index_for(corpus, event));
    let pairs = match_events(
        corpus,
        partition,
        &observed,
        &mut counts,
        &mut distributions,
    );
    for &event in &observed {
        row(
            outputs.events,
            &json!({
                "split": SPLITS[partition.splits[event.record(corpus)]],
                "event": event_value(corpus, partition, event),
            }),
        )?;
    }
    let mut truth_turns = BTreeSet::new();
    for (stop, continuation) in pairs {
        truth_turns.insert(stop.turn);
        truth_turns.insert(continuation.turn);
        row(
            outputs.pairs,
            &json!({
                "split": SPLITS[partition.splits[stop.record(corpus)]],
                "stop": event_value(corpus, partition, stop),
                "continuation": event_value(corpus, partition, continuation),
            }),
        )?;
    }
    let mut truth_order: Vec<_> = truth_turns.into_iter().collect();
    truth_order.sort_by_key(|&index| {
        let turn = &corpus.turns[index];
        (&corpus.records[turn.record].id, turn.ordinal)
    });
    for index in truth_order {
        let turn = &corpus.turns[index];
        let record = &corpus.records[turn.record];
        let start = turn.opening_offset + turn.opening.len();
        row(
            outputs.truth,
            &json!({
                "record": record.id,
                "turn": turn.ordinal,
                "ordinal": record.ordinal,
                "opening_start": turn.opening_offset,
                "answer_start": start,
                "answer_end": start + turn.answer.len(),
                "answer_ids": turn.answer,
                "eos": EOS,
                "eos_supervised": turn.eos_supervised,
                "user": (turn.ordinal == 0).then_some(turn.user.as_str()),
            }),
        )?;
    }
    for (i, split) in distributions.iter().enumerate() {
        for (&key, distribution) in split {
            row(
                outputs.strata,
                &json!({
                    "split": SPLITS[i],
                    "distribution": distribution_value(key, distribution),
                }),
            )?;
        }
    }
    let split_summary = std::array::from_fn::<_, 3, _>(|i| {
        json!({
            "split": SPLITS[i],
            "counts": counts[i],
            "all_strata": distributions[i].len(),
            "strata_with_observed_stop": distributions[i].iter()
                .filter(|(_, distribution)| distribution.stops_before_conflict_exclusion > 0)
                .map(|(&key, distribution)| distribution_value(key, distribution)).collect::<Vec<_>>(),
        })
    });
    Ok(json!({
        "opening_and_eligible_boundary_ambiguity": ambiguity,
        "registered_train_and_held_supply_gate": counts[0].matched_pairs > 0 && counts[2].matched_pairs > 0,
        "splits": split_summary,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{RecordKey, Turn};

    fn turn(
        record: usize,
        opening: Vec<u32>,
        answer: &[u32],
        text: &str,
        eos_supervised: bool,
    ) -> Turn {
        Turn {
            record,
            ordinal: 0,
            user: format!("fixture {record}"),
            opening_offset: record * 500,
            opening,
            answer: answer.to_vec(),
            answer_text: text.to_owned(),
            eos_supervised,
        }
    }

    #[test]
    fn matched_population_rejects_conflicting_prefix_masked_end_and_punctuation() -> Result<()> {
        let corpus = Corpus {
            records: (0..6)
                .map(|ordinal| RecordKey {
                    id: format!("fixture-{ordinal}"),
                    ordinal,
                })
                .collect(),
            turns: vec![
                turn(0, vec![0, 28], &[3, 4], "A!", true),
                turn(1, vec![0, 28], &[3, 4, 5], "A!B", false),
                turn(2, vec![0, 40, 28], &[3, 4], "A!", true),
                turn(3, vec![0, 41, 28], &[3, 4, 5], "A!B", false),
                turn(4, vec![0, 42, 28], &[3, 4], "A!", false),
                turn(5, vec![0; WINDOW - 1], &[3, 4, 5], "A!B", true),
            ],
            visible_tokens: 0,
            token_bytes: vec![
                vec![],
                vec![],
                vec![],
                b"A".to_vec(),
                b"!".to_vec(),
                b"B".to_vec(),
            ],
        };
        let partition = Partition {
            parent: (0..6).collect(),
            splits: vec![0; 6],
            audit: Value::Null,
        };
        let mut events = Vec::new();
        let mut pairs = Vec::new();
        let mut truth = Vec::new();
        let result = census(
            &corpus,
            &partition,
            Outputs {
                events: &mut events,
                pairs: &mut pairs,
                truth: &mut truth,
                strata: &mut std::io::sink(),
            },
        )?;
        let pair: Value = serde_json::from_slice(&pairs)?;
        let counts = &result["splits"][0]["counts"];
        assert_eq!(counts["stop_candidates"], 2);
        assert_eq!(counts["continue_candidates"], 2);
        assert_eq!(counts["removed_ambiguous_stops"], 1);
        assert_eq!(counts["removed_ambiguous_continues"], 1);
        assert_eq!(counts["history_eos_mask_zero"], 3);
        assert_eq!(counts["internal_ids_punctuation_or_space_only"], 5);
        assert_eq!(counts["internal_ids_without_target_room"], 2);
        assert_eq!(counts["matched_pairs"], 1);
        assert_eq!(pair["stop"]["record"], "fixture-2");
        assert_eq!(pair["continuation"]["record"], "fixture-3");
        assert_eq!(pair["stop"]["target"], 1);
        assert_eq!(pair["continuation"]["target"], 5);
        Ok(())
    }

    #[test]
    fn native_bytes_preserve_observable_unicode_and_character_boundaries() -> Result<()> {
        let token_bytes = vec![
            b"A".to_vec(),
            vec![0xc3],
            vec![0xa9],
            "。".as_bytes().to_vec(),
            b" ".to_vec(),
        ];
        let shape = shapes(
            "Aé。 ",
            &[0, 1, 2, 3, 4],
            &token_bytes,
            &punctuation_class()?,
        )?;
        assert!(!shape[1].completed_alnum && shape[1].ends_mid_char);
        assert_eq!(shape[1].after_class, Some(CharClass::Alphanumeric));
        assert!(shape[2].completed_alnum && shape[2].split_alnum);
        assert_eq!(shape[3].after_class, Some(CharClass::OtherPunctuation));
        assert!(shape[3].punctuation_or_space_only && !shape[3].whitespace_only);
        assert_eq!(shape[4].after_class, Some(CharClass::Whitespace));
        assert!(shape[4].whitespace_only);
        Ok(())
    }
}
