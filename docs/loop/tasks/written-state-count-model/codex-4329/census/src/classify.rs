use super::{require, Audit, Counts, CLASSES_SHA, CONTEXT, MARKER, SPACE};
use crate::inputs::{self, Colon};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::Path;

const BOS_USER: [u32; 4] = [0, 55, 2728, 28];
const NEXT_USER: [u32; 4] = [201, 55, 2728, 28];
const PARAGRAPH: [u32; 2] = [201, 201];

/// Historical #4315 fact span. Use this unchanged for the historical census.
fn historical_span(ids: &[u32]) -> Option<Range<usize>> {
    let body = ids.strip_prefix(&BOS_USER)?.strip_suffix(&MARKER)?;
    if body.iter().any(|&id| id < 2)
        || body.windows(MARKER.len()).any(|ids| ids == MARKER)
        || body.windows(NEXT_USER.len()).any(|ids| ids == NEXT_USER)
    {
        return None;
    }
    let mut splits = body
        .windows(PARAGRAPH.len())
        .enumerate()
        .filter_map(|(position, ids)| (ids == PARAGRAPH).then_some(position));
    let end = splits.next()?;
    if end == 0 || end + PARAGRAPH.len() >= body.len() || splits.next().is_some() {
        return None;
    }
    Some(BOS_USER.len()..BOS_USER.len() + end)
}

pub fn serving_span(ids: &[u32], prefix_start: usize) -> Option<Range<usize>> {
    if ids.len() > CONTEXT {
        return None;
    }
    let conversational = ids.strip_suffix(&[SPACE]).unwrap_or(ids);
    let span = historical_span(conversational)?;
    Some(prefix_start + span.start..prefix_start + span.end)
}

pub fn historical(
    colon_path: &Path,
    classes: &[u8],
    tape: &[u32],
    mask: &[u8],
) -> Audit<(Value, BTreeMap<usize, Colon>)> {
    let mut rows = inputs::rows::<Colon>(colon_path)?;
    let mut counts = Counts::new();
    let mut openings = BTreeMap::new();
    let mut targets = BTreeSet::new();
    let mut unique = BTreeSet::new();
    let mut parsed = 0;
    let mut total = 0;
    let mut bos = 0;
    for (offset, &id) in tape.iter().enumerate() {
        if id == 0 {
            bos = offset;
        }
        if offset <= bos || tape[offset - 1] != 28 || mask[offset] != 1 {
            continue;
        }
        let row = inputs::next(&mut rows)?;
        require(
            row.offset == offset
                && row.target == id
                && row.prefix == tape[bos.max(offset.saturating_sub(CONTEXT))..offset],
            &format!("colon tape mismatch at offset {offset}"),
        )?;
        let role = row.prefix.ends_with(&MARKER);
        require(
            role == row.genuine_assistant,
            "historical marker/provenance mismatch",
        )?;
        total += 1;
        let kind = if role {
            targets.insert(row.target);
            let category = match historical_span(&row.prefix) {
                None => "unparsed_genuine",
                Some(span) => {
                    parsed += 1;
                    if !eligible(classes[id as usize]) {
                        "nonlexical_genuine"
                    } else {
                        match row.prefix[span]
                            .iter()
                            .filter(|&&value| value == id)
                            .count()
                        {
                            0 => "source_absent",
                            1 => {
                                unique.insert(id);
                                "source_unique"
                            }
                            _ => "source_ambiguous",
                        }
                    }
                }
            };
            require(
                openings.insert(offset, row).is_none(),
                "duplicate opening offset",
            )?;
            category
        } else {
            "other_colon"
        };
        *counts.entry(kind).or_default() += 1;
    }
    require(rows.next().is_none(), "unmatched colon rows")?;
    let result = json!({
        "schema": "uor-r4.canonical-donor-source-admission/1",
        "rows": total,
        "genuine": openings.len(),
        "parsed_genuine": parsed,
        "partition": counts,
        "supported_source_target_ids": unique,
        "distinct_genuine_target_ids": targets.len()
    });
    Ok((result, openings))
}

fn eligible(class: u8) -> bool {
    class != 0 && class != 9
}

fn category(class: u8, matches: usize) -> &'static str {
    if !eligible(class) {
        return "nonlexical";
    }
    match matches {
        0 => "absent",
        1 => "unique",
        _ => "duplicate",
    }
}

#[derive(Clone, Serialize)]
pub struct Occurrence {
    pub offset: usize,
    pub prompt_index: usize,
    pub user_index: usize,
    pub user_protocol_bytes: [usize; 2],
    pub standalone_word: bool,
}

#[derive(Serialize)]
pub struct Target {
    pub id: u32,
    pub class: u8,
    pub surface: String,
    pub token_bytes: Vec<u8>,
    pub target_offset: usize,
    pub target_answer_index: usize,
    pub target_answer_protocol_bytes: [usize; 2],
    pub target_standalone_word: bool,
    pub raw_exact: Vec<Occurrence>,
    pub window_exact: Vec<Occurrence>,
    pub framed_exact: Vec<Occurrence>,
    pub prior_history_exact_offsets: Vec<usize>,
    pub prior_user_content_exact_offsets: Vec<usize>,
    pub prior_assistant_content_exact_offsets: Vec<usize>,
    pub raw_kind: &'static str,
    pub window_kind: &'static str,
    pub framed_kind: &'static str,
    pub framed_prompt_offsets: Option<[usize; 2]>,
    pub phase_window_start: Option<usize>,
    pub window_current_user_tokens: Option<usize>,
    pub unshifted_context_len: usize,
}

pub struct TurnContext<'a> {
    pub tape: &'a [u32],
    pub classes: &'a [u8],
    pub token_bytes: &'a [Vec<u8>],
    pub prefix_start: usize,
    pub answer_start: usize,
    pub user: Range<usize>,
    pub prior_end: usize,
    pub prior_user: &'a [Range<usize>],
    pub prior_assistant: &'a [Range<usize>],
    pub user_bytes: &'a [usize],
    pub answer_bytes: &'a [usize],
    pub user_content: &'a str,
    pub answer_content: &'a str,
}

pub fn byte_bounds(tokens: &[u32], token_bytes: &[Vec<u8>]) -> Vec<usize> {
    let mut bounds = Vec::with_capacity(tokens.len() + 1);
    bounds.push(0);
    let mut offset = 0;
    for &id in tokens {
        offset += token_bytes[id as usize].len();
        bounds.push(offset);
    }
    bounds
}

/// A strict single-word check, based on the original ID bytes and the text on
/// both sides. A BPE piece touching a letter or word joiner cannot pass.
fn standalone_word(text: &str, [start, end]: [usize; 2]) -> bool {
    let Some(surface) = text.get(start..end) else {
        return false;
    };
    let word = surface.trim();
    if word.is_empty() || !word.chars().all(char::is_alphanumeric) {
        return false;
    }
    let word_start = start + surface.len() - surface.trim_start().len();
    let word_end = word_start + word.len();
    let word_char = |c: char| c.is_alphanumeric() || matches!(c, '_' | '\'' | '’' | '-');
    !text[..word_start]
        .chars()
        .next_back()
        .is_some_and(word_char)
        && !text[word_end..].chars().next().is_some_and(word_char)
}

impl TurnContext<'_> {
    pub fn target(&self, offset: usize) -> Target {
        let id = self.tape[offset];
        let class = self.classes[id as usize];
        let token_bytes = self.token_bytes[id as usize].clone();
        let phase_window_start = (self.answer_start - self.prefix_start <= CONTEXT)
            .then(|| self.prefix_start.max(offset.saturating_sub(CONTEXT)));
        let framed =
            phase_window_start.and_then(|start| serving_span(&self.tape[start..offset], start));
        let mut raw_exact = Vec::new();
        let mut window_exact = Vec::new();
        let mut framed_exact = Vec::new();
        for position in self.user.clone().filter(|&i| self.tape[i] == id) {
            let user_index = position - self.user.start;
            let bounds = [self.user_bytes[user_index], self.user_bytes[user_index + 1]];
            let occurrence = Occurrence {
                offset: position,
                prompt_index: position - self.prefix_start,
                user_index,
                user_protocol_bytes: bounds,
                standalone_word: standalone_word(self.user_content, bounds),
            };
            if phase_window_start.is_some_and(|start| position >= start) {
                window_exact.push(occurrence.clone());
            }
            if framed
                .as_ref()
                .is_some_and(|range| range.contains(&position))
            {
                framed_exact.push(occurrence.clone());
            }
            raw_exact.push(occurrence);
        }
        let prior_history_exact_offsets = (self.prefix_start..self.prior_end)
            .filter(|&i| self.tape[i] == id)
            .collect();
        let prior_matches = |spans: &[Range<usize>]| {
            spans
                .iter()
                .cloned()
                .flatten()
                .filter(|&i| self.tape[i] == id)
                .collect()
        };
        let target_answer_index = offset - self.answer_start;
        let target_answer_protocol_bytes = [
            self.answer_bytes[target_answer_index],
            self.answer_bytes[target_answer_index + 1],
        ];
        let raw_kind = category(class, raw_exact.len());
        let window_kind =
            phase_window_start.map_or("unservable", |_| category(class, window_exact.len()));
        let framed_kind = phase_window_start.map_or("unservable", |_| {
            framed
                .as_ref()
                .map_or("unframed", |_| category(class, framed_exact.len()))
        });
        Target {
            id,
            class,
            surface: String::from_utf8_lossy(&token_bytes).into_owned(),
            token_bytes,
            target_offset: offset,
            target_answer_index,
            target_answer_protocol_bytes,
            target_standalone_word: standalone_word(
                self.answer_content,
                target_answer_protocol_bytes,
            ),
            raw_exact,
            window_exact,
            framed_exact,
            prior_history_exact_offsets,
            prior_user_content_exact_offsets: prior_matches(self.prior_user),
            prior_assistant_content_exact_offsets: prior_matches(self.prior_assistant),
            raw_kind,
            window_kind,
            framed_kind,
            framed_prompt_offsets: framed.map(|range| [range.start, range.end]),
            phase_window_start,
            window_current_user_tokens: phase_window_start
                .map(|start| self.user.end - self.user.start.max(start)),
            unshifted_context_len: offset - self.prefix_start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: u32 = 1042;
    const OTHER: u32 = 701;

    #[test]
    fn fact_fence_excludes_question_duplicates_and_whole_turn_keeps_them() {
        let tokens = [
            BOS_USER.as_slice(),
            &[TOKEN, PARAGRAPH[0], PARAGRAPH[1], TOKEN],
            &MARKER,
            &[TOKEN],
        ]
        .concat();
        let mut classes = vec![0; super::super::VOCAB];
        classes[TOKEN as usize] = 1;
        let mut bytes = vec![Vec::new(); super::super::VOCAB];
        bytes[TOKEN as usize] = b"Elm".to_vec();
        let user_bytes = [0, 3, 4, 5, 8];
        let offset = tokens.len() - 1;
        let turn = TurnContext {
            tape: &tokens,
            classes: &classes,
            token_bytes: &bytes,
            prefix_start: 0,
            answer_start: offset,
            user: 4..8,
            prior_end: 1,
            prior_user: &[],
            prior_assistant: &[],
            user_bytes: &user_bytes,
            answer_bytes: &[0, 3],
            user_content: "Elm\n\nElm",
            answer_content: "Elm",
        };
        let target = turn.target(offset);
        assert_eq!(target.framed_kind, "unique");
        assert_eq!(target.framed_exact[0].offset, 4);
        assert_eq!(target.window_kind, "duplicate");
        assert_eq!(target.window_exact[1].offset, 7);
        assert!(target.target_standalone_word);
        assert!(target
            .window_exact
            .iter()
            .all(|entry| entry.standalone_word));
    }

    #[test]
    fn framed_route_refuses_nested_role_ids_and_slid_bos() {
        let plain = [BOS_USER.as_slice(), &[TOKEN], &PARAGRAPH, &[OTHER], &MARKER].concat();
        assert_eq!(serving_span(&plain, 100), Some(104..105));
        assert_eq!(
            serving_span(&[plain.as_slice(), &[SPACE]].concat(), 100),
            Some(104..105)
        );
        assert_eq!(serving_span(&[&plain[1..], &[SPACE]].concat(), 101), None);
        let nested = [
            BOS_USER.as_slice(),
            &NEXT_USER,
            &[TOKEN],
            &PARAGRAPH,
            &[OTHER],
            &MARKER,
        ]
        .concat();
        assert_eq!(serving_span(&nested, 0), None);
    }

    #[test]
    fn partial_words_and_invalid_utf8_boundaries_cannot_certify_whole_word() {
        assert!(standalone_word(" Elm.", [0, 4]));
        assert!(standalone_word(" Élan", [0, 6]));
        assert!(!standalone_word(" Élan", [0, 2]));
        assert!(!standalone_word(" Elms.", [0, 4]));
        assert!(!standalone_word(" an Elm-wood example", [3, 7]));
        assert!(!standalone_word(" Salem", [3, 6]));
    }
}

pub fn canonical_classes(path: &Path) -> Audit<Vec<u8>> {
    let bytes = inputs::checked_data(path, CLASSES_SHA)?;
    require(bytes.len() == super::VOCAB, "invalid canonical classes")?;
    Ok(bytes)
}
