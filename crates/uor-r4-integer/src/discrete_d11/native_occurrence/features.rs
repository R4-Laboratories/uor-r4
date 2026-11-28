use std::{cmp::Reverse, ops::Range};

use super::{CONTEXT, VOCABULARY};
use crate::discrete_d11::native_opening::{ASSISTANT_MARKER, SPACE};

pub const ACTIVE: usize = 16;
pub const WEIGHTS: usize = ACTIVE * 32;
const USER: [u32; 4] = [0, 55, 2728, 28];
const NEXT_USER: [u32; 4] = [201, 55, 2728, 28];
const LINE_BREAK: [u32; 2] = [201, 201];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Candidate {
    pub position: u16,
    pub token: u16,
    pub features: [u16; ACTIVE],
}

pub fn classify_token(id: u32, bytes: &[u8]) -> u8 {
    let start = bytes.iter().position(|byte| !byte.is_ascii_whitespace());
    let end = bytes.iter().rposition(|byte| !byte.is_ascii_whitespace());
    let (Some(start), Some(end)) = (start, end) else {
        return 0;
    };
    if id < 2 {
        return 0;
    }
    let first = bytes[start];
    if start == end && b".,;:!?".contains(&first) {
        return 9;
    }
    let lexical = match first {
        b'a'..=b'z' => 1,
        b'A'..=b'Z' => 3,
        b'0'..=b'9' => 5,
        _ => 7,
    };
    lexical + u8::from(start != 0)
}

fn spans(ids: &[u32]) -> Option<(Range<usize>, Range<usize>)> {
    if ids.len() > CONTEXT || ids.iter().any(|&id| id as usize >= VOCABULARY) {
        return None;
    }
    let conversational = ids.strip_suffix(&[SPACE]).unwrap_or(ids);
    let body = conversational
        .strip_prefix(&USER)?
        .strip_suffix(&ASSISTANT_MARKER)?;
    if body.iter().any(|&id| id < 2)
        || body.windows(NEXT_USER.len()).any(|ids| ids == NEXT_USER)
        || body
            .windows(ASSISTANT_MARKER.len())
            .any(|ids| ids == ASSISTANT_MARKER)
    {
        return None;
    }
    let mut splits = body
        .windows(LINE_BREAK.len())
        .enumerate()
        .filter_map(|(position, ids)| (ids == LINE_BREAK).then_some(position));
    let split = splits.next()?;
    if split == 0 || split + LINE_BREAK.len() == body.len() || splits.next().is_some() {
        return None;
    }
    let fact_end = USER.len() + split;
    let question_start = fact_end + LINE_BREAK.len();
    let question_end = USER.len() + body.len();
    Some((USER.len()..fact_end, question_start..question_end))
}

fn eligible(class: u8) -> bool {
    class != 0 && class != 9
}

#[derive(Clone, Copy, Default)]
struct Anchor {
    fact: usize,
    question: usize,
    quality: u8,
}

pub(super) struct Prompt<'a> {
    ids: &'a [u32],
    classes: &'a [u8; VOCABULARY],
    roots: &'a [[u8; 4]],
    fact: Range<usize>,
    question: Range<usize>,
    fact_count: [u8; VOCABULARY],
    question_count: [u8; VOCABULARY],
    anchors: [Anchor; CONTEXT],
    anchors_len: usize,
    word_starts: [u8; CONTEXT],
}

impl<'a> Prompt<'a> {
    pub(super) fn new(
        ids: &'a [u32],
        classes: &'a [u8; VOCABULARY],
        roots: &'a [[u8; 4]],
    ) -> Option<Self> {
        let (fact, question) = spans(ids)?;
        if roots.len() != VOCABULARY {
            return None;
        }
        let mut prompt = Self {
            ids,
            classes,
            roots,
            fact,
            question,
            fact_count: [0; VOCABULARY],
            question_count: [0; VOCABULARY],
            anchors: [Anchor::default(); CONTEXT],
            anchors_len: 0,
            word_starts: [0; CONTEXT],
        };
        let mut word_starts = 0;
        for position in prompt.fact.clone() {
            prompt.fact_count[prompt.token(position)] += 1;
            word_starts += u8::from(prompt.starts_word(position));
            prompt.word_starts[position] = word_starts;
        }
        for position in prompt.question.clone() {
            prompt.question_count[prompt.token(position)] += 1;
        }
        for position in prompt.fact.clone() {
            let token = prompt.token(position);
            if !eligible(prompt.class(position)) || prompt.question_count[token] == 0 {
                continue;
            }
            let Some(question) = prompt
                .question
                .clone()
                .find(|&position| prompt.token(position) == token)
            else {
                continue;
            };
            let upper = prompt.uppercase(position);
            let unique = prompt.fact_count[token] == 1 && prompt.question_count[token] == 1;
            prompt.anchors[prompt.anchors_len] = Anchor {
                fact: position,
                question,
                quality: 1 + 2 * u8::from(upper) + u8::from(unique),
            };
            prompt.anchors_len += 1;
        }
        Some(prompt)
    }

    pub(super) fn fact_bounds(&self) -> [u16; 2] {
        [self.fact.start as u16, self.fact.end as u16]
    }

    fn token(&self, position: usize) -> usize {
        self.ids[position] as usize
    }

    fn class(&self, position: usize) -> u8 {
        self.classes[self.token(position)]
    }

    fn follows_gap(&self, position: usize) -> bool {
        position == self.fact.start || self.ids[position - 1] == SPACE
    }

    fn starts_word(&self, position: usize) -> bool {
        let class = self.class(position);
        matches!(class, 2 | 4 | 6 | 8) || (eligible(class) && self.follows_gap(position))
    }

    fn uppercase(&self, position: usize) -> bool {
        let class = self.class(position);
        class == 4 || (class == 3 && self.follows_gap(position))
    }

    fn root_overlap(&self, a: Option<usize>, b: Option<usize>) -> u8 {
        let (Some(a), Some(b)) = (a, b) else {
            return 0;
        };
        self.roots[self.token(a)]
            .iter()
            .zip(&self.roots[self.token(b)])
            .filter(|(x, y)| x == y)
            .count() as u8
    }

    pub(super) fn candidate(&self, position: usize) -> Option<Candidate> {
        if !self.fact.contains(&position) || !eligible(self.class(position)) {
            return None;
        }
        let anchor = self.anchors[..self.anchors_len]
            .iter()
            .min_by_key(|a| (Reverse(a.quality), position.abs_diff(a.fact), a.fact));
        let left = previous(&self.fact, position);
        let right = next(&self.fact, position);
        let left_boundary = (self.fact.start..position)
            .rfind(|&p| self.class(p) == 9)
            .unwrap_or(self.fact.start - 1);
        let right_boundary = (position + 1..self.fact.end)
            .find(|&p| self.class(p) == 9)
            .unwrap_or(self.fact.end);
        let other_upper = self
            .fact
            .clone()
            .filter(|&p| p != position && self.uppercase(p))
            .min_by_key(|&p| (position.abs_diff(p), p));
        let query_left = anchor.and_then(|a| previous(&self.question, a.question));
        let query_right = anchor.and_then(|a| next(&self.question, a.question));
        let word_delta = anchor
            .map(|a| i16::from(self.word_starts[position]) - i16::from(self.word_starts[a.fact]));
        let position_as_i16 = position as i16;
        let token = self.token(position);
        let class = self.class(position);
        let categories = [
            class,
            delta_bucket(anchor.map(|a| position_as_i16 - a.fact as i16)),
            delta_bucket(word_delta),
            anchor.map_or(0, |a| a.quality),
            5 * self.fact_count[token].min(4) + self.question_count[token].min(4),
            left.map_or(0, |p| self.class(p)),
            right.map_or(0, |p| self.class(p)),
            (position - left_boundary).min(15) as u8,
            (right_boundary - position).min(15) as u8,
            anchor.map_or(5, |a| self.root_overlap(Some(position), Some(a.fact))),
            5 * self.root_overlap(Some(position), left) + self.root_overlap(Some(position), right),
            5 * self.root_overlap(left, query_left) + self.root_overlap(right, query_right),
            5 * left.map_or(0, |p| self.fact_count[self.token(p)].min(4))
                + right.map_or(0, |p| self.fact_count[self.token(p)].min(4)),
            delta_bucket(other_upper.map(|p| position_as_i16 - p as i16)),
            anchor.map_or(0, |a| {
                6 * word_delta.unwrap_or(0).unsigned_abs().min(4) as u8 + a.quality
            }),
            5 * (right_boundary - position).min(5) as u8 + class.min(4),
        ];
        Some(Candidate {
            position: position as u16,
            token: token as u16,
            features: std::array::from_fn(|slot| 32 * slot as u16 + u16::from(categories[slot])),
        })
    }

    pub(super) fn candidates(&self) -> impl Iterator<Item = Candidate> + '_ {
        self.fact
            .clone()
            .filter_map(|position| self.candidate(position))
    }
}

fn previous(span: &Range<usize>, position: usize) -> Option<usize> {
    (position > span.start).then(|| position - 1)
}

fn next(span: &Range<usize>, position: usize) -> Option<usize> {
    (position + 1 < span.end).then_some(position + 1)
}

fn delta_bucket(delta: Option<i16>) -> u8 {
    delta.map_or(29, |value| (value.clamp(-14, 14) + 14) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete_d11::native_occurrence::{OpeningChoice, OpeningDecision, State};

    #[test]
    fn occurrence_ablation_retains_framing_and_removes_a_selectable_source() {
        let mut classes = Box::new([0; VOCABULARY]);
        classes[51] = 2;
        let state = State {
            weights: [0; WEIGHTS],
            no_read: -1,
            classes,
            state_sha256: String::new(),
            fit_sha256: String::new(),
        };
        let roots = [[0; 4]; VOCABULARY];
        let mut ids = Vec::from(USER);
        ids.extend([51, LINE_BREAK[0], LINE_BREAK[1], 80]);
        ids.extend(ASSISTANT_MARKER);

        for _ in 0..2 {
            let source = state.choose(&ids, &roots, OpeningChoice::Learned).unwrap();
            assert_eq!((source.position, source.token), (4, 51));
            assert!(state
                .choose(&ids, &roots, OpeningChoice::Fallback)
                .is_none());
            let trace = state.probe(&ids, &roots, false);
            assert!(matches!(trace.decision, OpeningDecision::Disabled));
            assert_eq!((trace.parsed_start, trace.parsed_end), (Some(4), Some(5)));
            assert!(trace.best_score.is_none());
            ids.push(SPACE);
        }
    }

    #[test]
    fn only_causal_fact_positions_enter_the_reader() {
        let mut classes = [0; VOCABULARY];
        for (id, class) in [
            (50, 4),
            (51, 2),
            (16, 9),
            (60, 4),
            (61, 2),
            (80, 2),
            (33, 9),
        ] {
            classes[id] = class;
        }
        let mut roots = [[0; 4]; VOCABULARY];
        roots[51] = [1, 2, 3, 4];
        roots[50] = [1, 2, 5, 6];
        roots[80] = [1, 6, 5, 6];
        let mut ids = Vec::from(USER);
        ids.extend([50, 51, 16, 60, 61]);
        ids.extend(LINE_BREAK);
        ids.extend([80, 50, 33]);
        ids.extend(ASSISTANT_MARKER);
        let prompt = Prompt::new(&ids, &classes, &roots).unwrap();
        assert_eq!(prompt.fact_bounds(), [4, 9]);
        let positions: Vec<_> = prompt
            .candidates()
            .map(|candidate| candidate.position)
            .collect();
        assert_eq!(positions, [4, 5, 7, 8]);
        let expected = [2, 15, 15, 4, 5, 4, 9, 2, 1, 2, 10, 19, 6, 15, 10, 7];
        let features = std::array::from_fn(|slot| 32 * slot as u16 + expected[slot]);
        assert_eq!(prompt.candidate(5).unwrap().features, features);
        assert!(prompt.candidate(12).is_none());

        ids.push(SPACE);
        let continued = Prompt::new(&ids, &classes, &roots).unwrap();
        assert_eq!(continued.candidate(5).unwrap().features, features);
        ids.push(SPACE);
        assert!(Prompt::new(&ids, &classes, &roots).is_none());
        ids.truncate(ids.len() - 2);
        ids.insert(9, 201);
        assert!(Prompt::new(&ids, &classes, &roots).is_none());
        ids.remove(9);
        ids.splice(5..5, NEXT_USER);
        assert!(Prompt::new(&ids, &classes, &roots).is_none());
    }

    #[test]
    fn byte_classes_never_require_a_lexical_label() {
        for (id, bytes, expected) in [
            (0, b"START".as_slice(), 0),
            (10, b"\r\n", 0),
            (10, b"river", 1),
            (10, b" river", 2),
            (10, b"River", 3),
            (10, b" River", 4),
            (10, b"9a", 5),
            (10, b" 9a", 6),
            (10, b"'s", 7),
            (10, b" 's", 8),
            (10, b" ? ", 9),
            (10, " Å".as_bytes(), 8),
        ] {
            assert_eq!(classify_token(id, bytes), expected);
        }
    }
}
