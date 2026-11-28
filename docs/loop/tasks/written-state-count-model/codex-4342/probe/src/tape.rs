use std::{cmp::Reverse, collections::BTreeMap, ops::Range};

use serde::Deserialize;
use uor_r4_tokenizer::{ByteBpeTokenizer, dialogue::EncodedUserOpening};

use crate::{
    Result, Span, check,
    contrast::{BANKS, Keys, RELATION_BANK},
    oracle::{RelationOracle, Roots},
};

pub const ARMS: [&str; 5] = [
    "geometry",
    "count",
    "directed_geometry",
    "directed_count",
    "swapped_geometry",
];
const ROLE_COPY_STOPS: [u32; 4] = [16, 1, 0, 201];
pub const MAX_SPAN: usize = 8;
const BANK_WIDTH: u16 = 256;
const NO_READ_BASE: u16 = 224;

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tape {
    pub case_id: String,
    pub arm: String,
    pub user: [usize; 2],
    pub fact: [usize; 2],
    pub question: [usize; 2],
    pub candidates: Vec<Candidate>,
    pub no_read_keys: Keys,
    pub relation_endpoints: Option<Vec<Endpoints>>,
    pub relation_tokens: Option<Vec<Token>>,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub start: usize,
    pub end: usize,
    pub keys: Keys,
}

impl Candidate {
    pub fn span(&self) -> Span {
        Span {
            start: self.start,
            end: self.end,
        }
    }
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Endpoints {
    pub query: [Option<usize>; 2],
    pub source: [Option<usize>; 2],
    pub query_anchor: Option<[usize; 2]>,
    pub source_anchor: Option<usize>,
    pub clause: [usize; 2],
}

impl Endpoints {
    pub fn missing(&self) -> bool {
        self.query.contains(&None) || self.source.contains(&None)
    }

    pub fn missing_query(&self) -> bool {
        self.query.contains(&None)
    }

    pub fn missing_source(&self) -> bool {
        self.source.contains(&None)
    }
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Token {
    pub id: u32,
    pub bytes: Vec<u8>,
    pub root: u8,
}

#[derive(Debug)]
struct Word {
    tokens: Span,
    spelling: String,
    capitalized: bool,
}

fn words(tokens: &[Vec<u8>], bounds: [usize; 2]) -> Result<Vec<Word>> {
    let bytes = tokens[bounds[0]..bounds[1]].concat();
    // Only legacy matching tokenizes a clipped role. Original-user word
    // reconstruction receives the entire role, including split characters.
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return Ok(Vec::new());
    };
    let starts: Vec<_> = tokens[bounds[0]..bounds[1]]
        .iter()
        .scan(0, |pos, token| {
            let start = *pos;
            *pos += token.len();
            Some(start)
        })
        .collect();
    let covering = |byte| bounds[0] + starts.partition_point(|&start| start <= byte) - 1;
    let mut result = Vec::new();
    let mut text_span: Option<Range<usize>> = None;
    for (pos, ch) in text.char_indices() {
        match (ch.is_alphanumeric(), &mut text_span) {
            (true, Some(span)) => span.end = pos + ch.len_utf8(),
            (true, None) => text_span = Some(pos..pos + ch.len_utf8()),
            (false, _) => {
                if let Some(span) = text_span.take() {
                    result.push(word(text, span, covering)?);
                }
            }
        }
    }
    if let Some(span) = text_span {
        result.push(word(text, span, covering)?);
    }
    Ok(result)
}

fn word(text: &str, span: Range<usize>, covering: impl Fn(usize) -> usize) -> Result<Word> {
    let raw = text.get(span.clone()).ok_or("invalid lexical byte edges")?;
    Ok(Word {
        tokens: Span {
            start: covering(span.start),
            end: covering(span.end - 1) + 1,
        },
        spelling: raw.chars().flat_map(char::to_lowercase).collect(),
        capitalized: raw.chars().next().is_some_and(char::is_uppercase),
    })
}

pub fn terminal(byte: u8) -> bool {
    matches!(byte, b'.' | b'?' | b'!' | b'\n')
}

pub fn copyable(id: u32) -> bool {
    !ROLE_COPY_STOPS.contains(&id)
}

fn bounds(tokens: &[Vec<u8>], user: [usize; 2]) -> ([usize; 2], [usize; 2]) {
    let opening = tokens[user[0]..user[1]].concat();
    let ends: Vec<_> = tokens[user[0]..user[1]]
        .iter()
        .scan(0, |pos, bytes| {
            *pos += bytes.len();
            Some(*pos)
        })
        .collect();
    let containing = |byte| user[0] + ends.partition_point(|&end| end <= byte);
    if let Some(mark) = opening.iter().rposition(|&byte| byte == b'?') {
        let end = containing(mark) + 1;
        let before = &opening[..mark];
        match before
            .iter()
            .rposition(|&byte| terminal(byte))
            .or_else(|| before.iter().rposition(|&byte| byte == b':'))
        {
            Some(sep) => {
                let sep = containing(sep);
                ([user[0], sep], [sep + 1, end])
            }
            None => ([user[0], end], [end.saturating_sub(32).max(user[0]), end]),
        }
    } else if let Some(sep) = opening.windows(2).rposition(|pair| pair == b"\n\n") {
        (
            [user[0], containing(sep)],
            [containing(sep + 1) + 1, user[1]],
        )
    } else {
        ([user[0], user[0]], user)
    }
}

#[derive(Debug)]
struct Site {
    source: usize,
    query: usize,
    quality: u8,
}

fn matched_anchors(source: &[Word], query: &[Word]) -> Vec<Site> {
    let mut counts = BTreeMap::new();
    for word in source {
        *counts.entry(&word.spelling).or_insert(0) += 1;
    }
    let mut first_question = BTreeMap::new();
    for (index, word) in query.iter().enumerate() {
        first_question.entry(&word.spelling).or_insert(index);
    }
    let mut anchors = Vec::new();
    for word in source.iter().filter(|word| word.spelling.len() >= 3) {
        let Some(&query_index) = first_question.get(&word.spelling) else {
            continue;
        };
        anchors.push(Site {
            source: word.tokens.start,
            query: query_index,
            quality: 1
                + 2 * u8::from(counts[&word.spelling] <= 2)
                + 4 * u8::from(word.capitalized && query[query_index].capitalized),
        });
    }
    anchors
}

fn candidates(
    tokens: &[Vec<u8>],
    ids: &[u32],
    fact: [usize; 2],
    words: &[Word],
) -> Vec<(Span, [usize; 2])> {
    let mut result = Vec::new();
    let mut starts = vec![false; fact[1] - fact[0]];
    let mut ends = starts.clone();
    for word in words {
        starts[word.tokens.start - fact[0]] = true;
        ends[word.tokens.end - fact[0] - 1] = true;
    }
    let mut clause_start = fact[0];
    let clause_ends = tokens[fact[0]..fact[1]]
        .iter()
        .enumerate()
        .filter(|(_, token)| token.iter().copied().any(terminal))
        .map(|(offset, _)| fact[0] + offset)
        .chain(std::iter::once(fact[1]));
    for clause_end in clause_ends {
        for start in clause_start..clause_end {
            if !starts[start - fact[0]] || !copyable(ids[start]) {
                continue;
            }
            for end in start + 1..=(start + MAX_SPAN).min(clause_end) {
                if !copyable(ids[end - 1]) {
                    break;
                }
                if ends[end - fact[0] - 1] {
                    result.push((Span { start, end }, [clause_start, clause_end]));
                }
            }
        }
        clause_start = clause_end + 1;
    }
    result
}

fn derive_endpoints(
    span: Span,
    clause: [usize; 2],
    question: [usize; 2],
    whole_words: &[Word],
    legacy_query: &[Word],
    anchors: &[Site],
) -> Endpoints {
    let anchor = anchors
        .iter()
        .filter(|anchor| (clause[0]..clause[1]).contains(&anchor.source))
        .min_by_key(|anchor| {
            (
                Reverse(anchor.quality),
                span.start.abs_diff(anchor.source),
                anchor.source,
            )
        });
    let query_anchor = anchor.map(|anchor| &legacy_query[anchor.query]);
    let query = query_anchor
        .filter(|legacy| {
            whole_words
                .iter()
                .any(|word| word.tokens == legacy.tokens && word.spelling == legacy.spelling)
        })
        .map_or([None; 2], |word| {
            neighbors(whole_words, word.tokens, question)
        });
    Endpoints {
        source: neighbors(whole_words, span, clause),
        query,
        query_anchor: query_anchor.map(|word| [word.tokens.start, word.tokens.end]),
        source_anchor: anchor.map(|anchor| anchor.source),
        clause,
    }
}

fn neighbors(words: &[Word], focus: Span, bounds: [usize; 2]) -> [Option<usize>; 2] {
    let complete = |word: &&Word| word.tokens.start >= bounds[0] && word.tokens.end <= bounds[1];
    let left = words
        .iter()
        .rev()
        .filter(complete)
        .find(|word| word.tokens.end <= focus.start)
        .map(|word| word.tokens.end - 1);
    let right = words
        .iter()
        .filter(complete)
        .find(|word| word.tokens.start >= focus.end)
        .map(|word| word.tokens.start);
    [left, right]
}

fn key_layout(keys: &Keys, no_read: bool) -> bool {
    keys.iter().enumerate().all(|(bank, &key)| {
        key / BANK_WIDTH == bank as u16 && (key % BANK_WIDTH >= NO_READ_BASE) == no_read
    })
}

fn same_except_relation(old: &Keys, new: &Keys) -> bool {
    (0..BANKS).all(|bank| bank == RELATION_BANK || old[bank] == new[bank])
}

pub fn verify(
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
    old: &[&Tape],
    rows: &[&Tape],
    roots: &Roots,
    oracle: &RelationOracle,
) -> Result<()> {
    let user = opening.user_range();
    let expected_user = [user.start, user.end];
    let ids = opening.tokens();
    let decoded: Vec<_> = ids
        .iter()
        .map(|&id| tokenizer.decode_bytes(&[id]))
        .collect();
    let (fact, question) = bounds(&decoded, expected_user);
    let source_words = words(&decoded, fact)?;
    let query_words = words(&decoded, question)?;
    let original_words = words(&decoded, expected_user)?;
    let anchors = matched_anchors(&source_words, &query_words);
    let expected_candidates = candidates(&decoded, ids, fact, &source_words);
    check(
        old == &rows[..2],
        "pre-change legacy opening or keys differ",
    )?;
    for (index, row) in rows.iter().enumerate() {
        let new_arm = index >= 2;
        check(row.arm == ARMS[index], "capture arm order differs")?;
        check(
            row.user == expected_user && row.fact == fact && row.question == question,
            "original user, fact, or question role bounds differ",
        )?;
        check(
            key_layout(&row.no_read_keys, true),
            "invalid NoRead key address",
        )?;
        check(
            row.candidates.len() == expected_candidates.len(),
            "candidate list length differs",
        )?;
        check(
            new_arm == row.relation_endpoints.is_some() && new_arm == row.relation_tokens.is_some(),
            "unexpected relation metadata presence",
        )?;
        if let (Some(endpoints), Some(tokens)) = (&row.relation_endpoints, &row.relation_tokens) {
            check(
                row.no_read_keys == rows[0].no_read_keys,
                "NoRead keys changed",
            )?;
            check(
                endpoints.len() == expected_candidates.len(),
                "candidate endpoint list length differs",
            )?;
            check(
                tokens.len() == ids.len()
                    && tokens
                        .iter()
                        .zip(ids.iter().zip(&decoded))
                        .all(|(actual, (&id, bytes))| {
                            actual.id == id
                                && actual.bytes == *bytes
                                && Some(actual.root) == roots.block_2(id)
                        }),
                "original token IDs, bytes or roots malformed",
            )?;
            check(
                rows[2].relation_tokens.as_ref() == Some(tokens),
                "relation arms use different original token roots",
            )?;
        }
        for (candidate_index, (&(span, clause), observed)) in
            expected_candidates.iter().zip(&row.candidates).enumerate()
        {
            check(
                span == observed.span(),
                "candidate endpoints or order changed",
            )?;
            check(
                key_layout(&observed.keys, false),
                "invalid candidate key address",
            )?;
            if let Some(endpoints) = &row.relation_endpoints {
                let expected = derive_endpoints(
                    span,
                    clause,
                    question,
                    &original_words,
                    &query_words,
                    &anchors,
                );
                check(
                    endpoints[candidate_index] == expected,
                    "original lexical neighbor, anchor or clause differs",
                )?;
                check(
                    same_except_relation(&rows[0].candidates[candidate_index].keys, &observed.keys),
                    "a bank other than 10 changed",
                )?;
                let [ql, qr] = expected.query;
                let [sl, sr] = expected.source;
                let expected_slot = match [ql, qr, sl, sr] {
                    [Some(ql), Some(qr), Some(sl), Some(sr)] => {
                        let endpoint_ids = [ids[ql], ids[qr], ids[sl], ids[sr]];
                        if index == 3 {
                            RelationOracle::count_slot(endpoint_ids)
                        } else {
                            let [Some(ql), Some(qr), Some(sl), Some(sr)] =
                                endpoint_ids.map(|id| roots.block_2(id))
                            else {
                                return Err("missing artifact token root".into());
                            };
                            oracle.geometry_slot([ql, qr, sl, sr], index == 4)
                        }
                    }
                    _ => 0,
                };
                check(
                    observed.keys[RELATION_BANK] % BANK_WIDTH == u16::from(expected_slot),
                    "bank 10 address differs from independent endpoint replay",
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_for_tokens(
        tokens: &[&[u8]],
        clause: [usize; 2],
        question: [usize; 2],
    ) -> Result<Endpoints> {
        let tokens: Vec<_> = tokens.iter().map(|bytes| bytes.to_vec()).collect();
        let original = words(&tokens, [0, tokens.len()])?;
        let source = words(&tokens, clause)?;
        let legacy = words(&tokens, question)?;
        let anchors = matched_anchors(&source, &legacy);
        Ok(derive_endpoints(
            Span { start: 3, end: 4 },
            clause,
            question,
            &original,
            &legacy,
            &anchors,
        ))
    }

    #[test]
    fn clipped_left_neighbor_is_unavailable() -> Result<()> {
        let tokens = [
            b"Amber".as_slice(),
            b" Keeper",
            b" sent",
            b" Falcon",
            b" toward",
            b" cya",
            b"n",
            b" Keeper",
            b" asks",
            b"?",
        ];
        let endpoint = expected_for_tokens(&tokens, [0, 5], [6, 10])?;
        assert_eq!(endpoint.source, [Some(2), Some(4)]);
        assert_eq!(endpoint.query_anchor, Some([7, 8]));
        assert_eq!(endpoint.source_anchor, Some(1));
        assert_eq!(endpoint.query, [None, Some(8)]);
        Ok(())
    }

    #[test]
    fn clipped_legacy_anchor_keeps_its_identity_without_query_neighbors() -> Result<()> {
        let tokens = [
            b"Amber".as_slice(),
            b" Suffix",
            b" sent",
            b" Falcon",
            b" toward",
            b" Pre",
            b"Suffix",
            b" asks",
            b"?",
        ];
        let endpoint = expected_for_tokens(&tokens, [0, 5], [6, 9])?;
        assert_eq!(endpoint.query_anchor, Some([6, 7]));
        assert_eq!(endpoint.source_anchor, Some(1));
        assert_eq!(endpoint.query, [None; 2]);
        Ok(())
    }

    #[test]
    fn invalid_utf8_clipped_query_preserves_original_whole_words() -> Result<()> {
        let tokens = [
            b"Amber".as_slice(),
            b" Keeper",
            b" sent",
            b" Falcon",
            b" toward",
            b" \xF0\x90",
            b"\x90\xB7",
            b" Keeper",
            b" asks",
            b"?",
        ];
        let endpoint = expected_for_tokens(&tokens, [0, 5], [6, 10])?;
        let original = words(&tokens.map(|bytes| bytes.to_vec()), [0, 10])?;
        assert!(
            original
                .iter()
                .any(|word| word.tokens == Span { start: 5, end: 7 })
        );
        assert_eq!(endpoint.source, [Some(2), Some(4)]);
        assert_eq!(endpoint.query_anchor, None);
        assert_eq!(endpoint.source_anchor, None);
        assert_eq!(endpoint.query, [None; 2]);
        Ok(())
    }
}
