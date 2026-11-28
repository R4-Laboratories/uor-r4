use super::{add, require, Counts, Result, Turn, SPACE, WINDOW};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    ops::Range,
};
use uor_r4_tokenizer::ByteBpeTokenizer;

// Pinned against native_interval::MAX_SPAN and discrete_d11::ROLE_COPY_STOPS.
const SOURCE_CAP: usize = 8;
const COPY_STOPS: [u32; 4] = [16, 1, 0, 201];
const COMMON: &str = "a an and are as at be been but by can could did do does for from had has have he her here him his how i if in is it its may me my no not of on one or our out she so than that the their them then there these they this those to up us was we were what when where which who will with would yes you your";

#[derive(Default)]
pub(super) struct Census {
    pub counts: Counts,
    boundaries: Counts,
    turns: Counts,
    stops: Vec<Observation>,
    switches: Vec<(Observation, Span)>,
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct Span {
    token_start: usize,
    token_end: usize,
    byte_start: usize,
    byte_end: usize,
    ids: Vec<u32>,
}

#[derive(Serialize)]
struct Observation {
    record: String,
    train_ordinal: usize,
    turn: usize,
    user: String,
    answer: String,
    earlier_history: Vec<(String, String)>,
    earlier_ids: Vec<u32>,
    leading_space: usize,
    first: Span,
    supervised_eos: bool,
}

impl Observation {
    fn new(turn: &Turn<'_>, first: Span, leading_space: usize) -> Self {
        Self {
            record: turn.record.to_owned(),
            train_ordinal: turn.ordinal,
            turn: turn.index,
            user: turn.user.to_owned(),
            answer: turn.answer.to_owned(),
            earlier_history: turn
                .history
                .iter()
                .map(|message| (message.role.to_owned(), message.content.to_owned()))
                .collect(),
            earlier_ids: turn.history_ids.to_vec(),
            leading_space,
            first,
            supervised_eos: turn.terminal,
        }
    }

    fn join_key(&self) -> impl Ord + '_ {
        (
            &self.earlier_history,
            &self.earlier_ids,
            self.leading_space,
            &self.first,
            &self.user[..self.first.byte_end],
        )
    }
}

struct Source<'a> {
    ids: &'a [u32],
    spaced: String,
    original: &'a str,
    token_edges: Vec<usize>,
    original_edges: Vec<usize>,
    absolute: usize,
}

impl<'a> Source<'a> {
    fn new(
        tokenizer: &ByteBpeTokenizer,
        ids: &'a [u32],
        original: &'a str,
        absolute: usize,
    ) -> Result<Self> {
        let (spaced, original_edges) = normalize_with_offsets(original);
        let token_edges = token_edges(tokenizer, ids);
        require(
            tokenizer.decode_bytes(ids) == spaced.as_bytes()
                && token_edges.last() == Some(&spaced.len()),
            "source byte offsets differ from the tokenizer",
        )?;
        Ok(Self {
            ids,
            spaced,
            original,
            token_edges,
            original_edges,
            absolute,
        })
    }

    fn check(&self, wanted: &[u32]) -> Match {
        let mut occurrences = self
            .ids
            .windows(wanted.len())
            .enumerate()
            .filter(|(_, ids)| *ids == wanted)
            .map(|(start, _)| start);
        let Some(start) = occurrences.next() else {
            return Match::Absent;
        };
        if occurrences.next().is_some() {
            return Match::Ambiguous;
        }
        if wanted.iter().any(|id| COPY_STOPS.contains(id)) {
            return Match::WriterDelimiter;
        }
        let end = start + wanted.len();
        let bytes = &self.spaced.as_bytes()[self.token_edges[start]..self.token_edges[end]];
        let padding = bytes
            .iter()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        let lo = self.token_edges[start] + padding;
        let hi = self.token_edges[end];
        let Some(surface) = self.spaced.get(lo..hi) else {
            return Match::PartialWord;
        };
        if !surface.chars().any(char::is_alphanumeric) {
            return Match::NonContent;
        }
        if !word_edges(&self.spaced, lo..hi) {
            return Match::PartialWord;
        }
        let first = self.original_edges[lo - 1];
        let last = self.original_edges[hi - 1];
        if self.original.get(first..last) != Some(surface) {
            return Match::NormalizedOnly;
        }
        if self
            .spaced
            .char_indices()
            .filter(|(off, _)| {
                self.spaced[*off..].starts_with(surface)
                    && word_edges(&self.spaced, *off..*off + surface.len())
            })
            .nth(1)
            .is_some()
        {
            return Match::Ambiguous;
        }
        if wanted.len() == 1 && common_word(surface) {
            return Match::CommonWord;
        }
        Match::Unique(Span {
            token_start: self.absolute + start,
            token_end: self.absolute + end,
            byte_start: first,
            byte_end: last,
            ids: wanted.to_vec(),
        })
    }
}

enum Match {
    Absent,
    Ambiguous,
    WriterDelimiter,
    NonContent,
    PartialWord,
    NormalizedOnly,
    CommonWord,
    Unique(Span),
}

impl Match {
    fn reason(&self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Ambiguous => "ambiguous",
            Self::WriterDelimiter => "writer_delimiter",
            Self::NonContent => "noncontent",
            Self::PartialWord => "partial_word",
            Self::NormalizedOnly => "normalized_only",
            Self::CommonWord => "one_id_common_word",
            Self::Unique(_) => "unique_content",
        }
    }
}

fn common_word(surface: &str) -> bool {
    let key = surface.trim().to_ascii_lowercase();
    COMMON.split_ascii_whitespace().any(|word| word == key)
}

fn normalize_with_offsets(original: &str) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut edges = vec![0];
    let mut chars = original.char_indices().peekable();
    while let Some((start, character)) = chars.next() {
        if character == '\r' {
            let end = start + 1 + usize::from(chars.next_if(|(_, ch)| *ch == '\n').is_some());
            text.push('\n');
            edges.push(end);
        } else {
            text.push(character);
            edges.extend(start + 1..=start + character.len_utf8());
        }
    }
    let trimmed = text.trim_start();
    let lo = text.len() - trimmed.len();
    let hi = lo + trimmed.trim_end().len();
    (format!(" {}", &text[lo..hi]), edges[lo..=hi].to_vec())
}

fn word_edges(text: &str, bytes: Range<usize>) -> bool {
    let Some(surface) = text.get(bytes.clone()) else {
        return false;
    };
    let begins_word = surface.chars().next().is_some_and(char::is_alphanumeric);
    let ends_word = surface
        .chars()
        .next_back()
        .is_some_and(char::is_alphanumeric);
    (!begins_word
        || !text[..bytes.start]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric))
        && (!ends_word
            || !text[bytes.end..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric))
}

fn token_edges(tokenizer: &ByteBpeTokenizer, ids: &[u32]) -> Vec<usize> {
    let mut edges = vec![0];
    let mut offset = 0;
    for id in ids {
        offset += tokenizer.decode_bytes(&[*id]).len();
        edges.push(offset);
    }
    edges
}

fn content_tokens(tokenizer: &ByteBpeTokenizer, ids: &[u32]) -> Result<Vec<bool>> {
    let edges = token_edges(tokenizer, ids);
    let mut content = vec![false; ids.len()];
    // A byte-BPE character may span IDs. It becomes observable in the ID
    // containing its last byte, including a budget ending within that character.
    let bytes = tokenizer.decode_bytes(ids);
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(error) if error.error_len().is_none() => {
            std::str::from_utf8(&bytes[..error.valid_up_to()])?
        }
        Err(error) => return Err(error.into()),
    };
    for (offset, character) in text
        .char_indices()
        .filter(|(_, character)| character.is_alphanumeric())
    {
        let end = offset + character.len_utf8();
        let token = edges[1..].partition_point(|&edge| edge < end);
        content[token] = true;
    }
    Ok(content)
}

fn flag(seen: &mut BTreeSet<&'static str>, counts: &mut Counts, event: &'static str) {
    seen.insert(event);
    add(counts, event, 1);
}

impl Census {
    pub(super) fn scan(
        &mut self,
        tokenizer: &ByteBpeTokenizer,
        turn: &Turn<'_>,
        source_ids: &[u32],
    ) -> Result<()> {
        add(&mut self.counts, "openings", 1);
        add(
            &mut self.counts,
            "visible_answer_ids",
            turn.answer_ids.len(),
        );
        add(
            &mut self.counts,
            if turn.terminal {
                "supervised_eos"
            } else {
                "visible_history_eos"
            },
            1,
        );
        if turn.prefix.len() > WINDOW {
            add(&mut self.counts, "full_prefix_exceeds_192", 1);
            return Ok(());
        }
        add(&mut self.counts, "full_prefix_fits_192", 1);
        if turn.index > 0 {
            add(&mut self.counts, "later_full_prefix_fits_192", 1);
        }
        let room = WINDOW - turn.prefix.len();
        let space = usize::from(turn.answer_ids.first() == Some(&SPACE));
        let eos_fits = turn.answer_ids.len() < room;
        if eos_fits {
            add(
                &mut self.counts,
                if turn.terminal {
                    "supervised_eos_fits"
                } else {
                    "visible_history_eos_fits"
                },
                1,
            );
            if turn.answer_ids.len() - space > SOURCE_CAP {
                add(
                    &mut self.counts,
                    "visible_eos_fits_but_reply_over_copy_cap",
                    1,
                );
            } else if turn.terminal {
                add(
                    &mut self.counts,
                    "supervised_eos_within_copy_cap_upper_bound",
                    1,
                );
            }
        }
        let reply = &turn.answer_ids[..turn.answer_ids.len().min(room)];
        let content = content_tokens(tokenizer, reply)?;
        if content
            .get(space + 1..)
            .is_some_and(|later| later.contains(&true))
        {
            add(&mut self.counts, "internal_content_upper_bound", 1);
        }
        if reply.len() <= space {
            return Ok(());
        }
        let source = Source::new(tokenizer, source_ids, turn.user, turn.user_token_start)?;
        let mut observed = BTreeSet::new();
        for size in 1..=SOURCE_CAP.min(reply.len() - space) {
            let end = space + size;
            let matched = source.check(&reply[space..end]);
            if matches!(matched, Match::Absent) {
                continue;
            }
            flag(&mut observed, &mut self.boundaries, "raw_opening_id_match");
            if size == 1 {
                flag(&mut observed, &mut self.boundaries, "raw_one_id");
            }
            if end == turn.answer_ids.len() && eos_fits {
                let event = if turn.terminal {
                    "raw_supervised_stop"
                } else {
                    "raw_visible_history_stop"
                };
                flag(&mut observed, &mut self.boundaries, event);
            }
            if content[end..].contains(&true) {
                flag(&mut observed, &mut self.boundaries, "raw_internal_content");
            }
            flag(&mut observed, &mut self.boundaries, matched.reason());
            let Match::Unique(first) = matched else {
                continue;
            };
            if size == 1 {
                flag(&mut observed, &mut self.boundaries, "unique_one_id_content");
            }
            if end == turn.answer_ids.len() && eos_fits {
                if turn.terminal {
                    flag(
                        &mut observed,
                        &mut self.boundaries,
                        "unique_supervised_stop",
                    );
                    self.stops.push(Observation::new(turn, first, space));
                } else {
                    flag(
                        &mut observed,
                        &mut self.boundaries,
                        "unique_visible_history_stop",
                    );
                }
                continue;
            }
            if end == reply.len() {
                flag(
                    &mut observed,
                    &mut self.boundaries,
                    "unique_next_action_clipped",
                );
                continue;
            }
            let rest = &turn.answer_ids[end..];
            let next = if content[end] {
                "unique_internal_immediate_content"
            } else if content[end..].contains(&true) {
                "unique_internal_content_after_punctuation_or_spacing"
            } else if eos_fits {
                "unique_internal_punctuation_or_spacing_only"
            } else {
                "unique_internal_tail_content_not_observed"
            };
            flag(&mut observed, &mut self.boundaries, next);
            if content[end..].contains(&true) {
                flag(
                    &mut observed,
                    &mut self.boundaries,
                    "unique_internal_content",
                );
            }
            let source_end = first.token_end - source.absolute;
            if source.ids.get(source_end) == rest.first() {
                flag(
                    &mut observed,
                    &mut self.boundaries,
                    "source_next_id_contiguous",
                );
            }
            if eos_fits && rest.len() <= SOURCE_CAP {
                if let Match::Unique(second) = source.check(rest) {
                    if first.token_end == second.token_start {
                        flag(
                            &mut observed,
                            &mut self.boundaries,
                            "exact_contiguous_suffix",
                        );
                        self.switches
                            .push((Observation::new(turn, first, space), second));
                    } else if first.token_end < second.token_start
                        || second.token_end <= first.token_start
                    {
                        flag(&mut observed, &mut self.boundaries, "exact_disjoint_suffix");
                        self.switches
                            .push((Observation::new(turn, first, space), second));
                    } else {
                        flag(
                            &mut observed,
                            &mut self.boundaries,
                            "exact_overlapping_suffix",
                        );
                    }
                }
            }
        }
        for event in observed {
            add(&mut self.turns, event, 1);
        }
        Ok(())
    }

    pub(super) fn finish(&self, private: &mut impl Write) -> Result<Value> {
        let mut joins = BTreeMap::<_, Vec<&Observation>>::new();
        for stop in &self.stops {
            joins.entry(stop.join_key()).or_default().push(stop);
            writeln!(
                private,
                "{}",
                json!({"role": "stop_candidate", "first": stop})
            )?;
        }
        let mut possible = 0;
        let mut possible_contiguous = 0;
        for (continuation, second) in &self.switches {
            let adjacent = continuation.first.token_end == second.token_start;
            writeln!(
                private,
                "{}",
                json!({
                    "role": if adjacent { "contiguous_candidate" } else { "disjoint_candidate" },
                    "first": continuation,
                    "second": second,
                })
            )?;
            for stop in joins.get(&continuation.join_key()).into_iter().flatten() {
                if stop.user == continuation.user || stop.record == continuation.record {
                    continue;
                }
                if adjacent {
                    possible_contiguous += 1;
                } else {
                    possible += 1;
                }
                writeln!(
                    private,
                    "{}",
                    json!({
                        "role": "liberal_original_address_join",
                        "stop_record": stop.record,
                        "stop_turn": stop.turn,
                        "continue_record": continuation.record,
                        "continue_turn": continuation.turn,
                        "adjacent": adjacent,
                    })
                )?;
            }
        }
        Ok(json!({
            "boundary_counts": self.boundaries,
            "turn_counts": self.turns,
            "unique_stop_candidates": self.stops.len(),
            "suffix_candidates": self.switches.len(),
            "liberal_different_request_noncontiguous_pairs": possible,
            "liberal_different_request_contiguous_pairs": possible_contiguous,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};
    use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};

    #[test]
    fn preserves_raw_positions_across_trim_crlf_and_unicode() {
        let raw = "\tAlé\r\nBeta  ";
        let (text, edges) = normalize_with_offsets(raw);
        assert_eq!(text, " Alé\nBeta");
        assert_eq!(&raw[edges[0]..edges[4]], "Alé");
        assert_eq!(&raw[edges[4]..edges[5]], "\r\n");
        assert_eq!(&raw[edges[5]..edges[9]], "Beta");
    }

    #[test]
    fn word_boundaries_refuse_substrings_and_invalid_utf8_edges() {
        let source = " nβeta Beta omega";
        assert!(!word_edges(source, 2..5));
        assert!(!word_edges(source, 3..5));
        assert!(word_edges(source, 8..12));
        assert!(!word_edges(source, 8..11));
    }

    #[test]
    fn strong_source_checks_refuse_repeated_subword_and_common_hits() {
        let source = Source {
            ids: &[40, 41, 42, 43, 44, 45],
            spaced: " the Beta Betamax Beta omega".into(),
            original: "the Beta Betamax Beta omega",
            token_edges: vec![0, 4, 9, 14, 17, 22, 28],
            original_edges: (0..=27).collect(),
            absolute: 10,
        };
        assert!(matches!(source.check(&[40]), Match::CommonWord));
        assert!(matches!(source.check(&[41]), Match::Ambiguous));
        assert!(matches!(source.check(&[42]), Match::PartialWord));
        assert!(matches!(
            source.check(&[45]),
            Match::Unique(Span {
                token_start: 15,
                ..
            })
        ));
        assert!(matches!(source.check(&[16]), Match::Absent));
    }

    #[test]
    fn tokenizer_replay_accepts_two_actions_at_one_original_boundary() -> Result<()> {
        let path =
            env::var_os("UOR_RELEASE_TEST_TOKENIZER").ok_or("missing bound test tokenizer")?;
        let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(path)?)
            .ok_or("invalid bound test tokenizer")?;
        let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
        let fact = "Episode: 016; Resolution: 480p\n\n";
        let stop = format!("{fact}Return only the episode number");
        let more = format!("{fact}Return the episode number followed by the resolution");

        let scan = |census: &mut Census, record: &str, request: &str, answer: &str| -> Result<()> {
            let user = Message {
                role: "user",
                content: request,
            };
            let opening = encoder.encode_assistant_prefix(&[user.clone()]);
            let whole = encoder.encode_document(&[
                user,
                Message {
                    role: "assistant",
                    content: answer,
                },
            ]);
            let source = tokenizer.encode(&super::super::spaced(request));
            let end = opening.tokens.len() - super::super::MARKER.len();
            let start = end - source.len();
            let answer_ids = whole
                .tokens
                .strip_prefix(opening.tokens.as_slice())
                .and_then(|reply| reply.strip_suffix(&[super::super::EOS]))
                .ok_or("fixture protocol EOS")?;
            require(
                whole.tokens.len() <= WINDOW
                    && whole.response_mask.last() == Some(&1)
                    && opening.tokens[start..end] == source,
                "fixture exceeds EOS budget or lacks its original source",
            )?;
            census.scan(
                &tokenizer,
                &Turn {
                    record,
                    ordinal: 0,
                    index: 0,
                    user: request,
                    answer,
                    history: &[],
                    history_ids: &[0],
                    user_token_start: start,
                    prefix: &opening.tokens,
                    answer_ids,
                    terminal: true,
                },
                &source,
            )
        };

        let mut positive = Census::default();
        scan(&mut positive, "stop", &stop, "016")?;
        scan(&mut positive, "continue", &more, "016 480p")?;
        let mut private = Vec::new();
        let joined = positive.finish(&mut private)?;
        assert_eq!(joined["unique_stop_candidates"], 1, "{joined}");
        assert_eq!(joined["suffix_candidates"], 1, "{joined}");
        assert_eq!(
            joined["liberal_different_request_noncontiguous_pairs"], 1,
            "{joined}"
        );
        let first = &positive.stops[0].first;
        let second = &positive.switches[0].1;
        assert_eq!(&stop[first.byte_start..first.byte_end], "016");
        assert_eq!(&more[second.byte_start..second.byte_end], "480p");
        assert!(first.token_end < second.token_start);

        // An equal lexical value and compatible output do not repair an incorrect
        // original address, even when the source and requests are otherwise intact.
        positive.stops[0].first.token_start += 1;
        let wrong_address = positive.finish(&mut Vec::new())?;
        assert_eq!(
            wrong_address["liberal_different_request_noncontiguous_pairs"],
            0
        );

        let mut duplicate = Census::default();
        scan(&mut duplicate, "stop", &stop, "016")?;
        scan(&mut duplicate, "duplicate", &stop, "016 480p")?;
        let duplicate_request = duplicate.finish(&mut Vec::new())?;
        assert_eq!(duplicate_request["suffix_candidates"], 1);
        assert_eq!(
            duplicate_request["liberal_different_request_noncontiguous_pairs"],
            0
        );
        Ok(())
    }
}
