//! Supplied exact-source and release controls for protocol-2 native serving.
//! Positions address the immutable original user opening. Source authorization
//! says where a value occurs. The separately supplied actions are interventions.

use serde::{Deserialize, Serialize};
use uor_r4_tokenizer::dialogue::EncodedUserOpening;

use super::{
    native_copy::EOS, native_interval, native_opening::SPACE, native_whole_turn::TOKENIZER_CID,
    ExactCopyState, ROLE_COPY_STOPS,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.verified-span-release/1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub authority: Option<Authority>,
    pub claim: Option<Claim>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Authority {
    pub prompt_ids: Option<Vec<u32>>,
    pub first: Option<Span>,
    pub second: Option<Span>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub first: Option<Span>,
    pub second: Option<Span>,
    pub first_action: Option<Action>,
    pub second_action: Option<TerminalAction>,
}

/// Exclusive original source endpoint and its exact token IDs.
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub ids: Option<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Native,
    Stop,
    Continue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalAction {
    Native,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    First,
    Second,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Copy,
    NativeSpace,
    OpeningMismatch,
    VerifiedEnd,
    Cap,
    Delimiter,
    Rejected,
    Ordinary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Commit {
    Ordinary,
    Copy,
    SuppliedStop,
}

#[derive(Debug, Serialize)]
pub struct Step {
    pub phase: Option<Phase>,
    pub event: Event,
    pub second_event: Option<Event>,
    /// Original source position observed at this step, including a delimiter.
    /// During same-step continuation, these two fields refer to the second span.
    pub source_pos: Option<u32>,
    pub source_id: Option<u32>,
    /// Ordinary writer result on the actual self-fed prefix, after any legacy
    /// period release that was eligible on the actual ordinary-writer history.
    pub ordinary_id: u32,
    pub committed_id: u32,
    pub commit: Commit,
    pub action: Option<Action>,
    pub start: Option<u32>,
    pub cursor: Option<u32>,
    pub end: Option<u32>,
    pub prompt_end: u32,
}

#[derive(Clone, Copy)]
struct VerifiedSpan {
    cursor: ExactCopyState,
    end: u32,
}

#[derive(Clone, Copy)]
struct Active {
    phase: Phase,
    span: Option<VerifiedSpan>,
}

pub struct Session<'a> {
    prompt: &'a [u32],
    prompt_end: u32,
    pending: Option<Active>,
    second: Option<VerifiedSpan>,
    first_action: Option<Action>,
    second_action: Option<Action>,
    leading_space: bool,
}

impl<'a> Session<'a> {
    /// Bind a user opening from the tokenizer that defines the writer's fixed token IDs.
    pub fn new(opening: &'a EncodedUserOpening, request: &Request) -> Result<Self> {
        if opening.tokenizer_cid() != TOKENIZER_CID {
            return Err(invalid(
                "verified span release requires the pinned tokenizer",
            ));
        }
        Self::bind(opening.tokens(), opening.user_range(), request)
    }

    /// Enter the same exact cursor path with an interval read from the bound
    /// opening. The experiment keeps the soundness checks at the write boundary.
    pub fn for_interval(
        opening: &'a EncodedUserOpening,
        span: native_interval::Span,
        leading_space: bool,
        supplied_stop: bool,
    ) -> Result<Self> {
        let user = opening.user_range();
        let start = usize::from(span.start);
        let end = usize::from(span.end);
        if opening.tokenizer_cid() != TOKENIZER_CID
            || start < user.start
            || start >= end
            || end > user.end
            || end - start > native_interval::MAX_SPAN
            || opening.tokens()[start..end]
                .iter()
                .any(|id| ROLE_COPY_STOPS.contains(id))
        {
            return Err(invalid("learned interval escapes the original copy window"));
        }
        let prompt = opening.tokens();
        let prompt_end = u32::try_from(prompt.len())
            .map_err(|_| invalid("interval original prompt exceeds u32 addresses"))?;
        Ok(Self {
            prompt,
            prompt_end,
            pending: Some(Active {
                phase: Phase::First,
                span: Some(VerifiedSpan {
                    cursor: ExactCopyState::new(Some(u32::from(span.start)), prompt_end),
                    end: u32::from(span.end),
                }),
            }),
            second: None,
            first_action: Some(if supplied_stop {
                Action::Stop
            } else {
                Action::Native
            }),
            second_action: None,
            leading_space,
        })
    }

    fn bind(
        prompt: &'a [u32],
        user_range: std::ops::Range<usize>,
        request: &Request,
    ) -> Result<Self> {
        let prompt_end = u32::try_from(prompt.len())
            .map_err(|_| invalid("verified span original prompt exceeds u32 addresses"))?;
        let authority = request
            .authority
            .as_ref()
            .filter(|proof| proof.prompt_ids.as_deref() == Some(prompt));
        let claim = request.claim.as_ref();
        let authorize = |expected: Option<&Span>, presented: Option<&Span>| {
            let span = expected.filter(|&span| Some(span) == presented)?;
            let (start, end) = (span.start?, span.end?);
            // Accept only exact, complete IDs from the captured user content.
            // The assistant framing and the changing causal window are excluded.
            (user_range.start <= start as usize
                && start < end
                && end as usize <= user_range.end
                && prompt.get(start as usize..end as usize) == Some(span.ids.as_deref()?))
            .then(|| VerifiedSpan {
                cursor: ExactCopyState::new(Some(start), prompt_end),
                end,
            })
        };
        let first = authorize(
            authority.and_then(|proof| proof.first.as_ref()),
            claim.and_then(|claim| claim.first.as_ref()),
        );
        let second = authorize(
            authority.and_then(|proof| proof.second.as_ref()),
            claim.and_then(|claim| claim.second.as_ref()),
        );
        Ok(Self {
            prompt,
            prompt_end,
            pending: Some(Active {
                phase: Phase::First,
                span: first,
            }),
            second,
            first_action: claim.and_then(|claim| claim.first_action),
            second_action: claim
                .and_then(|claim| claim.second_action)
                .map(|action| match action {
                    TerminalAction::Native => Action::Native,
                    TerminalAction::Stop => Action::Stop,
                }),
            leading_space: false,
        })
    }

    /// Choose the final token on the same decode step as the ordinary writer.
    /// The caller self-feeds exactly `committed_id` and closes at EOS.
    pub fn step(&mut self, ordinary_id: u32) -> Result<Step> {
        let active = self.pending;
        let span = active.and_then(|active| active.span);
        let mut step = Step {
            phase: active.map(|active| active.phase),
            event: Event::Ordinary,
            second_event: None,
            source_pos: None,
            source_id: None,
            ordinary_id,
            committed_id: ordinary_id,
            commit: Commit::Ordinary,
            action: None,
            start: span.map(|span| span.cursor.source),
            cursor: span.map(|span| span.cursor.cursor),
            end: span.map(|span| span.end),
            prompt_end: self.prompt_end,
        };
        let Some(active) = active else {
            return Ok(step);
        };
        if self.leading_space {
            self.leading_space = false;
            step.event = if ordinary_id == SPACE {
                Event::NativeSpace
            } else {
                self.pending = None;
                Event::OpeningMismatch
            };
            return Ok(step);
        }
        let (event, mut next) = self.advance(active, &mut step)?;
        step.event = event;
        if event == Event::VerifiedEnd {
            step.action = match active.phase {
                Phase::First => self.first_action,
                Phase::Second => self.second_action,
            };
            match step.action {
                Some(Action::Stop) => {
                    step.committed_id = EOS;
                    step.commit = Commit::SuppliedStop;
                }
                Some(Action::Continue) => {
                    let second = Active {
                        phase: Phase::Second,
                        span: self.second,
                    };
                    let (second_event, next_second) = self.advance(second, &mut step)?;
                    step.second_event = Some(second_event);
                    next = next_second;
                }
                Some(Action::Native) | None => {}
            }
        }
        self.pending = next;
        Ok(step)
    }

    fn advance(&self, active: Active, step: &mut Step) -> Result<(Event, Option<Active>)> {
        let Some(span) = active.span else {
            return Ok((Event::Rejected, None));
        };
        // Check the independently authenticated end before the legacy cap.
        // An exact eight-token span releases here without a ninth cursor read.
        if span.cursor.cursor == span.end {
            return Ok((Event::VerifiedEnd, None));
        }
        let exact = span
            .cursor
            .write(self.prompt, step.ordinary_id, ROLE_COPY_STOPS)?;
        match exact.release {
            0 => {
                step.source_pos = Some(span.cursor.cursor);
                step.source_id = Some(exact.exact_id);
                step.committed_id = exact.emitted;
                step.commit = Commit::Copy;
                let next = Active {
                    phase: active.phase,
                    span: Some(VerifiedSpan {
                        cursor: exact.state,
                        end: span.end,
                    }),
                };
                Ok((Event::Copy, Some(next)))
            }
            1 => {
                step.source_pos = Some(span.cursor.cursor);
                step.source_id = self.prompt.get(span.cursor.cursor as usize).copied();
                Ok((Event::Delimiter, None))
            }
            3 => Ok((Event::Cap, None)),
            _ => Err(invalid("verified span cursor escaped the original prompt")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use uor_r4_tokenizer::{dialogue::DialogueProtocol, ByteBpeTokenizer};

    const ORDINARY: u32 = 91;
    const PROMPT: [u32; 23] = [
        0, 101, 30, 31, 32, 33, 34, 35, 36, 37, 38, 30, 31, 32, 33, 34, 35, 36, 37, 38, 16, 39, 200,
    ];

    fn source(start: usize, end: usize) -> Value {
        json!({"start": start, "end": end, "ids": PROMPT[start..end]})
    }

    fn request(first: Value, second: Value, action: &str) -> Value {
        json!({
            "authority": {"prompt_ids": PROMPT, "first": first, "second": second},
            "claim": {"first": first, "second": second, "first_action": action, "second_action": "stop"},
        })
    }

    fn session(control: Value) -> Session<'static> {
        let request = serde_json::from_value(control).expect("diagnostic JSON");
        Session::bind(&PROMPT, 2..22, &request).expect("original prompt")
    }

    #[test]
    fn rejects_an_alternate_tokenizer_with_a_different_eos_id() {
        let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(
            br#"{
                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                "added_tokens": [
                    {"id": 2, "content": "<|bos|>"},
                    {"id": 3, "content": "<|eos|>"},
                    {"id": 4, "content": "<|unk|>"},
                    {"id": 5, "content": "User:"},
                    {"id": 6, "content": "\n"},
                    {"id": 7, "content": "Assistant:"},
                    {"id": 8, "content": " x"}
                ]
            }"#,
        )
        .expect("valid byte BPE tokenizer");
        let protocol = DialogueProtocol::literal_roles_v2(&tokenizer).expect("valid protocol 2");
        assert_ne!(protocol.eos_id, EOS);
        let opening = protocol
            .bind(&tokenizer)
            .expect("bound protocol")
            .encode_single_user_opening("x")
            .expect("valid user opening");
        let range = opening.user_range();
        assert_eq!(opening.tokens()[range.clone()], [8]);
        let span = json!({"start": range.start, "end": range.end, "ids": [8]});
        let request = serde_json::from_value(json!({
            "authority": {"prompt_ids": opening.tokens(), "first": span},
            "claim": {"first": span, "first_action": "stop"},
        }))
        .expect("complete stop request");
        assert!(matches!(
            Session::new(&opening, &request),
            Err(crate::IntegerError::Invalid(message))
                if message == "verified span release requires the pinned tokenizer"
        ));
    }

    #[test]
    fn distinguishes_verified_end_at_seven_or_eight_from_the_cap_at_nine() {
        for length in [7, 8, 9] {
            let mut session = session(request(source(2, 2 + length), Value::Null, "stop"));
            for (offset, &expected) in PROMPT[2..2 + length.min(8)].iter().enumerate() {
                let step = session.step(EOS).expect("copy over ordinary EOS");
                assert_eq!(step.event, Event::Copy);
                assert_eq!(step.commit, Commit::Copy);
                assert_eq!(step.source_pos, Some(2 + offset as u32));
                assert_eq!(step.source_id, Some(expected));
                assert_eq!(step.committed_id, expected);
            }
            let release = session.step(ORDINARY).expect("release");
            assert_eq!(release.phase, Some(Phase::First));
            assert_eq!(release.source_pos, None);
            assert_eq!(release.source_id, None);
            if length == 9 {
                assert_eq!(release.event, Event::Cap);
                assert_eq!(release.action, None);
                assert_eq!(release.committed_id, ORDINARY);
                let fallback = session.step(ORDINARY + 1).expect("ordinary self-feed");
                assert_eq!(fallback.event, Event::Ordinary);
                assert_eq!(fallback.committed_id, ORDINARY + 1);
            } else {
                assert_eq!(release.event, Event::VerifiedEnd);
                assert_eq!(release.cursor, release.end);
                assert_eq!(release.commit, Commit::SuppliedStop);
                assert_eq!(release.committed_id, EOS);
            }
        }
    }

    #[test]
    fn continues_on_the_release_step_and_requires_separate_terminal_action() {
        for supplied_terminal_stop in [true, false] {
            let mut control = request(source(2, 3), source(11, 19), "continue");
            if !supplied_terminal_stop {
                control["claim"]["second_action"] = Value::Null;
            }
            let mut session = session(control);
            assert_eq!(session.step(ORDINARY).unwrap().source_pos, Some(2));
            let continued = session.step(ORDINARY).unwrap();
            assert_eq!(continued.event, Event::VerifiedEnd);
            assert_eq!(continued.second_event, Some(Event::Copy));
            assert_eq!(continued.source_pos, Some(11));
            assert_eq!(continued.committed_id, PROMPT[11]);
            for position in 12..19 {
                let next = session.step(ORDINARY).unwrap();
                assert_eq!(next.phase, Some(Phase::Second));
                assert_eq!(next.source_pos, Some(position));
                assert_eq!(next.committed_id, PROMPT[position as usize]);
            }
            let end = session.step(ORDINARY).unwrap();
            assert_eq!(end.event, Event::VerifiedEnd);
            assert_eq!(end.phase, Some(Phase::Second));
            assert_eq!(end.source_pos, None);
            assert_eq!(
                end.committed_id,
                if supplied_terminal_stop {
                    EOS
                } else {
                    ORDINARY
                }
            );
        }
    }

    #[test]
    fn refuses_wrong_same_id_address_early_late_and_incomplete_claims() {
        let good = request(source(2, 9), Value::Null, "stop");
        let changed_claims = [
            source(11, 18),
            source(2, 8),
            source(2, 10),
            json!({"start": 2}),
        ];
        for changed in changed_claims {
            let mut control = good.clone();
            control["claim"]["first"] = changed;
            let step = session(control).step(ORDINARY).unwrap();
            assert_eq!(step.event, Event::Rejected);
            assert_eq!(step.commit, Commit::Ordinary);
            assert_eq!(step.source_pos, None);
            assert_eq!(step.committed_id, ORDINARY);
        }
        let mut wrong_prompt = good;
        wrong_prompt["authority"]["prompt_ids"][22] = json!(201);
        assert_eq!(
            session(wrong_prompt).step(ORDINARY).unwrap().event,
            Event::Rejected
        );
        let mut wrong_ids = request(source(2, 3), Value::Null, "stop");
        wrong_ids["authority"]["first"]["ids"][0] = json!(51);
        wrong_ids["claim"]["first"]["ids"][0] = json!(51);
        assert_eq!(
            session(wrong_ids).step(ORDINARY).unwrap().event,
            Event::Rejected
        );
        let framing = request(source(22, 23), Value::Null, "stop");
        assert_eq!(
            session(framing).step(ORDINARY).unwrap().event,
            Event::Rejected
        );
    }

    #[test]
    fn incorrect_second_proof_cannot_borrow_the_verified_first_endpoint() {
        let mut control = request(source(2, 3), source(11, 12), "continue");
        control["claim"]["second"] = source(2, 3);
        let mut session = session(control);
        assert_eq!(session.step(ORDINARY).unwrap().committed_id, PROMPT[2]);
        let end = session.step(ORDINARY).unwrap();
        assert_eq!(end.event, Event::VerifiedEnd);
        assert_eq!(end.second_event, Some(Event::Rejected));
        assert_eq!(end.commit, Commit::Ordinary);
        assert_eq!(end.committed_id, ORDINARY);
        assert_eq!(
            session.step(ORDINARY + 1).unwrap().committed_id,
            ORDINARY + 1
        );
    }

    #[test]
    fn native_action_uses_the_ordinary_id_at_the_verified_endpoint() {
        for writer_id in [ORDINARY, EOS] {
            let mut session = session(request(source(2, 3), source(11, 12), "native"));
            assert_eq!(session.step(ORDINARY).unwrap().committed_id, PROMPT[2]);
            let end = session.step(writer_id).unwrap();
            assert_eq!(end.event, Event::VerifiedEnd);
            assert_eq!(end.action, Some(Action::Native));
            assert_eq!(end.second_event, None);
            assert_eq!(end.commit, Commit::Ordinary);
            assert_eq!(end.committed_id, writer_id);
        }
    }

    #[test]
    fn delimiter_inside_an_authorized_span_keeps_ordinary_commit() {
        let mut session = session(request(source(19, 22), Value::Null, "stop"));
        assert_eq!(session.step(ORDINARY).unwrap().committed_id, PROMPT[19]);
        let delimiter = session.step(ORDINARY).unwrap();
        assert_eq!(delimiter.event, Event::Delimiter);
        assert_eq!(delimiter.source_pos, Some(20));
        assert_eq!(delimiter.source_id, Some(16));
        assert_eq!(delimiter.action, None);
        assert_eq!(delimiter.committed_id, ORDINARY);
    }
}
