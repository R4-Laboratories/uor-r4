//! Finite replay of original typed TRAIN source intervals.
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{self, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::{
    ROLE_COPY_STOPS, VerifiedEndpointCopy,
    native_copy::EOS,
    native_interval::MAX_SPAN,
    native_span_release::{
        Action, Authority, Claim, Commit, Event, Phase, Request, Session, Span, Step,
    },
};
use uor_r4_tokenizer::{
    ByteBpeTokenizer,
    dialogue::{DialogueProtocol, EncodedUserOpening},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const HANDOFF: u32 = 91;
const AFTER: u32 = 92;

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

fn pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        format!("{:x}", Sha256::digest(&bytes)) == sha,
        "input SHA-256 mismatch",
    )?;
    Ok(bytes)
}

#[derive(Deserialize)]
struct Census {
    cases: Vec<Case>,
    same_fact_inverse: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    case_id: String,
    tokens: usize,
    own: Address,
    own_original_bytes: [usize; 2],
    named_opponent: Option<Address>,
    opponent_original_bytes: Option<[usize; 2]>,
}

#[derive(Deserialize)]
struct Address {
    status: String,
    span: Option<Range>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct Range {
    start: u32,
    end: u32,
}

impl Range {
    fn indices(self) -> std::ops::Range<usize> {
        self.start as usize..self.end as usize
    }

    fn span(self, prompt: &[u32]) -> Span {
        Span {
            start: Some(self.start),
            end: Some(self.end),
            ids: prompt.get(self.indices()).map(<[u32]>::to_vec),
        }
    }
}

#[derive(Deserialize)]
struct Blind {
    case_id: String,
    text: String,
}

#[derive(Serialize)]
struct Trace {
    case_id: String,
    opening_tokens: usize,
    role: &'static str,
    original_bytes: [usize; 2],
    absolute: Range,
    expected_ids: Vec<u32>,
    fixed: Vec<Step>,
    variable_native: Vec<Step>,
    variable_stop: Vec<Step>,
}

fn request(prompt: &[u32], authorized: Range, claimed: Range, action: Action) -> Request {
    Request {
        authority: Some(Authority {
            prompt_ids: Some(prompt.to_vec()),
            first: Some(authorized.span(prompt)),
            second: None,
        }),
        claim: Some(Claim {
            first: Some(claimed.span(prompt)),
            second: None,
            first_action: Some(action),
            second_action: None,
        }),
    }
}

fn licensed_step(
    writer: &mut VerifiedEndpointCopy<'_>,
    prompt: &[u32],
    end: u32,
    ordinary: u32,
    action: Action,
) -> Result<Step> {
    let before = writer.state();
    let (exact, verified_end) = writer.write(ordinary)?;
    let (event, source_pos, source_id) = match (exact.copied, exact.release, verified_end) {
        (1, 0, false) => (Event::Copy, Some(before.cursor), Some(exact.exact_id)),
        (0, 1, false) => (
            Event::Delimiter,
            Some(before.cursor),
            Some(prompt[before.cursor as usize]),
        ),
        (0, 0, true) => (Event::VerifiedEnd, None, None),
        (0, 0, false) if before.active == 0 => (Event::Ordinary, None, None),
        _ => return Err("licensed cursor escaped its authorized range".into()),
    };
    let active = before.active != 0;
    let stopped = verified_end && action == Action::Stop;
    require(
        exact.state.words()
            == [
                exact.copied,
                before.source,
                before.cursor + exact.copied,
                before.prompt_end,
            ],
        "unexpected change to the exact persisted cursor words",
    )?;
    Ok(Step {
        phase: active.then_some(Phase::First),
        event,
        second_event: None,
        source_pos,
        source_id,
        ordinary_id: ordinary,
        committed_id: if stopped { EOS } else { exact.emitted },
        commit: if stopped {
            Commit::SuppliedStop
        } else if exact.copied == 1 {
            Commit::Copy
        } else {
            Commit::Ordinary
        },
        action: verified_end.then_some(action),
        start: active.then_some(before.source),
        cursor: active.then_some(before.cursor),
        end: active.then_some(end),
        prompt_end: before.prompt_end,
    })
}

fn execute(
    opening: &EncodedUserOpening,
    address: Range,
    ids: &[u32],
    variable: bool,
    action: Action,
) -> Result<Vec<Step>> {
    let request = request(opening.tokens(), address, address, action);
    let mut step: Box<dyn FnMut(u32) -> Result<Step> + '_> = if variable {
        let span = address.span(opening.tokens());
        let mut writer = VerifiedEndpointCopy::new(opening, opening.tokens(), &span, &span)?;
        Box::new(move |ordinary| {
            licensed_step(&mut writer, opening.tokens(), address.end, ordinary, action)
        })
    } else {
        let mut writer = Session::new(opening, &request)?;
        Box::new(move |ordinary| Ok(writer.step(ordinary)?))
    };
    let mut steps = Vec::new();
    let past_delimiter = ids
        .iter()
        .position(|id| ROLE_COPY_STOPS.contains(id))
        .unwrap_or(ids.len());
    let budget = if variable { ids.len() } else { MAX_SPAN };
    let copied = past_delimiter.min(budget);
    for (offset, &expected) in ids[..copied].iter().enumerate() {
        let step = step(EOS)?;
        let position = address.start + offset as u32;
        require(
            step.event == Event::Copy
                && step.commit == Commit::Copy
                && step.start == Some(address.start)
                && step.cursor == Some(position)
                && step.end == Some(address.end)
                && step.prompt_end as usize == opening.tokens().len()
                && step.source_pos == Some(position)
                && step.source_id == Some(expected)
                && step.committed_id == expected,
            "cursor did not commit the expected source ID and original address",
        )?;
        steps.push(step);
    }

    let end = step(HANDOFF)?;
    require(
        end.start == Some(address.start)
            && end.cursor == Some(address.start + copied as u32)
            && end.end == Some(address.end)
            && end.prompt_end as usize == opening.tokens().len(),
        "release cursor lost the original fence or endpoint",
    )?;
    if copied == ids.len() {
        let stop = action == Action::Stop;
        require(
            end.event == Event::VerifiedEnd
                && end.action == Some(action)
                && end.source_pos.is_none()
                && end.source_id.is_none()
                && end.committed_id == if stop { EOS } else { HANDOFF }
                && end.commit
                    == if stop {
                        Commit::SuppliedStop
                    } else {
                        Commit::Ordinary
                    },
            "verified end did not hand off without another source read",
        )?;
    } else if !variable && copied == MAX_SPAN {
        require(
            end.event == Event::Cap
                && end.source_pos.is_none()
                && end.source_id.is_none()
                && end.action.is_none()
                && end.committed_id == HANDOFF,
            "fixed writer did not preserve the eight-token cap",
        )?;
    } else {
        require(
            end.event == Event::Delimiter
                && end.source_pos == end.cursor
                && end.source_id == Some(ids[copied])
                && end.action.is_none()
                && end.committed_id == HANDOFF,
            "copy delimiter failed to release before the authorized end",
        )?;
    }
    steps.push(end);
    for ordinary in [AFTER, EOS] {
        let step = step(ordinary)?;
        require(
            step.event == Event::Ordinary
                && step.commit == Commit::Ordinary
                && step.source_pos.is_none()
                && step.source_id.is_none()
                && step.committed_id == ordinary,
            "released cursor read source again",
        )?;
        steps.push(step);
    }
    Ok(steps)
}

fn replay(
    tokenizer: &ByteBpeTokenizer,
    opening: &EncodedUserOpening,
    case: &Case,
    text: &str,
    role: &'static str,
    original_bytes: [usize; 2],
    address: &Address,
) -> Result<Trace> {
    let absolute = address.span.ok_or("census interval missing edges")?;
    let tokens = opening.tokens();
    require(
        tokens.len() == case.tokens && tokens.len() <= 512,
        "whole protocol opening length differs from census",
    )?;
    let ids = tokens
        .get(absolute.indices())
        .ok_or("census token edge out of bounds")?;
    let [start, end] = original_bytes;
    let source = text
        .as_bytes()
        .get(start..end)
        .ok_or("invalid source byte edge")?;
    let surface = tokenizer.decode_bytes(ids);
    require(
        surface == source || surface.strip_prefix(b" ") == Some(source),
        "original bytes and the source IDs at the archived coordinates differ",
    )?;

    let fixed = execute(opening, absolute, ids, false, Action::Native)?;
    let variable_native = execute(opening, absolute, ids, true, Action::Native)?;
    let variable_stop = execute(opening, absolute, ids, true, Action::Stop)?;
    Ok(Trace {
        case_id: case.case_id.clone(),
        opening_tokens: tokens.len(),
        role,
        original_bytes,
        absolute,
        expected_ids: ids.to_vec(),
        fixed,
        variable_native,
        variable_stop,
    })
}

fn verify_controls(tokenizer: &ByteBpeTokenizer) -> Result<()> {
    let encoder = DialogueProtocol::literal_roles_v2(tokenizer)?.bind(tokenizer)?;
    let opening = encoder
        .encode_single_user_opening(&"red ".repeat(220))
        .ok_or("invalid repeated-ID control opening")?;
    let prompt = opening.tokens();
    let user = opening.user_range();
    let start = user.start as u32 + 2;
    let authorized = Range {
        start,
        end: start + 12,
    };
    let shifted = Range {
        start: start + 1,
        end: authorized.end + 1,
    };
    let earlier = Range {
        start,
        end: authorized.end - 1,
    };
    let later = Range {
        start,
        end: authorized.end + 1,
    };
    let good = authorized.span(prompt);
    require(
        prompt.len() > 192
            && good.ids == shifted.span(prompt).ids
            && !prompt[authorized.indices()]
                .iter()
                .any(|id| ROLE_COPY_STOPS.contains(id)),
        "same-ID shifted or unchanged source controls are degenerate",
    )?;
    for altered in [shifted, earlier, later] {
        require(
            VerifiedEndpointCopy::new(&opening, prompt, &good, &altered.span(prompt)).is_err(),
            "claim with a different original address escaped fixed authority",
        )?;
    }
    let mut changed_prompt = prompt.to_vec();
    changed_prompt[200] ^= 1;
    require(
        VerifiedEndpointCopy::new(&opening, &changed_prompt, &good, &good).is_err(),
        "changed authority prompt after token 192 was accepted",
    )?;
    let mut wrong_ids = authorized.span(prompt);
    wrong_ids.ids.as_mut().ok_or("missing control IDs")?[0] ^= 1;
    require(
        VerifiedEndpointCopy::new(&opening, prompt, &wrong_ids, &wrong_ids).is_err(),
        "changed IDs in both claim and authority were accepted",
    )?;
    for illegal in [
        Range { start, end: start },
        Range {
            start,
            end: start - 1,
        },
        Range {
            start: user.start as u32 - 1,
            end: start,
        },
        Range {
            start: user.end as u32 - 1,
            end: user.end as u32 + 1,
        },
        Range {
            start,
            end: prompt.len() as u32 + 1,
        },
    ] {
        let span = illegal.span(prompt);
        require(
            VerifiedEndpointCopy::new(&opening, prompt, &span, &span).is_err(),
            "illegal source range was accepted",
        )?;
    }
    let oversized = encoder
        .encode_single_user_opening(&"red ".repeat(520))
        .ok_or("invalid over-capacity control opening")?;
    let oversized_span = authorized.span(oversized.tokens());
    require(
        oversized.tokens().len() > 512
            && VerifiedEndpointCopy::new(
                &oversized,
                oversized.tokens(),
                &oversized_span,
                &oversized_span,
            )
            .is_err(),
        "oversized opening was accepted",
    )?;

    let wrong = later.span(prompt);
    let mut coherent_wrong = VerifiedEndpointCopy::new(&opening, prompt, &wrong, &wrong)?;
    for position in later.start..later.end {
        let step = licensed_step(&mut coherent_wrong, prompt, later.end, EOS, Action::Native)?;
        require(
            step.source_pos == Some(position) && step.source_id == Some(prompt[position as usize]),
            "coherent wrong endpoint failed to follow its own provided source address",
        )?;
    }
    let release = licensed_step(
        &mut coherent_wrong,
        prompt,
        later.end,
        HANDOFF,
        Action::Native,
    )?;
    require(
        release.event == Event::VerifiedEnd && release.cursor == Some(later.end),
        "coherent wrong endpoint did not demonstrate overshoot of the original target",
    )?;

    let delimiter = encoder
        .encode_single_user_opening("red red red. blue blue blue")
        .ok_or("invalid delimiter control opening")?;
    let user = delimiter.user_range();
    let range = Range {
        start: user.start as u32,
        end: user.end as u32,
    };
    let source = &delimiter.tokens()[user];
    require(
        source
            .iter()
            .position(|id| ROLE_COPY_STOPS.contains(id))
            .is_some_and(|position| position > 0 && position + 1 < source.len()),
        "delimiter control lacks an interior source stop",
    )?;
    execute(&delimiter, range, source, true, Action::Stop)?;
    Ok(())
}

fn run(root: &Path, tokenizer_path: &Path, audit_path: &Path) -> Result<serde_json::Value> {
    let audit: Census = serde_json::from_slice(&pinned(
        audit_path,
        "1c713dc602681d887aa641b913b8cb9f92a96829701d170263500c1c94496e03",
    )?)?;
    let blind = pinned(
        &root.join("supply/blind.jsonl"),
        "54219a05ae0a1402ad09f92e5133e376adb73df577e5f7075c7db576f5d3a9ca",
    )?;
    let mut prompts = BTreeMap::new();
    for line in std::str::from_utf8(&blind)?.lines() {
        let row: Blind = serde_json::from_str(line)?;
        require(
            prompts.insert(row.case_id, row.text).is_none(),
            "repeated blind prompt",
        )?;
    }
    require(
        prompts.len() == 73
            && audit.cases.len() == 73
            && audit
                .cases
                .iter()
                .map(|row| &row.case_id)
                .collect::<BTreeSet<_>>()
                .len()
                == 73
            && audit.same_fact_inverse.len() == 2,
        "frozen roster or reversal pair differs",
    )?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&pinned(
        tokenizer_path,
        "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
    )?)
    .ok_or("invalid pinned tokenizer")?;
    verify_controls(&tokenizer)?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let opening = |case: &Case| {
        let text = prompts.get(&case.case_id).ok_or("missing blind prompt")?;
        let opening = encoder
            .encode_single_user_opening(text)
            .ok_or("invalid protocol opening")?;
        Ok::<_, Box<dyn std::error::Error>>((text, opening))
    };

    let mut over_width = Vec::new();
    for case in &audit.cases {
        if case.own.status == "over-eight-tokens" {
            let (text, opening) = opening(case)?;
            over_width.push(replay(
                &tokenizer,
                &opening,
                case,
                text,
                "own",
                case.own_original_bytes,
                &case.own,
            )?);
        }
    }
    require(over_width.len() == 23, "over-width case count differs")?;
    let mut reversal = Vec::new();
    for case in &audit.same_fact_inverse {
        let (text, opening) = opening(case)?;
        reversal.push(replay(
            &tokenizer,
            &opening,
            case,
            text,
            "own",
            case.own_original_bytes,
            &case.own,
        )?);
        reversal.push(replay(
            &tokenizer,
            &opening,
            case,
            text,
            "opposing",
            case.opponent_original_bytes
                .ok_or("missing reversal opposing byte edges")?,
            case.named_opponent
                .as_ref()
                .ok_or("missing reversal opposing token edges")?,
        )?);
    }
    require(
        reversal[0].expected_ids == reversal[3].expected_ids
            && reversal[1].expected_ids == reversal[2].expected_ids
            && reversal[0].expected_ids != reversal[1].expected_ids,
        "source IDs differ across the authenticated reversal",
    )?;
    Ok(serde_json::json!({
        "schema": "uor-r4/verified-endpoint-finite-replay/1",
        "finite_controls": "pass",
        "over_width": over_width,
        "reversal": reversal,
    }))
}

fn main() -> Result<()> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let [root, tokenizer, audit] = args.as_slice() else {
        return Err("expected VERIFIED_RESTORE TOKENIZER CENSUS_AUDIT".into());
    };
    let value = run(Path::new(root), Path::new(tokenizer), Path::new(audit))?;
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &value)?;
    writeln!(stdout)?;
    Ok(())
}
