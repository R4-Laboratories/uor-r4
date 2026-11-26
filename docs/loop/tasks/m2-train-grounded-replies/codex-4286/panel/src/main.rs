#![forbid(unsafe_code)]
#![recursion_limit = "256"]

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, fs, path::Path};
use uor_r4_integer::{
    discrete_d11::IntegerDiscreteModel,
    role_action::{
        instruction::{
            bound::Span,
            learned::{Occurrence, Query, TokenSpan},
        },
        Action, Turn,
    },
};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol},
    ByteBpeTokenizer,
};

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

const FIRST_USER: &str = "Hello.";
const FRESH_ROWS: usize = 256;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Design {
    wordings: [Wording; 4],
    facts: [(String, String, String); 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wording {
    family: String,
    reference: String,
    rival: String,
    neutral: String,
    near_before: String,
    near_after: String,
    far_before: String,
    far_after: String,
    authorize: String,
    decline: String,
    heading: String,
    order: String,
    ending: String,
    request_first_mentioned: bool,
}

impl Wording {
    fn scope(&self, noun: &str, take: bool, far: bool, governor_first: bool) -> String {
        let governor = if take { &self.authorize } else { &self.decline };
        let template = match (far, governor_first) {
            (false, true) => &self.near_before,
            (false, false) => &self.near_after,
            (true, true) => &self.far_before,
            (true, false) => &self.far_after,
        };
        template
            .replace("{noun}", noun)
            .replace("{governor}", governor)
    }

    fn command(&self, first: &str, second: &str, horizon: u32) -> (String, String) {
        let order = self
            .order
            .replace("{first}", first)
            .replace("{second}", second);
        let numeral = if horizon == 3 { "three" } else { "four" };
        (order, self.ending.replace("{count}", numeral))
    }
}

fn raw_span(source: Occurrence) -> Span {
    Span {
        start: source.bytes.start - 1,
        end: source.bytes.end - 1,
    }
}

#[derive(Deserialize)]
struct History {
    user: String,
    prompt_ids: Vec<u32>,
    reply_ids: Vec<u32>,
    stop: String,
    reply_text: String,
}

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write(path: &Path, value: &Value) -> Result<()> {
    let mut data = serde_json::to_vec(value)?;
    data.push(b'\n');
    fs::write(path, data)?;
    Ok(())
}

fn ids(value: &Value) -> Result<Vec<u32>> {
    Ok(serde_json::from_value(value.clone())?)
}

fn array(value: &Value) -> Result<&Vec<Value>> {
    value.as_array().ok_or_else(|| "missing array".into())
}

fn string(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| "missing string".into())
}

fn unique_index<T: PartialEq>(haystack: &[T], needle: &[T]) -> Result<usize> {
    check(!needle.is_empty(), "empty registered literal")?;
    let mut matches = haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, window)| (window == needle).then_some(index));
    let start = matches.next().ok_or("registered literal absent")?;
    check(matches.next().is_none(), "registered literal is ambiguous")?;
    Ok(start)
}

fn location(text: &str, clause: &str, needle: &str) -> Result<Span> {
    let clause_start = unique_index(text.as_bytes(), clause.as_bytes())?;
    let start = clause_start + unique_index(clause.as_bytes(), needle.as_bytes())?;
    Ok(Span {
        start: u32::try_from(start)?,
        end: u32::try_from(start + needle.len())?,
    })
}
fn fact_spans(
    codec: &ByteBpeTokenizer,
    suffix: &[u32],
    job: &str,
    home: &str,
) -> Result<Vec<Value>> {
    let mut facts = Vec::new();
    for (role, value, declaration, opener) in [
        (
            "job",
            job,
            format!("My current job is {job}."),
            " Your job is",
        ),
        (
            "home",
            home,
            format!("I currently live in {home}."),
            " You live in",
        ),
    ] {
        let body = codec.encode(&format!(" {value}."));
        let source = unique_index(suffix, &body)?;
        check(
            body.len() >= 3 && body.last() == Some(&16),
            "unusable original fact body",
        )?;
        facts.push(json!({"role":role,"value":value,"fact":declaration,
            "opener_text":opener,"opener":codec.encode(opener),"body":body,
            "source_start":source,"source_period":source+body.len()-1}));
    }
    check(
        facts[0]["body"][0] == facts[1]["body"][0] && facts[0]["body"][1] != facts[1]["body"][1],
        "the two fact bodies must share the first token and diverge at the second",
    )?;
    for (index, rival) in [(0, 1), (1, 0)] {
        let offset = facts[rival]["source_start"].clone();
        facts[index]["decoy_offsets"] = json!([offset]);
        facts[index]["rival_offset"] = offset;
        facts[index]["rival_next_id"] = facts[rival]["body"][1].clone();
    }
    Ok(facts)
}

fn literal_reply(
    codec: &ByteBpeTokenizer,
    facts: &[Value],
    first: usize,
    horizon: u32,
    eos: u32,
) -> Result<Value> {
    let mut literal = String::new();
    let mut tokens = Vec::new();
    let mut units = Vec::new();
    let mut decision = None;
    for i in 0..horizon {
        let mut fact = facts[(first + i as usize) % 2].clone();
        literal.push_str(string(&fact["opener_text"])?);
        literal.push_str(&format!(" {}.", string(&fact["value"])?));
        fact["output_start"] = json!(tokens.len());
        tokens.extend(ids(&fact["opener"])?);
        tokens.extend(ids(&fact["body"])?);
        units.push(fact);
        if i == 2 {
            decision = Some(tokens.len());
        }
    }
    check(
        codec.encode(&literal) == tokens,
        "literal tokenization differs from declared source units",
    )?;
    check(
        codec.decode(&tokens) == literal,
        "literal reply fails the tokenizer round trip",
    )?;
    tokens.push(eos);
    check(tokens.len() <= 48, "literal reply exceeds the ordinary cap")?;
    Ok(json!({
        "requested_role":facts[first]["role"],"requested_horizon":horizon,
        "pending":if horizon==3 {"stop"} else {"continue"},
        "expected_text":literal,"expected":tokens,"expected_stop":"eos","units":units,
        "decision_offset":decision
    }))
}

fn occurrences(
    query: &Query<'_>,
    prompt: &[u32],
    raw: &str,
    codec: &ByteBpeTokenizer,
    sources: [Span; 3],
    neutral: Span,
    owner: Span,
) -> Result<([Occurrence; 3], Occurrence, usize)> {
    let mut candidates = None;
    let mut neutral_candidate = None;
    let mut multiplicity = [0; 4];
    let other = if sources[0] == owner {
        sources[1]
    } else {
        sources[0]
    };
    let mut count = 0;
    query.scope_triples(|a, b, r| {
        let tuple = [a, b, r].map(|i| query.scope_candidates()[i].source);
        let bytes = tuple.map(raw_span);
        count += 1;
        if bytes == sources {
            multiplicity[0] += 1;
            candidates = Some(tuple);
        }
        if bytes == [sources[1], sources[0], sources[2]] {
            multiplicity[1] += 1;
        }
        if bytes == [neutral, other, sources[2]] {
            multiplicity[2] += 1;
            neutral_candidate = Some(tuple[0]);
        }
        if bytes == [other, neutral, sources[2]] {
            multiplicity[3] += 1;
        }
    });
    check(
        multiplicity == [1; 4],
        &format!("gold/exchange/neutral/exchanged-neutral tuples must be distinct and unique: {multiplicity:?}"),
    )?;
    let tuple = candidates.ok_or("unique native scope tuple missing")?;
    let neutral_candidate = neutral_candidate.ok_or("unique native neutral rival missing")?;
    for (occurrence, raw_range) in tuple
        .into_iter()
        .chain([neutral_candidate])
        .zip(sources.into_iter().chain([neutral]))
    {
        let surface = raw
            .get(raw_range.start as usize..raw_range.end as usize)
            .ok_or("raw oracle source outside prompt")?;
        let covered = codec.decode(
            prompt
                .get(occurrence.tokens.start as usize..occurrence.tokens.end as usize)
                .ok_or("native source outside prompt")?,
        );
        check(
            query.surface(occurrence.bytes) == Some(surface)
                && covered.trim_matches(|c: char| !c.is_alphanumeric()) == surface,
            "exact native scope source does not cover the oracle noun",
        )?;
    }
    let owner_index = usize::from(sources[1] == owner);
    check(
        query.surface(neutral_candidate.bytes) == query.surface(tuple[owner_index].bytes)
            && neutral_candidate != tuple[owner_index]
            && neutral_candidate.tokens != tuple[owner_index].tokens,
        "the neutral noun must share the referent's surface and retain a different address",
    )?;
    Ok((tuple, neutral_candidate, count))
}

fn native_action(query: &Query<'_>, reference: Occurrence, raw: Span) -> Result<Occurrence> {
    let expected = Span {
        start: raw.start + 1,
        end: raw.end + 1,
    };
    let mut matches = query
        .action_candidates(reference.bytes)?
        .into_iter()
        .filter(|candidate| candidate.source.bytes == expected);
    let source = matches
        .next()
        .ok_or("registered action excluded by native Query")?
        .source;
    check(matches.next().is_none(), "registered action proposed twice")?;
    Ok(source)
}

fn covering_tokens(codec: &ByteBpeTokenizer, turn: &Turn<'_>, bytes: Span) -> Result<[usize; 2]> {
    let mut next = 0;
    let mut start = None;
    for address in turn.user_content.clone() {
        let current = next;
        next += codec
            .token_byte_len(turn.prompt[address])
            .ok_or("unmapped user token")?;
        if current <= bytes.start as usize && (bytes.start as usize) < next {
            start = Some(address);
        }
        if current < bytes.end as usize && (bytes.end as usize) <= next {
            return Ok([
                start.ok_or("governor byte start outside user tape")?,
                address + 1,
            ]);
        }
    }
    Err("governor byte end outside user tape".into())
}

fn verify_copy(
    turn: &Turn<'_>,
    facts: &[Value],
    signatures: &[[u32; 2]],
    prefix_len: usize,
) -> Result<()> {
    for fact in facts {
        let source = prefix_len
            + fact["source_start"]
                .as_u64()
                .ok_or("missing source start")? as usize;
        let period = prefix_len + fact["source_period"].as_u64().ok_or("missing period")? as usize;
        let body = ids(&fact["body"])?;
        let matching = signatures
            .iter()
            .filter_map(|&signature| turn.resolve(Action::copy(signature)))
            .filter(|copy| copy.source as usize == source && copy.period as usize == period)
            .count();
        check(
            matching == 1 && turn.prompt.get(source..=period) == Some(body.as_slice()),
            "native writer declaration signature does not uniquely cover the original body",
        )?;
    }
    Ok(())
}

fn punctuation(codec: &ByteBpeTokenizer, text: &str) -> Result<u32> {
    match codec.encode(text).as_slice() {
        [token] => Ok(*token),
        _ => Err("punctuation isn't a single native token".into()),
    }
}

fn governor(
    codec: &ByteBpeTokenizer,
    turn: &Turn<'_>,
    bytes: Span,
    noun: Span,
    native_noun: Occurrence,
    far: bool,
    governor_first: bool,
) -> Result<Value> {
    let native_bytes = Span {
        start: bytes.start + 1,
        end: bytes.end + 1,
    };
    let [start, end] = covering_tokens(codec, turn, native_bytes)?;
    let raw_user = codec.decode(&turn.prompt[turn.user_content.clone()]);
    let declared = &raw_user[native_bytes.start as usize..native_bytes.end as usize];
    let covered = codec.decode(&turn.prompt[start..end]);
    check(
        covered.trim_matches(|ch: char| !ch.is_alphanumeric()) == declared,
        "covering governor tokens do not identify the exact original verb",
    )?;
    let native = Occurrence {
        bytes: native_bytes,
        tokens: TokenSpan {
            start: start.try_into()?,
            end: end.try_into()?,
        },
    };
    let noun_start = native_noun.tokens.start as usize;
    let noun_end = native_noun.tokens.end as usize;
    let gap = if governor_first {
        noun_start.checked_sub(end)
    } else {
        start.checked_sub(noun_end)
    }
    .ok_or("governor and noun overlap or occur in the wrong order")?;
    let visible = if governor_first {
        end.saturating_sub(start.max(noun_start.saturating_sub(6)))
    } else {
        end.min(noun_end + 6).saturating_sub(start)
    };
    check(
        if far {
            gap >= 8 && visible == 0
        } else {
            gap < 6 && visible > 0
        },
        &format!("governor visibility disagrees with distance: gap={gap}, visible={visible}"),
    )?;
    Ok(json!({"bytes":bytes,"tokens":[start,end],"native":native,
        "noun":noun,"native_noun":native_noun.tokens,"gap":gap,"visible_tokens":visible}))
}

fn generate(
    codec: &ByteBpeTokenizer,
    protocol: &DialogueProtocol,
    encoder: &DialogueEncoder<'_>,
    model: &IntegerDiscreteModel,
    history: &History,
    design: &Design,
) -> Result<Vec<Value>> {
    let mut first_prompt = vec![protocol.bos_id];
    first_prompt.extend(encoder.encode_user_prefix(FIRST_USER, false).tokens);
    let preface = [&history.prompt_ids[..], &history.reply_ids[..]].concat();
    check(
        history.user == FIRST_USER
            && history.prompt_ids == first_prompt
            && history.stop == "eos"
            && history.reply_ids.last() == Some(&protocol.eos_id)
            && history.reply_ids.len() > 1
            && !history.reply_ids[..history.reply_ids.len() - 1].contains(&protocol.eos_id)
            && codec.decode(&history.reply_ids[..history.reply_ids.len() - 1])
                == history.reply_text,
        "earlier response does not close the stock first-turn prompt",
    )?;
    let period = punctuation(codec, ".")?;
    let other_sentence_ends = [punctuation(codec, "?")?, punctuation(codec, "!")?];
    let roots = model.packed_codebook_roots();
    let reader = model
        .role_action()
        .ok_or("missing native role-action reader schema")?;
    let signatures = reader.signatures();
    let mut rows = Vec::new();
    for wording in &design.wordings {
        check(
            wording.authorize != wording.decline
                && wording.authorize.len() == wording.decline.len(),
            "governor alternatives must be distinct and keep noun byte slots",
        )?;
        for (fact_index, (fact, job, home)) in design.facts.iter().enumerate() {
            for cell in 0u8..32 {
                let take = cell & 1 != 0;
                let far = cell & 2 != 0;
                let neutral_before = cell & 4 != 0;
                let job_requested = cell & 8 != 0;
                let governor_first = cell & 16 != 0;
                let horizon = 3 + fact_index as u32;
                let (role, other, first) = if job_requested {
                    ("job", "home", 0)
                } else {
                    ("home", "job", 1)
                };
                let mentioned = if wording.request_first_mentioned {
                    role
                } else {
                    other
                };
                let reference_clause = wording.scope(&wording.reference, take, far, governor_first);
                let rival_clause = wording.scope(&wording.rival, !take, far, governor_first);
                let neutral = wording.neutral.replace("{noun}", &wording.reference);
                // The scope proposer requires the different noun to follow both
                // same-spelling occurrences for all four ordered tuples to exist.
                let first_sentence = if neutral_before {
                    format!("{neutral}, and {reference_clause}")
                } else {
                    format!("{reference_clause}, while {neutral}")
                };
                let heading = wording.heading.replace("{noun}", &wording.reference);
                let (opening, ending) = wording.command(role, other, horizon);
                let user = format!("My current job is {job}. I currently live in {home}. For this task, {first_sentence}. Separately, {rival_clause}. {heading} \"{opening}. {ending}.\"");
                check(
                    user.is_ascii() && !user.contains(':'),
                    "invalid user-text wire fixture",
                )?;
                let (suffix, user_range) = encoder.encode_user_prefix_with_range(&user, true);
                let user_range = user_range.ok_or("fresh current-user range missing")?;
                let facts = fact_spans(codec, &suffix.tokens, job, home)
                    .map_err(|problem| format!("{fact}: {problem}"))?;
                let literal = literal_reply(codec, &facts, first, horizon, protocol.eos_id)?;
                let prompt = [&preface[..], &suffix.tokens[..]].concat();
                let expected_tape = format!(" {user}");
                let turn = Turn {
                    prompt: &prompt,
                    user_content: (preface.len() + user_range.start)
                        ..(preface.len() + user_range.end),
                    period,
                    other_sentence_ends,
                    eos: protocol.eos_id,
                };
                check(
                    codec.decode(&prompt[turn.user_content.clone()]) == expected_tape,
                    "encoded tape is not the original user text",
                )?;
                let ref_owner = location(&user, &reference_clause, &wording.reference)?;
                let rival_owner = location(&user, &rival_clause, &wording.rival)?;
                let reference = location(&user, &heading, &wording.reference)?;
                let sources = if take {
                    [ref_owner, rival_owner, reference]
                } else {
                    [rival_owner, ref_owner, reference]
                };
                let term = |authorizes| {
                    if authorizes {
                        &wording.authorize
                    } else {
                        &wording.decline
                    }
                };
                let ref_governor = location(&user, &reference_clause, term(take))?;
                let rival_governor = location(&user, &rival_clause, term(!take))?;
                let query = Query::new(&turn, signatures, &roots, codec)?
                    .ok_or_else(|| format!("{} {fact} cell {cell}: native Query rejected {} current-user tokens after {} declaration tokens", wording.family, turn.user_content.len(), facts[1]["source_period"]))?;
                let neutral_bytes = location(&user, &neutral, &wording.reference)?;
                let (native_scope, neutral_rival, scope_tuple_count) = occurrences(
                    &query,
                    &prompt,
                    &user,
                    codec,
                    sources,
                    neutral_bytes,
                    ref_owner,
                )
                .map_err(|problem| format!("{} cell {cell}: {problem}", wording.family))?;
                let opening_bytes = location(&user, &opening, &opening)?;
                let ending_bytes = location(&user, &ending, &ending)?;
                let native_role = native_action(&query, native_scope[2], opening_bytes)?;
                let native_pending = native_action(&query, native_scope[2], ending_bytes)?;
                verify_copy(&turn, &facts, signatures, preface.len()).map_err(|problem| {
                    format!("{} {fact} cell {cell}: {problem}", wording.family)
                })?;
                let id = sha(user.as_bytes());
                let family = &wording.family;
                let group = format!("{family}/{fact}");
                let (native_owner, native_rival) = if take {
                    (native_scope[0], native_scope[1])
                } else {
                    (native_scope[1], native_scope[0])
                };
                let referent_governor = governor(
                    codec,
                    &turn,
                    ref_governor,
                    ref_owner,
                    native_owner,
                    far,
                    governor_first,
                )?;
                let other_governor = governor(
                    codec,
                    &turn,
                    rival_governor,
                    rival_owner,
                    native_rival,
                    far,
                    governor_first,
                )?;
                let (affirmative, excluded) = if take {
                    (&referent_governor, &other_governor)
                } else {
                    (&other_governor, &referent_governor)
                };
                let pair_id = format!("{group}/{}", cell & !1);
                rows.push(json!({
                    "id":id,"case_id":id,"cohort":"fresh","fold":8,"family":family,"wrapper":family,"fact":fact,"cell":cell,
                    "distance":if far {"far"} else {"near"},
                    "governor_order":if governor_first {"before"} else {"after"},
                    "placement_name":if neutral_before {"before"} else {"after"},
                    "admission":if take {"take"} else {"defer"},
                    "gold_clause_order":"referenced_first",
                    "referenced_mentioned_first":!neutral_before,
                    "requested_role":role,"requested_horizon":horizon,"role_mentioned_first":mentioned,
                    "pair_id":pair_id,"admission_pair":pair_id,
                    "distance_pair":format!("{group}/{}",cell & !2),
                    "neutral_pair":format!("{group}/{}",cell & !4),
                    "role_pair":format!("{group}/{}",cell & !8),
                    "order_pair":format!("{group}/{}",cell & !16),
                    "fact_horizon_pair":format!("{family}/{cell}"),
                    "turns":[FIRST_USER,user],"first_prompt":first_prompt,"second_suffix":suffix.tokens,"prompt":prompt,
                    "script":format!("scripts/{id}.json"),
                    "earlier":{"prompt":history.prompt_ids,"ids":history.reply_ids,"stop":history.stop,"text":history.reply_text},
                    "user_content":{"start":turn.user_content.start,"end":turn.user_content.end},
                    "fact_spans":facts,"effective_grounded":{"nominal":literal},
                    "role_phrase":opening_bytes,"pending_phrase":ending_bytes,
                    "gold_binding":{"affirmed":sources[0],"excluded":sources[1],"referenced":sources[2],
                        "first_role":role,"pending":literal["pending"]},
                    "scope_clauses":[reference_clause,rival_clause],
                    "structural_history":{"prompt":history.prompt_ids,"ids":history.reply_ids,"stop":history.stop,"text":history.reply_text},
                    "structural_prompt_sha256":sha(&prompt.iter().flat_map(|id|id.to_le_bytes()).collect::<Vec<_>>()),
                    "native_scope":native_scope,"native_actions":[native_role,native_pending],
                    "neutral_rival":neutral_rival,"neutral_rival_bytes":neutral_bytes,
                    "scope_tuple_count":scope_tuple_count,
                    "governors":{
                        "referent":referent_governor,"rival":other_governor,
                        "affirmative":affirmative,"excluded":excluded
                    }
                }));
            }
        }
    }
    check(rows.len() == FRESH_ROWS, "wrong fresh enrollment")?;
    Ok(rows)
}

fn mask_governors(text: &str, spans: &Value) -> Result<String> {
    let mut ranges: Vec<Span> = ["referent", "rival"]
        .into_iter()
        .map(|field| serde_json::from_value(spans[field]["bytes"].clone()))
        .collect::<std::result::Result<_, _>>()?;
    ranges.sort_unstable_by_key(|range| std::cmp::Reverse(range.start));
    let mut masked = text.to_owned();
    for span in ranges {
        masked.replace_range(span.start as usize..span.end as usize, "{}");
    }
    Ok(masked)
}

fn token_crossover(row: &Value, peer: &Value) -> Result<bool> {
    let prompt = ids(&row["prompt"])?;
    let opposed = ids(&peer["prompt"])?;
    if prompt.len() != opposed.len() {
        return Ok(false);
    }
    let mut governor_tokens = BTreeSet::new();
    for physical_site in ["referent", "rival"] {
        let expected: [usize; 2] =
            serde_json::from_value(row["governors"][physical_site]["tokens"].clone())?;
        let other: [usize; 2] =
            serde_json::from_value(peer["governors"][physical_site]["tokens"].clone())?;
        if expected != other || prompt[expected[0]..expected[1]] == opposed[other[0]..other[1]] {
            return Ok(false);
        }
        governor_tokens.extend(expected[0]..expected[1]);
    }
    Ok(prompt
        .iter()
        .zip(opposed)
        .enumerate()
        .all(|(offset, (&id, other))| id == other || governor_tokens.contains(&offset)))
}

fn fixture(rows: &[Value], eos: u32) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut families = BTreeSet::new();
    for block in rows.chunks_exact(32) {
        families.insert(string(&block[0]["family"])?);
        for (cell, row) in block.iter().enumerate() {
            let id = string(&row["id"])?;
            let text = string(&row["turns"][1])?;
            let peer = &block[cell ^ 1];
            let peer_text = string(&peer["turns"][1])?;
            let gold = &row["gold_binding"];
            let reverse = &peer["gold_binding"];
            check(
                row["cell"] == cell
                    && row["fold"] == 8
                    && ids.insert(id)
                    && sha(text.as_bytes()) == id,
                "duplicate or misidentified held row",
            )?;
            check(
                row["admission"] != peer["admission"]
                    && row["admission_pair"] == peer["admission_pair"]
                    && row["effective_grounded"] == peer["effective_grounded"]
                    && row["fact_spans"] == peer["fact_spans"]
                    && gold["affirmed"] == reverse["excluded"]
                    && gold["excluded"] == reverse["affirmed"]
                    && gold["referenced"] == reverse["referenced"]
                    && row["native_scope"][0] == peer["native_scope"][1]
                    && row["native_scope"][1] == peer["native_scope"][0]
                    && row["native_scope"][2] == peer["native_scope"][2]
                    && row["neutral_rival"] == peer["neutral_rival"]
                    && mask_governors(text, &row["governors"])?
                        == mask_governors(peer_text, &peer["governors"])?
                    && token_crossover(row, peer)?,
                &format!(
                    "{} cell {cell}: opposed pair changed a noun address or a non-governor byte",
                    row["family"]
                ),
            )?;
            let teacher = &row["effective_grounded"]["nominal"];
            check(
                array(&teacher["expected"])?.last() == Some(&json!(eos)),
                "literal reference missing terminal EOS",
            )?;
        }
    }
    check(
        ids.len() == FRESH_ROWS && families.len() == 4,
        "wrong number of distinct rows or families",
    )
}

fn emit(base: &Path, rows: &[Value]) -> Result<()> {
    fs::create_dir(base)?;
    for dir in [
        "scripts",
        "structural-controls",
        "structural-controls/correct",
        "structural-controls/scope-swap",
    ] {
        fs::create_dir(base.join(dir))?;
    }
    let mut enrollment = String::new();
    let mut jobs = String::new();
    for row in rows {
        let id = string(&row["id"])?;
        write(
            &base.join("scripts").join(format!("{id}.json")),
            &json!({
                "schema":"uor-r4.chat-script/1",
                "turns":[{"user":row["turns"][0]},{"user":row["turns"][1]}]
            }),
        )?;
        enrollment.push_str(&format!("{id}\n"));
        jobs.push_str(&format!("8\t{id}\n"));
        let mut sidecar = row["gold_binding"].clone();
        sidecar["prompt_sha256"] = row["structural_prompt_sha256"].clone();
        write(
            &base
                .join("structural-controls/correct")
                .join(format!("{id}.json")),
            &sidecar,
        )?;
        let affirm = sidecar["affirmed"].clone();
        sidecar["affirmed"] = sidecar["excluded"].clone();
        sidecar["excluded"] = affirm;
        write(
            &base
                .join("structural-controls/scope-swap")
                .join(format!("{id}.json")),
            &sidecar,
        )?;
    }
    fs::write(base.join("ids.txt"), enrollment)?;
    fs::write(base.join("jobs.tsv"), jobs)?;
    write(&base.join("reference.json"), &json!(rows))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [codec_path, artifact_path, history_path, design_path, output] = args.as_slice() else {
        return Err("usage: governor-links-panel CODEC ARTIFACT HISTORY GRAMMAR NEW_DIR".into());
    };
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(codec_path)?)
        .ok_or("invalid tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_version(&codec, 2)?;
    let encoder = protocol.bind(&codec)?;
    // Read the originating ABI's roots and copy signatures. Do not call inference.
    let model = IntegerDiscreteModel::load(Path::new(artifact_path))?;
    let history = serde_json::from_value(read(Path::new(history_path))?)?;
    let design = serde_json::from_value(read(Path::new(design_path))?)?;
    let rows = generate(&codec, &protocol, &encoder, &model, &history, &design)?;
    fixture(&rows, protocol.eos_id)?;
    emit(Path::new(output), &rows)?;
    println!("256 rows / 4 families / 128 exact governor-only pairs / 1,024 registered native tuples / 512 native actions / 512 original copy bodies");
    Ok(())
}
