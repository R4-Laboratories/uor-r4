use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
use uor_r4_tokenizer::{
    dialogue::{DialogueProtocol, Message},
    ByteBpeTokenizer,
};

const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const USER: [u32; 4] = [0, 55, 2728, 28];
const ASSISTANT: [u32; 6] = [201, 35, 560, 652, 714, 28];
const PERIOD: u32 = 16;
const EOS: u32 = 1;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
struct Source {
    pos: usize,
    id: u32,
    byte_start: usize,
    byte_end: usize,
}

#[derive(Serialize)]
struct Truth {
    case_id: String,
    group: &'static str,
    family: &'static str,
    fact_order: usize,
    question_target: String,
    fenced: bool,
    supported: bool,
    reply: String,
    prompt_ids: Vec<u32>,
    sources: Vec<Source>,
    rival: Option<Source>,
    same_id_rival: Option<Source>,
    expected_ids: Vec<u32>,
    period_step: Option<usize>,
    eos_id: u32,
}

#[derive(Serialize)]
struct Prompt<'a> {
    case_id: &'a str,
    prompt: &'a str,
}

struct Case {
    group: &'static str,
    family: &'static str,
    fact_order: usize,
    question_target: String,
    fenced: bool,
    facts: Vec<(String, Option<&'static str>)>,
    question: String,
    answer_facts: Vec<usize>,
    rival_fact: Option<usize>,
    same_id_fact: Option<usize>,
}

fn guard(ok: bool, message: impl FnOnce() -> String) -> Result<()> {
    if !ok {
        return Err(message().into());
    }
    Ok(())
}

fn write_jsonl(file: &mut impl Write, row: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *file, row)?;
    writeln!(file)?;
    Ok(())
}

fn single_reply(question: String) -> String {
    format!("{question} Give only the value followed by a period.")
}

fn order(first: usize, sides: [usize; 2]) -> [usize; 2] {
    [sides[first], sides[1 - first]]
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (group, family, fenced, facts, values, queries) in [
        (
            "fenced_prior_template",
            "museum_badge",
            true,
            [
                "Tessa wears the badge labeled gold.",
                "Nolan wears the badge labeled white.",
            ],
            ["gold", "white"],
            [
                "Tell me the badge label worn by the volunteer Tessa.",
                "Tell me the badge label worn by the volunteer Nolan.",
            ],
        ),
        (
            "fenced_unique",
            "delivery_crate",
            true,
            [
                "The Carter delivery has a crate marked blue.",
                "The Gomez delivery has a crate marked red.",
            ],
            ["blue", "red"],
            [
                "What mark identifies the crate for Carter?",
                "What mark identifies the crate for Gomez?",
            ],
        ),
        (
            "fenced_unique",
            "cafe_break",
            true,
            [
                "After loading the car, Layla ordered tea.",
                "After closing the shop, Hugo ordered water.",
            ],
            ["tea", "water"],
            ["What did Layla order?", "What did Hugo order?"],
        ),
        (
            "natural_twin",
            "delivery_crate",
            false,
            [
                "The Carter delivery has a crate marked blue.",
                "The Gomez delivery has a crate marked red.",
            ],
            ["blue", "red"],
            [
                "What mark identifies the crate for Carter?",
                "What mark identifies the crate for Gomez?",
            ],
        ),
        (
            "natural_supported",
            "breakfast",
            false,
            [
                "Mina brought bread to our picnic.",
                "Daniel showed up with milk.",
            ],
            ["bread", "milk"],
            [
                "What did Mina bring to the picnic?",
                "What did Daniel bring to the picnic?",
            ],
        ),
        (
            "duplicate_fenced",
            "cabinet_tag",
            true,
            [
                "The cabinet assigned to Rosa has a label reading green.",
                "The cabinet assigned to Oscar has a label reading green.",
            ],
            ["green", "green"],
            [
                "Which label is on Rosa's cabinet?",
                "Which label is on Oscar's cabinet?",
            ],
        ),
        (
            "duplicate_natural",
            "recycling_bin",
            false,
            [
                "Sofia put out a bin marked paper.",
                "Malik also put out a bin marked paper.",
            ],
            ["paper", "paper"],
            [
                "What does the mark on Sofia's bin say?",
                "What does the mark on Malik's bin say?",
            ],
        ),
    ] {
        for first in 0..2 {
            for (target, query) in queries.iter().enumerate() {
                let mut factual = Vec::new();
                if family == "museum_badge" {
                    factual.push((
                        "The curator registered two volunteer badges.".to_owned(),
                        None,
                    ));
                }
                let offset = factual.len();
                factual.extend(
                    order(first, [0, 1]).map(|side| (facts[side].to_owned(), Some(values[side]))),
                );
                let gold = offset + usize::from(target != first);
                let other = offset + usize::from(target == first);
                cases.push(Case {
                    group,
                    family,
                    fact_order: first,
                    question_target: target.to_string(),
                    fenced,
                    facts: factual,
                    question: single_reply((*query).to_owned()),
                    answer_facts: vec![gold],
                    rival_fact: Some(other),
                    same_id_fact: (values[0] == values[1]).then_some(other),
                });
            }
        }
    }

    let values = ["blue", "red", "green", "white"];
    let stations = ["North", "South", "East", "West"];
    for first in 0..2 {
        let physical = if first == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        for target in 0..2 {
            let second = 1 - target;
            for extended in [false, true] {
                let mut requested = vec![target];
                if extended {
                    requested.extend([second, 2, 3]);
                }
                let address = |side| if first == 0 { side } else { 3 - side };
                let first_station = stations[target];
                let second_station = stations[second];
                let question = if extended {
                    format!("Give the {first_station} code followed by a period. Then give the {second_station}, East, and West codes, in that order, separated by spaces.")
                } else {
                    format!("Give only the {first_station} code followed by a period.")
                };
                cases.push(Case {
                    group: if extended {
                        "release_more"
                    } else {
                        "release_stop"
                    },
                    family: "station_code",
                    fact_order: first,
                    question_target: target.to_string(),
                    fenced: true,
                    facts: physical
                        .iter()
                        .map(|&side| {
                            (
                                format!(
                                    "The {} station recorded code {}.",
                                    stations[side], values[side]
                                ),
                                Some(values[side]),
                            )
                        })
                        .collect(),
                    question,
                    answer_facts: requested.into_iter().map(address).collect(),
                    rival_fact: Some(address(second)),
                    same_id_fact: None,
                });
            }
        }
    }

    for (group, family, candidates, question) in [
        (
            "negated",
            "denied_color",
            [
                ("Daria's folder is not blue.", Some("blue")),
                ("Ruben's folder is red.", Some("red")),
            ],
            "What actual color is Daria's folder?",
        ),
        (
            "negated",
            "correction_without_replacement",
            [
                (
                    "The notice denies that Elena's box is green.",
                    Some("green"),
                ),
                (
                    "The notice confirms that Aiden's box is white.",
                    Some("white"),
                ),
            ],
            "What color does the notice confirm for Elena's box?",
        ),
        (
            "absent",
            "omitted_value",
            [
                ("Inez picked up a folder yesterday.", None),
                ("Marcos picked up the folder marked white.", Some("white")),
            ],
            "What mark was on the folder picked up by Inez?",
        ),
        (
            "absent",
            "third_subject",
            [
                ("Jenna received a parcel marked red.", Some("red")),
                ("Owen received a parcel marked white.", Some("white")),
            ],
            "What mark was on the parcel Tess received?",
        ),
        (
            "question_only",
            "unconfirmed_desk",
            [
                ("Did Iris take the desk marked gold?", Some("gold")),
                ("Did Jonah take the desk marked white?", Some("white")),
            ],
            "Which mark is known to be on the desk Iris took?",
        ),
        (
            "question_only",
            "conditional_delivery",
            [
                (
                    "If the gate is open, Felix might get the blue box.",
                    Some("blue"),
                ),
                ("Could Soren get the green box?", Some("green")),
            ],
            "What color box did Felix actually get?",
        ),
    ] {
        for first in 0..2 {
            cases.push(Case {
                group,
                family,
                fact_order: first,
                question_target: "unsupported".to_owned(),
                fenced: family != "conditional_delivery",
                facts: order(first, [0, 1])
                    .map(|side| (candidates[side].0.to_owned(), candidates[side].1))
                    .to_vec(),
                question: format!(
                    "{question} If the information is missing, say unknown followed by a period."
                ),
                answer_facts: vec![],
                rival_fact: None,
                same_id_fact: None,
            });
        }
    }
    cases
}

fn prepare(tokenizer: &ByteBpeTokenizer, case_id: String, case: Case) -> Result<(String, Truth)> {
    let mut prompt = String::new();
    let mut known_ranges = Vec::new();
    for (fact, value) in &case.facts {
        if !prompt.is_empty() {
            prompt.push(' ');
        }
        let range = if let Some(value) = value {
            let needle = format!(" {value}");
            let local_start = fact.find(&needle).ok_or("authored value absent")?;
            guard(fact.match_indices(&needle).count() == 1, || {
                format!("ambiguous label {needle} in {fact}")
            })?;
            let start = prompt.len() + local_start;
            Some((start, start + needle.len()))
        } else {
            None
        };
        known_ranges.push(range);
        prompt.push_str(fact);
    }
    prompt.push_str(if case.fenced { "\n\n" } else { " " });
    prompt.push_str(&case.question);

    let protocol = DialogueProtocol::literal_roles_version(tokenizer, 2)?;
    let prefix = protocol
        .bind(tokenizer)?
        .encode_assistant_prefix(&[Message {
            role: "user",
            content: &prompt,
        }])
        .tokens;
    let encoded_body = prefix
        .strip_prefix(&USER)
        .and_then(|ids| ids.strip_suffix(&ASSISTANT))
        .ok_or("unexpected protocol 2 framing")?;
    let mut spans = Vec::new();
    let mut serialized_body = vec![b' '];
    serialized_body.extend_from_slice(prompt.as_bytes());
    let mut offset = 0;
    for (position, &id) in encoded_body.iter().enumerate() {
        let bytes = tokenizer.decode_bytes(&[id]);
        let next = offset + bytes.len();
        guard(
            serialized_body.get(offset..next) == Some(bytes.as_slice()),
            || {
                format!(
                    "{case_id}: token {} diverged at content byte {offset}",
                    position + USER.len()
                )
            },
        )?;
        if offset > 0 {
            spans.push(Source {
                pos: position + USER.len(),
                id,
                byte_start: offset - 1,
                byte_end: next - 1,
            });
        }
        offset = next;
    }
    guard(offset == serialized_body.len(), || {
        format!("{case_id}: incomplete bytes")
    })?;
    let separators = encoded_body
        .windows(2)
        .filter(|ids| *ids == [201, 201])
        .count();
    guard(separators == usize::from(case.fenced), || {
        format!("{case_id}: bad fence cardinality")
    })?;

    let source = |fact_index: usize| -> Result<Source> {
        let (start, end) = known_ranges[fact_index].ok_or("missing source fact value")?;
        spans
            .iter()
            .find(|span| (span.byte_start, span.byte_end) == (start, end))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "{case_id}: source {} is not one whole ID in raw context",
                    &prompt[start..end]
                )
                .into()
            })
    };
    let sources = case
        .answer_facts
        .iter()
        .map(|&fact| source(fact))
        .collect::<Result<Vec<_>>>()?;
    let rival = case.rival_fact.map(source).transpose()?;
    let same_id_rival = case.same_id_fact.map(source).transpose()?;
    if let Some(gold) = sources.first() {
        let same_token = spans.iter().filter(|span| span.id == gold.id).count();
        guard(
            same_token == 1 + usize::from(same_id_rival.is_some())
                && same_id_rival
                    .as_ref()
                    .is_none_or(|rival| gold.pos != rival.pos && gold.id == rival.id),
            || format!("{case_id}: wrong whole-ID occurrence count or duplicate"),
        )?;
        if case.fenced {
            let span_end = USER.len()
                + encoded_body
                    .windows(2)
                    .position(|ids| ids == [201, 201])
                    .ok_or("missing fenced split")?;
            guard(sources.iter().all(|site| site.pos < span_end), || {
                format!("{case_id}: expected source beyond fence")
            })?;
        }
    }

    let supported = !sources.is_empty();
    let reply = if let Some(gold) = sources.first() {
        let mut reply = format!("{}.", &prompt[gold.byte_start..gold.byte_end]);
        for span in &sources[1..] {
            reply.push_str(&prompt[span.byte_start..span.byte_end]);
        }
        reply
    } else {
        " unknown.".to_owned()
    };
    let mut expected_ids = tokenizer.encode(&reply);
    let period_step = expected_ids.iter().position(|&id| id == PERIOD);
    guard(tokenizer.decode(&expected_ids) == reply, || {
        format!("{case_id}: reply fails tokenizer round trip")
    })?;
    guard(period_step.is_some(), || {
        format!("{case_id}: period is not an independent original ID")
    })?;
    if supported {
        let required: Vec<_> = std::iter::once(sources[0].id)
            .chain(std::iter::once(PERIOD))
            .chain(sources[1..].iter().map(|span| span.id))
            .collect();
        guard(expected_ids == required, || {
            format!("{case_id}: expected words use other IDs than the raw occurrences")
        })?;
    }
    expected_ids.push(EOS);

    Ok((
        prompt,
        Truth {
            case_id,
            group: case.group,
            family: case.family,
            fact_order: case.fact_order,
            question_target: case.question_target,
            fenced: case.fenced,
            supported,
            reply,
            prompt_ids: prefix,
            sources,
            rival,
            same_id_rival,
            expected_ids,
            period_step,
            eos_id: EOS,
        },
    ))
}

fn main() -> Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [tokenizer_path, output] = arguments.as_slice() else {
        return Err("usage: ordinary-source-panel TOKENIZER.json NEW_OUTPUT_DIR".into());
    };
    let bytes = fs::read(tokenizer_path)?;
    guard(format!("{:x}", Sha256::digest(&bytes)) == TOKENIZER, || {
        "unbound tokenizer digest".to_owned()
    })?;
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes).ok_or("invalid bound tokenizer")?;
    let mut verified = Vec::new();
    let mut all_inputs = HashSet::new();
    for (index, case) in cases().into_iter().enumerate() {
        let (prompt, truth) = prepare(&tokenizer, format!("c{:03}", index + 1), case)?;
        guard(all_inputs.insert(truth.prompt_ids.clone()), || {
            format!("duplicate complete prompt {}", truth.case_id)
        })?;
        guard(truth.prompt_ids.len() + 12 <= 192, || {
            format!("{} exceeds client generation window", truth.case_id)
        })?;
        verified.push((prompt, truth));
    }
    let output = Path::new(output);
    fs::create_dir(output)?;
    let mut prompts = BufWriter::new(File::create(output.join("prompts.jsonl"))?);
    let mut keys = BufWriter::new(File::create(output.join("truth.jsonl"))?);
    for (prompt, truth) in &verified {
        write_jsonl(
            &mut prompts,
            &Prompt {
                case_id: &truth.case_id,
                prompt,
            },
        )?;
        write_jsonl(&mut keys, truth)?;
    }
    prompts.flush()?;
    keys.flush()?;
    Ok(())
}
