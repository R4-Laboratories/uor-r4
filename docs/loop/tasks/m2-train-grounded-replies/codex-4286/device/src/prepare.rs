use crate::{
    checked,
    packet::{self, Address, Edge, Input, Label, Row, Site, Span},
    rank, require, sha, Result, CODEC_SHA, FIT_SHA, MODEL_SHA,
};
use serde::Deserialize;
use std::{collections::BTreeSet, fs, path::Path};
use uor_r4_integer::{
    discrete_d11::IntegerDiscreteModel,
    role_action::{
        instruction::learned::{Learned, Occurrence, Query},
        Turn,
    },
};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol},
    ByteBpeTokenizer,
};

#[derive(Deserialize)]
struct Gold {
    affirmed: Span,
    excluded: Span,
    referenced: Span,
}

#[derive(Deserialize)]
struct History {
    prompt: Vec<u32>,
    ids: Vec<u32>,
    stop: String,
}

#[derive(Deserialize)]
struct Governor {
    bytes: Span,
    tokens: [u32; 2],
    native: Address,
    noun: Span,
    native_noun: Span,
}

#[derive(Deserialize)]
struct Governors {
    affirmative: Governor,
    excluded: Governor,
}

#[derive(Deserialize)]
struct Reference {
    case_id: String,
    pair_id: String,
    family: String,
    distance: String,
    admission: String,
    turns: [String; 2],
    first_prompt: Vec<u32>,
    second_suffix: Vec<u32>,
    structural_history: History,
    prompt: Vec<u32>,
    structural_prompt_sha256: String,
    gold_binding: Gold,
    native_scope: [Address; 3],
    neutral_rival_bytes: Span,
    neutral_rival: Address,
    governors: Governors,
}

struct Sources<'a> {
    model: &'a IntegerDiscreteModel,
    codec: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    protocol: DialogueProtocol,
    roots: Vec<u32>,
    period: u32,
    other_sentence_ends: [u32; 2],
}

fn raw_source(site: Occurrence) -> Result<Address> {
    require(
        site.bytes.start >= 1 && site.bytes.start < site.bytes.end,
        "bad native byte span",
    )?;
    Ok(Address {
        bytes: Span {
            start: site.bytes.start - 1,
            end: site.bytes.end - 1,
        },
        tokens: Span {
            start: site.tokens.start,
            end: site.tokens.end,
        },
    })
}

struct Tape<'a> {
    raw: &'a str,
    start: u32,
    bounds: Vec<usize>,
}

impl<'a> Tape<'a> {
    fn new(raw: &'a str, start: usize, ids: &[u32], codec: &ByteBpeTokenizer) -> Result<Self> {
        require(
            codec.decode_bytes(ids) == format!(" {raw}").as_bytes(),
            "current user tape changed",
        )?;
        let mut bounds = vec![0];
        let mut end = 0;
        for &id in ids {
            end += codec
                .token_byte_len(id)
                .ok_or("ID has no exact BPE bytes")?;
            bounds.push(end);
        }
        require(
            end == 1 + raw.len(),
            "BPE boundaries differ from authored bytes",
        )?;
        Ok(Self {
            raw,
            start: u32::try_from(start)?,
            bounds,
        })
    }

    fn address(&self, bytes: Span) -> Result<Address> {
        bytes.text(self.raw)?;
        let framed = bytes.frame()?;
        let low = framed.start as usize;
        let high = framed.end as usize;
        let start = self.bounds.partition_point(|&offset| offset <= low) - 1;
        let end = self.bounds.partition_point(|&offset| offset < high);
        let frame = format!(" {}", self.raw);
        let covered = frame.as_bytes();
        for outside in [
            &covered[self.bounds[start]..low],
            &covered[high..self.bounds[end]],
        ] {
            require(
                !std::str::from_utf8(outside)?
                    .chars()
                    .any(char::is_alphanumeric),
                "BPE address envelops another lexeme",
            )?;
        }
        Ok(Address {
            bytes,
            tokens: Span {
                start: self.start + u32::try_from(start)?,
                end: self.start + u32::try_from(end)?,
            },
        })
    }

    fn verify(&self, bytes: Span, declared: Address) -> Result<Address> {
        let addr = self.address(bytes)?;
        require(
            declared.bytes == addr.bytes.frame()? && declared.tokens == addr.tokens,
            "authored span does not match declared native address",
        )?;
        Ok(addr)
    }
}

impl<'a> Sources<'a> {
    fn new(model: &'a IntegerDiscreteModel, codec: &'a ByteBpeTokenizer) -> Result<Self> {
        let protocol = DialogueProtocol::literal_roles_v2(codec)?;
        let single = |text| match codec.encode(text).as_slice() {
            [id] => Ok(*id),
            _ => Err("sentence marker is not a single BPE token"),
        };
        Ok(Self {
            model,
            codec,
            encoder: protocol.bind(codec)?,
            protocol,
            roots: model.packed_codebook_roots(),
            period: single(".")?,
            other_sentence_ends: [single("?")?, single("!")?],
        })
    }

    fn stage(&self, row: Reference) -> Result<(Row, Label, [Input; 5])> {
        let mut first = vec![self.protocol.bos_id];
        first.extend(self.encoder.encode_user_prefix(&row.turns[0], false).tokens);
        let history = &row.structural_history;
        let eos = self.protocol.eos_id;
        require(
            first == history.prompt
                && first == row.first_prompt
                && history.stop == "eos"
                && history.ids.len() > 1
                && history.ids.last() == Some(&eos)
                && !history.ids[..history.ids.len() - 1].contains(&eos),
            "first-turn history chain changed",
        )?;
        let (suffix, user_range) = self
            .encoder
            .encode_user_prefix_with_range(&row.turns[1], true);
        let user_range = user_range.ok_or("missing current user range")?;
        require(
            suffix.tokens == row.second_suffix,
            "authored second frame changed",
        )?;
        let prefix = first.len() + history.ids.len();
        let user_content = prefix + user_range.start..prefix + user_range.end;
        let prompt = [&first[..], &history.ids, &suffix.tokens].concat();
        let prompt_bytes: Vec<_> = prompt.iter().flat_map(|id| id.to_le_bytes()).collect();
        require(
            prompt == row.prompt
                && sha(&prompt_bytes) == row.structural_prompt_sha256
                && sha(row.turns[1].as_bytes()) == row.case_id,
            "authored prompt identity changed",
        )?;
        let tape = Tape::new(
            &row.turns[1],
            user_content.start,
            &prompt[user_content.clone()],
            self.codec,
        )?;
        let turn = Turn {
            prompt: &prompt,
            user_content,
            period: self.period,
            other_sentence_ends: self.other_sentence_ends,
            eos,
        };
        let reader = self.model.role_action().ok_or("missing native reader")?;
        let query = Query::new(&turn, reader.signatures(), &self.roots, self.codec)?
            .ok_or("authored prompt is outside the native proposer domain")?;
        let mut sites = Vec::new();
        for place in query.scope_candidates() {
            let address = raw_source(place.source)?;
            require(
                tape.address(address.bytes)? == address,
                "native candidate BPE span changed",
            )?;
            sites.push(Site {
                address,
                point: place.point,
            });
        }
        let mut tuples = Vec::new();
        query.scope_triples(|a, e, h| tuples.push([a, e, h]));
        let geometry = Row { sites, tuples };

        let raw_gold = [
            row.gold_binding.affirmed,
            row.gold_binding.excluded,
            row.gold_binding.referenced,
        ];
        let [a, e, h] = row.native_scope;
        let gold = [
            tape.verify(raw_gold[0], a)?,
            tape.verify(raw_gold[1], e)?,
            tape.verify(raw_gold[2], h)?,
        ];
        let [a, e, h] = gold;
        let neutral = tape.verify(row.neutral_rival_bytes, row.neutral_rival)?;
        let heading = h.bytes.text(tape.raw)?;
        let owns = |x: Address| {
            Ok::<_, Box<dyn std::error::Error>>(
                x.bytes.text(tape.raw)?.eq_ignore_ascii_case(heading),
            )
        };
        let [has_a, has_e] = [owns(a)?, owns(e)?];
        require(
            has_a != has_e && owns(neutral)?,
            "neutral and original owners must share the heading spelling",
        )?;
        let other = if has_a { e } else { a };
        require(
            geometry.contains(gold)
                && geometry.contains([e, a, h])
                && geometry.contains([neutral, other, h])
                && geometry.contains([other, neutral, h])
                && gold
                    .iter()
                    .all(|p| p.bytes != neutral.bytes && p.tokens != neutral.tokens),
            "missing exact/exchanged/neutral proposal at its original addresses",
        )?;
        let sources = [row.governors.affirmative, row.governors.excluded];
        let mut verb = Vec::with_capacity(2);
        for (source, noun) in sources.into_iter().zip([a, e]) {
            let governor = tape.verify(source.bytes, source.native)?;
            let gap = rank::gap(governor.tokens, noun.tokens);
            require(
                source.noun == noun.bytes
                    && source.native_noun == noun.tokens
                    && Span::from_pair(source.tokens) == governor.tokens
                    && gold.iter().chain([&neutral]).all(|point| {
                        point.bytes.end <= governor.bytes.start
                            || governor.bytes.end <= point.bytes.start
                    })
                    && match row.distance.as_str() {
                        "near" => gap < 6,
                        "far" => gap >= 6,
                        _ => false,
                    },
                "signed governor address, distance, or noun witness changed",
            )?;
            verb.push(governor);
        }
        require(
            verb[0].bytes.end <= verb[1].bytes.start || verb[1].bytes.end <= verb[0].bytes.start,
            "signed verb spans overlap",
        )?;
        let signed = [verb[0].tokens, verb[1].tokens];
        let mut unordered = [a.tokens, e.tokens];
        unordered.sort_unstable();
        let [plus, minus] = signed;
        let exact = [
            Edge {
                governor: plus,
                noun: a.tokens,
            },
            Edge {
                governor: minus,
                noun: e.tokens,
            },
        ];
        let mut exchange = [
            Edge {
                noun: e.tokens,
                ..exact[0]
            },
            Edge {
                noun: a.tokens,
                ..exact[1]
            },
        ];
        let mut exact = exact;
        // Edge order follows source address so no list position encodes a sign.
        exact.sort_unstable_by_key(|edge| edge.governor);
        exchange.sort_unstable_by_key(|edge| edge.governor);
        let label = Label {
            case_id: row.case_id,
            pair_id: row.pair_id,
            family: row.family,
            distance: row.distance,
            admission: row.admission,
            prompt_sha256: row.structural_prompt_sha256,
            gold,
            neutral,
        };
        Ok((
            geometry,
            label,
            [
                Input::Native,
                Input::Governors(signed),
                Input::Nouns(unordered),
                Input::Links {
                    governors: signed,
                    edges: exact,
                },
                Input::Links {
                    governors: signed,
                    edges: exchange,
                },
            ],
        ))
    }
}

fn check_banks(model: &IntegerDiscreteModel, fit: &Learned) -> Result<()> {
    let embedded = model
        .role_action()
        .and_then(|r| r.bound_instruction())
        .ok_or("missing D11 role banks")?;
    require(
        embedded == fit,
        "GPU fit scope and action banks are not the exact fixed D11 bank words",
    )
}

pub fn verify_model_banks(artifact: &str, fit: &Learned) -> Result<()> {
    let model = IntegerDiscreteModel::from_bytes(&checked(artifact, MODEL_SHA)?)?;
    check_banks(&model, fit)
}

pub fn run(reference: &str, artifact: &str, codec: &str, fit: &str, out: &Path) -> Result<()> {
    let model = IntegerDiscreteModel::from_bytes(&checked(artifact, MODEL_SHA)?)?;
    check_banks(&model, &rank::pooled(&checked(fit, FIT_SHA)?)?)?;
    require(
        model.tokenizer_sha256() == Some(CODEC_SHA),
        "fixed D11 tokenizer binding changed",
    )?;
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked(codec, CODEC_SHA)?)
        .ok_or("invalid tokenizer")?;
    let source = Sources::new(&model, &codec)?;
    let input: Vec<Reference> = serde_json::from_slice(&fs::read(reference)?)?;
    let mut ids = BTreeSet::new();
    let (mut rows, mut notes, mut arms) = (Vec::new(), Vec::new(), Vec::new());
    for record in input {
        require(ids.insert(record.case_id.clone()), "duplicate case ID")?;
        let id = record.case_id.clone();
        let (row, label, inputs) = source
            .stage(record)
            .map_err(|e| format!("case {id}: {e}"))?;
        rows.push(row);
        notes.push(label);
        arms.push(inputs);
    }
    packet::store(out, &rows, &notes, &arms)
}
