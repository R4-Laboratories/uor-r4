#![forbid(unsafe_code)]
#[path = "../../../codex-4363/pairs/src/dialogue.rs"]
mod dialogue;

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    ops::Range,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const CONTEXT: usize = 2048;
const CHUNK: usize = 257;
const CAP: usize = 1088;
const TRAIN: &str = "a268cd3fed9f327b3f7c1a105a3341b0a08513d1328ab72d5e3ec44d0896bbd7";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const ARTIFACT: &str = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";

#[derive(Deserialize)]
struct Pair {
    pair: usize,
    a: Side,
    b: Side,
}
#[derive(Deserialize)]
struct Side {
    component_sha256: String,
    literal_question_sha256: String,
    original_article_title: String,
    original_section_title: String,
    original_dialogue_id: String,
    original_target_turn_id: String,
    original_target: String,
    original_questions: Vec<String>,
    original_target_support_scalar_start: usize,
    original_target_support_bytes: Range<usize>,
    role_messages: Vec<dialogue::RoleText>,
    turn: usize,
    prefix: Vec<u16>,
    target: Vec<u16>,
    first_content_id: u16,
    stratum: String,
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn rank(purpose: &str, payload: &str) -> [u8; 32] {
    Sha256::digest(format!("uor-r4-quac-train-intake-v1\0{purpose}\0{payload}")).into()
}
fn pinned(path: &str, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        sha(&bytes) == expected,
        "pinned intake source digest differs",
    )?;
    Ok(bytes)
}
fn normalized(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
fn permutation(bytes: &[u8]) -> Result<Vec<u16>> {
    require(
        bytes.get(..16) == Some(b"UORBFY03\x01\0\0\0\0\x10\0\0"),
        "invalid pinned UORBFY03 layout",
    )?;
    let table = bytes
        .get(60..60 + 8192)
        .ok_or("missing static vocabulary permutation")?;
    let slots: Vec<_> = table
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    require(
        slots.iter().copied().collect::<BTreeSet<_>>() == (0..4096).collect(),
        "static label table is not a bijection",
    )?;
    Ok(slots)
}
fn pair_key(pair: &Pair, slots: &[u16]) -> (bool, [u8; 32]) {
    let same_cell = slots[usize::from(pair.a.first_content_id)] / 64
        == slots[usize::from(pair.b.first_content_id)] / 64;
    let hash = rank(
        "pair",
        &format!(
            "{}\0{}\0{}",
            pair.pair, pair.a.component_sha256, pair.b.component_sha256
        ),
    );
    (same_cell, hash)
}
fn family(pair: &Pair) -> Result<String> {
    let a = pair
        .a
        .original_questions
        .last()
        .ok_or("missing target question")?;
    let b = pair
        .b
        .original_questions
        .last()
        .ok_or("missing target question")?;
    require(
        a == b
            && pair.a.first_content_id != pair.b.first_content_id
            && pair.a.first_content_id < 4096
            && pair.b.first_content_id < 4096,
        "graph-first TRAIN pair has mismatched literal question or opening IDs",
    )?;
    Ok(normalized(a))
}
fn third(pair: &Pair) -> bool {
    pair.a.turn == 3 && pair.b.turn == 3
}

fn loader_valid(prefix: &[u16], target: &[u16], stratum: &str) -> bool {
    let total = prefix.len() + target.len();
    prefix.first() == Some(&0)
        && prefix.iter().skip(1).all(|&id| id != 0 && id < 4096)
        && prefix.iter().filter(|&&id| id == 1).count() == 2
        && target.len() >= 2
        && target.last() == Some(&1)
        && target[..target.len() - 1]
            .iter()
            .all(|&id| id > 1 && id < 4096)
        && total <= CAP
        && matches!(stratum, "near" | "far" | "intermediate")
        && prefix.len() % CHUNK != 0
        && prefix.len() / CHUNK == (total - 1) / CHUNK
}
fn canonical_match(
    renderer: &dialogue::Render<'_>,
    side: &Side,
    prefix: &[u16],
    target: &[u16],
) -> Result<()> {
    let (a, b) = renderer
        .encode(&side.role_messages, &side.original_target, 2)?
        .map_err(|reason| format!("original role arrays refused: {reason}"))?;
    require(
        a.into_iter().eq(prefix.iter().copied().map(u32::from))
            && b.into_iter().eq(target.iter().copied().map(u32::from)),
        "canonical role reconstruction differs from tape",
    )
}
fn source_rejoin(side: &Side) -> Result<()> {
    require(
        side.turn == 3 && side.original_questions.len() == 3 && side.role_messages.len() == 5,
        "intake row lacks authentic first three questions and preceding replies",
    )?;
    require(
        sha(side.original_questions[2].as_bytes()) == side.literal_question_sha256
            && side.role_messages[2].content == side.original_questions[1]
            && side.role_messages[4].content == side.original_questions[2],
        "literal original question does not match runtime role array",
    )?;
    let source = side.role_messages[0]
        .content
        .strip_suffix(&format!("\n\n{}", side.original_questions[0]))
        .and_then(|s| {
            s.strip_prefix(&format!(
                "{}\n{}\n\n",
                side.original_article_title, side.original_section_title
            ))
        })
        .ok_or("initial runtime user text lacks complete original source framing")?;
    let rejoin_count = source
        .match_indices("\n\n")
        .filter(|(i, _)| {
            let body = &source[i + 2..];
            let span = side.original_target_support_bytes.clone();
            body.get(span.clone()) == Some(side.original_target.as_str())
                && body
                    .get(..span.start)
                    .is_some_and(|p| p.chars().count() == side.original_target_support_scalar_start)
        })
        .count();
    require(
        rejoin_count == 1,
        "source body and positive extractive target could not be rejoined uniquely",
    )
}
fn mutate_marker(
    tokenizer: &ByteBpeTokenizer,
    renderer: &dialogue::Render<'_>,
    actual: &Side,
) -> Result<Value> {
    let assistant = tokenizer.encode("Assistant:");
    require(
        !assistant.is_empty(),
        "terminal assistant marker not encoded",
    )?;
    let marker_id: Vec<_> = assistant.into_iter().map(|id| id as u16).collect();
    let body = actual
        .prefix
        .strip_suffix(marker_id.as_slice())
        .ok_or("canonical opener marker absent")?;
    let from = body
        .len()
        .checked_sub(1)
        .ok_or("no preceding token for marker shift")?;
    let mut shifted = actual.prefix.clone();
    shifted[from..].rotate_left(1);
    require(
        shifted != actual.prefix,
        "shifted-marker mutation did not change tape",
    )?;
    require(
        loader_valid(&shifted, &actual.target, &actual.stratum),
        "shifted marker failed simple format before canonical test",
    )?;
    require(
        canonical_match(renderer, actual, &shifted, &actual.target).is_err(),
        "canonical receipt accepted shifted marker",
    )?;
    Ok(
        json!({"mutation":"move terminal Assistant marker one ID earlier without altering length or EOS count",
        "positive_real_prefix_accepted":true,"negative_passed_structural_loader":true,"negative_rejected_by_canonical_readback":true}),
    )
}

fn file(path: &Path) -> Result<BufWriter<fs::File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    ))
}
fn emit(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writeln!(writer)?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [training, tokenizer, artifact, output] = args.as_slice() else {
        return Err(
            "expected PINNED_GRAPH_FIRST_TRAIN PINNED_TOKENIZER STATIC_LABEL_ARTIFACT NEW_OUTPUT"
                .into(),
        );
    };
    let train_bytes = pinned(training, TRAIN)?;
    let pairs: Vec<Pair> = std::str::from_utf8(&train_bytes)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&pinned(tokenizer, TOKENIZER)?)
        .ok_or("invalid native tokenizer")?;
    let renderer = dialogue::Render::new(&tokenizer)?;
    let slots = permutation(&pinned(artifact, ARTIFACT)?)?;
    let mut families = BTreeMap::<String, Vec<&Pair>>::new();
    for pair in &pairs {
        families.entry(family(pair)?).or_default().push(pair);
    }
    let (dominant, dominant_cases) = families
        .iter()
        .min_by_key(|(family, group)| (std::cmp::Reverse(group.len()), rank("family", family)))
        .ok_or("missing input families")?;
    let d: Vec<_> = dominant_cases
        .iter()
        .copied()
        .filter(|p| third(p))
        .collect();
    let first = d
        .iter()
        .copied()
        .min_by_key(|p| pair_key(p, &slots))
        .ok_or("no actual third-turn pair in dominant TRAIN family")?;
    let earlier: BTreeSet<_> = [&first.a, &first.b]
        .into_iter()
        .flat_map(|s| s.original_questions.iter().take(2))
        .map(|q| normalized(q))
        .collect();
    let other: Vec<_> = families
        .iter()
        .filter(|(f, _)| *f != dominant && !earlier.contains(*f))
        .flat_map(|(_, pairs)| pairs.iter().copied())
        .filter(|p| third(p))
        .collect();
    let second = other
        .iter()
        .copied()
        .min_by_key(|p| pair_key(p, &slots))
        .ok_or("no unseen-in-batch actual third-turn pair")?;
    let selected = [(first, "dominant"), (second, "unseen_other_family")];
    let docs = [&first.a, &first.b, &second.a, &second.b];
    require(
        docs.iter()
            .map(|s| &s.component_sha256)
            .collect::<BTreeSet<_>>()
            .len()
            == 4,
        "intake source components repeat",
    )?;
    for s in docs {
        source_rejoin(s)?;
        require(
            loader_valid(&s.prefix, &s.target, &s.stratum),
            "row is incompatible with original #4361 tape loader",
        )?;
        require(
            s.target.first() == Some(&s.first_content_id),
            "target first ID differs",
        )?;
        canonical_match(&renderer, s, &s.prefix, &s.target)?;
        let expected_text = format!(
            " {}",
            s.original_target
                .replace("\r\n", "\n")
                .replace('\r', "\n")
                .trim()
        );
        let answer_ids: Vec<_> = s.target[..s.target.len() - 1]
            .iter()
            .copied()
            .map(u32::from)
            .collect();
        require(
            tokenizer.decode_bytes(&answer_ids) == expected_text.as_bytes(),
            "target reverse text differs",
        )?;
    }
    let finite = mutate_marker(&tokenizer, &renderer, &first.a)?;
    let output = Path::new(output);
    fs::DirBuilder::new().mode(0o700).create(output)?;
    let mut tape = file(&output.join("train.jsonl"))?;
    let mut messages = file(&output.join("role-and-source.jsonl"))?;
    let mut tokens = file(&output.join("tokens.u16le"))?;
    let mut mask = file(&output.join("response-mask.u8"))?;
    let mut aggregate_rows = Vec::new();
    let mut offset = 0;
    for (i, s) in docs.into_iter().enumerate() {
        let id = i + 1;
        let (pair, which) = selected[i / 2];
        let total = s.prefix.len() + s.target.len();
        for token in s.prefix.iter().chain(&s.target) {
            tokens.write_all(&token.to_le_bytes())?;
        }
        let loss_mask: Vec<_> = (0..total).map(|p| u8::from(p >= s.prefix.len())).collect();
        mask.write_all(&loss_mask)?;
        let mask_transitions: Vec<_> = loss_mask
            .windows(2)
            .enumerate()
            .filter_map(|(at, bytes)| (bytes[0] != bytes[1]).then_some(at + 1))
            .collect();
        require(
            mask_transitions == [s.prefix.len()],
            "response mask has unexpected state transitions",
        )?;
        let chunks = total.div_ceil(CHUNK);
        require(
            loss_mask.chunks(CHUNK).all(|c| c[0] == 0),
            "reset would lose a supervised onset",
        )?;
        emit(
            &mut tape,
            &json!({"id":id,"prefix":s.prefix,"expected":s.target,"stratum":s.stratum}),
        )?;
        emit(
            &mut messages,
            &json!({
                "id":id,"source_graph_pair":pair.pair,"source_component_sha256":s.component_sha256,
                "original_article_title":s.original_article_title,"original_section_title":s.original_section_title,
                "original_dialogue_id":s.original_dialogue_id,"original_target_turn_id":s.original_target_turn_id,
                "original_target":s.original_target,"original_questions":s.original_questions,"role_messages":s.role_messages,
                "original_target_support_bytes":s.original_target_support_bytes,
                "original_target_support_scalar_start":s.original_target_support_scalar_start,"token_index":offset,
            }),
        )?;
        aggregate_rows.push(json!({
            "id":id,"pair":which,"token_index":offset,"u16le_byte_offset":2*offset,"mask_byte_offset":offset,
            "prefix_tokens":s.prefix.len(),"target_tokens_including_eos":s.target.len(),"document_tokens":total,
            "zero_based_mask_transitions":mask_transitions,"first_target_index":s.prefix.len(),"final_eos_index":total-1,
            "eos_count_in_causal_prefix":2,"chunk_index":s.prefix.len()/CHUNK,"chunk_target_offset":s.prefix.len()%CHUNK,
            "full_transitions":total-1,"reset_causal_transitions":total-chunks,"reset_accounting_padding":chunks-1,
            "static_coarse_label":slots[usize::from(s.first_content_id)]/64,
        }));
        offset += total;
    }
    tape.flush()?;
    messages.flush()?;
    tokens.flush()?;
    mask.flush()?;
    let sum = |k: &str| {
        aggregate_rows
            .iter()
            .map(|r| r[k].as_u64().unwrap_or_default())
            .sum::<u64>()
    };
    let stats = json!({
        "evidence_class":"static TRAIN-only instrument intake",
        "input_training_roster_sha256":TRAIN,"tokenizer_sha256":TOKENIZER,"static_label_artifact_sha256":ARTIFACT,
        "selector_sha256":sha(include_bytes!("../SELECTOR.md")),
        "tape_format_clarification_sha256":sha(include_bytes!("../FORMAT.md")),
        "canonical_protocol":renderer.protocol_identity()?,
        "input_train_pairs":pairs.len(),"input_train_families":families.len(),"dominant_family_pairs":dominant_cases.len(),
        "dominant_family_actual_third_pair_pool":d.len(),"other_eligible_actual_third_pair_pool":other.len(),
        "eligible_different_coarse_dominant":d.iter().filter(|p|!pair_key(p,&slots).0).count(),
        "eligible_different_coarse_other":other.iter().filter(|p|!pair_key(p,&slots).0).count(),
        "selected":[
            {"source_graph_pair":first.pair,"rank_sha256":hex::encode(pair_key(first,&slots).1),"different_coarse":!pair_key(first,&slots).0},
            {"source_graph_pair":second.pair,"rank_sha256":hex::encode(pair_key(second,&slots).1),"different_coarse":!pair_key(second,&slots).0}
        ],
        "distinct_original_components":4,"target_families":2,"documents":4,"total_token_ids":offset,
        "full_transitions":sum("full_transitions"),"reset_causal_transitions":sum("reset_causal_transitions"),
        "reset_accounting_padding":sum("reset_accounting_padding"),"supervised_target_transitions":sum("target_tokens_including_eos"),
        "rows":aggregate_rows,"fresh_finite_negative":finite,
        "checks":"four canonical role arrays and gold IDs equal prior sealed rows; all first user source spans and answer bytes rejoined; two prior EOS and final target EOS; unchanged #4361 structural row contract"
    });
    let mut out = file(&output.join("aggregate.json"))?;
    serde_json::to_writer_pretty(&mut out, &stats)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
