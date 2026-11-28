mod dialogue;
mod inputs;

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
};

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CONTEXT: usize = 2048;
const PRIMARY_CONTEXT: usize = 512;
const WINDOW: usize = 32;
const MAX_PAIRS: usize = 64;
const TRAIN: &str = "c804c3b084a49d11f2bfcd2a5cf741862df29b42e3442ccb609305e049aba590";
const TUNE: &str = "99b714a67415d92e42750577110d9ce405625d93adabbd906739bbfca60d9c78";
const CHAT: &str = "e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef";
const COQA: &str = "b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6";
const SUPPLEMENT: &str = "483bf82f99e7ba454986726abc98b9bdee66b0df230b6a2c089a862dd364feee";

fn present<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| message.into())
}

fn require(value: bool, message: impl Into<String>) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(hex::encode(digest(&bytes)) == sha, "input SHA-256 mismatch")?;
    Ok(bytes)
}

fn rank(domain: &str, value: &str) -> [u8; 32] {
    digest(format!("uor-r4/compact-carry-cohort/v1\0{domain}\0{value}").as_bytes())
}

#[derive(Serialize, Clone)]

struct Row {
    prefix: Vec<u16>,
    expected: Vec<u16>,
    group: [String; 3],
    target_turn: usize,
    first_content: u16,
}

#[derive(Clone)]

struct Candidate {
    row: Row,
    question: String,
    final_suffix_len: usize,
    answer: String,
    passage: String,
    contaminated: bool,
}

#[derive(Serialize)]
struct Pair<'a> {
    pair_id: usize,
    a: &'a Row,
    b: &'a Row,
    common_suffix_ids: usize,
    stratum: &'static str,
    length_stratum: &'static str,
}

fn contains(container: &str, value: &str) -> bool {
    container.contains(&format!(" {value} "))
}

fn unique_answer(container: &str, value: &str) -> bool {
    let words: Vec<_> = container.split_whitespace().collect();
    let target: Vec<_> = value.split_whitespace().collect();
    words
        .windows(target.len())
        .filter(|window| *window == target)
        .take(2)
        .count()
        == 1
}

fn write_private(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    ))
}

fn window_hash(tokens: &[u16]) -> u64 {
    tokens.iter().fold(0_u64, |hash, &token| {
        hash.wrapping_mul(1_140_071_481_932_319_849)
            .wrapping_add(u64::from(token) + 1)
    })
}

fn find_overlap(candidates: &mut [Candidate], path: &Path, sha: &str) -> Result<usize> {
    let bytes = pinned(path, sha)?;
    require(bytes.len() % 2 == 0, "misaligned token file")?;
    let tokens: Vec<_> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    drop(bytes);
    let sequences: Vec<Vec<u16>> = candidates
        .iter()
        .map(|c| {
            c.row
                .prefix
                .iter()
                .chain(&c.row.expected)
                .copied()
                .collect()
        })
        .collect();
    let mut wanted: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for (index, sequence) in sequences.iter().enumerate() {
        if !candidates[index].contaminated {
            for (start, window) in sequence.windows(WINDOW).enumerate() {
                wanted
                    .entry(window_hash(window))
                    .or_default()
                    .push((index, start));
            }
        }
    }
    let mut found = 0;
    let mut windows = tokens.windows(WINDOW);
    let first = present(windows.next(), "token file too short")?;
    let mut hash = window_hash(first);
    let high = 1_140_071_481_932_319_849_u64.wrapping_pow((WINDOW - 1) as u32);
    for (start, window) in std::iter::once(first).chain(windows).enumerate() {
        if start > 0 {
            hash = hash
                .wrapping_sub((u64::from(tokens[start - 1]) + 1).wrapping_mul(high))
                .wrapping_mul(1_140_071_481_932_319_849)
                .wrapping_add(u64::from(window[WINDOW - 1]) + 1);
        }
        if let Some(hits) = wanted.get(&hash) {
            for &(index, offset) in hits {
                if !candidates[index].contaminated
                    && sequences[index][offset..offset + WINDOW] == *window
                {
                    candidates[index].contaminated = true;
                    found += 1;
                }
            }
        }
    }
    Ok(found)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (runs, previews, out, context_cap) = match args.as_slice() {
        [runs, previews, out] => (runs, previews, out, CONTEXT),
        [runs, previews, out, option] if option == "--512-only" => {
            (runs, previews, out, PRIMARY_CONTEXT)
        }
        _ => return Err("expected RUNS PREVIEWS NEW_OUTPUT [--512-only]".into()),
    };
    let runs = Path::new(runs);
    let out = Path::new(out);
    let previous = runs.join("20261128-codex-lexical-transfer");
    let source = previous.join("input/coqa-train-v1.0.json");
    let pool = inputs::Pool::Train(&source, COQA);
    let mut input = inputs::load(runs, Path::new(previews), &pool)?;
    let renderer = dialogue::Render::new(&input.tokenizer)?;
    inputs::planted_exclusions()?;
    renderer.planted_controls()?;
    let space = input.tokenizer.encode(" ");
    require(
        space.as_slice() == [223],
        "standalone space disagrees with Gate0",
    )?;
    let protocol =
        uor_r4_tokenizer::dialogue::DialogueProtocol::literal_roles_v2(&input.tokenizer)?;
    let encoder = protocol.bind(&input.tokenizer)?;

    let previous_truth = pinned(&previous.join("supplement-1/truth.jsonl"), SUPPLEMENT)?;
    let mut prior_sources = HashSet::new();
    let mut prior_files = HashSet::new();
    let mut prior_passages = HashSet::new();
    let mut prior_questions = HashSet::new();
    for line in std::str::from_utf8(&previous_truth)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let source = &row["source"];
        for (field, set) in [
            ("id", &mut prior_sources),
            ("normalized_passage_sha256", &mut prior_passages),
        ] {
            set.insert(present(source[field].as_str(), "invalid supplement source")?.to_owned());
        }
        prior_files.insert((
            present(source["source"].as_str(), "invalid supplement domain")?.to_owned(),
            present(source["filename"].as_str(), "invalid supplement filename")?.to_owned(),
        ));
        for question in present(row["questions"].as_array(), "invalid supplement questions")? {
            prior_questions.insert(inputs::normalize(present(
                question.as_str(),
                "invalid supplement question",
            )?));
        }
    }
    require(
        prior_sources.len() == 200,
        "invalid supplemental source census",
    )?;
    input
        .data
        .sort_by_cached_key(|doc| (rank("source", &doc.id), doc.id.clone()));
    let mut candidates = Vec::new();
    let mut standalone_space_excluded = 0;
    let mut seen = HashSet::new();
    for doc in &input.data {
        require(seen.insert(&doc.id), "duplicate CoQA ID")?;
        if !matches!(doc.source.as_str(), "gutenberg" | "cnn" | "wikipedia")
            || !doc.valid_turns()
            || doc.story.trim().is_empty()
            || doc.filename.is_empty()
        {
            continue;
        }
        let hash = inputs::passage_digest(&doc.story);
        let hash_text = hex::encode(hash);
        if input.exclusions.source_hit(doc, hash).is_some()
            || prior_sources.contains(&doc.id)
            || prior_files.contains(&doc.key())
            || prior_passages.contains(&hash_text)
        {
            continue;
        }
        let passage = format!(" {} ", inputs::normalize(&doc.story));
        for turn in 2..=doc.questions.len().min(5) {
            let questions: Vec<_> = doc.questions[..turn]
                .iter()
                .map(|q| q.input_text.clone())
                .collect();
            let answers: Vec<_> = doc.answers[..turn]
                .iter()
                .map(|a| a.input_text.clone())
                .collect();
            let history = dialogue::messages(&doc.story, &questions, &answers);
            if input
                .exclusions
                .prompt_hit(&history[0].content, questions.iter())
                .is_some()
                || input.exclusions.prior_question_hit(questions.iter())
                || questions
                    .iter()
                    .any(|question| prior_questions.contains(&inputs::normalize(question)))
            {
                continue;
            }
            let answer = inputs::normalize(&answers[turn - 1]);
            if answer.chars().count() < 2
                || matches!(
                    answer.as_str(),
                    "yes" | "no" | "unknown" | "none" | "not mentioned"
                )
                || !unique_answer(&passage, &answer)
            {
                continue;
            }
            let Ok((prefix, gold)) = renderer.encode(&history, &answers[turn - 1], turn - 1)?
            else {
                continue;
            };
            if prefix.len() > context_cap {
                continue;
            }
            let question = questions[turn - 1].clone();
            let suffix = encoder.encode_user_prefix(&question, true).tokens;
            if suffix.len() < 3 || !prefix.ends_with(&suffix) {
                continue;
            }
            let first_content = *present(
                gold.iter().skip_while(|&&id| id == space[0]).next(),
                "missing first content",
            )? as u16;
            if u32::from(first_content) == protocol.eos_id {
                continue;
            }
            if gold[0] != u32::from(first_content) {
                standalone_space_excluded += 1;
                continue;
            }
            candidates.push(Candidate {
                row: Row {
                    prefix: prefix.into_iter().map(|id| id as u16).collect(),
                    expected: gold.into_iter().map(|id| id as u16).collect(),
                    group: [
                        doc.id.clone(),
                        format!("{}/{}", doc.source, doc.filename),
                        hash_text.clone(),
                    ],
                    target_turn: turn,
                    first_content,
                },
                question,
                final_suffix_len: suffix.len(),
                answer,
                passage: passage.clone(),
                contaminated: false,
            });
        }
    }
    let eligible = candidates.len();
    let eligible_groups = candidates
        .iter()
        .map(|c| &c.row.group[0])
        .collect::<HashSet<_>>()
        .len();
    let mut by_question = HashMap::<&str, usize>::new();
    for candidate in &candidates {
        *by_question.entry(&candidate.question).or_default() += 1;
    }
    let recurring_questions = by_question.values().filter(|&&count| count > 1).count();
    candidates = candidates
        .iter()
        .filter(|c| by_question[c.question.as_str()] > 1)
        .map(Candidate::clone)
        .collect();
    let matched_question_candidates = candidates.len();
    let corpus = runs.join("20261127-antigravity-gate0-d87-proto2/corpus");
    let train_overlap = find_overlap(
        &mut candidates,
        &corpus.join("train-100m.tokens.u16"),
        TRAIN,
    )?;
    let tune_overlap = find_overlap(&mut candidates, &corpus.join("dev-tune.tokens.u16"), TUNE)?;
    let chat_overlap = find_overlap(
        &mut candidates,
        &runs.join("../inputs/chat-v0-p2/heldout.u16"),
        CHAT,
    )?;
    let mut groups = HashMap::<&str, Vec<&Candidate>>::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| !candidate.contaminated)
    {
        groups
            .entry(&candidate.question)
            .or_default()
            .push(candidate);
    }
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by_cached_key(|(q, _)| (rank("question", q), *q));
    let mut used = [HashSet::new(), HashSet::new(), HashSet::new()];
    let mut pairs = Vec::new();
    let mut selected_questions = HashSet::new();
    for (_, members) in &mut groups {
        members.sort_by_cached_key(|candidate| {
            let id = &candidate.row.group[0];
            let turn = candidate.row.target_turn;
            (rank("member", &format!("{id}\0{turn}")), id, turn)
        });
    }
    for (min_context, max_context, length_stratum) in [
        (0, PRIMARY_CONTEXT, "both_at_most_512"),
        (PRIMARY_CONTEXT + 1, context_cap, "maximum_513_to_2048"),
    ] {
        for (question, members) in &groups {
            for (i, a) in members.iter().enumerate() {
                if a.row.prefix.len() > max_context {
                    continue;
                }
                if used
                    .iter()
                    .zip(&a.row.group)
                    .any(|(set, key)| set.contains(key))
                {
                    continue;
                }
                let b = members[i + 1..].iter().find(|b| {
                    b.row.prefix.len() <= max_context
                        && a.row.prefix.len().max(b.row.prefix.len()) >= min_context
                        && a.row.expected[0] != b.row.expected[0]
                        && a.row.group.iter().zip(&b.row.group).all(|(x, y)| x != y)
                        && used
                            .iter()
                            .zip(&b.row.group)
                            .all(|(set, key)| !set.contains(key))
                        && !contains(&a.passage, &b.answer)
                        && !contains(&b.passage, &a.answer)
                });
                if let Some(b) = b {
                    require(
                        a.row.first_content == a.row.expected[0]
                            && b.row.first_content == b.row.expected[0]
                            && a.row.expected[0] != b.row.expected[0],
                        "pair lacks distinct first emitted content IDs",
                    )?;
                    for (set, (ag, bg)) in used.iter_mut().zip(a.row.group.iter().zip(&b.row.group))
                    {
                        set.insert(ag.clone());
                        set.insert(bg.clone());
                    }
                    let common_suffix_ids = a
                        .row
                        .prefix
                        .iter()
                        .rev()
                        .zip(b.row.prefix.iter().rev())
                        .take_while(|(x, y)| x == y)
                        .count();
                    require(
                        common_suffix_ids >= a.final_suffix_len
                            && a.final_suffix_len == b.final_suffix_len,
                        "final prompt token identity mismatch",
                    )?;
                    pairs.push(Pair {
                        pair_id: pairs.len() + 1,
                        a: &a.row,
                        b: &b.row,
                        common_suffix_ids,
                        stratum: "identical_literal_final_question",
                        length_stratum,
                    });
                    selected_questions.insert(*question);
                    if pairs.len() == MAX_PAIRS {
                        break;
                    }
                }
            }
            if pairs.len() == MAX_PAIRS {
                break;
            }
        }
        if pairs.len() == MAX_PAIRS {
            break;
        }
    }

    fs::DirBuilder::new().mode(0o700).create(out)?;
    let pair_path = out.join("pairs.jsonl");
    let index_path = out.join("selected-index.jsonl");
    let mut tape = write_private(&pair_path)?;
    let mut index = write_private(&index_path)?;
    for pair in &pairs {
        serde_json::to_writer(&mut tape, pair)?;
        writeln!(tape)?;
        serde_json::to_writer(
            &mut index,
            &serde_json::json!({
                "pair_id": pair.pair_id,
                "a": { "group": pair.a.group, "target_turn": pair.a.target_turn },
                "b": { "group": pair.b.group, "target_turn": pair.b.target_turn },
            }),
        )?;
        writeln!(index)?;
    }
    tape.flush()?;
    index.flush()?;
    let mut receipt = write_private(&out.join("aggregate.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &serde_json::json!({
            "schema": "uor-r4/compact-carry-cohort/2",
            "context_cap": context_cap, "protocol": renderer.protocol_identity()?, "coqa_sha256": COQA,
            "screen": { "train_sha256": TRAIN, "dev_tune_sha256": TUNE, "chat_v0_p2_sha256": CHAT, "prior_supplement_sha256": SUPPLEMENT,
                "contiguous_tokens": WINDOW, "train_excluded": train_overlap, "additional_dev_tune_excluded": tune_overlap,
                "additional_chat_v0_p2_excluded": chat_overlap },
            "prior_input_receipts": input.receipts,
            "eligible_extractable_turns": eligible, "standalone_space_turns_excluded": standalone_space_excluded,
            "candidate_turns_with_recurring_question": matched_question_candidates,
            "eligible_conversation_groups": eligible_groups, "recurring_literal_question_keys_before_overlap_screen": recurring_questions,
            "selected_literal_question_keys": selected_questions.len(),
            "pairs": pairs.len(), "unique_groups": used[0].len(), "pair_tape_sha256": hex::encode(digest(&fs::read(pair_path)?)),
            "selected_index_sha256": hex::encode(digest(&fs::read(index_path)?)),
            "pairs_both_at_most_512": pairs.iter().filter(|p| p.length_stratum == "both_at_most_512").count(),
            "sum_prefix_ids": pairs.iter().map(|p| p.a.prefix.len() + p.b.prefix.len()).sum::<usize>(),
            "maximum_prefix_ids": pairs.iter().map(|p| p.a.prefix.len().max(p.b.prefix.len())).max(),
            "common_suffix_id_range": [pairs.iter().map(|p| p.common_suffix_ids).min(), pairs.iter().map(|p| p.common_suffix_ids).max()],
            "scope": "CoQA original passage, source filename, and conversation ID; worker/author identity unavailable"
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    Ok(())
}
