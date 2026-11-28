mod dialogue;
mod exposure;
mod inputs;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
};

use serde_json::Value;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CONTEXT: usize = 2048;
const WINDOW: usize = 32;
const CAP: usize = 1088;
const TRAIN: &str = "c804c3b084a49d11f2bfcd2a5cf741862df29b42e3442ccb609305e049aba590";
const TUNE: &str = "99b714a67415d92e42750577110d9ce405625d93adabbd906739bbfca60d9c78";
const CHAT: &str = "e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef";
const COQA: &str = "b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6";

struct Candidate {
    tokens: Vec<u16>,
    domain: String,
    contaminated: bool,
    component: usize,
    group: [String; 3],
    prefix_len: usize,
    stratum: &'static str,
    later_in_chunk: bool,
    question: String,
    history: Vec<dialogue::RoleText>,
}

fn rank(purpose: &str, id: &str) -> [u8; 32] {
    digest(format!("uor-r4-grounded-reset-4361-v1\0{purpose}\0{id}").as_bytes())
}

fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}

fn write_file(path: &Path) -> Result<BufWriter<fs::File>> {
    Ok(BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    ))
}

fn domains(candidates: &[Candidate]) -> BTreeMap<&str, [usize; 4]> {
    let mut out = BTreeMap::new();
    for c in candidates.iter().filter(|c| !c.contaminated) {
        let values = out.entry(c.domain.as_str()).or_insert([0; 4]);
        values[0] += 1;
        values[1] += usize::from(c.tokens.len() <= 512);
        values[2] += usize::from(c.tokens.len() <= 1088);
        values[3] += usize::from(c.tokens.len() <= 2048);
    }
    out
}
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

const RESERVED: &str = "d6dd93ab4cbacf4eb6ea6fdaf42e5264dda4a325b78c14f2fcf038c9485adfb7";
const RESERVED_REJECTED: &str = "a335a338dda285c192582256797c6dd2a8afbd633fea90acf38d670314e4b005";

fn add_group(deny: &mut [HashSet<String>; 3], fields: [&str; 3]) {
    for (set, value) in deny.iter_mut().zip(fields) {
        set.insert(value.to_owned());
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [runs, previews, reserved, rejected, output] = args.as_slice() else {
        return Err("expected RUNS PREVIEW_DIR RESERVED_INDEX REJECTED_INDEX NEW_OUTPUT".into());
    };
    let runs = Path::new(runs);
    let prev = runs.join("20261128-codex-lexical-transfer");
    let source = prev.join("input/coqa-train-v1.0.json");
    let pool = inputs::Pool::Train(&source, COQA);
    let input = inputs::load(runs, Path::new(previews), &pool)?;
    let renderer = dialogue::Render::new(&input.tokenizer)?;
    let mut deny = [HashSet::new(), HashSet::new(), HashSet::new()];
    let previous_truth = pinned(&prev.join("supplement-1/truth.jsonl"), SUPPLEMENT)?;
    let mut prior_questions = HashSet::new();
    for line in std::str::from_utf8(&previous_truth)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let doc = &row["source"];
        let filename = format!(
            "{}/{}",
            present(doc["source"].as_str(), "missing source domain")?,
            present(doc["filename"].as_str(), "missing filename")?
        );
        add_group(
            &mut deny,
            [
                present(doc["id"].as_str(), "missing ID")?,
                &filename,
                present(
                    doc["normalized_passage_sha256"].as_str(),
                    "missing passage digest",
                )?,
            ],
        );
        for question in present(row["questions"].as_array(), "missing prior questions")? {
            prior_questions.insert(inputs::normalize(present(
                question.as_str(),
                "missing question",
            )?));
        }
    }
    require(deny[0].len() == 200, "prior supplement count mismatch")?;
    let mut reserved_ids = HashSet::new();
    for (path, sha) in [(reserved, RESERVED), (rejected, RESERVED_REJECTED)] {
        let reserved_data = pinned(Path::new(path), sha)?;
        let mut roster = HashSet::new();
        for line in std::str::from_utf8(&reserved_data)?.lines() {
            let row: Value = serde_json::from_str(line)?;
            for side in ["a", "b"] {
                let g = &row[side]["group"];
                let fields = [
                    present(g[0].as_str(), "missing reserved ID")?,
                    present(g[1].as_str(), "missing reserved filename")?,
                    present(g[2].as_str(), "missing reserved passage digest")?,
                ];
                reserved_ids.insert(fields[0].to_owned());
                roster.insert(fields[0].to_owned());
                add_group(&mut deny, fields);
            }
        }
        require(
            roster.len() == 128,
            "reserved roster article count mismatch",
        )?;
    }
    require(
        reserved_ids.len() == 151,
        "union of reserved article IDs mismatch",
    )?;
    let groups: Vec<_> = input
        .data
        .iter()
        .map(|doc| {
            [
                doc.id.clone(),
                format!("{}/{}", doc.source, doc.filename),
                hex::encode(inputs::passage_digest(&doc.story)),
            ]
        })
        .collect();
    let mut parents: Vec<_> = (0..groups.len()).collect();
    let mut owners = [HashMap::new(), HashMap::new(), HashMap::new()];
    for (i, group) in groups.iter().enumerate() {
        for (owner, key) in owners.iter_mut().zip(group) {
            if let Some(&j) = owner.get(key.as_str()) {
                let a = root(&mut parents, i);
                let b = root(&mut parents, j);
                parents[a] = b;
            } else {
                owner.insert(key.as_str(), i);
            }
        }
    }
    let roots: Vec<_> = (0..groups.len()).map(|i| root(&mut parents, i)).collect();
    let mut component_ids: HashMap<usize, Vec<&str>> = HashMap::new();
    let mut excluded = HashSet::new();
    for (i, (doc, group)) in input.data.iter().zip(&groups).enumerate() {
        component_ids.entry(roots[i]).or_default().push(&doc.id);
        if input
            .exclusions
            .source_hit(doc, inputs::passage_digest(&doc.story))
            .is_some()
            || deny.iter().zip(group).any(|(set, key)| set.contains(key))
        {
            excluded.insert(roots[i]);
        }
    }
    let component_ids: HashMap<_, _> = component_ids
        .into_iter()
        .map(|(key, mut ids)| {
            ids.sort_unstable();
            (key, ids.join("\0"))
        })
        .collect();
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut candidates = Vec::new();
    let normalize = inputs::normalize;
    for (i, doc) in input.data.iter().enumerate() {
        if !matches!(doc.source.as_str(), "cnn" | "gutenberg" | "wikipedia")
            || excluded.contains(&roots[i])
            || !doc.valid_turns()
            || doc.answers.len() < 3
            || doc.filename.is_empty()
        {
            continue;
        }
        *counts
            .entry("three_turn_groups_before_grounding_or_local_dedupe")
            .or_default() += 1;
        let questions: Vec<_> = doc.questions[..3]
            .iter()
            .map(|t| t.input_text.clone())
            .collect();
        let answers: Vec<_> = doc.answers[..3]
            .iter()
            .map(|t| t.input_text.clone())
            .collect();
        let history = dialogue::messages(&doc.story, &questions, &answers);
        if input
            .exclusions
            .prompt_hit(&history[0].content, questions.iter())
            .is_some()
            || input.exclusions.prior_question_hit(questions.iter())
            || questions
                .iter()
                .any(|q| prior_questions.contains(&normalize(q)))
        {
            continue;
        }
        let offsets: Vec<_> = doc
            .story
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(doc.story.len()))
            .collect();
        let mut all_offset_backed = true;
        let mut all_exact_answer = true;
        for a in &doc.answers[..3] {
            let Some((start, end, span)) = a
                .span_start
                .zip(a.span_end)
                .zip(a.span_text.as_deref())
                .and_then(|((s, e), t)| {
                    Some((usize::try_from(s).ok()?, usize::try_from(e).ok()?, t))
                })
            else {
                all_offset_backed = false;
                break;
            };
            let Some(slice) = offsets
                .get(start)
                .zip(offsets.get(end))
                .and_then(|(&s, &e)| doc.story.get(s..e))
            else {
                all_offset_backed = false;
                break;
            };
            let normalized = normalize(slice);
            if normalized.is_empty() || normalized != normalize(span) {
                all_offset_backed = false;
                break;
            }
            let answer = normalize(&a.input_text);
            let answer_words: Vec<_> = answer.split_whitespace().collect();
            let span_words: Vec<_> = normalized.split_whitespace().collect();
            if answer_words.is_empty()
                || !span_words
                    .windows(answer_words.len())
                    .any(|window| window == answer_words)
            {
                all_exact_answer = false;
            }
        }
        if !all_offset_backed {
            continue;
        }
        *counts
            .entry("first_three_original_offsets_rejoin")
            .or_default() += 1;
        if !all_exact_answer {
            continue;
        }
        *counts
            .entry("first_three_answer_values_literal_within_offsets")
            .or_default() += 1;
        let Ok((prefix, gold)) = renderer.encode(&history, &answers[2], 2)? else {
            continue;
        };
        let gold_first = gold[0];
        let complete_len = prefix.len() + gold.len();
        if gold_first == 223 || complete_len > CAP {
            continue;
        }
        let target = prefix.len();
        if target / 257 != (target - 1) / 257 || (complete_len - 1) / 257 != target / 257 {
            *counts
                .entry("target_or_eos_straddles_257_boundary")
                .or_default() += 1;
            continue;
        }
        let start = offsets[doc.answers[2].span_start.unwrap() as usize];
        let end = offsets[doc.answers[2].span_end.unwrap() as usize];
        let (stratum, later_in_chunk, _) = exposure::exposure(
            &history,
            &doc.story,
            start..end,
            &answers[2],
            &prefix,
            &input.tokenizer,
        )?;
        *counts
            .entry("strict_eligible_complete_documents_before_component_representative")
            .or_default() += 1;
        candidates.push(Candidate {
            tokens: prefix.into_iter().chain(gold).map(|id| id as u16).collect(),
            domain: doc.source.clone(),
            contaminated: false,
            component: roots[i],
            group: groups[i].clone(),
            prefix_len: target,
            stratum,
            later_in_chunk,
            question: questions[2].clone(),
            history,
        });
        if complete_len <= 512 {
            *counts.entry("complete_document_at_most_512").or_default() += 1;
        }
        if complete_len <= 2048 {
            *counts.entry("complete_document_at_most_2048").or_default() += 1;
        }
    }
    let before = serde_json::to_value(domains(&candidates))?;
    let corpus = runs.join("20261127-antigravity-gate0-d87-proto2/corpus");
    let train_hits = find_overlap(
        &mut candidates,
        &corpus.join("train-100m.tokens.u16"),
        TRAIN,
    )?;
    let post_train = serde_json::to_value(domains(&candidates))?;
    let tune_hits = find_overlap(&mut candidates, &corpus.join("dev-tune.tokens.u16"), TUNE)?;
    let post_tune = serde_json::to_value(domains(&candidates))?;
    let chat_hits = find_overlap(
        &mut candidates,
        &runs.join("../inputs/chat-v0-p2/heldout.u16"),
        CHAT,
    )?;
    let post_chat = serde_json::to_value(domains(&candidates))?;
    let output = Path::new(output);
    fs::DirBuilder::new().mode(0o700).create(output)?;
    let split = write_splits(output, &candidates, &component_ids)?;
    let mut receipt = write_file(&output.join("aggregate.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &serde_json::json!({
            "counts": counts, "coqa_sha256": COQA, "reserved_4360_index_sha256": RESERVED,
            "reserved_4360_rejected_index_sha256": RESERVED_REJECTED,
            "reserved_4360_union_article_ids": reserved_ids.len(),
            "domain_count_columns": ["all", "complete_le_512", "complete_le_1088", "complete_le_2048"],
            "before_overlap": before,
            "after_train": post_train, "after_tune": post_tune, "after_chat": post_chat,
            "train_excluded": train_hits, "additional_tune_excluded": tune_hits, "additional_chat_excluded": chat_hits,
            "train_sha256": TRAIN, "tune_sha256": TUNE, "chat_sha256": CHAT,
            "screen": "all contiguous 32-ID windows over complete canonical three-response document, including BOS and every EOS; hashes verify with exact IDs",
            "selector_sha256": hex::encode(digest(include_bytes!("../SELECTOR.md"))),
            "assignment": split,
            "status": "data build only, no model execution"
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    Ok(())
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
    let mut wanted: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.contaminated {
            for (start, window) in candidate.tokens.windows(WINDOW).enumerate() {
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
                    && candidates[index].tokens[offset..offset + WINDOW] == *window
                {
                    candidates[index].contaminated = true;
                    found += 1;
                }
            }
        }
    }
    Ok(found)
}

fn write_splits(
    output: &Path,
    candidates: &[Candidate],
    identities: &HashMap<usize, String>,
) -> Result<Value> {
    let mut reps = HashMap::<usize, &Candidate>::new();
    let rep_key = |c: &Candidate| (rank("representative", &c.group[0]), c.group[0].clone());
    for c in candidates.iter().filter(|c| !c.contaminated) {
        let old = reps.entry(c.component).or_insert(c);
        if rep_key(c) < rep_key(old) {
            *old = c;
        }
    }
    let mut domains = BTreeMap::<&str, Vec<&Candidate>>::new();
    for rep in reps.values() {
        domains.entry(&rep.domain).or_default().push(rep);
    }
    let mut splits = [Vec::new(), Vec::new(), Vec::new()];
    let mut assignments = BTreeMap::new();
    for (domain, members) in &mut domains {
        members.sort_by_cached_key(|c| {
            let identity = &identities[&c.component];
            (rank("assignment", identity), identity)
        });
        let train = members.len() * 70 / 100;
        let dev = members.len() * 10 / 100;
        assignments.insert(*domain, [train, dev, members.len() - train - dev]);
        splits[0].extend_from_slice(&members[..train]);
        splits[1].extend_from_slice(&members[train..train + dev]);
        splits[2].extend_from_slice(&members[train + dev..]);
    }
    let order_key = |c: &&Candidate| {
        let identity = &identities[&c.component];
        (rank("order", identity), identity)
    };
    for split in &mut splits {
        split.sort_by_cached_key(order_key);
    }
    let available_train = splits[0].len();
    let train_preserved: HashSet<_> = ["near", "far"]
        .iter()
        .filter_map(|&s| {
            splits[0]
                .iter()
                .find(|c| c.stratum == s)
                .map(|c| c.component)
        })
        .collect();
    let mut selected: Vec<_> = splits[0]
        .iter()
        .copied()
        .filter(|c| train_preserved.contains(&c.component))
        .collect();
    selected.extend(
        splits[0]
            .iter()
            .copied()
            .filter(|c| !train_preserved.contains(&c.component))
            .take(256 - selected.len()),
    );
    selected.sort_by_cached_key(order_key);
    splits[0] = selected;

    let mut prior = HashSet::new();
    let mut rollup = BTreeMap::new();
    let mut gates = splits[0].len() == 256 && splits[2].len() >= 200;
    for (label, split) in ["train", "development", "held"].into_iter().zip(&splits) {
        let path = output.join(format!("{label}.jsonl"));
        let mut file = write_file(&path)?;
        let mut sidecar = if label == "held" {
            Some(write_file(&output.join("held-messages.jsonl"))?)
        } else {
            None
        };
        let mut strata = BTreeMap::from([("near", 0), ("far", 0), ("intermediate", 0)]);
        let mut later = 0;
        let mut actual_domains = BTreeMap::new();
        let mut questions = BTreeMap::<&str, Vec<u16>>::new();
        for (i, c) in split.iter().enumerate() {
            require(prior.insert(c.component), "component appears twice")?;
            let id = i + 1;
            let prefix = &c.tokens[..c.prefix_len];
            let expected = &c.tokens[c.prefix_len..];
            serde_json::to_writer(
                &mut file,
                &serde_json::json!({
                    "id": id, "prefix": prefix, "expected": expected,
                    "stratum": c.stratum, "later_in_chunk": c.later_in_chunk,
                    "question_sha256": hex::encode(digest(c.question.as_bytes())),
                    "source_group": c.group,
                }),
            )?;
            writeln!(file)?;
            if let Some(sidecar) = &mut sidecar {
                serde_json::to_writer(
                    &mut *sidecar,
                    &serde_json::json!({"id": id, "messages": c.history}),
                )?;
                writeln!(sidecar)?;
            }
            *strata.get_mut(c.stratum).unwrap() += 1;
            later += usize::from(c.later_in_chunk);
            *actual_domains.entry(c.domain.clone()).or_insert(0) += 1;
            questions.entry(&c.question).or_default().push(expected[0]);
        }
        file.flush()?;
        if let Some(mut sidecar) = sidecar {
            sidecar.flush()?;
        }
        gates &= strata["near"] >= 1 && strata["far"] >= 1;
        let pairs: usize = questions
            .values()
            .map(|group| {
                let mut ids = HashMap::new();
                for &id in group {
                    *ids.entry(id).or_insert(0usize) += 1;
                }
                (group.len() / 2).min(group.len() - ids.values().copied().max().unwrap_or(0))
            })
            .sum();
        rollup.insert(label, serde_json::json!({
            "rows": split.len(), "strata": strata, "later_value_in_chunk": later, "domains": actual_domains,
            "distinct_exact_questions": questions.len(), "same_question_distinct_first_id_disjoint_pairs": pairs,
            "maximum_complete_ids": split.iter().map(|c| c.tokens.len()).max(),
            "sha256": hex::encode(digest(&fs::read(path)?)),
        }));
    }
    Ok(serde_json::json!({
        "available_components": reps.len(), "available_train": available_train,
        "available_domain_counts_train_development_held": assignments, "selected": rollup,
        "descriptive_data_gates_pass": gates,
        "held_messages_sha256": hex::encode(digest(&fs::read(output.join("held-messages.jsonl"))?)),
    }))
}
