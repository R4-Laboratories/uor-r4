#![forbid(unsafe_code)]
mod dialogue;
mod exposure;
mod inputs;
mod pairing;
mod prior;

use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    ops::Range,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CONTEXT: usize = 2048;
const CAP: usize = 1088;
const CHUNK: usize = 257;
const WINDOW: usize = 32;
const COQA: &str = "b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6";
const PREVIEW_MANIFEST: &str = "45da31cf48d4a18719b5218a6ca1e22b461ef519a4642200dc1d46bef091fd00";
const HISTORICAL: [(&str, &str, &str); 3] = [
    (
        "historical_train",
        "20261127-antigravity-gate0-d87-proto2/corpus/train-100m.tokens.u16",
        "c804c3b084a49d11f2bfcd2a5cf741862df29b42e3442ccb609305e049aba590",
    ),
    (
        "historical_dev",
        "20261127-antigravity-gate0-d87-proto2/corpus/dev-tune.tokens.u16",
        "99b714a67415d92e42750577110d9ce405625d93adabbd906739bbfca60d9c78",
    ),
    (
        "historical_prior",
        "../inputs/chat-v0-p2/heldout.u16",
        "e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef",
    ),
];

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
fn sha(bytes: &[u8]) -> String {
    hex::encode(digest(bytes))
}
fn pinned(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    require(
        sha(&bytes) == expected,
        format!("digest differs for {}", path.display()),
    )?;
    Ok(bytes)
}
fn rank(purpose: &str, identity: &str) -> [u8; 32] {
    digest(format!("uor-r4-natural-pairs-v1\0{purpose}\0{identity}").as_bytes())
}
fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}
fn allowed_domain(doc: &inputs::Conversation) -> bool {
    matches!(doc.source.as_str(), "cnn" | "gutenberg" | "wikipedia")
}
fn cluster(groups: &[[String; 3]]) -> Vec<usize> {
    let mut parents: Vec<_> = (0..groups.len()).collect();
    let mut owners: [HashMap<&str, usize>; 3] = Default::default();
    for (i, group) in groups.iter().enumerate() {
        for (owner, key) in owners.iter_mut().zip(group) {
            if let Some(&previous) = owner.get(key.as_str()) {
                let a = root(&mut parents, i);
                let b = root(&mut parents, previous);
                parents[a] = b;
            } else {
                owner.insert(key, i);
            }
        }
    }
    (0..groups.len()).map(|i| root(&mut parents, i)).collect()
}
fn normalized_has(haystack: &str, needle: &str) -> bool {
    let needle: Vec<_> = needle.split_whitespace().collect();
    !needle.is_empty()
        && haystack
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(needle.len())
            .any(|w| w == needle)
}

#[derive(Clone, Debug, Serialize)]
struct Support {
    #[serde(flatten)]
    bytes: Range<usize>,
    scalar_start: usize,
    scalar_end: usize,
    exact_raw_annotation: bool,
}

fn offsets(story: &str) -> Vec<usize> {
    story
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(story.len()))
        .collect()
}

fn source_support(
    story: &str,
    offsets: &[usize],
    answer: &inputs::Turn,
) -> Option<(Support, bool)> {
    let scalar_start = usize::try_from(answer.span_start?).ok()?;
    let scalar_end = usize::try_from(answer.span_end?).ok()?;
    let annotation = answer.span_text.as_deref()?;
    let bytes = *offsets.get(scalar_start)?..*offsets.get(scalar_end)?;
    let slice = story.get(bytes.clone())?;
    let normalized = inputs::normalize(slice);
    if normalized.is_empty() || normalized != inputs::normalize(annotation) {
        return None;
    }
    let support = Support {
        bytes,
        scalar_start,
        scalar_end,
        exact_raw_annotation: slice == annotation,
    };
    let literal = normalized_has(&normalized, &inputs::normalize(&answer.input_text));
    Some((support, literal))
}

fn first_content(tokenizer: &ByteBpeTokenizer, first_id: u32) -> bool {
    let bytes = tokenizer.decode_bytes(&[first_id]);
    // A byte-level ID can legally stop partway through a scalar. Such an ID cannot
    // establish alphanumeric content on its own and is excluded conservatively.
    std::str::from_utf8(&bytes).is_ok_and(|s| s.chars().any(char::is_alphanumeric))
}

fn preview_pin(previews: &Path) -> Result<()> {
    let manifest = pinned(&previews.join("PREVIEWS.sha256"), PREVIEW_MANIFEST)?;
    for line in std::str::from_utf8(&manifest)?.lines() {
        let (sha, filename) = present(line.split_once("  "), "invalid preview manifest")?;
        require(
            Path::new(filename).components().count() == 1,
            "invalid preview entry",
        )?;
        pinned(&previews.join(filename), sha)?;
    }
    let actual = fs::read_dir(previews)?
        .filter_map(|item| item.ok()?.file_name().into_string().ok())
        .filter(|name| name.ends_with(".jsonl"))
        .count();
    require(actual == 4, "preview file inventory changed")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Split {
    Train,
    Held,
}

struct Candidate<'a> {
    doc: usize,
    component: usize,
    component_id: &'a str,
    question: &'a str,
    answer_normalized: String,
    turn: usize,
    first: u32,
    level: usize,
    split: Split,
    family_split: Split,
    strict: Option<Strict>,
}

struct Strict {
    tokens: Vec<u16>,
    prefix_len: usize,
    support: Vec<Support>,
    stratum: &'static str,
    later_in_chunk: bool,
}

fn get_history(doc: &inputs::Conversation, turns: usize) -> Vec<dialogue::RoleText> {
    let questions: Vec<_> = doc.questions[..turns]
        .iter()
        .map(|q| q.input_text.clone())
        .collect();
    let answers: Vec<_> = doc.answers[..turns]
        .iter()
        .map(|a| a.input_text.clone())
        .collect();
    dialogue::messages(&doc.story, &questions, &answers)
}

struct Screening<'a> {
    exclusion: &'a inputs::Exclusions,
    prior: &'a prior::Prior,
    renderer: &'a dialogue::Render<'a>,
    tokenizer: &'a ByteBpeTokenizer,
    rejections: BTreeMap<&'static str, usize>,
    byte_exact: usize,
    target_only: bool,
}

impl Screening<'_> {
    fn reject(&mut self, reason: &'static str) {
        *self.rejections.entry(reason).or_default() += 1;
    }

    fn screen(
        &mut self,
        doc: &inputs::Conversation,
        candidate: &mut Candidate<'_>,
        char_offsets: &[usize],
    ) -> Result<()> {
        let questions = &doc.questions[..candidate.turn];
        let answers = &doc.answers[..candidate.turn];
        if questions
            .iter()
            .chain(answers)
            .any(|turn| turn.input_text.trim().is_empty())
        {
            self.reject("empty_original_turn");
            return Ok(());
        }
        let history = get_history(doc, candidate.turn);
        let question_texts = || questions.iter().map(|q| q.input_text.as_str());
        if self
            .exclusion
            .prompt_hit(&history[0].content, question_texts())
            .is_some()
        {
            self.reject("d87_exact_prompt_or_template");
            return Ok(());
        }
        if self.exclusion.prior_question_hit(question_texts()) {
            self.reject("historical_4352_panel_or_preview_question");
            return Ok(());
        }
        if question_texts().any(|q| self.prior.question_families.contains(&inputs::normalize(q))) {
            self.reject("previous_4360_4361_or_lexical_question");
            return Ok(());
        }
        candidate.level = 1;
        let required_answers = if self.target_only {
            &answers[candidate.turn - 1..]
        } else {
            answers
        };
        let Some(support) = required_answers
            .iter()
            .map(|a| source_support(&doc.story, char_offsets, a))
            .collect::<Option<Vec<_>>>()
        else {
            self.reject("missing_or_invalid_original_evidence_offsets");
            return Ok(());
        };
        candidate.level = 2;
        if support.iter().any(|(_, literal)| !literal) {
            self.reject("answer_absent_from_own_original_evidence_span");
            return Ok(());
        }
        if support.iter().all(|(s, _)| s.exact_raw_annotation) {
            self.byte_exact += 1;
        }
        candidate.level = 3;
        let reply = &answers[candidate.turn - 1].input_text;
        let (prefix, gold) = match self.renderer.encode(&history, reply, candidate.turn - 1)? {
            Ok(encoded) => encoded,
            Err(reason) => {
                self.reject(reason);
                return Ok(());
            }
        };
        let first = gold[0];
        if !first_content(self.tokenizer, first) {
            self.reject("actual_opener_is_not_a_complete_content_id");
            return Ok(());
        }
        require(
            first == candidate.first,
            "strict opener and relaxed first content selection differ",
        )?;
        candidate.level = 4;
        let complete = prefix.len() + gold.len();
        if complete > CAP {
            self.reject("complete_dialogue_above_1088");
            return Ok(());
        }
        let target = prefix.len();
        if target / CHUNK != (target - 1) / CHUNK || (complete - 1) / CHUNK != target / CHUNK {
            self.reject("opening_predecessor_or_eos_crosses_chunk");
            return Ok(());
        }
        let support: Vec<_> = support.into_iter().map(|(span, _)| span).collect();
        let (stratum, later_in_chunk, _) = exposure::exposure(
            &history,
            &doc.story,
            present(support.last(), "missing target source support")?
                .bytes
                .clone(),
            reply,
            &prefix,
            self.tokenizer,
        )?;
        candidate.level = 5;
        candidate.strict = Some(Strict {
            tokens: prefix.into_iter().chain(gold).map(|id| id as u16).collect(),
            prefix_len: target,
            support,
            stratum,
            later_in_chunk,
        });
        Ok(())
    }
}

const BASE: u64 = 1_140_071_481_932_319_849;
fn window_hash(tokens: &[u16]) -> u64 {
    tokens.iter().fold(0_u64, |hash, &token| {
        hash.wrapping_mul(BASE).wrapping_add(u64::from(token) + 1)
    })
}
fn overlap(candidates: &mut [Candidate<'_>], tokens: &[u16], level: usize) -> Result<usize> {
    let mut wanted = HashMap::<u64, Vec<(usize, usize)>>::new();
    for (index, candidate) in candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.level == level)
    {
        let strict = present(
            candidate.strict.as_ref(),
            "missing strict row during overlap scan",
        )?;
        for (start, window) in strict.tokens.windows(WINDOW).enumerate() {
            wanted
                .entry(window_hash(window))
                .or_default()
                .push((index, start));
        }
    }
    let high = BASE.wrapping_pow((WINDOW - 1) as u32);
    let mut hash = window_hash(present(tokens.get(..WINDOW), "token stream too short")?);
    let mut found = 0;
    let mut contaminated = vec![false; candidates.len()];
    for (start, window) in tokens.windows(WINDOW).enumerate() {
        if start > 0 {
            hash = hash
                .wrapping_sub((u64::from(tokens[start - 1]) + 1).wrapping_mul(high))
                .wrapping_mul(BASE)
                .wrapping_add(u64::from(window[WINDOW - 1]) + 1);
        }
        if let Some(hits) = wanted.get(&hash) {
            for &(index, offset) in hits {
                let strict = present(candidates[index].strict.as_ref(), "overlap row missing")?;
                if !contaminated[index] && strict.tokens[offset..offset + WINDOW] == *window {
                    contaminated[index] = true;
                    found += 1;
                }
            }
        }
    }
    for (index, candidate) in candidates.iter_mut().enumerate() {
        if candidate.level == level && !contaminated[index] {
            candidate.level += 1;
        }
    }
    Ok(found)
}

fn output_file(path: &Path) -> Result<BufWriter<fs::File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    ))
}

fn side(
    candidate: &Candidate<'_>,
    docs: &[inputs::Conversation],
    groups: &[[String; 3]],
) -> Result<Value> {
    let doc = &docs[candidate.doc];
    let strict = present(
        candidate.strict.as_ref(),
        "packing candidate lacks source support",
    )?;
    let support: Vec<_> = strict
        .support
        .iter()
        .zip(
            doc.answers[..candidate.turn]
                .iter()
                .skip(candidate.turn - strict.support.len()),
        )
        .map(|(support, original)| {
            json!({"offsets": support, "annotation": original.span_text,
            "source_slice": &doc.story[support.bytes.clone()], "original_turn": original.turn_id})
        })
        .collect();
    Ok(json!({
        "source_group": groups[candidate.doc],
        "component_sha256": sha(candidate.component_id.as_bytes()),
        "literal_question_sha256": sha(candidate.question.as_bytes()),
        "normalized_question_family_sha256": sha(inputs::normalize(candidate.question).as_bytes()),
        "turn": candidate.turn, "source_support": support,
        "role_messages": get_history(doc, candidate.turn),
        "original_target": doc.answers[candidate.turn - 1].input_text,
        "prefix": strict.tokens[..strict.prefix_len], "target": strict.tokens[strict.prefix_len..],
        "first_content_id": candidate.first, "stratum": strict.stratum,
        "later_in_chunk": strict.later_in_chunk,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--finite"] {
        return finite();
    }
    let [runs, previews, mode, output] = args.as_slice() else {
        return Err("expected RUNS PREVIEWS MODE NEW_OUTPUT".into());
    };
    let target_only = match mode.as_str() {
        "strict" => false,
        "target-only" => true,
        _ => return Err("expected strict or target-only".into()),
    };
    let (runs, previews, output) = (Path::new(runs), Path::new(previews), Path::new(output));
    require(!output.exists(), "output already exists")?;
    preview_pin(previews)?;
    let source = runs.join("20261128-codex-lexical-transfer/input/coqa-train-v1.0.json");
    let input = inputs::load(runs, previews, &inputs::Pool::Train(&source, COQA))?;
    let renderer = dialogue::Render::new(&input.tokenizer)?;
    renderer.planted_controls()?;
    inputs::planted_exclusions()?;
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
    let prior = prior::Prior::load(runs, &input.data, &groups)?;
    let components = cluster(&groups);
    let mut members = BTreeMap::<usize, Vec<&str>>::new();
    let mut historical = BTreeSet::new();
    let mut prev_panels = BTreeSet::new();
    for (i, (doc, group)) in input.data.iter().zip(&groups).enumerate() {
        members.entry(components[i]).or_default().push(&doc.id);
        if input
            .exclusions
            .source_hit(doc, inputs::passage_digest(&doc.story))
            .is_some()
        {
            historical.insert(components[i]);
        }
        if prior
            .identity
            .iter()
            .zip(group)
            .any(|(set, key)| set.contains(key))
        {
            prev_panels.insert(components[i]);
        }
    }
    let identities: BTreeMap<_, _> = members
        .into_iter()
        .map(|(component, mut ids)| {
            ids.sort_unstable();
            (component, ids.join("\0"))
        })
        .collect();
    let component_splits: BTreeMap<_, _> = identities
        .iter()
        .map(|(&component, ids)| {
            let hash = rank("component-split", ids);
            let side = if u16::from_be_bytes([hash[0], hash[1]]) < 52429 {
                Split::Train
            } else {
                Split::Held
            };
            (component, side)
        })
        .collect();

    let inventory: Vec<_> = input
        .data
        .iter()
        .enumerate()
        .filter(|(i, doc)| {
            allowed_domain(doc)
                && doc.valid_turns()
                && !doc.filename.is_empty()
                && !prev_panels.contains(&components[*i])
                && !historical.contains(&components[*i])
        })
        .collect();
    let mut families = HashMap::<String, ([u8; 32], Split)>::new();
    for &(i, doc) in &inventory {
        let component = components[i];
        for question in &doc.questions {
            let family = inputs::normalize(&question.input_text);
            if family.is_empty() {
                continue;
            }
            let hash = rank(
                "family-owner",
                &format!("{family}\0{}", identities[&component]),
            );
            let current = families
                .entry(family)
                .or_insert((hash, component_splits[&component]));
            if hash < current.0 {
                *current = (hash, component_splits[&component]);
            }
        }
    }
    let mut screening = Screening {
        exclusion: &input.exclusions,
        prior: &prior,
        renderer: &renderer,
        tokenizer: &input.tokenizer,
        rejections: BTreeMap::new(),
        byte_exact: 0,
        target_only,
    };
    let mut candidates = Vec::new();
    let space = input.tokenizer.encode(" ");
    let excluded_space = if space.len() == 1 {
        Some(space[0])
    } else {
        None
    };
    for &(i, doc) in &inventory {
        let component = components[i];
        let offsets = offsets(&doc.story);
        for (j, (question, answer)) in doc.questions.iter().zip(&doc.answers).enumerate().skip(2) {
            let family = inputs::normalize(&question.input_text);
            let answer_norm = inputs::normalize(&answer.input_text);
            if family.is_empty() || answer_norm.is_empty() {
                screening.reject("empty_normalized_question_or_answer");
                continue;
            }
            let canon = answer.input_text.replace("\r\n", "\n").replace('\r', "\n");
            let encoded = input.tokenizer.encode(&format!(" {}", canon.trim()));
            let Some(first) = encoded
                .iter()
                .copied()
                .find(|id| Some(*id) != excluded_space && *id > 2 && *id < 4096)
            else {
                screening.reject("missing_opening_token");
                continue;
            };
            let mut candidate = Candidate {
                doc: i,
                component,
                component_id: &identities[&component],
                question: &question.input_text,
                answer_normalized: answer_norm,
                turn: j + 1,
                first,
                level: 0,
                split: component_splits[&component],
                family_split: families[&family].1,
                strict: None,
            };
            screening.screen(doc, &mut candidate, &offsets)?;
            candidates.push(candidate);
        }
    }
    let mut overlap_hits = BTreeMap::new();
    for (idx, (name, path, expected)) in HISTORICAL.into_iter().enumerate() {
        let bytes = pinned(&runs.join(path), expected)?;
        require(bytes.len() % 2 == 0, "misaligned token stream")?;
        let tokens: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        drop(bytes);
        overlap_hits.insert(
            name,
            json!({
                "additional_rejected_rows": overlap(&mut candidates, &tokens, 5 + idx)?,
                "path":path, "sha256":expected, "stream_tokens": tokens.len(),
            }),
        );
    }

    let mut stages = BTreeMap::new();
    for (level, name) in [
        "0_source_screen_and_relaxed_opening",
        "1_donor_preview_and_prior_questions",
        "2_all_original_offsets_rejoin",
        "3_all_answer_values_within_their_offsets",
        "4_canonical_roles_with_real_content_opening",
        "5_length_and_chunk",
        "6_after_historical_train",
        "7_after_historical_dev",
        "8_after_historical_prior",
    ]
    .into_iter()
    .enumerate()
    {
        let (summary, _) = pairing::summarize(
            candidates.iter().filter(|row| row.level >= level),
            pairing::Relation::Any,
        );
        stages.insert(name, summary);
    }
    fs::DirBuilder::new().mode(0o700).create(output)?;
    let mut graph = output_file(&output.join("eligible-graph.jsonl"))?;
    for row in candidates.iter().filter(|row| row.level == 8) {
        serde_json::to_writer(
            &mut graph,
            &json!({
                "component": sha(row.component_id.as_bytes()),
                "question": row.question,
                "answer": row.answer_normalized,
                "first": row.first,
                "split": row.split,
                "family_split": row.family_split,
                "turn": row.turn,
                "stratum": present(row.strict.as_ref(), "eligible graph row lacks source support")?.stratum,
            }),
        )?;
        writeln!(graph)?;
    }
    graph.flush()?;
    let mut splits = BTreeMap::new();
    let mut train_families = BTreeSet::new();
    let mut held_families = BTreeSet::new();
    for (label, split) in [("train", Split::Train), ("held", Split::Held)] {
        let rows = || {
            candidates
                .iter()
                .filter(|row| row.level == 8 && row.split == split)
        };
        let (component_only, _) = pairing::summarize(rows(), pairing::Relation::Any);
        let selected = || rows().filter(|row| row.split == row.family_split);
        let (summary, pairs) = pairing::summarize(selected(), pairing::Relation::Any);
        let (same_turn, _) = pairing::summarize(selected(), pairing::Relation::SameTurn);
        let (near_far, _) = pairing::summarize(selected(), pairing::Relation::NearFar);
        let families = if split == Split::Train {
            &mut train_families
        } else {
            &mut held_families
        };
        families.extend(selected().map(|row| inputs::normalize(row.question)));
        let mut file = output_file(&output.join(format!("{label}-pairs.jsonl")))?;
        for (index, (a, b)) in pairs.iter().enumerate() {
            let row = json!({
                "pair": index + 1, "split": split,
                "a": side(a, &input.data, &groups)?,
                "b": side(b, &input.data, &groups)?,
            });
            serde_json::to_writer(&mut file, &row)?;
            writeln!(file)?;
        }
        file.flush()?;
        splits.insert(label, json!({
            "inventory_components_before_eligibility": inventory.iter().filter(
                |(i, _)| split == component_splits[&components[*i]]
            ).map(|(i, _)| components[*i]).collect::<BTreeSet<_>>().len(),
            "component_partition_only":component_only, "component_and_question_family_partition":summary,
            "same_turn_ordinal_within_final_split":same_turn, "near_far_within_final_split":near_far,
        }));
    }
    require(
        train_families.is_disjoint(&held_families),
        "question family leakage across partitions",
    )?;
    let aggregate = json!({
        "evidence_class": "static authenticated corpus feasibility, no model execution",
        "repo_anchor": "6faaf355b1887ceb1a5dbb08398f31f401d18df4",
        "selector_sha256": sha(include_bytes!("../SELECTOR.md")),
        "tokenizer_protocol": renderer.protocol_identity()?,
        "preview_manifest_sha256": PREVIEW_MANIFEST,
        "historical_4352_sources": input.receipts,
        "prior_source_rosters": prior.receipts,
        "source_summary": {
            "articles": input.data.len(), "components": identities.len(),
            "components_excluded_by_4352_including_dev_and_previews": historical.len(),
            "components_excluded_by_4360_4361_and_lexical": prev_panels.len(),
            "components_excluded_by_union": historical.union(&prev_panels).count(),
            "articles_in_three_domains_with_valid_original_turn_inventory": inventory.len(),
            "prior_4360_4361_lexical_normalized_question_keys": prior.question_families.len(),
            "reserved_question_families_in_inventory": families.len(),
            "grounded_rows_all_evidence_offsets_exact_byte_equal_to_annotations": screening.byte_exact,
        },
        "rows_first_exclusion_by_reason":screening.rejections,
        "raw_stage_supply_before_partition": stages, "overlap":overlap_hits, "split_supply":splits,
        "normalized_selected_family_overlap": train_families.intersection(&held_families).count(),
    });
    let mut file = output_file(&output.join("aggregate.json"))?;
    serde_json::to_writer_pretty(&mut file, &aggregate)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn finite() -> Result<()> {
    let groups = [
        ["a".into(), "f1".into(), "p1".into()],
        ["b".into(), "f1".into(), "p2".into()],
        ["c".into(), "f2".into(), "p2".into()],
        ["d".into(), "f3".into(), "p3".into()],
    ];
    let joined = cluster(&groups);
    require(
        joined[0] == joined[2] && joined[0] != joined[3],
        "transitive component control failed",
    )?;
    let answer = |text, start, end| inputs::Turn {
        input_text: text,
        turn_id: 3,
        span_start: Some(start),
        span_end: Some(end),
        span_text: Some("A pear".to_owned()),
    };
    let story = "é A pear sits.";
    let story_offsets = offsets(story);
    require(
        source_support(story, &story_offsets, &answer("pear".into(), 2, 8))
            .is_some_and(|(s, literal)| s.bytes == (3..9) && literal && s.exact_raw_annotation),
        "valid multi-byte source offset rejected",
    )?;
    require(
        source_support(story, &story_offsets, &answer("pear".into(), 3, 8)).is_none(),
        "shifted source annotation accepted",
    )?;
    require(
        source_support(story, &story_offsets, &answer("pea".into(), 2, 8))
            .is_some_and(|(_, literal)| !literal),
        "partial word was admitted as a grounded answer",
    )?;
    let mut rows = pairing::finite_rows();
    pairing::finite_controls(&rows)?;
    let (summary, picked) = pairing::summarize(rows.iter(), pairing::Relation::Any);
    require(
        summary.upper_bound_disjoint_pairs == 2
            && picked.len() == 2
            && summary.literal_question_keys_with_pairs == 1,
        "exact-question / distinct-opener / component-disjoint finite fixture failed",
    )?;
    let window: Vec<u16> = (0..WINDOW as u16).collect();
    let retained = |tokens: Vec<u16>| Strict {
        tokens,
        prefix_len: 1,
        support: vec![],
        stratum: "near",
        later_in_chunk: false,
    };
    rows[0].strict = Some(retained(window.clone()));
    let mut changed = window.clone();
    changed[WINDOW - 1] += 1;
    rows[1].strict = Some(retained(changed));
    rows[0].level = 5;
    rows[1].level = 5;
    let tokens = [vec![31, 17, 19], window, vec![8, 7]].concat();
    require(
        overlap(&mut rows, &tokens, 5)? == 1 && rows[0].level == 5 && rows[1].level == 6,
        "exact overlap positive and one-ID mutant regression failed",
    )?;
    println!(
        "finite controls passed: Unicode scalar offsets, corruption, literal word boundaries, transitive components, exact-question pair, rolling 32-ID match and one-ID mutant"
    );
    Ok(())
}
