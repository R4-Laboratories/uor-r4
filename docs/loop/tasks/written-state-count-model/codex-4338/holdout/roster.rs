use super::*;
use serde_json::{json, Value};
use std::cmp::Reverse;

const SALT: &str = "uor-r4/codex-4338/published-dev-answerability-v1";
const PARENT: &str = "20261127-codex-grounded-interval";
const MAX_COPY: usize = 8;

fn digest(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn rank(kind: &str, value: &str) -> String {
    digest(format!("{SALT}/{kind}\n{value}"))
}

fn read_rows(
    runs: &Path,
    relative: &str,
    receipts: &mut BTreeMap<String, String>,
) -> Result<Vec<Value>> {
    let bytes = fs::read(runs.join(relative))?;
    receipts.insert(relative.to_owned(), digest(&bytes));
    BufReader::new(bytes.as_slice())
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn protect(blocked: &mut Blocklist, article: &Article) {
    blocked.titles.insert(normalize(&article.title));
    for paragraph in &article.paragraphs {
        let normalized = normalize(&paragraph.context);
        blocked.shingles.extend(shingles(&normalized));
        blocked.passages.insert(normalized);
        blocked
            .prior_questions
            .extend(paragraph.qas.iter().map(|q| normalize(&q.question)));
    }
}

fn annotate_window(
    question: &Question,
    annotation: &Annotation,
    char_offset: usize,
) -> Result<Question> {
    let entry = Annotation {
        text: annotation.text.clone(),
        answer_start: annotation
            .answer_start
            .checked_sub(char_offset)
            .ok_or("window dropped annotation")?,
    };
    // Full-source annotation consensus is checked before remapping. The
    // rendering check needs only its one already verified representative.
    let (answers, plausible_answers) = if question.is_impossible {
        (Vec::new(), vec![entry])
    } else {
        (vec![entry], Vec::new())
    };
    Ok(Question {
        id: question.id.clone(),
        question: question.question.clone(),
        is_impossible: question.is_impossible,
        answers,
        plausible_answers,
    })
}

fn boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = vec![0];
    let mut delimiter = false;
    for (at, ch) in text.char_indices() {
        if delimiter && ch.is_whitespace() {
            boundaries.push(at);
        }
        delimiter = matches!(ch, '.' | '?' | '!');
    }
    boundaries.push(text.len());
    boundaries
}

fn window_ranges(text: &str, cover: [usize; 2]) -> Vec<[usize; 2]> {
    let boundaries = boundaries(text);
    let mut candidates = vec![[0, text.len()]];
    for (index, &start) in boundaries.iter().enumerate() {
        for &end in &boundaries[index + 1..] {
            let source = text[start..end].trim();
            let start = start + text[start..end].len() - text[start..end].trim_start().len();
            if start <= cover[0] && cover[1] <= start + source.len() {
                candidates.push([start, start + source.len()]);
            }
        }
    }
    candidates.sort_by_key(|&[start, end]| (Reverse(end - start), start));
    candidates.dedup();
    candidates
}

fn section(
    candidate: &Candidate<'_>,
    original: [usize; 2],
    annotation: &Annotation,
    agreement: usize,
) -> Value {
    json!({
        "published_id": candidate.question.id,
        "question": candidate.question.question,
        "published_impossible": candidate.question.is_impossible,
        "prompt": candidate.body,
        "prefix": candidate.input,
        "user_bounds": candidate.user_bounds,
        "original_annotation_bytes": original,
        "original_annotation_char": annotation.answer_start,
        "published_annotation_agreement": agreement,
        "label": candidate.label,
    })
}

fn prepare_pair(
    render: &Render<'_>,
    blocked: &Blocklist,
    paragraph: &Paragraph,
    positive: &Question,
    negative: &Question,
    seen_questions: &HashSet<String>,
) -> Result<Option<Value>> {
    let Ok((pos_annot, pos_range, pos_agreement)) = verified_annotation(paragraph, positive, true)
    else {
        return Ok(None);
    };
    let Ok((neg_annot, neg_range, neg_agreement)) = verified_annotation(paragraph, negative, true)
    else {
        return Ok(None);
    };
    let cover = [
        pos_range[0].min(neg_range[0]),
        pos_range[1].max(neg_range[1]),
    ];
    for [start, end] in window_ranges(&paragraph.context, cover) {
        let char_offset = paragraph.context[..start].chars().count();
        let pos = annotate_window(positive, pos_annot, char_offset)?;
        let neg = annotate_window(negative, neg_annot, char_offset)?;
        let window = Paragraph {
            context: paragraph.context[start..end].to_owned(),
            qas: Vec::new(),
        };
        let Ok(p) = render.candidate(&window, &pos, false, blocked, seen_questions) else {
            continue;
        };
        let Ok(n) = render.candidate(&window, &neg, false, blocked, seen_questions) else {
            continue;
        };
        if !p.label.whole_span_ids || p.label.source_ids.len() > MAX_COPY {
            continue;
        }
        let raw_passage = format!(" {}", window.context);
        let passage_ids = render.tokenizer.encode(&raw_passage);
        let body_start = p.user_bounds[0];
        let body_end = body_start + passage_ids.len();
        if p.input.get(body_start..body_end) != Some(passage_ids.as_slice())
            || n.input.get(body_start..body_end) != Some(passage_ids.as_slice())
        {
            continue;
        }
        return Ok(Some(json!({
            "window_bytes": [start, end],
            "full_paragraph": start == 0 && end == paragraph.context.len(),
            "shared_passage_ids": passage_ids,
            "shared_passage": window.context,
            "positive": section(&p, pos_range, pos_annot, pos_agreement),
            "negative": section(&n, neg_range, neg_annot, neg_agreement),
        })));
    }
    Ok(None)
}

pub fn run(runs: &Path, inputs: &Path, output: &Path) -> Result<()> {
    let train: Dataset = serde_json::from_slice(&checked_bytes(&inputs.join(TRAIN), PINS[0].1)?)?;
    let dev: Dataset = serde_json::from_slice(&checked_bytes(&inputs.join(DEV), PINS[1].1)?)?;
    if train.version != "v2.0" || dev.version != "v2.0" {
        return Err(bad_input("unexpected official SQuAD version"));
    }
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked_bytes(
        &runs.join(TOKENIZER),
        PINS[2].1,
    )?)
    .ok_or("invalid tokenizer")?;
    let render = Render::new(&tokenizer)?;
    let mut counts = Counts::new();
    let mut blocked = Blocklist::load(runs)?;
    let mut receipts = BTreeMap::new();
    let mut seen_source_ids = HashSet::new();
    for article in &train.data {
        protect(&mut blocked, article);
        for question in article.paragraphs.iter().flat_map(|p| &p.qas) {
            seen_source_ids.insert(question.id.clone());
        }
    }
    for relative in history::POOLS.iter().copied().chain([
        "20261127-codex-grounded-interval/data/gate-v2/train-truth.jsonl",
        "20261127-codex-grounded-interval/data/gate-v2/protected-truth.jsonl",
        "20261127-codex-grounded-interval/data/fit-v1/train-truth.jsonl",
        "20261127-codex-grounded-interval/data/held-v1/held-truth.jsonl",
        "20261127-codex-grounded-interval/next-supply-v1/questions.jsonl",
    ]) {
        for row in read_rows(runs, relative, &mut receipts)? {
            if let Some(article) = row["article"]
                .as_str()
                .or_else(|| row["source_origin"]["article"].as_str())
            {
                blocked.titles.insert(normalize(article));
            }
            if let Some(question) = row["question"].as_str() {
                blocked.prior_questions.insert(normalize(question));
            }
            for field in ["case_id", "source_id"] {
                if let Some(id) = row[field].as_str() {
                    seen_source_ids.insert(id.to_owned());
                }
            }
            if let Some(text) = ["text", "prompt", "user"]
                .into_iter()
                .find_map(|field| row[field].as_str())
            {
                let normalized = normalize(text);
                blocked.shingles.extend(shingles(&normalized));
                blocked.prior_prompts.insert(normalized);
            }
        }
    }
    let split_tsv = fs::read(runs.join(PARENT).join("next-supply-pilot-v1/split.tsv"))?;
    receipts.insert(
        format!("{PARENT}/next-supply-pilot-v1/split.tsv"),
        digest(&split_tsv),
    );
    for line in std::str::from_utf8(&split_tsv)?.lines() {
        let title: String =
            serde_json::from_str(line.split('\t').nth(2).ok_or("bad positive split line")?)?;
        blocked.titles.insert(normalize(&title));
    }
    let mut titles: Vec<_> = dev
        .data
        .iter()
        .filter(|a| !blocked.titles.contains(&normalize(&a.title)))
        .collect();
    titles.sort_by_cached_key(|a| rank("article", &normalize(&a.title)));
    count(&mut counts, "official_dev_articles", dev.data.len());
    count(&mut counts, "unexposed_title_candidates", titles.len());
    fs::create_dir(output)?;
    fs::write(output.join("prior-positive-split.tsv"), split_tsv)?;
    let mut roster = BufWriter::new(File::create(output.join("roster.jsonl"))?);
    let mut title_split = BufWriter::new(File::create(output.join("articles.jsonl"))?);
    let mut seen_questions = HashSet::new();
    let mut seen_passages = HashSet::new();
    let mut seen_shingles = HashSet::new();
    let mut min_prefix = usize::MAX;
    let mut max_prefix = 0;
    let mut max_copy = 0;
    for (article_index, article) in titles.into_iter().enumerate() {
        let split = if article_index % 3 == 0 {
            "validation"
        } else {
            "locked"
        };
        jsonl(
            &mut title_split,
            &json!({"article": article.title, "split": split}),
        )?;
        let mut article_pairs = 0;
        for (paragraph_index, paragraph) in article.paragraphs.iter().enumerate() {
            count(&mut counts, "paragraphs_screened", 1);
            let normalized = normalize(&paragraph.context);
            if blocked.passages.contains(&normalized)
                || seen_passages.contains(&normalized)
                || shingles(&normalized)
                    .any(|s| blocked.shingles.contains(&s) || seen_shingles.contains(&s))
            {
                count(&mut counts, "blocked_passage", 1);
                continue;
            }
            if paragraph.context.trim() != paragraph.context
                || paragraph.context.contains(['\n', '\r'])
            {
                count(&mut counts, "not_verbatim_single_line", 1);
                continue;
            }
            let mut questions: Vec<_> = paragraph
                .qas
                .iter()
                .filter(|q| {
                    !seen_source_ids.contains(&q.id)
                        && !blocked.prior_questions.contains(&normalize(&q.question))
                })
                .collect();
            questions.sort_by(|a, b| a.id.cmp(&b.id));
            let mut pairs = Vec::new();
            for p in questions.iter().copied().filter(|q| !q.is_impossible) {
                for n in questions.iter().copied().filter(|q| q.is_impossible) {
                    if normalize(&p.question) == normalize(&n.question) {
                        continue;
                    }
                    if let Some(pair) =
                        prepare_pair(&render, &blocked, paragraph, p, n, &seen_questions)?
                    {
                        let full = pair["full_paragraph"]
                            .as_bool()
                            .ok_or("missing full paragraph flag")?;
                        let same_span = pair["positive"]["original_annotation_bytes"]
                            == pair["negative"]["original_annotation_bytes"];
                        let edits = question_edit(&normalize(&p.question), &normalize(&n.question));
                        pairs.push(((!full, !same_span, edits, &p.id, &n.id), pair));
                    }
                }
            }
            let Some((_, mut pair)) = pairs.into_iter().min_by(|a, b| a.0.cmp(&b.0)) else {
                count(&mut counts, "no_eligible_pair", 1);
                continue;
            };
            seen_passages.insert(normalized.clone());
            seen_shingles.extend(shingles(&normalized));
            for side in ["positive", "negative"] {
                let row = &pair[side];
                let prefix_count = row["prefix"].as_array().ok_or("missing prefix")?.len();
                min_prefix = min_prefix.min(prefix_count);
                max_prefix = max_prefix.max(prefix_count);
                seen_source_ids
                    .insert(row["published_id"].as_str().ok_or("missing ID")?.to_owned());
                seen_questions.insert(normalize(
                    row["question"].as_str().ok_or("missing question")?,
                ));
            }
            max_copy = max_copy.max(
                pair["positive"]["label"]["source_ids"]
                    .as_array()
                    .ok_or("missing source IDs")?
                    .len(),
            );
            pair["article"] = json!(article.title);
            pair["paragraph"] = json!(paragraph_index);
            pair["passage_sha256"] = json!(digest(paragraph.context.as_bytes()));
            pair["split"] = json!(split);
            let full = pair["full_paragraph"]
                .as_bool()
                .ok_or("missing full paragraph flag")?;
            count(
                &mut counts,
                match (split, full) {
                    ("validation", true) => "validation_full_pairs",
                    ("validation", false) => "validation_window_pairs",
                    (_, true) => "locked_full_pairs",
                    (_, false) => "locked_window_pairs",
                },
                1,
            );
            article_pairs += 1;
            jsonl(&mut roster, &pair)?;
        }
        if article_pairs > 0 {
            count(
                &mut counts,
                if split == "validation" {
                    "validation_contributing_articles"
                } else {
                    "locked_contributing_articles"
                },
                1,
            );
        }
    }
    roster.flush()?;
    title_split.flush()?;
    if counts
        .get("locked_contributing_articles")
        .copied()
        .unwrap_or_default()
        == 0
        || counts
            .get("validation_contributing_articles")
            .copied()
            .unwrap_or_default()
            == 0
    {
        return Err(bad_input("answerability partition has no rows"));
    }
    let mut summary = BufWriter::new(File::create(output.join("summary.json"))?);
    serde_json::to_writer_pretty(
        &mut summary,
        &json!({
            "schema": "natural-answerability-roster/1",
            "pins": PINS,
            "history_files": receipts,
            "counts": counts,
            "minimum_prefix_ids": min_prefix,
            "maximum_prefix_ids": max_prefix,
            "maximum_positive_copy_ids": max_copy,
            "semantic_adjudication": "NOT_RUN; published labels only",
        }),
    )?;
    writeln!(summary)?;
    summary.flush()?;
    Ok(())
}
