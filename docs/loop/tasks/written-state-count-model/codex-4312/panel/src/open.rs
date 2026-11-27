use crate::build::{bank_normalize, donor_continuations, donor_hit, Prior};
use crate::*;
use flate2::read::MultiGzDecoder;
use std::collections::{HashSet, VecDeque};
use uor_r4_tokenizer::dialogue::{DialogueEncoder, Message};

const OASST_SHA: &str = "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b";
const MIN_REQUEST_WORDS: usize = 4;

#[derive(Deserialize)]
struct Tree {
    message_tree_id: String,
    tree_state: String,
    prompt: Node,
}

#[derive(Deserialize)]
struct Node {
    message_id: String,
    text: String,
    role: String,
    lang: String,
    review_result: Option<bool>,
    deleted: bool,
    rank: Option<i64>,
    synthetic: bool,
    #[serde(default)]
    replies: Vec<Node>,
}

#[derive(Default, Serialize)]
pub(crate) struct Census {
    trees: usize,
    known_prior_tree: usize,
    historical_holdout: usize,
    no_admitted_open_turn: usize,
    duplicate_opening: usize,
    distinct_eligible_trees: usize,
}

fn usable(node: &Node, role: &str) -> bool {
    node.role == role
        && node.lang == "en"
        && node.review_result == Some(true)
        && !node.deleted
        && !node.synthetic
        && !node.text.trim().is_empty()
        && !node.text.contains("<|")
}

fn user_admitted(node: &Node, bank: &HashSet<String>, continuations: &HashSet<String>) -> bool {
    usable(node, "prompter")
        && !donor_hit(bank, &node.text)
        && !continuations.contains(&bank_normalize(&node.text))
}

fn open_ended(request: &str) -> bool {
    let terms = crate::rubric::words(request);
    const REQUESTS: &[&str] = &[
        "explain",
        "help",
        "suggest",
        "write",
        "show",
        "describe",
        "how",
        "outline",
        "argue",
        "teach",
        "translate",
        "invent",
        "rephrase",
        "generate",
        "provide",
        "ways",
        "more",
        "thought",
        "think",
        "opinion",
        "consider",
        "examples",
        "example",
        "story",
        "compare",
        "recommend",
        "improve",
        "rewrite",
        "design",
        "plan",
        "summarize",
        "elaborate",
        "detail",
        "discuss",
        "advise",
        "ideas",
        "why",
        "thoughts",
        "strategies",
    ];
    terms.len() >= MIN_REQUEST_WORDS
        && (terms.iter().any(|word| REQUESTS.contains(&word.as_str()))
            || matches!(terms.first().map(String::as_str), Some("what" | "which")))
}

fn select_history<'a>(
    tree: &'a Tree,
    bank: &HashSet<String>,
    continuations: &HashSet<String>,
    encoder: &DialogueEncoder<'_>,
    context: usize,
) -> Result<Option<(Vec<&'a Node>, Vec<u16>)>> {
    let mut paths = VecDeque::from([vec![&tree.prompt]]);
    while let Some(path) = paths.pop_front() {
        let node = path.last().ok_or("empty source path")?;
        if (path.len() == 3 || path.len() == 5) && open_ended(&node.text) {
            let messages: Vec<_> = path
                .iter()
                .map(|n| Message {
                    role: if n.role == "assistant" {
                        "assistant"
                    } else {
                        "user"
                    },
                    content: &n.text,
                })
                .collect();
            let prefix = encoder.encode_assistant_prefix(&messages);
            if prefix.tokens.len() <= context
                && prefix.special_token_occurrences == 0
                && prefix.emitted_turns == path.len()
                && prefix
                    .tokens
                    .iter()
                    .filter(|&&t| t == u32::from(EOS))
                    .count()
                    == path.len() / 2
            {
                return Ok(Some((path, ids(&prefix.tokens)?)));
            }
        }
        if path.len() == 5 {
            continue;
        }
        let mut children: Vec<_> = node
            .replies
            .iter()
            .filter(|next| {
                if node.role == "assistant" {
                    user_admitted(next, bank, continuations)
                } else {
                    usable(next, "assistant")
                }
            })
            .collect();
        children.sort_by(|a, b| {
            a.rank
                .unwrap_or(i64::MAX)
                .cmp(&b.rank.unwrap_or(i64::MAX))
                .then_with(|| a.message_id.cmp(&b.message_id))
        });
        for child in children {
            let mut next = path.clone();
            next.push(child);
            paths.push_back(next);
        }
    }
    Ok(None)
}

pub(crate) fn build(
    path: &Path,
    prior: &[Prior],
    bank: &HashSet<String>,
    encoder: &DialogueEncoder<'_>,
    context: usize,
) -> Result<(Vec<Prompt>, Vec<Value>, Census)> {
    check_pin(path, OASST_SHA)?;
    let prior: HashSet<_> = prior
        .iter()
        .filter_map(|p| p.id.strip_prefix("oasst2:"))
        .collect();
    if prior.is_empty() {
        return Err("no previously used OASST2 trees to exclude".into());
    }
    let continuations = donor_continuations()?;
    let mut census = Census::default();
    let mut candidates = Vec::new();
    for line in BufReader::new(MultiGzDecoder::new(File::open(path)?)).lines() {
        let tree: Tree = serde_json::from_str(&line?)?;
        census.trees += 1;
        if prior.contains(tree.message_tree_id.as_str()) {
            census.known_prior_tree += 1;
            continue;
        }
        let digest = Sha256::digest(tree.message_tree_id.as_bytes());
        let mut prefix_digest = [0; 8];
        prefix_digest.copy_from_slice(&digest[..8]);
        if u64::from_be_bytes(prefix_digest) % 1000 < 50 {
            census.historical_holdout += 1;
            continue;
        }
        let normalized = bank_normalize(&tree.prompt.text);
        if tree.tree_state != "ready_for_export"
            || !user_admitted(&tree.prompt, bank, &continuations)
        {
            census.no_admitted_open_turn += 1;
            continue;
        }
        let Some((history, prefix)) =
            select_history(&tree, bank, &continuations, encoder, context)?
        else {
            census.no_admitted_open_turn += 1;
            continue;
        };
        let id = opaque_id("open", &tree.message_tree_id);
        let messages: Vec<_> = history
            .iter()
            .map(|node| {
                json!({
                    "role":if node.role == "assistant" { "assistant" } else { "user" },
                    "content":node.text, "source_message_id":node.message_id,
                })
            })
            .collect();
        candidates.push((hash("open-rank", &tree.message_tree_id), normalized, tree.message_tree_id.clone(),
            Prompt { id: id.clone(), domain: Domain::Open, prefix, cap: OPEN_CAP },
            json!({"id":id,"messages":messages,"source_tree_id":tree.message_tree_id,
                "source":"OpenAssistant/oasst2","revision":"179dd21fc55192153d94adb0e0ce8f69e222bf75",
                "earlier_human_assistant_turns":history.len()/2}),
        ));
    }
    candidates.sort_by_key(|(rank, _, _, _, _)| *rank);
    let mut seen_opening = HashSet::new();
    let mut seen_tree = HashSet::new();
    let mut rows = Vec::new();
    let mut sources = Vec::new();
    for (_, opening, tree, prompt, source) in candidates {
        if !seen_opening.insert(opening) || !seen_tree.insert(tree) {
            census.duplicate_opening += 1;
            continue;
        }
        census.distinct_eligible_trees += 1;
        if rows.len() < OPEN_ROWS {
            rows.push(prompt);
            sources.push(source);
        }
    }
    if rows.len() != OPEN_ROWS {
        return Err(format!(
            "open human panel underfilled, {} of {OPEN_ROWS}; census {}",
            rows.len(),
            serde_json::to_string(&census)?
        )
        .into());
    }
    Ok((rows, sources, census))
}

pub(crate) const fn source_pin() -> &'static str {
    OASST_SHA
}
