use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

const SAMPLE_SALT: &str = "uor-r4/codex/natural-reciprocal-transfer-20261127/validation-passage/v1";
const QUOTA: usize = 4;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Pair {
    article: String,
    passage_sha256: String,
    left: String,
    right: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    articles: usize,
    passages: usize,
    pairs: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    Review,
    AlternateWithinPassage,
    OutsideArticleQuota,
}

#[derive(Serialize)]
struct Candidate {
    rank: usize,
    disposition: Disposition,
    pair: Pair,
}

#[derive(Serialize)]
struct Article {
    title: String,
    passages: usize,
    pairs: usize,
    selected: usize,
    shortage: usize,
    candidates: Vec<Candidate>,
}

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

fn invalid(message: &str) -> Error {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn passage_rank(family: &str, pair: &Pair) -> String {
    digest(format!(
        "{SAMPLE_SALT}\n{family}\n{}\n{}",
        pair.article, pair.passage_sha256
    ))
}

fn valid_text(text: &str) -> bool {
    !text.trim().is_empty() && !text.chars().any(char::is_control)
}

fn sha256_text(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn read_pairs(path: &Path) -> Result<(String, Vec<Pair>)> {
    let input = fs::read(path)?;
    let pairs = std::str::from_utf8(&input)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    Ok((digest(input), pairs))
}

fn register(family: &str, pairs: Vec<Pair>, expected: Shape) -> Result<Vec<Article>> {
    let length = pairs.len();
    let mut articles = BTreeMap::<String, BTreeMap<String, Vec<Pair>>>::new();
    let mut passages = BTreeMap::<String, String>::new();
    let mut questions = BTreeMap::<String, String>::new();
    for mut pair in pairs {
        if !valid_text(&pair.article)
            || !sha256_text(&pair.passage_sha256)
            || !valid_text(&pair.left)
            || !valid_text(&pair.right)
            || pair.left == pair.right
        {
            return Err(invalid("invalid validation candidate identity"));
        }
        if family == "two_answer" && pair.right < pair.left {
            std::mem::swap(&mut pair.left, &mut pair.right);
        }
        let title = passages
            .entry(pair.passage_sha256.clone())
            .or_insert_with(|| pair.article.clone());
        if *title != pair.article {
            return Err(invalid("passage is repeated across articles"));
        }
        for question in [&pair.left, &pair.right] {
            let passage = questions
                .entry(question.clone())
                .or_insert_with(|| pair.passage_sha256.clone());
            if *passage != pair.passage_sha256 {
                return Err(invalid("question is repeated across passages"));
            }
        }
        articles
            .entry(pair.article.clone())
            .or_default()
            .entry(pair.passage_sha256.clone())
            .or_default()
            .push(pair);
    }
    let found = Shape {
        articles: articles.len(),
        passages: passages.len(),
        pairs: length,
    };
    if found != expected {
        return Err(invalid(&format!(
            "{family} validation coverage differs from the seal: {found:?}, expected {expected:?}"
        )));
    }

    articles
        .into_iter()
        .map(|(title, passages)| rank_article(family, title, passages))
        .collect()
}

fn rank_article(
    family: &str,
    title: String,
    passages: BTreeMap<String, Vec<Pair>>,
) -> Result<Article> {
    let passages_count = passages.len();
    let mut ranked = passages.into_values().collect::<Vec<_>>();
    ranked.sort_by_cached_key(|pairs| {
        (
            passage_rank(family, &pairs[0]),
            pairs[0].passage_sha256.clone(),
        )
    });
    let selected = passages_count.min(QUOTA);
    let mut candidates = Vec::new();
    for (index, mut pairs) in ranked.into_iter().enumerate() {
        pairs.sort_unstable();
        if pairs.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid("duplicate mechanical pair"));
        }
        for (pair_index, pair) in pairs.into_iter().enumerate() {
            let disposition = match (pair_index == 0, index < QUOTA) {
                (true, true) => Disposition::Review,
                (true, false) => Disposition::OutsideArticleQuota,
                (false, _) => Disposition::AlternateWithinPassage,
            };
            candidates.push(Candidate {
                rank: index + 1,
                disposition,
                pair,
            });
        }
    }
    Ok(Article {
        title,
        passages: passages_count,
        pairs: candidates.len(),
        selected,
        shortage: QUOTA - selected,
        candidates,
    })
}

fn main() -> Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    let [positives, answerability] = paths.as_slice() else {
        return Err(invalid(
            "usage: validation-roster TWO_ANSWER_METADATA.jsonl ANSWERABILITY_METADATA.jsonl",
        ));
    };
    let (positive_hash, positive_pairs) = read_pairs(Path::new(positives))?;
    let (answerability_hash, answerability_pairs) = read_pairs(Path::new(answerability))?;
    let two_answer = register(
        "two_answer",
        positive_pairs,
        Shape {
            articles: 7,
            passages: 77,
            pairs: 187,
        },
    )?;
    let answerability = register(
        "answerability",
        answerability_pairs,
        Shape {
            articles: 11,
            passages: 256,
            pairs: 256,
        },
    )?;
    if two_answer
        .iter()
        .any(|left| answerability.iter().any(|right| left.title == right.title))
    {
        return Err(invalid("article belongs to both validation families"));
    }
    let quota_shortfall = two_answer
        .iter()
        .chain(&answerability)
        .any(|article| article.shortage > 0);
    let manifest = serde_json::json!({
        "schema": "natural-reciprocal-validation-selection/1",
        "salt": SAMPLE_SALT,
        "quota_per_article": QUOTA,
        "quota_shortfall": quota_shortfall,
        "input_sha256": {"two_answer": positive_hash, "answerability": answerability_hash},
        "two_answer": two_answer,
        "answerability": answerability,
    });
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> Vec<Pair> {
        let article = "Synthetic Δ".to_string();
        let mut pairs: Vec<_> = (0..6)
            .map(|index| Pair {
                article: article.clone(),
                passage_sha256: digest(index.to_string()),
                left: format!("l{index}"),
                right: format!("r{index}"),
            })
            .collect();
        let mut alternate = pairs[0].clone();
        alternate.right = "z0".to_owned();
        pairs.push(alternate);
        pairs
    }

    #[test]
    fn rank_is_order_invariant_and_accounts_for_every_pair() -> Result<()> {
        let shape = Shape {
            articles: 1,
            passages: 6,
            pairs: 7,
        };
        let mut pairs = synthetic();
        let first = register("two_answer", pairs.clone(), shape)?;
        pairs.reverse();
        let second = register("two_answer", pairs, shape)?;
        assert_eq!(serde_json::to_vec(&first)?, serde_json::to_vec(&second)?);
        let article = &first[0];
        assert_eq!(article.selected, 4);
        assert_eq!(article.pairs, 7);
        let reviewed: Vec<_> = article
            .candidates
            .iter()
            .filter(|candidate| matches!(candidate.disposition, Disposition::Review))
            .map(|candidate| candidate.pair.left.as_str())
            .collect();
        assert_eq!(reviewed, ["l0", "l1", "l5", "l4"]);
        assert_eq!(
            article
                .candidates
                .iter()
                .filter(|candidate| matches!(
                    candidate.disposition,
                    Disposition::AlternateWithinPassage
                ))
                .count(),
            1
        );
        assert_eq!(
            article
                .candidates
                .iter()
                .filter(|candidate| matches!(
                    candidate.disposition,
                    Disposition::OutsideArticleQuota
                ))
                .count(),
            2
        );
        Ok(())
    }

    #[test]
    fn accepts_shortage_as_a_report_and_rejects_duplicate_pair() -> Result<()> {
        let mut pairs = synthetic();
        pairs.truncate(2);
        assert_eq!(
            register(
                "two_answer",
                pairs.clone(),
                Shape {
                    articles: 1,
                    passages: 2,
                    pairs: 2
                }
            )?[0]
                .shortage,
            2
        );
        pairs.push(pairs[0].clone());
        assert!(
            register(
                "two_answer",
                pairs,
                Shape {
                    articles: 1,
                    passages: 2,
                    pairs: 3
                }
            )
            .is_err()
        );
        Ok(())
    }
}
