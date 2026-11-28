use crate::{Candidate, Split};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
pub enum Relation {
    Any,
    SameTurn,
    NearFar,
}

fn compatible(a: &Candidate<'_>, b: &Candidate<'_>, relation: Relation) -> bool {
    if a.component == b.component
        || a.question != b.question
        || a.first == b.first
        || a.answer_normalized == b.answer_normalized
    {
        return false;
    }
    match relation {
        Relation::Any => true,
        Relation::SameTurn => a.turn == b.turn,
        Relation::NearFar => matches!(
            (
                a.strict.as_ref().map(|s| s.stratum),
                b.strict.as_ref().map(|s| s.stratum)
            ),
            (Some("near"), Some("far")) | (Some("far"), Some("near"))
        ),
    }
}

#[derive(Serialize)]
pub struct Counts {
    rows: usize,
    components: usize,
    literal_question_keys: usize,
    pub literal_question_keys_with_pairs: usize,
    literal_candidate_row_pairs: usize,
    participating_components: usize,
    sum_local_component_pair_bound: usize,
    pub upper_bound_disjoint_pairs: usize,
    greedy_disjoint_pairs: usize,
}

pub fn summarize<'a>(
    candidates: impl Iterator<Item = &'a Candidate<'a>>,
    relation: Relation,
) -> (Counts, Vec<(&'a Candidate<'a>, &'a Candidate<'a>)>) {
    let mut questions = BTreeMap::<&str, Vec<&Candidate<'_>>>::new();
    let mut components = BTreeSet::new();
    let mut count = 0;
    for row in candidates {
        questions.entry(row.question).or_default().push(row);
        components.insert(row.component);
        count += 1;
    }
    let mut participating = BTreeSet::new();
    let mut used = BTreeSet::new();
    let mut pairs = Vec::new();
    let mut useful_questions = 0;
    let mut edges = 0;
    let mut local_bound = 0;
    for rows in questions.values_mut() {
        rows.sort_unstable_by_key(|row| (row.component, row.turn, row.doc));
        let mut local_matches = BTreeSet::new();
        for (i, &a) in rows.iter().enumerate() {
            for &b in &rows[i + 1..] {
                if !compatible(a, b, relation) {
                    continue;
                }
                edges += 1;
                local_matches.extend([a.component, b.component]);
                if !used.contains(&a.component) && !used.contains(&b.component) {
                    used.extend([a.component, b.component]);
                    pairs.push((a, b));
                }
            }
        }
        if !local_matches.is_empty() {
            useful_questions += 1;
            local_bound += local_matches.len() / 2;
            participating.extend(local_matches);
        }
    }
    (
        Counts {
            rows: count,
            components: components.len(),
            literal_question_keys: questions.len(),
            literal_question_keys_with_pairs: useful_questions,
            literal_candidate_row_pairs: edges,
            participating_components: participating.len(),
            sum_local_component_pair_bound: local_bound,
            upper_bound_disjoint_pairs: local_bound.min(participating.len() / 2),
            greedy_disjoint_pairs: pairs.len(),
        },
        pairs,
    )
}

pub fn finite_rows() -> Vec<Candidate<'static>> {
    [
        (1, "Where is it?", "attic", 5),
        (2, "Where is it?", "cellar", 6),
        (1, "Where is it?", "shed", 7),
        (3, "Where is it? ", "yard", 8),
        (4, "Where is it?", "attic", 6),
        (5, "Where is it?", "cellar", 5),
    ]
    .into_iter()
    .map(
        |(component, question, answer_normalized, first)| Candidate {
            doc: component,
            component,
            component_id: "",
            question,
            answer_normalized: answer_normalized.into(),
            turn: 3,
            first,
            level: 0,
            split: Split::Train,
            family_split: Split::Train,
            strict: None,
        },
    )
    .collect()
}

pub fn finite_controls(rows: &[Candidate<'_>]) -> crate::Result<()> {
    let a = &rows[0];
    crate::require(
        compatible(a, &rows[1], Relation::Any),
        "valid pair rejected",
    )?;
    crate::require(
        !compatible(a, &rows[2], Relation::Any),
        "same component admitted",
    )?;
    crate::require(
        !compatible(a, &rows[3], Relation::Any),
        "nonliteral question admitted",
    )?;
    crate::require(
        !compatible(a, &rows[4], Relation::Any),
        "identical normalized answer admitted",
    )?;
    crate::require(
        !compatible(a, &rows[5], Relation::Any),
        "identical first ID admitted",
    )
}
