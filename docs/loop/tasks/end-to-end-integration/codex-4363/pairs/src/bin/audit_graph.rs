use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct Row {
    question: String,
    component: String,
    answer: String,
    first: u32,
    split: String,
    family_split: String,
    turn: usize,
    stratum: String,
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn row_edges<'a>(
    rows: impl Iterator<Item = &'a Row>,
    relation: impl Fn(&Row, &Row) -> bool,
) -> Value {
    let mut by_prompt = BTreeMap::<&str, Vec<&Row>>::new();
    let mut counted_rows = 0;
    for row in rows {
        by_prompt.entry(&row.question).or_default().push(row);
        counted_rows += 1;
    }
    let mut all_members = HashSet::new();
    let mut distinct_component_edges = BTreeSet::new();
    let mut question_edges = 0;
    let mut row_edges = 0;
    let mut local_bound = 0;
    for group in by_prompt.values() {
        let mut local_members = HashSet::new();
        for a in group {
            for b in group.iter().filter(|b| a.component < b.component) {
                if a.first != b.first && a.answer != b.answer && relation(a, b) {
                    row_edges += 1;
                    local_members.extend([&a.component, &b.component]);
                    distinct_component_edges.insert((&a.component, &b.component));
                }
            }
        }
        question_edges += usize::from(!local_members.is_empty());
        local_bound += local_members.len() / 2;
        all_members.extend(local_members);
    }
    let mut used = HashSet::<&String>::new();
    let mut independent_greedy = 0;
    for (a, b) in &distinct_component_edges {
        if !used.contains(a) && !used.contains(b) {
            used.extend([*a, *b]);
            independent_greedy += 1;
        }
    }
    json!({
        "eligible_rows": counted_rows,
        "literal_question_keys_with_edges": question_edges,
        "eligible_row_edges": row_edges,
        "distinct_component_edges": distinct_component_edges.len(),
        "greedy_matching_by_component_digest": independent_greedy,
        "upper_bound_disjoint_component_pairs": local_bound.min(all_members.len() / 2),
    })
}

fn check_pack(
    all: &[Row],
    packed: &[Value],
    split: &str,
    used: &mut BTreeSet<String>,
) -> Result<()> {
    for (index, pair) in packed.iter().enumerate() {
        if pair["split"] != split || pair["pair"] != index + 1 {
            return Err("packed split or pair ID mismatch".into());
        }
        let mut matches = Vec::new();
        for side in ["a", "b"] {
            let item = &pair[side];
            let component = item["component_sha256"]
                .as_str()
                .ok_or("missing source component")?;
            if !used.insert(component.to_owned()) {
                return Err("component reused in packing".into());
            }
            let expected = item["first_content_id"]
                .as_u64()
                .ok_or("missing actual first ID")?;
            if item["target"][0].as_u64() != Some(expected) {
                return Err("actual opening token differs".into());
            }
            let question = item["role_messages"]
                .as_array()
                .and_then(|rows| rows.last())
                .and_then(|final_user| final_user["content"].as_str())
                .ok_or("missing final user question")?;
            if item["literal_question_sha256"] != hex::encode(Sha256::digest(question.as_bytes())) {
                return Err("original question bytes differ from their digest".into());
            }
            let row = all
                .iter()
                .find(|row| {
                    row.component == component
                        && row.question == question
                        && row.turn as u64 == item["turn"].as_u64().unwrap_or_default()
                        && u64::from(row.first) == expected
                        && row.split == split
                        && row.family_split == split
                })
                .ok_or("packed side missing from independently recounted graph")?;
            matches.push(row);
        }
        let [a, b] = matches.as_slice() else {
            return Err("incomplete packed pair".into());
        };
        if a.question != b.question || a.answer == b.answer || a.first == b.first {
            return Err("packed sides do not establish the opposed exact-question endpoint".into());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let out = std::env::args()
        .nth(1)
        .ok_or("expected data build directory")?;
    let out = Path::new(&out);
    let rows: Vec<Row> = read(&out.join("eligible-graph.jsonl"))?;
    let mut used = BTreeSet::new();
    let mut split = BTreeMap::new();
    for name in ["train", "held"] {
        let in_component_split = || rows.iter().filter(|row| row.split == name);
        let final_graph = || in_component_split().filter(|row| row.family_split == name);
        let packed = read(&out.join(format!("{name}-pairs.jsonl")))?;
        check_pack(&rows, &packed, name, &mut used)?;
        split.insert(name, json!({
            "component_partition_only":row_edges(in_component_split(), |_, _| true),
            "component_and_question_family_partition":row_edges(final_graph(), |_, _| true),
            "same_turn":row_edges(final_graph(), |a, b| a.turn == b.turn),
            "near_far":row_edges(final_graph(), |a, b| matches!((a.stratum.as_str(), b.stratum.as_str()), ("near", "far") | ("far", "near"))),
            "validated_actual_packing":packed.len(),
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "whole_graph":row_edges(rows.iter(), |_, _| true), "splits":split,
        }))?
    );
    Ok(())
}
