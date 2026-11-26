use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{json, Value};

use crate::packet::{Address, Row};
use crate::{check, read_json, read_lines, sha, Result};

pub fn address(span: Address) -> Value {
    json!({"bytes": span.bytes, "tokens": span.tokens})
}

fn interval(raw: &Value, adjustment: u64) -> Result<Value> {
    let [a, b] = if let Some(values) = raw.as_array().filter(|v| v.len() == 2) {
        [&values[0], &values[1]]
    } else {
        [&raw["start"], &raw["end"]]
    };
    let offset = |v: &Value| -> Result<u64> {
        Ok(v.as_u64()
            .and_then(|n| n.checked_sub(adjustment))
            .ok_or("invalid original interval")?)
    };
    let a = offset(a)?;
    let b = offset(b)?;
    check(a < b, "empty original interval")?;
    Ok(json!([a, b]))
}

fn span(raw: &Value, adjustment: u64) -> Result<Value> {
    Ok(
        json!({"bytes": interval(&raw["bytes"], adjustment)?, "tokens": interval(&raw["tokens"], 0)?}),
    )
}

fn field(value: &Value, name: &str) -> Result<String> {
    Ok(value[name]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing {name}"))?
        .to_owned())
}

pub struct Reference {
    pub pair: String,
    pub id: String,
    pub digest: String,
    pub admission: String,
    pub distance: String,
    pub family: String,
    pub gold: [Value; 3],
    pub governors: [Value; 2],
    pub text: String,
    pub prompt: Vec<u32>,
    pub history: Value,
    pub selector: Option<Value>,
}

impl Reference {
    pub fn read(
        folder: &Path,
        coordinates: &Path,
        rows: &[Row],
        labels: &[[[usize; 2]; 2]],
    ) -> Result<Vec<Self>> {
        let data = read_json(&folder.join("reference.json"))?;
        let refs = data.as_array().ok_or("missing reference rows")?;
        let coords = read_lines(coordinates)?;
        check(
            refs.len() == rows.len() && rows.len() == coords.len() && rows.len() == labels.len(),
            "wrong reference cardinality",
        )?;
        let mut ids = BTreeSet::new();
        let mut digests = BTreeSet::new();
        let mut result = Vec::with_capacity(rows.len());
        for (i, (((row, label), item), coord)) in
            rows.iter().zip(labels).zip(refs).zip(&coords).enumerate()
        {
            let id = field(item, "id")?;
            let digest = field(item, "structural_prompt_sha256")?;
            check(
                ids.insert(id.clone()) && digests.insert(digest.clone()),
                "repeated prompt identity or digest",
            )?;
            check(
                coord["ordinal"] == i && coord["id"] == id && coord["prompt_sha256"] == digest,
                format!("coordinate identity differs at {i}"),
            )?;
            check(
                coord["tuples"] == json!(row.tuples),
                format!("physical tuple order differs at {i}"),
            )?;
            let text = item["turns"][1]
                .as_str()
                .ok_or("original user turn missing")?;
            let prompt: Vec<u32> = serde_json::from_value(item["prompt"].clone())?;
            let tape = prompt
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>();
            check(
                sha(&tape) == digest && sha(text.as_bytes()) == id,
                format!("original prompt or BPE digest mismatch at {i}"),
            )?;
            for (name, occurrences) in [("words", &row.words), ("sites", &row.sites)] {
                let expected = coord[name]
                    .as_array()
                    .ok_or("original tape has no occurrences")?;
                check(
                    expected.len() == occurrences.len(),
                    format!("original {name} count differs at {i}"),
                )?;
                for (site, frozen) in occurrences.iter().zip(expected) {
                    let [start, end] = site.address.bytes.map(|v| v as usize);
                    check(
                        address(site.address) == span(frozen, 0)? && text.get(start..end).is_some(),
                        format!("original {name} address differs at {i}"),
                    )?;
                }
            }
            let target = item["native_scope"]
                .as_array()
                .filter(|x| x.len() == 3)
                .ok_or("wrong original target arity")?;
            let gold = [
                span(&target[0], 1)?,
                span(&target[1], 1)?,
                span(&target[2], 1)?,
            ];
            let governors = [
                span(&item["governors"]["affirmative"], 0)?,
                span(&item["governors"]["excluded"], 0)?,
            ];
            for (sign, [word, site]) in label.iter().copied().enumerate() {
                let gov = row
                    .words
                    .get(word)
                    .ok_or("reference word outside physical row")?;
                let noun = row
                    .sites
                    .get(site)
                    .ok_or("reference noun outside physical row")?;
                check(
                    address(gov.address) == governors[sign] && address(noun.address) == gold[sign],
                    format!("original signed supervision differs at {i} sign {sign}"),
                )?;
            }
            let head = row
                .sites
                .iter()
                .position(|site| address(site.address) == gold[2])
                .ok_or("heading outside original tape")?;
            check(
                row.tuples.contains(&[label[0][1], label[1][1], head]),
                format!("original tuple unavailable at {i}"),
            )?;
            result.push(Self {
                pair: field(item, "pair_id")?,
                id,
                digest,
                admission: field(item, "admission")?,
                distance: field(item, "distance")?,
                family: field(item, "family")?,
                gold,
                governors,
                text: text.to_owned(),
                prompt,
                history: item["structural_history"].clone(),
                selector: item.get("contrast").cloned(),
            });
        }
        Ok(result)
    }

    fn earliest(&self, got: &Value) -> Option<usize> {
        let addresses = got["tuple"]["addresses"].as_array()?;
        (0..3)
            .find(|&i| {
                addresses
                    .get(i)
                    .is_none_or(|actual| actual != &self.gold[i])
            })
            .or_else(|| {
                (0..2)
                    .find(|&i| got["links"][i]["governor"] != self.governors[i])
                    .map(|i| i + 3)
            })
    }

    pub fn exact(&self, got: &Value) -> bool {
        got["tuple"]["ties"] == 1
            && got["tuple"]["addresses"]
                .as_array()
                .is_some_and(|v| v == &self.gold)
    }

    pub fn full(&self, got: &Value) -> bool {
        self.exact(got)
            && (0..2).all(|sign| {
                got["links"][sign]["governor"] == self.governors[sign]
                    && got["links"][sign]["noun"] == self.gold[sign]
            })
    }

    fn site(&self, span: &Value) -> Value {
        let interval = span["bytes"]
            .as_array()
            .and_then(|a| a.first().zip(a.get(1)));
        let surface = interval.and_then(|(from, to)| {
            self.text
                .get(from.as_u64()? as usize..to.as_u64()? as usize)
        });
        json!({"address":span,"surface":surface})
    }

    pub fn witness(&self, ordinal: usize, got: &Value) -> Value {
        let first = self.earliest(got);
        let kind = first.map(|i| {
            [
                "affirmed",
                "excluded",
                "heading",
                "affirmative_governor",
                "excluded_governor",
            ][i]
        });
        let expected = first.map(|i| {
            if i < 3 {
                self.site(&self.gold[i])
            } else {
                self.site(&self.governors[i - 3])
            }
        });
        let observed = first.map(|i| {
            if i < 3 {
                self.site(&got["tuple"]["addresses"][i])
            } else {
                self.site(&got["links"][i - 3]["governor"])
            }
        });
        json!({"row":ordinal,"case_id":self.id,"pair_id":self.pair,"distance":self.distance,"admission":self.admission,
            "prompt_sha256":self.digest,"first_error":kind.unwrap_or("no_unique_joint_witness"),"expected":expected,"observed":observed,
            "gold_nouns":self.gold,"gold_governors":self.governors,"tuple":got["tuple"],"links":got["links"]})
    }
}

pub type Ids = BTreeSet<String>;

#[derive(Default)]
pub struct Pairs {
    pub exact_near: Ids,
    pub exact_far: Ids,
    pub full_near: Ids,
    pub full_far: Ids,
    pub exact_family: BTreeMap<String, usize>,
    pub exact_prompts: usize,
    pub both_governors: usize,
    pub first_near: Option<Value>,
    pub first_far: Option<Value>,
    pub first_lost_protected: Option<Value>,
}

impl Pairs {
    pub fn read(refs: &[Reference], captures: &[Value], protected: &Ids) -> Result<Self> {
        Self::read_matched(refs, captures, protected, false)
    }

    pub fn read_matched(
        refs: &[Reference],
        captures: &[Value],
        protected: &Ids,
        contrast: bool,
    ) -> Result<Self> {
        check(
            refs.len() == captures.len(),
            "prediction cardinality differs",
        )?;
        let mut grades = Self::default();
        let mut pairs = BTreeMap::<&str, Vec<(usize, bool, bool)>>::new();
        for (row, (target, raw)) in refs.iter().zip(captures).enumerate() {
            check(
                raw["row"] == row,
                format!("physical row order differs at {row}"),
            )?;
            let exact = target.exact(raw);
            let full = target.full(raw);
            grades.exact_prompts += usize::from(exact);
            grades.both_governors += usize::from(
                (0..2).all(|sign| raw["links"][sign]["governor"] == target.governors[sign]),
            );
            pairs
                .entry(&target.pair)
                .or_default()
                .push((row, exact, full));
            if full {
                continue;
            }
            let first = match target.distance.as_str() {
                "near" => &mut grades.first_near,
                "far" => &mut grades.first_far,
                _ => return Err(format!("unsupported distance at row {row}").into()),
            };
            first.get_or_insert_with(|| target.witness(row, raw));
            if protected.contains(&target.pair) {
                grades
                    .first_lost_protected
                    .get_or_insert_with(|| target.witness(row, raw));
            }
        }
        for (id, endpoints) in pairs {
            let [(i, a_exact, a_full), (j, b_exact, b_full)] = endpoints.as_slice() else {
                return Err(
                    format!("opposed pair {id} does not have two prompt identities").into(),
                );
            };
            let [a, b] = [&refs[*i], &refs[*j]];
            let matched = if contrast {
                a.admission == b.admission && a.selector != b.selector
            } else {
                matches!(
                    (a.admission.as_str(), b.admission.as_str()),
                    ("take", "defer") | ("defer", "take")
                )
            };
            check(
                matched && a.distance == b.distance && a.family == b.family,
                format!("wrong opposed identity for {id}"),
            )?;
            let (exact, full) = match a.distance.as_str() {
                "near" => (&mut grades.exact_near, &mut grades.full_near),
                "far" => (&mut grades.exact_far, &mut grades.full_far),
                _ => return Err(format!("unsupported distance for {id}").into()),
            };
            let family = grades.exact_family.entry(a.family.clone()).or_default();
            if *a_exact && *b_exact {
                exact.insert(id.into());
                *family += 1;
            }
            if *a_full && *b_full {
                full.insert(id.into());
            }
        }
        Ok(grades)
    }

    pub fn value(&self) -> Value {
        json!({"exact_near":self.exact_near,"exact_far":self.exact_far,"full_near":self.full_near,"full_far":self.full_far,
            "exact_family":self.exact_family,"exact_prompts":self.exact_prompts,"both_governors":self.both_governors,
            "first_near":self.first_near,"first_far":self.first_far,"first_lost_protected":self.first_lost_protected})
    }
}
