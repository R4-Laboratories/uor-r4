use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Error, ErrorKind},
    path::Path,
};

pub const BANKS: usize = 16;
pub const SLOTS: usize = 256;
pub const WORDS: usize = BANKS * SLOTS;
pub const BOUND: i16 = 4096;
pub const MARGIN: i32 = 1;
pub type Keys = [u16; BANKS];
pub type Words = [i16; WORDS];
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn invalid(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Error::new(ErrorKind::InvalidData, message.into()).into()
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Geometry,
    Count,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Span {
    pub start: u16,
    pub end: u16,
}

#[derive(Deserialize)]
struct Candidate {
    #[serde(flatten)]
    span: Span,
    keys: Keys,
}

#[derive(Deserialize)]
struct Features {
    case_id: String,
    arm: Arm,
    #[serde(flatten)]
    bounds: Bounds,
    candidates: Vec<Candidate>,
    no_read_keys: Keys,
    #[serde(default)]
    scores: Option<ColdScores>,
}

#[derive(Deserialize)]
pub struct ColdScores {
    pub candidates: Vec<i32>,
    pub selection: ColdSelection,
}

#[derive(Deserialize)]
pub struct ColdSelection {
    pub span: Option<Span>,
    pub best_candidate_score: Option<i32>,
    pub no_read_score: i32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Bounds {
    pub user: [u16; 2],
    pub fact: [u16; 2],
    pub question: [u16; 2],
}

impl Bounds {
    fn valid(self) -> bool {
        self.user[0] <= self.fact[0]
            && self.fact[0] <= self.fact[1]
            && self.fact[1] <= self.user[1]
            && self.user[0] <= self.question[0]
            && self.question[0] <= self.question[1]
            && self.question[1] <= self.user[1]
            && (self.fact[1] <= self.question[0]
                || (self.fact[1] == self.question[1]
                    && self.fact[0] <= self.question[0]
                    && self.question[1] - self.question[0] <= 32))
    }
}

// A missing field is malformed. An explicit null means NoRead.
#[derive(Deserialize)]
struct Gold {
    case_id: String,
    #[serde(deserialize_with = "Option::deserialize")]
    start: Option<u16>,
    #[serde(deserialize_with = "Option::deserialize")]
    end: Option<u16>,
    group: String,
    #[serde(default)]
    swapped_case_id: Option<String>,
}

pub struct Row {
    pub case_id: String,
    pub gold: Option<Span>,
    pub group: String,
    pub swapped_case_id: Option<String>,
    pub bounds: Bounds,
    pub choices: Vec<Option<Span>>,
    pub keys: BTreeMap<Arm, Vec<Keys>>,
    pub cold: BTreeMap<Arm, ColdScores>,
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let rows = BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(&line?)
                .map_err(|e| invalid(format!("{}:{}: {e}", path.display(), i + 1)))
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Err(invalid(format!("empty JSONL: {}", path.display())));
    }
    Ok(rows)
}

fn validate_keys(case_id: &str, keys: &Keys) -> Result<()> {
    if keys
        .iter()
        .enumerate()
        .any(|(bank, &key)| usize::from(key) / SLOTS != bank)
    {
        return Err(invalid(format!(
            "{case_id}: expected one key from each bank"
        )));
    }
    Ok(())
}

pub fn read_rows(features: &Path, gold: Option<&Path>) -> Result<Vec<Row>> {
    let mut values = BTreeMap::new();
    let mut order = Vec::new();
    for features in read_jsonl::<Features>(features)? {
        if features.arm == Arm::Geometry {
            order.push(features.case_id.clone());
        }
        let key = (features.case_id.clone(), features.arm);
        if values.insert(key, features).is_some() {
            return Err(invalid("duplicate case/arm in feature tape"));
        }
    }
    let labels = if let Some(gold) = gold {
        let labels = read_jsonl::<Gold>(gold)?;
        if !order
            .iter()
            .map(String::as_str)
            .eq(labels.iter().map(|row| row.case_id.as_str()))
        {
            return Err(invalid("feature and gold case order differ"));
        }
        labels
    } else {
        order
            .into_iter()
            .map(|case_id| Gold {
                case_id,
                start: None,
                end: None,
                group: String::new(),
                swapped_case_id: None,
            })
            .collect()
    };
    let mut rows = Vec::new();
    for label in labels {
        let gold = match (label.start, label.end) {
            (Some(start), Some(end)) if start < end && end - start <= 8 => {
                Some(Span { start, end })
            }
            (None, None) => None,
            _ => return Err(invalid(format!("{}: invalid gold span", label.case_id))),
        };
        let bounds = values
            .get(&(label.case_id.clone(), Arm::Geometry))
            .ok_or_else(|| invalid(format!("{}: missing geometry features", label.case_id)))?
            .bounds;
        if !bounds.valid() {
            return Err(invalid(format!(
                "{}: invalid original user bounds",
                label.case_id
            )));
        }
        let mut row = Row {
            case_id: label.case_id,
            gold,
            group: label.group,
            swapped_case_id: label.swapped_case_id,
            bounds,
            choices: Vec::new(),
            keys: BTreeMap::new(),
            cold: BTreeMap::new(),
        };
        for arm in [Arm::Geometry, Arm::Count] {
            let Some(features) = values.remove(&(row.case_id.clone(), arm)) else {
                return Err(invalid(format!(
                    "{}: missing features for {arm:?}",
                    row.case_id
                )));
            };
            if let Some(scores) = features.scores {
                row.cold.insert(arm, scores);
            }
            if features.bounds != bounds {
                return Err(invalid(format!(
                    "{}: unequal original user bounds across arms",
                    row.case_id
                )));
            }
            validate_keys(&row.case_id, &features.no_read_keys)?;
            let mut choices = vec![None];
            let mut keys = vec![features.no_read_keys];
            for candidate in features.candidates {
                let span = candidate.span;
                if span.start >= span.end
                    || span.end - span.start > 8
                    || span.start < bounds.fact[0]
                    || span.end > bounds.fact[1]
                    || choices.last().is_some_and(|prior| *prior >= Some(span))
                {
                    return Err(invalid(format!(
                        "{}: unordered, repeated or invalid candidate",
                        row.case_id
                    )));
                }
                validate_keys(&row.case_id, &candidate.keys)?;
                choices.push(Some(span));
                keys.push(candidate.keys);
            }
            if row.choices.is_empty() {
                row.choices = choices;
            } else if row.choices != choices {
                return Err(invalid(format!(
                    "{}: unequal candidates across arms",
                    row.case_id
                )));
            }
            row.keys.insert(arm, keys);
        }
        rows.push(row);
    }
    if !values.is_empty() {
        return Err(invalid(
            "feature tape includes cases absent from the gold tape",
        ));
    }
    Ok(rows)
}

pub fn gold_index(row: &Row) -> Option<usize> {
    row.choices.binary_search(&row.gold).ok()
}

pub fn score(words: &Words, keys: &Keys) -> i32 {
    keys.iter()
        .map(|&key| i32::from(words[usize::from(key)]))
        .sum()
}

pub fn winner(words: &Words, choices: &[Keys]) -> (usize, i32) {
    let mut index = 0;
    let mut best = score(words, &choices[0]);
    for (i, keys) in choices.iter().enumerate().skip(1) {
        let value = score(words, keys);
        if value > best {
            (index, best) = (i, value);
        }
    }
    (index, best)
}
