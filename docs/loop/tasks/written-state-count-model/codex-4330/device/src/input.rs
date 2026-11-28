use crate::{require, Result, CONTEXT, NONE, STATE_WORDS, STRIDE, VOCAB, WEIGHT_LIMIT};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

const ROOTS_SHA: &str = "875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a";
const CLASSES_SHA: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";
const INITIAL_SHA: &str = "c4c2cb4139f1af761da5c5d6dcc656dab81ad9a813646c3efef7be672ca5a08a";
pub const USER: [u32; 4] = [0, 55, 2728, 28];
pub const ASSISTANT: [u32; 6] = [201, 35, 560, 652, 714, 28];

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pinned(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        sha(&bytes) == expected,
        &format!("frozen input SHA-256 differs for {}", path.display()),
    )?;
    Ok(bytes)
}

pub fn weights_bytes(weights: &[i16]) -> Vec<u8> {
    weights.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn check_weights(weights: &[i16]) -> Result<()> {
    require(
        weights.len() == STATE_WORDS
            && weights
                .iter()
                .all(|&word| (-WEIGHT_LIMIT..=WEIGHT_LIMIT).contains(&word)),
        "expected 513 signed i16 weight coordinates in [-4096,4096]",
    )
}

pub fn weights(path: &Path, initial: bool) -> Result<Vec<i16>> {
    let bytes = if initial {
        pinned(path, INITIAL_SHA)?
    } else {
        fs::read(path)?
    };
    require(
        bytes.len() == 2 * STATE_WORDS,
        "wrong i16 state byte length",
    )?;
    let result: Vec<_> = bytes
        .chunks_exact(2)
        .map(|x| i16::from_le_bytes([x[0], x[1]]))
        .collect();
    check_weights(&result)?;
    Ok(result)
}

pub struct Geometry {
    pub roots: Vec<u32>,
    pub classes: Vec<u8>,
}

impl Geometry {
    pub fn load(roots: &Path, classes: &Path) -> Result<Self> {
        let root_bytes = pinned(roots, ROOTS_SHA)?;
        let classes = pinned(classes, CLASSES_SHA)?;
        require(
            root_bytes.len() == VOCAB * 4 && classes.len() == VOCAB,
            "bad geometry length",
        )?;
        let roots = root_bytes
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect::<Vec<_>>();
        require(
            roots.iter().all(|&root| root <= u16::MAX as u32)
                && classes.iter().all(|&class| class <= 9),
            "class or packed geometric root is out of range",
        )?;
        Ok(Self { roots, classes })
    }

    pub fn identity() -> serde_json::Value {
        serde_json::json!({"roots_sha256": ROOTS_SHA, "classes_sha256": CLASSES_SHA})
    }
}

pub fn eligible(class: u8) -> bool {
    class != 0 && class != 9
}

#[derive(Clone, Deserialize)]
pub struct Row {
    pub case_id: String,
    pub prefix: Vec<u32>,
    pub user_bounds: [u16; 2],
    pub target: Option<u16>,
}

impl Row {
    pub fn validate(&self, classes: &[u8]) -> Result<()> {
        let [start, end] = self.user_bounds.map(usize::from);
        require(
            !self.case_id.trim().is_empty()
                && self.prefix.len() > USER.len() + ASSISTANT.len()
                && self.prefix.len() <= CONTEXT
                && self.prefix.starts_with(&USER)
                && self.prefix.ends_with(&ASSISTANT)
                && start == USER.len()
                && end == self.prefix.len() - ASSISTANT.len()
                && self.prefix.iter().all(|&id| (id as usize) < VOCAB)
                && self.prefix[start..end].iter().all(|&id| id > 2),
            &format!(
                "invalid protocol-2 prefix or current user range for {}",
                self.case_id
            ),
        )?;
        require(
            self.target.is_none_or(|p| {
                let p = usize::from(p);
                (start..end).contains(&p) && eligible(classes[self.prefix[p] as usize])
            }),
            &format!(
                "gold position is outside eligible current-user tokens for {}",
                self.case_id
            ),
        )
    }

    pub fn target_word(&self) -> u32 {
        self.target.map_or(NONE, u32::from)
    }
}

pub fn rows(path: &Path, classes: &[u8], labeled: bool) -> Result<(Vec<Row>, String)> {
    let bytes = fs::read(path)?;
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for line in std::str::from_utf8(&bytes)?.lines() {
        let value: serde_json::Value = serde_json::from_str(line)?;
        require(
            !labeled || value.get("target").is_some(),
            "fit row needs an explicit target or null",
        )?;
        let mut row: Row = serde_json::from_value(value)?;
        if !labeled {
            row.target = None;
        }
        row.validate(classes)?;
        require(seen.insert(row.case_id.clone()), "duplicate case ID")?;
        rows.push(row);
    }
    require(!rows.is_empty(), "empty row tape")?;
    rows.sort_unstable_by(|left, right| left.case_id.cmp(&right.case_id));
    Ok((rows, sha(&bytes)))
}

pub fn pack<'a>(rows: impl IntoIterator<Item = &'a Row>, labeled: bool) -> Vec<u32> {
    let mut packed = Vec::new();
    for row in rows {
        let start = packed.len();
        packed.resize(start + STRIDE, 0);
        let words = &mut packed[start..];
        words[0] = row.prefix.len() as u32;
        words[1] = if labeled { row.target_word() } else { NONE };
        words[2] = u32::from(row.user_bounds[0]);
        words[3] = u32::from(row.user_bounds[1]);
        words[8..8 + row.prefix.len()].copy_from_slice(&row.prefix);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_eval_has_no_labels_and_all_current_user_positions_are_preserved() {
        let mut row = Row {
            case_id: "negative-fixture".to_owned(),
            prefix: [USER.as_slice(), &[1042, 201, 55, 2728, 28], &ASSISTANT].concat(),
            user_bounds: [4, 9],
            target: Some(4),
        };
        let mut classes = vec![0; VOCAB];
        classes[1042] = 4;
        row.validate(&classes).unwrap();
        assert_eq!(pack([&row], false)[0..4], [15, NONE, 4, 9]);
        assert_eq!(pack([&row], true)[0..4], [15, 4, 4, 9]);
        row.target = Some(5);
        assert!(row.validate(&classes).is_err());
        row.user_bounds = [4, 8];
        assert!(row.validate(&classes).is_err());
    }
}
