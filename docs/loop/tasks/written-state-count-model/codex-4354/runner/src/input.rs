//! Frozen #4352 TRAIN tapes and the #4353 replay oracle. No panel paths are accepted.

use crate::{fail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub const ARTIFACT: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
pub const STATE: &str = "368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778";
pub const SEEDS: [u32; 2] = [435300, 435301];

const TRAIN: &str = "5dd411d7dee1335784efc5868adf58a6979da18d4b743809e36b12bd44dc6829";
const LABELS: &str = "e26f239df8aba2e977bf5372dac8cdc9511e516dcf38914443cdcd9cd5431872";
const REFUSED: &str = "cf3f2d6b82aeb08818daf6a718febb98341c1730b89eca0871ae05c69d7b6398";
const RECEIPT: &str = "98098f7b57fe8239eb97500929deebd60871dcd23b66d15d3969756f095c9f0e";
const FIT: &str = "843e02e999ad8672462d024960c4ef064c10a0a02df234ba1aacd137b0de0045";
const GEOMETRY: [&str; 2] = [
    "02234649fda6e65e715a7a82faf393922af5b46a41b58cfa0c32ebe1fcbe57d4",
    "dd7983ff762cacc93ce93113fcaa442160d0138023de894a6914643e44fd344d",
];
const GROUP_DOMAIN: &[u8] = b"uor-r4/lexical-margin/group-v1\0";
const MARKER: [u16; 6] = [201, 35, 560, 652, 714, 28];
const SPACE: u16 = 223;
const CAP: usize = 2048;
const VOCABULARY: u16 = 4096;

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Deserialize)]
struct Key {
    group: u64,
    turn: u8,
}

#[derive(Clone, Copy, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Boundary {
    Immediate,
    StandaloneSpace,
}

#[derive(Deserialize)]
struct Input {
    #[serde(flatten)]
    key: Key,
    boundary: Boundary,
    full_opening: Vec<u16>,
    lexical_prefix: Vec<u16>,
}

#[derive(Deserialize)]
struct Label {
    #[serde(flatten)]
    key: Key,
    lexical_payload_offset: usize,
    target: u16,
}

#[derive(Deserialize)]
struct Refusal {
    #[serde(flatten)]
    key: Key,
    boundary: Boundary,
    reason: String,
    opening_len: usize,
    lexical_prefix_len: usize,
}

pub struct Train {
    pub prefixes: Vec<Vec<u16>>,
    pub targets: Vec<u16>,
    pub groups: Vec<u64>,
    pub ordinals: Vec<u32>,
    entries: BTreeMap<Key, Entry>,
}

struct Entry {
    admitted_ordinal: Option<u32>,
    refusal_ordinal: Option<u32>,
    boundary: Boundary,
    prefix_len: usize,
    row_sha256: [u8; 32],
}

fn read_pinned(path: &Path, digest: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if hex::encode(Sha256::digest(&bytes)) != digest {
        return Err(fail(format!("input checksum differs: {}", path.display())));
    }
    Ok(bytes)
}

fn rows(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    bytes.split_inclusive(|&byte| byte == b'\n')
}

fn boundary_width(boundary: Boundary) -> usize {
    usize::from(boundary == Boundary::StandaloneSpace)
}

pub fn load(root: &Path) -> Result<Train> {
    let train = root.join("train");
    let inputs = read_pinned(&train.join("inputs.jsonl"), TRAIN)?;
    let labels = read_pinned(&train.join("labels.jsonl"), LABELS)?;
    let refused = read_pinned(&train.join("refused.jsonl"), REFUSED)?;
    read_pinned(&train.join("receipt.json"), RECEIPT)?;
    let (input_rows, label_rows) = (rows(&inputs), rows(&labels));
    if input_rows.count() != 3891 || label_rows.count() != 3891 || rows(&refused).count() != 454 {
        return Err(fail("frozen TRAIN row counts differ"));
    }

    let mut train = Train {
        prefixes: Vec::with_capacity(3891),
        targets: Vec::with_capacity(3891),
        groups: Vec::with_capacity(3891),
        ordinals: Vec::with_capacity(3891),
        entries: BTreeMap::new(),
    };
    let mut previous = None;
    for (ordinal, (input_bytes, label_bytes)) in rows(&inputs).zip(rows(&labels)).enumerate() {
        let input: Input = serde_json::from_slice(input_bytes)?;
        let label: Label = serde_json::from_slice(label_bytes)?;
        let precursor = if input.boundary == Boundary::StandaloneSpace {
            &[SPACE][..]
        } else {
            &[]
        };
        if previous.is_some_and(|key| key >= input.key)
            || input.key != label.key
            || !(2..VOCABULARY).contains(&label.target)
            || label.target == SPACE
            || label.lexical_payload_offset < input.lexical_prefix.len()
            || input.lexical_prefix.len() > CAP
            || input.full_opening.first() != Some(&0)
            || !input.full_opening.ends_with(&MARKER)
            || input.full_opening[1..].contains(&0)
            || input.full_opening.iter().filter(|&&id| id == 1).count()
                != usize::from(input.key.turn)
            || input.lexical_prefix.iter().any(|&id| id >= VOCABULARY)
            || input
                .lexical_prefix
                .strip_prefix(input.full_opening.as_slice())
                != Some(precursor)
        {
            return Err(fail(format!(
                "invalid TRAIN opening or label join at ordinal {ordinal}"
            )));
        }
        previous = Some(input.key);
        let mut hash = Sha256::new();
        hash.update(Sha256::digest(input_bytes));
        hash.update(Sha256::digest(label_bytes));
        train.entries.insert(
            input.key,
            Entry {
                admitted_ordinal: Some(u32::try_from(ordinal)?),
                refusal_ordinal: None,
                boundary: input.boundary,
                prefix_len: input.lexical_prefix.len(),
                row_sha256: hash.finalize().into(),
            },
        );
        train.prefixes.push(input.lexical_prefix);
        train.targets.push(label.target);
        train.groups.push(input.key.group);
        train.ordinals.push(u32::try_from(ordinal)?);
    }

    let mut previous = None;
    for (ordinal, bytes) in rows(&refused).enumerate() {
        let refusal: Refusal = serde_json::from_slice(bytes)?;
        if train.entries.contains_key(&refusal.key)
            || previous.is_some_and(|key| key >= refusal.key)
            || refusal.reason != "full_opening_over_cap"
            || refusal.opening_len <= CAP
            || refusal
                .opening_len
                .checked_add(boundary_width(refusal.boundary))
                != Some(refusal.lexical_prefix_len)
        {
            return Err(fail(format!("invalid TRAIN refusal at ordinal {ordinal}")));
        }
        previous = Some(refusal.key);
        train.entries.insert(
            refusal.key,
            Entry {
                admitted_ordinal: None,
                refusal_ordinal: Some(u32::try_from(ordinal)?),
                boundary: refusal.boundary,
                prefix_len: refusal.lexical_prefix_len,
                row_sha256: Sha256::digest(bytes).into(),
            },
        );
    }

    let mut next_group = 0;
    let mut second_turns = 0;
    for key in train.entries.keys() {
        match key.turn {
            0 if key.group == next_group => next_group += 1,
            1 if key.group + 1 == next_group => second_turns += 1,
            _ => return Err(fail("frozen TRAIN group/turn order differs")),
        }
    }
    if (next_group, second_turns) != (4060, 285) {
        return Err(fail("frozen TRAIN group/turn counts differ"));
    }
    Ok(train)
}

pub fn orders(seed: u32, ordinals: &[u32]) -> Vec<Vec<u32>> {
    (0..8u32)
        .map(|epoch| {
            let mut ids = ordinals.to_vec();
            ids.sort_by_cached_key(|id| {
                let mut hash = Sha256::new();
                for value in [seed, epoch, *id] {
                    hash.update(value.to_le_bytes());
                }
                (<[u8; 32]>::from(hash.finalize()), *id)
            });
            ids
        })
        .collect()
}

fn group_digest(group: u64) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(GROUP_DOMAIN);
    hash.update(group.to_le_bytes());
    hash.finalize().into()
}

fn withheld_digest(digest: &[u8; 32]) -> bool {
    let mut first = [0; 8];
    first.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(first) % 5 == 0
}

pub fn is_withheld(group: u64) -> bool {
    withheld_digest(&group_digest(group))
}

#[derive(Default)]
struct Partition {
    groups: usize,
    admitted: usize,
    refused: usize,
    admitted_immediate: usize,
    admitted_standalone_space: usize,
    admitted_prefix_max: usize,
    admitted_prefix_tokens: usize,
    group_hashes: Sha256,
    data_hashes: Sha256,
}

impl Partition {
    fn entry(&mut self, key: &Key, entry: &Entry) {
        self.data_hashes.update(key.group.to_le_bytes());
        self.data_hashes.update([key.turn]);
        self.data_hashes.update(entry.row_sha256);
        if entry.admitted_ordinal.is_some() {
            self.admitted += 1;
            self.admitted_immediate += usize::from(entry.boundary == Boundary::Immediate);
            self.admitted_standalone_space += boundary_width(entry.boundary);
            self.admitted_prefix_max = self.admitted_prefix_max.max(entry.prefix_len);
            self.admitted_prefix_tokens += entry.prefix_len;
        } else {
            self.refused += 1;
        }
    }

    fn summary(self) -> Value {
        serde_json::json!({
            "groups": self.groups,
            "admitted": self.admitted,
            "refused": self.refused,
            "admitted_immediate": self.admitted_immediate,
            "admitted_standalone_space": self.admitted_standalone_space,
            "admitted_prefix_max": self.admitted_prefix_max,
            "admitted_prefix_tokens": self.admitted_prefix_tokens,
            "group_sha256": hex::encode(self.group_hashes.finalize()),
            "data_sha256": hex::encode(self.data_hashes.finalize()),
        })
    }
}

#[derive(Serialize)]
struct Group {
    group: u64,
    group_sha256: String,
    withheld: bool,
    admitted_ordinals: Vec<u32>,
    refusal_ordinals: Vec<u32>,
}

fn split_bytes(train: &Train) -> Result<Vec<u8>> {
    let mut fit = Partition::default();
    let mut withheld = Partition::default();
    let mut groups: Vec<Group> = Vec::with_capacity(4060);
    for (key, entry) in &train.entries {
        if key.turn == 0 {
            groups.push(Group {
                group: key.group,
                group_sha256: hex::encode(group_digest(key.group)),
                withheld: is_withheld(key.group),
                admitted_ordinals: Vec::new(),
                refusal_ordinals: Vec::new(),
            });
        }
        let group = groups
            .last_mut()
            .ok_or_else(|| fail("TRAIN split missing first turn"))?;
        let partition = if group.withheld {
            &mut withheld
        } else {
            &mut fit
        };
        if key.turn == 0 {
            partition.groups += 1;
            partition.group_hashes.update(key.group.to_le_bytes());
        }
        group.admitted_ordinals.extend(entry.admitted_ordinal);
        group.refusal_ordinals.extend(entry.refusal_ordinal);
        partition.entry(key, entry);
    }
    let mut bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": "uor-r4.lexical-margin.group-split/1",
        "assignment": "SHA-256 of ASCII uor-r4/lexical-margin/group-v1 + NUL + u64le group, first u64le mod 5 == 0 is withheld",
        "group_hash_encoding": "SHA-256 of ascending u64le group keys",
        "data_hash_encoding": "SHA-256 of u64le group + u8 turn + row hash in ascending (group, turn) order. Row hash is SHA-256 of the paired raw input and label line hashes for admitted rows, or the raw refusal line hash",
        "ordinal_encoding": "zero-based original row in the frozen admitted or refusal JSONL tape",
        "source": {
            "inputs_sha256": TRAIN,
            "labels_sha256": LABELS,
            "refusals_sha256": REFUSED,
            "receipt_sha256": RECEIPT,
            "cap": CAP,
            "tokenizer_sha256": "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
            "tokenizer_protocol": "blake3:4c9778ead027688600d2211e50cbfae98ac15f932a0827c01ec2389c521fef27",
        },
        "fit": fit.summary(),
        "withheld": withheld.summary(),
        "groups": groups,
    }))?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn seal_split(train: &Train, path: &Path) -> Result<String> {
    let bytes = split_bytes(train)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o400)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

pub fn verify_split(train: &Train, path: &Path) -> Result<()> {
    if fs::read(path)? != split_bytes(train)? {
        return Err(fail(
            "sealed internal split manifest differs from frozen TRAIN assignment",
        ));
    }
    Ok(())
}

fn historical_fit(root: &Path) -> Result<Value> {
    let bytes = read_pinned(&root.join("prior/fit.json"), FIT)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn historical_weights(root: &Path, index: usize) -> Result<Vec<u8>> {
    let seed = SEEDS[index];
    read_pinned(
        &root.join(format!("prior/geometry-{seed}.lex")),
        GEOMETRY[index],
    )
}

pub fn verify_historical_inputs(root: &Path) -> Result<()> {
    historical_fit(root)?;
    for index in 0..SEEDS.len() {
        historical_weights(root, index)?;
    }
    Ok(())
}

pub fn verify_history(root: &Path, seed: u32, weights: &[i16; 256], epochs: &Value) -> Result<()> {
    let index = SEEDS
        .iter()
        .position(|&value| value == seed)
        .ok_or_else(|| fail("unknown historical geometry seed"))?;
    let old = historical_weights(root, index)?;
    // #4353 uses the shared 584-byte envelope: state digest, UORLEX01, i16le words, digest.
    let mut encoded = Vec::with_capacity(584);
    encoded.extend(hex::decode(STATE)?);
    encoded.extend(b"UORLEX01");
    encoded.extend(weights.iter().flat_map(|word| word.to_le_bytes()));
    encoded.extend(Sha256::digest(&encoded));
    if old != encoded {
        return Err(fail(format!(
            "geometry-{seed} final native weight words differ from #4353"
        )));
    }
    let fit = historical_fit(root)?;
    let name = format!("geometry-{seed}");
    let old_epochs = fit["arms"]
        .as_array()
        .and_then(|arms| {
            arms.iter()
                .find(|arm| arm["arm"].as_str() == Some(name.as_str()))
        })
        .map(|arm| &arm["epochs"])
        .ok_or_else(|| fail(format!("missing #4353 epoch counters for {name}")))?;
    if epochs != old_epochs {
        let difference = match (epochs.as_array(), old_epochs.as_array()) {
            (Some(actual), Some(expected)) => actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .map_or(actual.len().min(expected.len()), |index| index),
            _ => 0,
        };
        return Err(fail(format!(
            "{name} counters differ from #4353 at epoch {}",
            difference + 1
        )));
    }
    Ok(())
}
