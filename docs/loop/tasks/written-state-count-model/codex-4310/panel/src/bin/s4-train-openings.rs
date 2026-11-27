#![forbid(unsafe_code)]

//! Descriptive corpus census and readback of existing GPU output. Runs no model.
//! Run on the corpus host with the pinned files in the order printed by usage.
//! C420_TRAIN_ROOT contains train.tokens.u16 and train.mask.u8 from claude-420.
//! The probe IDs and individual corpus records stay private.

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::Path;
use uor_r4_tokenizer::{
    dialogue::{DialogueProtocol, Message},
    ByteBpeTokenizer,
};

type AuditResult<T> = Result<T, Box<dyn Error>>;
const N: usize = 16;
const LENGTHS: [usize; 5] = [1, 2, 4, 8, N];
const VOCAB: u16 = 4096;
const R_SHA: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
const D87_TOK_SHA: &str = "d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89";
const P4_TOK_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const FIT_SHA: &str = "e5b1c14eb96686b7fdee3f78bd70634f79b114e2912d5c53aca635c791edcc52";
const HELD_SHA: &str = "d8630bf60f06cc65ff4b32732cf5b1b5aba2a2837008408a207e32b950d24ef0";
const GENERATION_SHA: &str = "31b3f86fc602f1ecc82d89b0632907792d9fe52b1d9a2941b16896012aea81a8";
const TRAIN_SHA: &str = "c6cba29ead4c745fb47b3f87007d982e72defc9f04102a268709ba0d34d51460";
const MASK_SHA: &str = "2bdbb283e7dcb12709a0cc3c4ce86f3c5d7032aab0b3171f85ebe3b1bdb1a2a8";

#[derive(Deserialize)]
struct PanelRow {
    id: String,
    domain: String,
    prefix: Vec<u16>,
    gold: Vec<u16>,
    free_run: bool,
}

#[derive(Deserialize)]
struct Generation {
    id: String,
    domain: String,
    arm: String,
    gold: Vec<u16>,
    tokens: Vec<u16>,
}

fn require(condition: bool, message: &str) -> AuditResult<()> {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

fn pinned(path: &Path, digest: &str) -> AuditResult<Vec<u8>> {
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != digest {
        return Err(format!("input digest mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}

fn tokenizer(path: &Path, digest: &str) -> AuditResult<ByteBpeTokenizer> {
    let bytes = pinned(path, digest)?;
    ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes).ok_or_else(|| "invalid tokenizer".into())
}

fn jsonl<T: serde::de::DeserializeOwned>(path: &Path, digest: &str) -> AuditResult<Vec<T>> {
    let bytes = pinned(path, digest)?;
    std::str::from_utf8(&bytes)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn ids(ids: Vec<u32>) -> AuditResult<Vec<u16>> {
    ids.into_iter().map(|id| Ok(u16::try_from(id)?)).collect()
}

fn uort(raw: &[u8]) -> AuditResult<Vec<u16>> {
    let (header, payload) = raw.split_at_checked(64).ok_or("short UORT header")?;
    require(
        &header[..8] == b"UORT\x01\x00\x00\x00"
            && u32::from_le_bytes(header[16..20].try_into()?) == u32::from(VOCAB)
            && payload.len() % 2 == 0
            && u64::from_le_bytes(header[8..16].try_into()?) == (payload.len() / 2) as u64,
        "invalid UORT header or payload",
    )?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

#[derive(Default)]
struct Openings {
    first_ids: BTreeMap<u16, usize>,
    eligible_prefix: [usize; N],
    matching_prefix: [usize; N],
    rows: usize,
}

impl Openings {
    fn add(&mut self, tokens: &[u16], probe: &[u16]) {
        self.rows += 1;
        if let Some(&first) = tokens.first() {
            *self.first_ids.entry(first).or_default() += 1;
        }
        for n in 1..=probe.len() {
            let offset = n - 1;
            if tokens.len() >= n {
                self.eligible_prefix[offset] += 1;
                if tokens[..n] == probe[..n] {
                    self.matching_prefix[offset] += 1;
                }
            }
        }
    }

    fn report(&self) -> Value {
        let mut ranked: Vec<_> = self.first_ids.iter().collect();
        ranked.sort_unstable_by_key(|(&id, &count)| (std::cmp::Reverse(count), id));
        json!({
            "rows": self.rows,
            "first_id_distinct": self.first_ids.len(),
            "top_first_ids": ranked.into_iter().take(5).map(|(id, count)| {
                json!({"id": id, "count": count})
            }).collect::<Vec<_>>(),
            "eligible_by_prefix_length": LENGTHS.map(|n| self.eligible_prefix[n - 1]),
            "matches_by_prefix_length": LENGTHS.map(|n| self.matching_prefix[n - 1]),
        })
    }
}

fn scan(
    tokens: &[u16],
    marker: &[u16],
    user_marker: &[u16],
    probe: &[u16],
    mask: &[u8],
) -> AuditResult<Value> {
    let mut record_counts = BTreeMap::new();
    let mut context = BTreeMap::<usize, Openings>::new();
    let mut anywhere = [0usize; N];
    let mut anywhere_records = [0usize; N];
    let mut records_with_probe_opening = 0;
    let mut positions = 0;
    let mut all = Openings::default();
    let mut unmasked_marker_candidates = 0;
    for record in tokens.split_inclusive(|&id| id == 1) {
        require(
            record.first() == Some(&0)
                && record.last() == Some(&1)
                && record[1..].iter().all(|&id| id > 0 && id < VOCAB)
                && record[1..].starts_with(user_marker),
            "invalid D87 record framing",
        )?;
        let record_mask = &mask[positions..positions + record.len()];
        require(record_mask[0] == 0, "BOS marked as response")?;
        let mut seen = [false; N];
        let mut record_turns = 0;
        let mut record_match = false;
        for (index, &id) in record.iter().enumerate() {
            if id == probe[0] {
                for (offset, &want) in probe.iter().enumerate() {
                    if record.get(index + offset) != Some(&want) {
                        break;
                    }
                    anywhere[offset] += 1;
                    seen[offset] = true;
                }
            }
            let marker_start = index
                .checked_sub(marker.len())
                .filter(|&start| record.get(start..index) == Some(marker));
            let response_start =
                index != 0 && record_mask[index - 1] == 0 && record_mask[index] == 1;
            if !response_start {
                unmasked_marker_candidates += usize::from(marker_start.is_some());
                continue;
            }
            require(
                id != 1
                    && marker_start
                        .is_some_and(|start| record_mask[start..index].iter().all(|&m| m == 0)),
                "masked assistant response has no matching role marker",
            )?;
            record_turns += 1;
            let response_end = record_mask[index..]
                .iter()
                .position(|&byte| byte == 0)
                .map_or(record.len(), |end| index + end)
                .min(record.len() - 1);
            let response = &record[index..response_end];
            all.add(response, probe);
            context
                .entry(record_turns)
                .or_default()
                .add(response, probe);
            record_match |= response.first() == Some(&probe[0]);
        }
        require((1..=2).contains(&record_turns), "wrong D87 turn count")?;
        records_with_probe_opening += usize::from(record_match);
        *record_counts.entry(record_turns).or_insert(0usize) += 1;
        for (total, found) in anywhere_records.iter_mut().zip(seen) {
            *total += usize::from(found);
        }
        positions += record.len();
    }
    let record_count: usize = record_counts.values().sum();
    require(record_count == 4060, "wrong R record count")?;
    Ok(json!({
        "records": record_count,
        "tokens_including_bos_and_eos": positions,
        "records_by_assistant_markers": record_counts,
        "records_with_probe_first_id_after_assistant_marker": records_with_probe_opening,
        "assistant_openings": all.report(),
        "openings_by_turn_number": context.into_iter().map(|(context, counts)| {
            (context, counts.report())
        }).collect::<BTreeMap<_, _>>(),
        "excluded_marker_matches_without_role_transition": unmasked_marker_candidates,
        "any_position_prefix_occurrences_by_length": LENGTHS.map(|n| anywhere[n - 1]),
        "any_position_records_with_prefix_by_length": LENGTHS.map(|n| anywhere_records[n - 1]),
    }))
}

fn response_mask(root: &Path, r: &[u16]) -> AuditResult<Vec<u8>> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let mut train = BufReader::new(File::open(root.join("train.tokens.u16"))?);
    let mut mask = BufReader::new(File::open(root.join("train.mask.u8"))?);
    let mut header = [0u8; 64];
    train.read_exact(&mut header)?;
    let count = u64::from_le_bytes(header[8..16].try_into()?);
    require(
        &header[..8] == b"UORT\x01\x00\x00\x00" && count == 2_091_560_001,
        "wrong TRAIN UORT header",
    )?;
    let mut train_sha = Sha256::new();
    train_sha.update(header);
    let mut mask_sha = Sha256::new();
    const BLOCK: usize = 1 << 18;
    let mut tokens = vec![0u8; 2 * BLOCK];
    let mut masks = vec![0u8; BLOCK];
    let mut result = Vec::with_capacity(r.len());
    let mut record = 0;
    for start in (0..count as usize).step_by(BLOCK) {
        let len = BLOCK.min(count as usize - start);
        train.read_exact(&mut tokens[..2 * len])?;
        mask.read_exact(&mut masks[..len])?;
        train_sha.update(&tokens[..2 * len]);
        mask_sha.update(&masks[..len]);
        for (offset, pair) in tokens[..2 * len].chunks_exact(2).enumerate() {
            let id = u16::from_le_bytes([pair[0], pair[1]]);
            if record % 128 == 0 {
                require(
                    r.get(result.len()) == Some(&id) && masks[offset] <= 1,
                    "reconstructed R and pinned R differ",
                )?;
                result.push(masks[offset]);
            }
            record += usize::from(id == 1);
        }
    }
    require(
        train.read(&mut header)? == 0
            && mask.read(&mut header)? == 0
            && record == 519_580
            && result.len() == r.len()
            && format!("{:x}", train_sha.finalize()) == TRAIN_SHA
            && format!("{:x}", mask_sha.finalize()) == MASK_SHA,
        "TRAIN source identity or reconstructed R size mismatch",
    )?;
    Ok(result)
}

fn main() -> AuditResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    require(
        args.len() == 7,
        "usage: s4-train-openings R D87_TOKENIZER P4_TOKENIZER FIT HELD GPU_GENERATION C420_TRAIN_ROOT",
    )?;
    let [r_path, d87, p4, fit, held, generation, train_root] = args.as_slice() else {
        return Err("missing paths".into());
    };
    let r = uort(&pinned(Path::new(r_path), R_SHA)?)?;
    let d87 = tokenizer(Path::new(d87), D87_TOK_SHA)?;
    let p4 = tokenizer(Path::new(p4), P4_TOK_SHA)?;
    let protocol = DialogueProtocol::literal_roles_v2(&p4)?;
    let encoder = protocol.bind(&p4)?;
    let example = [Message {
        role: "user",
        content: "Example?",
    }];
    let history = encoder.encode_open_history(&example).tokens;
    let prefix = encoder.encode_assistant_prefix(&example).tokens;
    let marker = ids(prefix
        .strip_prefix(history.as_slice())
        .ok_or("assistant prefix rewrote its example history")?
        .to_vec())?;
    let d87_marker = d87.encode("\nAssistant:");
    require(
        d87.decode_bytes(&d87_marker) == b"\nAssistant:"
            && p4.decode_bytes(&marker.iter().copied().map(u32::from).collect::<Vec<_>>())
                == b"\nAssistant:",
        "assistant marker surface mismatch",
    )?;
    let d87_marker = ids(d87_marker)?;
    let user_marker = ids(d87.encode("\nUser:"))?;
    let differing_token_ids = (0..u32::from(VOCAB))
        .filter(|&id| d87.decode_bytes(&[id]) != p4.decode_bytes(&[id]))
        .count();
    let fit: Vec<PanelRow> = jsonl(Path::new(fit), FIT_SHA)?;
    let held: Vec<PanelRow> = jsonl(Path::new(held), HELD_SHA)?;
    let generation: Vec<Generation> = jsonl(Path::new(generation), GENERATION_SHA)?;
    let chat: BTreeMap<_, _> = held
        .iter()
        .filter(|row| row.domain == "conversation" && row.free_run)
        .map(|row| (row.id.as_str(), row))
        .collect();
    require(chat.len() == 24, "wrong free chat cohort size")?;
    for row in &fit {
        if row.domain == "conversation" {
            require(
                row.prefix.ends_with(&marker),
                "fit assistant marker mismatch",
            )?;
        }
    }
    for row in chat.values() {
        require(
            row.prefix.ends_with(&marker),
            "held assistant marker mismatch",
        )?;
    }
    let mut occurrences = BTreeMap::new();
    let mut ids_by_arm = BTreeMap::<_, BTreeSet<_>>::new();
    for entry in generation
        .iter()
        .filter(|entry| entry.domain == "conversation")
    {
        require(
            chat.get(entry.id.as_str())
                .is_some_and(|row| row.gold == entry.gold)
                && ids_by_arm.entry(&entry.arm).or_default().insert(&entry.id),
            "generation does not match registered free chat",
        )?;
        if entry.arm == "learned" {
            let prefix = entry
                .tokens
                .get(..N)
                .ok_or("learned output shorter than 16")?;
            *occurrences.entry(prefix).or_insert(0usize) += 1;
        }
    }
    require(
        ids_by_arm.len() == 6 && ids_by_arm.values().all(|ids| ids.len() == chat.len()),
        "generation has incomplete conversation arms",
    )?;
    let (probe, _) = occurrences
        .into_iter()
        .max_by_key(|&(prefix, count)| (count, std::cmp::Reverse(prefix)))
        .ok_or("no learned opening")?;
    require(
        probe[0] == 799
            && probe
                .iter()
                .all(|&id| d87.decode_bytes(&[u32::from(id)]) == p4.decode_bytes(&[u32::from(id)])),
        "probe opener ID or probe token byte identities differ",
    )?;
    let mask = response_mask(Path::new(train_root), &r)?;
    let r_report = scan(&r, &d87_marker, &user_marker, probe, &mask)?;
    let mut cohorts = BTreeMap::<&str, Openings>::new();
    for row in fit.iter().filter(|row| row.domain == "conversation") {
        cohorts
            .entry("human_fit_gold")
            .or_default()
            .add(&row.gold, probe);
    }
    for row in chat.values() {
        cohorts
            .entry("human_free_gold")
            .or_default()
            .add(&row.gold, probe);
    }
    for entry in generation
        .iter()
        .filter(|entry| entry.domain == "conversation")
    {
        cohorts
            .entry(&entry.arm)
            .or_default()
            .add(&entry.tokens, probe);
    }
    let value = json!({
        "schema": "uor-r4.s4-train-openings/1",
        "classification": "post-hoc descriptive corpus census and readback of existing GPU generation",
        "binding": {
            "r_sha256": R_SHA,
            "d87_tokenizer_sha256": D87_TOK_SHA,
            "p4_tokenizer_sha256": P4_TOK_SHA,
            "natural_fit_sha256": FIT_SHA,
            "natural_held_sha256": HELD_SHA,
            "gpu_generation_sha256": GENERATION_SHA,
            "full_train_sha256": TRAIN_SHA,
            "full_train_response_mask_sha256": MASK_SHA,
        },
        "r_provenance": "D87 donor TRAIN, every 128th original record beginning from index zero. The source packs reasoning followed by the final answer when reasoning exists. R segment and class labels encode the split rather than assistant roles.",
        "interpretation_limit": "Exposure in R does not assign causality to an individual example or fitted component. The response mask authenticates onset but does not separate the donor reasoning and final answer.",
        "differing_single_id_token_bytes_in_vocabulary": differing_token_ids,
        "p4_newline_and_assistant_marker_ids": marker,
        "d87_newline_and_assistant_marker_ids": d87_marker,
        "corpus_markers_authenticated_by_original_role_mask": true,
        "probe_first_id": probe[0],
        "prefix_lengths": LENGTHS,
        "probe_definition": "Most frequent exact first 16 IDs of the already saved learned free conversations. Counts at each length use the nested prefix of that same probe.",
        "r": r_report,
        "natural_conversation_openings": cohorts.into_iter().map(|(name, counts)| {
            (name, counts.report())
        }).collect::<BTreeMap<_, _>>(),
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
