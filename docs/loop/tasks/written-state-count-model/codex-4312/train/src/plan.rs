use crate::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};
use uor_r4_discrete::{
    corpus::{sha256, BosIdentity, CorpusSource, FilePin, PreparedCorpus, Split},
    training::{
        CoordinateBlock, CounterfactualPath, DriverConfig, PreparedDriver, SchedulePolicy,
        DRIVER_SCHEMA,
    },
    ObjectivePolicy, ReadInitialization, ReadSpec, Spec,
};

pub const ARMS: [&str; 2] = ["reasoning-first", "answer-first"];
pub const SEED: u32 = 4312;
pub const BATCH: u32 = 256;
pub const K: u32 = 16;
pub const VOCAB: u32 = 4096;
pub const CONTEXT: u32 = 192;
const EXPECTED_GROUPS: usize = 4060;
const MAX_PER_GROUP: u64 = 64;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub name: String,
    pub full: CorpusSource,
    pub geometry: DriverConfig,
    pub lineage_sha256: String,
    pub targets_map: FilePin,
    pub geometry_order: FilePin,
    pub segments: FilePin,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub donor_manifest: FilePin,
    pub groups: FilePin,
    pub group_count: usize,
    pub full_targets: u64,
    pub geometry_targets: u64,
    pub geometry_batches: u64,
    pub arms: [Arm; 2],
}

fn file_pin(path: PathBuf) -> Result<FilePin> {
    Ok(FilePin {
        sha256: sha256(&fs::read(&path)?),
        path,
    })
}

pub fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(path, &bytes)
}

fn source(split: Split, tokens: FilePin, mask_path: PathBuf) -> Result<CorpusSource> {
    let mask = fs::read(&mask_path)?;
    if mask.iter().any(|&value| value > 1) {
        return Err(format!("mask {} is not binary", mask_path.display()).into());
    }
    let expected_eligible_targets = mask.iter().map(|&value| u64::from(value)).sum();
    Ok(CorpusSource {
        split,
        tokens,
        mask: FilePin {
            path: mask_path,
            sha256: sha256(&mask),
        },
        expected_payload_tokens: u64::try_from(mask.len())?,
        expected_eligible_targets,
    })
}

pub fn bos() -> BosIdentity {
    BosIdentity {
        token: String::from("<|bos|>"),
        id: 0,
    }
}

pub fn driver_config(train: CorpusSource, dev: CorpusSource, tokenizer: FilePin) -> DriverConfig {
    DriverConfig {
        schema: DRIVER_SCHEMA.into(),
        spec: Spec {
            vocabulary: VOCAB,
            blocks: 4,
            context: CONTEXT,
            kept_cells: 8,
        },
        read_spec: ReadSpec::Flock { kept: 8 },
        objective_policy: ObjectivePolicy::NearestAddressMarginV1,
        read_initialization: ReadInitialization::ActiveLeafV1,
        normalization_mode: Default::default(),
        position: None,
        address_metric: Default::default(),
        relation_learning_policy: Default::default(),
        counterfactual_path: CounterfactualPath::RowsV1,
        codebook_initialization: Default::default(),
        coordinate_block: CoordinateBlock(K),
        seed: SEED,
        batch_rows: BATCH,
        train,
        dev,
        tokenizer,
        bos: bos(),
        schedule: SchedulePolicy::InterleavedReadEvery8V1,
    }
}

pub fn uort(ids: &[u16]) -> Vec<u8> {
    let mut data = vec![0; 64];
    data[..4].copy_from_slice(b"UORT");
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..16].copy_from_slice(&(ids.len() as u64).to_le_bytes());
    data[16..20].copy_from_slice(&VOCAB.to_le_bytes());
    data.extend(ids.iter().flat_map(|token| token.to_le_bytes()));
    data
}

fn read_arm(donor: &Path, arm: &str, dev: &CorpusSource, tokenizer: &FilePin) -> Result<Arm> {
    let named = |suffix| donor.join(format!("{arm}.{suffix}"));
    let token_pin = file_pin(named("tokens.u16"))?;
    let full = source(Split::Train, token_pin.clone(), named("mask.u8"))?;
    let geometry_source = source(Split::Train, token_pin, named("geometry-mask.u8"))?;
    let geometry = driver_config(geometry_source, dev.clone(), tokenizer.clone());
    let driver = PreparedDriver::load(geometry.clone())?;
    Ok(Arm {
        name: arm.to_owned(),
        full,
        geometry,
        lineage_sha256: driver.lineage_sha256()?,
        targets_map: file_pin(named("targets.u32"))?,
        geometry_order: file_pin(named("geometry-targets.u32"))?,
        segments: file_pin(named("segment.u8"))?,
    })
}

fn check_events(arms: &[Arm; 2]) -> Result<()> {
    let [a, b] = arms;
    if a.full.tokens.sha256 == b.full.tokens.sha256
        || a.full.expected_payload_tokens != b.full.expected_payload_tokens
        || a.full.expected_eligible_targets != b.full.expected_eligible_targets
        || a.geometry.train.expected_eligible_targets != b.geometry.train.expected_eligible_targets
        || a.geometry.train.expected_eligible_targets == 0
        || a.geometry.train.expected_payload_tokens != a.full.expected_payload_tokens
        || b.geometry.train.expected_payload_tokens != b.full.expected_payload_tokens
    {
        return Err("arm payload, eligible-target, or geometry inventories differ".into());
    }
    let open = |arm: &Arm| -> Result<_> {
        let corpus = PreparedCorpus::load(&arm.full, VOCAB, &bos())?;
        let mask = arm.geometry.train.mask.read()?;
        let target_map = arm.targets_map.read()?;
        let segments = arm.segments.read()?;
        let geometry_order = arm.geometry_order.read()?;
        let mut order = geometry_order.chunks_exact(4);
        if corpus.tokens().len() != segments.len()
            || target_map.len()
                != corpus
                    .eligible_len()
                    .checked_mul(4)
                    .ok_or("map length overflow")?
        {
            return Err("source event-map lengths differ".into());
        }
        let mut visited = vec![false; mask.len()];
        let mut full = vec![false; mask.len()];
        for &index in corpus.eligible_positions() {
            full[index] = true;
        }
        let mut geometry_total = 0u64;
        let mut events = Vec::with_capacity(corpus.eligible_len());
        for offset in target_map.chunks_exact(4) {
            let at = u32::from_le_bytes(offset.try_into()?) as usize;
            if !full.get(at).copied().unwrap_or(false) || std::mem::replace(&mut visited[at], true)
            {
                return Err("source events must bijectively address legal unique targets".into());
            }
            let selected = mask[at];
            geometry_total += u64::from(selected);
            if selected == 1 {
                let order_position = order.next().ok_or("geometry event order too short")?;
                if order_position != offset {
                    return Err("geometry order differs from canonical source events".into());
                }
            }
            events.push(
                corpus.tokens()[at] | (u32::from(segments[at]) << 16) | (u32::from(selected) << 24),
            );
        }
        if geometry_total != arm.geometry.train.expected_eligible_targets
            || order.len() != 0
            || !order.remainder().is_empty()
        {
            return Err("geometry mask selects positions outside the full target inventory".into());
        }
        Ok(events)
    };
    if open(a)? != open(b)? {
        return Err("matched source events differ in token, segment or geometry selection".into());
    }
    Ok(())
}

pub fn prepare(donor: &Path, tokenizer: &Path, out: &Path) -> Result<()> {
    let manifest = file_pin(donor.join("receipt.json"))?;
    let donor_receipt: serde_json::Value = serde_json::from_slice(&manifest.read()?)?;
    if donor_receipt["schema"] != "uor-r4.visible-answer-donor/1"
        || donor_receipt["groups"].as_u64() != Some(EXPECTED_GROUPS as u64)
        || donor_receipt["geometry_seed"].as_u64() != Some(u64::from(SEED))
        || donor_receipt["geometry_max_targets_per_group"].as_u64() != Some(MAX_PER_GROUP)
        || donor_receipt["reconstructed_original_r_byte_equal"] != true
        || donor_receipt["same_ordered_original_groups"] != true
    {
        return Err("donor receipt did not establish authenticated source framing or common target sampling".into());
    }
    let groups = file_pin(donor.join("groups.jsonl"))?;
    let group_count = BufReader::new(fs::File::open(&groups.path)?)
        .lines()
        .try_fold(0usize, |count, line| -> Result<usize> {
            let row: serde_json::Value = serde_json::from_str(&line?)?;
            if !row["geometry_targets_both_arms"]
                .as_u64()
                .is_some_and(|selected| (1..=MAX_PER_GROUP).contains(&selected))
            {
                return Err("donor group lacks its required geometry coverage".into());
            }
            Ok(count + 1)
        })?;
    if group_count != EXPECTED_GROUPS {
        return Err(format!("expected {EXPECTED_GROUPS} donor groups, got {group_count}").into());
    }

    fs::create_dir(out)?;
    // The driver always binds a separate DEV source. This fixture is never
    // trained or evaluated in this campaign. The real held scoring is external.
    let dev_tokens = out.join("unused-dev.tokens.u16");
    let dev_mask = out.join("unused-dev.mask.u8");
    write_new(&dev_tokens, &uort(&[0, 1]))?;
    write_new(&dev_mask, &[0, 1])?;
    let dev = source(Split::Dev, file_pin(dev_tokens)?, dev_mask)?;
    let tokenizer = file_pin(tokenizer.to_owned())?;
    let arms = [
        read_arm(donor, ARMS[0], &dev, &tokenizer)?,
        read_arm(donor, ARMS[1], &dev, &tokenizer)?,
    ];
    for pin in std::iter::once(&groups).chain(arms.iter().flat_map(|arm| {
        [
            &arm.full.tokens,
            &arm.full.mask,
            &arm.geometry.train.mask,
            &arm.targets_map,
            &arm.geometry_order,
            &arm.segments,
        ]
    })) {
        let name = pin
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("invalid donor file name")?;
        if donor_receipt["files"][name]["sha256"].as_str() != Some(pin.sha256.as_str()) {
            return Err(format!("donor receipt digest differs for {name}").into());
        }
    }
    check_events(&arms)?;
    let full_targets = arms[0].full.expected_eligible_targets;
    let geometry_targets = arms[0].geometry.train.expected_eligible_targets;
    if geometry_targets > group_count as u64 * MAX_PER_GROUP
        || donor_receipt["geometry_target_count"].as_u64() != Some(geometry_targets)
    {
        return Err("geometry target dose exceeds the group bound".into());
    }
    let plan = Plan {
        schema: "uor-r4.codex-4312.train-plan/1".to_owned(),
        donor_manifest: manifest,
        groups,
        group_count,
        full_targets,
        geometry_targets,
        geometry_batches: geometry_targets.div_ceil(u64::from(BATCH)),
        arms,
    };
    let plan_path = out.join("plan.json");
    write_json(&plan_path, &plan)?;
    let plan_digest = sha256(&fs::read(plan_path)?);
    println!(
        "{}",
        serde_json::json!({
            "plan_sha256": plan_digest,
            "donor_manifest_sha256": plan.donor_manifest.sha256,
            "reasoning_first_lineage_sha256": plan.arms[0].lineage_sha256,
            "answer_first_lineage_sha256": plan.arms[1].lineage_sha256,
            "reasoning_first_geometry_order_sha256": plan.arms[0].geometry_order.sha256,
            "answer_first_geometry_order_sha256": plan.arms[1].geometry_order.sha256,
            "full_targets": plan.full_targets,
            "geometry_targets": plan.geometry_targets,
            "geometry_batches": plan.geometry_batches,
        })
    );
    Ok(())
}
