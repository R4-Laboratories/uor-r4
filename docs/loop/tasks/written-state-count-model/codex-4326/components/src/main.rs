#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

mod geometry;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const WORDS: usize = 4096;
const PREFIX_BYTES: usize = 20_483 * 2;
const STATE_BYTES: usize = PREFIX_BYTES + WORDS * 2;
const ROWS: usize = 768;
const EPOCHS: usize = 4;
const SLOTS: usize = ROWS * EPOCHS * 2;
const COORDINATE_CAP: usize = 16;
const MARGIN: i64 = 8;

const SCHEDULE_SHA: &str = "1cb1cf2e2410adbce7430fd7db86911bc8ee1046b3d4d62ef7d401e99737650a";
const AUDIT_SHA: &str = "ddc4eb82f04fdce0b8adf9f831e48801d26916c64c4a5254ac5636fc89d44ac8";
const PREFIX_SHA: &str = "bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91";
const INITIAL_SHA: &str = "52f84dc839628abfe20f18f43a579393f4c29da79052247a68ee815e3622cd38";
const JOINT_SHA: &str = "2216c08d143fa15de31a7c31e009c791b5e56ad845f35bd89327afdbdcc29525";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Slot {
    epoch: usize,
    train_index: usize,
    pair_id: String,
    side: u8,
    first_stage: usize,
    first_copy: bool,
    joint_stage: usize,
    joint_copy: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Audit {
    slot: usize,
    before: i32,
    after: i32,
    link_visits: usize,
    unique_coordinate_visits: usize,
    changed_coordinates: Vec<usize>,
}

#[derive(Default, Serialize)]
struct Stage {
    slots: usize,
    copy_targets: usize,
    stop_targets: usize,
    link_visits: usize,
    unique_coordinate_visits: usize,
    capped_coordinate_opportunities: usize,
    violated_margins: usize,
    actual_coordinate_writes: usize,
    repeated_coordinate_writes: usize,
    max_abs_running_weight: u16,
}

#[derive(Serialize)]
struct Receipt {
    states_sha256: [String; 4],
    state_bytes: usize,
    checked_residual_words: usize,
    checked_slots: usize,
    full_matches_archived_bytes: bool,
    max_abs_running_joint_weight: u16,
    stages_period_then_post_b: [Stage; 2],
    nonzero_residual_by_component: [usize; 4],
    touched_by_both: usize,
    opposing_net: usize,
    exactly_cancelled: usize,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn verify_sha(bytes: &[u8], expected: &str, name: &str) -> Result<()> {
    let actual = sha(bytes);
    if actual != expected {
        return Err(format!("{name} SHA-256 mismatch: {actual}").into());
    }
    Ok(())
}

fn pinned(source: &Path, name: &str, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(source.join(name))?;
    verify_sha(&bytes, expected, name)?;
    Ok(bytes)
}

fn rows<T: for<'a> Deserialize<'a>>(bytes: &[u8]) -> Result<Vec<T>> {
    std::str::from_utf8(bytes)?
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line)
                .map_err(|error| format!("line {}: {error}", index + 1).into())
        })
        .collect()
}

fn check_slot(index: usize, tape: &[Slot], audit: &Audit) -> Result<(usize, i16, usize, bool)> {
    let slot = &tape[index];
    let epoch = index / (ROWS * 2);
    let offset = index % (ROWS * 2);
    let stage = usize::from(offset >= ROWS);
    if slot.epoch != epoch
        || slot.joint_stage != stage
        || slot.first_stage != 0
        || slot.train_index >= ROWS
        || slot.pair_id.is_empty()
        || slot.side > 1
    {
        return Err(format!("invalid scheduled row at slot {index}").into());
    }
    let first = &tape[slot.train_index];
    if first.pair_id != slot.pair_id
        || first.side != slot.side
        || first.first_copy != slot.first_copy
        || (stage == 0 && (slot.train_index != offset || slot.joint_copy != slot.first_copy))
        || (stage == 1 && (!slot.first_copy || slot.joint_copy != (offset % 2 == 1)))
    {
        return Err(format!("inconsistent row identity or target at slot {index}").into());
    }

    let support = audit.unique_coordinate_visits;
    let opportunity = support.min(COORDINATE_CAP);
    let sign: i16 = if slot.joint_copy { 1 } else { -1 };
    let violated = i64::from(sign) * i64::from(audit.before) < MARGIN;
    let writes = audit.changed_coordinates.len();
    if audit.slot != index
        || !(1..=WORDS).contains(&support)
        || audit.link_visits < support
        || writes != if violated { opportunity } else { 0 }
        || i64::from(audit.after) != i64::from(audit.before) + i64::from(sign) * writes as i64
    {
        return Err(format!("inconsistent signed margin/update cap at slot {index}").into());
    }
    for (visit, &address) in audit.changed_coordinates.iter().enumerate() {
        if address >= WORDS || audit.changed_coordinates[..visit].contains(&address) {
            return Err(format!("invalid or duplicate address at slot {index}").into());
        }
    }
    Ok((stage, sign, opportunity, violated))
}

fn state_bytes(prefix: &[u8], residual: &[i16]) -> Vec<u8> {
    prefix
        .iter()
        .copied()
        .chain(residual.iter().flat_map(|word| word.to_le_bytes()))
        .collect()
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn reconstruct(source: &Path, dest: &Path) -> Result<()> {
    let schedule: Vec<Slot> = rows(&pinned(source, "schedule.jsonl", SCHEDULE_SHA)?)?;
    let audit: Vec<Audit> = rows(&pinned(source, "joint-slots.jsonl", AUDIT_SHA)?)?;
    let original = pinned(source, "joint.i16le", JOINT_SHA)?;
    if schedule.len() != SLOTS || audit.len() != SLOTS || original.len() != STATE_BYTES {
        return Err("invalid schedule, audit, or full state length".into());
    }
    let prefix = &original[..PREFIX_BYTES];
    verify_sha(prefix, PREFIX_SHA, "frozen linked prefix")?;

    let mut partials = [[0i16; WORDS]; 2];
    let mut touched = [[false; WORDS]; 2];
    let mut chronological = [0i16; WORDS];
    let mut stages = [Stage::default(), Stage::default()];
    let mut max_abs_joint = 0;
    for (index, audit) in audit.iter().enumerate() {
        let (stage, sign, opportunity, violated) = check_slot(index, &schedule, audit)?;
        let stats = &mut stages[stage];
        stats.slots += 1;
        stats.copy_targets += usize::from(sign == 1);
        stats.stop_targets += usize::from(sign == -1);
        stats.link_visits += audit.link_visits;
        stats.unique_coordinate_visits += audit.unique_coordinate_visits;
        stats.capped_coordinate_opportunities += opportunity;
        stats.violated_margins += usize::from(violated);
        stats.actual_coordinate_writes += audit.changed_coordinates.len();
        for &address in &audit.changed_coordinates {
            stats.repeated_coordinate_writes +=
                usize::from(touched[0][address] || touched[1][address]);
            touched[stage][address] = true;
            let partial = &mut partials[stage][address];
            *partial = partial.checked_add(sign).ok_or("i16 component overflow")?;
            let joint = &mut chronological[address];
            *joint = joint
                .checked_add(sign)
                .ok_or("i16 chronological overflow")?;
            stats.max_abs_running_weight = stats.max_abs_running_weight.max(partial.unsigned_abs());
            max_abs_joint = max_abs_joint.max(joint.unsigned_abs());
        }
    }

    let mut receipt = Receipt {
        states_sha256: Default::default(),
        state_bytes: STATE_BYTES,
        checked_residual_words: WORDS,
        checked_slots: SLOTS,
        full_matches_archived_bytes: true,
        max_abs_running_joint_weight: max_abs_joint,
        stages_period_then_post_b: stages,
        nonzero_residual_by_component: [0; 4],
        touched_by_both: 0,
        opposing_net: 0,
        exactly_cancelled: 0,
    };
    for (address, joint) in chronological.iter().enumerate() {
        let period = partials[0][address];
        let post_b = partials[1][address];
        if period.checked_add(post_b) != Some(*joint) {
            return Err(format!("component sum differs or overflows at address {address}").into());
        }
        receipt.nonzero_residual_by_component[1] += usize::from(period != 0);
        receipt.nonzero_residual_by_component[2] += usize::from(post_b != 0);
        receipt.nonzero_residual_by_component[3] += usize::from(*joint != 0);
        receipt.touched_by_both += usize::from(touched[0][address] && touched[1][address]);
        let opposing = (period < 0 && post_b > 0) || (period > 0 && post_b < 0);
        receipt.opposing_net += usize::from(opposing);
        receipt.exactly_cancelled += usize::from(opposing && *joint == 0);
    }

    let states = [
        ("zero.i16le", state_bytes(prefix, &[0; WORDS])),
        ("period.i16le", state_bytes(prefix, &partials[0])),
        ("post-b.i16le", state_bytes(prefix, &partials[1])),
        ("full.i16le", state_bytes(prefix, &chronological)),
    ];
    verify_sha(&states[0].1, INITIAL_SHA, "zero residual state")?;
    if states[3].1 != original {
        return Err("reconstructed full native state differs from archived bytes".into());
    }

    fs::create_dir(dest)?;
    let mut manifest = String::new();
    for (index, (name, bytes)) in states.iter().enumerate() {
        let digest = sha(bytes);
        manifest.push_str(&format!("{digest}  {name}\n"));
        receipt.states_sha256[index] = digest;
        write_new(&dest.join(name), bytes)?;
    }
    write_new(
        &dest.join("reconstruction.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    write_new(&dest.join("SHA256SUMS"), manifest.as_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [source, dest] => reconstruct(Path::new(source), Path::new(dest)),
        [mode, prior, roots, components, dest] if mode == "geometry" => geometry::run(
            Path::new(prior),
            Path::new(roots),
            Path::new(components),
            Path::new(dest),
        ),
        _ => Err("usage: release-component-accounting FIT_INPUT NEW_COMPONENT_STATE_DIR | geometry PRIOR_RUN ROOTS COMPONENT_STATE_DIR NEW_GEOMETRY_DIR".into()),
    }
}
