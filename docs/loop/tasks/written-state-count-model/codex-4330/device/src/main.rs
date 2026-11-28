#![deny(unsafe_op_in_unsafe_fn)]

mod finite;
mod gpu;
mod input;

use gpu::Device;
use input::{sha, weights_bytes, Geometry};
use serde::Serialize;
use serde_json::json;
use std::{
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const CONTEXT: usize = 192;
const VOCAB: usize = 4096;
const STRIDE: usize = CONTEXT + 8;
const FEATURES: usize = 16;
const STATE_WORDS: usize = FEATURES * 32 + 1;
const EPOCHS: usize = 4;
const WEIGHT_LIMIT: i16 = 4096;
const MARGIN: i32 = 8;
const NONE: u32 = u32::MAX;

fn require(condition: bool, why: &str) -> Result<()> {
    if !condition {
        return Err(why.into());
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn write_jsonl(path: &Path, values: impl IntoIterator<Item = impl Serialize>) -> Result<String> {
    let mut file = BufWriter::new(File::create_new(path)?);
    for value in values {
        serde_json::to_writer(&mut file, &value)?;
        writeln!(file)?;
    }
    file.flush()?;
    Ok(sha(&fs::read(path)?))
}

fn fit(image: &Path, geometry: Geometry, initial: &Path, tape: &Path, out: &Path) -> Result<()> {
    let initial = input::weights(initial, true)?;
    let (rows, tape_sha) = input::rows(tape, &geometry.classes, true)?;
    require(
        rows.iter().any(|row| row.target.is_some()) && rows.iter().any(|row| row.target.is_none()),
        "scientific fit tape needs both source and NoRead supervision",
    )?;
    fs::create_dir(out)?;
    let gpu = Device::open(image, &geometry)?;
    let first = gpu.fit(&rows, &initial)?;
    let repeated = gpu.fit(&rows, &initial)?;
    require(
        first == repeated,
        "repeat fit differed in updates or saved state",
    )?;
    let mut snapshot_sha256 = Vec::new();
    for (index, state) in first.snapshots.iter().enumerate() {
        let bytes = weights_bytes(state);
        let path = out.join(format!("epoch-{}.i16le", index + 1));
        File::create_new(&path)?.write_all(&bytes)?;
        require(
            fs::read(path)? == bytes,
            "saved epoch i16 words failed readback",
        )?;
        snapshot_sha256.push(sha(&bytes));
    }
    let state = weights_bytes(first.snapshots.last().ok_or("fit returned no state")?);
    let weights_path = out.join("weights.i16le");
    File::create_new(&weights_path)?.write_all(&state)?;
    require(
        input::weights(&weights_path, false)? == first.snapshots[EPOCHS - 1],
        "independent fitted i16 readback differs",
    )?;
    let events_sha = write_jsonl(&out.join("fit-events.jsonl"), &first.events)?;
    let updates_by_epoch = first
        .events
        .chunks_exact(rows.len())
        .map(|events| {
            let positive = events
                .iter()
                .filter(|e| e.update && e.gold_pos.is_some())
                .count();
            let negative = events
                .iter()
                .filter(|e| e.update && e.gold_pos.is_none())
                .count();
            json!({"positive":positive, "negative":negative})
        })
        .collect::<Vec<_>>();
    write_json(
        &out.join("fit.json"),
        &json!({
            "schema": "uor-r4.whole-turn-ordered-fit/1",
            "gpu": gpu.name,
            "cubin_sha256": gpu.image_sha,
            "geometry": Geometry::identity(),
            "train_tape_sha256": tape_sha,
            "initial_state_sha256": sha(&weights_bytes(&initial)),
            "state_sha256": sha(&state),
            "snapshots_sha256": snapshot_sha256,
            "fit_events_sha256": events_sha,
            "rows_per_epoch": rows.len(),
            "positive_rows_per_epoch": rows.iter().filter(|row| row.target.is_some()).count(),
            "epochs": EPOCHS,
            "margin": MARGIN,
            "weight_bound": WEIGHT_LIMIT,
            "changed_words_from_initial": first.snapshots[EPOCHS - 1].iter().zip(&initial)
                .filter(|(new, original)| new != original).count(),
            "updates_by_epoch": updates_by_epoch,
            "repeat_fit_exact": true,
        }),
    )
}

fn score(
    image: &Path,
    geometry: Geometry,
    initial: &Path,
    fitted: &Path,
    tape: &Path,
    out: &Path,
) -> Result<()> {
    let initial = input::weights(initial, true)?;
    let fitted = input::weights(fitted, false)?;
    let (rows, tape_sha) = input::rows(tape, &geometry.classes, false)?;
    fs::create_dir(out)?;
    let gpu = Device::open(image, &geometry)?;
    let ordered = rows.iter().collect::<Vec<_>>();
    let reversed = rows.iter().rev().collect::<Vec<_>>();
    let decisions = gpu.score(&ordered, &initial, &fitted, &geometry.classes)?;
    let confirm = gpu.score(&reversed, &initial, &fitted, &geometry.classes)?;
    require(
        decisions.iter().eq(confirm.iter().rev()),
        "reversed GPU evaluation differed field by field",
    )?;
    let decisions_sha = write_jsonl(&out.join("decisions.jsonl"), &decisions)?;
    write_json(
        &out.join("score.json"),
        &json!({
            "schema": "uor-r4.whole-turn-ordered-score/1",
            "gpu": gpu.name,
            "cubin_sha256": gpu.image_sha,
            "geometry": Geometry::identity(),
            "eval_tape_sha256": tape_sha,
            "initial_state_sha256": sha(&weights_bytes(&initial)),
            "fitted_state_sha256": sha(&weights_bytes(&fitted)),
            "decisions_sha256": decisions_sha,
            "rows": decisions.len(),
            "cases_with_framed_candidates": decisions.iter().filter(|row| !row.old_framed.candidates.is_empty()).count(),
            "labels_removed_before_device": true,
            "reverse_eval_exact": true,
        }),
    )
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let paths = args.iter().map(Path::new).collect::<Vec<_>>();
    match paths.as_slice() {
        [mode, image, roots, classes, initial, tape, out] if mode.as_os_str() == "fit" => {
            fit(image, Geometry::load(roots, classes)?, initial, tape, out)
        }
        [mode, image, roots, classes, initial, fitted, tape, out] if mode.as_os_str() == "score" => {
            score(image, Geometry::load(roots, classes)?, initial, fitted, tape, out)
        }
        [mode, image, roots, classes, out] if mode.as_os_str() == "fixture" => {
            finite::run(image, Geometry::load(roots, classes)?, out)
        }
        _ => Err("whole-turn-device fit CUBIN ROOTS CLASSES INITIAL TRAIN NEW_OUT\nwhole-turn-device score CUBIN ROOTS CLASSES INITIAL FITTED EVAL NEW_OUT\nwhole-turn-device fixture CUBIN ROOTS CLASSES NEW_OUT".into()),
    }
}
