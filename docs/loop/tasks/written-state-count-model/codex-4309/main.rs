use serde_json::json;
use std::{error::Error, fs, path::PathBuf, time::Instant};
use uor_r4_discrete::p5_brier_core::extract_s1_from_uorlut01;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}

mod alias_eval;
mod alias_train;
#[allow(dead_code)]
#[path = "../codex-4308/pilot/src/engine.rs"]
mod engine;
#[allow(dead_code)]
#[path = "../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
pub mod lawful_likelihood;
#[allow(dead_code)]
#[path = "../codex-4308/pilot/src/measure.rs"]
mod measure;
#[path = "../codex-4308/pilot/src/replay.rs"]
mod replay;
#[allow(dead_code)]
#[path = "../codex-4308/pilot/src/rows.rs"]
mod rows;
#[path = "../codex-4308/pilot/src/sample.rs"]
mod sample;

const TRUNK: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
const P4: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const STANDARD: &str = "f681fad83f59376bf5b6a3598c1af0e89834bc6b98e6645c1b918544c990c0ac";

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "finite" {
        let gpu = engine::P5Gpu::new(0)?;
        gpu.check_finite_parity()?;
        alias_train::finite(&gpu)?;
        println!("finite GPU comparison passed");
        return Ok(());
    }
    if args.len() != 7 || args[0] != "run" {
        return Err(fail(
            "expected finite or run P4 TRUNK R DEV_TUNE UOR_CHAT NEW_OUT",
        ));
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    let (p4, trunk, train_path, dev_path, chat, out) = (
        &paths[0], &paths[1], &paths[2], &paths[3], &paths[4], &paths[5],
    );
    fs::create_dir(out)?;
    let start = Instant::now();
    measure::verify_file(trunk, TRUNK)?;
    let seed = extract_s1_from_uorlut01(&fs::read(trunk)?)?;
    let train = sample::Corpus::load(train_path, sample::TRAIN_SHA)?;
    let dev = sample::Corpus::load(dev_path, sample::DEV_SHA)?;
    if (
        train.record_count(),
        train.target_count(),
        dev.record_count(),
        dev.target_count(),
    ) != (4060, 15947584, 2670, 10807417)
        || seed.roots.len() != 16384
    {
        return Err(fail("input census or native root count changed"));
    }
    let reader = rows::RowsGpu::new(p4)?;
    if reader.artifact_sha256 != P4 {
        return Err(fail("incorrect ordinary P4 identity"));
    }
    let gpu = engine::P5Gpu::new(0)?;
    let load_us = start.elapsed().as_micros();
    let stage = Instant::now();
    let (standard, penalty) = alias_train::train(&gpu, &reader, &seed, &train, out)?;
    let train_us = stage.elapsed().as_micros();
    measure::verify_file(&out.join("states/standard-15.s1"), STANDARD)?;
    let stage = Instant::now();
    let witnesses = alias_eval::run(&gpu, &reader, &seed, &standard, &penalty, &dev, out)?;
    let eval_us = stage.elapsed().as_micros();
    drop(gpu);
    drop(reader);
    let stage = Instant::now();
    alias_eval::reload(
        &witnesses,
        p4,
        trunk,
        &out.join("states/penalty-15.s1"),
        chat,
        out,
    )?;
    let reload_us = stage.elapsed().as_micros();
    measure::json_file(
        &out.join("complete.json"),
        &json!({
            "complete":true, "load_us":load_us, "train_us":train_us,
            "eval_us":eval_us, "ordinary_reload_us":reload_us, "total_us":start.elapsed().as_micros(),
            "baseline_checkpoint_sha256": STANDARD, "frozen_p4_sha256":P4,
            "trunk_sha256":TRUNK,"train_sha256":sample::TRAIN_SHA,"dev_sha256":sample::DEV_SHA,
            "geometric_epoch":standard.epoch,"penalty_epoch":penalty.epoch,
        }),
    )?;
    Ok(())
}
