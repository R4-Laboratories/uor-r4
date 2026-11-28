#![forbid(unsafe_code)]

mod events;
mod groups;
mod input;

use serde_json::json;
use std::{
    env,
    error::Error,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const EOS: u32 = 1;
const WINDOW: usize = 192;
const SPLITS: [&str; 3] = ["train", "dev", "held"];

fn require(good: bool, message: &str) -> Result<()> {
    if !good {
        return Err(message.into());
    }
    Ok(())
}

fn fresh(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [tokenizer, donor, records, output] = args.as_slice() else {
        return Err("expected tokenizer, corrected DONOR directory, records JSONL and private output directory".into());
    };
    let output = Path::new(output);
    let corpus = input::replay(Path::new(tokenizer), Path::new(donor), Path::new(records))?;
    let partition = groups::assign(&corpus)?;
    let mut events = fresh(&output.join("events.jsonl"))?;
    let mut pairs = fresh(&output.join("pairs.jsonl"))?;
    let mut truth = fresh(&output.join("truth.jsonl"))?;
    let mut strata = fresh(&output.join("strata.jsonl"))?;
    let census = events::census(
        &corpus,
        &partition,
        events::Outputs {
            events: &mut events,
            pairs: &mut pairs,
            truth: &mut truth,
            strata: &mut strata,
        },
    )?;
    events.flush()?;
    pairs.flush()?;
    truth.flush()?;
    strata.flush()?;
    let receipt = json!({
        "schema": "uor-r4.ordinary-generic-termination-train-supply/1",
        "source": "corrected #4314 donor-v2, every-128th official D87 TRAIN records",
        "prior_exposure": "same subset used in #4312/#4314 and inspected in #4344; all three splits are internal",
        "records": corpus.records.len(),
        "turns": corpus.turns.len(),
        "visible_original_ids": corpus.visible_tokens,
        "cap_including_target_ids": WINDOW,
        "partitions": partition.audit,
        "visible_answer_overlap": groups::answer_overlap(&corpus, &partition),
        "census": census,
    });
    let mut summary = fresh(&output.join("summary.json"))?;
    serde_json::to_writer_pretty(&mut summary, &receipt)?;
    summary.write_all(b"\n")?;
    summary.flush()?;
    Ok(())
}
