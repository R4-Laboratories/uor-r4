#![forbid(unsafe_code)]

mod adapt;
mod report;
mod verify;
mod wire;
mod writer;

use std::path::Path;
use wire::{Evidence, PROPOSED, Result, Roster, pinned, require};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let out = std::io::stdout().lock();
    match args.as_slice() {
        [command, roster_path, pin] if command == "coverage" => {
            let roster: Roster = pinned(Path::new(roster_path), pin)?;
            let checked = verify::roster(&roster)?;
            serde_json::to_writer(out, &report::coverage(&roster, &checked, pin))?;
            return Ok(());
        }
        [command, mode, blind, gold, truth, tokenized, features] if command == "opening" => {
            serde_json::to_writer(
                out,
                &adapt::opening(mode, blind, gold, truth, tokenized, features)?,
            )?;
            return Ok(());
        }
        [command, study, manifest, identities, directory] if command == "validation" => {
            let study = match study.as_str() {
                "4340" => wire::Study::Original4340,
                "4341" => wire::Study::Expanded4341,
                _ => return Err(usage().into()),
            };
            serde_json::to_writer(
                out,
                &adapt::validation(study, manifest, identities, directory)?,
            )?;
            return Ok(());
        }
        [command, class, blind, gold, truth, tokenized, features] if command == "controls" => {
            serde_json::to_writer(
                out,
                &adapt::controls(class, [blind, gold, truth, tokenized, features])?,
            )?;
            return Ok(());
        }
        [command, requests, cold] if command == "swaps" => {
            serde_json::to_writer(out, &adapt::swaps(requests, cold)?)?;
            return Ok(());
        }
        [command, arm, blind, cold, gpu] if command == "panel" => {
            serde_json::to_writer(out, &adapt::panel(arm, blind, cold, gpu)?)?;
            return Ok(());
        }
        _ => {}
    }
    let (mode, args) = args.split_first().ok_or_else(usage)?;
    let (roster_path, roster_sha, evidence_path, evidence_sha, extra) = match args {
        [roster, pin, evidence, evidence_pin, remaining @ ..] => {
            (roster, pin, evidence, evidence_pin, remaining)
        }
        _ => return Err(usage().into()),
    };
    require(
        (mode == "source" && extra.is_empty()) || (mode == "writer" && extra.len() == 2),
        usage(),
    )?;
    let frozen: Roster = pinned(Path::new(roster_path), roster_sha)?;
    let checked = verify::roster(&frozen)?;
    let evidence: Evidence = pinned(Path::new(evidence_path), evidence_sha)?;
    let models = verify::evidence(&frozen, roster_sha, &checked, &evidence)?;
    let source = report::score(&frozen, &checked, &models, roster_sha, evidence_sha);
    if let [writer_path, writer_sha] = extra {
        let artifact = evidence
            .models
            .iter()
            .find(|row| row.arm == PROPOSED)
            .ok_or_else(usage)?;
        let receipt = pinned(Path::new(writer_path), writer_sha)?;
        let grade = writer::grade(
            &source,
            &checked,
            &models[&PROPOSED],
            &artifact.artifact_sha256,
            writer_sha,
            &receipt,
        )?;
        serde_json::to_writer(out, &grade)?;
    } else {
        serde_json::to_writer(out, &source)?;
    }
    Ok(())
}

fn usage() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "usage: validation-transfer-score coverage ROSTER ROSTER_SHA\n       validation-transfer-score source ROSTER ROSTER_SHA EVIDENCE EVIDENCE_SHA\n       validation-transfer-score writer ROSTER ROSTER_SHA EVIDENCE EVIDENCE_SHA WRITER WRITER_SHA",
    )
}

#[cfg(test)]
mod tests;
