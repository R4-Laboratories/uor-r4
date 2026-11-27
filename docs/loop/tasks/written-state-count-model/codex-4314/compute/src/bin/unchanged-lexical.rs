//! Score the delayed lexical positions through the frozen original D11 CUDA gate and writer.
use d11::{artifact::ArtifactTables, device::DeviceScorer, kernel::pack_rows_with_targets};
use serde::Deserialize;
use serde_json::json;
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
};
use uor_r4_integer::discrete_d11::native_opening::{COLON, SPACE};
#[derive(Deserialize)]
struct Input {
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    pair_id: String,
    side: u32,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: unchanged-lexical ORIGINAL-D11 LEXICAL-STEPS.JSONL NEW-OBS.JSONL".into(),
        );
    }
    let (base, _client) = ArtifactTables::from_bytes(&fs::read(&args[0])?)?;
    if base.artifact_sha256 != "47214eda18119b1e555cec858f9356c93246a23c9f34686962bc19ba05716049" {
        return Err("wrong frozen base".into());
    }
    let mut rows = Vec::new();
    for line in BufReader::new(File::open(&args[1])?).lines() {
        let row: Input = serde_json::from_str(&line?)?;
        if row.prefix.last() != Some(&COLON) {
            rows.push(row);
        }
    }
    if rows.len() != 2 || rows.iter().any(|r| r.prefix.last() != Some(&SPACE)) {
        return Err("delayed lexical cohort changed".into());
    }
    let gpu = DeviceScorer::new(0, &base)?;
    let run = |order: &[usize]| -> Result<_, Box<dyn Error>> {
        let pref = order
            .iter()
            .map(|&i| rows[i].prefix.as_slice())
            .collect::<Vec<_>>();
        let targets = order.iter().map(|&i| rows[i].target).collect::<Vec<_>>();
        let ids = order
            .iter()
            .map(|&i| [rows[i].target, rows[i].rival, SPACE])
            .collect::<Vec<_>>();
        Ok(gpu.score_token_probes(&pack_rows_with_targets(&pref, &targets)?, &ids)?)
    };
    let (prod, taps) = run(&[0, 1])?;
    let (rev, rtaps) = run(&[1, 0])?;
    if !prod.iter().eq(rev.iter().rev()) || !taps.iter().eq(rtaps.iter().rev()) {
        return Err("delayed-step CUDA replay differs".into());
    }
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?,
    );
    for ((r, decision), tap) in rows.iter().zip(&prod).zip(&taps) {
        if decision.status != 0 {
            return Err("frozen writer rejected a delayed lexical input".into());
        }
        let mut scores = Vec::new();
        let mut ranks = Vec::new();
        for start in [1013usize, 1036, 1059] {
            let probe = &tap[start..start + 23];
            let positive = probe[8] == 1;
            scores.push(if positive {
                u64::from(probe[0]) + (u64::from(probe[1]) << 32)
            } else {
                0
            });
            ranks.push(probe[10]);
        }
        serde_json::to_writer(
            &mut out,
            &json!({"pair_id":r.pair_id,"side":r.side,"target":r.target,"rival":r.rival,
            "scores":scores,"ranks":ranks,"selected":tap[6],"emitted":decision.prediction,"previous_token":SPACE,
            "state":"unchanged_base_after_space","base_sha256":base.artifact_sha256}),
        )?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}
