use crate::{
    claim_path, digest, open_panel, require, require_digest, write_json, write_lines, Job, Prompt,
    Result, ARTIFACT_SHA, DECODE_LIMIT, OLD_CHAT_SHA,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};
use uor_r4_tokenizer::ByteBpeTokenizer;

const WORKERS: usize = 6;

#[derive(Deserialize, Serialize)]
pub struct Record {
    pub id: String,
    pub arm: String,
    pub client: Value,
    pub stderr: String,
}

pub fn baseline(arm: &str) -> bool {
    matches!(
        arm,
        "historical_auth"
            | "base_auth"
            | "new_auth"
            | "new_auth_quiet"
            | "historical_stock"
            | "base_stock"
            | "new_stock"
    )
}

pub struct Inputs<'a> {
    pub tokenizer: &'a Path,
    pub panel: &'a Path,
    pub old: &'a Path,
    pub base: &'a Path,
    pub new: &'a Path,
    pub artifact: &'a Path,
}

fn capture_one(inputs: &Inputs<'_>, prompt: &Prompt, job: &Job) -> Result<Record> {
    let binary = match job.arm.as_str() {
        "historical_auth" | "historical_stock" => inputs.old,
        "base_auth" | "base_stock" => inputs.base,
        _ => inputs.new,
    };
    let mut cmd = Command::new(binary);
    cmd.arg("--discrete")
        .arg(inputs.artifact)
        .arg("--tokenizer")
        .arg(inputs.tokenizer)
        .args([
            "--protocol",
            "2",
            "--say",
            &prompt.text,
            "--temperature",
            "0",
            "--seed",
            "0",
        ])
        .arg("--max-tokens")
        .arg(DECODE_LIMIT.to_string());
    if !job.arm.ends_with("_stock") {
        cmd.args(["--authenticated-copy", "--native-period-release"]);
        if !job.arm.ends_with("_quiet") {
            cmd.arg("--native-occurrence-trace");
        }
    }
    if !baseline(&job.arm) {
        cmd.arg("--verified-span-release")
            .arg(claim_path(inputs.panel, job));
    }
    let output = cmd.output()?;
    let stderr = String::from_utf8(output.stderr)?;
    require(output.status.success(), || {
        format!(
            "{}/{}: client {} ({stderr})",
            job.id, job.arm, output.status
        )
    })?;
    let mut client: Value = serde_json::from_slice(&output.stdout)?;
    client
        .as_object_mut()
        .ok_or("client must be a JSON object")?
        .remove("artifact");
    Ok(Record {
        id: job.id.clone(),
        arm: job.arm.clone(),
        client,
        stderr,
    })
}

pub fn includes(group: &str, job: &Job) -> bool {
    match group {
        "historical" => job.arm.starts_with("historical_") || job.arm.starts_with("base_"),
        "new" => job.arm.starts_with("new_"),
        "diagnostic" => !baseline(&job.arm),
        "all" => true,
        _ => false,
    }
}

pub fn capture(
    tokenizer: &ByteBpeTokenizer,
    inputs: &Inputs<'_>,
    out: &Path,
    group: &str,
) -> Result<()> {
    require(
        matches!(group, "all" | "historical" | "new" | "diagnostic"),
        || "unknown capture group".into(),
    )?;
    require_digest(inputs.old, OLD_CHAT_SHA)?;
    require_digest(inputs.artifact, ARTIFACT_SHA)?;
    let (prompts, _, jobs) = open_panel(tokenizer, inputs.panel)?;
    let jobs = jobs
        .iter()
        .filter(|job| includes(group, job))
        .map(|job| {
            let prompt = prompts
                .iter()
                .find(|prompt| prompt.id == job.id)
                .ok_or("no scheduled prompt")?;
            Ok((job, prompt))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut identity = json!({
        "group":group,
        "jobs":jobs.len(),
        "tokenizer_sha256":digest(inputs.tokenizer)?,
        "artifact_sha256":ARTIFACT_SHA,
        "historical_chat_sha256":OLD_CHAT_SHA,
        "base_chat_sha256":digest(inputs.base)?,
        "new_chat_sha256":null,
    });
    if group != "historical" {
        identity["new_chat_sha256"] = digest(inputs.new)?.into();
    }
    fs::create_dir(out)?;
    write_json(&out.join("inputs.json"), &identity)?;

    let next = AtomicUsize::new(0);
    let mut records = thread::scope(|scope| -> Result<Vec<(usize, Record)>> {
        let mut workers = Vec::new();
        for _ in 0..WORKERS.min(jobs.len()) {
            let next = &next;
            let jobs = &jobs;
            workers.push(scope.spawn(move || -> Result<Vec<_>> {
                let mut rows = Vec::new();
                while let Some((index, (job, prompt))) = {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    jobs.get(index).map(|row| (index, row))
                } {
                    rows.push((index, capture_one(inputs, prompt, job)?));
                }
                Ok(rows)
            }));
        }
        let mut records = Vec::new();
        for worker in workers {
            records.extend(
                worker
                    .join()
                    .map_err(|_| "client capture worker panicked")??,
            );
        }
        Ok(records)
    })?;
    records.sort_unstable_by_key(|(index, _)| *index);
    let records = records
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let pending = out.join("records.pending");
    write_lines(&pending, &records)?;
    fs::rename(pending, out.join("records.jsonl"))?;
    Ok(())
}
