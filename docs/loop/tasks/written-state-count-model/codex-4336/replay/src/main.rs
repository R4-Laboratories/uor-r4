#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, BufRead};
use std::path::Path;
use uor_r4_integer::discrete_d11::{native_copy, native_source_choice, IntegerDiscreteModel};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Trace {
    context: usize,
    artifact_sha256: String,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    authenticated_copy_steps: Vec<Value>,
    verified_span_release: SpanTrace,
}

#[derive(Deserialize)]
struct SpanTrace {
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    ordinary_id: u32,
    committed_id: u32,
    commit: String,
}

#[derive(Default, Serialize)]
struct Counts {
    transcripts: usize,
    transitions: usize,
    commits_different_from_ordinary: usize,
    successors_after_changed_commit: usize,
    successors_with_changed_read_code: usize,
    successors_with_changed_recurrence_code: usize,
    successors_with_changed_native_proposal: usize,
}

fn check(model: &IntegerDiscreteModel, trace: Trace, totals: &mut Counts) -> Result<()> {
    let limit = model.shape().context as usize;
    let count = trace.generated_ids.len();
    if trace.artifact_sha256 != model.artifact_sha256()
        || limit != trace.context
        || trace.prompt_ids.is_empty()
        || trace.prompt_ids.len() > limit
        || count == 0
        || count != trace.verified_span_release.steps.len()
        || count != trace.authenticated_copy_steps.len()
    {
        return Err("inconsistent artifact, prompt, window or trace lengths".into());
    }
    let mut prefix = trace.prompt_ids.clone();
    let mut history = Vec::new();
    let release = native_source_choice::OrdinaryPeriodRelease::new(model)?;
    let mut prior_ordinary = None;
    for (offset, step) in trace.verified_span_release.steps.iter().enumerate() {
        let mut event = if offset == 0 {
            native_copy::Session::new(model)?.step(&prefix)?
        } else {
            native_copy::observe_generate(model, &prefix)?
        };
        if let Some(prior) = prior_ordinary {
            let mut alternative = prefix.clone();
            *alternative.last_mut().ok_or("empty self-fed window")? = prior;
            let other = native_copy::observe_generate(model, &alternative)?;
            totals.successors_after_changed_commit += 1;
            totals.successors_with_changed_read_code +=
                usize::from(event.read_code != other.read_code);
            totals.successors_with_changed_recurrence_code +=
                usize::from(event.recurrence_code != other.recurrence_code);
            totals.successors_with_changed_native_proposal +=
                usize::from(event.native_proposal != other.native_proposal);
        }
        release.apply(&trace.prompt_ids, &history, &mut event);
        if event.committed_id != step.ordinary_id
            || trace.generated_ids[offset] != step.committed_id
        {
            return Err(
                format!("ordinary proposal or self-fed ID mismatch at offset {offset}").into(),
            );
        }
        prior_ordinary = (event.committed_id != step.committed_id).then_some(event.committed_id);
        totals.commits_different_from_ordinary += usize::from(prior_ordinary.is_some());
        match step.commit.as_str() {
            "ordinary" if event.committed_id == step.committed_id => {}
            "copy" | "supplied_stop" => {
                event.committed_id = step.committed_id;
                event.copied = false;
            }
            _ => return Err(format!("invalid commit at offset {offset}").into()),
        }
        if serde_json::to_value(&event)? != trace.authenticated_copy_steps[offset] {
            return Err(format!(
                "original writer fields or actual history differ at offset {offset}"
            )
            .into());
        }
        if step.committed_id == native_copy::EOS && offset + 1 != count {
            return Err("tokens after EOS".into());
        }
        history.push(event);
        if prefix.len() == limit {
            prefix.remove(0);
        }
        prefix.push(step.committed_id);
        totals.transitions += 1;
    }
    totals.transcripts += 1;
    Ok(())
}

fn main() -> Result<()> {
    let artifact = std::env::args()
        .nth(1)
        .ok_or("usage: check-span-self-feed BALANCED < client-records.jsonl")?;
    let model = IntegerDiscreteModel::load(Path::new(&artifact))?;
    let mut totals = Counts::default();
    for line in io::stdin().lock().lines() {
        check(&model, serde_json::from_str(&line?)?, &mut totals)?;
    }
    if totals.transcripts == 0 || totals.successors_after_changed_commit == 0 {
        return Err("no usable self-fed intervention".into());
    }
    println!("{}", serde_json::to_string(&totals)?);
    Ok(())
}
