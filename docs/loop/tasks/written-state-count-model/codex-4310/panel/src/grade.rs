use super::{
    file_sha, short_reply, verify, Domain, PanelRow, Result, Split, BOS, EOS, TOKENIZER_SHA,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::ByteBpeTokenizer;

const ARMS: [&str; 6] = [
    "frozen",
    "zero",
    "learned",
    "wrong-address",
    "shuffled-role",
    "role-blind",
];
const P4_CONTEXT: usize = 2048;

#[derive(Deserialize)]
struct Generation {
    id: String,
    domain: Domain,
    arm: String,
    tokens: Vec<u16>,
    ended_eos: bool,
    cap: usize,
    gold: Vec<u16>,
    steps: Vec<TraceStep>,
}

#[derive(Deserialize)]
struct TraceStep {
    context_tokens: usize,
    token_id: u16,
    s4_source: Option<Source>,
}

#[derive(Deserialize)]
struct Source {
    resolved_role: Option<u8>,
    source_pos: Option<usize>,
    source_token: Option<u16>,
    transfer_q48: u64,
}

#[derive(Default)]
struct Exposure {
    admitted: usize,
    active: usize,
    from_prompt: usize,
    from_generated_reply: usize,
    first: Option<serde_json::Value>,
    first_active: Option<serde_json::Value>,
}

#[derive(Default, Serialize)]
struct Counts {
    rows: usize,
    exact_complete: usize,
    exact_through_eos: usize,
    exact_through_cap: usize,
    first_token_correct: usize,
    ended_eos: usize,
    ended_cap: usize,
    premature_eos: usize,
    censored_cap: usize,
    generated_tokens: usize,
    matched_prefix_tokens: usize,
    differing_from_frozen: usize,
    edit_gains_over_frozen: usize,
    edit_losses_vs_frozen: usize,
    source_trace_steps: usize,
    source_steps_admitted: usize,
    source_steps_active: usize,
    source_steps_from_prompt: usize,
    source_steps_from_generated_reply: usize,
}

fn shared_prefix(left: &[u16], right: &[u16]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

fn observe_token(tokenizer: &ByteBpeTokenizer, token: Option<&u16>) -> serde_json::Value {
    token.map_or(serde_json::Value::Null, |&id| {
        let bytes = tokenizer.decode_bytes(&[u32::from(id)]);
        json!({ "id": id, "bytes_hex": hex::encode(&bytes), "text": String::from_utf8_lossy(&bytes) })
    })
}

fn decode(tokenizer: &ByteBpeTokenizer, ids: &[u16]) -> String {
    tokenizer.decode(&ids.iter().copied().map(u32::from).collect::<Vec<_>>())
}

fn exposure(
    tokenizer: &ByteBpeTokenizer,
    row: &PanelRow,
    generated: &Generation,
) -> Result<Exposure> {
    let mut out = Exposure::default();
    if generated.steps.len() != generated.tokens.len() {
        return Err("generation must trace every emitted token".into());
    }
    for (index, step) in generated.steps.iter().enumerate() {
        if step.s4_source.is_some() != (generated.arm != "frozen") {
            return Err("source trace presence disagrees with the selected arm".into());
        }
        let context_len = row.prefix.len() + index;
        if step.token_id != generated.tokens[index]
            || step.context_tokens != context_len.min(P4_CONTEXT)
        {
            return Err("generation trace token or context count disagrees".into());
        }
        let Some(source) = &step.s4_source else {
            continue;
        };
        let (Some(pos), Some(token)) = (source.source_pos, source.source_token) else {
            if source.transfer_q48 != 0
                || source.source_pos.is_some()
                || source.source_token.is_some()
            {
                return Err("incomplete or active unadmitted source trace".into());
            }
            continue;
        };
        if pos >= step.context_tokens {
            return Err("source is outside causal context".into());
        }
        let full_pos = context_len - step.context_tokens + pos;
        let original = row.prefix.get(full_pos).or_else(|| {
            full_pos
                .checked_sub(row.prefix.len())
                .and_then(|generated_pos| generated.tokens.get(generated_pos))
        });
        if original != Some(&token) {
            return Err(format!(
                "source token identity disagrees for {} {} step {}",
                row.id,
                generated.arm,
                index + 1
            )
            .into());
        }
        out.admitted += 1;
        let from_prompt = full_pos < row.prefix.len();
        out.from_prompt += usize::from(from_prompt);
        out.from_generated_reply += usize::from(!from_prompt);
        let active = source.transfer_q48 != 0;
        out.active += usize::from(active);
        if out.first.is_none() || (active && out.first_active.is_none()) {
            let witness = json!({
                "emitted_token_number": index + 1,
                "source_token_number_in_history": full_pos + 1,
                "source_token": observe_token(tokenizer, Some(&token)),
                "source_from_prompt": from_prompt,
                "resolved_role_vote": source.resolved_role,
                "transfer_q48": source.transfer_q48,
            });
            if out.first.is_none() {
                out.first = Some(witness.clone());
            }
            if active && out.first_active.is_none() {
                out.first_active = Some(witness);
            }
        }
    }
    Ok(out)
}

fn read_lines<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

pub(crate) fn run(
    tokenizer_path: &Path,
    panel_path: &Path,
    generation_path: &Path,
    out: &Path,
) -> Result<()> {
    verify(tokenizer_path, TOKENIZER_SHA)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer_path)?)
        .ok_or("invalid tokenizer")?;
    let mut panel = BTreeMap::new();
    for row in read_lines::<PanelRow>(panel_path)? {
        if row.split != Split::Held || !row.free_run {
            continue;
        }
        if row.prefix.first() != Some(&BOS)
            || row.gold.is_empty()
            || row.gold.len() > row.cap
            || row.new_teacher_tokens.len() != row.gold.len()
            || ((row.domain == Domain::Conversation) != (row.gold.last() == Some(&EOS)))
        {
            return Err(format!("invalid held row {}", row.id).into());
        }
        if panel.insert(row.id.clone(), row).is_some() {
            return Err("duplicate held row".into());
        }
    }
    if panel.is_empty() {
        return Err("no held free-run rows".into());
    }
    let mut generations = BTreeMap::new();
    for generated in read_lines::<Generation>(generation_path)? {
        let row = panel
            .get(&generated.id)
            .ok_or_else(|| format!("unexpected generation {}", generated.id))?;
        if !ARMS.contains(&generated.arm.as_str())
            || generated.domain != row.domain
            || generated.gold != row.gold
            || generated.cap != row.cap
            || generated.tokens.is_empty()
            || generated.tokens.len() > generated.cap
            || generated
                .tokens
                .iter()
                .any(|&id| id == BOS || usize::from(id) >= tokenizer.vocab_size())
            || generated.ended_eos != (generated.tokens.last() == Some(&EOS))
            || (!generated.ended_eos && generated.tokens.len() != row.cap)
            || generated.tokens[..generated.tokens.len() - 1].contains(&EOS)
        {
            return Err(format!("invalid generation {} {}", generated.id, generated.arm).into());
        }
        let key = (generated.id.clone(), generated.arm.clone());
        if generations.insert(key, generated).is_some() {
            return Err("duplicate generation".into());
        }
    }
    let mut details = Vec::new();
    let mut counts = BTreeMap::<&str, BTreeMap<&str, Counts>>::new();
    for row in panel.values() {
        let observed: BTreeSet<_> = row.prefix.iter().copied().collect();
        let frozen = &generations
            .get(&(row.id.clone(), "frozen".to_owned()))
            .ok_or_else(|| format!("missing frozen generation {}", row.id))?
            .tokens;
        let band = if row.domain == Domain::Prose {
            "prose"
        } else if short_reply(row) {
            "short_chat"
        } else {
            "long_chat"
        };
        for arm in ARMS {
            let generated = generations
                .get(&(row.id.clone(), arm.to_owned()))
                .ok_or_else(|| format!("missing generation {} {arm}", row.id))?;
            let predicted = &generated.tokens;
            if arm == "zero" && predicted != frozen {
                return Err(format!(
                    "zero arm and frozen P4 emitted different tokens for {}",
                    row.id
                )
                .into());
            }
            let sources = exposure(&tokenizer, row, generated)?;
            let common = shared_prefix(&row.gold, predicted);
            let exact = row.gold == *predicted;
            let exact_eos = exact && row.domain == Domain::Conversation;
            let exact_cap = exact && row.domain == Domain::Prose;
            let premature_eos = generated.ended_eos
                && (row.domain == Domain::Prose || predicted.len() < row.gold.len());
            let censored_cap = !generated.ended_eos && row.domain == Domain::Conversation;
            let frozen_exact = row.gold == *frozen;
            let frozen_common = shared_prefix(frozen, predicted);
            let groups: &[&str] = if row.domain == Domain::Conversation {
                &[band, "conversation"]
            } else {
                &[band]
            };
            for &name in groups {
                let cell = counts.entry(name).or_default().entry(arm).or_default();
                cell.rows += 1;
                cell.exact_complete += usize::from(exact);
                cell.exact_through_eos += usize::from(exact_eos);
                cell.exact_through_cap += usize::from(exact_cap);
                cell.first_token_correct += usize::from(common > 0);
                cell.ended_eos += usize::from(generated.ended_eos);
                cell.ended_cap += usize::from(!generated.ended_eos);
                cell.premature_eos += usize::from(premature_eos);
                cell.censored_cap += usize::from(censored_cap);
                cell.generated_tokens += predicted.len();
                cell.matched_prefix_tokens += common;
                cell.differing_from_frozen += usize::from(predicted != frozen);
                cell.edit_gains_over_frozen += usize::from(exact && !frozen_exact);
                cell.edit_losses_vs_frozen += usize::from(!exact && frozen_exact);
                cell.source_trace_steps += generated.steps.len();
                cell.source_steps_admitted += sources.admitted;
                cell.source_steps_active += sources.active;
                cell.source_steps_from_prompt += sources.from_prompt;
                cell.source_steps_from_generated_reply += sources.from_generated_reply;
            }
            let first_diff = (!exact).then(|| {
                json!({
                    "zero_based_index": common,
                    "token_number": common + 1,
                    "gold": observe_token(&tokenizer, row.gold.get(common)),
                    "emitted": observe_token(&tokenizer, predicted.get(common)),
                    "gold_absent_at_teacher_position": row.new_teacher_tokens.get(common),
                })
            });
            let first_diff_from_frozen = (predicted != frozen).then(|| {
                json!({
                    "zero_based_index": frozen_common,
                    "token_number": frozen_common + 1,
                    "frozen": observe_token(&tokenizer, frozen.get(frozen_common)),
                    "emitted": observe_token(&tokenizer, predicted.get(frozen_common)),
                })
            });
            details.push(json!({
                "id": row.id,
                "band": band,
                "arm": arm,
                "end": if generated.ended_eos { "eos" } else { "cap" },
                "exact_complete": exact,
                "exact_through_eos": exact_eos,
                "exact_through_cap": exact_cap,
                "premature_eos": premature_eos,
                "censored_cap": censored_cap,
                "first_admitted_source": sources.first,
                "first_active_source": sources.first_active,
                "first_wrong": first_diff,
                "first_diff_from_frozen": first_diff_from_frozen,
                "matched_prefix_tokens": common,
                "new_emitted_tokens": predicted.iter().filter(|&&token| token != EOS && !observed.contains(&token)).count(),
                "prefix_text": decode(&tokenizer, &row.prefix),
                "gold_text": decode(&tokenizer, &row.gold),
                "emitted_text": decode(&tokenizer, predicted),
            }));
        }
    }
    let report = json!({
        "schema": "uor-r4.s4-source-generation-grade/1",
        "held_sha256": file_sha(panel_path)?,
        "generation_sha256": file_sha(generation_path)?,
        "tokenizer_sha256": TOKENIZER_SHA,
        "by_domain_and_chat_band": counts,
        "details": details,
    });
    let mut writer = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(out)?);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}
