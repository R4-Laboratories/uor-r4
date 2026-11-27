//! Run the shipped D11 CLI on plain user turns, preserving actual generated history.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::{env, fs};

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::dialogue::DialogueProtocol;
use uor_r4_tokenizer::ByteBpeTokenizer;

const TOKENIZER_SHA256: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const MAX_TOKENS: usize = 16;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Conversation {
    id: String,
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    user: String,
    accept: Vec<String>,
}

fn refusal(output: Output, contains: &str) -> Result<Value> {
    let stderr = String::from_utf8(output.stderr)?;
    if output.status.success() || !output.stdout.is_empty() || !stderr.contains(contains) {
        return Err(format!("expected refusal containing {contains:?}, got {stderr:?}").into());
    }
    Ok(json!({"exit_code": output.status.code(), "stderr": stderr}))
}

fn probe_interfaces(
    chat: &Path,
    source: &Path,
    artifact: &Path,
    vocab: &Path,
    prefix: &[u32],
) -> Result<()> {
    let mut child = Command::new(source)
        .arg(artifact)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().ok_or("missing source-choice stdin")?;
    writeln!(input, "{}", json!({"prefix": prefix}))?;
    drop(input);
    let no_positions = refusal(child.wait_with_output()?, "missing field `positions`")?;
    let stack = Command::new(chat)
        .arg("--stack")
        .arg(artifact)
        .arg("--tokenizer")
        .arg(vocab)
        .args(["--say", "What color is snow?", "--max-tokens", "16"])
        .output()?;
    let stack_stderr = String::from_utf8(stack.stderr)?;
    if stack.status.success() || !stack.stdout.is_empty() || stack_stderr.is_empty() {
        return Err("stack interface did not refuse the discrete artifact as expected".into());
    }
    println!(
        "{}",
        json!({"kind":"interface", "grounded_without_positions":no_positions,
        "stack": {"exit_code":stack.status.code(),"stderr":stack_stderr}})
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [chat, source, artifact, vocab, panel, artifact_sha, arm] = args.as_slice() else {
        return Err("usage: paired-ordinary-dialogue-probe UOR_CHAT SOURCE_CHOICE ARTIFACT TOKENIZER PANEL ARTIFACT_SHA rowwise|paired|base".into());
    };
    let arm = arm.to_str().ok_or("non-text arm")?;
    if !matches!(arm, "rowwise" | "paired" | "base") {
        return Err(format!("unexpected arm {arm}").into());
    }
    let expected_sha = artifact_sha.to_str().ok_or("non-text artifact digest")?;
    let tokenizer_bytes = fs::read(vocab)?;
    if format!("{:x}", Sha256::digest(&tokenizer_bytes)) != TOKENIZER_SHA256 {
        return Err("tokenizer digest changed".into());
    }
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&tokenizer_bytes).ok_or("invalid tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let encoder = protocol.bind(&tokenizer)?;
    let conversations: Vec<Conversation> = serde_json::from_slice(&fs::read(panel)?)?;
    let mut seen_ids = BTreeSet::new();
    let mut seen_prompts = BTreeSet::new();
    for conversation in &conversations {
        if conversation.id.is_empty()
            || !seen_ids.insert(&conversation.id)
            || conversation.turns.is_empty()
        {
            return Err("invalid or repeated conversation".into());
        }
        for turn in &conversation.turns {
            if turn.user.trim().is_empty()
                || !seen_prompts.insert(&turn.user)
                || turn.accept.is_empty()
                || turn.accept.iter().any(|text| text.trim().is_empty())
            {
                return Err("invalid or repeated turn".into());
            }
        }
    }
    let mut dialogs_exact = 0;
    let mut turn_counts = [[0usize; 4]; 2];
    use uor_r4_tokenizer::dialogue::Message;
    for (index, conversation) in conversations.iter().enumerate() {
        let mut history = Vec::new();
        let mut conversation_exact = true;
        for (turn_idx, turn) in conversation.turns.iter().enumerate() {
            let followup = turn_idx > 0;
            let appended = if followup {
                encoder.encode_user_prefix(&turn.user, true)
            } else {
                encoder.encode_assistant_prefix(&[Message {
                    role: "user",
                    content: &turn.user,
                }])
            };
            if appended.special_token_occurrences != 0 || appended.emitted_turns != 1 {
                return Err("unexpected tokens in ordinary user message".into());
            }
            history.extend(appended.tokens);
            if index == 0 && !followup && arm != "base" {
                probe_interfaces(
                    Path::new(chat),
                    Path::new(source),
                    Path::new(artifact),
                    Path::new(vocab),
                    &history,
                )?;
            }
            let mut cmd = Command::new(chat);
            cmd.arg("--discrete")
                .arg(artifact)
                .args(["--protocol", "2", "--max-tokens"])
                .arg(MAX_TOKENS.to_string());
            if followup {
                let ids = history
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                cmd.arg("--prompt-ids").arg(ids);
            } else {
                cmd.arg("--tokenizer")
                    .arg(vocab)
                    .arg("--say")
                    .arg(&turn.user);
            }
            let output = cmd.output()?;
            if !output.status.success() {
                return Err(format!(
                    "{} turn {}: {}",
                    conversation.id,
                    turn_idx + 1,
                    String::from_utf8_lossy(&output.stderr)
                )
                .into());
            }
            let record: Value = serde_json::from_slice(&output.stdout)?;
            let ids: Vec<u32> = serde_json::from_value(record["generated_ids"].clone())?;
            let prompt: Vec<u32> = serde_json::from_value(record["prompt_ids"].clone())?;
            if record["artifact_sha256"] != expected_sha
                || prompt != history
                || ids.len() > MAX_TOKENS
            {
                return Err(format!(
                    "invalid CLI response identity for {} turn {}",
                    conversation.id,
                    turn_idx + 1
                )
                .into());
            }
            let eos_position = ids.iter().position(|id| *id == protocol.eos_id);
            let eos = eos_position.is_some();
            let decoded = tokenizer.decode(&ids[..eos_position.unwrap_or(ids.len())]);
            if !followup
                && (record["text"] != decoded
                    || record["stop"] != (if eos { "eos" } else { "max_tokens" }))
            {
                return Err("text mode EOS or text differs from the raw ID trace".into());
            }
            let text = protocol.reply_text(&decoded);
            let text_match = turn.accept.iter().any(|target| target == text);
            let exact = eos && text_match;
            conversation_exact &= exact;
            let counts = &mut turn_counts[usize::from(followup)];
            counts[0] += 1;
            counts[1] += usize::from(eos);
            counts[2] += usize::from(text_match);
            counts[3] += usize::from(exact);
            println!(
                "{}",
                json!({"kind":"turn", "conversation":conversation.id, "turn":turn_idx + 1,
                "user":turn.user,"accepted":turn.accept,"text":text,"emitted":ids,
                "text_exact":text_match,"eos":eos,"exact_through_eos":exact,"prompt_ids":prompt,
                "eos_position":eos_position,"model_schema":record["model_schema"],
                "decoding": record["decoding"],"artifact_sha256":expected_sha})
            );
            history.extend(&ids[..eos_position.map_or(ids.len(), |i| i + 1)]);
        }
        dialogs_exact += usize::from(conversation_exact);
    }
    let counts = |ix: usize| {
        let [turns, eos, text_exact, exact] = turn_counts[ix];
        json!({"turns":turns,"eos":eos,"text_exact":text_exact,"exact_through_eos":exact})
    };
    let out = io::stdout();
    writeln!(
        out.lock(),
        "{}",
        json!({"kind":"summary", "arm":arm,
        "conversations":conversations.len(),"complete_conversations":dialogs_exact,
        "max_tokens": MAX_TOKENS, "first": counts(0), "followup": counts(1)})
    )?;
    Ok(())
}
