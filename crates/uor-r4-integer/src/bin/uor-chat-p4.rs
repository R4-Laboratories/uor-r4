//! Host CLI for the native ordinary P4 compile graph.

use std::{
    env,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use serde_json::json;
use uor_r4_count_runtime::p4_serving::{P4Session, P4_MAX_CONTEXT};
use uor_r4_integer::{
    discrete_d11::count_deployment::{
        load_client_artifact, ClientArtifactFile, DiscreteCountModel, SCHEMA_P4,
    },
    ordinary_p4::{
        encode_discrete_messages, load_discrete_tokenizer, load_lexical, step, MAX_MESSAGE_BYTES,
    },
    IntegerError, Result,
};
use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};

const USAGE: &str = "Usage: uor-chat --discrete P4 (--prompt-ids CSV | --tokenizer JSON (--messages FILE | --say TEXT)) [--protocol 2] [--max-tokens N] [--p1-trace] [--p4-lexical FILE] [--read-mode enabled|no_read]";

enum Input {
    Messages(PathBuf),
    Say(String),
    Ids(Vec<u32>),
}

struct Args {
    artifact: PathBuf,
    input: Input,
    tokenizer: Option<PathBuf>,
    lexical: Option<PathBuf>,
    protocol: u8,
    cap: usize,
    trace: bool,
    read: bool,
}

fn invalid(message: &str) -> IntegerError {
    IntegerError::Invalid(message.into())
}

fn input_once(input: &mut Option<Input>, new: Input) -> Result<()> {
    if input.is_some() {
        return Err(invalid("specify one P4 input"));
    }
    *input = Some(new);
    Ok(())
}

fn parse() -> Result<Option<Args>> {
    let mut parts = env::args().skip(1);
    let (mut artifact, mut input, mut tokenizer, mut lexical) = (None, None, None, None);
    let (mut cap, mut protocol, mut trace, mut read) = (128, 2, false, true);
    while let Some(flag) = parts.next() {
        if flag == "--help" || flag == "-h" {
            println!("{USAGE}");
            return Ok(None);
        }
        if flag == "--version" || flag == "-V" {
            println!("uor-chat {}", env!("CARGO_PKG_VERSION"));
            return Ok(None);
        }
        if flag == "--p1-trace" {
            trace = true;
            continue;
        }
        let value = parts
            .next()
            .ok_or_else(|| invalid("missing P4 option value"))?;
        match flag.as_str() {
            "--discrete" => artifact = Some(PathBuf::from(value)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value)),
            "--p4-lexical" => lexical = Some(PathBuf::from(value)),
            "--messages" => input_once(&mut input, Input::Messages(value.into()))?,
            "--say" => input_once(&mut input, Input::Say(value))?,
            "--prompt-ids" => {
                let ids = value
                    .split(',')
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .map(str::parse)
                    .collect::<std::result::Result<Vec<u32>, _>>()
                    .map_err(|_| invalid("invalid original-ID prompt"))?;
                input_once(&mut input, Input::Ids(ids))?;
            }
            "--max-tokens" => cap = value.parse().map_err(|_| invalid("invalid token cap"))?,
            "--protocol" => {
                protocol = value
                    .parse()
                    .map_err(|_| invalid("invalid dialogue protocol"))?
            }
            "--read-mode" | "-m" => {
                read = match value.as_str() {
                    "enabled" | "on" => true,
                    "no_read" | "noread" | "off" => false,
                    _ => return Err(invalid("invalid P4 read mode")),
                }
            }
            _ => return Err(invalid("option unavailable in the P4-only client")),
        }
    }
    let artifact = artifact.ok_or_else(|| invalid("P4 requires --discrete"))?;
    let input = input.ok_or_else(|| invalid("missing P4 input"))?;
    if (!matches!(input, Input::Ids(_)) && tokenizer.is_none())
        || (matches!(input, Input::Messages(_)) && protocol != 2)
    {
        return Err(invalid(
            "P4 text input requires a tokenizer and --messages requires protocol 2",
        ));
    }
    Ok(Some(Args {
        artifact,
        input,
        tokenizer,
        lexical,
        protocol,
        cap,
        trace,
        read,
    }))
}

fn run(args: Args) -> Result<()> {
    let start = Instant::now();
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(&args.artifact, false)?
    else {
        return Err(invalid("--discrete requires an embedded P4 deployment"));
    };
    let deployment =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha256, identity)?;
    let engine = deployment
        .p4
        .ok_or_else(|| invalid("--discrete requires an embedded P4 deployment"))?;
    let identity = deployment
        .identity
        .ok_or_else(|| invalid("missing P4 deployment identity"))?;
    let model = deployment.model;
    let mut text = None;
    let prompt = match args.input {
        Input::Ids(ids) => ids,
        input => {
            let path = args
                .tokenizer
                .as_deref()
                .ok_or_else(|| invalid("missing tokenizer"))?;
            let (tokenizer, digest) = load_discrete_tokenizer(&model, path)?;
            let protocol = DialogueProtocol::literal_roles_version(&tokenizer, args.protocol)
                .map_err(|err| IntegerError::Invalid(err.to_string()))?;
            let encoder = protocol
                .bind(&tokenizer)
                .map_err(|err| IntegerError::Invalid(err.to_string()))?;
            let tokens = match input {
                Input::Messages(path) => {
                    let mut bytes = Vec::new();
                    File::open(path)?
                        .take((MAX_MESSAGE_BYTES + 1) as u64)
                        .read_to_end(&mut bytes)?;
                    encode_discrete_messages(&bytes, &encoder, &protocol)?
                }
                Input::Say(content) => {
                    encoder
                        .encode_assistant_prefix(&[Message {
                            role: "user",
                            content: &content,
                        }])
                        .tokens
                }
                Input::Ids(_) => return Err(invalid("unexpected raw IDs in text input")),
            };
            text = Some((tokenizer, protocol, digest));
            tokens
        }
    };
    if prompt.is_empty()
        || prompt.len() > P4_MAX_CONTEXT
        || prompt.iter().any(|&id| id >= model.shape().vocabulary)
    {
        return Err(invalid(
            "P4 prompt exceeds the causal window or original vocabulary",
        ));
    }
    let lexical = args
        .lexical
        .as_deref()
        .map(|path: &Path| load_lexical(path, &engine.inner))
        .transpose()?;
    let mut session = P4Session::new();
    let ids: Vec<u16> = prompt.iter().map(|&id| id as u16).collect();
    session
        .sync_prompt(&ids)
        .map_err(|err| IntegerError::Invalid(err.to_string()))?;
    let decode = Instant::now();
    let mut generated = Vec::new();
    let mut trace = Vec::new();
    for _ in 0..args.cap {
        let (id, observation) = step(
            &mut session,
            &model,
            &engine.inner,
            args.read,
            args.trace,
            None,
            lexical.as_ref().map(|(head, _)| head),
        )?;
        generated.push(u32::from(id));
        if let Some(observation) = observation {
            trace.push(observation);
        }
        if id == 1 {
            break;
        }
        session
            .observe_generated(id)
            .map_err(|err| IntegerError::Invalid(err.to_string()))?;
    }
    if env::var_os("UOR_P1_TIMING").is_some() {
        eprintln!(
            "{}",
            json!({"schema":"uor-r4.p1-client-timing/1",
            "open_and_prefill_ns":decode.duration_since(start).as_nanos(),
            "decode_ns":decode.elapsed().as_nanos(),"tokens":generated.len(),"traced":args.trace})
        );
    }
    let mut record = json!({"schema":"uor-r4.uor-chat-discrete-parity/1",
        "artifact":args.artifact,"artifact_sha256":identity.artifact_sha256,
        "model_schema":SCHEMA_P4,"context":P4_MAX_CONTEXT,"prompt_ids":prompt,"generated_ids":generated,
        "p4_sha256":engine.artifact_sha256,"p4_state_sha256":engine.artifact_sha256,
        "p4_trunk_sha256":model.artifact_sha256(),
        "decoding":"greedy P4 S0-S6 integer runtime (UORKNC03 + S3 + S2-skip + CausalSessionCache + S4 + S6)"});
    if args.trace {
        record["p4_steps"] = trace.into();
        record["p4_components"] = engine.component_digests.clone();
    }
    if let Some((head, digest)) = lexical {
        record["decoding"] = "greedy P4 with signed native lexical ranking".into();
        record["p4_lexical_sha256"] = digest.into();
        record["p4_lexical_addressing"] = if head.is_ordered() {
            "ordered"
        } else {
            "geometry"
        }
        .into();
    }
    if let Some((tokenizer, protocol, digest)) = text {
        let visible = generated.strip_suffix(&[protocol.eos_id]);
        record["schema"] = "uor-r4.uor-chat-discrete-text/1".into();
        record["stop"] = if visible.is_some() {
            "eos"
        } else {
            "max_tokens"
        }
        .into();
        record["text"] = tokenizer.decode(visible.unwrap_or(&generated)).into();
        record["tokenizer_sha256"] = digest.into();
        record["protocol"] = protocol.schema.into();
    }
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

pub(super) fn main() {
    if let Err(error) = parse().and_then(|args| args.map(run).transpose()) {
        eprintln!("uor-chat: {error}");
        process::exit(1);
    }
}
