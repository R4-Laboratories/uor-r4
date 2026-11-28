//! Cold text serving through the lossless V3 reference transition.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;
use uor_r4_integer::ordinary_p4::{encode_discrete_messages, MAX_MESSAGE_BYTES};
use uor_r4_native_vector::butterfly_rotor_v3::ButterflyV3Model;
use uor_r4_tokenizer::{dialogue::DialogueProtocol, ByteBpeTokenizer};

const BOS: u16 = 0;
const EOS: u16 = 1;
const MAX_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROMPT_TOKENS: usize = 32_768;
const MAX_OUTPUT_TOKENS: usize = 1_024;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    checkpoint_sha256: String,
    tokenizer_sha256: String,
    dialogue_protocol: u32,
    vocab_size: usize,
    bos_id: u16,
    eos_id: u16,
}

impl Manifest {
    fn parse(bytes: &[u8], trusted_sha256: &str) -> Result<Self> {
        verify_hash(bytes, trusted_sha256, "manifest")?;
        let manifest: Self = serde_json::from_slice(bytes)?;
        if manifest.schema != "uor-r4.butterfly-v3-chat/1"
            || manifest.dialogue_protocol != 2
            || !(2..=4096).contains(&manifest.vocab_size)
            || manifest.bos_id != BOS
            || manifest.eos_id != EOS
        {
            return Err("unsupported Butterfly V3 chat manifest".into());
        }
        Ok(manifest)
    }
}

fn verify_hash(bytes: &[u8], pin: &str, label: &str) -> Result<()> {
    let actual = hex::encode(Sha256::digest(bytes));
    if actual != pin {
        return Err(format!("{label} SHA-256 mismatches its trusted pin").into());
    }
    Ok(())
}

fn read_bounded(path: &Path, max_bytes: usize, label: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(format!("{label} exceeds {max_bytes} bytes").into());
    }
    Ok(bytes)
}

fn bound_tokenizer(bytes: &[u8], manifest: &Manifest) -> Result<ByteBpeTokenizer> {
    verify_hash(bytes, &manifest.tokenizer_sha256, "tokenizer")?;
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(bytes).ok_or("invalid tokenizer JSON")?;
    if tokenizer.vocab_size() != manifest.vocab_size
        || tokenizer.token_id("<|bos|>") != Some(u32::from(BOS))
        || tokenizer.token_id("<|eos|>") != Some(u32::from(EOS))
    {
        return Err("tokenizer does not match manifest vocabulary or special IDs".into());
    }
    Ok(tokenizer)
}

fn prompt_ids(messages: &[u8], tokenizer: &ByteBpeTokenizer) -> Result<Vec<u16>> {
    let protocol = DialogueProtocol::literal_roles_v2(tokenizer)?;
    let encoder = protocol.bind(tokenizer)?;
    let ids = encode_discrete_messages(messages, &encoder, &protocol)?;
    if ids.len() > MAX_PROMPT_TOKENS {
        return Err("prompt exceeds the Butterfly V3 client token limit".into());
    }
    ids.into_iter()
        .map(|id| {
            if (id as usize) < tokenizer.vocab_size() {
                Ok(id as u16)
            } else {
                Err("prompt token outside the bound vocabulary".into())
            }
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
enum Stop {
    Eos,
    Cap,
    InvalidBos,
}

impl Stop {
    fn name(&self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::Cap => "cap",
            Self::InvalidBos => "invalid_bos",
        }
    }
}

fn decode_greedy(
    mut token: u16,
    cap: usize,
    mut step: impl FnMut(u16) -> Result<u16>,
) -> Result<(Vec<u32>, Stop)> {
    let mut ids = Vec::new();
    while ids.len() < cap {
        match token {
            EOS => return Ok((ids, Stop::Eos)),
            BOS => return Ok((ids, Stop::InvalidBos)),
            _ => ids.push(u32::from(token)),
        }
        if ids.len() < cap {
            token = step(token)?;
        }
    }
    Ok((ids, Stop::Cap))
}

fn run(args: &[OsString]) -> Result<Stop> {
    let [manifest_path, manifest_sha256, checkpoint_path, tokenizer_path, messages_path, cap, trace @ ..] =
        args
    else {
        return Err(
            "expected MANIFEST TRUSTED_MANIFEST_SHA256 CHECKPOINT TOKENIZER MESSAGES MAX_TOKENS"
                .into(),
        );
    };
    let trace_path = match trace {
        [] => None,
        [flag, path] if flag == "--trace" => Some(path),
        _ => return Err("expected optional --trace PATH".into()),
    };
    let cap: usize = cap.to_str().ok_or("invalid token cap")?.parse()?;
    if !(1..=MAX_OUTPUT_TOKENS).contains(&cap) {
        return Err("MAX_TOKENS must be in 1..=1024".into());
    }
    let manifest_bytes = read_bounded(Path::new(manifest_path), 4096, "manifest")?;
    let manifest = Manifest::parse(
        &manifest_bytes,
        manifest_sha256.to_str().ok_or("invalid manifest hash")?,
    )?;
    let tokenizer_bytes =
        read_bounded(Path::new(tokenizer_path), MAX_TOKENIZER_BYTES, "tokenizer")?;
    let tokenizer = bound_tokenizer(&tokenizer_bytes, &manifest)?;
    let messages = read_bounded(Path::new(messages_path), MAX_MESSAGE_BYTES, "messages")?;
    let ids = prompt_ids(&messages, &tokenizer)?;
    let checkpoint = read_bounded(Path::new(checkpoint_path), MAX_ARTIFACT_BYTES, "checkpoint")?;
    verify_hash(&checkpoint, &manifest.checkpoint_sha256, "checkpoint")?;
    let model = ButterflyV3Model::from_bytes(&checkpoint)?;
    if model.cfg.vocab != manifest.vocab_size {
        return Err("checkpoint vocabulary does not match manifest".into());
    }
    let mut session = model.new_session();
    let mut next = None;
    for &id in &ids {
        next = Some(model.step(id, &mut session)?.greedy_token);
    }
    let first = next.ok_or("empty encoded prompt")?;
    let (response_ids, stop) = decode_greedy(first, cap, |id| {
        Ok(model.step(id, &mut session)?.greedy_token)
    })?;
    if let Some(path) = trace_path {
        let mut emitted = response_ids.clone();
        match stop {
            Stop::Eos => emitted.push(u32::from(EOS)),
            Stop::InvalidBos => emitted.push(u32::from(BOS)),
            Stop::Cap => {}
        }
        let mut prefix_digest = Sha256::new();
        for id in &ids {
            prefix_digest.update(id.to_le_bytes());
        }
        let trace = serde_json::json!({
            "prefix": ids,
            "prefix_u16le_sha256": hex::encode(prefix_digest.finalize()),
            "emitted": emitted,
            "stop": stop.name()
        });
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        serde_json::to_writer(&mut file, &trace)?;
        writeln!(file)?;
    }
    io::stdout()
        .lock()
        .write_all(&tokenizer.decode_bytes(&response_ids))?;
    Ok(stop)
}

pub(super) fn usage() {
    eprintln!("Usage: uor-chat --butterfly-v3 MANIFEST TRUSTED_MANIFEST_SHA256 CHECKPOINT TOKENIZER MESSAGES MAX_TOKENS [--trace PATH]");
}

pub(super) fn try_main() -> bool {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--butterfly-v3")) {
        return false;
    }
    let args = args.collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--help" {
        usage();
        return true;
    }
    match run(&args) {
        Ok(Stop::Eos) => eprintln!("stop=eos"),
        Ok(Stop::Cap) => eprintln!("stop=cap"),
        Ok(Stop::InvalidBos) => {
            eprintln!("stop=invalid_bos");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("Butterfly V3: {error}");
            std::process::exit(1);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_tokenizer() -> Vec<u8> {
        let mut vocab = serde_json::Map::new();
        let specials = ["<|bos|>", "<|eos|>", "<|unk|>"];
        for (id, token) in specials.iter().enumerate() {
            vocab.insert((*token).into(), json!(id));
        }
        let mut spilled = 0u32;
        for byte in 0..=255u32 {
            let scalar = if (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte)
            {
                byte
            } else {
                let scalar = 256 + spilled;
                spilled += 1;
                scalar
            };
            vocab.insert(char::from_u32(scalar).unwrap().to_string(), json!(byte + 3));
        }
        json!({
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
            "added_tokens": specials.iter().enumerate().map(|(id, token)| json!({
                "id": id, "content": token
            })).collect::<Vec<_>>(),
            "model": {"type": "BPE", "vocab": vocab, "merges": []}
        })
        .to_string()
        .into_bytes()
    }

    fn fixture_manifest(tokenizer: &[u8]) -> serde_json::Value {
        json!({
            "schema": "uor-r4.butterfly-v3-chat/1",
            "checkpoint_sha256": hex::encode(Sha256::digest(b"fixture")),
            "tokenizer_sha256": hex::encode(Sha256::digest(tokenizer)),
            "dialogue_protocol": 2, "vocab_size": 259, "bos_id": 0, "eos_id": 1
        })
    }

    fn parse_fixture(value: &serde_json::Value) -> Result<Manifest> {
        let bytes = serde_json::to_vec(value)?;
        Manifest::parse(&bytes, &hex::encode(Sha256::digest(&bytes)))
    }

    #[test]
    fn manifest_and_inputs_fail_closed() -> Result<()> {
        let tokenizer = fixture_tokenizer();
        let mut manifest = fixture_manifest(&tokenizer);
        assert!(Manifest::parse(&serde_json::to_vec(&manifest)?, "different").is_err());
        assert!(Manifest::parse(b"{", &hex::encode(Sha256::digest(b"{"))).is_err());
        let bound = parse_fixture(&manifest)?;
        assert!(verify_hash(b"fixture", &bound.checkpoint_sha256, "checkpoint").is_ok());
        assert!(verify_hash(b"changed", &bound.checkpoint_sha256, "checkpoint").is_err());
        assert!(bound_tokenizer(b"changed", &bound).is_err());
        manifest["dialogue_protocol"] = json!(1);
        assert!(parse_fixture(&manifest).is_err());
        manifest["dialogue_protocol"] = json!(2);
        manifest["bos_id"] = json!(1);
        assert!(parse_fixture(&manifest).is_err());
        Ok(())
    }

    #[test]
    fn shared_chat_encoding_uses_original_ids_and_preserves_completed_eos() -> Result<()> {
        let bytes = fixture_tokenizer();
        let manifest = parse_fixture(&fixture_manifest(&bytes))?;
        let tokenizer = bound_tokenizer(&bytes, &manifest)?;
        let messages = serde_json::to_vec(&json!([
            {"role":"user","content":"Hi"},
            {"role":"assistant","content":"Hello"},
            {"role":"user","content":"é"}
        ]))?;
        let ids = prompt_ids(&messages, &tokenizer)?;
        assert_eq!(ids.first(), Some(&BOS));
        assert_eq!(ids.iter().filter(|&&id| id == EOS).count(), 1);
        assert!(ids.windows(2).any(|pair| pair == [198, 172]));
        assert_eq!(tokenizer.decode_bytes(&[198, 172]), "é".as_bytes());
        assert!(prompt_ids(b"[]", &tokenizer).is_err());
        assert!(prompt_ids(br#"[{"role":"assistant","content":"bad"}]"#, &tokenizer).is_err());
        assert!(prompt_ids(br#"[{"role":"user","content":"<|eos|>"}]"#, &tokenizer).is_err());
        Ok(())
    }

    #[test]
    fn greedy_uses_observed_tokens_and_respects_eos_and_cap() -> Result<()> {
        let (ids, stop) = decode_greedy(198, 3, |token| match token {
            198 => Ok(172),
            172 => Ok(EOS),
            _ => Err("unexpected self-fed ID".into()),
        })?;
        assert_eq!(ids, [198, 172]);
        assert_eq!(stop, Stop::Eos);
        assert_eq!(
            decode_greedy(198, 1, |_| Err("stepped past cap".into()))?,
            (vec![198], Stop::Cap)
        );
        assert_eq!(
            decode_greedy(BOS, 2, |_| Err("stepped BOS".into()))?,
            (vec![], Stop::InvalidBos)
        );
        Ok(())
    }

    #[test]
    fn codec_rejects_pinned_but_malformed_artifact() -> Result<()> {
        let manifest = parse_fixture(&fixture_manifest(&fixture_tokenizer()))?;
        verify_hash(b"fixture", &manifest.checkpoint_sha256, "checkpoint")?;
        assert!(ButterflyV3Model::from_bytes(b"fixture").is_err());
        Ok(())
    }

    #[test]
    fn cold_cli_imports_a_synthetic_checkpoint_and_stops_at_observed_eos() -> Result<()> {
        use uor_r4_native_vector::butterfly_rotor::LossKind;
        use uor_r4_native_vector::butterfly_rotor_v3::ButterflyV3Config;

        let mut model = ButterflyV3Model::new(ButterflyV3Config {
            vocab: 259,
            dim: 16,
            layers: 1,
            ffn_branches: 1,
            pkm_sub_keys: 2,
            pkm_key_dim: 4,
            loss_kind: LossKind::Sparsemax64,
            seed: 1,
        })?;
        model.embeddings.fill(0);
        model.coarse_centroids.fill(0);
        let checkpoint = model.to_bytes();
        let tokenizer = fixture_tokenizer();
        let mut manifest = fixture_manifest(&tokenizer);
        manifest["checkpoint_sha256"] = json!(hex::encode(Sha256::digest(&checkpoint)));
        let bytes = serde_json::to_vec(&manifest)?;
        let base = std::env::temp_dir().join(format!("uor-bfy03-cli-test-{}", std::process::id()));
        std::fs::create_dir(&base)?;
        let check = || -> Result<Stop> {
            let mut paths = Vec::new();
            for (file, data) in [
                ("manifest.json", bytes.as_slice()),
                ("model.uorbfy03", checkpoint.as_slice()),
                ("tokenizer.json", tokenizer.as_slice()),
                (
                    "messages.json",
                    br#"[{"role":"user","content":"Hi"}]"#.as_slice(),
                ),
            ] {
                let path = base.join(file);
                std::fs::write(&path, data)?;
                paths.push(path.into_os_string());
            }
            paths.insert(1, hex::encode(Sha256::digest(&bytes)).into());
            paths.push("2".into());
            let trace_path = base.join("trace.json");
            paths.push("--trace".into());
            paths.push(trace_path.clone().into_os_string());
            let stop = run(&paths)?;
            let trace: serde_json::Value = serde_json::from_slice(&std::fs::read(trace_path)?)?;
            assert_eq!(trace["prefix"][0], BOS);
            assert_eq!(trace["emitted"], json!([EOS]));
            assert_eq!(trace["stop"], "eos");
            assert_eq!(
                trace["prefix_u16le_sha256"].as_str().map(str::len),
                Some(64)
            );
            Ok(stop)
        };
        let result = check();
        std::fs::remove_dir_all(&base)?;
        assert_eq!(result?, Stop::Eos);
        Ok(())
    }
}
