//! Rejoin both held tapes to their upstream CoQA sources and check canonical encoding.

use std::{
    collections::{HashMap, HashSet},
    env, fs, io,
    path::Path,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::{
    ByteBpeTokenizer,
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Hash = [u8; 32];

const DEV_SHA256: &str = "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a";
const TRAIN_SHA256: &str = "b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6";
const TOKENIZER_SHA256: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

#[derive(Deserialize)]
struct Dataset {
    data: Vec<Conversation>,
}

#[derive(Deserialize)]
struct Conversation {
    id: String,
    source: String,
    filename: String,
    story: String,
    questions: Vec<Turn>,
    answers: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    turn_id: usize,
    input_text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Blind {
    id: String,
    messages: Vec<RawMessage>,
    prefix: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Truth {
    id: String,
    source: Source,
    story: String,
    questions: Vec<String>,
    answers: Vec<String>,
    gold: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    dataset: String,
    split: String,
    source: String,
    id: String,
    filename: String,
    passage_sha256: String,
    normalized_passage_sha256: String,
    target_turn: usize,
}

struct Pool {
    split: &'static str,
    domain: &'static str,
}

const DEV: Pool = Pool {
    split: "dev",
    domain: "uor-r4/4352/coqa",
};
const SUPPLEMENT: Pool = Pool {
    split: "train",
    domain: "uor-r4/4352/coqa-train-as-held",
};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn ensure(test: bool, message: &'static str) -> Result<()> {
    if !test {
        return Err(invalid(message).into());
    }
    Ok(())
}

fn pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    ensure(
        hex::encode(Sha256::digest(&bytes)) == sha,
        "source SHA-256 differs",
    )?;
    Ok(bytes)
}

fn dataset(path: &Path, sha: &str) -> Result<Dataset> {
    serde_json::from_slice(&pinned(path, sha)?)
        .map_err(|_| invalid("invalid CoQA source structure").into())
}

fn tape<T: DeserializeOwned>(root: &Path, name: &str) -> Result<(Vec<T>, String)> {
    let bytes = fs::read(root.join(name))?;
    let text = std::str::from_utf8(&bytes).map_err(|_| invalid("non-UTF-8 tape"))?;
    ensure(
        text.is_empty() || text.ends_with('\n'),
        "missing tape line terminator",
    )?;
    let rows = text
        .lines()
        .map(|line| serde_json::from_str(line).map_err(|_| invalid("invalid tape row schema")))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((rows, hex::encode(Sha256::digest(&bytes))))
}

fn story_hash(story: &str) -> Hash {
    let spaced: String = story
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    Sha256::digest(spaced.split_whitespace().collect::<Vec<_>>().join(" ")).into()
}

fn envelope_hash(pool: &Pool, purpose: &str, value: &str) -> Hash {
    let mut hash = Sha256::new();
    hash.update(pool.domain);
    hash.update(b"/");
    hash.update(purpose);
    hash.update(b"/v1\0");
    hash.update(value);
    hash.finalize().into()
}

#[derive(Default)]
struct Footprint<'a> {
    ids: HashSet<&'a str>,
    files: HashSet<(&'a str, &'a str)>,
    stories: HashSet<Hash>,
}

impl<'a> Footprint<'a> {
    fn add(&mut self, doc: &'a Conversation, story: Hash) {
        self.ids.insert(&doc.id);
        self.files.insert((&doc.source, &doc.filename));
        self.stories.insert(story);
    }

    fn overlaps(&self, doc: &Conversation, story: Hash) -> bool {
        self.ids.contains(doc.id.as_str())
            || self
                .files
                .contains(&(doc.source.as_str(), doc.filename.as_str()))
            || self.stories.contains(&story)
    }
}

fn original_messages<'a>(doc: &'a Conversation, first: &'a str, turn: usize) -> Vec<Message<'a>> {
    let mut out = Vec::with_capacity(2 * turn);
    for i in 0..turn {
        out.push(Message {
            role: "user",
            content: if i == 0 {
                first
            } else {
                &doc.questions[i].input_text
            },
        });
        out.push(Message {
            role: "assistant",
            content: &doc.answers[i].input_text,
        });
    }
    out
}

fn verify_row(
    blind: &Blind,
    truth: &Truth,
    original: &Conversation,
    pool: &Pool,
    protocol: &DialogueProtocol,
    encoder: &DialogueEncoder<'_>,
) -> Result<()> {
    let turn = truth.source.target_turn;
    ensure(
        (2..=5).contains(&turn)
            && original.questions.len() == original.answers.len()
            && original.questions.len() >= turn,
        "bad source turn count",
    )?;
    ensure(
        truth.source.dataset == "CoQA v1.0"
            && truth.source.split == pool.split
            && truth.source.source == original.source
            && truth.source.filename == original.filename
            && matches!(original.source.as_str(), "cnn" | "gutenberg" | "wikipedia"),
        "source attribution differs",
    )?;
    ensure(
        truth.story == original.story
            && truth.source.passage_sha256
                == hex::encode(Sha256::digest(original.story.as_bytes()))
            && truth.source.normalized_passage_sha256 == hex::encode(story_hash(&original.story)),
        "complete source passage differs",
    )?;
    ensure(
        truth.questions.len() == turn && truth.answers.len() == turn,
        "tape role history length differs",
    )?;
    for i in 0..turn {
        ensure(
            original.questions[i].turn_id == i + 1
                && original.answers[i].turn_id == i + 1
                && original.questions[i].input_text == truth.questions[i]
                && original.answers[i].input_text == truth.answers[i],
            "original QA or order differs",
        )?;
    }

    let first = format!("{}\n\n{}", original.story, original.questions[0].input_text);
    let complete = original_messages(original, &first, turn);
    let pending = &complete[..complete.len() - 1];
    ensure(
        blind.messages.len() == pending.len()
            && blind
                .messages
                .iter()
                .zip(pending)
                .all(|(a, b)| a.role == b.role && a.content == b.content),
        "blind history differs from original causal roles",
    )?;
    let prefix = encoder.encode_assistant_prefix(pending);
    let document = encoder.encode_document(&complete);
    let fixed_vocab = |&id: &u32| id < 4096 && id != protocol.unk_id;
    ensure(
        prefix.special_token_occurrences == 0
            && document.special_token_occurrences == 0
            && prefix.emitted_turns == pending.len()
            && document.emitted_turns == complete.len()
            && prefix.tokens.len() <= 2048
            && prefix.tokens.first() == Some(&protocol.bos_id)
            && prefix.tokens.iter().all(fixed_vocab)
            && document.tokens.iter().all(fixed_vocab),
        "noncanonical encoding or token budget",
    )?;
    ensure(
        blind.prefix == prefix.tokens,
        "blind prefix differs from canonical encoding",
    )?;
    ensure(
        document.tokens.strip_prefix(prefix.tokens.as_slice()) == Some(truth.gold.as_slice())
            && truth.gold.len() > 1
            && truth.gold.last() == Some(&protocol.eos_id)
            && truth.gold[..truth.gold.len() - 1]
                .iter()
                .all(|&id| id != protocol.bos_id && id != protocol.eos_id),
        "canonical reference through EOS differs",
    )?;
    let earlier = encoder.encode_open_history(&pending[..pending.len() - 1]);
    let current = encoder.encode_user_prefix(
        pending
            .last()
            .ok_or_else(|| invalid("missing current user"))?
            .content,
        true,
    );
    ensure(
        prefix.tokens.strip_prefix(earlier.tokens.as_slice()) == Some(current.tokens.as_slice())
            && earlier
                .tokens
                .iter()
                .filter(|&&id| id == protocol.eos_id)
                .count()
                == turn - 1
            && !current.tokens.contains(&protocol.eos_id),
        "closed-history or pending-user composition differs",
    )?;
    for end in 1..turn {
        let closed = encoder.encode_open_history(&pending[..2 * end]);
        ensure(
            closed.tokens.last() == Some(&protocol.eos_id)
                && prefix.tokens.starts_with(&closed.tokens)
                && closed
                    .tokens
                    .iter()
                    .filter(|&&id| id == protocol.eos_id)
                    .count()
                    == end,
            "assistant turn closure differs",
        )?;
    }
    Ok(())
}

#[derive(Serialize)]
struct Receipt {
    rows: usize,
    blind_sha256: String,
    truth_sha256: String,
}

#[derive(Default)]
struct Selected<'a> {
    source: Footprint<'a>,
    opaque_ids: HashSet<String>,
    prefix_reference: HashSet<(Vec<u32>, Vec<u32>)>,
}

fn verify_pool<'a>(
    pool: &Pool,
    folder: &Path,
    originals: &'a Dataset,
    all_dev: Option<&Footprint<'_>>,
    selected: &mut Selected<'a>,
    protocol: &DialogueProtocol,
    encoder: &DialogueEncoder<'_>,
) -> Result<Receipt> {
    let index: HashMap<_, _> = originals
        .data
        .iter()
        .map(|doc| (doc.id.as_str(), doc))
        .collect();
    ensure(
        index.len() == originals.data.len(),
        "duplicate original CoQA ID",
    )?;
    let (blind, blind_sha256) = tape::<Blind>(folder, "blind.jsonl")?;
    let (truth, truth_sha256) = tape::<Truth>(folder, "truth.jsonl")?;
    let rows = blind.len();
    ensure(
        rows == truth.len() && rows <= 200,
        "tape length mismatch or cap exceeded",
    )?;
    let mut prior_rank = None;
    for (ordinal, (blind, truth)) in blind.into_iter().zip(truth).enumerate() {
        let expected_id = hex::encode(&envelope_hash(pool, "blind", &ordinal.to_string())[..12]);
        ensure(
            blind.id == expected_id
                && truth.id == expected_id
                && selected.opaque_ids.insert(expected_id),
            "blind/truth opaque identity or order differs",
        )?;
        let doc = index
            .get(truth.source.id.as_str())
            .ok_or_else(|| invalid("selected source missing from pinned split"))?;
        let rank = (envelope_hash(pool, "source", &doc.id), doc.id.as_str());
        ensure(
            prior_rank.is_none_or(|previous| previous < rank),
            "source selection order differs",
        )?;
        prior_rank = Some(rank);
        let story = story_hash(&doc.story);
        ensure(
            !selected.source.overlaps(doc, story)
                && all_dev.is_none_or(|deny| !deny.overlaps(doc, story)),
            "source ID, filename or complete passage overlap",
        )?;
        verify_row(&blind, &truth, doc, pool, protocol, encoder)?;
        ensure(
            selected.prefix_reference.insert((blind.prefix, truth.gold)),
            "duplicate prefix/reference pair",
        )?;
        selected.source.add(doc, story);
    }
    Ok(Receipt {
        rows,
        blind_sha256,
        truth_sha256,
    })
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [
        dev,
        supplemental,
        tokenizer,
        dev_output,
        supplemental_output,
    ] = args.as_slice()
    else {
        return Err(invalid(
            "usage: verify_pools DEV_SOURCE TRAIN_SOURCE TOKENIZER DEV_OUTPUT TRAIN_AS_HELD_OUTPUT",
        )
        .into());
    };
    let dev = dataset(Path::new(dev), DEV_SHA256)?;
    let supplemental = dataset(Path::new(supplemental), TRAIN_SHA256)?;
    let bytes = pinned(Path::new(tokenizer), TOKENIZER_SHA256)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes)
        .ok_or_else(|| invalid("invalid tokenizer"))?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let encoder = protocol.bind(&tokenizer)?;
    let mut all_dev = Footprint::default();
    for doc in &dev.data {
        all_dev.add(doc, story_hash(&doc.story));
    }
    ensure(
        dev.data.len() == 500 && all_dev.ids.len() == 500,
        "incomplete whole-DEV exclusion roster",
    )?;
    let mut selected = Selected::default();
    let primary = verify_pool(
        &DEV,
        Path::new(dev_output),
        &dev,
        None,
        &mut selected,
        &protocol,
        &encoder,
    )?;
    let transfer = verify_pool(
        &SUPPLEMENT,
        Path::new(supplemental_output),
        &supplemental,
        Some(&all_dev),
        &mut selected,
        &protocol,
        &encoder,
    )?;
    println!(
        "{}",
        serde_json::json!({
            "schema": "uor-r4/4352/independent-held-verification/v1",
            "dev": primary,
            "train_as_held": transfer,
            "supplemental_exclusion_dev_ids": all_dev.ids.len(),
            "supplemental_exclusion_dev_files": all_dev.files.len(),
            "supplemental_exclusion_dev_passages": all_dev.stories.len(),
            "unique_opaque_ids": selected.opaque_ids.len(),
            "unique_prefix_reference_pairs": selected.prefix_reference.len(),
            "status": "PASS"
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("verify_pools: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truth_injection_into_a_valid_blind_row() {
        let valid =
            r#"{"id":"fixture","messages":[{"role":"user","content":"Question"}],"prefix":[0,5]}"#;
        assert!(serde_json::from_str::<Blind>(valid).is_ok());
        let tampered = valid.replace("\"prefix\"", "\"gold\":[6,1],\"prefix\"");
        assert!(serde_json::from_str::<Blind>(&tampered).is_err());
    }
}
