use std::{
    env,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::{
    native_occurrence::OpeningAssessment, native_opening, native_whole_turn, IntegerDiscreteModel,
};
use uor_r4_tokenizer::{dialogue::DialogueProtocol, ByteBpeTokenizer};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Request {
    case_id: Value,
    #[serde(alias = "prompt")]
    text: String,
}

#[derive(Serialize)]
struct Reply {
    case_id: Value,
    old_framed: OpeningAssessment,
    expanded_frozen: OpeningAssessment,
    expanded_fitted: OpeningAssessment,
}

fn main() -> CliResult<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [base, frozen, fitted, tokenizer, requests] = args.as_slice() else {
        return Err("usage: verify-whole-turn-occurrence BALANCED.d11 FROZEN.d11 FITTED.d11 TOKENIZER.json INPUT.jsonl".into());
    };
    let load = |p: &Path| -> CliResult<_> {
        Ok(IntegerDiscreteModel::from_reader(BufReader::new(
            File::open(p)?,
        ))?)
    };
    let base = load(Path::new(base))?;
    let frozen = load(Path::new(frozen))?;
    let fitted = load(Path::new(fitted))?;
    if native_whole_turn::balanced_words(&base)? != native_whole_turn::expanded_words(&frozen)? {
        return Err("expanded frozen words differ from pinned framed words".into());
    }
    native_whole_turn::expanded_words(&fitted)?;

    let tokenizer = fs::read(tokenizer)?;
    if hex::encode(Sha256::digest(&tokenizer)) != native_opening::CANONICAL_TOKENIZER_SHA256 {
        return Err("whole-turn verification requires the pinned tokenizer".into());
    }
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&tokenizer).ok_or("invalid tokenizer JSON")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let encoder = protocol.bind(&tokenizer)?;
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    for line in BufReader::new(File::open(requests)?).lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let opening = encoder
            .encode_single_user_opening(&request.text)
            .ok_or("request must encode a plain protocol-2 user opening")?;
        serde_json::to_writer(
            &mut writer,
            &Reply {
                case_id: request.case_id,
                old_framed: native_whole_turn::assess(&base, &opening)?,
                expanded_frozen: native_whole_turn::assess(&frozen, &opening)?,
                expanded_fitted: native_whole_turn::assess(&fitted, &opening)?,
            },
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
