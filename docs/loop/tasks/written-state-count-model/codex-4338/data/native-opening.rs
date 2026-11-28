#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::json;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, ErrorKind, Write},
    path::PathBuf,
};
use uor_r4_tokenizer::{dialogue::DialogueProtocol, ByteBpeTokenizer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpeningRow {
    case_id: String,
    text: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [tokenizer, prompts] = args.as_slice() else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "usage: whole-turn-supervision TOKENIZER BLIND_PROMPT_JSONL",
        )
        .into());
    };
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "invalid tokenizer JSON"))?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let mut output = BufWriter::new(io::stdout().lock());
    for line in BufReader::new(File::open(prompts)?).lines() {
        let row: OpeningRow = serde_json::from_str(&line?)?;
        let opening = encoder
            .encode_single_user_opening(&row.text)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    format!("{}: invalid protocol-2 opening", row.case_id),
                )
            })?;
        let user = opening.user_range();
        let user_ids = &opening.tokens()[user.clone()];
        if tokenizer.decode_bytes(user_ids) != format!(" {}", row.text).as_bytes() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "{}: decoded opening differs from the blind prompt",
                    row.case_id
                ),
            )
            .into());
        }
        let mut cursor = 0;
        let byte_bounds: Vec<_> = user_ids
            .iter()
            .map(|&id| {
                let bytes = tokenizer.decode_bytes(&[id]);
                let start = cursor + bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
                cursor += bytes.len();
                [start, cursor]
            })
            .collect();
        serde_json::to_writer(
            &mut output,
            &json!({"case_id": row.case_id, "user": [user.start, user.end],
                "ids": opening.tokens(), "byte_bounds": byte_bounds}),
        )?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}
