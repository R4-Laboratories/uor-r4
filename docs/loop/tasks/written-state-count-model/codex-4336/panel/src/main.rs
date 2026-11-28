#![forbid(unsafe_code)]

use span_panel::{capture, freeze, grade, tokenizer, Result};
use std::path::Path;

fn main() -> Result<()> {
    let argv = std::env::args().collect::<Vec<_>>();
    let args = argv.iter().skip(1).map(String::as_str).collect::<Vec<_>>();
    match args.as_slice() {
        ["freeze", tok, author, panel] => {
            freeze(&tokenizer(Path::new(tok))?, Path::new(author), Path::new(panel))
        }
        ["capture", tok, panel, old, base, new, artifact, out, group] => {
            let inputs = capture::Inputs {
                tokenizer: Path::new(tok), panel: Path::new(panel), old: Path::new(old),
                base: Path::new(base), new: Path::new(new), artifact: Path::new(artifact),
            };
            capture::capture(&tokenizer(inputs.tokenizer)?, &inputs, Path::new(out), group)
        }
        ["run", tok, panel, old, base, new, artifact, out] => {
            let inputs = capture::Inputs {
                tokenizer: Path::new(tok), panel: Path::new(panel), old: Path::new(old),
                base: Path::new(base), new: Path::new(new), artifact: Path::new(artifact),
            };
            let tokenizer = tokenizer(inputs.tokenizer)?;
            let out = Path::new(out);
            capture::capture(&tokenizer, &inputs, out, "all")?;
            grade::grade(&tokenizer, inputs.panel, out, &[&out.join("records.jsonl")])
        }
        ["grade", tok, panel, out, records @ ..] if !records.is_empty() => {
            let records = records.iter().map(Path::new).collect::<Vec<_>>();
            grade::grade(&tokenizer(Path::new(tok))?, Path::new(panel), Path::new(out), &records)
        }
        _ => Err(concat!(
            "usage: span-panel freeze TOKENIZER AUTHOR NEW_PANEL\n",
            "       span-panel capture TOKENIZER PANEL OLD_CHAT BASE_CHAT NEW_CHAT ARTIFACT NEW_CAPTURE GROUP\n",
            "       span-panel run TOKENIZER PANEL OLD_CHAT BASE_CHAT NEW_CHAT ARTIFACT NEW_CAPTURE\n",
            "       span-panel grade TOKENIZER PANEL OUTPUT RECORDS [RECORDS ...]"
        ).into()),
    }
}
