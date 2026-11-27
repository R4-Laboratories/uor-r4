mod plan;

#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
mod training;

use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [cmd, donor, tokenizer, out] if cmd == "prepare" => {
            plan::prepare(Path::new(donor), Path::new(tokenizer), Path::new(out))
        }
        #[cfg(feature = "cuda")]
        [cmd, plan, go, sha, out] if cmd == "fit" => {
            training::fit(Path::new(plan), Path::new(go), sha.to_str().ok_or("non-UTF8 GO SHA")?, Path::new(out))
        }
        #[cfg(feature = "cuda")]
        [cmd, out] if cmd == "finite" => training::finite(Path::new(out)),
        _ => Err("usage: visible-answer-train prepare DONOR_DIR TOKENIZER PLAN_DIR | finite OUT_DIR | fit PLAN.json GO.json GO_SHA OUT_DIR (finite/fit require CUDA)".into()),
    }
}
