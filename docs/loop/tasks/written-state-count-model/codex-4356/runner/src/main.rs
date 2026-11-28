use std::{error::Error, path::Path};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}

#[cfg(feature = "cuda")]
mod conflicts;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/grade.rs"]
mod grade;
#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/input.rs"]
mod input;
#[cfg(any(feature = "cuda", test))]
mod measure;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "../../../antigravity-17/p4-qual/src/cuda.rs"]
mod p4_cuda;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod trial {
    include!("../../../codex-4354/runner/src/trial.rs");
    include!("experiment.rs");
}

fn verify_input(root: &Path, split: &Path) -> Result<input::Train> {
    let train = input::load(root)?;
    input::verify_historical_inputs(root)?;
    input::verify_split(&train, split)?;
    Ok(train)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        #[cfg(feature = "cuda")]
        [cmd] if cmd == "finite" => {
            device::finite()?;
            device::finite_election()?;
            device::finite_center()?;
            println!("finite PASS");
        }
        [cmd, root, split] if cmd == "verify-input" => {
            let train = verify_input(Path::new(root), Path::new(split))?;
            println!("verified {} TRAIN admissions", train.targets.len());
        }
        #[cfg(feature = "cuda")]
        [cmd, p4, root, split, prior, out] if cmd == "run" => trial::run_center(
            Path::new(p4),
            Path::new(root),
            Path::new(split),
            Path::new(prior),
            Path::new(out),
        )?,
        _ => {
            return Err(fail(
                "expected finite, verify-input INPUT SPLIT, or run P4 INPUT SPLIT PREVIOUS_TRIAL FRESH_OUT",
            ));
        }
    }
    Ok(())
}
