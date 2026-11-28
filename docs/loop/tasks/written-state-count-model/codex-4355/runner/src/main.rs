use std::error::Error;
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail(message: impl Into<String>) -> Box<dyn Error> { message.into().into() }

#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/input.rs"]
mod input;
#[allow(dead_code)]
#[path = "../../../codex-4354/runner/src/grade.rs"]
mod grade;
mod measure;
#[cfg(feature = "cuda")]
mod device;
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
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        #[cfg(feature="cuda")]
        [cmd] if cmd=="finite" => { device::finite()?; device::finite_election()?; println!("finite PASS"); }
        [cmd, input_root, split] if cmd=="verify-input" => {
            let root=Path::new(input_root);
            let rows=input::load(root)?;
            input::verify_historical_inputs(root)?;
            input::verify_split(&rows,Path::new(split))?;
            println!("verified {} TRAIN admissions",rows.targets.len());
        }
        #[cfg(feature="cuda")]
        [cmd,p4,root,split,replay,out] if cmd=="run" =>
            trial::run_election(Path::new(p4),Path::new(root),Path::new(split),Path::new(replay),Path::new(out))?,
        _ => return Err(fail("expected finite, verify-input INPUT SPLIT, or run P4 INPUT SPLIT HISTORICAL_TRAJECTORIES NEW_OUTPUT")),
    }
    Ok(())
}
