use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}

#[cfg(feature = "cuda")]
#[path = "../../../codex-4308/pilot/src/sample.rs"]
mod sample;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "../../../antigravity-17/p4-qual/src/cuda.rs"]
mod p4_cuda;

#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
mod trial;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    trial::run(std::env::args_os().skip(1).collect())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    Err(fail("build with --features cuda on the leased GPU host"))
}
