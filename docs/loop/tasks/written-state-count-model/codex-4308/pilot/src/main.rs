use std::error::Error;
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}
#[cfg(feature = "cuda")]
mod engine;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "../../../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
pub mod lawful_likelihood;
#[cfg(feature = "cuda")]
mod measure;
#[cfg(feature = "cuda")]
mod replay;
#[cfg(feature = "cuda")]
mod rows;
mod sample;
#[cfg(feature = "cuda")]
mod trial;
#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "finite" {
        engine::P5Gpu::new(0)?.check_finite_parity()?;
        println!("finite GPU parity passed");
        return Ok(());
    }
    if args.len() != 7 || args[0] != "run" {
        return Err(fail(
            "usage: p5-wick-pilot finite | run P4 TRUNK R DEV_TUNE UOR_CHAT NEW_OUT",
        ));
    }
    trial::run(&args[1..])
}
#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    Err(fail("this measurement requires Wick CUDA"))
}
