use std::error::Error;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "../../../antigravity-17/p4-qual/src/cuda.rs"]
mod p4_cuda;
#[cfg(feature = "cuda")]
mod trial;
#[cfg(feature = "cuda")]
pub use trial::{Arm, Engines};
#[cfg(feature = "cuda")]
mod held;
#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    trial::run()
}
#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    Err(fail("build with --features cuda for the registered device"))
}
