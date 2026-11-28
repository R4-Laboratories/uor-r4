use std::{error::Error, fs, path::Path};
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}
#[cfg(feature = "cuda")]
mod device;
mod grade;
mod input;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "../../../antigravity-17/p4-qual/src/cuda.rs"]
mod p4_cuda;
#[cfg(feature = "cuda")]
mod trial;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice(){
        [cmd,root,split] if cmd == "seal-split" => {
            let train=input::load(Path::new(root))?;
            input::verify_historical_inputs(Path::new(root))?;
            println!("{}",input::seal_split(&train,Path::new(split))?);
        }
        [cmd,root,split] if cmd == "verify-input" => {
            let train=input::load(Path::new(root))?;
            input::verify_historical_inputs(Path::new(root))?;
            input::verify_split(&train,Path::new(split))?;
            use sha2::{Digest,Sha256};
            println!("{} rows; split {}",train.targets.len(),hex::encode(Sha256::digest(fs::read(split)?)));
        }
        #[cfg(feature="cuda")]
        [cmd] if cmd=="finite" => {device::finite()?;println!("finite PASS");}
        #[cfg(feature="cuda")]
        [cmd,p4,root,split,out] if cmd=="run" => trial::run(Path::new(p4),Path::new(root),Path::new(split),Path::new(out))?,
        _ => return Err(fail("expected seal-split INPUT NEW_FILE, verify-input INPUT SPLIT, finite, or run P4 INPUT SPLIT NEW_OUTPUT (device commands require cuda)")),
    }
    Ok(())
}
