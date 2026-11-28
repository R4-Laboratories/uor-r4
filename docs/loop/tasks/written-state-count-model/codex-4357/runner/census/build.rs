use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    for name in ["MIDPOINT_CUBIN", "P4_QUAL_CUBIN"] {
        println!("cargo:rerun-if-env-changed={name}");
        let path = PathBuf::from(env::var_os(name).ok_or("missing audited cubin path")?);
        if !path.is_absolute() || !path.is_file() {
            return Err(format!("{name} must be an absolute path to an audited cubin").into());
        }
        let path = path.display();
        println!("cargo:rerun-if-changed={path}");
        println!("cargo:rustc-env={name}={path}");
    }
    Ok(())
}
