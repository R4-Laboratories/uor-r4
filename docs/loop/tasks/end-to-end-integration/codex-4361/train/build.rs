#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::{env, error::Error, fs, path::Path, process::Command};

fn pin(path: &Path, expected: &str) -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed={}", path.display());
    let actual = format!("{:x}", Sha256::digest(fs::read(path)?));
    if actual != expected {
        return Err(format!("source SHA-256 mismatch for {}: {actual}", path.display()).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=UOR_NVCC_ARCH");
    let manifest = env::current_dir()?;
    let source = manifest
        .join("../../antigravity-5/butterfly-v3/src")
        .canonicalize()?;
    let backend = source.join("cuda_backend.rs");
    let cu = source.join("butterfly_kernels_v3.cu");
    pin(
        &backend,
        "9822da970dc304fd0d13979667d515475a75b27e0e527fcbf845218500ac2ad2",
    )?;
    pin(
        &cu,
        "d703e9b224bdf005e447ca318a330eaccff299b768511a331593443fab69f939",
    )?;
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    let arch = env::var("UOR_NVCC_ARCH").unwrap_or_else(|_| "compute_89".to_owned());
    if arch != "compute_89" {
        return Err("this experimental registration binds compute_89".into());
    }
    let cubin = out.join("butterfly_kernels_v3.sm_89.cubin");
    let nvcc = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let status = Command::new(nvcc)
        .args(["-cubin", "-arch=sm_89", "-O3", "-x", "cu"])
        .arg(&cu)
        .arg("-o")
        .arg(&cubin)
        .status()?;
    if !status.success() {
        return Err("nvcc failed to build pinned sm89 source".into());
    }
    println!("cargo:rustc-env=BFY_V3_INT_CUBIN={}", cubin.display());
    fs::write(
        out.join("backend.rs"),
        format!("#[path = {:?}] mod cuda_backend;\n", backend),
    )?;
    Ok(())
}
