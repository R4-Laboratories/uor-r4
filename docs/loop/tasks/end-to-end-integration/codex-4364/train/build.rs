#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::{env, error::Error, fs, path::Path, process::Command};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn pin(path: &Path, expected: &str) -> Result<()> {
    println!("cargo:rerun-if-changed={}", path.display());
    let hash = format!("{:x}", Sha256::digest(fs::read(path)?));
    if hash != expected {
        return Err(format!("source SHA-256 mismatch for {}: {hash}", path.display()).into());
    }
    Ok(())
}

fn main() -> Result<()> {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    let manifest = env::current_dir()?;
    let original = manifest.join("../../antigravity-5/butterfly-v3/src");
    let patch = manifest.join("opening.patch");
    let helper = manifest.join("opening_margin.cuh");
    println!("cargo:rerun-if-changed={}", patch.display());
    println!("cargo:rerun-if-changed={}", helper.display());
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    for (file, sha) in [
        (
            "cuda_backend.rs",
            "9822da970dc304fd0d13979667d515475a75b27e0e527fcbf845218500ac2ad2",
        ),
        (
            "butterfly_kernels_v3.cu",
            "d703e9b224bdf005e447ca318a330eaccff299b768511a331593443fab69f939",
        ),
    ] {
        let source = original.join(file);
        pin(&source, sha)?;
        fs::copy(source, out.join(file))?;
    }
    fs::copy(helper, out.join("opening_margin.cuh"))?;
    let status = Command::new("patch")
        .current_dir(&out)
        .args(["-f", "-p1", "-i"])
        .arg(patch)
        .status()?;
    if !status.success() {
        return Err("source-pinned opening patch did not apply".into());
    }
    let cuda = out.join("butterfly_kernels_v3.cu");
    let cubin = out.join("checker.sm_89.cubin");
    let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args(["-cubin", "-arch=sm_89", "-O3"])
        .arg(cuda)
        .arg("-o")
        .arg(&cubin)
        .status()?;
    if !status.success() {
        return Err("nvcc failed to compile sm_89 checker".into());
    }
    println!("cargo:rustc-env=BFY_V3_INT_CUBIN={}", cubin.display());
    fs::write(
        out.join("opening-backend.rs"),
        format!(
            "#[path = {:?}] mod cuda_backend;\n",
            out.join("cuda_backend.rs")
        ),
    )?;
    Ok(())
}
