use std::{env, path::PathBuf, process::Command};

fn build() -> Result<(), Box<dyn std::error::Error>> {
    let p4 = "../../antigravity-17/p4-qual/src";
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=S4_CUDA_ARCH");
    for source in [
        "src/s4.cu".to_owned(),
        format!("{p4}/p4_kernels.cu"),
        format!("{p4}/cuda.rs"),
        "../../../../../../crates/uor-r4-discrete/src/kn_score_division.cuh".to_owned(),
    ] {
        println!("cargo:rerun-if-changed={source}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }
    let arch = env::var("S4_CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    let cubin = out.join(format!("s4_output.{arch}.cubin"));
    let ptx = out.join(format!("s4_output.{arch}.ptx"));
    for (kind, path) in [("-cubin", &cubin), ("-ptx", &ptx)] {
        let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
            .args([
                kind,
                &format!("-arch={arch}"),
                "-O3",
                "-I",
                "../../../../../../crates/uor-r4-discrete/src",
                "src/s4.cu",
                "-o",
            ])
            .arg(path)
            .status()?;
        if !status.success() {
            return Err(format!("AOT nvcc {kind} compile failed for {arch}").into());
        }
    }
    println!("cargo:rustc-env=P4_QUAL_CUBIN={}", cubin.display());
    println!("cargo:rustc-env=S4_CUBIN={}", cubin.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    build()
}
