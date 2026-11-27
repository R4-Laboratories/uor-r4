use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-changed=../../antigravity-17/p4-qual/src/p4_kernels.cu");
    println!("cargo:rerun-if-changed=../../antigravity-17/p4-qual/src/cuda.rs");
    println!("cargo:rerun-if-changed=../../../../../../crates/uor-r4-discrete/src");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    for (name, key) in [("rows", "P5_ROWS_CUBIN"), ("engine", "P5_PILOT_CUBIN")] {
        let src = format!("src/{name}.cu");
        let cubin = out_dir.join(format!("p5_{name}.sm_89.cubin"));
        println!("cargo:rerun-if-changed={src}");
        let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
            .args([
                "-cubin",
                "-arch=sm_89",
                "-O3",
                "-I",
                "../../../../../../crates/uor-r4-discrete/src",
                "-x",
                "cu",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&cubin)
            .status()
            .expect("invoke ahead-of-time nvcc on Wick");
        assert!(status.success(), "failed to build {src}");
        println!("cargo:rustc-env={key}={}", cubin.display());
        if name == "rows" {
            println!("cargo:rustc-env=P4_QUAL_CUBIN={}", cubin.display());
        }
    }
    println!("cargo:rustc-env=P4_QUAL_CAP_MAJOR=8");
    println!("cargo:rustc-env=P4_QUAL_CAP_MINOR=9");
}
