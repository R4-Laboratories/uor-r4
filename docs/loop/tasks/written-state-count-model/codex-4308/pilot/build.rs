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
    let mut kernels = vec![
        ("rows", "P5_ROWS_CUBIN", "src/rows.cu"),
        ("engine", "P5_PILOT_CUBIN", "src/engine.cu"),
    ];
    if env::var_os("CARGO_FEATURE_ALIAS_TRAINING").is_some() {
        kernels.push((
            "alias_adjust",
            "P5_ALIAS_ADJUST_CUBIN",
            "../../codex-4309/alias_adjust.cu",
        ));
    }
    for (name, key, src) in kernels {
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
            .arg(src)
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
