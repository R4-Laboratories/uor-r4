use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/lookup_v3.cuh");
    println!("cargo:rerun-if-changed=../../../../../../crates/uor-r4-discrete/src/kn_score_division.cuh");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let div_inc = manifest_dir
        .join("../../../../../../crates/uor-r4-discrete/src")
        .canonicalize()
        .expect("canonicalize division include");
    let cu = manifest_dir.join("src/lookup_v3.cuh");
    let cubin = out_dir.join("lookup_v3.sm_89.cubin");

    let status = Command::new("nvcc")
        .arg("-cubin")
        .arg("-arch=sm_89")
        .arg("-O3")
        .arg("-x")
        .arg("cu")
        .arg("-I")
        .arg(&div_inc)
        .arg(&cu)
        .arg("-o")
        .arg(&cubin)
        .status()
        .expect("invoke nvcc");
    assert!(status.success(), "nvcc failed building lookup_v3.sm_89.cubin");
    println!("cargo:rustc-env=V3_BATCH_CUBIN={}", cubin.display());
}
