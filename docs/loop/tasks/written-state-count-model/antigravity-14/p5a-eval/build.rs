#![forbid(unsafe_code)]

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/p5a_s4_kernels.cu");
    println!("cargo:rerun-if-env-changed=UOR_NVCC_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let arch = env::var("UOR_NVCC_ARCH").unwrap_or_else(|_| "compute_89".to_owned());
    let digits = arch
        .strip_prefix("compute_")
        .filter(|d| d.len() >= 2 && d.bytes().all(|b| b.is_ascii_digit()))
        .expect(
            "UOR_NVCC_ARCH must be compute_<decimal capability> (e.g. compute_89 or compute_120)",
        );
    let (cap_major, cap_minor) = digits.split_at(digits.len() - 1);

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let cu = manifest_dir.join("src/p5a_s4_kernels.cu");
    let cubin = out_dir.join(format!("p5a_s4_kernels.sm_{digits}.cubin"));

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let status = Command::new(nvcc)
        .arg("-cubin")
        .arg(format!("-arch=sm_{digits}"))
        .arg("-O3")
        .arg("-x")
        .arg("cu")
        .arg(&cu)
        .arg("-o")
        .arg(&cubin)
        .status()
        .expect("invoke nvcc");
    assert!(status.success(), "nvcc failed building {}", cubin.display());
    println!("cargo:rustc-env=P5A_EVAL_CUBIN={}", cubin.display());
    println!("cargo:rustc-env=P5A_EVAL_CAP_MAJOR={cap_major}");
    println!("cargo:rustc-env=P5A_EVAL_CAP_MINOR={cap_minor}");
}
