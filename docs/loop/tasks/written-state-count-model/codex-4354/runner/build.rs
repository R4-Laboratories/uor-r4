use std::{env, fs, path::Path, path::PathBuf, process::Command};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const INTEGER: &str = "../../../../../../crates/uor-r4-integer/src/discrete_d11.rs";
const D11: &str = "../../../../../labs/m1-d11-artifact-scorer/instrument/src";
const P4: &str = "../../antigravity-17/p4-qual/src";
const DISCRETE: &str = "../../../../../../crates/uor-r4-discrete/src";

fn table(source: &str, name: &str, expected: usize) -> Result<Vec<i32>> {
    let prefix = format!("const {name}:");
    let declaration = source
        .lines()
        .skip_while(|line| !line.starts_with(&prefix))
        .take_while(|line| line.trim() != "];")
        .collect::<Vec<_>>()
        .join("\n");
    let values = declaration
        .split_once("= [")
        .ok_or_else(|| format!("missing native D11 table {name}"))?
        .1
        .split(|c: char| !c.is_ascii_digit() && c != '-')
        .filter(|item| !item.is_empty())
        .map(str::parse::<i32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.len() != expected {
        return Err(format!("wrong number of values in native D11 table {name}").into());
    }
    Ok(values)
}

fn write_tables(out: &Path) -> Result<()> {
    use std::fmt::Write as _;
    let source = fs::read_to_string(INTEGER)?;
    let mut header = String::from("#pragma once\n#include <stdint.h>\n");
    for (name, shape, width, count, signed) in [
        ("AXES", "[512]", 512, 512, false),
        ("COMPOSE", "[4096]", 4096, 4096, false),
        ("ROOTS", "[16][8]", 8, 128, true),
        ("BFS", "[256]", 256, 256, false),
        ("BRANCH", "[16][3]", 3, 48, false),
    ] {
        let values = table(&source, name, count)?;
        let ty = if signed { "int8_t" } else { "uint8_t" };
        let require_nested = width != count;
        write!(header, "D11_CONST {ty} D11_{name}{shape} = {{")?;
        for row in values.chunks_exact(width) {
            if require_nested {
                header.push('{');
            }
            for &item in row {
                if signed {
                    i8::try_from(item)?;
                } else {
                    u8::try_from(item)?;
                }
                write!(header, "{item},")?;
            }
            if require_nested {
                header.push_str("},");
            }
        }
        header.push_str("};\n");
        if name == "BFS" {
            let raw = values
                .into_iter()
                .map(u8::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            fs::write(out.join("lexical-bfs.bin"), raw)?;
        }
    }
    fs::write(out.join("d11_tables.h"), header)?;
    Ok(())
}

fn main() -> Result<()> {
    for source in [
        INTEGER.to_owned(),
        format!("{D11}/kernel_core.h"),
        format!("{P4}/cuda.rs"),
        format!("{P4}/p4_kernels.cu"),
        format!("{DISCRETE}/kn_score_division.cuh"),
        "src/retention.cu".to_owned(),
        "../../codex-4353/runner/src/lexical.cu".to_owned(),
    ] {
        println!("cargo:rerun-if-changed={source}");
    }
    for name in ["NVCC", "LEXICAL_CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }
    let arch = env::var("LEXICAL_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_owned());
    if !arch
        .strip_prefix("sm_")
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|d| d.is_ascii_digit()))
    {
        return Err("LEXICAL_CUDA_ARCH must be sm_<decimal capability>".into());
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    let nvcc = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    write_tables(&out)?;
    for (name, source, key) in [
        ("p4", format!("{P4}/p4_kernels.cu"), "P4_QUAL_CUBIN"),
        (
            "retention",
            "src/retention.cu".to_owned(),
            "RETENTION_CUBIN",
        ),
    ] {
        let cubin = out.join(format!("{name}.{arch}.cubin"));
        let ptx = out.join(format!("{name}.{arch}.ptx"));
        for (kind, path) in [("-cubin", &cubin), ("-ptx", &ptx)] {
            let status = Command::new(&nvcc)
                .args([
                    kind,
                    &format!("-arch={arch}"),
                    "-std=c++17",
                    "-O3",
                    "-I",
                    DISCRETE,
                    "-I",
                    D11,
                    "-I",
                ])
                .arg(&out)
                .arg(&source)
                .arg("-o")
                .arg(path)
                .status()?;
            if !status.success() {
                return Err(format!("AOT nvcc {kind} failed for {source} on {arch}").into());
            }
        }
        println!("cargo:rustc-env={key}={}", cubin.display());
    }
    Ok(())
}
