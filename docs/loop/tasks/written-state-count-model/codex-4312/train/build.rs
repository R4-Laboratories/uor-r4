use std::{env, fs, path::PathBuf, process::Command};

const SOURCE: &str = "../../../../../../crates/uor-r4-integer/src/discrete_d11.rs";
const SCORER: &str = "../../../../../labs/m1-d11-artifact-scorer/instrument/src";
const ENTRIES: [&str; 2] = ["visible_count_rows", "visible_unigram_best"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for file in [SOURCE, "src/fit.cu", &format!("{SCORER}/kernel_core.h")] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rerun-if-env-changed=UOR_NVCC_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUOBJDUMP");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }

    let arch = env::var("UOR_NVCC_ARCH")?;
    let capability = arch
        .strip_prefix("compute_")
        .filter(|digits| !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit()))
        .ok_or("UOR_NVCC_ARCH must be compute_<decimal capability>")?;
    let image_target = format!("sm_{capability}");
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR missing")?);
    let mut header = String::from("#include <stdint.h>\n");
    let source = fs::read_to_string(SOURCE)?;
    for (name, ty, dims, count, row) in [
        ("AXES", "uint8_t", "[512]", 512, 0),
        ("COMPOSE", "uint8_t", "[4096]", 4096, 0),
        ("ROOTS", "int8_t", "[16][8]", 128, 8),
        ("BFS", "uint8_t", "[256]", 256, 0),
        ("BRANCH", "uint8_t", "[16][3]", 48, 3),
    ] {
        let prefix = format!("const {name}:");
        let definition = source
            .lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| source.split_once(line).map(|(_, rest)| (line, rest)))
            .ok_or_else(|| format!("D11 geometry is missing {name}"))?;
        let value = if let Some((_, body)) = definition.0.split_once("= [") {
            format!("{body}{}", definition.1)
        } else {
            return Err(format!("{name} has no array literal").into());
        };
        let words = value
            .split_once("];")
            .ok_or("D11 array terminator missing")?
            .0
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .filter(|word| !word.is_empty())
            .map(str::parse::<i16>)
            .collect::<Result<Vec<_>, _>>()?;
        if words.len() != count
            || words.iter().any(|value| {
                if ty == "int8_t" {
                    i8::try_from(*value).is_err()
                } else {
                    u8::try_from(*value).is_err()
                }
            })
        {
            return Err(format!("D11 geometry shape or storage differs for {name}").into());
        }
        let fragments = if row == 0 {
            words.iter().map(ToString::to_string).collect::<Vec<_>>()
        } else {
            words
                .chunks_exact(row)
                .map(|row| {
                    format!(
                        "{{{}}}",
                        row.iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect()
        };
        header.push_str(&format!(
            "D11_CONST {ty} D11_{name}{dims} = {{{}}};\n",
            fragments.join(",")
        ));
    }
    let reciprocals = std::iter::repeat_n(String::from("0ULL"), 2)
        .chain((2..=243).map(|divisor| format!("{}ULL", (1u128 << 64) / divisor)))
        .collect::<Vec<_>>();
    header.push_str(&format!(
        "D11_CONST uint64_t D11_SHUFFLE_RECIPROCAL[244] = {{{}}};\n",
        reciprocals.join(",")
    ));
    fs::write(out.join("d11_tables.h"), header)?;

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let ptx = out.join("visible_fit.ptx");
    if !Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "-std=c++17",
            &format!("-arch={arch}"),
            "-I",
            SCORER,
            "-I",
        ])
        .arg(&out)
        .args(["src/fit.cu", "-o"])
        .arg(&ptx)
        .status()?
        .success()
    {
        return Err("ahead-of-time CUDA count build failed".into());
    }
    admit_integer_ptx(&fs::read_to_string(&ptx)?)?;

    let cubin = out.join("visible_fit.cubin");
    if !Command::new(nvcc)
        .args([
            "-cubin",
            &format!("-arch={arch}"),
            &format!("--gpu-code={image_target}"),
            "-o",
        ])
        .arg(&cubin)
        .arg(ptx)
        .status()?
        .success()
    {
        return Err("ahead-of-time count cubin assembly failed".into());
    }
    if !fs::read(&cubin)?.starts_with(b"\x7fELF") {
        return Err("count build emitted no ELF cubin".into());
    }
    let disassembler = env::var("CUOBJDUMP").unwrap_or_else(|_| "cuobjdump".into());
    let sass = Command::new(disassembler)
        .arg("--dump-sass")
        .arg(&cubin)
        .output()?;
    if !sass.status.success() {
        return Err("count cubin SASS disassembly failed".into());
    }
    fs::write(out.join("visible_fit.sass"), &sass.stdout)?;
    let sass = String::from_utf8(sass.stdout)?;
    admit_integer_sass(&sass)?;
    println!("cargo:rustc-env=VISIBLE_FIT_CUBIN_TARGET={image_target}");
    Ok(())
}

fn admit_integer_sass(text: &str) -> Result<(), String> {
    const FORBIDDEN: &[&str] = &[
        "FADD", "FMUL", "FFMA", "FMA", "FSET", "FSETP", "FSEL", "FCHK", "FMNMX", "FRND", "F2I",
        "I2F", "F2F", "DADD", "DMUL", "DFMA", "DSET", "DSETP", "DMNMX", "HADD2", "HMUL2", "HFMA2",
        "HSET2", "HSETP2", "HMNMX2", "MUFU", "RRO", "HMMA", "IMMA", "BMMA", "DMMA", "QMMA", "GMMA",
        "WMMA",
    ];
    for line in text.lines() {
        let Some((_, instruction)) = line.split_once("*/") else {
            continue;
        };
        if let Some(word) = instruction
            .split_whitespace()
            .find(|term| !term.starts_with('@'))
        {
            if FORBIDDEN.iter().any(|opcode| word.starts_with(opcode)) {
                return Err(format!("noninteger SASS operation {word}"));
            }
        }
    }
    Ok(())
}

fn admit_integer_ptx(text: &str) -> Result<(), String> {
    let text = text
        .lines()
        .map(|line| line.split("//").next().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    let mut entries = Vec::new();
    let mut expect_name = false;
    for word in text
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_'))
        .filter(|word| !word.is_empty())
    {
        if word.split('.').any(|part| {
            matches!(
                part,
                "f16"
                    | "f16x2"
                    | "f32"
                    | "f64"
                    | "bf16"
                    | "bf16x2"
                    | "tf32"
                    | "e4m3"
                    | "e5m2"
                    | "mma"
                    | "wmma"
                    | "wgmma"
                    | "tensor"
                    | "tensormap"
                    | "tcgen05"
            )
        }) {
            return Err(format!("noninteger PTX operation {word}"));
        }
        if expect_name {
            entries.push(word);
            expect_name = false;
        } else if word == ".entry" {
            expect_name = true;
        }
    }
    entries.sort_unstable();
    if expect_name || entries != ENTRIES {
        return Err(format!("count CUDA entry inventory differs: {entries:?}"));
    }
    Ok(())
}
