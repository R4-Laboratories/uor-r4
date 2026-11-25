//! Rust replacement for the Foundry Studio Python staging, provenance, and dist scripts.
//! Runs cleanly under `cargo run --release --locked --bin stamp-shell -- <subcommand>`.

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256, Sha512};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MIB: usize = 1024 * 1024;
const MONACO_VERSION: &str = "0.52.2";
const MONACO_URL: &str = "https://registry.npmjs.org/monaco-editor/-/monaco-editor-0.52.2.tgz";
const ESBUILD_VERSION: &str = "0.25.5";
const ESBUILD_URL: &str = "https://registry.npmjs.org/esbuild-wasm/-/esbuild-wasm-0.25.5.tgz";

struct AssetPolicy {
    name: &'static str,
    version: &'static str,
    url: &'static str,
    lock_rel: &'static str,
    destination: &'static str,
    profile: &'static str,
    required: &'static [&'static str],
    prefix: &'static str,
    strip: &'static str,
    selected: &'static [&'static str],
}

const EDITOR_POLICY: AssetPolicy = AssetPolicy {
    name: "monaco-editor",
    version: MONACO_VERSION,
    url: MONACO_URL,
    lock_rel: "vendor/prism/package-lock.json",
    destination: "editor",
    profile: "foundry/editor-assets/1",
    required: &[
        "LICENSE",
        "vs/loader.js",
        "vs/editor/editor.main.js",
        "vs/editor/editor.main.css",
        "vs/base/worker/workerMain.js",
        "vs/language/json/jsonWorker.js",
        "vs/language/typescript/tsWorker.js",
    ],
    prefix: "package/min/vs/",
    strip: "package/min/",
    selected: &["package/LICENSE", "package/ThirdPartyNotices.txt"],
};

const COMPILER_POLICY: AssetPolicy = AssetPolicy {
    name: "esbuild-wasm",
    version: ESBUILD_VERSION,
    url: ESBUILD_URL,
    lock_rel: "runtime/package-lock.json",
    destination: "compiler",
    profile: "foundry/compiler-assets/1",
    required: &["LICENSE.md", "esm/browser.min.js", "esbuild.wasm"],
    prefix: "",
    strip: "package/",
    selected: &["package/LICENSE.md", "package/esm/browser.min.js", "package/esbuild.wasm"],
};

#[derive(Serialize)]
struct ReceiptFileRow {
    bytes: usize,
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct PackageSource {
    integrity: String,
    name: String,
    url: String,
    version: String,
}

#[derive(Serialize)]
struct AssetReceipt {
    files: Vec<ReceiptFileRow>,
    profile: String,
    source: PackageSource,
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_file(path: &Path) -> String {
    let data = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    sha256_hex(&data)
}

fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 { TABLE[((triple >> 6) & 0x3f) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[(triple & 0x3f) as usize] as char } else { '=' });
    }
    out
}

fn const_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn copy_dir_recursive(src: &Path, dst: &Path, skip_tests_and_sw: bool) -> usize {
    if dst.exists() {
        fs::remove_dir_all(dst).unwrap_or_else(|e| panic!("remove {}: {e}", dst.display()));
    }
    fs::create_dir_all(dst).unwrap_or_else(|e| panic!("mkdir {}: {e}", dst.display()));
    let mut count = 0usize;
    let mut stack = vec![src.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|r| r.expect("dir entry"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let p = entry.path();
            let rel = p.strip_prefix(src).expect("relative");
            let srel = rel.to_string_lossy().replace('\\', "/");
            if skip_tests_and_sw
                && (srel == "sw.template.js"
                    || srel.ends_with(".test.mjs")
                    || srel.ends_with(".test.js")
                    || srel.ends_with(".test.ts"))
            {
                continue;
            }
            let target = dst.join(rel);
            let meta = fs::symlink_metadata(&p).unwrap_or_else(|e| panic!("stat {}: {e}", p.display()));
            if meta.is_dir() {
                fs::create_dir_all(&target).unwrap_or_else(|e| panic!("mkdir {}: {e}", target.display()));
                stack.push(p);
            } else if meta.is_file() {
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent).unwrap_or_else(|e| panic!("mkdir {}: {e}", parent.display()));
                }
                fs::copy(&p, &target).unwrap_or_else(|e| panic!("copy {} -> {}: {e}", p.display(), target.display()));
                count += 1;
            }
        }
    }
    count
}

fn collect_files_sorted(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    if !dir.exists() {
        return;
    }
    let mut entries: Vec<_> = fs::read_dir(dir).expect("read_dir").map(|r| r.expect("entry")).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let p = entry.path();
        let meta = fs::metadata(&p).expect("metadata");
        if meta.is_dir() {
            collect_files_sorted(root, &p, out);
        } else if meta.is_file() {
            out.push(p);
        }
    }
}

fn valid_archive_path(name: &str) -> bool {
    if name.is_empty() || name.len() > 1024 {
        return false;
    }
    if !(name == "package" || name.starts_with("package/")) {
        return false;
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'@' | b'+' | b' ' | b'/' | b'-'))
    {
        return false;
    }
    !name.split('/').any(|part| part.is_empty() || part == "." || part == "..")
}

fn parse_tar_octal(field: &[u8]) -> Option<usize> {
    if !field.is_empty() && (field[0] & 0x80) != 0 {
        let mut val = 0u64;
        for &b in &field[1..] {
            val = (val << 8) | (b as u64);
        }
        return usize::try_from(val).ok();
    }
    let trimmed = std::str::from_utf8(field)
        .ok()?
        .trim_matches(|c: char| c == '\0' || c == ' ');
    if trimmed.is_empty() {
        return Some(0);
    }
    usize::from_str_radix(trimmed, 8).ok()
}

fn stage_locked_asset(root: &Path, policy: &AssetPolicy) -> usize {
    let lock_path = root.join(policy.lock_rel);
    let lock_text = fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", lock_path.display()));
    let lock_json: Value = serde_json::from_str(&lock_text).expect("invalid package-lock JSON");
    let pkg_key = format!("node_modules/{}", policy.name);
    let row = lock_json["packages"][&pkg_key]
        .as_object()
        .unwrap_or_else(|| panic!("missing lock entry {pkg_key}"));
    let version = row.get("version").and_then(Value::as_str).unwrap_or("");
    let resolved = row.get("resolved").and_then(Value::as_str).unwrap_or("");
    let integrity = row.get("integrity").and_then(Value::as_str).unwrap_or("");
    let is_dev = row.get("dev").and_then(Value::as_bool).unwrap_or(false);
    assert!(
        version == policy.version
            && resolved == policy.url
            && !is_dev
            && integrity.starts_with("sha512-")
            && integrity.len() == 7 + 88
            && integrity.ends_with("=="),
        "invalid locked package metadata for {}",
        policy.name
    );

    let tmp_dir = std::env::temp_dir().join(format!("foundry-pkg-{}-{}", policy.name, std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let out = Command::new("npm")
        .args([
            "pack",
            "--offline",
            "--ignore-scripts",
            "--json",
            "--pack-destination",
            &tmp_dir.to_string_lossy(),
            policy.url,
        ])
        .current_dir(root)
        .output()
        .expect("run npm pack --offline");
    assert!(
        out.status.success(),
        "npm pack --offline {} failed: {}",
        policy.name,
        String::from_utf8_lossy(&out.stderr)
    );
    let tgz_path = tmp_dir.join(format!("{}-{}.tgz", policy.name, policy.version));
    let compressed = fs::read(&tgz_path).expect("read packed tgz");
    assert!(compressed.len() <= 32 * MIB, "asset tgz too large");
    let mut sha512 = Sha512::new();
    sha512.update(&compressed);
    let actual_integrity = format!("sha512-{}", base64_encode(&sha512.finalize()));
    assert!(
        const_time_eq(actual_integrity.as_bytes(), integrity.as_bytes()),
        "package integrity mismatch for {}: expected {}, got {}",
        policy.name,
        integrity,
        actual_integrity
    );

    let mut gzip_child = Command::new("gzip")
        .arg("-dc")
        .arg(&tgz_path)
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn gzip -dc");
    let mut expanded = Vec::new();
    gzip_child
        .stdout
        .take()
        .expect("gzip stdout")
        .take((256 * MIB + 1) as u64)
        .read_to_end(&mut expanded)
        .expect("read decompressed tar");
    let gz_status = gzip_child.wait().expect("wait gzip");
    assert!(gz_status.success(), "gzip -dc failed");
    assert!(expanded.len() <= 256 * MIB, "expanded archive exceeds 256 MiB");
    let _ = fs::remove_dir_all(&tmp_dir);

    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut total = 0usize;
    let mut pkg_json: Option<Value> = None;
    let mut pax_path: Option<String> = None;
    let mut gnu_longname: Option<String> = None;
    let mut offset = 0usize;
    let mut member_idx = 0usize;

    while offset + 512 <= expanded.len() {
        let hdr = &expanded[offset..offset + 512];
        offset += 512;
        if hdr.iter().all(|&b| b == 0) {
            break;
        }
        member_idx += 1;
        assert!(member_idx <= 16384, "too many archive entries");
        let typeflag = hdr[156];
        let size = parse_tar_octal(&hdr[124..136]).expect("tar size");
        assert!(size <= 32 * MIB, "tar entry exceeds 32 MiB");
        let data_end = offset.checked_add(size).expect("tar offset overflow");
        assert!(data_end <= expanded.len(), "truncated tar entry");
        let data = &expanded[offset..data_end];
        let padded = (size + 511) & !511;
        offset = offset.checked_add(padded).expect("tar pad overflow");

        if typeflag == b'L' {
            let s = std::str::from_utf8(data).expect("utf8 gnu longname").trim_end_matches('\0');
            gnu_longname = Some(s.to_string());
            continue;
        }
        if typeflag == b'x' {
            let text = std::str::from_utf8(data).unwrap_or("");
            for line in text.lines() {
                if let Some(rest) = line.split_once(' ').map(|x| x.1) {
                    if let Some(p) = rest.strip_prefix("path=") {
                        pax_path = Some(p.to_string());
                    }
                }
            }
            continue;
        }
        if typeflag == b'g' {
            continue;
        }

        let raw_name = if let Some(p) = pax_path.take() {
            gnu_longname.take();
            p
        } else if let Some(p) = gnu_longname.take() {
            p
        } else {
            let name_part = std::str::from_utf8(&hdr[0..100]).expect("tar name").trim_end_matches('\0');
            let prefix_part = std::str::from_utf8(&hdr[345..500]).expect("tar prefix").trim_end_matches('\0');
            if prefix_part.is_empty() {
                name_part.to_string()
            } else {
                format!("{prefix_part}/{name_part}")
            }
        };

        let is_dir = typeflag == b'5' || raw_name.ends_with('/');
        let name = if is_dir {
            raw_name.trim_end_matches('/').to_string()
        } else {
            raw_name
        };
        assert!(valid_archive_path(&name), "invalid archive member name: {name}");
        assert!(seen.insert(name.clone()), "duplicate archive member: {name}");
        if is_dir {
            continue;
        }
        assert!(typeflag == b'0' || typeflag == 0, "unsupported archive entry type for {name}");

        let selected = (!policy.prefix.is_empty() && name.starts_with(policy.prefix))
            || policy.selected.contains(&name.as_str());
        if selected || name == "package/package.json" {
            if name == "package/package.json" {
                assert!(data.len() <= 65536, "package.json too large");
                pkg_json = Some(serde_json::from_slice(data).expect("parse package.json"));
            } else {
                let target = if let Some(rest) = name.strip_prefix(policy.strip) {
                    rest.to_string()
                } else {
                    name.strip_prefix("package/").unwrap_or(&name).to_string()
                };
                total += data.len();
                assert!(files.len() < 4096 && total <= 64 * MIB, "extracted asset limit exceeded");
                files.insert(target, data.to_vec());
            }
        }
    }

    let pkg_meta = pkg_json.expect("missing package/package.json");
    assert_eq!(pkg_meta.get("name").and_then(Value::as_str), Some(policy.name));
    assert_eq!(pkg_meta.get("version").and_then(Value::as_str), Some(policy.version));
    for req in policy.required {
        let body = files.get(*req).unwrap_or_else(|| panic!("missing required asset {req}"));
        assert!(!body.is_empty(), "empty required asset {req}");
    }

    let rows: Vec<ReceiptFileRow> = files
        .iter()
        .map(|(p, body)| ReceiptFileRow {
            bytes: body.len(),
            path: p.clone(),
            sha256: sha256_hex(body),
        })
        .collect();
    let receipt = AssetReceipt {
        files: rows,
        profile: policy.profile.to_string(),
        source: PackageSource {
            integrity: integrity.to_string(),
            name: policy.name.to_string(),
            url: policy.url.to_string(),
            version: policy.version.to_string(),
        },
    };
    let receipt_bytes = serde_json::to_string(&receipt).expect("serialize receipt") + "\n";

    let dst = root.join("shell").join(policy.destination);
    if dst.exists() {
        fs::remove_dir_all(&dst).expect("clean asset dest");
    }
    fs::create_dir_all(&dst).expect("mkdir asset dest");
    for (rel, body) in &files {
        let target = dst.join(rel);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).expect("mkdir parent");
        }
        fs::write(&target, body).expect("write asset file");
    }
    fs::write(dst.join("package.json"), receipt_bytes.as_bytes()).expect("write asset receipt");
    receipt.files.len()
}

fn cmd_stamp(studio_root: &Path) {
    let prism_dist_assets = studio_root.join("vendor/prism/dist/assets");
    assert!(
        prism_dist_assets.is_dir(),
        "{} is missing: run Vite build first",
        prism_dist_assets.display()
    );
    let assets = studio_root.join("shell/assets");
    let asset_count = copy_dir_recursive(&prism_dist_assets, &assets, false);
    let html_to_image = studio_root.join("vendor/prism/node_modules/html-to-image/dist/html-to-image.js");
    assert!(
        html_to_image.is_file(),
        "{} is missing: run npm ci in vendor/prism first",
        html_to_image.display()
    );
    fs::copy(&html_to_image, assets.join("html-to-image.js")).expect("copy html-to-image.js");

    let stale_app = studio_root.join("shell/app");
    if stale_app.exists() {
        fs::remove_dir_all(&stale_app).expect("remove stale shell/app");
    }
    let index_html = fs::read_to_string(studio_root.join("shell/index.html")).expect("read shell/index.html");
    fs::write(studio_root.join("shell/404.html"), index_html.as_bytes()).expect("write shell/404.html");
    println!("placed {} renderer files under shell/assets and wrote shell/404.html", asset_count);

    let src = studio_root.join("vendor/prism/scaffold");
    let dst = studio_root.join("shell/scaffold");
    if dst.exists() {
        fs::remove_dir_all(&dst).expect("clean shell/scaffold");
    }
    copy_dir_recursive(&src.join("src"), &dst.join("src"), false);
    for name in ["package.json", "tailwind.config.ts", "index.html", "AI_RULES.md"] {
        let p = src.join(name);
        if p.exists() {
            fs::copy(&p, dst.join(name)).expect("copy scaffold root file");
        }
    }

    let mut all_scaffold = Vec::new();
    collect_files_sorted(&dst, &dst, &mut all_scaffold);
    for p in all_scaffold {
        if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
            if matches!(ext, "tsx" | "ts" | "md" | "html" | "json") {
                let text = fs::read_to_string(&p).expect("read scaffold text");
                let new_text = text.replace("https://www.dyad.sh/", "https://uor-foundation.github.io/foundry/");
                if new_text != text {
                    fs::write(&p, new_text.replace("\r\n", "\n")).expect("rewrite scaffold url");
                }
            }
        }
    }

    let rules_path = dst.join("AI_RULES.md");
    let rules_old = fs::read_to_string(&rules_path).expect("read AI_RULES.md");
    let brand_note = "\n\n## Brand\n\n- The app wears the Hologram brand kit: its colours, fonts and radius are set for you. Use the shadcn/ui components and the theme classes (`bg-background`, `text-foreground`, `bg-primary`, `text-muted-foreground`, `border-border`, `bg-card`, `text-destructive`); never raw palette colours (`bg-blue-500`), hex values, or a new `:root` theme.\n";
    fs::write(&rules_path, format!("{}{}", rules_old.trim_end_matches('\n'), brand_note))
        .expect("write AI_RULES.md");

    let app_path = dst.join("src/App.tsx");
    let app_src = fs::read_to_string(&app_path).expect("read scaffold App.tsx");
    fs::write(
        &app_path,
        app_src.replacen("<BrowserRouter>", "<BrowserRouter basename={import.meta.env.BASE_URL}>", 1),
    )
    .expect("write scaffold App.tsx");

    let main_tsx = dst.join("src/main.tsx");
    if main_tsx.exists() {
        let compat_main = concat!(
            "import type { ComponentType } from \"react\";\n",
            "import { createRoot } from \"react-dom/client\";\n",
            "import * as AppMod from \"./App.tsx\";\n",
            "import \"./globals.css\";\n\n",
            "const RootApp = (AppMod as { default?: ComponentType; App?: ComponentType }).default ?? (AppMod as { App?: ComponentType }).App;\n",
            "if (!RootApp) throw new Error(\"src/App.tsx must export a default or App React component\");\n",
            "createRoot(document.getElementById(\"root\")!).render(<RootApp />);\n",
        );
        fs::write(&main_tsx, compat_main).expect("write scaffold main.tsx");
    }

    let mut scaffold_src_files = Vec::new();
    collect_files_sorted(&dst, &dst.join("src"), &mut scaffold_src_files);
    let mut rels: Vec<String> = scaffold_src_files
        .into_iter()
        .map(|p| p.strip_prefix(&dst).expect("rel").to_string_lossy().replace('\\', "/"))
        .collect();
    rels.sort();
    fs::write(dst.join("files.json"), serde_json::to_string(&rels).expect("files.json"))
        .expect("write files.json");
    println!("staged {} scaffold files", rels.len());

    let ed_count = stage_locked_asset(studio_root, &EDITOR_POLICY);
    println!("staged {ed_count} locked editor assets and licenses");
    let comp_count = stage_locked_asset(studio_root, &COMPILER_POLICY);
    println!("staged {comp_count} locked compiler assets and licenses");

    let lex_src = studio_root.parent().expect("foundry root").join("app/src/InteractionModel.lex.tex");
    fs::copy(&lex_src, studio_root.join("shell/interaction-model.lex.tex")).expect("copy InteractionModel.lex.tex");

    for args in [&["tools/host_bundle.mjs"][..], &["tools/host_bundle.mjs", "--check"][..]] {
        let st = Command::new("node")
            .args(args)
            .current_dir(studio_root)
            .status()
            .expect("run host_bundle.mjs");
        assert!(st.success(), "node {} failed", args.join(" "));
    }
}

fn canonical_json(value: Value) -> Vec<u8> {
    fn sort_val(v: Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut sorted = BTreeMap::new();
                for (k, val) in map {
                    sorted.insert(k, sort_val(val));
                }
                let mut out = Map::new();
                for (k, val) in sorted {
                    out.insert(k, val);
                }
                Value::Object(out)
            }
            Value::Array(arr) => Value::Array(arr.into_iter().map(sort_val).collect()),
            other => other,
        }
    }
    serde_json::to_string(&sort_val(value)).expect("canonical JSON").into_bytes()
}

fn append_tar_entry(tar_out: &mut Vec<u8>, name: &str, data: &[u8]) {
    let mut hdr = [0u8; 512];
    let name_bytes = name.as_bytes();
    assert!(name_bytes.len() <= 100, "tar path > 100 chars not expected here: {name}");
    hdr[..name_bytes.len()].copy_from_slice(name_bytes);
    hdr[100..108].copy_from_slice(b"0000644\0");
    hdr[108..116].copy_from_slice(b"0000000\0");
    hdr[116..124].copy_from_slice(b"0000000\0");
    let size_str = format!("{:011o}\0", data.len());
    hdr[124..136].copy_from_slice(size_str.as_bytes());
    hdr[136..148].copy_from_slice(b"00000000000\0");
    hdr[148..156].copy_from_slice(b"        ");
    hdr[156] = b'0';
    hdr[257..263].copy_from_slice(b"ustar\0");
    hdr[263..265].copy_from_slice(b"00");
    let cksum: u32 = hdr.iter().map(|&b| b as u32).sum();
    let cksum_str = format!("{:06o}\0 ", cksum);
    hdr[148..156].copy_from_slice(cksum_str.as_bytes());
    tar_out.extend_from_slice(&hdr);
    tar_out.extend_from_slice(data);
    let rem = data.len() % 512;
    if rem != 0 {
        tar_out.extend_from_slice(&vec![0u8; 512 - rem]);
    }
}

fn cmd_provenance(foundry_root: &Path, shell_dir: &Path, studio_root: &Path) {
    let studio = foundry_root.join("studio");
    let upstream_text = fs::read_to_string(studio.join("generated/upstream-provenance.json"))
        .expect("read upstream-provenance.json");
    let upstream: Value = serde_json::from_str(&upstream_text).expect("parse upstream-provenance.json");

    let checks = [
        ("kernel_ir_sha256", studio.join("generated/kernel.ir")),
        ("coverage_sha256", studio.join("generated/coverage.json")),
        ("roots_sha256", studio.join("generated/roots.json")),
        ("generated_core_sha256", studio.join("generated/hologram_forge_core.rs")),
    ];
    let mut actual = BTreeMap::new();
    for (field, path) in checks {
        let digest = sha256_file(&path);
        let expected = upstream[field].as_str().expect("upstream hash string");
        assert!(
            digest == expected,
            "provenance mismatch for {field}: expected {expected}, got {digest}"
        );
        actual.insert(field, digest);
    }

    let mut lcnf_obj = Map::new();
    lcnf_obj.insert("coverage_sha256".into(), Value::String(actual["coverage_sha256"].clone()));
    lcnf_obj.insert("kernel_ir_sha256".into(), Value::String(actual["kernel_ir_sha256"].clone()));
    lcnf_obj.insert("roots_sha256".into(), Value::String(actual["roots_sha256"].clone()));
    lcnf_obj.insert("schema".into(), Value::String("prismpm/lcnf-manifest/1".into()));
    let lcnf_sha = sha256_hex(&canonical_json(Value::Object(lcnf_obj)));
    assert_eq!(
        Some(lcnf_sha.as_str()),
        upstream["lcnf_manifest_sha256"].as_str(),
        "lcnf_manifest_sha256 mismatch"
    );

    let mut target_obj = Map::new();
    target_obj.insert("contract".into(), Value::String("hologram:guest/core-wasm@1".into()));
    target_obj.insert("export".into(), Value::String("holo_run".into()));
    target_obj.insert("input_allocation_cap".into(), Value::Number(268435456.into()));
    target_obj.insert("maximum_pages".into(), Value::Number(4096.into()));
    target_obj.insert("response_maximum".into(), Value::Number(268435456.into()));
    target_obj.insert("schema".into(), Value::String("lean4-prod/core-wasm-target/1".into()));
    let target_profile_id = sha256_hex(&canonical_json(Value::Object(target_obj)));
    assert_eq!(
        Some(target_profile_id.as_str()),
        upstream["target_profile_id"].as_str(),
        "target_profile_id mismatch"
    );

    let cargo_toml = fs::read_to_string(studio.join("core/Cargo.toml")).expect("read Cargo.toml");
    let cargo_name = cargo_toml
        .lines()
        .find_map(|l| l.strip_prefix("name = "))
        .map(|v| v.trim_matches('"').to_string())
        .expect("package name");
    let cargo_version = cargo_toml
        .lines()
        .find_map(|l| l.strip_prefix("version = "))
        .map(|v| v.trim_matches('"').to_string())
        .expect("package version");

    let mut members = vec![
        studio.join("core/Cargo.toml"),
        studio.join("core/Cargo.lock"),
        studio.join("generated/hologram_forge_core.rs"),
    ];
    collect_files_sorted(&studio, &studio.join("core/src"), &mut members);
    members.sort();
    let mut tar_bytes = Vec::new();
    for m in members {
        let rel = m.strip_prefix(&studio).expect("rel").to_string_lossy().replace('\\', "/");
        let raw = fs::read(&m).expect("read crate member");
        let norm: Vec<u8> = raw
            .windows(2)
            .enumerate()
            .filter_map(|(idx, w)| if w == b"\r\n" { None } else { Some(raw[idx]) })
            .chain(raw.last().copied())
            .collect();
        append_tar_entry(&mut tar_bytes, &rel, &norm);
    }
    tar_bytes.extend_from_slice(&[0u8; 1024]);
    let mut gz = Command::new("gzip")
        .args(["-9", "-n", "-c"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn gzip");
    let (gz_out, write_res) = std::thread::scope(|s| {
        let mut gz_stdin = gz.stdin.take().expect("gzip stdin");
        let tar_ref = &tar_bytes;
        let writer = s.spawn(move || gz_stdin.write_all(tar_ref));
        let output = gz.wait_with_output().expect("gzip output");
        let res = writer.join().expect("gzip stdin writer thread panicked");
        (output, res)
    });
    assert!(
        gz_out.status.success(),
        "gzip crate tar failed ({}): {}",
        gz_out.status,
        String::from_utf8_lossy(&gz_out.stderr)
    );
    write_res.expect("write tar to gzip");
    let cargo_crate_sha256 = sha256_hex(&gz_out.stdout);

    let manifest_text = fs::read_to_string(shell_dir.join("manifest.json")).expect("read manifest.json");
    let closure = serde_json::from_str::<Value>(&manifest_text).expect("parse manifest")["closure"]
        .as_str()
        .expect("closure")
        .to_string();
    let lock_text = fs::read_to_string(studio_root.join("foundry-assets.lock.json")).expect("read assets lock");
    let upstream_commit = serde_json::from_str::<Value>(&lock_text).expect("parse assets lock")["upstream_commit"]
        .as_str()
        .expect("upstream_commit")
        .to_string();

    let prov = serde_json::json!({
        "schema": "hologram-forge/provenance/1",
        "source_repo": "UOR-Foundation/uor-r4",
        "foundry_upstream_commit": upstream_commit,
        "attestation_id": upstream["attestation_id"],
        "build_id": upstream["build_id"],
        "source_id": upstream["source_id"],
        "semantic_id": upstream["semantic_id"],
        "compiler_semantics_id": upstream["compiler_semantics_id"],
        "snapshot_id": upstream["snapshot_id"],
        "stdlib_semantics_id": upstream["stdlib_semantics_id"],
        "prism_stdlib_crate_sha256": upstream["prism_stdlib_crate_sha256"],
        "lexlean_commit": upstream["lexlean_commit"],
        "lexlean_package_sha256": upstream["lexlean_package_sha256"],
        "lean4_prod_commit": upstream["lean4_prod_commit"],
        "hologram_live_commit": upstream["hologram_live_commit"],
        "uor_hologram_commit": upstream["uor_hologram_commit"],
        "target_profile_id": target_profile_id,
        "lean_manifest_sha256": upstream["lean_manifest_sha256"],
        "lcnf_manifest_sha256": lcnf_sha,
        "kernel_ir_sha256": actual["kernel_ir_sha256"],
        "coverage_sha256": actual["coverage_sha256"],
        "roots_sha256": actual["roots_sha256"],
        "generated_core_sha256": actual["generated_core_sha256"],
        "cargo_name": cargo_name,
        "cargo_version": cargo_version,
        "cargo_crate_sha256": cargo_crate_sha256,
        "cargo_crate_kind": "deterministic tar of Cargo.toml, Cargo.lock, src/** and generated/hologram_forge_core.rs (cargo package refuses the git dependency)",
        "view_model_id": upstream["view_model_id"],
        "core_wasm_sha256": sha256_file(&shell_dir.join("core.wasm")),
        "browser_projection_sha256": closure,
        "prismpm_commit": upstream["prismpm_commit"],
    });
    fs::write(
        shell_dir.join("provenance.json"),
        serde_json::to_string_pretty(&prov).expect("pretty prov") + "\n",
    )
    .expect("write provenance.json");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default_studio = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("studio dir")
        .to_path_buf();
    match args.first().map(String::as_str) {
        Some("stamp") => {
            let root = args.get(1).map(PathBuf::from).unwrap_or(default_studio);
            cmd_stamp(&root);
        }
        Some("provenance") => {
            assert!(args.len() >= 4, "usage: stamp-shell provenance <foundry_root> <shell_dir> <studio_root>");
            cmd_provenance(Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]));
        }
        Some("stage-dist") => {
            assert!(args.len() >= 3, "usage: stamp-shell stage-dist <shell_dir> <dist_dir>");
            let copied = copy_dir_recursive(Path::new(&args[1]), Path::new(&args[2]), true);
            println!("staged {copied} clean bundle files into {}", args[2]);
        }
        _ => panic!("usage: stamp-shell <stamp|provenance|stage-dist> ..."),
    }
}
