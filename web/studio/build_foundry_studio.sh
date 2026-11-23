#!/usr/bin/env bash
set -euo pipefail
# Build the full Foundry Web Studio + real UOR-R4 integer WASM + RV64IMAC MVM Micro-Linux WASM
# and emit a self-contained static bundle in web/studio/dist/.

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
studio_root="$repo_root/web/studio"
foundry_root="$studio_root/foundry"
foundry_studio="$foundry_root/studio"
shell_dir="$foundry_studio/shell"
dist_dir="$studio_root/dist"

export PATH="$HOME/.cargo/bin:$PATH"

fetch_args=()
if [ -d /tmp/foundry-probe/studio ]; then
    fetch_args+=(--seed-dir /tmp/foundry-probe)
fi
echo "== 1/9 fetch locked Foundry binary assets =="
node "$repo_root/scripts/studio/fetch-foundry-assets.mjs" ${fetch_args[@]+"${fetch_args[@]}"}

echo "== 2/9 build UOR-R4 integer serving WASM + synthetic-stack.lut + tokenizer.json =="
bash "$studio_root/build_serve_wasm.sh" --no-verify

echo "== 3/9 build UOR-R4 RV64IMAC MVM WASM =="
bash "$studio_root/build_mvm_wasm.sh"

echo "== 4/9 stage UOR-R4 + MVM runtime packages into Foundry shell =="
rm -rf "$shell_dir/pkg" "$shell_dir/pkg-mvm" "$shell_dir/uor"
mkdir -p "$shell_dir/pkg" "$shell_dir/pkg-mvm" "$shell_dir/uor/js"
cp "$studio_root/pkg/uor_r4_integer.js" "$studio_root/pkg/uor_r4_integer_bg.wasm" "$studio_root/pkg/synthetic-stack.lut" "$studio_root/pkg/tokenizer.json" "$shell_dir/pkg/"
cp "$studio_root/pkg-mvm/uor_r4_mvm.js" "$studio_root/pkg-mvm/uor_r4_mvm_bg.wasm" "$shell_dir/pkg-mvm/"
cp "$studio_root/r4_worker.js" "$studio_root/coi-serviceworker.js" "$shell_dir/"
cp "$studio_root/assets/js/uor_model_worker.js" "$studio_root/assets/js/uor_tokenizer_artifact.js" "$shell_dir/uor/js/"

rm -rf "$shell_dir/models"
node -e "import('$repo_root/scripts/studio/catalog.mjs').then(m => m.writeStudioCatalog({ dest: '$shell_dir' }))"
node "$repo_root/scripts/studio/build-standards-index.mjs" "$shell_dir/standards-index.json"

echo "== 5/9 build Foundry core.wasm (hologram-forge-core) =="
cargo build --release --locked --manifest-path "$foundry_studio/core/Cargo.toml" --target wasm32-unknown-unknown
target_base="${CARGO_TARGET_DIR:-$foundry_studio/core/target}"
cp "$target_base/wasm32-unknown-unknown/release/hologram_forge_core.wasm" "$shell_dir/core.wasm"

echo "== 6/9 install Foundry JS deps (ignoring native Electron/sqlite scripts) & build Vite renderer =="
npm ci --prefix "$foundry_studio/runtime" --ignore-scripts --no-audit --no-fund
if [ ! -d "$foundry_studio/vendor/prism/node_modules" ] || [ ! -f "$foundry_studio/vendor/prism/node_modules/.package-lock.json" ]; then
    (
        cd "$foundry_studio/vendor/prism"
        ELECTRON_SKIP_BINARY_DOWNLOAD=1 npm ci --prefer-offline --ignore-scripts --engine-strict=false --force --no-audit --no-fund --loglevel=error
    )
else
    # Ensure npm's offline tarball store holds monaco-editor for stamp_shell.py.
    npm cache add monaco-editor@0.52.2 --prefer-offline >/dev/null 2>&1 || true
fi
(
    cd "$foundry_studio/vendor/prism"
    npx vite build --config vite.shell.config.mts --base ./ --outDir dist --emptyOutDir
)

echo "== 7/9 stamp Foundry shell assets, Monaco editor, esbuild-wasm compiler, and host.bundle.js =="
python3 "$foundry_studio/scripts/stamp_shell.py"

echo "== 8/9 project Foundry shell index.html, holo.html, manifest.json & seal closure =="
cargo run --release --locked --manifest-path "$foundry_studio/core/Cargo.toml" --bin project-shell
cp "$shell_dir/index.html" "$shell_dir/404.html"

python3 - "$foundry_root" "$shell_dir" "$studio_root" <<'PY'
import gzip, hashlib, io, json, pathlib, re, sys, tarfile

foundry = pathlib.Path(sys.argv[1])
studio = foundry / "studio"
shell = pathlib.Path(sys.argv[2])
studio_root = pathlib.Path(sys.argv[3])

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha_path(p):
    return sha_bytes(pathlib.Path(p).read_bytes())

def canonical(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")

upstream = json.loads((studio / "generated" / "upstream-provenance.json").read_text(encoding="utf-8"))
gen_checks = {
    "kernel_ir_sha256": studio / "generated" / "kernel.ir",
    "coverage_sha256": studio / "generated" / "coverage.json",
    "roots_sha256": studio / "generated" / "roots.json",
    "generated_core_sha256": studio / "generated" / "hologram_forge_core.rs",
}
actual = {}
for field, path in gen_checks.items():
    digest = sha_path(path)
    actual[field] = digest
    if digest != upstream[field]:
        sys.exit(f"provenance mismatch for {field}: expected {upstream[field]}, got {digest}")

lcnf_manifest = canonical({
    "coverage_sha256": actual["coverage_sha256"],
    "kernel_ir_sha256": actual["kernel_ir_sha256"],
    "roots_sha256": actual["roots_sha256"],
    "schema": "prismpm/lcnf-manifest/1",
})
lcnf_manifest_sha256 = sha_bytes(lcnf_manifest)
if lcnf_manifest_sha256 != upstream["lcnf_manifest_sha256"]:
    sys.exit(f"lcnf_manifest_sha256 mismatch: expected {upstream['lcnf_manifest_sha256']}, got {lcnf_manifest_sha256}")

target_profile = canonical({
    "contract": "hologram:guest/core-wasm@1",
    "export": "holo_run",
    "input_allocation_cap": 268435456,
    "maximum_pages": 4096,
    "response_maximum": 268435456,
    "schema": "lean4-prod/core-wasm-target/1",
})
target_profile_id = sha_bytes(target_profile)
if target_profile_id != upstream["target_profile_id"]:
    sys.exit(f"target_profile_id mismatch: expected {upstream['target_profile_id']}, got {target_profile_id}")

pkg = re.search(
    r'\[package\]\s*\nname = "([^"]+)"\s*\nversion = "([^"]+)"',
    (studio / "core" / "Cargo.toml").read_text(encoding="utf-8"),
)
cargo_name, cargo_version = pkg.group(1), pkg.group(2)

members = sorted(
    [studio / "core" / "Cargo.toml", studio / "core" / "Cargo.lock", studio / "generated" / "hologram_forge_core.rs"]
    + [p for p in (studio / "core" / "src").rglob("*") if p.is_file()]
)
tar_buf = io.BytesIO()
with gzip.GzipFile(fileobj=tar_buf, mode="wb", compresslevel=9, mtime=0) as gz:
    with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for path in members:
            info = tarfile.TarInfo(str(path.relative_to(studio)).replace("\\", "/"))
            data = path.read_bytes().replace(b"\r\n", b"\n")
            info.size, info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = len(data), 0, 0, 0, "", "", 0o644
            tar.addfile(info, io.BytesIO(data))
cargo_crate_sha256 = sha_bytes(tar_buf.getvalue())

closure = json.loads((shell / "manifest.json").read_text(encoding="utf-8"))["closure"]
upstream_commit = json.loads((studio_root / "foundry-assets.lock.json").read_text(encoding="utf-8"))["upstream_commit"]

prov = {
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
    "lcnf_manifest_sha256": lcnf_manifest_sha256,
    "kernel_ir_sha256": actual["kernel_ir_sha256"],
    "coverage_sha256": actual["coverage_sha256"],
    "roots_sha256": actual["roots_sha256"],
    "generated_core_sha256": actual["generated_core_sha256"],
    "cargo_name": cargo_name,
    "cargo_version": cargo_version,
    "cargo_crate_sha256": cargo_crate_sha256,
    "cargo_crate_kind": "deterministic tar of Cargo.toml, Cargo.lock, src/** and generated/hologram_forge_core.rs (cargo package refuses the git dependency)",
    "view_model_id": upstream["view_model_id"],
    "core_wasm_sha256": sha_path(shell / "core.wasm"),
    "browser_projection_sha256": closure,
    "prismpm_commit": upstream["prismpm_commit"],
}
(shell / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
PY

node "$repo_root/scripts/studio/seal-shell-closure.mjs" "$shell_dir" --template "$shell_dir/sw.template.js"

echo "== 9/9 stage clean bundle to web/studio/dist/ =="
rm -rf "$dist_dir"
mkdir -p "$dist_dir"
python3 -c '
import pathlib, shutil
src = pathlib.Path("'"$shell_dir"'")
dst = pathlib.Path("'"$dist_dir"'")
for p in sorted(src.rglob("*")):
    rel = p.relative_to(src)
    srel = str(rel).replace("\\", "/")
    if srel == "sw.template.js" or srel.endswith(".test.mjs") or srel.endswith(".test.js") or srel.endswith(".test.ts"):
        continue
    target = dst / rel
    if p.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    elif p.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
'
for req in app.json q/pkg/holospaces_web.js q/pkg/holospaces_web_bg.wasm core.wasm host.bundle.js manifest.json sw.js standards-index.json v13/uor_foundry_v13_wasm.js v13/uor_foundry_v13_wasm_bg.wasm v86/libv86.js v86/v86.wasm v86/seabios.bin v86/vgabios.bin images/buildroot-bzimage.bin; do
    [ -f "$dist_dir/$req" ] || { echo "build_foundry_studio: missing required dist file $req" >&2; exit 1; }
done
printf 'Built Foundry Studio in %s (%s files)\n' "$dist_dir" "$(find "$dist_dir" -type f | wc -l | tr -d ' ')"
