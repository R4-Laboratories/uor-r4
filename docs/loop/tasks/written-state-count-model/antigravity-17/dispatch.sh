#!/usr/bin/env bash
set -euo pipefail

# Phase P4 PR 2 Wick RTX 4090 qualification driver (antigravity-17, #4063).
# Runs Gates 1-4 on Wick RTX 4090 ($0.00 pod spend) and seals receipts.sha256.

REPO_ROOT="${REPO_ROOT:-/tmp/p4-qual-run/repo}"
OUT_DIR="${OUT_DIR:-/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual}"
WICK_GPU_LOCK="/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock"
FALLBACK_LOCK="/tmp/uor-antigravity-p4-qual.lock"

TRUNK_PATH="/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11"
TOKENIZER_PATH="/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json"
MANIFEST_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.manifest.json"
A1S_WEIGHTS_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.weights"
COUNTS_V3_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate3-fullc/full-c.v3"
S3_T02_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p2-fallback-devtune/tables/golden_shell.uors3t02"
S3_ERR_P_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p5b_s3_err_p.i8"
S3_ERR_B_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p5b_s3_err_b.i8"
S2_S01_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-eval-r3/evidence/artifacts/s2_skip.uors2s01"
S2_ERRS_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p5b_s2_skip_errs.i8"
S4_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5a-s4-r3/artifacts/c_ppmi_s4.uors4p01"
S6_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p4_full_errwrite.uors6b01"
HELDOUT_PATH="/mnt/plexpool/uor-r4-corpus/inputs/chat-v0-p2/heldout.u16"

# L-002: never write into an existing / previously sealed output root.
if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: OUT_DIR $OUT_DIR already exists (L-002 refuse overwrite)" >&2
    exit 1
fi

# Acquire Wick shared GPU lock (non-blocking flock) so we never collide with another lab.
if [[ -d "$(dirname "$WICK_GPU_LOCK")" ]]; then
    exec 9>"$WICK_GPU_LOCK"
else
    exec 9>"$FALLBACK_LOCK"
fi
if ! flock -n 9; then
    echo "ERROR: Wick GPU lock is held by another job" >&2
    exit 1
fi

# Verify sm_89 capability and >= 6,000 MiB free VRAM before launching compute.
GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)"
echo "GPU: $GPU_INFO"
python3 - "$GPU_INFO" <<'PY'
import sys
parts = [p.strip() for p in sys.argv[1].split(",")]
name, cc, free_mib, util = parts[0], parts[1], int(parts[2]), int(parts[3])
if cc != "8.9":
    raise SystemExit(f"ERROR: expected compute_cap 8.9 (sm_89), got {cc}")
if free_mib < 6000:
    raise SystemExit(f"ERROR: insufficient free VRAM ({free_mib} MiB < 6000 MiB)")
print(f"GPU preflight OK: {name} cc={cc} free={free_mib} MiB util={util}%")
PY

mkdir -p "$OUT_DIR/evidence" "$OUT_DIR/artifacts" "$OUT_DIR/audit"
CRATE_DIR="$REPO_ROOT/docs/loop/tasks/written-state-count-model/antigravity-17/p4-qual"
TARGET_DIR="/tmp/p4-qual-run/target"
export CARGO_TARGET_DIR="$TARGET_DIR"
export UOR_NVCC_ARCH="compute_89"
export CUDA_DISABLE_PTX_JIT=1

echo "=== Gate 1: Build AOT sm_89 CUDA binary & audit PTX/SASS for zero float/MMA ==="
# Clean any prior p4-qual build script outdirs so exactly one fresh cubin exists.
rm -rf "$TARGET_DIR/release/build"/p4-qual-*
(cd "$CRATE_DIR" && cargo build --release --offline --features cuda)
BIN="$TARGET_DIR/release/p4-qual"
mapfile -t CUBINS < <(find "$TARGET_DIR/release/build" -name 'p4_kernels.sm_89.cubin' | sort)
if [[ "${#CUBINS[@]}" -ne 1 || ! -f "${CUBINS[0]}" ]]; then
    echo "ERROR: expected exactly 1 compiled sm_89 cubin, found ${#CUBINS[@]}" >&2
    exit 1
fi
CUBIN="${CUBINS[0]}"
CUBIN_SHA="$(sha256sum "$CUBIN" | awk '{print $1}')"
echo "cubin=$CUBIN sha256=$CUBIN_SHA"

nvcc -ptx -arch=sm_89 -O3 -I "$REPO_ROOT/crates/uor-r4-discrete/src" \
    -x cu "$CRATE_DIR/src/p4_kernels.cu" -o "$OUT_DIR/audit/p4_kernels.ptx"
cuobjdump --dump-sass "$CUBIN" > "$OUT_DIR/audit/p4_kernels.sass"

python3 - "$OUT_DIR/audit/p4_kernels.ptx" "$OUT_DIR/audit/p4_kernels.sass" "$CUBIN_SHA" "$OUT_DIR/evidence/gate1_audit.json" <<'PY'
import json, re, sys
ptx_path, sass_path, cubin_sha, out_path = sys.argv[1:5]
ptx = open(ptx_path).read()
sass = open(sass_path).read()

required_kernels = ["p4_parity_batch_kernel", "p4_bar_a_windows_kernel"]
missing_ptx = [k for k in required_kernels if not re.search(rf'\.entry\s+{re.escape(k)}\b', ptx)]
missing_sass = [k for k in required_kernels if not re.search(rf'Function\s*:\s*{re.escape(k)}\b', sass)]

ptx_float_re = re.compile(r'\.(?:f16|f16x2|bf16|bf16x2|f32|f64|tf32|e4m3|e5m2|e4m3x2|e5m2x2|approx|ftz)\b')
ptx_mma_re = re.compile(r'\b(?:mma|wmma|wgmma)\.')
# Match all SASS float/half/double/transcendental/conversion/MMA opcodes (including *32I, *P, *2 variants).
sass_float_re = re.compile(
    r'\b(?:FADD|FMUL|FFMA|FMA|FSET|FSETP|FSEL|FCHK|FMNMX|FRND|FCMP|FSWZADD|'
    r'F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DSETP|DMNMX|'
    r'HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO)[A-Z0-9_]*\b'
)
sass_mma_re = re.compile(r'\b(?:HMMA|IMMA|BMMA|DMMA|QMMA|GMMA|WMMA)[A-Z0-9_]*\b')

ptx_f = [l.strip() for l in ptx.splitlines() if not l.strip().startswith('//') and ptx_float_re.search(l)]
ptx_m = [l.strip() for l in ptx.splitlines() if not l.strip().startswith('//') and ptx_mma_re.search(l)]
sass_f = [l.strip() for l in sass.splitlines() if sass_float_re.search(l)]
sass_m = [l.strip() for l in sass.splitlines() if sass_mma_re.search(l)]

ok = (
    len(missing_ptx) == 0
    and len(missing_sass) == 0
    and len(ptx_f) == 0
    and len(ptx_m) == 0
    and len(sass_f) == 0
    and len(sass_m) == 0
)
res = {
    "cubin_sha256": cubin_sha,
    "required_kernels": required_kernels,
    "missing_ptx_kernels": missing_ptx,
    "missing_sass_kernels": missing_sass,
    "ptx_float_hits": len(ptx_f),
    "ptx_mma_hits": len(ptx_m),
    "sass_float_hits": len(sass_f),
    "sass_mma_hits": len(sass_m),
    "pass": ok,
}
with open(out_path, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
if not ok:
    print("Missing PTX:", missing_ptx, "Missing SASS:", missing_sass)
    print("PTX floats:", ptx_f[:10])
    print("SASS floats:", sass_f[:10])
    print("PTX MMA:", ptx_m[:10], "SASS MMA:", sass_m[:10])
    sys.exit(1)
PY

echo "=== Gate 2: Synthetic finite GPU-vs-CPU parity ==="
"$BIN" finite --out "$OUT_DIR/evidence/finite.json"

echo "=== Gate 3a: Pack frozen Stage B5b/B5a/B4/B1a artifacts into p4.d11 (<= 4 GB) ==="
"$BIN" pack \
    --trunk "$TRUNK_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --manifest "$MANIFEST_PATH" \
    --a1s-weights "$A1S_WEIGHTS_PATH" \
    --counts-v3 "$COUNTS_V3_PATH" \
    --s3-t02 "$S3_T02_PATH" \
    --s3-err-p "$S3_ERR_P_PATH" \
    --s3-err-b "$S3_ERR_B_PATH" \
    --s2-s01 "$S2_S01_PATH" \
    --s2-errs "$S2_ERRS_PATH" \
    --s4 "$S4_PATH" \
    --s6 "$S6_PATH" \
    --out "$OUT_DIR/artifacts/p4.d11" \
    --report "$OUT_DIR/evidence/pack_manifest.json"

echo "=== Gate 3b: Generate & verify 96-prompt CUDA-vs-CPU parity fixture ==="
"$BIN" parity-fixture \
    --p4 "$OUT_DIR/artifacts/p4.d11" \
    --heldout "$HELDOUT_PATH" \
    --out "$OUT_DIR/artifacts/cuda_parity_96.json" \
    --summary "$OUT_DIR/evidence/parity_fixture_summary.json"

echo "=== Gate 4: Bar A (chat-v0-p2 heldout.u16) GPU PRLL scoring ==="
"$BIN" bar-a \
    --p4 "$OUT_DIR/artifacts/p4.d11" \
    --heldout "$HELDOUT_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --out "$OUT_DIR/evidence/bar_a_report.json"

echo "=== Cross-check cubin_sha256 & pass flags across all Gate 1-4 evidence ==="
python3 - "$CUBIN_SHA" "$OUT_DIR" <<'PY'
import hashlib, json, os, sys
cubin_sha, out_dir = sys.argv[1:3]
ev_dir = os.path.join(out_dir, "evidence")
g1 = json.load(open(os.path.join(ev_dir, "gate1_audit.json")))
g2 = json.load(open(os.path.join(ev_dir, "finite.json")))
g3p = json.load(open(os.path.join(ev_dir, "pack_manifest.json")))
g3f = json.load(open(os.path.join(ev_dir, "parity_fixture_summary.json")))
g4 = json.load(open(os.path.join(ev_dir, "bar_a_report.json")))

fix_sha = hashlib.sha256(open(os.path.join(out_dir, "artifacts", "cuda_parity_96.json"), "rb").read()).hexdigest()
assert g1["pass"] is True and g1["cubin_sha256"] == cubin_sha
assert g2["pass"] is True and g2["cubin_sha256"] == cubin_sha
assert g3p["within_4gb_cap"] is True
assert g3f["pass"] is True and g3f["cubin_sha256"] == cubin_sha
assert g3f["prompt_count"] >= 64
assert g3f["wick_cuda_vs_cpu_step_diffs"] == 0
assert g3f["wick_cuda_vs_cpu_score_word_diffs"] == 0
assert g3f["fixture_sha256"] == fix_sha
assert g4["cubin_sha256"] == cubin_sha
assert g4["determinism_repeat_diffs"] == 0
assert g4["cpu_spot_checks"]["diffs"] == 0
assert g4["cpu_spot_checks"]["positions_checked"] >= 512
print("CROSS-CHECK OK: cubin_sha256 pinned across Gates 1-4:", cubin_sha)
PY

echo "=== Seal receipts.sha256 ==="
(cd "$OUT_DIR" && find artifacts evidence audit -type f | sort | xargs sha256sum > receipts.sha256)
(cd "$OUT_DIR" && sha256sum -c receipts.sha256)
echo "ALL WICK GATES 1-4 COMPLETE"
