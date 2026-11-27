#!/usr/bin/env bash
set -euo pipefail

# Phase P5 PR 2 Wick RTX 4090 training & sealed DEV-test driver (antigravity-19, #4063).
# Runs Gates 1-5 on Wick RTX 4090 ($0.00 pod spend) and seals receipts.sha256.

export PATH="/usr/local/cuda/bin:$PATH"

REPO_ROOT="${REPO_ROOT:-/tmp/p5-train-run/repo}"
OUT_DIR="${OUT_DIR:-/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5-qual-r2}"
WICK_GPU_LOCK="/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock"
FALLBACK_LOCK="/tmp/uor-antigravity-p5-qual.lock"

P4_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11"
SEED1_PATH="/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11"
TOKENIZER_PATH="/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json"
R_PATH="/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16"
DEVTUNE_PATH="/home/callard/uor-codex/20261118-diagnostics/inputs/c420/dev-tune.tokens.u16"
DEVTEST_DIR="/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/inputs/c420"
RSTAR_PATH="/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/rstar/run-v1/rstar-devtest.masses.u64"

MIN_EPOCHS="${MIN_EPOCHS:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
K_BLOCK="${K_BLOCK:-64}"

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

# Verify sm_89 capability and >= 8,000 MiB free VRAM before launching compute.
GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)"
echo "GPU: $GPU_INFO"
python3 - "$GPU_INFO" <<'PY'
import sys
parts = [p.strip() for p in sys.argv[1].split(",")]
name, cc, free_mib, util = parts[0], parts[1], int(parts[2]), int(parts[3])
if cc != "8.9":
    raise SystemExit(f"ERROR: expected compute_cap 8.9 (sm_89), got {cc}")
if free_mib < 8000:
    raise SystemExit(f"ERROR: insufficient free VRAM ({free_mib} MiB < 8000 MiB)")
print(f"GPU preflight OK: {name} cc={cc} free={free_mib} MiB util={util}%")
PY

mkdir -p "$OUT_DIR/evidence" "$OUT_DIR/artifacts" "$OUT_DIR/checkpoints" "$OUT_DIR/audit"
CRATE_DIR="$REPO_ROOT/docs/loop/tasks/written-state-count-model/antigravity-19/p5-train"
TARGET_DIR="/tmp/p5-train-run/target"
export CARGO_TARGET_DIR="$TARGET_DIR"
export UOR_NVCC_ARCH="compute_89"
export CUDA_DISABLE_PTX_JIT=1

echo "=== Gate 1a: uor-r4-discrete --features cuda 4,096-row & p5_brier_core unit tests ==="
(cd "$REPO_ROOT" && cargo test --offline --locked -p uor-r4-discrete --features cuda --lib driver_config_admits_4096_rows_and_refuses_out_of_range)
(cd "$REPO_ROOT" && cargo test --offline --locked -p uor-r4-discrete --features cuda --lib p5_brier_core)

echo "=== Gate 1b: Build AOT sm_89 p5-train binary & audit PTX/SASS for zero float/MMA ==="
rm -rf "$TARGET_DIR/release/build"/p5-train-*
(cd "$CRATE_DIR" && cargo build --release --offline --locked --features cuda)
BIN="$TARGET_DIR/release/p5-train"
mapfile -t CUBINS < <(find "$TARGET_DIR/release/build" -name 'p5_kernels.sm_89.cubin' | sort)
if [[ "${#CUBINS[@]}" -ne 1 || ! -f "${CUBINS[0]}" ]]; then
    echo "ERROR: expected exactly 1 compiled sm_89 cubin, found ${#CUBINS[@]}" >&2
    exit 1
fi
CUBIN="${CUBINS[0]}"
CUBIN_SHA="$(sha256sum "$CUBIN" | awk '{print $1}')"
echo "cubin=$CUBIN sha256=$CUBIN_SHA"

nvcc -ptx -arch=sm_89 -O3 -I "$REPO_ROOT/crates/uor-r4-discrete/src" \
    -x cu "$CRATE_DIR/src/p5_kernels.cu" -o "$OUT_DIR/audit/p5_kernels.ptx"
cuobjdump --dump-sass "$CUBIN" > "$OUT_DIR/audit/p5_kernels.sass"

python3 - "$OUT_DIR/audit/p5_kernels.ptx" "$OUT_DIR/audit/p5_kernels.sass" "$CUBIN_SHA" "$OUT_DIR/evidence/gate1_audit.json" <<'PY'
import json, re, sys
ptx_path, sass_path, cubin_sha, out_path = sys.argv[1:5]
ptx = open(ptx_path).read()
sass = open(sass_path).read()

required_kernels = [
    "p5_slot_step_brier_kernel",
    "p5_batch_eval_roots_kernel",
    "p5_rowsk_score_kernel",
    "p5_rowsk_reduce_kernel",
    "p5_eval_records_multi_arm_kernel",
]
missing_ptx = [k for k in required_kernels if not re.search(rf'\.entry\s+{re.escape(k)}\b', ptx)]
missing_sass = [k for k in required_kernels if not re.search(rf'Function\s*:\s*{re.escape(k)}\b', sass)]

ptx_float_re = re.compile(r'\.(?:f16|f16x2|bf16|bf16x2|f32|f64|tf32|e4m3|e5m2|e4m3x2|e5m2x2|approx|ftz)\b')
ptx_mma_re = re.compile(r'\b(?:mma|wmma|wgmma)\.')
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

echo "=== Gate 2: Synthetic finite GPU-vs-CPU parity & live 4,096-row uor-r4-discrete check ==="
"$BIN" finite --out "$OUT_DIR/evidence/finite.json"

echo "=== Gate 3: R rows-K + BVG-K S1 training vs equal-trial re-seeded controls ==="
"$BIN" train \
    --p4 "$P4_PATH" \
    --seed1 "$SEED1_PATH" \
    --r-uort "$R_PATH" \
    --dev-tune "$DEVTUNE_PATH" \
    --ckpt-dir "$OUT_DIR/checkpoints" \
    --report "$OUT_DIR/evidence/train_report.json" \
    --min-epochs "$MIN_EPOCHS" \
    --max-epochs "$MAX_EPOCHS" \
    --k "$K_BLOCK"

echo "=== Gate 4: Single-shot sealed DEV-test 5-arm qualification gate ==="
"$BIN" eval-devtest \
    --p4 "$P4_PATH" \
    --ckpt-dir "$OUT_DIR/checkpoints" \
    --train-report "$OUT_DIR/evidence/train_report.json" \
    --devtest-dir "$DEVTEST_DIR" \
    --rstar "$RSTAR_PATH" \
    --out "$OUT_DIR/evidence/devtest_p5_report.json"

echo "=== Gate 5: Conditional p5.d11 pack (ONLY if verdict == P5_QUALIFIED) ==="
"$BIN" pack-if-pass \
    --devtest-report "$OUT_DIR/evidence/devtest_p5_report.json" \
    --p4 "$P4_PATH" \
    --seed1 "$SEED1_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --s1-ckpt "$OUT_DIR/checkpoints/s1_p5_from_seed1.uorp5s10" \
    --out-p5 "$OUT_DIR/artifacts/p5.d11" \
    --report "$OUT_DIR/evidence/pack_status.json"

echo "=== Cross-check cubin_sha256 & pass flags across Gates 1-5 ==="
python3 - "$CUBIN_SHA" "$OUT_DIR" <<'PY'
import json, os, sys
cubin_sha, out_dir = sys.argv[1:3]
ev_dir = os.path.join(out_dir, "evidence")
g1 = json.load(open(os.path.join(ev_dir, "gate1_audit.json")))
g2 = json.load(open(os.path.join(ev_dir, "finite.json")))
g3 = json.load(open(os.path.join(ev_dir, "train_report.json")))
g4 = json.load(open(os.path.join(ev_dir, "devtest_p5_report.json")))
g5 = json.load(open(os.path.join(ev_dir, "pack_status.json")))

assert g1["pass"] is True and g1["cubin_sha256"] == cubin_sha
assert g2["pass"] is True and g2["cubin_sha256"] == cubin_sha
assert g2["row_brier_diffs_vs_p4_session"] == 0 and g2["rowsk_diffs"] == 0
assert all(g2["bvgk_cases"].values())
assert g2["slide_batch_check"]["row_diffs"] == 0 and g2["slide_batch_check"]["rowsk_diffs"] == 0
assert g2["slide_batch_check"]["slid_rows_checked"] >= 7
assert g2["multi_arm_eval"]["records_exhaustively_checked"] == 6
assert g2["multi_arm_eval"]["eval_rec_diffs"] == 0 and g2["multi_arm_eval"]["eval_spot_diffs"] == 0
assert g2["multi_arm_eval"]["slide_positions_checked"] >= 10 and g2["multi_arm_eval"]["slide_tail_diffs"] == 0
assert g2["driver_4096_check"]["pass"] is True and g2["driver_4096_check"]["cuda_executed"] is True
assert g3["pass"] is True and g3["cubin_sha256"] == cubin_sha
t_s1 = g3["arms"]["s1_p5_from_seed1"]["trials"]
t_cn = g3["arms"]["ctrl_reseed_canonical"]["trials"]
t_rn = g3["arms"]["ctrl_reseed_random"]["trials"]
assert t_s1 == t_cn == t_rn == g3["equal_trials_per_arm"] and t_s1 > 0
assert g4["cubin_sha256"] == cubin_sha
assert g4["cpu_spot_checks"]["positions_checked"] >= 512
assert g4["cpu_spot_checks"]["diffs"] == 0
assert g4["determinism_repeat"]["diffs"] == 0
assert g4["verdict"] in ("P5_QUALIFIED", "P5_S1_FROZEN_AT_DEFAULT")
assert g5["verdict"] == g4["verdict"]
print("CROSS-CHECK OK: cubin_sha256 pinned across Gates 1-4:", cubin_sha, "verdict:", g4["verdict"])
PY

echo "=== Seal receipts.sha256 ==="
(cd "$OUT_DIR" && find artifacts checkpoints evidence audit -type f | sort | xargs sha256sum > receipts.sha256)
(cd "$OUT_DIR" && sha256sum -c receipts.sha256)
echo "ALL WICK P5 GATES 1-5 COMPLETE"
