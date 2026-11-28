#!/usr/bin/env bash
set -euo pipefail

# Gate 1 PR 2 Pure-Integer Butterfly-Rotor CUDA Pilot Driver (antigravity-3, #4064).
# Works identically on Wick RTX 4090 (sm_89) or Vast.ai RTX 5090 (sm_120) via auto-detected compute cap.
# Enforces L-002 fresh OUT_DIR, AOT cubin (CUDA_DISABLE_PTX_JIT=1), 0 PTX/SASS float/MMA, and 0-diff parity.

export PATH="/usr/local/cuda/bin:$PATH"

REPO_ROOT="${REPO_ROOT:-/tmp/gate1-pilot-run/repo}"
OUT_DIR="${OUT_DIR:?must set fresh OUT_DIR (L-002)}"
CORPUS_DIR="${CORPUS_DIR:-/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/corpus}"
HELDOUT_PATH="${HELDOUT_PATH:-/mnt/plexpool/uor-r4-corpus/inputs/chat-v0-p2/heldout.u16}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json}"
M2_DIR="${M2_DIR:-}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-128}"
SEQ_LEN="${SEQ_LEN:-256}"
LR_Q8="${LR_Q8:-2048}"
WD_SHIFT="${WD_SHIFT:-17}"
MAX_STEPS="${MAX_STEPS:-999999999}"

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: OUT_DIR $OUT_DIR already exists (L-002 refuse overwrite)" >&2
    exit 1
fi

# Wick GPU safety: never collide with another lab's job on shared Wick RTX 4090 (CHARTER §5.4).
WICK_LOCK="${WICK_LOCK:-/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock}"
if [[ -e "$WICK_LOCK" || -d "$(dirname "$WICK_LOCK")" ]]; then
    exec 9>>"$WICK_LOCK"
    if ! flock -n 9; then
        echo "ERROR: Wick GPU lock $WICK_LOCK is held by another session; refusing to collide." >&2
        exit 75
    fi
fi

ACTIVE_APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$ACTIVE_APPS" ]]; then
    echo "ERROR: GPU already has active compute process(es): $ACTIVE_APPS; refusing to collide." >&2
    exit 75
fi

GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)"
echo "GPU: $GPU_INFO"
CAP_DIGITS="$(python3 - "$GPU_INFO" <<'PY'
import sys
parts = [p.strip() for p in sys.argv[1].split(",")]
name, cc, free_mib, util = parts[0], parts[1], int(parts[2]), int(parts[3])
digits = cc.replace(".", "")
if free_mib < 6000:
    raise SystemExit(f"ERROR: insufficient free VRAM ({free_mib} MiB < 6000 MiB)")
print(digits)
PY
)"

export UOR_NVCC_ARCH="compute_${CAP_DIGITS}"
export CUDA_DISABLE_PTX_JIT=1
echo "AOT target: sm_${CAP_DIGITS} (${UOR_NVCC_ARCH}), CUDA_DISABLE_PTX_JIT=1"

mkdir -p "$OUT_DIR/evidence" "$OUT_DIR/artifacts" "$OUT_DIR/audit"
CRATE_DIR="$REPO_ROOT/docs/loop/tasks/end-to-end-integration/antigravity-3/butterfly-pilot"
TARGET_DIR="${TARGET_DIR:-/tmp/gate1-pilot-run/target}"
export CARGO_TARGET_DIR="$TARGET_DIR"

echo "=== Gate 1a: Build AOT sm_${CAP_DIGITS} butterfly-pilot & audit PTX/SASS for 0 float / 0 MMA ==="
rm -rf "$TARGET_DIR/release/build"/butterfly-pilot-*
(cd "$CRATE_DIR" && cargo test --bins && cargo build --release --features cuda)
BIN="$TARGET_DIR/release/butterfly-pilot"
mapfile -t CUBINS < <(find "$TARGET_DIR/release/build" -name "butterfly_kernels.sm_${CAP_DIGITS}.cubin" | sort)
if [[ "${#CUBINS[@]}" -ne 1 || ! -f "${CUBINS[0]}" ]]; then
    echo "ERROR: expected 1 compiled sm_${CAP_DIGITS} cubin, found ${#CUBINS[@]}" >&2
    exit 1
fi
CUBIN="${CUBINS[0]}"
CUBIN_SHA="$(sha256sum "$CUBIN" | awk '{print $1}')"
echo "cubin=$CUBIN sha256=$CUBIN_SHA"

nvcc -ptx -arch="sm_${CAP_DIGITS}" -O3 -x cu "$CRATE_DIR/src/butterfly_kernels.cu" -o "$OUT_DIR/audit/butterfly_kernels.ptx"
cuobjdump --dump-sass "$CUBIN" > "$OUT_DIR/audit/butterfly_kernels.sass"

python3 - "$OUT_DIR/audit/butterfly_kernels.ptx" "$OUT_DIR/audit/butterfly_kernels.sass" "$CUBIN_SHA" "$CAP_DIGITS" "$OUT_DIR/evidence/gate1_audit.json" <<'PY'
import json, re, sys
ptx_path, sass_path, cubin_sha, cap_digits, out_path = sys.argv[1:6]
ptx = open(ptx_path).read()
sass = open(sass_path).read()

required_kernels = [
    "bfy_batch_fwd_bwd_kernel",
    "bfy_lion_update_kernel",
    "bfy_eval_records_kernel",
]
missing_ptx = [k for k in required_kernels if not re.search(rf'\.entry\s+{re.escape(k)}\b', ptx)]
missing_sass = [k for k in required_kernels if not re.search(rf'Function\s*:\s*{re.escape(k)}\b', sass)]

ptx_float_re = re.compile(r'\.(?:f16|f16x2|bf16|bf16x2|f32|f64|tf32|e4m3|e5m2|e4m3x2|e5m2x2|approx|ftz)\b')
ptx_div_re = re.compile(r'\b(?:div|rem)\.(?:u|s|b)(?:16|32|64)\b')
ptx_mma_re = re.compile(r'\b(?:mma|wmma|wgmma)\.')
sass_float_re = re.compile(
    r'\b(?:FADD|FMUL|FFMA|FMA|FSET|FSETP|FSEL|FCHK|FMNMX|FRND|FCMP|FSWZADD|'
    r'F2I|I2F|F2F|F2FP|UF2FP|DADD|DMUL|DFMA|DSET|DSETP|DMNMX|DCHK|DMOV|'
    r'HADD|HMUL|HFMA|HSET|HSETP|HMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO)[A-Z0-9_]*\b'
)
# Blackwell sm_120 ptxas emits `HFMA2 R<n>, -RZ, RZ, <imm>` as a 32-bit integer constant-load alias
# (project precedent: antigravity-10/11/12). Only exempt exact `-RZ, RZ,` constant loads on sm_120.
sm120_mov_alias_re = re.compile(r'\bHFMA2\s+R\d+,\s*-RZ,\s*RZ,')
sass_mma_re = re.compile(r'\b(?:HMMA|IMMA|BMMA|DMMA|QMMA|GMMA|WMMA)[A-Z0-9_]*\b')

ptx_f = [l.strip() for l in ptx.splitlines() if not l.strip().startswith('//') and ptx_float_re.search(l)]
ptx_d = [l.strip() for l in ptx.splitlines() if not l.strip().startswith('//') and ptx_div_re.search(l)]
ptx_m = [l.strip() for l in ptx.splitlines() if not l.strip().startswith('//') and ptx_mma_re.search(l)]
sass_all_f = [l.strip() for l in sass.splitlines() if sass_float_re.search(l)]
if cap_digits == "120":
    sass_aliases = [l for l in sass_all_f if sm120_mov_alias_re.search(l)]
    sass_f = [l for l in sass_all_f if not sm120_mov_alias_re.search(l)]
else:
    sass_aliases = []
    sass_f = sass_all_f
sass_m = [l.strip() for l in sass.splitlines() if sass_mma_re.search(l)]

ok = (
    len(missing_ptx) == 0
    and len(missing_sass) == 0
    and len(ptx_f) == 0
    and len(ptx_d) == 0
    and len(ptx_m) == 0
    and len(sass_f) == 0
    and len(sass_m) == 0
)
res = {
    "arch": f"sm_{cap_digits}",
    "cubin_sha256": cubin_sha,
    "required_kernels": required_kernels,
    "missing_ptx_kernels": missing_ptx,
    "missing_sass_kernels": missing_sass,
    "ptx_float_hits": len(ptx_f),
    "ptx_div_hits": len(ptx_d),
    "ptx_mma_hits": len(ptx_m),
    "sass_sm120_mov_aliases": len(sass_aliases),
    "sass_float_hits": len(sass_f),
    "sass_mma_hits": len(sass_m),
    "pass": ok,
}
with open(out_path, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
if not ok:
    print("Missing PTX:", missing_ptx, "Missing SASS:", missing_sass)
    print("PTX floats:", ptx_f[:10], "PTX div/rem:", ptx_d[:10])
    print("SASS floats:", sass_f[:10])
    print("PTX MMA:", ptx_m[:10], "SASS MMA:", sass_m[:10])
    sys.exit(1)
PY

echo "=== Gate 2: Finite GPU-vs-CPU 0-diff parity (D=32 & D=256, Hui-Belkin & Sparsemax64) ==="
"$BIN" finite --out "$OUT_DIR/evidence/finite.json"

echo "=== Gate 2b: Materialize & pin deterministic m2-fresh256 panel BEFORE training ==="
if [[ -z "$M2_DIR" || ! -f "$M2_DIR/inputs.json" ]]; then
    M2_DIR="$OUT_DIR/m2-fresh256"
    mkdir -p "$M2_DIR"
    (cd "$REPO_ROOT/docs/labs/m2-fresh-heldout-panel/instrument" && cargo run --release -- --emit-dir "$M2_DIR")
fi
PIN_M2_IN="1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a"
PIN_M2_LB="3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be"
GOT_M2_IN="$(sha256sum "$M2_DIR/inputs.json" | awk '{print $1}')"
GOT_M2_LB="$(sha256sum "$M2_DIR/labels.json" | awk '{print $1}')"
if [[ "$GOT_M2_IN" != "$PIN_M2_IN" || "$GOT_M2_LB" != "$PIN_M2_LB" ]]; then
    echo "ERROR: m2-fresh256 SHA-256 mismatch before training (in=$GOT_M2_IN lb=$GOT_M2_LB)" >&2
    exit 1
fi
M2_ARGS=(--m2-dir "$M2_DIR")

run_arm() {
    local name="$1"
    local layers="$2"
    local loss="$3"
    echo "=== Gate 3/4 Arm: $name (D=256, L=$layers, loss=$loss, epochs=$EPOCHS) ==="
    "$BIN" train \
        --corpus-dir "$CORPUS_DIR" \
        --heldout "$HELDOUT_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        "${M2_ARGS[@]}" \
        --out-ckpt "$OUT_DIR/artifacts/${name}.uorbfy01" \
        --out-report "$OUT_DIR/evidence/${name}_report.json" \
        --out-replay "$OUT_DIR/evidence/${name}_replay_fixture.json" \
        --dim 256 \
        --layers "$layers" \
        --pkm-sub-keys 64 \
        --loss "$loss" \
        --loss-k 8 \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --seq-len "$SEQ_LEN" \
        --lr-q8 "$LR_Q8" \
        --wd-shift "$WD_SHIFT" \
        --max-steps-per-epoch "$MAX_STEPS" \
        --seed 20261127
}

run_arm "hb_l4_d256" 4 "hui-belkin"
run_arm "hb_l8_d256" 8 "hui-belkin"
run_arm "sp_l8_d256" 8 "sparsemax"

echo "=== Sealing receipts.sha256 ==="
(cd "$OUT_DIR" && sha256sum evidence/*.json artifacts/*.uorbfy01 > receipts.sha256 && sha256sum -c receipts.sha256)
echo "GATE1_PILOT_COMPLETE: $OUT_DIR"
