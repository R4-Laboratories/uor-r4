#!/usr/bin/env bash
set -euo pipefail

# Gate 1 `UORBFY03` Capacity, Stereographic Selectivity & Loss Breakthrough (`antigravity-5`)
# Runs host unit tests, AOT sm_<CC> release build, Gate 1 (exact embedded cubin PTX/SASS audit:
# 0 PTX float/div/MMA, 0 arithmetic SASS float/MMA), Gate 2 (0-diff GPU-vs-CPU parity across 4
# V3 configs), and parallel 2x RTX 5090 training (GPU 0: v3_d256_l8_ffn4_k32; GPU 1:
# v3_d512_l6_ffn4_k64) on train-100m. Sealed `dev-test.*` is NEVER opened.

OUT_DIR="${1:?usage: dispatch.sh <OUT_DIR> <CORPUS_DIR> <HELDOUT_U16> <TOKENIZER_JSON> <M2_DIR>}"
CORPUS_DIR="${2:?missing CORPUS_DIR}"
HELDOUT_U16="${3:?missing HELDOUT_U16}"
TOKENIZER_JSON="${4:?missing TOKENIZER_JSON}"
M2_DIR="${5:?missing M2_DIR}"

# L-002: refuse writing into an existing/sealed report root
if [[ -e "$OUT_DIR" ]]; then
  echo "ERROR: OUT_DIR already exists (L-002 refuse overwrite): $OUT_DIR" >&2
  exit 1
fi

export PATH="/usr/local/cuda/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$SCRIPT_DIR/butterfly-v3"
TARGET_DIR="${CARGO_TARGET_DIR:-/workspace/bfy-v3-target}"
export CARGO_TARGET_DIR="$TARGET_DIR"
mkdir -p "$OUT_DIR/audit" "$OUT_DIR/evidence" "$OUT_DIR/artifacts"

export CUDA_DISABLE_PTX_JIT=1

CAP_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:].')"
if [[ -z "$CAP_RAW" ]]; then
  echo "FATAL: could not detect GPU compute capability" >&2
  exit 1
fi
export UOR_NVCC_ARCH="compute_${CAP_RAW}"
SM_ARCH="sm_${CAP_RAW}"
echo "=== Detected GPU arch: $SM_ARCH ($UOR_NVCC_ARCH), CUDA_DISABLE_PTX_JIT=1 ==="

echo "=== Running host tests & building butterfly-v3 (--release --features cuda) ==="
rm -rf "$TARGET_DIR/release/build"/butterfly-v3-*
(cd "$CRATE_DIR" && cargo test --bins && cargo build --release --features cuda)
BIN="$TARGET_DIR/release/butterfly-v3"

mapfile -t INT_CUBINS < <(find "$TARGET_DIR/release/build" -name "butterfly_kernels_v3.${SM_ARCH}.cubin" | sort)
if [[ "${#INT_CUBINS[@]}" -ne 1 || ! -f "${INT_CUBINS[0]}" ]]; then
  echo "ERROR: expected 1 compiled ${SM_ARCH} pure-int v3 cubin, found ${#INT_CUBINS[@]}" >&2
  exit 1
fi

INT_CUBIN="${INT_CUBINS[0]}"
cp "$INT_CUBIN" "$OUT_DIR/audit/butterfly_kernels_v3.cubin"

INT_CU="$CRATE_DIR/src/butterfly_kernels_v3.cu"
INT_PTX="$OUT_DIR/audit/butterfly_kernels_v3.ptx"
INT_SASS="$OUT_DIR/audit/butterfly_kernels_v3.sass"

nvcc -ptx -arch="$SM_ARCH" -O3 -x cu "$INT_CU" -o "$INT_PTX"
cuobjdump --dump-sass "$INT_CUBIN" > "$INT_SASS"

# ---------------------------------------------------------------------------
# Gate 1: Strict AOT PTX/SASS Audit (0 PTX float/div/MMA, 0 arithmetic SASS float/MMA)
# ---------------------------------------------------------------------------
python3 - "$INT_PTX" "$INT_SASS" "$INT_CUBIN" "$OUT_DIR/evidence/gate1_audit.json" "$CAP_RAW" <<'PY'
import hashlib, json, re, sys

int_ptx_p, int_sass_p, int_cubin_p, out_p, cap_digits = sys.argv[1:6]
sm_arch = f"sm_{cap_digits}"

def sha256_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def ptx_code_lines(text):
    out = []
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith((".file", ".loc", ".section", ".version", ".target", ".address_size")):
            continue
        out.append(line)
    return out

int_ptx = open(int_ptx_p).read()
int_sass = open(int_sass_p).read()

req_int = ["bfy_v3_batch_fwd_bwd_kernel", "bfy_v3_adamw_update_kernel", "bfy_v3_eval_records_kernel"]
miss_int_ptx = [k for k in req_int if not re.search(rf'\.entry\s+{re.escape(k)}\b', int_ptx)]
miss_int_sass = [k for k in req_int if not re.search(rf'Function\s*:\s*{re.escape(k)}\b', int_sass)]

ptx_float_re = re.compile(r'\.(?:f16|f16x2|bf16|bf16x2|f32|f64|tf32|e4m3|e5m2|e4m3x2|e5m2x2|approx|ftz)\b')
ptx_div_re = re.compile(r'\b(?:div|rem)\.(?:u|s|b)(?:16|32|64)\b')
ptx_mma_re = re.compile(r'\b(?:mma|wmma|wgmma|tcgen05\.mma)\b')
sass_float_re = re.compile(
    r'\b(?:FADD|FMUL|FFMA|FMA|FSET|FSETP|FSEL|FCHK|FMNMX|FRND|FCMP|FSWZADD|'
    r'F2I|I2F|F2F|F2FP|UF2FP|DADD|DMUL|DFMA|DSET|DSETP|DMNMX|DCHK|DMOV|'
    r'HADD|HMUL|HFMA|HSET|HSETP|HMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO)[A-Z0-9_]*\b'
)
sm120_mov_alias_re = re.compile(r'\bHFMA2\s+R\d+,\s*-RZ,\s*RZ,')
sass_mma_re = re.compile(r'\b(?:HMMA|IMMA|BMMA|DMMA|QMMA|GMMA|WMMA|UTCMMA)[A-Z0-9_]*\b')

int_ptx_lines = ptx_code_lines(int_ptx)
int_ptx_float = [l for l in int_ptx_lines if ptx_float_re.search(l)]
int_ptx_div = [l for l in int_ptx_lines if ptx_div_re.search(l)]
int_ptx_mma = [l for l in int_ptx_lines if ptx_mma_re.search(l)]

int_sass_mov_aliases = 0
int_sass_float = []
int_sass_mma = []
for line in int_sass.splitlines():
    if sass_mma_re.search(line):
        int_sass_mma.append(line.strip())
    if sass_float_re.search(line):
        if cap_digits == "120" and sm120_mov_alias_re.search(line):
            int_sass_mov_aliases += 1
        else:
            int_sass_float.append(line.strip())

int_pass = (
    not miss_int_ptx and not miss_int_sass
    and len(int_ptx_float) == 0 and len(int_ptx_div) == 0 and len(int_ptx_mma) == 0
    and len(int_sass_float) == 0 and len(int_sass_mma) == 0
)

res = {
    "arch": sm_arch,
    "gate1_pure_int_v3": {
        "cubin_path": int_cubin_p,
        "cubin_sha256": sha256_file(int_cubin_p),
        "required_kernels": req_int,
        "missing_ptx_kernels": miss_int_ptx,
        "missing_sass_kernels": miss_int_sass,
        "ptx_float_hits": len(int_ptx_float),
        "ptx_div_hits": len(int_ptx_div),
        "ptx_mma_hits": len(int_ptx_mma),
        "sass_sm120_mov_aliases": int_sass_mov_aliases,
        "sass_float_hits": len(int_sass_float),
        "sass_mma_hits": len(int_sass_mma),
        "pass": int_pass,
    },
    "pass": int_pass,
}
open(out_p, 'w').write(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
if not int_pass:
    if int_ptx_float:
        print("INT PTX FLOAT SAMPLE:", int_ptx_float[:10], file=sys.stderr)
    if int_ptx_div:
        print("INT PTX DIV SAMPLE:", int_ptx_div[:10], file=sys.stderr)
    if int_sass_float:
        print("INT SASS FLOAT SAMPLE:", int_sass_float[:10], file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# Gate 2: Finite Pure-Int GPU-vs-CPU Exact Parity (`0` diffs across 4 V3 configs)
# ---------------------------------------------------------------------------
echo "=== Gate 2: Finite GPU-vs-CPU Exact Parity (UORBFY03, 4 configs) ==="
CUDA_VISIBLE_DEVICES=0 "$BIN" finite --out "$OUT_DIR/evidence/finite.json"

# ---------------------------------------------------------------------------
# Gate 3: Parallel 2-Arm `UORBFY03` Training on `train-100m` (DEV-test sealed)
# ---------------------------------------------------------------------------
N_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
echo "=== Detected $N_GPUS GPU(s) for 2-arm UORBFY03 Gate 1 run ==="

run_arm1() {
  echo "=== [GPU 0] Arm 1: v3_d256_l8_ffn4_k32 (D=256, L=8, ffn4, kd=32, k=32, freq-sorted, learned PKM) ==="
  CUDA_VISIBLE_DEVICES=0 "$BIN" train-int \
    --corpus-dir "$CORPUS_DIR" --heldout "$HELDOUT_U16" --tokenizer "$TOKENIZER_JSON" --m2-dir "$M2_DIR" \
    --dim 256 --layers 8 --ffn-branches 4 --pkm-sub-keys 64 --pkm-key-dim 32 --loss-k 32 \
    --epochs 1 --batch 128 --seq-len 256 \
    --lr-q8 4096 --wd-shift 19 --eps 16 --seed 20261128 --auto-sft-below-mbpb 2000000 \
    --out-ckpt "$OUT_DIR/artifacts/v3_d256_l8_ffn4_k32.uorbfy03" \
    --out-replay "$OUT_DIR/evidence/v3_d256_l8_ffn4_k32_replay_fixture.json" \
    --out-report "$OUT_DIR/evidence/v3_d256_l8_ffn4_k32_report.json"
}

run_arm2() {
  local DEV="$1"
  echo "=== [GPU $DEV] Arm 2: v3_d512_l6_ffn4_k64 (D=512, L=6, ffn4, kd=32, k=64, freq-sorted, learned PKM) ==="
  CUDA_VISIBLE_DEVICES="$DEV" "$BIN" train-int \
    --corpus-dir "$CORPUS_DIR" --heldout "$HELDOUT_U16" --tokenizer "$TOKENIZER_JSON" --m2-dir "$M2_DIR" \
    --dim 512 --layers 6 --ffn-branches 4 --pkm-sub-keys 64 --pkm-key-dim 32 --loss-k 64 \
    --epochs 1 --batch 128 --seq-len 256 \
    --lr-q8 4096 --wd-shift 19 --eps 16 --seed 20261128 --auto-sft-below-mbpb 2000000 \
    --out-ckpt "$OUT_DIR/artifacts/v3_d512_l6_ffn4_k64.uorbfy03" \
    --out-replay "$OUT_DIR/evidence/v3_d512_l6_ffn4_k64_replay_fixture.json" \
    --out-report "$OUT_DIR/evidence/v3_d512_l6_ffn4_k64_report.json"
}

if [[ "$N_GPUS" -ge 2 ]]; then
  run_arm1 > "$OUT_DIR/gpu0.log" 2>&1 &
  PID0=$!
  run_arm2 1 > "$OUT_DIR/gpu1.log" 2>&1 &
  PID1=$!
  FAIL=0
  wait "$PID0" || FAIL=1
  wait "$PID1" || FAIL=1
  echo "--- GPU 0 LOG ---"
  cat "$OUT_DIR/gpu0.log"
  echo "--- GPU 1 LOG ---"
  cat "$OUT_DIR/gpu1.log"
  if [[ "$FAIL" -ne 0 ]]; then
    echo "FATAL: one or more parallel GPU arms failed" >&2
    exit 1
  fi
else
  run_arm1
  run_arm2 0
fi

# ---------------------------------------------------------------------------
# Seal receipts.sha256
# ---------------------------------------------------------------------------
echo "=== Sealing receipts.sha256 ==="
(
  cd "$OUT_DIR"
  sha256sum \
    evidence/finite.json \
    evidence/gate1_audit.json \
    evidence/v3_d256_l8_ffn4_k32_replay_fixture.json \
    evidence/v3_d256_l8_ffn4_k32_report.json \
    evidence/v3_d512_l6_ffn4_k64_replay_fixture.json \
    evidence/v3_d512_l6_ffn4_k64_report.json \
    artifacts/v3_d256_l8_ffn4_k32.uorbfy03 \
    artifacts/v3_d512_l6_ffn4_k64.uorbfy03 \
    > receipts.sha256
  sha256sum -c receipts.sha256
)
echo "GATE1_V3_COMPLETE: $OUT_DIR"
