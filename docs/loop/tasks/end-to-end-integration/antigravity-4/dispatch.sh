#!/usr/bin/env bash
set -euo pipefail

# Gate 1 Diagnostic & Pure-Integer AdamW Repair (`antigravity-4`) Dispatcher
# Runs host unit tests, AOT sm_<CC> release build, Gate 1a (Pure-Int exact embedded cubin PTX/SASS audit),
# Gate 1b (Diagnostic F32 0-transcendental/0-MMA audit), Gate 2 (0-diff finite Pure-Int AdamW parity +
# F32 central finite-difference gradient check), and the 2x2 factorial on train-100m.
# Supports 2x RTX 5090 parallel execution (GPU 0: Pure-Int Arms 1&2; GPU 1: Diag F32 Arms 3&4)
# or 1x GPU sequential fallback. Sealed `dev-test.*` is NEVER opened.

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
CRATE_DIR="$SCRIPT_DIR/butterfly-diag"
TARGET_DIR="${CARGO_TARGET_DIR:-/workspace/bfy-diag-target}"
export CARGO_TARGET_DIR="$TARGET_DIR"
mkdir -p "$OUT_DIR/audit" "$OUT_DIR/evidence" "$OUT_DIR/artifacts"

export CUDA_DISABLE_PTX_JIT=1

# Detect GPU compute capability via nvidia-smi (fail-closed)
CAP_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:].')"
if [[ -z "$CAP_RAW" ]]; then
  echo "FATAL: could not detect GPU compute capability" >&2
  exit 1
fi
export UOR_NVCC_ARCH="compute_${CAP_RAW}"
SM_ARCH="sm_${CAP_RAW}"
echo "=== Detected GPU arch: $SM_ARCH ($UOR_NVCC_ARCH), CUDA_DISABLE_PTX_JIT=1 ==="

# ---------------------------------------------------------------------------
# Build release AOT CUDA binary FIRST (plus host unit tests) and audit the
# EXACT `.cubin` artifacts embedded into `$BIN` by `build.rs`!
# ---------------------------------------------------------------------------
echo "=== Running host tests & building butterfly-diag (--release --features cuda) ==="
rm -rf "$TARGET_DIR/release/build"/butterfly-diag-*
(cd "$CRATE_DIR" && cargo test --bins && cargo build --release --features cuda)
BIN="$TARGET_DIR/release/butterfly-diag"

mapfile -t INT_CUBINS < <(find "$TARGET_DIR/release/build" -name "butterfly_kernels.${SM_ARCH}.cubin" | sort)
mapfile -t F32_CUBINS < <(find "$TARGET_DIR/release/build" -name "diag_f32_kernels.${SM_ARCH}.cubin" | sort)
if [[ "${#INT_CUBINS[@]}" -ne 1 || ! -f "${INT_CUBINS[0]}" ]]; then
  echo "ERROR: expected 1 compiled ${SM_ARCH} pure-int cubin, found ${#INT_CUBINS[@]}" >&2
  exit 1
fi
if [[ "${#F32_CUBINS[@]}" -ne 1 || ! -f "${F32_CUBINS[0]}" ]]; then
  echo "ERROR: expected 1 compiled ${SM_ARCH} diag-f32 cubin, found ${#F32_CUBINS[@]}" >&2
  exit 1
fi

INT_CUBIN="${INT_CUBINS[0]}"
F32_CUBIN="${F32_CUBINS[0]}"
cp "$INT_CUBIN" "$OUT_DIR/audit/butterfly_kernels.cubin"
cp "$F32_CUBIN" "$OUT_DIR/audit/diag_f32_kernels.cubin"

INT_CU="$CRATE_DIR/src/butterfly_kernels.cu"
F32_CU="$CRATE_DIR/src/diag_f32_kernels.cu"
INT_PTX="$OUT_DIR/audit/butterfly_kernels.ptx"
INT_SASS="$OUT_DIR/audit/butterfly_kernels.sass"
F32_PTX="$OUT_DIR/audit/diag_f32_kernels.ptx"
F32_SASS="$OUT_DIR/audit/diag_f32_kernels.sass"

nvcc -ptx -arch="$SM_ARCH" -O3 -x cu "$INT_CU" -o "$INT_PTX"
cuobjdump --dump-sass "$INT_CUBIN" > "$INT_SASS"

nvcc -ptx -arch="$SM_ARCH" -O3 -x cu "$F32_CU" -o "$F32_PTX"
cuobjdump --dump-sass "$F32_CUBIN" > "$F32_SASS"

# ---------------------------------------------------------------------------
# Gate 1a & 1b: Strict AOT PTX/SASS Audits (Pure-Int 0-float/div/MMA; Diag F32 0-trans/0-MMA)
# ---------------------------------------------------------------------------
python3 - "$INT_PTX" "$INT_SASS" "$INT_CUBIN" "$F32_PTX" "$F32_SASS" "$F32_CUBIN" "$OUT_DIR/evidence/gate1_audit.json" "$CAP_RAW" <<'PY'
import hashlib, json, re, sys

int_ptx_p, int_sass_p, int_cubin_p, f32_ptx_p, f32_sass_p, f32_cubin_p, out_p, cap_digits = sys.argv[1:9]
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
f32_ptx = open(f32_ptx_p).read()
f32_sass = open(f32_sass_p).read()

req_int = ["bfy_batch_fwd_bwd_kernel", "bfy_adamw_update_kernel", "bfy_eval_records_kernel"]
req_f32 = ["diag_f32_batch_fwd_bwd_kernel", "diag_f32_adamw_update_kernel", "diag_f32_eval_records_kernel"]

miss_int_ptx = [k for k in req_int if not re.search(rf'\.entry\s+{re.escape(k)}\b', int_ptx)]
miss_int_sass = [k for k in req_int if not re.search(rf'Function\s*:\s*{re.escape(k)}\b', int_sass)]
miss_f32_ptx = [k for k in req_f32 if not re.search(rf'\.entry\s+{re.escape(k)}\b', f32_ptx)]
miss_f32_sass = [k for k in req_f32 if not re.search(rf'Function\s*:\s*{re.escape(k)}\b', f32_sass)]

# Gate 1a: Strict antigravity-3 regexes (+ wgmma, tcgen05, WMMA, UTCMMA)
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

# Gate 1b: Diagnostic F32 audit (CHARTER §3.2 zero transcendentals/squashing & §3.3 zero MMA)
f32_trans_ptx_re = re.compile(r'\b(?:ex2|lg2|tanh|sin|cos)\b')
f32_trans_sass_re = re.compile(r'\bMUFU\.(?:EX2|LG2|TANH|SIN|COS)\b')
f32_ptx_lines = ptx_code_lines(f32_ptx)
f32_ptx_trans = [l for l in f32_ptx_lines if f32_trans_ptx_re.search(l)]
f32_sass_trans = [l.strip() for l in f32_sass.splitlines() if f32_trans_sass_re.search(l)]
f32_ptx_mma = [l for l in f32_ptx_lines if ptx_mma_re.search(l)]
f32_sass_mma = [l.strip() for l in f32_sass.splitlines() if sass_mma_re.search(l)]

int_pass = (
    not miss_int_ptx and not miss_int_sass
    and len(int_ptx_float) == 0 and len(int_ptx_div) == 0 and len(int_ptx_mma) == 0
    and len(int_sass_float) == 0 and len(int_sass_mma) == 0
)
f32_pass = (
    not miss_f32_ptx and not miss_f32_sass
    and len(f32_ptx_trans) == 0 and len(f32_sass_trans) == 0
    and len(f32_ptx_mma) == 0 and len(f32_sass_mma) == 0
)

res = {
    "arch": sm_arch,
    "gate1a_pure_int": {
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
    "gate1b_diag_f32": {
        "cubin_path": f32_cubin_p,
        "cubin_sha256": sha256_file(f32_cubin_p),
        "required_kernels": req_f32,
        "missing_ptx_kernels": miss_f32_ptx,
        "missing_sass_kernels": miss_f32_sass,
        "ptx_exp_log_hits": len(f32_ptx_trans),
        "sass_exp_log_hits": len(f32_sass_trans),
        "ptx_mma_hits": len(f32_ptx_mma),
        "sass_mma_hits": len(f32_sass_mma),
        "pass": f32_pass,
    },
    "pass": int_pass and f32_pass,
}
open(out_p, 'w').write(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
if not res["pass"]:
    if int_ptx_float:
        print("INT PTX FLOAT SAMPLE:", int_ptx_float[:10], file=sys.stderr)
    if int_sass_float:
        print("INT SASS FLOAT SAMPLE:", int_sass_float[:10], file=sys.stderr)
    if f32_ptx_trans:
        print("F32 PTX TRANS SAMPLE:", f32_ptx_trans[:10], file=sys.stderr)
    if f32_sass_trans:
        print("F32 SASS TRANS SAMPLE:", f32_sass_trans[:10], file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# Gate 2: Finite Pure-Integer GPU-vs-CPU Exact Parity (0 diffs) + F32 FD Check
# ---------------------------------------------------------------------------
echo "=== Gate 2: Finite GPU-vs-CPU Exact Parity & Diag F32 Adjoint Check ==="
CUDA_VISIBLE_DEVICES=0 "$BIN" finite --out "$OUT_DIR/evidence/finite.json"

# ---------------------------------------------------------------------------
# Gate 3: 2x2 Factorial on train-100m (DEV-test strictly sealed)
# ---------------------------------------------------------------------------
N_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
echo "=== Detected $N_GPUS GPU(s) for 2x2 factorial ==="

run_gpu0() {
  echo "=== [GPU 0] Arm 1: int_adamw_hb8 (Pure-Int Q15 AdamW, k=8, identity perm, frozen PKM keys) ==="
  CUDA_VISIBLE_DEVICES=0 "$BIN" train-int \
    --corpus-dir "$CORPUS_DIR" --heldout "$HELDOUT_U16" --tokenizer "$TOKENIZER_JSON" --m2-dir "$M2_DIR" \
    --dim 256 --layers 4 --pkm-sub-keys 64 --loss-k 8 --epochs 1 --batch 128 --seq-len 256 \
    --lr-q8 4096 --wd-shift 17 --eps 16 --seed 20261127 --auto-sft-below-mbpb 2000000 \
    --out-ckpt "$OUT_DIR/artifacts/int_adamw_hb8.uorbfy01" \
    --out-replay "$OUT_DIR/evidence/int_adamw_hb8_replay_fixture.json" \
    --out-report "$OUT_DIR/evidence/int_adamw_hb8_report.json"

  echo "=== [GPU 0] Arm 2: int_adamw_brier1_bal (Pure-Int Q15 AdamW, k=1 Brier, TRAIN-balanced perm, learned PKM keys) ==="
  CUDA_VISIBLE_DEVICES=0 "$BIN" train-int \
    --corpus-dir "$CORPUS_DIR" --heldout "$HELDOUT_U16" --tokenizer "$TOKENIZER_JSON" --m2-dir "$M2_DIR" \
    --dim 256 --layers 4 --pkm-sub-keys 64 --loss-k 1 --balanced-perm --learn-pkm-keys \
    --epochs 1 --batch 128 --seq-len 256 \
    --lr-q8 4096 --wd-shift 17 --eps 16 --seed 20261127 --auto-sft-below-mbpb 2000000 \
    --out-ckpt "$OUT_DIR/artifacts/int_adamw_brier1_bal.uorbfy01" \
    --out-replay "$OUT_DIR/evidence/int_adamw_brier1_bal_replay_fixture.json" \
    --out-report "$OUT_DIR/evidence/int_adamw_brier1_bal_report.json"
}

run_gpu1() {
  local DEV="$1"
  echo "=== [GPU $DEV] Arm 3: diag_f32_adamw_hb8 (CHARTER §3.7 F32 twin, k=8, identity perm, frozen PKM keys) ==="
  CUDA_VISIBLE_DEVICES="$DEV" "$BIN" train-diag-f32 \
    --corpus-dir "$CORPUS_DIR" --heldout "$HELDOUT_U16" --tokenizer "$TOKENIZER_JSON" \
    --dim 256 --layers 4 --pkm-sub-keys 64 --loss-k 8 --epochs 1 --batch 128 --seq-len 256 \
    --lr-q8 4096 --wd-shift 17 --eps 16 --seed 20261127 \
    --out-report "$OUT_DIR/evidence/diag_f32_adamw_hb8_report.json"

  echo "=== [GPU $DEV] Arm 4: diag_f32_adamw_brier1_bal (CHARTER §3.7 F32 twin, k=1 Brier, TRAIN-balanced perm, learned PKM keys) ==="
  CUDA_VISIBLE_DEVICES="$DEV" "$BIN" train-diag-f32 \
    --corpus-dir "$CORPUS_DIR" --heldout "$HELDOUT_U16" --tokenizer "$TOKENIZER_JSON" \
    --dim 256 --layers 4 --pkm-sub-keys 64 --loss-k 1 --balanced-perm --learn-pkm-keys \
    --epochs 1 --batch 128 --seq-len 256 \
    --lr-q8 4096 --wd-shift 17 --eps 16 --seed 20261127 \
    --out-report "$OUT_DIR/evidence/diag_f32_adamw_brier1_bal_report.json"
}

if [[ "$N_GPUS" -ge 2 ]]; then
  run_gpu0 > "$OUT_DIR/gpu0.log" 2>&1 &
  PID0=$!
  run_gpu1 1 > "$OUT_DIR/gpu1.log" 2>&1 &
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
  run_gpu0
  run_gpu1 0
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
    evidence/int_adamw_hb8_replay_fixture.json \
    evidence/int_adamw_hb8_report.json \
    evidence/int_adamw_brier1_bal_replay_fixture.json \
    evidence/int_adamw_brier1_bal_report.json \
    evidence/diag_f32_adamw_hb8_report.json \
    evidence/diag_f32_adamw_brier1_bal_report.json \
    artifacts/int_adamw_hb8.uorbfy01 \
    artifacts/int_adamw_brier1_bal.uorbfy01 \
    > receipts.sha256
  sha256sum -c receipts.sha256
)
echo "GATE1_DIAG_COMPLETE: $OUT_DIR"
