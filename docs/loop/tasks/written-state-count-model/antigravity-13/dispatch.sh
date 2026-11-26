#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: dispatch.sh <go.json>" >&2
  exit 2
fi

GO_JSON="$1"
if [[ ! -f "$GO_JSON" ]]; then
  echo "missing GO.json: $GO_JSON" >&2
  exit 2
fi

# Hard lock: Stage B5 must never touch sealed DEV-test.
if grep -Eiq 'dev-test|rstar-devtest|023784b7|ec048217' "$GO_JSON"; then
  echo "REFUSED (exit 99): sealed DEV-test pattern detected in $GO_JSON" >&2
  exit 99
fi

eval "$(
  python3 - "$GO_JSON" <<'PY'
import json, shlex, sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    go = json.load(fh)

def need_str(obj, key):
    val = obj.get(key)
    if not isinstance(val, str) or not val:
        raise SystemExit(f"GO.json missing string field: {key}")
    return val

fields = {
    "RUN_DIR": need_str(go, "run_dir"),
    "P5_EVAL_BIN": need_str(go, "p5_eval_bin"),
    "P5_EVAL_BIN_SHA": need_str(go, "p5_eval_bin_sha256"),
    "CUBIN_PATH": need_str(go, "cubin_path"),
    "CUBIN_SHA": need_str(go, "cubin_sha256"),
    "ORDER_MD": need_str(go, "order_md"),
    "ORDER_SHA": need_str(go, "order_sha256"),
    "SHEET_MD": need_str(go, "sheet_md"),
    "SHEET_SHA": need_str(go, "sheet_sha256"),
    "FIT_MD": need_str(go, "fit_md"),
    "FIT_SHA": need_str(go, "fit_sha256"),
    "DISPATCH_SH": need_str(go, "dispatch_sh"),
    "DISPATCH_SHA": need_str(go, "dispatch_sha256"),
    "P5_CARGO_TOML": need_str(go, "p5_cargo_toml"),
    "P5_CARGO_TOML_SHA": need_str(go, "p5_cargo_toml_sha256"),
    "P5_CARGO_LOCK": need_str(go, "p5_cargo_lock"),
    "P5_CARGO_LOCK_SHA": need_str(go, "p5_cargo_lock_sha256"),
    "P5_BUILD_RS": need_str(go, "p5_build_rs"),
    "P5_BUILD_RS_SHA": need_str(go, "p5_build_rs_sha256"),
    "P5_MAIN_RS": need_str(go, "p5_main_rs"),
    "P5_MAIN_RS_SHA": need_str(go, "p5_main_rs_sha256"),
    "P5_CUDA_RS": need_str(go, "p5_cuda_rs"),
    "P5_CUDA_RS_SHA": need_str(go, "p5_cuda_rs_sha256"),
    "P5_C_PPMI_RS": need_str(go, "p5_c_ppmi_rs"),
    "P5_C_PPMI_RS_SHA": need_str(go, "p5_c_ppmi_rs_sha256"),
    "P5_REPLY_EVAL_RS": need_str(go, "p5_reply_eval_rs"),
    "P5_REPLY_EVAL_RS_SHA": need_str(go, "p5_reply_eval_rs_sha256"),
    "P5_KERNELS_CU": need_str(go, "p5_kernels_cu"),
    "P5_KERNELS_CU_SHA": need_str(go, "p5_kernels_cu_sha256"),
    "C_UORT": need_str(go, "c_uort"),
    "C_UORT_SHA": need_str(go, "c_uort_sha256"),
    "COMPACT_V3": need_str(go, "compact_v3"),
    "COMPACT_V3_SHA": need_str(go, "compact_v3_sha256"),
    "R_UORT": need_str(go, "r_uort"),
    "R_UORT_SHA": need_str(go, "r_uort_sha256"),
    "A1S_WEIGHTS": need_str(go, "a1s_weights"),
    "A1S_WEIGHTS_SHA": need_str(go, "a1s_weights_sha256"),
    "A1S_SEAL": need_str(go, "a1s_seal"),
    "A1S_SEAL_SHA": need_str(go, "a1s_seal_sha256"),
    "TRUNK": need_str(go, "trunk"),
    "TRUNK_SHA": need_str(go, "trunk_sha256"),
    "TOKENIZER_JSON": need_str(go, "tokenizer_json"),
    "TOKENIZER_SHA": need_str(go, "tokenizer_sha256"),
    "FRESH256_DIR": need_str(go, "fresh256_dir"),
    "FRESH256_PLAN_SHA": need_str(go, "fresh256_plan_sha256"),
    "FRESH256_INPUTS_SHA": need_str(go, "fresh256_inputs_sha256"),
    "FRESH256_LABELS_SHA": need_str(go, "fresh256_labels_sha256"),
    "FRESH256_MANIFEST_SHA": need_str(go, "fresh256_manifest_sha256"),
}

for k, v in fields.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"

# L-002 guard: refuse to write into a pre-existing run directory.
if [[ -e "$RUN_DIR" ]]; then
  echo "REFUSED (L-002): RUN_DIR already exists ($RUN_DIR); use a fresh output root" >&2
  exit 2
fi

# Wick lock discipline when running on Wick host.
WICK_LOCK="/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock"
if [[ -e "$WICK_LOCK" ]]; then
  exec 9<"$WICK_LOCK"
  flock -n 9 || {
    echo "REFUSED: Wick GPU lock is held ($WICK_LOCK)" >&2
    exit 75
  }
fi

export CUDA_DISABLE_PTX_JIT=1
export PATH="/usr/local/cuda/bin:$PATH"

CHECKS=0

verify_sha() {
  local label="$1"
  local path="$2"
  local want="$3"
  if [[ ! -f "$path" ]]; then
    echo "missing file for $label: $path" >&2
    exit 1
  fi
  local got
  got="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$got" != "$want" ]]; then
    echo "SHA-256 mismatch for $label ($path): got $got want $want" >&2
    exit 1
  fi
  CHECKS=$((CHECKS + 1))
}

mkdir -p "$RUN_DIR" "$RUN_DIR/artifacts"

echo "[Stage B5] Verifying GPU compute capability (sm_89), binaries, and frozen inputs..."
GPU_INFO="$(nvidia-smi --query-gpu=compute_cap,utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)"
GPU_CAP="$(printf '%s' "$GPU_INFO" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
GPU_UTIL="$(printf '%s' "$GPU_INFO" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
GPU_FREE_MIB="$(printf '%s' "$GPU_INFO" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
if [[ "$GPU_CAP" != "8.9" ]]; then
  echo "REFUSED: expected compute_cap 8.9, got '$GPU_CAP' ($GPU_INFO)" >&2
  exit 1
fi
if [[ "$GPU_UTIL" -gt 15 || "$GPU_FREE_MIB" -lt 4096 ]]; then
  echo "REFUSED: GPU not ready (util=${GPU_UTIL}%, free_mib=${GPU_FREE_MIB})" >&2
  exit 75
fi
CHECKS=$((CHECKS + 1))

[[ -x "$P5_EVAL_BIN" ]] || { echo "missing executable P5_EVAL_BIN: $P5_EVAL_BIN" >&2; exit 1; }
verify_sha "p5_eval_bin" "$P5_EVAL_BIN" "$P5_EVAL_BIN_SHA"
verify_sha "cubin_path" "$CUBIN_PATH" "$CUBIN_SHA"

verify_sha "ORDER.md" "$ORDER_MD" "$ORDER_SHA"
verify_sha "SHEET.md" "$SHEET_MD" "$SHEET_SHA"
verify_sha "FIT.md" "$FIT_MD" "$FIT_SHA"
verify_sha "dispatch.sh" "$DISPATCH_SH" "$DISPATCH_SHA"
verify_sha "p5-eval/Cargo.toml" "$P5_CARGO_TOML" "$P5_CARGO_TOML_SHA"
verify_sha "p5-eval/Cargo.lock" "$P5_CARGO_LOCK" "$P5_CARGO_LOCK_SHA"
verify_sha "p5-eval/build.rs" "$P5_BUILD_RS" "$P5_BUILD_RS_SHA"
verify_sha "p5-eval/src/main.rs" "$P5_MAIN_RS" "$P5_MAIN_RS_SHA"
verify_sha "p5-eval/src/cuda.rs" "$P5_CUDA_RS" "$P5_CUDA_RS_SHA"
verify_sha "p5-eval/src/c_ppmi.rs" "$P5_C_PPMI_RS" "$P5_C_PPMI_RS_SHA"
verify_sha "p5-eval/src/reply_eval.rs" "$P5_REPLY_EVAL_RS" "$P5_REPLY_EVAL_RS_SHA"
verify_sha "p5-eval/src/p5_s4_kernels.cu" "$P5_KERNELS_CU" "$P5_KERNELS_CU_SHA"

verify_sha "c_uort" "$C_UORT" "$C_UORT_SHA"
verify_sha "compact_v3" "$COMPACT_V3" "$COMPACT_V3_SHA"
verify_sha "r_uort" "$R_UORT" "$R_UORT_SHA"
verify_sha "a1s_weights" "$A1S_WEIGHTS" "$A1S_WEIGHTS_SHA"
verify_sha "a1s_seal" "$A1S_SEAL" "$A1S_SEAL_SHA"
verify_sha "trunk" "$TRUNK" "$TRUNK_SHA"
verify_sha "tokenizer_json" "$TOKENIZER_JSON" "$TOKENIZER_SHA"
verify_sha "fresh256/plan.json" "$FRESH256_DIR/plan.json" "$FRESH256_PLAN_SHA"
verify_sha "fresh256/inputs.json" "$FRESH256_DIR/inputs.json" "$FRESH256_INPUTS_SHA"
verify_sha "fresh256/labels.json" "$FRESH256_DIR/labels.json" "$FRESH256_LABELS_SHA"
verify_sha "fresh256/manifest.json" "$FRESH256_DIR/manifest.json" "$FRESH256_MANIFEST_SHA"

# Verify embedded FIT SHA-256 table matches every file listed above.
python3 - "$FIT_MD" "$ORDER_SHA" "$SHEET_SHA" "$DISPATCH_SHA" "$P5_CARGO_TOML_SHA" "$P5_CARGO_LOCK_SHA" "$P5_BUILD_RS_SHA" "$P5_MAIN_RS_SHA" "$P5_CUDA_RS_SHA" "$P5_C_PPMI_RS_SHA" "$P5_REPLY_EVAL_RS_SHA" "$P5_KERNELS_CU_SHA" <<'PY'
import sys
fit_text = open(sys.argv[1], "r", encoding="utf-8").read()
if "TEST FITNESS: FIT" not in fit_text:
    raise SystemExit("FIT.md missing 'TEST FITNESS: FIT'")
for h in sys.argv[2:]:
    if h not in fit_text:
        raise SystemExit(f"FIT.md missing expected SHA-256: {h}")
PY
CHECKS=$((CHECKS + 1))

echo "[Gate 1] Non-vacuous PTX + SASS zero-float / zero-MMA audit..."
nvcc -ptx -arch=sm_89 -O3 -x cu "$P5_KERNELS_CU" -o "$RUN_DIR/p5_s4_kernels.ptx"
cuobjdump --dump-sass "$CUBIN_PATH" > "$RUN_DIR/p5_s4_kernels.sass"
test "$(grep -c '\.entry p3_s4_vote_batch' "$RUN_DIR/p5_s4_kernels.ptx")" -ge 1
test "$(grep -c 'p3_s4_vote_batch' "$RUN_DIR/p5_s4_kernels.sass")" -ge 1

PTX_HITS="$(
  grep -vE '^[[:space:]]*(//|\.(version|target|address_size|file|loc))' "$RUN_DIR/p5_s4_kernels.ptx" \
    | grep -Eic '\.(f16|f32|f64|bf16)\b' || true
)"
SASS_HITS="$(
  grep -Eic '\b(FADD|FADD32I|FMUL|FMUL32I|FFMA|FFMA32I|FMA|FSET|FSETP|FRND|FCMP|FSEL|FMNMX|MUFU|DADD|DADD32I|DMUL|DMUL32I|DFMA|DSET|DSETP|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMMA|IMMA|BMMA|WMMA|QMMA|DMMA|F2I|F2IP|I2F|I2FP|F2F|F2FP|FCHK|FSWZADD|RRO)\b' "$RUN_DIR/p5_s4_kernels.sass" || true
)"
{
  echo "cubin_path=$CUBIN_PATH"
  echo "cubin_sha256=$CUBIN_SHA"
  echo "ptx_float_opcode_hits=$PTX_HITS"
  echo "float_opcode_hits=$SASS_HITS"
} > "$RUN_DIR/sass-check.txt"
if [[ "$PTX_HITS" -ne 0 || "$SASS_HITS" -ne 0 ]]; then
  cat "$RUN_DIR/sass-check.txt" >&2
  echo "Gate 1 FAILED: float/MMA opcodes detected" >&2
  exit 1
fi
CHECKS=$((CHECKS + 1))

echo "[Gate 2] Finite GPU vs CPU parity (64 probes) + embedded cubin SHA-256 bind..."
timeout 60 /usr/bin/time -v -o "$RUN_DIR/finite.time" \
  "$P5_EVAL_BIN" finite "$RUN_DIR/finite.json"
python3 - "$RUN_DIR/finite.json" "$CUBIN_SHA" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert d["cubin_sha256"] == sys.argv[2], (d["cubin_sha256"], sys.argv[2])
assert d["s4_checked"] == 64, d
assert d["s4_divergent"] == 0, d
assert d["divergent"] == 0, d
PY
CHECKS=$((CHECKS + 1))

echo "[Gate 3] Full C-corpus PPMI S4 build, hinge training, and GPU-voted evaluation..."
timeout 2400 /usr/bin/time -v -o "$RUN_DIR/reply.time" \
  "$P5_EVAL_BIN" reply \
  "$C_UORT" \
  "$COMPACT_V3" \
  "$A1S_WEIGHTS" \
  "$A1S_SEAL" \
  "$TRUNK" \
  "$TOKENIZER_JSON" \
  "$FRESH256_DIR" \
  "$R_UORT" \
  "$RUN_DIR/artifacts" \
  "$RUN_DIR/reply_report.json"

test -f "$RUN_DIR/artifacts/c_ppmi_s4.uors4p01"
CHECKS=$((CHECKS + 1))

python3 - "$RUN_DIR/reply_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert r["s4_gpu_vs_cpu_divergences"] == 0, r["s4_gpu_vs_cpu_divergences"]
art = r["s4_artifact"]
assert art["c_tokens_scanned"] == 2075608357, art["c_tokens_scanned"]
assert art["mworld_spans_extracted"] == 24, art["mworld_spans_extracted"]
assert art["total_hinge_updates"] > 0, art["total_hinge_updates"]

f = r["r1_r2_m2_fresh256"]
assert f["rows"] == 256, f["rows"]
assert f["pairs"] == 128, f["pairs"]

r4 = r["r4_heldout_phrasing"]
assert r4["rows"] == 240, r4["rows"]
assert r4["pairs"] == 120, r4["pairs"]

print(
    f"Stage B5 metrics: R1={f['active']['pairs_both_complete_distinct']}/{f['pairs']} "
    f"R2={f['active']['row_hits']}/{f['rows']} | "
    f"R4={r4['active_row_hits']}/{r4['rows']} (stem={r4['active_in_stem_rows_of_120']}/120, "
    f"oov={r4['active_oov_synonym_rows_of_120']}/120) | "
    f"R4_NoMem={r4['no_memory_row_hits']}/{r4['rows']} R4_Shuf={r4['shuffled_role_row_hits']}/{r4['rows']}"
)
PY
CHECKS=$((CHECKS + 1))

(
  cd "$RUN_DIR"
  sha256sum \
    p5_s4_kernels.ptx \
    sass-check.txt \
    finite.json \
    finite.time \
    reply_report.json \
    reply.time \
    artifacts/c_ppmi_s4.uors4p01 > receipts.sha256
  sha256sum --strict -c receipts.sha256
)
CHECKS=$((CHECKS + 1))

echo "checks_verified=$CHECKS"
echo "STAGE_B5_P3_S4_COMPLETE"
