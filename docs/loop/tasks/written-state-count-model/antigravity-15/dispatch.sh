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
    "P5B_EVAL_BIN": need_str(go, "p5b_eval_bin"),
    "P5B_EVAL_BIN_SHA": need_str(go, "p5b_eval_bin_sha256"),
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
    "P5B_CARGO_TOML": need_str(go, "p5b_cargo_toml"),
    "P5B_CARGO_TOML_SHA": need_str(go, "p5b_cargo_toml_sha256"),
    "P5B_CARGO_LOCK": need_str(go, "p5b_cargo_lock"),
    "P5B_CARGO_LOCK_SHA": need_str(go, "p5b_cargo_lock_sha256"),
    "P5B_BUILD_RS": need_str(go, "p5b_build_rs"),
    "P5B_BUILD_RS_SHA": need_str(go, "p5b_build_rs_sha256"),
    "P5B_MAIN_RS": need_str(go, "p5b_main_rs"),
    "P5B_MAIN_RS_SHA": need_str(go, "p5b_main_rs_sha256"),
    "P5B_CUDA_RS": need_str(go, "p5b_cuda_rs"),
    "P5B_CUDA_RS_SHA": need_str(go, "p5b_cuda_rs_sha256"),
    "P5B_CACHE_RS": need_str(go, "p5b_cache_eval_rs"),
    "P5B_CACHE_RS_SHA": need_str(go, "p5b_cache_eval_rs_sha256"),
    "P5B_S3_RS": need_str(go, "p5b_s3_decode_rs"),
    "P5B_S3_RS_SHA": need_str(go, "p5b_s3_decode_rs_sha256"),
    "P5B_DEVTEST_RS": need_str(go, "p5b_devtest_eval_rs"),
    "P5B_DEVTEST_RS_SHA": need_str(go, "p5b_devtest_eval_rs_sha256"),
    "P5B_REPLY_RS": need_str(go, "p5b_reply_verify_rs"),
    "P5B_REPLY_RS_SHA": need_str(go, "p5b_reply_verify_rs_sha256"),
    "P5B_KERNELS_CU": need_str(go, "p5b_kernels_cu"),
    "P5B_KERNELS_CU_SHA": need_str(go, "p5b_kernels_cu_sha256"),
    "DEVTEST_DIR": need_str(go, "devtest_dir"),
    "DEVTEST_CENSUS_SHA": need_str(go, "devtest_census_sha256"),
    "DEVTEST_TAPE": need_str(go, "devtest_tape"),
    "DEVTEST_TAPE_SHA": need_str(go, "devtest_tape_sha256"),
    "RSTAR_MASSES": need_str(go, "rstar_masses"),
    "RSTAR_MASSES_SHA": need_str(go, "rstar_masses_sha256"),
    "R_UORT": need_str(go, "r_uort"),
    "R_UORT_SHA": need_str(go, "r_uort_sha256"),
    "B1A_R_TAPE": need_str(go, "b1a_r_tape"),
    "B1A_R_PICK": need_str(go, "b1a_r_pick"),
    "COMPACT_V3": need_str(go, "compact_v3"),
    "COMPACT_V3_SHA": need_str(go, "compact_v3_sha256"),
    "A1S_WEIGHTS": need_str(go, "a1s_weights"),
    "A1S_WEIGHTS_SHA": need_str(go, "a1s_weights_sha256"),
    "A1S_SEAL": need_str(go, "a1s_seal"),
    "A1S_SEAL_SHA": need_str(go, "a1s_seal_sha256"),
    "GOLDEN_SHELL": need_str(go, "golden_shell"),
    "GOLDEN_SHELL_SHA": need_str(go, "golden_shell_sha256"),
    "S2_SKIP": need_str(go, "s2_skip"),
    "S2_SKIP_SHA": need_str(go, "s2_skip_sha256"),
    "S4_UORS4P01": need_str(go, "s4_uors4p01"),
    "S4_UORS4P01_SHA": need_str(go, "s4_uors4p01_sha256"),
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

# L-002 guard
if [[ -e "$RUN_DIR" ]]; then
  echo "REFUSED (L-002): RUN_DIR already exists ($RUN_DIR)" >&2
  exit 2
fi

WICK_LOCK="/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock"
if [[ -e "$WICK_LOCK" ]]; then
  exec 9<"$WICK_LOCK"
  flock -n 9 || {
    echo "REFUSED: Wick GPU lock held ($WICK_LOCK)" >&2
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
  [[ -f "$path" ]] || { echo "missing $label: $path" >&2; exit 1; }
  local got
  got="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$got" != "$want" ]]; then
    echo "SHA-256 mismatch for $label ($path): got $got want $want" >&2
    exit 1
  fi
  CHECKS=$((CHECKS + 1))
}

mkdir -p "$RUN_DIR" "$RUN_DIR/artifacts"

GPU_INFO="$(nvidia-smi --query-gpu=compute_cap,utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)"
GPU_CAP="$(printf '%s' "$GPU_INFO" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
GPU_UTIL="$(printf '%s' "$GPU_INFO" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
GPU_FREE_MIB="$(printf '%s' "$GPU_INFO" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
if [[ "$GPU_CAP" != "8.9" ]]; then
  echo "REFUSED: expected sm_89 compute_cap 8.9, got '$GPU_CAP'" >&2
  exit 1
fi
if [[ "$GPU_UTIL" -gt 15 || "$GPU_FREE_MIB" -lt 4096 ]]; then
  echo "REFUSED: GPU busy/low memory ($GPU_INFO)" >&2
  exit 75
fi
CHECKS=$((CHECKS + 1))

[[ -x "$P5B_EVAL_BIN" ]] || { echo "missing executable $P5B_EVAL_BIN" >&2; exit 1; }
verify_sha "p5b_eval_bin" "$P5B_EVAL_BIN" "$P5B_EVAL_BIN_SHA"
verify_sha "cubin_path" "$CUBIN_PATH" "$CUBIN_SHA"
verify_sha "ORDER.md" "$ORDER_MD" "$ORDER_SHA"
verify_sha "SHEET.md" "$SHEET_MD" "$SHEET_SHA"
verify_sha "FIT.md" "$FIT_MD" "$FIT_SHA"
verify_sha "dispatch.sh" "$DISPATCH_SH" "$DISPATCH_SHA"
verify_sha "p5b-eval/Cargo.toml" "$P5B_CARGO_TOML" "$P5B_CARGO_TOML_SHA"
verify_sha "p5b-eval/Cargo.lock" "$P5B_CARGO_LOCK" "$P5B_CARGO_LOCK_SHA"
verify_sha "p5b-eval/build.rs" "$P5B_BUILD_RS" "$P5B_BUILD_RS_SHA"
verify_sha "p5b-eval/src/main.rs" "$P5B_MAIN_RS" "$P5B_MAIN_RS_SHA"
verify_sha "p5b-eval/src/cuda.rs" "$P5B_CUDA_RS" "$P5B_CUDA_RS_SHA"
verify_sha "p5b-eval/src/cache_eval.rs" "$P5B_CACHE_RS" "$P5B_CACHE_RS_SHA"
verify_sha "p5b-eval/src/s3_decode.rs" "$P5B_S3_RS" "$P5B_S3_RS_SHA"
verify_sha "p5b-eval/src/devtest_eval.rs" "$P5B_DEVTEST_RS" "$P5B_DEVTEST_RS_SHA"
verify_sha "p5b-eval/src/reply_verify.rs" "$P5B_REPLY_RS" "$P5B_REPLY_RS_SHA"
verify_sha "p5b-eval/src/p5b_kernels.cu" "$P5B_KERNELS_CU" "$P5B_KERNELS_CU_SHA"

verify_sha "dev-test.json" "$DEVTEST_DIR/dev-test.json" "$DEVTEST_CENSUS_SHA"
verify_sha "devtest_tape" "$DEVTEST_TAPE" "$DEVTEST_TAPE_SHA"
verify_sha "rstar_masses" "$RSTAR_MASSES" "$RSTAR_MASSES_SHA"
verify_sha "r_uort" "$R_UORT" "$R_UORT_SHA"
verify_sha "compact_v3" "$COMPACT_V3" "$COMPACT_V3_SHA"
verify_sha "a1s_weights" "$A1S_WEIGHTS" "$A1S_WEIGHTS_SHA"
verify_sha "a1s_seal" "$A1S_SEAL" "$A1S_SEAL_SHA"
verify_sha "golden_shell" "$GOLDEN_SHELL" "$GOLDEN_SHELL_SHA"
verify_sha "s2_skip" "$S2_SKIP" "$S2_SKIP_SHA"
verify_sha "s4_uors4p01" "$S4_UORS4P01" "$S4_UORS4P01_SHA"
verify_sha "trunk" "$TRUNK" "$TRUNK_SHA"
verify_sha "tokenizer_json" "$TOKENIZER_JSON" "$TOKENIZER_SHA"
verify_sha "fresh256/plan.json" "$FRESH256_DIR/plan.json" "$FRESH256_PLAN_SHA"
verify_sha "fresh256/inputs.json" "$FRESH256_DIR/inputs.json" "$FRESH256_INPUTS_SHA"
verify_sha "fresh256/labels.json" "$FRESH256_DIR/labels.json" "$FRESH256_LABELS_SHA"
verify_sha "fresh256/manifest.json" "$FRESH256_DIR/manifest.json" "$FRESH256_MANIFEST_SHA"

python3 - "$FIT_MD" "$ORDER_SHA" "$SHEET_SHA" "$DISPATCH_SHA" "$P5B_CARGO_TOML_SHA" "$P5B_CARGO_LOCK_SHA" "$P5B_BUILD_RS_SHA" "$P5B_MAIN_RS_SHA" "$P5B_CUDA_RS_SHA" "$P5B_CACHE_RS_SHA" "$P5B_S3_RS_SHA" "$P5B_DEVTEST_RS_SHA" "$P5B_REPLY_RS_SHA" "$P5B_KERNELS_CU_SHA" <<'PY'
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
nvcc -ptx -arch=sm_89 -O3 -x cu "$P5B_KERNELS_CU" -o "$RUN_DIR/p5b_kernels.ptx"
cuobjdump --dump-sass "$CUBIN_PATH" > "$RUN_DIR/p5b_kernels.sass"
for k in p3_brier_step_batch p3_em_step_batch p3_s3_golden_lookup_and_accum p3_s2_lookup_and_accum p3_s4_vote_batch; do
  test "$(grep -c "\.entry $k" "$RUN_DIR/p5b_kernels.ptx")" -ge 1
  test "$(grep -c "$k" "$RUN_DIR/p5b_kernels.sass")" -ge 1
done
PTX_HITS="$(
  grep -vE '^[[:space:]]*(//|\.(version|target|address_size|file|loc))' "$RUN_DIR/p5b_kernels.ptx" \
    | grep -Eic '\.(f16|f32|f64|bf16)\b' || true
)"
SASS_HITS="$(
  grep -Eic '\b(FADD|FADD32I|FMUL|FMUL32I|FFMA|FFMA32I|FMA|FSET|FSETP|FRND|FCMP|FSEL|FMNMX|MUFU|DADD|DADD32I|DMUL|DMUL32I|DFMA|DSET|DSETP|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMMA|IMMA|BMMA|WMMA|QMMA|DMMA|F2I|F2IP|I2F|I2FP|F2F|F2FP|FCHK|FSWZADD|RRO)\b' "$RUN_DIR/p5b_kernels.sass" || true
)"
{
  echo "cubin_path=$CUBIN_PATH"
  echo "cubin_sha256=$CUBIN_SHA"
  echo "ptx_float_opcode_hits=$PTX_HITS"
  echo "float_opcode_hits=$SASS_HITS"
} > "$RUN_DIR/sass-check.txt"
if [[ "$PTX_HITS" -ne 0 || "$SASS_HITS" -ne 0 ]]; then
  cat "$RUN_DIR/sass-check.txt" >&2
  exit 1
fi
CHECKS=$((CHECKS + 1))

echo "[Gate 2] Finite GPU vs CPU parity (525 probes) + cubin SHA bind..."
timeout 120 /usr/bin/time -v -o "$RUN_DIR/finite.time" \
  "$P5B_EVAL_BIN" finite "$RUN_DIR/finite.json"
python3 - "$RUN_DIR/finite.json" "$CUBIN_SHA" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["cubin_sha256"] == sys.argv[2], d
assert d["rows_checked"] == 525, d
assert d["divergent"] == 0, d
PY
CHECKS=$((CHECKS + 1))

echo "[Gate 3] Stage B4 replay verify + single-shot sealed DEV-test P3 scoring..."
timeout 2400 /usr/bin/time -v -o "$RUN_DIR/devtest.time" \
  "$P5B_EVAL_BIN" devtest \
  "$DEVTEST_DIR" "$R_UORT" "$TRUNK" "$B1A_R_TAPE" "$B1A_R_PICK" \
  "$DEVTEST_TAPE" "$RSTAR_MASSES" "$A1S_WEIGHTS" "$A1S_SEAL" \
  "$GOLDEN_SHELL" "$S2_SKIP" "$RUN_DIR/artifacts" "$RUN_DIR/devtest_report.json"

python3 - "$RUN_DIR/devtest_report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["spot_check_divergences"] == 0 and d["s3_spot_check_divergences"] == 0 and d["s2_spot_check_divergences"] == 0, d
assert d["shifted_batch_parity"] is True, d
assert d["pick_probes_checked"] >= 13000, d
assert d["spot_check_probes"] >= 25000, d
assert int(d["p1_a1s_baseline"]["all"]["bits_q12"]) == 246099676775, d
assert int(d["p1_a1s_baseline"]["novel"]["bits_q12"]) == 215795693587, d
for a in d["arms"]:
    if a["arm"] == "p4_full_errwrite":
        print(f"DEV-test p4_full_errwrite: micro_bpb={a['all']['bpb_micro']} pct_P1={a['pct_of_p1_bps']}bps vs_R*={a['ratio_vs_rstar_bps']}bps gate_1pct={a['gate_brier_1pct_fewer_bits']}")
print(f"Controlled errwrite - cache_static Q12 CI: {d['controlled_error_write_deltas']['p4_full_errwrite_minus_cache_static_all_q12']}")
PY
CHECKS=$((CHECKS + 1))

echo "[Gate 4] Cold-loaded Stage B5a S4 verification..."
timeout 600 /usr/bin/time -v -o "$RUN_DIR/reply_verify.time" \
  "$P5B_EVAL_BIN" reply-verify \
  "$S4_UORS4P01" "$COMPACT_V3" "$A1S_WEIGHTS" "$A1S_SEAL" "$TRUNK" "$TOKENIZER_JSON" "$FRESH256_DIR" "$RUN_DIR/reply_verify.json"
python3 - "$RUN_DIR/reply_verify.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["s4_gpu_vs_cpu_divergences"] == 0, r
print(f"Cold S4 reply_verify: R1={r['r1_r2_m2_fresh256']['active_pairs']}/128 R2={r['r1_r2_m2_fresh256']['active_row_hits']}/256 R4={r['r4_heldout_phrasing']['active_row_hits']}/240 Shuf={r['r4_heldout_phrasing']['shuffled_role_row_hits']}/240")
PY
CHECKS=$((CHECKS + 1))

(
  cd "$RUN_DIR"
  sha256sum \
    p5b_kernels.ptx \
    sass-check.txt \
    finite.json \
    finite.time \
    devtest_report.json \
    devtest.time \
    reply_verify.json \
    reply_verify.time \
    artifacts/p4_c_skip_static.uors6b01 \
    artifacts/p4_c_skip_errwrite.uors6b01 \
    artifacts/p4_cache_static.uors6b01 \
    artifacts/p4_full_errwrite.uors6b01 \
    artifacts/p5b_s3_err_p.i8 \
    artifacts/p5b_s3_err_b.i8 \
    artifacts/p5b_s2_skip_errs.i8 > receipts.sha256
  sha256sum --strict -c receipts.sha256
)
echo "checks_verified=$CHECKS"
echo "STAGE_B5B_SEALED_DEVTEST_COMPLETE"
