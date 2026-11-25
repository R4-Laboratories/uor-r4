#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH

test "$#" -eq 2
stage=$(realpath -e "$1")
test "$0" -ef "$stage/dispatch.sh"
cd "$stage"

check() { printf '%s  %s\n' "$2" "$1" | sha256sum --strict -c -; }
field() { jq -er --arg key "$1" '.[$key] | strings | select(length > 0)' GO.json; }

if jq -r '.. | strings' GO.json | grep -qiE 'dev-test|rstar-devtest'; then
    echo "FATAL: DEV-test / R* path forbidden in Stage B2-Fallback" >&2
    exit 99
fi

check GO.json "$2"
test "$(hostname -s)" = "$(field expected_host)"
test "$(field decision)" = GO
test "$(field phase)" = 'Stage B2-Fallback DEV-tune P2 S3 calibration'
test "$(field fit_verdict)" = FIT

check ORDER.md "$(field order_sha256)"
check SHEET.md "$(field sheet_sha256)"
check FIT.md "$(field fit_sha256)"
check dispatch.sh "$(field dispatch_sha256)"

gpu=$(field device_uuid)
cap=$(field expected_compute_cap)
NVCC_ARCH=$(field uor_nvcc_arch)
test "$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader)" = "$cap"

ROOT=$(field run_root)
PARENT_ROOT=$(dirname "$ROOT")
mkdir -p "$PARENT_ROOT"
test ! -e "$ROOT"
test "$(df --output=avail -B1 "$PARENT_ROOT" | awk 'NR==2{print $1}')" -ge 85899345920

BIN=$(field p2_fallback_bin)
CUBIN=$(field cubin_path)
C_UORT=$(field c_uort)
R_UORT=$(field r_uort)
TUNE_DIR=$(field tune_dir)
TRUNK=$(field trunk)
FINITE_C1=$(field finite_c1)
B1A_R_TAPE=$(field b1a_r_tape)
B1A_R_PICK=$(field b1a_r_pick)
B1A_TUNE_TAPE=$(field b1a_tune_tape)
B1A_A1S_DIR=$(field b1a_a1s_dir)

check "$BIN" "$(field p2_fallback_sha256)"
check "$CUBIN" "$(field cubin_sha256)"
check p2-fallback/Cargo.toml "$(field cargo_toml_sha256)"
check p2-fallback/Cargo.lock "$(field cargo_lock_sha256)"
check p2-fallback/build.rs "$(field build_rs_sha256)"
check p2-fallback/src/main.rs "$(field src_main_sha256)"
check p2-fallback/src/fallback.rs "$(field src_fallback_sha256)"
check p2-fallback/src/transport.rs "$(field src_transport_sha256)"
check p2-fallback/src/builder.rs "$(field src_builder_sha256)"
check p2-fallback/src/cuda.rs "$(field src_cuda_sha256)"
check p2-fallback/src/eval.rs "$(field src_eval_sha256)"
check p2-fallback/src/s3_kernels.cu "$(field src_cu_sha256)"

check "$C_UORT" cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a
check "$R_UORT" 5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a
check "$TUNE_DIR/dev-tune.json" 90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467
check "$TUNE_DIR/dev-tune.tokens.u16" 75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a
check "$TUNE_DIR/dev-tune.novel.u8" 7df42437260bb767e2b1bdfd0cc55a242bd6e0b632a5bcd754852a7ab8aca6af
check "$TUNE_DIR/dev-tune.mask.u8" cf359a0ec251aef3a27f18e245699f4cb685b97c53e704857258c8629542c6f6
check "$TUNE_DIR/dev-tune.segment.u8" 202efde41523829869fe0b728f174ab04e98229d8446c17b2190ec926ec4b9b4
check "$TUNE_DIR/id-bytes.u8" befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9
check "$TRUNK" 04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d
check "$FINITE_C1" f9a685845db9ee07c82e6ae66cdbeb2cf7747d14de41193b8e349def3ac543da
check "$B1A_R_TAPE" 8b6b39d25ace5d1dadfe50fd05d8638b6b5b3765784f20e8714f3832935d8c0a
check "$B1A_R_PICK" ba9e646a99ed1507235e95715c6e959469cd830085b6b492b01b2a083769bba9
check "$B1A_TUNE_TAPE" dc0785bf0002a17f9e416235e7b4aeb1308b218574ce6b5145b808244fc4f24d
check "$B1A_A1S_DIR/a1s.weights" e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293
check "$B1A_A1S_DIR/a1s.manifest.json" 26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec
check "$B1A_A1S_DIR/a1s.seal.json" b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071

LOCK=$(field gpu_lock)
if [ "$LOCK" != "none" ]; then
    test -f "$LOCK"
fi

admit_gpu() {
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    test "$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192
    test "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" -ge 10485760
}

run_gpu_guarded() {
    if [ "$LOCK" != "none" ]; then
        (
            exec 8<>"$LOCK"
            flock -n 8 || { echo "GPU lock busy ($1)" >&2; exit 75; }
            admit_gpu
            "${@:2}"
        )
    else
        admit_gpu
        "${@:2}"
    fi
}

export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DISABLE_PTX_JIT=1

mkdir "$ROOT"
mkdir -p "$ROOT/evidence/build" "$ROOT/evidence/verify" "$ROOT/tables" "$ROOT/r_tapes" "$ROOT/tune_tapes" "$ROOT/tmp"
cp GO.json ORDER.md SHEET.md FIT.md "$ROOT/evidence/"

# ---------------- Gate 1: PTX + SASS 0-Float / 0-Tensor-Core Audit ----------------
CUDA_BIN_DIR=/usr/local/cuda/bin
NVCC_BIN=$(command -v nvcc || echo "$CUDA_BIN_DIR/nvcc")
CUOBJDUMP=$(command -v cuobjdump || echo "$CUDA_BIN_DIR/cuobjdump")
"$NVCC_BIN" -ptx -arch="$NVCC_ARCH" -O3 -x cu p2-fallback/src/s3_kernels.cu -o "$ROOT/evidence/s3_kernels.ptx"
PTX_FLOAT_HITS=$(grep -cE '\.(f16|bf16|f32|f64|ftz|rn|rz|rm|rp|approx)' "$ROOT/evidence/s3_kernels.ptx" || true)
"$CUOBJDUMP" --dump-sass "$CUBIN" > "$ROOT/evidence/sass.asm"
FLOAT_RE='\b(FADD|FADD32I|FMUL|FMUL32I|FFMA|FFMA32I|FSET|FSETP|FRND|FCHK|FCMP|FSEL|FMNMX|F2F|F2FP|F2I|F2IP|I2F|I2FP|DADD|DADD32I|DMUL|DMUL32I|DFMA|DSET|DSETP|DMNMX|HADD2|HADD2_32I|HMUL2|HMUL2_32I|HFMA2|HFMA2_32I|HSET2|HSETP2|MUFU|RCP|RSQ|SQRT|EX2|LG2|SIN|COS|HMMA|IMMA|BMMA|DMMA|QMMA)\b'
if [ "$cap" = "12.0" ]; then
    SM120_MOV_ALIASES=$(grep -E "$FLOAT_RE" "$ROOT/evidence/sass.asm" | grep -cE 'HFMA2 R[0-9]+, -RZ, RZ,' || true)
    FLOAT_HITS=$(grep -E "$FLOAT_RE" "$ROOT/evidence/sass.asm" | grep -cvE 'HFMA2 R[0-9]+, -RZ, RZ,' || true)
else
    SM120_MOV_ALIASES=0
    FLOAT_HITS=$(grep -cE "$FLOAT_RE" "$ROOT/evidence/sass.asm" || true)
fi
printf 'cubin_sha256=%s\nnvcc_arch=%s\ncompute_cap=%s\nptx_float_opcode_hits=%s\nfloat_opcode_hits=%s\nsm120_const_mov_aliases=%s\n' \
    "$(field cubin_sha256)" "$NVCC_ARCH" "$cap" "$PTX_FLOAT_HITS" "$FLOAT_HITS" "$SM120_MOV_ALIASES" > "$ROOT/evidence/sass-check.txt"
test "$PTX_FLOAT_HITS" -eq 0
test "$FLOAT_HITS" -eq 0

C_SHA=cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a
R_SHA=5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a
TUNE_SHA=75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a
CUBIN_SHA=$(field cubin_sha256)
ARMS=(vsa hopf h4_600cell golden_shell rand tc_1 tc_2 tc_3 tc_4 tc_6 tc_8 tc_10 tc_12 splitmix)

# ---------------- Gate 2: Finite C128 Parity ----------------
run_gpu_guarded "Gate 2" /usr/bin/time -v -o "$ROOT/evidence/finite.time" "$BIN" finite "$FINITE_C1" "$ROOT/evidence/finite.json"
jq -e --arg cubin "$CUBIN_SHA" '
    .cubin_sha256 == $cubin and
    (.arms | length == 14) and
    ([.arms[] | .divergent == 0 and .bytes <= 8192] | all)
' "$ROOT/evidence/finite.json" >/dev/null

# ---------------- Gate 3: Build All 14 <= 100 MB Tables (<= 180s per arm) ----------------
for arm in "${ARMS[@]}"; do
    t0=$(date +%s)
    run_gpu_guarded "Gate 3 build $arm" timeout 180 /usr/bin/time -v -o "$ROOT/evidence/build/${arm}.time" "$BIN" build "$arm" "$C_UORT" "$C_SHA" \
        "$TRUNK" "$ROOT/tmp/${arm}" 100000000 \
        "$ROOT/tables/${arm}.uors3t02" "$ROOT/evidence/build/${arm}.meta.json"
    elapsed=$(( $(date +%s) - t0 ))
    test "$elapsed" -le 180
    test "$(stat -c %s "$ROOT/tables/${arm}.uors3t02")" -le 100000000
    jq -e --arg arm "$arm" '.arm == $arm and .bytes <= 100000000' "$ROOT/evidence/build/${arm}.meta.json" >/dev/null
    rm -rf "$ROOT/tmp/${arm}"
done

# ---------------- Gate 4: DEV-tune Pick + GPU Capture + Unlocked CPU Verify ----------------
run_gpu_guarded "Gate 4 pick" /usr/bin/time -v -o "$ROOT/evidence/devtune-pick.time" "$BIN" devtune-pick "$TRUNK" "$TUNE_DIR" "$ROOT/devtune-pick.rows"
test "$(stat -c %s "$ROOT/devtune-pick.rows")" -eq $((80 + 16 * 10807417))
test "$(dd if="$ROOT/devtune-pick.rows" bs=8 count=1 status=none)" = "R37PK001"

for arm in "${ARMS[@]}"; do
    run_gpu_guarded "Gate 4 capture $arm" bash -c '
        set -euo pipefail
        "$1" capture "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"
        "$1" capture "$2" "$3" "$4" "${11}" "${12}" "$7" "${13}" "${14}" "${15}"
    ' _ "$BIN" "$arm" "$ROOT/tables/${arm}.uors3t02" "$C_SHA" \
        "$R_UORT" "$R_SHA" "$TRUNK" \
        "$ROOT/r_tapes/${arm}.65536.features" "$ROOT/r_tapes/${arm}.65521.features" "$ROOT/r_tapes/${arm}.cap.json" \
        "$TUNE_DIR/dev-tune.tokens.u16" "$TUNE_SHA" \
        "$ROOT/tune_tapes/${arm}.65536.features" "$ROOT/tune_tapes/${arm}.65521.features" "$ROOT/tune_tapes/${arm}.cap.json"

    cmp "$ROOT/r_tapes/${arm}.65536.features" "$ROOT/r_tapes/${arm}.65521.features"
    cmp "$ROOT/tune_tapes/${arm}.65536.features" "$ROOT/tune_tapes/${arm}.65521.features"
    "$BIN" verify-tape "$arm" "$ROOT/tables/${arm}.uors3t02" "$C_SHA" \
        "$R_UORT" "$R_SHA" "$TRUNK" \
        "$ROOT/r_tapes/${arm}.65536.features" "$ROOT/r_tapes/${arm}.cap.json" \
        "$ROOT/evidence/verify/${arm}.r.json"
    "$BIN" verify-tape "$arm" "$ROOT/tables/${arm}.uors3t02" "$C_SHA" \
        "$TUNE_DIR/dev-tune.tokens.u16" "$TUNE_SHA" "$TRUNK" \
        "$ROOT/tune_tapes/${arm}.65536.features" "$ROOT/tune_tapes/${arm}.cap.json" \
        "$ROOT/evidence/verify/${arm}.tune.json"

    R_TAPE_SHA=$(sha256sum "$ROOT/r_tapes/${arm}.65536.features" | cut -d ' ' -f1)
    T_TAPE_SHA=$(sha256sum "$ROOT/tune_tapes/${arm}.65536.features" | cut -d ' ' -f1)
    jq -e --arg arm "$arm" --arg cubin "$CUBIN_SHA" --arg tape "$R_TAPE_SHA" '
        .arm == $arm and .rows == 15947584 and .probes_checked >= 12500 and
        .divergent_rows == 0 and .cubin_sha256 == $cubin and .tape_sha256 == $tape
    ' "$ROOT/evidence/verify/${arm}.r.json" >/dev/null
    jq -e --arg arm "$arm" --arg cubin "$CUBIN_SHA" --arg tape "$T_TAPE_SHA" '
        .arm == $arm and .rows == 10807417 and .probes_checked >= 12500 and
        .divergent_rows == 0 and .cubin_sha256 == $cubin and .tape_sha256 == $tape
    ' "$ROOT/evidence/verify/${arm}.tune.json" >/dev/null
done

# ---------------- Gate 5: R-Fit EM + DEV-tune Eval + U_geo vs U_count ----------------
/usr/bin/time -v -o "$ROOT/evidence/eval.time" "$BIN" eval \
    "$TRUNK" "$ROOT/r_tapes" "$TUNE_DIR" "$ROOT/tune_tapes" \
    "$B1A_R_TAPE" "$B1A_R_PICK" "$B1A_TUNE_TAPE" "$ROOT/devtune-pick.rows" \
    "$B1A_A1S_DIR/a1s.weights" "$B1A_A1S_DIR/a1s.seal.json" \
    "$ROOT/evidence/devtune-fallback.json"

jq -e '
    .p1_a1s_baseline.all.bits_q12 == "125750699315" and
    .u_count_devtune.bits_q12 == "128614178256" and
    (.arms | length == 14) and
    (.comparisons.fallback_arms | length == 4) and
    ([.arms[] | select(.arm == "vsa" or .arm == "hopf" or .arm == "h4_600cell" or .arm == "golden_shell") | .oracle_u_geo_subset.bits_q12 | length > 0] | all)
' "$ROOT/evidence/devtune-fallback.json" >/dev/null

(
    cd "$ROOT/evidence"
    find . -maxdepth 3 -type f ! -name 'sass.asm' ! -name 's3_kernels.ptx' ! -name 'receipts.sha256' | sort | xargs sha256sum > receipts.sha256
)
echo "STAGE_B2_FALLBACK_COMPLETE"
