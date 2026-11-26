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

if jq -r '.. | strings' GO.json | grep -qiE 'dev-test|rstar-devtest|023784b7|ec048217'; then
    echo "FATAL: DEV-test / R* path or sealed digest forbidden in Stage B4" >&2
    exit 99
fi

check GO.json "$2"
test "$(hostname -s)" = "$(field expected_host)"
test "$(field decision)" = GO
test "$(field phase)" = 'Stage B4 P3 Disjoint-C S2-skip, Causal Session Cache, and 2-Hop S4'
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
test "$(df --output=avail -B1 "$PARENT_ROOT" | awk 'NR==2{print $1}')" -ge 21474836480

BIN=$(field p4_eval_bin)
CUBIN=$(field cubin_path)
C_UORT=$(field c_uort)
COMPACT_V3=$(field compact_v3)
R_UORT=$(field r_uort)
TUNE_DIR=$(field tune_dir)
TRUNK=$(field trunk)
B1A_R_TAPE=$(field b1a_r_tape)
B1A_R_PICK=$(field b1a_r_pick)
B1A_TUNE_TAPE=$(field b1a_tune_tape)
B2_TUNE_PICK=$(field b2_tune_pick)
A1S_DIR=$(field a1s_dir)
GOLDEN_SHELL=$(field golden_shell)
TOKENIZER=$(field tokenizer_json)
FRESH256_DIR=$(field fresh256_dir)

check "$BIN" "$(field p4_eval_sha256)"
check "$CUBIN" "$(field cubin_sha256)"
check p4-eval/Cargo.toml "$(field cargo_toml_sha256)"
check p4-eval/Cargo.lock "$(field cargo_lock_sha256)"
check p4-eval/build.rs "$(field build_rs_sha256)"
check p4-eval/src/main.rs "$(field src_main_sha256)"
check p4-eval/src/cuda.rs "$(field src_cuda_sha256)"
check p4-eval/src/cache_eval.rs "$(field src_cache_eval_sha256)"
check p4-eval/src/s3_decode.rs "$(field src_s3_decode_sha256)"
check p4-eval/src/brier_eval.rs "$(field src_brier_eval_sha256)"
check p4-eval/src/reply_eval.rs "$(field src_reply_eval_sha256)"
check p4-eval/src/p4_brier_kernels.cu "$(field src_cu_sha256)"

check "$C_UORT" cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a
check "$COMPACT_V3" 75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233
check "$R_UORT" 5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a
check "$TUNE_DIR/dev-tune.json" 90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467
check "$TUNE_DIR/dev-tune.tokens.u16" 75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a
check "$TUNE_DIR/dev-tune.novel.u8" 7df42437260bb767e2b1bdfd0cc55a242bd6e0b632a5bcd754852a7ab8aca6af
check "$TUNE_DIR/dev-tune.mask.u8" cf359a0ec251aef3a27f18e245699f4cb685b97c53e704857258c8629542c6f6
check "$TUNE_DIR/dev-tune.segment.u8" 202efde41523829869fe0b728f174ab04e98229d8446c17b2190ec926ec4b9b4
check "$TUNE_DIR/id-bytes.u8" befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9
check "$TRUNK" 04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d
check "$B1A_R_TAPE" 8b6b39d25ace5d1dadfe50fd05d8638b6b5b3765784f20e8714f3832935d8c0a
check "$B1A_R_PICK" ba9e646a99ed1507235e95715c6e959469cd830085b6b492b01b2a083769bba9
check "$B1A_TUNE_TAPE" dc0785bf0002a17f9e416235e7b4aeb1308b218574ce6b5145b808244fc4f24d
check "$B2_TUNE_PICK" e2e3cf84db615517c84724700396fa6da2e4861fe92a5371149a0e593d3e5496
check "$A1S_DIR/a1s.weights" e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293
check "$A1S_DIR/a1s.manifest.json" 26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec
check "$A1S_DIR/a1s.seal.json" b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071
check "$GOLDEN_SHELL" e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c
check "$TOKENIZER" d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89
check "$FRESH256_DIR/plan.json" 0ed1ab98ab50bbad6c484c4e7e4ffeeb8bb7ad17d162251826302efe86d75715
check "$FRESH256_DIR/inputs.json" 1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a
check "$FRESH256_DIR/labels.json" 3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be
check "$FRESH256_DIR/manifest.json" 7c33f3f7e19edcb4b1a7aa5e5eab3bb0056db9351067d809f62f97b4a8880964

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
mkdir -p "$ROOT/evidence/artifacts"
cp GO.json ORDER.md SHEET.md FIT.md "$ROOT/evidence/"

# ---------------- Gate 1: PTX + SASS 0-Float / 0-Tensor-Core Audit ----------------
CUDA_BIN_DIR=/usr/local/cuda/bin
NVCC_BIN=$(command -v nvcc || echo "$CUDA_BIN_DIR/nvcc")
CUOBJDUMP=$(command -v cuobjdump || echo "$CUDA_BIN_DIR/cuobjdump")
"$NVCC_BIN" -ptx -arch="$NVCC_ARCH" -O3 -x cu p4-eval/src/p4_brier_kernels.cu -o "$ROOT/evidence/p4_brier_kernels.ptx"
PTX_FLOAT_HITS=$(grep -cE '\.(f16|bf16|f32|f64|ftz|rn|rz|rm|rp|approx)' "$ROOT/evidence/p4_brier_kernels.ptx" || true)
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

CUBIN_SHA=$(field cubin_sha256)

# ---------------- Gate 2: Multi-Kernel Finite GPU vs CPU Bit-Exactness ----------------
run_gpu_guarded "Gate 2 finite" /usr/bin/time -v -o "$ROOT/evidence/finite.time" "$BIN" finite "$ROOT/evidence/finite.json"
jq -e --arg cubin "$CUBIN_SHA" '
    .cubin_sha256 == $cubin and
    .brier_checked == 256 and .brier_divergent == 0 and
    .em_checked == 256 and .em_divergent == 0 and
    .s3_checked >= 5 and .s3_divergent == 0 and
    .s2_checked >= 4 and .s2_divergent == 0 and
    .s4_checked >= 4 and .s4_divergent == 0 and
    .divergent == 0
' "$ROOT/evidence/finite.json" >/dev/null

# ---------------- Gate 3: Disjoint-C S2-skip Streaming Build (<= 80 MB) ----------------
timeout 1800 /usr/bin/time -v -o "$ROOT/evidence/skip_build.time" "$BIN" build-skip \
    "$C_UORT" "$ROOT/evidence/artifacts/s2_skip.uors2s01" "$ROOT/evidence/skip_build.json"
jq -e '
    .c_corpus_sha256 == "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a" and
    .per_family_bytes[0] <= 20000000 and
    .per_family_bytes[1] <= 30000000 and
    .per_family_bytes[2] <= 30000000 and
    .total_bytes <= 80000000
' "$ROOT/evidence/skip_build.json" >/dev/null

# ---------------- Gate 4: Brier/EM + Disjoint-C Skip + Causal Cache Evaluation ----------------
run_gpu_guarded "Gate 4 brier" /usr/bin/time -v -o "$ROOT/evidence/brier.time" "$BIN" brier \
    "$TUNE_DIR" "$R_UORT" "$TRUNK" \
    "$B1A_R_TAPE" "$B1A_R_PICK" "$B1A_TUNE_TAPE" "$B2_TUNE_PICK" \
    "$A1S_DIR/a1s.weights" "$A1S_DIR/a1s.seal.json" "$GOLDEN_SHELL" \
    "$ROOT/evidence/artifacts/s2_skip.uors2s01" \
    "$ROOT/evidence/artifacts" "$ROOT/evidence/brier_report.json"

jq -e '
    .dev_tune_targets == 10807417 and
    .p1_a1s_baseline.all.bits_q12 == "125750699315" and
    .s2_skip_table.total_bytes <= 80000000 and
    .shifted_batch_parity == true and
    .spot_check_probes >= 12500 and
    .spot_check_divergences == 0 and
    .s3_spot_check_divergences == 0 and
    .s2_spot_check_divergences == 0 and
    (.arms | length == 4) and
    (.arms[0].epochs | length == 3) and
    (.arms[1].epochs | length == 3) and
    (.arms[2].epochs | length == 3) and
    (.arms[3].epochs | length == 3) and
    ((.arms[1].epochs[2].mean_brier_after_q32 | tonumber) < (.arms[1].epochs[0].mean_brier_before_q32 | tonumber)) and
    ((.arms[3].epochs[2].mean_brier_after_q32 | tonumber) < (.arms[3].epochs[0].mean_brier_before_q32 | tonumber))
' "$ROOT/evidence/brier_report.json" >/dev/null

test -f "$ROOT/evidence/artifacts/p4_c_skip_static.uors6b01"
test -f "$ROOT/evidence/artifacts/p4_c_skip_errwrite.uors6b01"
test -f "$ROOT/evidence/artifacts/p4_cache_static.uors6b01"
test -f "$ROOT/evidence/artifacts/p4_full_errwrite.uors6b01"

# ---------------- Gate 5: GPU-Voted Clean-Anchor 2-Hop S4 + R1 / R2 / R4 Panels ----------------
run_gpu_guarded "Gate 5 reply" /usr/bin/time -v -o "$ROOT/evidence/reply.time" "$BIN" reply \
    "$COMPACT_V3" "$A1S_DIR/a1s.weights" "$A1S_DIR/a1s.seal.json" \
    "$TRUNK" "$TOKENIZER" "$FRESH256_DIR" "$R_UORT" \
    "$ROOT/evidence/artifacts" "$ROOT/evidence/reply_report.json"

jq -e '
    .s4_gpu_vs_cpu_divergences == 0 and
    .s4_artifact.total_hinge_updates > 0 and
    .r1_r2_m2_fresh256.rows == 256 and
    .r1_r2_m2_fresh256.pairs == 128 and
    .r4_heldout_phrasing.rows == 240 and
    .r4_heldout_phrasing.pairs == 120
' "$ROOT/evidence/reply_report.json" >/dev/null

test -f "$ROOT/evidence/artifacts/s4_train.uors4p01"

# ---------------- Receipt Manifest ----------------
(
    cd "$ROOT/evidence"
    find . -maxdepth 3 -type f ! -name 'sass.asm' ! -name 'p4_brier_kernels.ptx' ! -name 'receipts.sha256' | sort | xargs sha256sum > receipts.sha256
)
echo "STAGE_B4_P3_EVAL_COMPLETE"
