#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

test "$#" -eq 2
stage=$(realpath -e "$1")
test "$0" -ef "$stage/dispatch.sh"
cd "$stage"

check() { printf '%s  %s\n' "$2" "$1" | sha256sum --strict -c -; }
field() { jq -er --arg key "$1" '.[$key] | strings | select(length > 0)' GO.json; }

check GO.json "$2"
test "$(hostname -s)" = wick
test "$(field decision)" = GO
test "$(field phase)" = 'DEV-test P1(b) UORKNC03 paired evaluation'
test "$(field fit_verdict)" = FIT
check dispatch.sh "$(field dispatch_sha256)"
check SHEET.md "$(field sheet_sha256)"
check FIT.md "$(field fit_sha256)"

gpu=GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41
test "$(field device_uuid)" = "$gpu"
test "$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader)" = 8.9
test "$(df --output=avail -B1 "$stage" | awk 'NR==2{print $1}')" -ge 21474836480

ROOT=$(field run_root)
EVAL_BIN=$(field eval_bin)
CUDA_V3=$(field cuda_v3_bin)
D37_DEVICE=$(field device_scorer)
COMPACT=$(field compact)
DEPLOY=$(field deploy_artifact)
RSTAR_MASSES=$(field rstar_masses)
RSTAR_JSON=$(field rstar_json)
TRUNK=$(field trunk)
C420=$(field c420_dir)
A1S_DIR=$(field a1s_dir)
A0_DIR=$(field a0_dir)

check "$EVAL_BIN" "$(field eval_sha256)"
check "$CUDA_V3" "$(field cuda_v3_sha256)"
check "$D37_DEVICE" "$(field device_scorer_sha256)"
check "$COMPACT" "$(field compact_sha256)"
check "$DEPLOY" "$(field deploy_artifact_sha256)"
check "$RSTAR_MASSES" "$(field rstar_masses_sha256)"
check "$RSTAR_JSON" "$(field rstar_json_sha256)"
check "$TRUNK" "$(field trunk_sha256)"
check "$C420/dev-test.json" "$(field census_sha256)"
check "$C420/dev-test.tokens.u16" "$(field dev_test_uort_sha256)"
check "$C420/id-bytes.u8" befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9
check "$C420/dev-test.novel.u8" 5bf3881bff8fb5abe5c15da2b6dfdfe60f4eaa709ee061d9d0405d618af10471
check "$A1S_DIR/a1s.weights" e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293
check "$A1S_DIR/a1s.manifest.json" 26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec
check "$A1S_DIR/a1s.seal.json" b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071
check "$A0_DIR/a1.weights" 9ec8168a20f250ef089db1b55d94b02f248d1ec39ee4f51d9692d98ee6be9c37
check "$A0_DIR/a1.manifest.json" 95b17f7bbb9059f4542f3505a9e7aeca5315bc540b8327a0d0a882cb8710e4c8
check "$A0_DIR/a1.seal.json" 5eddd7837bb67ed3b58b5a1262d711ade186719cc6a57ff31cc45d0042186b5b

LOCK=/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock
test -f "$LOCK"

admit_gpu() {
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    test "$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192
    test "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" -ge 10485760
}

export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DISABLE_PTX_JIT=1

# ---------------- Phase A: Verify Sealed R* & Artifact ----------------
test "$(stat -c %s "$RSTAR_MASSES")" -eq 170796608
test "$(stat -c %s "$DEPLOY")" -eq 3891016460
jq -e '
    .check_host == true and
    .sealed_run == true and
    .masses_sha256 == "59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de" and
    .masses_at_floor == 4 and
    .all.bits_q12 == "251672950854" and
    .novel.bits_q12 == "223758978608"
' "$RSTAR_JSON" >/dev/null

# ---------------- Phase B: Sealed DEV-test UORKNC03 UORCBF01 Capture ----------------
COUNT_OUT="$ROOT/count/run-v1"
C_SHA=cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a
COMPACT_SHA=75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233
UORT_SHA=023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b
CUBIN_SHA=4c329f9f09c5d571c4e46145680a884f684ea1793a1e2a5bcaf617d9f2651f6b
N_ROWS=21349576
if [ ! -f "$COUNT_OUT/count.sha256" ]; then
    test ! -e "$COUNT_OUT"
    mkdir -p "$COUNT_OUT"
    cp GO.json SHEET.md FIT.md "$COUNT_OUT/"
    (
        exec 8<>"$LOCK"
        flock -n 8 || { echo "shared Wick GPU lock busy (Phase B)" >&2; exit 75; }
        admit_gpu
        cd "$COUNT_OUT"
        nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,compute_cap,memory.free --format=csv > device.csv
        for width in 65536 65521; do
            /usr/bin/time -v -o "$width.time" "$CUDA_V3" cuda \
                "$COMPACT" "$C_SHA" "$COMPACT_SHA" "$C420/dev-test.tokens.u16" "$UORT_SHA" \
                "$COUNT_OUT/$width.features" "$width" > "$width.stdout" 2> "$width.stderr"
            test ! -s "$width.stdout"
            test "$(stat -c %s "$width.features")" -eq $((112 + 140 * N_ROWS))
            need=$((3804378885 + 64 * 1024 * 1024 + 198 * 8 + width * (24 + 11 * 32 + 4)))
            awk -v k="$(((N_ROWS + width - 1) / width))" -v need="$need" -v cubin="$CUBIN_SHA" '
                /^rows=21349576 batches=[0-9]+ uploads=1 upload_bytes=3804378885 cubin_sha256=[0-9a-f]+ upload_ms=[0-9]+ total_ms=[0-9]+ device_free_bytes=[0-9]+ required_device_bytes=[0-9]+$/ {
                    split($5,c,"="); split($8,f,"="); split($9,r,"=");
                    ok=($2 == "batches=" k && c[2] == cubin && r[2]+0==need && f[2]+0>=need)
                }
                END { exit !(NR==1 && ok) }
            ' "$width.stderr"
        done
    )
    cmp "$COUNT_OUT/65536.features" "$COUNT_OUT/65521.features"
    TAPE_SHA=$(sha256sum "$COUNT_OUT/65536.features" | cut -d ' ' -f1)
    /usr/bin/time -v -o "$COUNT_OUT/native.time" "$CUDA_V3" verify \
        "$COMPACT" "$C_SHA" "$COMPACT_SHA" "$C420/dev-test.tokens.u16" "$UORT_SHA" \
        "$COUNT_OUT/65536.features" > "$COUNT_OUT/native.json" 2> "$COUNT_OUT/native.stderr"
    jq -e --arg sha "$TAPE_SHA" '
        .rows == 21349576 and .probes_checked >= 25000 and .divergent_rows == 0 and .tape_sha256 == $sha
    ' "$COUNT_OUT/native.json" >/dev/null
    (cd "$COUNT_OUT" && sha256sum native.json 65536.features 65521.features 65536.time 65521.time native.time 65536.stderr 65521.stderr native.stderr device.csv > count.sha256)
fi
(cd "$COUNT_OUT" && sha256sum --strict -c count.sha256 >/dev/null)
cmp "$COUNT_OUT/65536.features" "$COUNT_OUT/65521.features"
TAPE_SHA=$(sha256sum "$COUNT_OUT/65536.features" | cut -d ' ' -f1)
jq -e --arg sha "$TAPE_SHA" '
    .rows == 21349576 and .probes_checked >= 25000 and .divergent_rows == 0 and .tape_sha256 == $sha
' "$COUNT_OUT/native.json" >/dev/null

# ---------------- Phase C: Frozen A1s & A0 Scoring + Paired Bootstrap ----------------
EVAL_STAGE="$ROOT/eval/stage-v1"
EVAL_OUT="$ROOT/eval/run-v1"
test ! -e "$EVAL_STAGE" && test ! -e "$EVAL_OUT"
mkdir -p "$EVAL_STAGE"
cp SHEET.md FIT.md "$EVAL_STAGE/"
jq --arg tape_sha "$TAPE_SHA" '.dev_test_tape_sha256 = $tape_sha' GO.json > "$EVAL_STAGE/GO.json"
EVAL_GO_SHA=$(sha256sum "$EVAL_STAGE/GO.json" | cut -d ' ' -f1)
(
    exec 8<>"$LOCK"
    flock -n 8 || { echo "shared Wick GPU lock busy (Phase C)" >&2; exit 75; }
    admit_gpu
    nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,compute_cap,memory.free --format=csv > "$ROOT/eval/device.csv"
    /usr/bin/time -v -o "$ROOT/eval/score.time" "$EVAL_BIN" score \
        "$C420" "$EVAL_STAGE/GO.json" "$EVAL_GO_SHA" \
        "$COMPACT" "$TRUNK" "$COUNT_OUT/65536.features" "$RSTAR_MASSES" \
        "$D37_DEVICE" "$A1S_DIR" "$A0_DIR" "$EVAL_OUT" \
        > "$ROOT/eval/score.stdout" 2> "$ROOT/eval/score.stderr"
)
cp "$EVAL_STAGE/GO.json" "$EVAL_OUT/GO.phase-c.json"
(cd "$EVAL_OUT" && sha256sum p1b-devtest.json per-record-q12.json GO.phase-c.json > eval.sha256)
echo "COMPLETE: P1B-V3 DEV-TEST PIPELINE SEALED"
