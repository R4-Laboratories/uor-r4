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
test "$(field phase)" = 'DEV-test P1(b) paired evaluation'
test "$(field fit_verdict)" = FIT
check dispatch.sh "$(field dispatch_sha256)"
check SHEET.md "$(field sheet_sha256)"
check FIT.md "$(field fit_sha256)"

gpu=GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41
test "$(field device_uuid)" = "$gpu"
test "$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader)" = 8.9
test "$(df --output=avail -B1 "$stage" | awk 'NR==2{print $1}')" -ge 21474836480

ROOT=$(field run_root)
KN_BIN=$(field kn_bin)
EVAL_BIN=$(field eval_bin)
D36_BATCH=$(field d36_batch)
D37_REPLAY=$(field d37_replay)
D37_DEVICE=$(field d37_device)
COMPACT=$(field compact)
TRUNK=$(field trunk)
C420=$(field c420_dir)
A1S_DIR=$(field a1s_dir)
A0_DIR=$(field a0_dir)

check "$KN_BIN" "$(field kn_sha256)"
check "$EVAL_BIN" "$(field eval_sha256)"
check "$D36_BATCH" "$(field d36_batch_sha256)"
check "$D37_REPLAY" "$(field d37_replay_sha256)"
check "$D37_DEVICE" "$(field device_scorer_sha256)"
check "$COMPACT" "$(field compact_sha256)"
check "$TRUNK" "$(field trunk_sha256)"
check "$C420/train.tokens.u16" "$(field train_sha256)"
check "$C420/dev-test.json" "$(field census_sha256)"
check "$C420/dev-test.tokens.u16" "$(field dev_test_uort_sha256)"
check "$C420/id-bytes.u8" befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9
check "$C420/dev-test.novel.u8" 5bf3881bff8fb5abe5c15da2b6dfdfe60f4eaa709ee061d9d0405d618af10471
check "$A1S_DIR/candidate.weights" b6b3da9e94ac33d0d8dc6c192c3117afc4fdc1f8e9c83dbf051010f5c2f31bb0
check "$A1S_DIR/candidate.manifest.json" 133e691cca11e6663163271c63d1f248aa8a9c575d7adf42ae4131316b74ca97
check "$A0_DIR/a1.seal.json" 5a7d108997da3378f3bee79599b95910155e646dc81d54ae1fafe6891d06267e

LOCK=/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock
test -f "$LOCK"

admit_gpu() {
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    test "$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192
    test "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" -ge 10485760
}

export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DISABLE_PTX_JIT=1

# ---------------- Phase A: Canonical R* on sealed DEV-test ----------------
RSTAR_OUT="$ROOT/rstar/run-v1"
if [ ! -f "$RSTAR_OUT/rstar.sha256" ]; then
    test ! -e "$RSTAR_OUT"
    mkdir -p "$RSTAR_OUT"
    cp GO.json SHEET.md FIT.md "$RSTAR_OUT/"
    (
        exec 8<>"$LOCK"
        flock -n 8 || { echo "shared Wick GPU lock busy (Phase A)" >&2; exit 75; }
        admit_gpu
        nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,compute_cap,memory.free --format=csv > "$RSTAR_OUT/device.csv"
        /usr/bin/time -v -o "$RSTAR_OUT/rstar.time" "$KN_BIN" \
            "$C420/train.tokens.u16" \
            "$C420/dev-test.tokens.u16" \
            "$C420/id-bytes.u8" \
            "$C420/dev-test.novel.u8" \
            10 16777216 "$RSTAR_OUT/rstar-devtest.json" \
            --sealed-run --check-host --profile-phases \
            --write-masses "$RSTAR_OUT/rstar-devtest.masses.u64" \
            > "$RSTAR_OUT/rstar.stdout" 2> "$RSTAR_OUT/rstar.stderr"
    )
    (cd "$RSTAR_OUT" && sha256sum rstar-devtest.json rstar-devtest.masses.u64 rstar.time rstar.stderr device.csv > rstar.sha256)
fi
(cd "$RSTAR_OUT" && sha256sum --strict -c rstar.sha256 >/dev/null)
test "$(stat -c %s "$RSTAR_OUT/rstar-devtest.masses.u64")" -eq 170796608
check "$RSTAR_OUT/rstar-devtest.masses.u64" 59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de
jq -e '
    .check_host == true and
    .sealed_run == true and
    .masses_sha256 == "59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de" and
    .masses_at_floor == 4 and
    .all.bits_q12 == "251672950854" and
    .novel.bits_q12 == "223758978608"
' "$RSTAR_OUT/rstar-devtest.json" >/dev/null

# ---------------- Phase B: Sealed DEV-test UORCBF01 Capture ----------------
COUNT_OUT="$ROOT/count/run-v1"
C_SHA=cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a
COMPACT_SHA=064b691a54895771df1a957b2f97c91163990b6c6b6e09504c43285a97c49ef8
UORT_SHA=023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b
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
            /usr/bin/time -v -o "$width.time" "$D36_BATCH" cuda \
                "$COMPACT" "$C_SHA" "$COMPACT_SHA" "$C420/dev-test.tokens.u16" "$UORT_SHA" \
                "$COUNT_OUT/$width.features" "$width" > "$width.stdout" 2> "$width.stderr"
            test ! -s "$width.stdout"
            test "$(stat -c %s "$width.features")" -eq $((112 + 140 * N_ROWS))
            need=$((3912334481 + 64 * 1024 * 1024 + 11 * 6 * 8 + 380 * width))
            awk -v k="$(((N_ROWS + width - 1) / width))" -v need="$need" '
                /^rows=21349576 batches=[0-9]+ uploads=1 upload_bytes=3912334481 upload_ms=[0-9]+ total_ms=[0-9]+ device_free_bytes=[0-9]+ required_device_bytes=[0-9]+$/ {
                    split($7,f,"="); split($8,r,"="); ok=($2 == "batches=" k && r[2]+0==need && f[2]+0>=need)
                }
                END { exit !(NR==1 && ok) }
            ' "$width.stderr"
        done
    )
    cmp "$COUNT_OUT/65536.features" "$COUNT_OUT/65521.features"
    TAPE_SHA=$(sha256sum "$COUNT_OUT/65536.features" | cut -d ' ' -f1)
    expected=$(od -An -v -j64 -tu2 "$C420/dev-test.tokens.u16" | awk -v nrows="$N_ROWS" '
        { for(i=1;i<=NF;i++){ v=$i+0; if(v==0){frames++;first=1;continue} if(first || v==1 || n%65536==0 || n%65536==65535 || n%4093==197) seen++; first=0; n++ } }
        END { if(n!=nrows || frames!=5321) exit 1; print seen }
    ')
    /usr/bin/time -v -o "$COUNT_OUT/native.time" "$D37_REPLAY" \
        "$COMPACT" "$C_SHA" "$COMPACT_SHA" "$C420/dev-test.tokens.u16" "$UORT_SHA" \
        "$COUNT_OUT/65536.features" 65536 > "$COUNT_OUT/native.json"
    jq -e --arg sha "$TAPE_SHA" --argjson expect "$expected" '
        .sha256 == $sha and .native_rows == $expect and .words_per_native_row == 23 and
        .rows == 21349576 and .records == 5321 and .width == 65536 and .bytes == (112 + 140 * 21349576)
    ' "$COUNT_OUT/native.json" >/dev/null
    (cd "$COUNT_OUT" && sha256sum native.json 65536.features 65521.features 65536.time 65521.time native.time 65536.stderr 65521.stderr device.csv > count.sha256)
fi
(cd "$COUNT_OUT" && sha256sum --strict -c count.sha256 >/dev/null)
cmp "$COUNT_OUT/65536.features" "$COUNT_OUT/65521.features"
TAPE_SHA=$(sha256sum "$COUNT_OUT/65536.features" | cut -d ' ' -f1)
jq -e --arg sha "$TAPE_SHA" '
    .sha256 == $sha and .words_per_native_row == 23 and
    .rows == 21349576 and .records == 5321 and .width == 65536 and .bytes == (112 + 140 * 21349576)
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
    /usr/bin/time -v -o "$ROOT/eval/score.time" "$EVAL_BIN" score \
        "$C420" "$EVAL_STAGE/GO.json" "$EVAL_GO_SHA" \
        "$COMPACT" "$TRUNK" "$COUNT_OUT/65536.features" "$RSTAR_OUT/rstar-devtest.masses.u64" \
        "$D37_DEVICE" "$A1S_DIR" "$A0_DIR" "$EVAL_OUT" \
        > "$ROOT/eval/score.stdout" 2> "$ROOT/eval/score.stderr"
)
cp "$EVAL_STAGE/GO.json" "$EVAL_OUT/GO.phase-c.json"
(cd "$EVAL_OUT" && sha256sum p1b-devtest.json per-record-q12.json GO.phase-c.json > eval.sha256)
echo "COMPLETE: P1B DEV-TEST PIPELINE SEALED"
