#!/usr/bin/env bash
# Invoke only on the top session's approved real-data run after FIT.
set -euo pipefail
umask 077
[[ $(hostname -s) == wick ]]
self=$(realpath "$0")
r=$(realpath "${1:?registered run root}")
bin=$(realpath "${2:?sealed ordinary client and packer directory}")
stage=$(realpath "${3:?frozen label-free inputs and private scoring truth}")
exe=$(realpath "${4:?trial executable}")
out=${5:?unused result directory}
base=$(dirname "$r")/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
base_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
device=$r/device-finite-r3
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin "$base_sha" "$base"
pin 92bcf275501e06a196f1ddd6ef998a8915b2b52a230d36590008f4cd8ed2c5e9 "$bin/uor-chat"
pin b042a06efa9b6fe5820240322e514733c39c9a7ae7cfad14dcd60950d6116814 "$bin/pack-native-interval"
pin 36666847e87330829c4081859588f50e0a2491eee07498e67934ce12f86c12e2 "$exe"
pin a98c09c8a09cc897beff0406b46e720fdc57308d11c2344d71d723ee8f6172c3 "$stage/FILES.sha256"
(cd "$stage"; sha256sum --strict --quiet -c FILES.sha256)
pin ccc4b76ac80f09b52bc672fd58fc0704dbdd621dd6f7bb4c7f63f67a1c0f6970 "$device/device-image.sha256"
(cd "$device"; sha256sum --strict --quiet -c device-image.sha256)
pin 71f558c88e662f16aeb09b589dd58d7d1689a334ddc36ffeecc73ea0a12ec1e6 "$device/termination-cuda"
mkdir -m 700 "$out"
out=$(realpath "$out")
cd "$out"
sha256sum "$bin/uor-chat" "$bin/pack-native-interval" "$exe" "$device/termination-cuda" "$stage/FILES.sha256" "$self" > TOOLS.sha256
for split in train held; do
    "$bin/uor-chat" --discrete "$base" --ordinary-termination-capture "$stage/$split.prefix.jsonl" > "$split.keys.jsonl"
    for arm in geometry ordered; do
        "$exe" wire "$stage" "$split" "$split.keys.jsonl" "$arm" "$split.$arm.bin"
    done
done
sha256sum TOOLS.sha256 *.keys.jsonl *.bin > FIT-INPUTS.sha256
export CUDA_DISABLE_PTX_JIT=1
export CUDA_VISIBLE_DEVICES=GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41
exec 8>>"$HOME/uor-codex/20261118-diagnostics/wick-codex-gpu.lock"
flock -n 8 || { echo 'shared device lock occupied' >&2; exit 2; }
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv >device-start.csv
for arm in geometry ordered; do
    for seed in 1 2; do
        run=$arm.$seed
        "$device/termination-cuda" train "train.$arm.bin" "$seed" actual "$run.words" "$run.stats"
        "$device/termination-cuda" replay "held.$arm.bin" "$run.words" "$run.scores"
        "$device/termination-cuda" train "train.$arm.bin" "$seed" actual "$run.repeat" "$run.repeat.stats"
        cmp "$run.words" "$run.repeat"
        cmp "$run.stats" "$run.repeat.stats"
    done
done
flock -u 8
for arm in geometry ordered; do
    for seed in 1 2; do
        run=$arm.$seed
        # The fit binding includes the state-producing TRAIN tape, seed and executable.
        { echo "arm=$arm seed=$seed epochs=32"; sha256sum "train.$arm.bin" "$device/termination-cuda"; } > "$run.fit"
        fit_sha=$(sha256sum "$run.fit" | cut -d' ' -f1)
        "$bin/pack-native-interval" ordinary-termination "$base" "$base_sha" "$arm" "$run.words" "$run.d11" "$fit_sha" > "$run.pack.json"
    done
done
sha256sum *.words *.scores *.stats *.fit *.d11 *.pack.json FIT-INPUTS.sha256 >FIT-OUTPUTS.sha256
