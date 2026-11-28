#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=${1:?absolute stage directory}
out=${2:?fresh absolute results directory}
uuid=${3:?leased GPU UUID}
[[ $stage == /* && $out == /* && $uuid == GPU-* ]] || exit 2

cd "$stage"
mkdir -m 700 "$out"
exec > "$out/dispatch.log" 2>&1
sample=
finish() {
    local rc=$?
    if [[ -n $sample ]]; then
        kill "$sample" 2>/dev/null || true
        wait "$sample" 2>/dev/null || true
    fi
    date -u +%FT%TZ > "$out/end.utc"
    printf '%s\n' "$rc" > "$out/status.new"
    mv "$out/status.new" "$out/status"
}
trap finish EXIT
date -u +%FT%TZ > "$out/start.utc"

# Require the approved manifest to bind each file the launch consumes.
for member in dispatch.sh bin/uor-lexical-centered-election p4.d11 internal-split.json input/MEMBERS.sha256 previous-members.sha256; do
    awk -v member="$member" '$2 == member { found++ } END { exit found != 1 }' approved.sha256
done
sha256sum --strict -c approved.sha256
printf "%s  %s\n" "9e42218516037cd2b98045ffd8a90fb8988cbe9f60bb437ed586e94fc3a6e21e" previous-members.sha256 | sha256sum --strict -c -
(cd previous && sha256sum --strict -c ../previous-members.sha256)
cp previous-members.sha256 "$out/previous-members.sha256"
cp approved.sha256 "$out/approved.sha256"
cp input/MEMBERS.sha256 "$out/input-members.sha256"
(
    cd input
    for member in train/{inputs.jsonl,labels.jsonl,refused.jsonl,receipt.json} \
        prior/{fit.json,geometry-435300.lex,geometry-435301.lex}; do
        awk -v member="$member" '$2 == member { found++ } END { exit found != 1 }' MEMBERS.sha256
    done
    sha256sum --strict -c MEMBERS.sha256
)
/usr/bin/time --version > "$out/timer.txt"
gpu_uuid=$(nvidia-smi --id="$uuid" --query-gpu=uuid --format=csv,noheader)
[[ $gpu_uuid == "$uuid" ]]
gpu_apps=$(nvidia-smi --id="$uuid" --query-compute-apps=pid --format=csv,noheader)
[[ -z $gpu_apps ]]
nvidia-smi --id="$uuid" --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$out/device.csv"
nvidia-smi --id="$uuid" --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader --loop=1 > "$out/gpu-samples.csv" &
sample=$!

CUDA_VISIBLE_DEVICES="$uuid" /usr/bin/time -v -o "$out/usage.txt" \
    ./bin/uor-lexical-centered-election run p4.d11 input internal-split.json previous "$out/trial"
