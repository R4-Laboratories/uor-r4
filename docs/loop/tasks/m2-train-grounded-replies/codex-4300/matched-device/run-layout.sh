#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
[[ $# == 6 ]]
layout=$(cd "$(dirname "$0")" && pwd -P)
layout_sha=$1
research=$(realpath -e "$2")
out=$(realpath -m "$3")
gpu=$4
auth=$(realpath -e "$5")
auth_sha=$6
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { [[ $1 =~ ^[0-9a-f]{64}$ && $(sha "$2") == "$1" ]]; }
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
pin "$layout_sha" "$layout/SHA256SUMS"
pin "$auth_sha" "$auth/SHA256SUMS"
verify "$layout"; verify "$auth"
[[ $(cat "$auth/payload-sha256.txt") == "$layout_sha" && $(cat "$auth/gpu-uuid.txt") == "$gpu" &&
   $(cat "$auth/hardware-host.txt") == "$(hostname -s)" && $gpu =~ ^GPU-[0-9a-fA-F-]+$ ]]
pin "$(cat "$auth/research-payload-sha256.txt")" "$research/SHA256SUMS"
(cd "$research" && sha256sum --strict --quiet -c "$auth/model.sha256")
grep -Fq 'TEST FITNESS: FIT' "$auth/FIT.md"
mkdir "$out"
date -u +%FT%T.%NZ > "$out/started-at.txt"
hostname -s > "$out/hardware-host.txt"
nvidia-smi --id="$gpu" --query-gpu=uuid,name,compute_cap,driver_version,memory.total,memory.free \
    --format=csv,noheader,nounits > "$out/gpu-identity.txt"
awk -F '[[:space:]]*,[[:space:]]*' -v uuid="$gpu" \
    'NR==1 && $1==uuid && $2=="NVIDIA GeForce RTX 4090" && $3=="8.9" && $6>=4096 {ok=1} END {exit !ok}' \
    "$out/gpu-identity.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader |
    awk -F ',' -v uuid="$gpu" '$1==uuid' > "$out/gpu-processes-before.txt"
test ! -s "$out/gpu-processes-before.txt"
export CUDA_VISIBLE_DEVICES="$gpu"
cp -- "$auth/inputs.sha256" "$out/"
printf '%s\n' "$layout_sha" > "$out/payload-sha256.txt"
printf '%s\n' "$auth_sha" > "$out/authorization-sha256.txt"
sha "$research/bin/switch-competition-sm89" > "$out/binary-sha256.txt"
printf 'layout\n' > "$out/phase.txt"

for arm in before after; do
    path="$out/$arm"
    cmd=("$research/bin/switch-competition-sm89" "$research/states/bank" "$layout/text.bin"
        "$research/states/$arm" "$layout/requests.tsv" "$path")
    printf '%q ' "${cmd[@]}" > "$path.command"
    printf '\n' >> "$path.command"
    (TIMEFORMAT='wall_s=%3R user_s=%3U system_s=%3S'; time "${cmd[@]}" > "$path.stdout" 2> "$path.stderr") \
        2> "$path.time"
    [[ $(wc -l < "$path/requests.jsonl") == 8 && $(wc -l < "$path/native.jsonl") == 8 ]]
done
(cd "$research" && sha256sum --strict --quiet -c "$auth/model.sha256")
verify "$layout"; verify "$auth"
date -u +%FT%T.%NZ > "$out/finished-at.txt"
seal "$out"
sha256sum "$out/SHA256SUMS"
