#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition
payload=$(realpath -e "${1:?returned result}")
phase=${2:?finite or capture}
control=$(realpath -e "${3:?provider evidence directory}")
[[ $# == 3 && $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs &&
    $control == "$study"/custody/* ]]
case "$phase" in
finite) [[ $payload == "$study"/device/finite-v* ]];;
capture) [[ $payload == "$study"/capture-v* ]];;
*) exit 64 ;;
esac
(cd "$payload" && sha256sum --quiet --strict -c SHA256SUMS)
pod=$(jq -er '.pod|select(test("^vast-[0-9]+$"))' "$control/lease.json")
host=$(cat "$payload/hardware-host.txt")
IFS=, read -r uuid model capability driver _ < "$payload/gpu-identity.txt"
model=${model# }; capability=${capability# }; driver=${driver# }
[[ $uuid =~ ^GPU-[0-9a-f-]+$ && $model == 'NVIDIA GeForce RTX 4090' && $capability == 8.9 ]]
grep -Fxq "$host" "$control/pod-probe.txt"
grep -Fq "$uuid" "$control/pod-probe.txt"
grep -Fq "pod=$pod gpu=0 queued" "$control/job.log"
grep -Eq '^# exit=0 end=' "$control/job.log"
raw=$(sha256sum "$control/raw.SHA256SUMS" | cut -d' ' -f1)
[[ $(cat "$control/raw-payload-sha256.txt") == "$raw" ]]
grep -Fq "$raw" "$control/job.log"
(cd "$payload" && sha256sum --strict --quiet -c "$control/raw.SHA256SUMS")
start=$(cut -d. -f1 "$payload/started-at.txt")
end=$(cut -d. -f1 "$payload/finished-at.txt")
jq -e --arg start "${start%Z}Z" --arg end "${end%Z}Z" '
    .lab=="codex" and .gpus==[0] and .session=="switch-source-trace" and
    (.started|fromdateiso8601)<=($start|fromdateiso8601) and
    ($start|fromdateiso8601)<=($end|fromdateiso8601) and ($end|fromdateiso8601)<=(.expires|fromdateiso8601)
' "$control/lease.json" >/dev/null
bin=$(cat "$payload/binary-sha256.txt")
[[ $bin == 7cd831ddec22535ecc94783cef2dae34db1d71a0428e4b40bfe8d5a43b4437d1 ]]
out="$payload-lease"; mkdir "$out"
cp "$control/"{lease.json,pod-probe.txt,job.log,raw.SHA256SUMS,raw-payload-sha256.txt} "$out/"
printf '0\n' > "$out/exit-code.txt"
jq -n --arg pod "$pod" --arg uuid "$uuid" --arg name "$model" --arg host "$host" --arg driver "$driver" \
    --arg phase "$phase" --arg bin "$bin" --arg raw "$(cat "$out/raw-payload-sha256.txt")" \
    --arg manifest "$(sha256sum "$payload/SHA256SUMS" | cut -d' ' -f1)" \
    '{provider:"vast",instance_id:$pod,gpu_name:$name,gpu_uuid:$uuid,hardware_host:$host,
      driver:$driver,phase:$phase,binary_sha256:$bin,payload_manifest_sha256:$manifest,raw_manifest_sha256:$raw}' > "$out/provider.json"
(cd "$out" && find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 sha256sum > .seal.pending &&
    mv .seal.pending SHA256SUMS && sha256sum --strict --quiet -c SHA256SUMS)
sha256sum "$out/SHA256SUMS"
