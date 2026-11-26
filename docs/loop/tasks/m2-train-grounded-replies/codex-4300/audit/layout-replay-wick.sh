#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
prior="$runs/20261126-codex-fixed-assignment-median"
audit="$study/audit"
here=$(cd "$(dirname "$0")" && pwd)
panel=$(realpath -e "${1:?layout panel}")
capture=$(realpath -e "${2:?returned layout capture}")
auth=$(realpath -e "${3:?sealed layout authorization}")
out=$(realpath -m "${4:?new independent replay directory}")
lease="$capture-lease"
primary_lease="$study/capture-v1-lease"
before="$prior/methods-v2/inputs/protected.state"
after="$prior/train-v1/states/ordered.state"
bank="$prior/methods-v2/inputs/primary.words"
build="$audit/build-v4"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
history="$audit/replay-v1"
[[ $# == 4 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs &&
   $here == "$audit"/recipe-v* && $panel == "$study"/matched-train/panel-v* &&
   $capture == "$study"/matched-device/capture-layout-v* && $auth == "$study"/matched-device/authorization-v*/layout-auth &&
   $out == "$audit"/layout-replay-v* ]]
for directory in "$panel" "$capture" "$lease" "$primary_lease" "$auth" "$build" "$history"; do
    (cd "$directory" && sha256sum --strict --quiet -c SHA256SUMS)
done
for directory in "$capture" "$auth"; do sha256sum --strict --quiet -c "$directory/inputs.sha256"; done
cmp "$capture/inputs.sha256" "$auth/inputs.sha256"
grep -Fxq "$(sha256sum "$panel/SHA256SUMS")" "$auth/inputs.sha256"
for payload in "$before" "$after" "$bank"; do grep -Fxq "$(sha256sum "$payload")" "$auth/inputs.sha256"; done
test "$(cat "$lease/exit-code.txt")" = 0
[[ $(cat "$capture/authorization-sha256.txt") == "$(sha256sum "$auth/SHA256SUMS" | cut -d' ' -f1)" ]]
[[ $(cat "$capture/payload-sha256.txt") == "$(cat "$auth/payload-sha256.txt")" ]]
jq -e --arg returned "$(sha256sum "$capture/SHA256SUMS" | cut -d' ' -f1)"     --slurpfile primary "$primary_lease/provider.json" '
    def identity: [.provider,.instance_id,.gpu_uuid,.gpu_name,.hardware_host,.binary_sha256];
    .phase=="layout" and .payload_manifest_sha256==$returned and .raw_manifest_sha256==$returned and
       identity==($primary[0]|identity)
' "$lease/provider.json" >/dev/null
uuid=$(jq -er .gpu_uuid "$lease/provider.json")
grep -Fq "$uuid" "$capture/gpu-identity.txt"
grep -Fq "$uuid" "$lease/pod-probe.txt"
cmp "$lease/raw.SHA256SUMS" "$capture/SHA256SUMS"
mkdir "$out"
"$build/switch-audit" replay "$panel" "$codec" "$before" "$bank" "$capture/before" - "$out/before.json"
"$build/switch-audit" replay "$panel" "$codec" "$after" "$bank" "$capture/after" - "$out/after.json"
jq -ne --slurpfile before "$out/before.json" --slurpfile after "$out/after.json" \
    --slurpfile historic_before "$history/historical-before.json" --slurpfile historic_after "$history/historical-after.json" \
    -f "$here/layout-replay.jq" > "$out/summary.json"
sha256sum "$0" "$here/layout-replay.jq" "$panel/SHA256SUMS" "$capture/SHA256SUMS" "$lease/SHA256SUMS" \
    "$auth/SHA256SUMS" "$build/SHA256SUMS" "$history/SHA256SUMS" "$codec" "$bank" "$before" "$after" > "$out/inputs.sha256"
(
    cd "$out"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
