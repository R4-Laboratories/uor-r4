#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
root="$study/matched-device"
prior="$runs/20261126-codex-fixed-assignment-median"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$root"/recipe-v* ]]
before="$prior/methods-v2/inputs/protected.state"
after="$prior/train-v1/states/ordered.state"
bank="$prior/methods-v2/inputs/primary.words"
binary="$study/device/build-v1/switch-competition-sm89"
research="$study/device/pod-payloads-v1/research"
finite="$study/device/finite-v1"
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { [[ $1 =~ ^[0-9a-f]{64}$ && $(sha "$2") == "$1" ]]; }
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
listed() { grep -Fxq "$(sha256sum -- "$2")" "$1"; }

mode=${1:?package, authorize, import, or replay}
v=${2:?new version}
[[ $v =~ ^v[1-9][0-9]*$ ]]
case "$mode" in
package)
    [[ $# == 5 ]]
    panel=$(realpath -e "$3"); reg=$(realpath -e "$5")
    [[ $panel == "$study"/matched-train/panel-v* && $reg == "$study/"* && -f $reg/TRAIN-LAYOUT.md ]]
    pin "$4" "$panel/SHA256SUMS"
    verify "$panel"; verify "$reg"
    [[ $(wc -l < "$panel/rows.jsonl") == 8 && $(wc -l < "$panel/requests.tsv") == 8 &&
       $(wc -l < "$panel/requests.jsonl") == 8 ]]
    jq -es 'length==8 and (group_by(.pair_id)|length)==4 and
        all(group_by(.pair_id)[]; length==2 and (map(.selector)|sort)==["first","later"])' \
        "$panel/requests.jsonl" >/dev/null
    dir="$root/package-$v"; payload="$dir/layout"; mkdir "$dir" "$payload"
    cp -- "$panel/ordered/text.bin" "$panel/requests.tsv" "$here/run-layout.sh" "$payload/"
    cp -- "$panel/SHA256SUMS" "$payload/wick-panel.SHA256SUMS"
    cp -- "$reg/TRAIN-LAYOUT.md" "$payload/"
    printf '%s\n' "$panel" > "$dir/panel.txt"
    printf '%s\n' "$reg" > "$dir/registration.txt"
    seal "$payload"
    tar -C "$dir" -czf "$dir/layout.tar.gz" layout
    sha256sum "$panel/SHA256SUMS" "$reg/SHA256SUMS" "$reg/TRAIN-LAYOUT.md" "$here/"{run-layout,stage-layout}.sh \
        > "$dir/inputs.sha256"
    seal "$dir"
    sha256sum "$payload/SHA256SUMS" "$dir/layout.tar.gz"
    ;;
authorize)
    [[ $# == 7 ]]
    package=$(realpath -e "$3"); grade=$(realpath -e "$4"); capture=$(realpath -e "$5")
    released=$(realpath -e "$6"); fit=$(realpath -e "$7")
    [[ $package == "$root"/package-v* && $grade == "$study"/grade-v* && $capture == "$study"/capture-v* &&
       $released == "$study"/release-v* && $fit == "$study"/audit/layout-FIT-v* ]]
    for dir in "$package" "$grade" "$capture" "$capture-lease" "$released" "$fit" "$finite-lease" "$research"; do verify "$dir"; done
    for dir in "$package" "$grade" "$capture" "$released"; do sha256sum --strict --quiet -c "$dir/inputs.sha256"; done
    listed "$grade/inputs.sha256" "$capture/SHA256SUMS"
    listed "$capture/inputs.sha256" "$released/SHA256SUMS"
    jq -e '.train_derived.all.pairs==32 and .train_derived.enrolled_prompts==64 and
        (.train_derived.matching_full_pair_signature|type=="array" and length==0)' "$grade/summary.json" >/dev/null
    provider="$capture-lease/provider.json"
    [[ $(cat "$capture-lease/exit-code.txt") == 0 ]]
    jq -e '.provider=="vast" and .phase=="capture" and (.gpu_name|contains("RTX 4090"))' "$provider" >/dev/null
    jq -e --arg json "$(jq -c 'del(.phase,.payload_manifest_sha256,.raw_manifest_sha256)' "$finite-lease/provider.json")" \
        'del(.phase,.payload_manifest_sha256,.raw_manifest_sha256)==($json|fromjson)' "$provider" >/dev/null
    cmp "$released/ordered.unpacked" "$after"
    panel=$(cat "$package/panel.txt"); reg=$(cat "$package/registration.txt")
    verify "$panel"; verify "$reg"
    cmp "$panel/SHA256SUMS" "$package/layout/wick-panel.SHA256SUMS"
    cmp "$reg/TRAIN-LAYOUT.md" "$package/layout/TRAIN-LAYOUT.md"
    grep -Fq 'TEST FITNESS: FIT' "$fit/FIT.md"
    for file in "$panel/SHA256SUMS" "$reg/SHA256SUMS" "$reg/TRAIN-LAYOUT.md" "$grade/SHA256SUMS" "$grade/summary.json" \
        "$binary" "$before" "$after"; do grep -Fq "$(sha "$file")" "$fit/FIT.md"; done
    gpu=$(jq -er .gpu_uuid "$provider"); hardware=$(jq -er .hardware_host "$provider")
    grep -Fq "$gpu" "$fit/FIT.md"
    dir="$root/authorization-$v"; auth="$dir/layout-auth"; mkdir "$dir" "$auth"
    cp -- "$fit/FIT.md" "$grade/summary.json" "$auth/"
    sha "$package/layout/SHA256SUMS" > "$auth/payload-sha256.txt"
    sha "$research/SHA256SUMS" > "$auth/research-payload-sha256.txt"
    printf '%s\n' "$gpu" > "$auth/gpu-uuid.txt"
    printf '%s\n' "$hardware" > "$auth/hardware-host.txt"
    jq -er .instance_id "$provider" > "$auth/instance-id.txt"
    printf '%s  %s\n' "$(sha "$binary")" bin/switch-competition-sm89 \
        "$(sha "$bank")" states/bank "$(sha "$before")" states/before "$(sha "$after")" states/after > "$auth/model.sha256"
    (cd "$research" && sha256sum --strict --quiet -c "$auth/model.sha256")
    sha256sum "$package/SHA256SUMS" "$package/layout/SHA256SUMS" "$panel/SHA256SUMS" "$reg/SHA256SUMS" \
        "$reg/TRAIN-LAYOUT.md" "$released/SHA256SUMS" "$grade/SHA256SUMS" "$grade/summary.json" \
        "$capture/SHA256SUMS" "$capture-lease/SHA256SUMS" "$finite-lease/SHA256SUMS" "$fit/SHA256SUMS" \
        "$binary" "$bank" "$before" "$after" > "$auth/inputs.sha256"
    seal "$auth"
    tar -C "$dir" -czf "$dir/layout-authorization.tar.gz" layout-auth
    seal "$dir"
    sha256sum "$auth/SHA256SUMS" "$dir/layout-authorization.tar.gz"
    ;;
import)
    [[ $# == 6 ]]
    tarball=$(realpath -e "$3"); output_sha=$4; auth=$(realpath -e "$5"); control=$(realpath -e "$6")
    [[ $auth == "$root"/authorization-v*/layout-auth && $control == "$root"/control-v* ]]
    verify "$auth"
    sha256sum --strict --quiet -c "$auth/inputs.sha256"
    name="capture-layout-$v"; dest="$root/$name"; lease="$dest-lease"
    [[ ! -e $dest && ! -e $lease ]]
    tar -tzf "$tarball" | awk -v name="$name" '$0 !~ ("^" name "/") || $0 ~ /(^|\/)\.\.(\/|$)/ {bad=1} END {exit bad}'
    tar -tvzf "$tarball" | awk '$0 !~ /^[-d]/ {bad=1} END {exit bad}'
    received="$root/import-$v"; mkdir "$received"
    tar -xzf "$tarball" -C "$received" --no-same-owner --no-same-permissions
    raw="$received/$name"
    pin "$output_sha" "$raw/SHA256SUMS"; verify "$raw"
    [[ $(cat "$raw/payload-sha256.txt") == "$(cat "$auth/payload-sha256.txt")" &&
       $(cat "$raw/authorization-sha256.txt") == "$(sha "$auth/SHA256SUMS")" &&
       $(cat "$raw/binary-sha256.txt") == "$(sha "$binary")" && $(cat "$raw/phase.txt") == layout &&
       $(cat "$raw/hardware-host.txt") == "$(cat "$auth/hardware-host.txt")" ]]
    cmp "$auth/inputs.sha256" "$raw/inputs.sha256"
    cmp "$raw/SHA256SUMS" "$control/raw.SHA256SUMS"
    [[ $(cat "$control/raw-payload-sha256.txt") == "$output_sha" ]]
    gpu=$(cat "$auth/gpu-uuid.txt"); instance=$(cat "$auth/instance-id.txt")
    grep -Fq "$gpu, NVIDIA GeForce RTX 4090, 8.9," "$raw/gpu-identity.txt"
    grep -Fxq "$(cat "$auth/hardware-host.txt")" "$control/pod-probe.txt"
    grep -Fq "$gpu" "$control/pod-probe.txt"
    grep -Fq "pod=$instance gpu=0 queued" "$control/job.log"
    grep -Eq '^# exit=0 end=' "$control/job.log"
    grep -Fq "$output_sha" "$control/job.log"
    started=$(cut -d. -f1 "$raw/started-at.txt"); ended=$(cut -d. -f1 "$raw/finished-at.txt")
    jq -e --arg pod "$instance" --arg start "${started%Z}Z" --arg end "${ended%Z}Z" '
        .pod==$pod and .lab=="codex" and .gpus==[0] and
        (.started|fromdateiso8601)<=($start|fromdateiso8601) and
        ($start|fromdateiso8601)<=($end|fromdateiso8601) and ($end|fromdateiso8601)<=(.expires|fromdateiso8601)
    ' "$control/lease.json" >/dev/null
    for arm in before after; do
        [[ $(wc -l < "$raw/$arm/native.jsonl") == 8 && $(wc -l < "$raw/$arm/requests.jsonl") == 8 ]]
    done
    mv -T -- "$raw" "$dest"; mkdir "$lease"
    cp -- "$control/"{lease.json,pod-probe.txt,job.log,raw.SHA256SUMS,raw-payload-sha256.txt} "$lease/"
    printf '0\n' > "$lease/exit-code.txt"
    jq -n --arg pod "$instance" --arg gpu "$gpu" --arg raw "$output_sha" --arg bin "$(sha "$binary")" \
        --arg hardware "$(cat "$auth/hardware-host.txt")" '
        {provider:"vast",instance_id:$pod,gpu_name:"NVIDIA GeForce RTX 4090",gpu_uuid:$gpu,hardware_host:$hardware,
         phase:"layout",binary_sha256:$bin,payload_manifest_sha256:$raw,raw_manifest_sha256:$raw}' > "$lease/provider.json"
    seal "$lease"
    sha256sum "$tarball" "$dest/SHA256SUMS" "$lease/SHA256SUMS" > "$received/inputs.sha256"
    seal "$received"
    sha256sum "$dest/SHA256SUMS" "$lease/SHA256SUMS" "$received/SHA256SUMS"
    ;;
replay)
    [[ $# == 7 ]]
    panel=$(realpath -e "$3"); capture=$(realpath -e "$4"); audit=$(realpath -e "$5")
    codec=$(realpath -e "$6"); methods=$(realpath -e "$7")
    [[ $panel == "$study"/matched-train/panel-v* && $capture == "$root"/capture-layout-v* &&
       $audit == "$study"/audit/build-v* && $methods == "$study"/methods-v* ]]
    for dir in "$panel" "$capture" "$capture-lease" "$audit" "$methods"; do verify "$dir"; done
    sha256sum --strict --quiet -c "$capture/inputs.sha256"
    listed "$capture/inputs.sha256" "$panel/SHA256SUMS"
    pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
    out="$root/replay-$v"; mkdir "$out"
    "$audit/switch-audit" census "$panel" "$codec" "$out/census.json"
    "$audit/switch-audit" replay "$panel" "$codec" "$before" "$bank" "$capture/before" - "$out/before.json"
    "$audit/switch-audit" replay "$panel" "$codec" "$after" "$bank" "$capture/after" - "$out/after.json"
    jq -nc --arg cohort matched-train --slurpfile meta "$panel/requests.jsonl" --slurpfile rows "$panel/rows.jsonl" \
        --slurpfile before "$capture/before/requests.jsonl" --slurpfile after "$capture/after/requests.jsonl" \
        -f "$methods/grade/row.jq" > "$out/rows.jsonl"
    [[ $(wc -l < "$out/rows.jsonl") == 8 ]]
    sha256sum "$panel/SHA256SUMS" "$capture/SHA256SUMS" "$capture-lease/SHA256SUMS" "$audit/SHA256SUMS" \
        "$methods/SHA256SUMS" "$codec" "$bank" "$before" "$after" > "$out/inputs.sha256"
    seal "$out"
    sha256sum "$out/SHA256SUMS"
    ;;
*) exit 64 ;;
esac
