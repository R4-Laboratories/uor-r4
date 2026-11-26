#!/usr/bin/env bash
# Bind the reviewed physical inputs before training, and the independent TRAIN replay before held.
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
src=$(cd "$(dirname "$0")" && pwd); op=${1:?train or held}; ver=${2:?new authorization version}; core=${3:?runtime methods}
[[ $src == "$s"/dispatch-v* && $core =~ ^methods-v[1-9][0-9]*$ ]]
source "$s/$core/common-wick.sh"; version "$ver"; verify "$here"
case $op in
train)
    # Arguments name the sealed equipment/fixtures; the independent reviewer records these in FIT.md.
    fit=${4:?fit build}; finite=${5:?CUDA finite}; pv=${6:?panel}; cv=${7:?parent fixture}; af=${8:?independent fixture}; review=${9:?approved review directory}
    adapter_v=${UOR_BRIDGE_CLIENT_METHODS:-v4}; fixture_v=${UOR_BRIDGE_CLIENT_FIXTURE:-v5}
    for v in "$fit" "$finite" "$pv" "$cv" "$adapter_v" "$fixture_v"; do version "$v"; done
    [[ $af == "$s"/audit/* && $review == "$s"/review-v* ]]
    shift 9
    test "$#" -ge 3
    extra_seals=()
    for independent in "$@"; do
        [[ $independent == "$s"/audit/* ]]; verify "$independent"; extra_seals+=("$independent/SHA256SUMS")
    done
    p="$s/panel/panel-$pv"; f="$s/fit/build-$fit"; ff="$s/fit/finite-$finite"; bound="$s/finite-core-$cv"
    adapter="$s/client/methods-$adapter_v"; client_fixture="$s/client/fixture-$fixture_v"
    for tree in "$p" "$f" "$ff" "$ff-lease" "$bound" "$af" "$adapter" "$client_fixture"; do verify "$tree"; done
    listed "$client_fixture/inputs.sha256" "$adapter/SHA256SUMS"
    jq -e '.schema=="uor-r4.codex-4297.client-selection-fixture/1" and .pass' "$client_fixture/receipt.json" >/dev/null
    ancestry; verify "$historical"
    grep -Eiq '(TEST FITNESS: FIT|verdict.{0,15}FIT)' "$review/FIT.md"
    grep -Fq "$(sha "$f/SHA256SUMS")" "$review/FIT.md"
    grep -Fq "$(sha "$p/SHA256SUMS")" "$review/FIT.md"
    grep -Fq "$(sha "$client_fixture/SHA256SUMS")" "$review/FIT.md"
    test "$(cat "$ff-lease/exit-code.txt")" = 0
    jq -e '.pass and .near_identity_substitution_rejected and .positive_writer_gate' "$bound/checks.json" >/dev/null
    dest="$s/release-$ver"; mkdir "$dest"
    cp "$review/"{FIT.md,DECISION.md} "$dest/"
    printf '%s\n' "$f/bridge-fit-sm89" > "$dest/fit-path"
    printf '%s\n' "$p" > "$dest/panel-path"; printf '%s\n' "$bound/bundle" > "$dest/packer-path"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader > "$dest/device.txt"
    sha256sum "$dest/"{FIT.md,DECISION.md,fit-path,panel-path,packer-path,device.txt} \
        "$here/SHA256SUMS" "$here/"{ORDER.md,PRE-RUN.md,measure-wick.sh,held-wick.sh,grade-outputs-wick.sh} \
        "$p/SHA256SUMS" "$f/SHA256SUMS" "$f/bridge-fit-sm89" "$ff/SHA256SUMS" "$ff-lease/SHA256SUMS" \
        "$bound/SHA256SUMS" "$bound/bundle" "$af/SHA256SUMS" "${extra_seals[@]}" "$adapter/SHA256SUMS" "$client_fixture/SHA256SUMS" \
        "$state" "$reference" "$old_pairs" "$baseline_raw" "$far_state" "$historical/SHA256SUMS" \
        "$mixed/panel/panel-v1/SHA256SUMS" "$prev/SHA256SUMS" "$reader" "$replay" "$lock" "$banks" \
        "$omission/measurement-v1/SHA256SUMS" "$src/authorize-wick.sh" > "$dest/inputs.sha256"
    sha256sum --strict --quiet -c "$dest/inputs.sha256"; sha256sum "$dest/inputs.sha256"
    ;;
held)
    mv=${4:?TRAIN version}; a=$(realpath "${5:?sealed TRAIN replay}"); version "$mv"
    [[ $a == "$s"/audit/train-v* ]]; measurement="$s/measurement-$mv"
    verify "$a"; verify "$measurement"; sha256sum --strict --quiet -c "$a/inputs.sha256" "$measurement/selection.sha256"
    release_file=$(cut -d' ' -f3- "$measurement/inputs.sha256"); release_inputs "$release_file" "$(cut -d' ' -f1 "$measurement/inputs.sha256")"
    listed "$a/inputs.sha256" "$measurement/SHA256SUMS"; listed "$a/inputs.sha256" "$panel/SHA256SUMS"
    jq -e '.pass==true and .batches==6' "$a/readback.json" >/dev/null
    dir="$s/train-verified-$ver"; mkdir "$dir"
    jq --arg sha "$(sha "$a/SHA256SUMS")" '{verified:.pass,batches,independent_manifest:$sha}' "$a/readback.json" > "$dir/verified.json"
    sha256sum "$measurement/"{SHA256SUMS,selection.json,selection.sha256} "$a/"{SHA256SUMS,readback.json,inputs.sha256} \
        "$here/SHA256SUMS" "$dir/verified.json" "$release_file" "$panel/SHA256SUMS" "$src/authorize-wick.sh" > "$dir/inputs.sha256"
    sha256sum --strict --quiet -c "$dir/inputs.sha256"; sha256sum "$dir/inputs.sha256"
    ;;
*) exit 64 ;;
esac
