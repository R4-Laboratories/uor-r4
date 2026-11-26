#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?train or held}; v=${2:?new release or permit}; version "$v"
case $mode in
train)
    panel=$(realpath "${3:?panel}"); fitter=$(realpath "${4:?CUDA binary}"); ff=$(realpath "${5:?fit finite}")
    core=$(realpath "${6:?core finite}"); client=$(realpath "${7:?writer adapter}"); cf=$(realpath "${8:?writer finite}")
    review=$(realpath "${9:?non-author review}"); shift 9
    [[ $panel == "$s"/panel/panel-v* && $fitter == "$s"/fit/build-v*/median-fit-sm89 &&
       $ff == "$s"/fit/* && $core == "$s"/core-fixture-v* && $client == "$s"/client/methods-v* &&
       $cf == "$s"/client/fixture-v* && $review == "$s"/review-v* && $# -ge 1 ]]
    files=()
    for dir in "$here" "$panel" "${fitter%/*}" "$ff" "$core" "$client" "$cf" "$review" "$@"; do
        [[ $dir == "$s/"* ]]; verify "$dir"; files+=("$dir/SHA256SUMS")
    done
    physical_pins
    grep -Fq 'TEST FITNESS: FIT' "$review/FIT.md"
    for file in "$here/SHA256SUMS" "$panel/SHA256SUMS" "$fitter" "$ff/SHA256SUMS" "$core/SHA256SUMS" "$cf/SHA256SUMS"; do
        grep -Fq "$(sha "$file")" "$review/FIT.md"; done
    listed "$cf/inputs.sha256" "$client/SHA256SUMS"
    jq -e '.pass and .positive_releases_and_swap_refusals==2' "$core/checks.json" >/dev/null
    jq -e '.pass and .forged_gate_rejected and .generated==0' "$cf/receipt.json" >/dev/null
    d="$s/release-$v"; mkdir "$d"; cp "$review/FIT.md" "$here/DECISION.md" "$d/"
    printf '%s\n' "$panel" > "$d/panel-path"; printf '%s\n' "$fitter" > "$d/fit-path"; printf '%s\n' "$client" > "$d/client-path"
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > "$d/device.txt"
    sha256sum "$d/"{FIT.md,DECISION.md,panel-path,fit-path,client-path,device.txt} "$fitter" "$reader" "$banks" "$replay" "$lock" \
        "$here/"{ORDER.md,PRE-RUN.md} "${files[@]}" > "$d/inputs.sha256"
    sha256sum --strict --quiet -c "$d/inputs.sha256"; seal "$d"; sha256sum "$d/inputs.sha256"
    ;;
held)
    tv=${3:?training version}; version "$tv"; audit=$(realpath "${4:?independent TRAIN audit}")
    [[ $audit == "$s"/audit/train-v* ]]; m="$s/train-$tv"
    verify "$m"; verify "$audit"; listed "$audit/inputs.sha256" "$m/SHA256SUMS"
    jq -e '.pass' "$audit/readback.json" >/dev/null
    read -r release_hash release_manifest < "$m/inputs.sha256"; released "${release_manifest%/*}" "$release_hash"
    d="$s/permit-$v"; mkdir "$d"
    jq --arg hash "$(sha "$audit/SHA256SUMS")" '{pass,independent_train_manifest:$hash}' "$audit/readback.json" > "$d/verified.json"
    sha256sum "$d/verified.json" "$audit/"{SHA256SUMS,readback.json} "$here/SHA256SUMS" "$release_manifest" \
        "$m/"{SHA256SUMS,selection.json,selection.sha256} > "$d/inputs.sha256"
    sha256sum --strict --quiet -c "$d/inputs.sha256"; seal "$d"; sha256sum "$d/inputs.sha256"
    ;;
*) exit 64 ;;
esac

