#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; v=${2:?new TRAIN version}; version "$v"; release=$(realpath "${3:?release}"); hash=${4:?release digest}
released "$release" "$hash"; m="$s/train-$v"
if [[ $mode == run ]]; then exec bash "$lock" "$m-lease" bash "$0" inner "$v" "$release" "$hash"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]
physical_pins; mkdir "$m" "$m/endpoint" "$m/ordered"
sha256sum "$release/inputs.sha256" > "$m/inputs.sha256"
t="$panel/mixed"; base="$here/inputs/protected.state"
for file in blind/text.bin ordered/text.bin links.bin reference.json blind/coordinates.jsonl; do cmp "$t/$file" "$parent/panel/panel-v1/mixed/$file"; done
/usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$m/fitter.time" \
    "$fitter" "$t/blind/text.bin" "$t/ordered/text.bin" "$t/links.bin" "$base" "$m/device"
for arm in endpoint ordered; do
    tape="$t/blind/text.bin"; if [[ $arm == ordered ]]; then tape="$t/ordered/text.bin"; fi
    cmp "$base" "$m/device/$arm/base.state"
    for id in $(states); do
        "$reader" learned "$banks" "$tape" "$m/device/$arm/$id.state" "$m/$arm/$id.raw.jsonl"
    done
    cmp <(jq -Sc . "$here/inputs/$arm.original.jsonl") <(jq -Sc . "$m/$arm/base.raw.jsonl")
done
bash "$here/reduce-train-wick.sh" "$m" "$panel" "$m"
bash "$here/pack-states-wick.sh" "$m"
sha256sum "$m/states/"*.state "$m/states/"*.bridge "$m/selection.json" > "$m/selection.sha256"
seal "$m"; sha256sum "$m/SHA256SUMS"

