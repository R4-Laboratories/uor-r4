#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; tv=${2:?TRAIN}; hv=${3:?new held}; version "$tv"; version "$hv"
permit=$(realpath "${4:?TRAIN replay permit}"); hash=${5:?permit digest}
[[ $permit == "$s"/permit-v* ]]
pin "$hash" "$permit/inputs.sha256"; sha256sum --strict --quiet -c "$permit/inputs.sha256"
jq -e '.pass' "$permit/verified.json" >/dev/null; listed "$permit/inputs.sha256" "$permit/verified.json"
m="$s/train-$tv"; h="$s/held-$hv"
for f in "$m/"{SHA256SUMS,selection.json,selection.sha256} "$here/SHA256SUMS"; do listed "$permit/inputs.sha256" "$f"; done
read -r release_hash release_manifest < "$m/inputs.sha256"; released "${release_manifest%/*}" "$release_hash"
if [[ $mode == run ]]; then exec bash "$lock" "$h-lease" bash "$0" inner "$tv" "$hv" "$permit" "$hash"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]; verify "$m"; sha256sum --strict --quiet -c "$m/selection.sha256"
mkdir "$h" "$h/unpacked"
for arm in endpoint ordered endpoint_full ordered_full; do
    "$here/inputs/bundle" unpack "$m/states/$arm.bridge" "$h/unpacked/$arm.state" > "$h/unpacked/$arm.mode"
    cmp "$m/states/$arm.state" "$h/unpacked/$arm.state"
done
for domain in held contrast; do
    d="$h/$domain"; path="$panel/$domain"; mkdir "$d"
    run_arm() {
        local name=$1 op=$2 geom=$3 file="$path/blind/text.bin" label=$2; shift 3
        if [[ $geom == ordered ]]; then file="$path/ordered/text.bin"; else [[ $geom == endpoint ]]; fi
        /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$d/$name.time" \
            "$reader" "$op" "$banks" "$file" "$@" "$d/$name.raw.jsonl"
        if [[ $op == correct || $op == exchanged ]]; then label=${op}_link; fi
        join_rows "$domain" "$path" "$label" "$d/$name.raw.jsonl" > "$d/$name.rows.jsonl"
        pair_rows "$domain" < "$d/$name.rows.jsonl" > "$d/$name.pairs.json"
    }
    run_arm native native endpoint
    run_arm unfitted unfitted endpoint
    run_arm gap governor_gap endpoint "$here/inputs/protected.state"
    run_arm ordered_gap governor_gap ordered "$here/inputs/protected.state"
    run_arm baseline learned endpoint "$here/inputs/protected.state"
    run_arm ordered_base learned ordered "$here/inputs/protected.state"
    run_arm far_only learned endpoint "$here/inputs/far_only.state"
    for arm in endpoint ordered endpoint_full ordered_full; do
        geom=$(cat "$h/unpacked/$arm.mode"); run_arm "$arm" learned "$geom" "$h/unpacked/$arm.state"
        gap=gap; file="$path/blind/text.bin"; if [[ $geom == ordered ]]; then gap=ordered_gap; file="$path/ordered/text.bin"; fi
        "$replay" "$file" "$banks" "$h/unpacked/$arm.state" "$d/$arm.raw.jsonl" "$d/$gap.raw.jsonl" > "$d/$arm.replayed.json"
        jq -e --argjson n "$(wc -l < "$d/$arm.raw.jsonl")" \
            '.pass and .learned.rows==$n and .learned.disagreements==0 and .governor_gap.disagreements==0' "$d/$arm.replayed.json" >/dev/null
    done
    run_arm correct_link correct endpoint "$path/aid.bin"
    run_arm exchanged_link exchanged endpoint "$path/aid.bin"
done
sha256sum "$permit/inputs.sha256" "$m/SHA256SUMS" "$panel/SHA256SUMS" "$here/SHA256SUMS" > "$h/inputs.sha256"
seal "$h"; sha256sum "$h/SHA256SUMS"

