#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; mv=${2:?measurement version}; hv=${3:?new held version}; version "$mv"; version "$hv"
permit=$(realpath "${4:?TRAIN replay permit}"); expected=${5:?permit sha}
[[ $permit == "$s"/train-verified-v*/inputs.sha256 ]]; pin "$expected" "$permit"; sha256sum --strict --quiet -c "$permit"
m="$s/measurement-$mv"; out="$s/held-$hv"
for f in "$m/"{SHA256SUMS,selection.json,selection.sha256} "$here/SHA256SUMS" "$(dirname "$permit")/verified.json"; do listed "$permit" "$f"; done
jq -e '.verified==true and .batches==6' "$(dirname "$permit")/verified.json" >/dev/null
manifest=$(cut -d' ' -f3- "$m/inputs.sha256"); release_inputs "$manifest" "$(cut -d' ' -f1 "$m/inputs.sha256")"
if [[ $mode == run ]]; then exec bash "$lock" "$out-lease" bash "$0" inner "$mv" "$hv" "$permit" "$expected"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]; verify "$panel"; verify "$m"; sha256sum --strict --quiet -c "$m/selection.sha256"
mkdir "$out" "$out/unpacked"
for arm in strongest all ordered-all; do
    "$packer" unpack "$m/states/$arm.bridge" "$out/unpacked/$arm.state" > "$out/unpacked/$arm.mode"
    cmp "$out/unpacked/$arm.state" "$m/states/$arm.state"
done
for domain in held contrast; do
    path="$panel/$domain"; d="$out/$domain"; mkdir "$d"
    read_arm(){
        local name=$1 kind=$2 geom=$3; shift 3
        local input="$path/blind/text.bin"
        if [[ $geom == ordered ]]; then input="$path/ordered/text.bin"; else [[ $geom == endpoint ]]; fi
        /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$d/$name.device-time.txt" \
            "$reader" "$kind" "$banks" "$input" "$@" "$d/$name.raw.jsonl"
        local label=$kind; if [[ $kind == correct || $kind == exchanged ]]; then label=${kind}_link; fi
        join_rows "$domain" "$path" "$label" "$d/$name.raw.jsonl" > "$d/$name.rows.jsonl"
        pair_rows "$domain" < "$d/$name.rows.jsonl" > "$d/$name.pairs.json"
    }
    read_arm native native endpoint; read_arm unfitted unfitted endpoint
    read_arm gap governor_gap endpoint "$state"; read_arm ordered_gap governor_gap ordered "$state"
    read_arm baseline learned endpoint "$state"; read_arm ordered_base learned ordered "$state"
    read_arm far_only learned endpoint "$far_state"
    for arm in strongest all ordered-all; do
        geom=$(cat "$out/unpacked/$arm.mode"); gap=gap; if [[ $geom == ordered ]]; then gap=ordered_gap; fi
        read_arm "$arm" learned "$geom" "$out/unpacked/$arm.state"
        input="$path/blind/text.bin"; if [[ $geom == ordered ]]; then input="$path/ordered/text.bin"; fi
        "$replay" "$input" "$banks" "$out/unpacked/$arm.state" "$d/$arm.raw.jsonl" "$d/$gap.raw.jsonl" > "$d/$arm.replayed.json"
        jq -e --argjson n "$(wc -l < "$d/$arm.raw.jsonl")" '.pass and .learned.rows==$n and
           .learned.disagreements==0 and .governor_gap.disagreements==0' "$d/$arm.replayed.json" >/dev/null
    done
    read_arm correct_link correct endpoint "$path/aid.bin"; read_arm exchanged_link exchanged endpoint "$path/aid.bin"
done
sha256sum "$permit" "$m/SHA256SUMS" "$panel/SHA256SUMS" "$here/SHA256SUMS" > "$out/inputs.sha256"
seal "$out"
