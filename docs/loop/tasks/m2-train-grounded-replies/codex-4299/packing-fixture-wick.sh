#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
v=${1:?new packing fixture version}; version "$v"
m="$s/train-v1"; out="$s/packing-fixture-$v"
verify "$here"; verify "$m"; physical_pins
names=$(states | jq -R . | jq -sc .)
mkdir "$out"

# Reuse the actual candidate bytes. Replace protected membership only in the
# deliberately infeasible elections. The packer consumes the regular election files.
for variant in both endpoint-only ordered-only none; do
    work="$out/$variant"; mkdir -p "$work/device"
    for arm in endpoint ordered; do
        mkdir "$work/$arm" "$work/device/$arm"
        cp "$m/device/$arm/"{base,full}.state "$work/device/$arm/"
        if [[ $variant == both || $variant == "$arm-only" ]]; then
            cp "$m/$arm/candidates.jsonl" "$work/$arm/candidates.jsonl"; ok=true
        else
            jq -c '.lost_protected=["fixture-missing-original-pair"]' "$m/$arm/candidates.jsonl" > "$work/$arm/candidates.jsonl"; ok=false
        fi
        jq -es --arg arm "$arm" --argjson schedule "$names" -f "$here/grade/elect.jq" \
            "$work/$arm/candidates.jsonl" > "$work/$arm/selected.json"
        jq -e --argjson ok "$ok" '.qualified==$ok and .chosen==(if $ok then "full" else "base" end)' \
            "$work/$arm/selected.json" >/dev/null
    done
    bash "$here/pack-states-wick.sh" "$work"
    for arm in endpoint ordered; do
        for lane in "$arm" "${arm}_full"; do
            state="$m/states/${arm}_full.state"; expected=$arm
            if [[ $lane == "$arm" && $variant != both && $variant != "$arm-only" ]]; then
                state="$here/inputs/protected.state"; expected=endpoint
            fi
            emitted="$work/states/$lane"
            actual=$("$here/inputs/bundle" unpack "$emitted.bridge" "$emitted.check")
            [[ $actual == "$expected" ]]; cmp "$state" "$emitted.check"
            if [[ $variant == both ]]; then cmp "$emitted.bridge" "$m/states/$lane.bridge"; fi
            jq -nc --arg case "$variant" --arg lane "$lane" --arg mode "$actual" --arg sha "$(sha "$emitted.check")" \
                '{case:$case,lane:$lane,mode:$mode,state_sha256:$sha}' >> "$out/lanes.jsonl"
        done
    done
done

# The two tags unpack to the same payload with different geometries.
bad="$out/wrong-tag"
"$here/inputs/bundle" pack ordered "$here/inputs/protected.state" "$bad.bridge"
wrong=$("$here/inputs/bundle" unpack "$bad.bridge" "$bad.state")
cmp "$here/inputs/protected.state" "$bad.state"
[[ $wrong == ordered && $wrong != endpoint ]]
if cmp -s "$bad.bridge" "$out/none/states/ordered.bridge"; then
    echo "Wrong operator was accepted as the endpoint fallback" >&2
    exit 1
fi

jq -es --arg protected "$(sha "$here/inputs/protected.state")" \
    'if length==16 then {pass:true,variants:4,unpacked_bundles:length,actual_election_bundles_identical:4,
        wrong_tag_changes_operator_with_identical_payload:true,protected_state_sha256:$protected,lanes:.}
     else error("incomplete fixture") end' "$out/lanes.jsonl" > "$out/receipt.json"
sha256sum "$here/SHA256SUMS" "$m/SHA256SUMS" "$here/inputs/protected.state" "$here/inputs/bundle" > "$out/inputs.sha256"
seal "$out"; sha256sum "$out/SHA256SUMS"
