#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
m=$(realpath -e "${1:?TRAIN or finite fixture directory}")
[[ $m == "$s/"* ]]
mkdir "$m/states"
for arm in endpoint ordered; do
    decision="$m/$arm/selected.json"
    selected=$(jq -er 'select((.qualified|type)=="boolean" and (.qualified or .chosen=="base"))
        | .chosen | select(.=="base" or .=="full" or test("^slot-(0[0-9]|1[0-5])$"))' "$decision")
    qualified=$(jq -r '.qualified' "$decision")
    for kind in selected full; do
        name=$arm; id=$selected; geom=$arm
        if [[ $kind == full ]]; then name=${arm}_full; id=full; fi
        state="$m/device/$arm/$id.state"
        if [[ $kind == selected && $qualified == false ]]; then
            geom=endpoint; state="$here/inputs/protected.state"
        fi
        output="$m/states/$name"
        cp "$state" "$output.state"
        "$here/inputs/bundle" pack "$geom" "$output.state" "$output.bridge"
        [[ $(stat -c %s "$output.bridge") == 13352 ]]
        [[ $("$here/inputs/bundle" unpack "$output.bridge" "$output.unpacked") == "$geom" ]]
        cmp "$output.state" "$output.unpacked"
    done
done
