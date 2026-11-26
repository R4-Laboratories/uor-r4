#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
v=${1:?new version}; version "$v"
panel=$(realpath -e "${2:?sealed panel}"); binary=$(realpath -e "${3:?sealed binary}")
finite=$(realpath -e "${4:?returned physical fixture}"); review=$(realpath -e "${5:?FIT directory}")
[[ $# == 5 && $panel == "$study"/panel/panel-v* && $binary == "$study"/device/build-v*/switch-competition-sm89 &&
   $finite == "$study"/device/finite-v* && $review == "$study"/audit/* ]]
for dir in "$here" "$panel" "${binary%/*}" "$finite" "$finite-lease" "$review"; do check_dir "$dir"; done
physical_inputs
[[ $(cat "$finite-lease/exit-code.txt") == 0 ]]
jq -e --arg payload "$(sha "$finite/SHA256SUMS")" --arg bin "$(sha "$binary")" '
    .provider=="vast" and .phase=="finite" and (.instance_id|test("^vast-[0-9]+$")) and
    (.gpu_name|contains("RTX 4090")) and (.gpu_uuid|test("^GPU-[0-9a-f-]+$")) and
    .binary_sha256==$bin and .payload_manifest_sha256==$payload
' "$finite-lease/provider.json" >/dev/null
grep -Fq 'TEST FITNESS: FIT' "$review/FIT.md"
for file in "$here/SHA256SUMS" "$here/PRE-RUN.md" "$here/VENUE.md" "$panel/SHA256SUMS" "$binary" \
    "$finite/SHA256SUMS" "$finite-lease/SHA256SUMS"; do
    grep -Fq "$(sha "$file")" "$review/FIT.md"
done
r="$study/release-$v"; mkdir "$r"
cp "$here/"{ORDER,PRE-RUN,DECISION,VENUE}.md "$review/FIT.md" "$r/"
[[ $("$bundle" unpack "$prior/train-v1/states/ordered.bridge" "$r/ordered.unpacked") == ordered ]]
cmp "$refit" "$r/ordered.unpacked"
jq -n --arg panel "$panel" --arg binary "$binary" --arg finite "$finite" --arg review "$review" \
    --slurpfile physical "$finite-lease/provider.json" \
    '{panel:$panel,binary:$binary,finite:$finite,review:$review,hardware:$physical[0]}' > "$r/paths.json"
sha256sum "$r/"{ORDER.md,PRE-RUN.md,DECISION.md,VENUE.md,FIT.md,paths.json,ordered.unpacked} "$here/SHA256SUMS" \
    "$panel/SHA256SUMS" "$binary" "${binary%/*}/SHA256SUMS" "$finite/SHA256SUMS" "$finite-lease/SHA256SUMS" \
    "$review/SHA256SUMS" "$bank" "$original" "$refit" "$bundle" "$lease" "$prior/train-v1/SHA256SUMS" \
    "$prior/held-v1/SHA256SUMS" "$prior/panel/panel-v1/SHA256SUMS" > "$r/inputs.sha256"
seal "$r"; sha256sum "$r/SHA256SUMS"
