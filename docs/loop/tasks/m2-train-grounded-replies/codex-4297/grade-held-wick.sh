#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mv=${1:?TRAIN version}; hv=${2:?held version}; gv=${3:?new grade version}
for v in "$mv" "$hv" "$gv"; do version "$v"; done
m="$s/measurement-$mv"; raw="$s/held-$hv"; g="$s/grade-$gv"
verify "$m"; verify "$raw"; sha256sum --strict --quiet -c "$raw/inputs.sha256"
listed "$raw/inputs.sha256" "$m/SHA256SUMS"
manifest=$(cut -d' ' -f3- "$m/inputs.sha256")
physical_methods=$(sed -n "s@^[a-f0-9]\{64\}  \($s/methods-v[0-9]*/SHA256SUMS\)\$@\1@p" "$raw/inputs.sha256")
[[ -f $physical_methods ]]; original=${physical_methods%/SHA256SUMS}
(source "$original/common-wick.sh"; release_inputs "$manifest" "$(cut -d' ' -f1 "$m/inputs.sha256")")
verify "$here"; panel=$(cat "$(dirname "$manifest")/panel-path")
# A reporting fix may use a newer sealed reducer. The original join, grading, and
# release criteria must remain byte-identical to the release used for the device run.
for file in original-join.jq original-pairs.jq original-summary.jq grade/{gate,elect,candidate,witness,contrast,contrast-pairs}.jq; do
    cmp "$original/$file" "$here/$file"
done
verify "$panel"; listed "$raw/inputs.sha256" "$panel/SHA256SUMS"
bash "$here/grade-outputs-wick.sh" "$m" "$raw" "$panel" "$g"
