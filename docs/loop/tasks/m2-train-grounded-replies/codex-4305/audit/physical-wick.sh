#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 5 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${5%/*}") == zfs ]]
build=$(realpath -e "$1")
seed=$(realpath -e "$2")
panel=$(realpath -e "$3")
physical=$(realpath -e "$4")
out=$5
for member in "$build" "$seed" "$panel/fresh" "$panel/near" "$panel/regression"; do
    (cd "$member" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd "$panel" && sha256sum --strict --quiet -c PANELS.sha256)
(cd "$physical" && sha256sum --strict --quiet -c raw.members)
raw=$physical/raw
audit=$build/argument-audit
state=$seed/slot3.state
bank=$seed/primary.words
cmp "$seed/protected.bridge" "$raw/ordinary/state.bridge"
cmp "$state" "$raw/ordinary/state.readback"
mkdir "$out"
for split in regression near fresh; do
    cmp "$raw/$split/baseline.jsonl" "$raw/$split/ordinary.jsonl"
    cmp "$raw/$split/baseline.jsonl" "$raw/ordinary/$split.native.jsonl"
    archived=()
    if [[ $split != fresh ]]; then
        archived+=("$seed/previous/$split/original.jsonl")
    fi
    "$audit" argument-captured "$panel/$split/ordered/text.bin" "$panel/$split/masks.tsv" \
        "$state" "$bank" "$raw/$split" "${archived[@]}" "$out/$split.json"
done
"$audit" physical-controls "$panel/fresh" "$seed/tokenizer.json" \
    "$raw/controls/gold.jsonl" "$raw/controls/swapped.jsonl" "$out/controls.json"
jq -e '.physical.signed_rows == [256,0] and .physical.full_rows == [256,0]' "$out/controls.json" > /dev/null
jq -s '
    {rows:(map(.rows)|add),
     native_scalar_fields:([.[].arms[].native_scalar_fields]|add),
     route_scalar_fields:([.[].arms[].route_scalar_fields]|add),
     archived_baselines_equal:([.[]|select(.archived_baseline_equal == true)]|length),
     original_off_equal:all(.[]; .native_off_equal),
     split:([.[] | {rows,masked_rows,arms}] )}
' "$out"/{regression,near,fresh}.json > "$out/summary.json"
sha256sum "$build/SHA256SUMS" "$seed/SHA256SUMS" "$panel/PANELS.sha256" \
    "$physical/raw.members" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
