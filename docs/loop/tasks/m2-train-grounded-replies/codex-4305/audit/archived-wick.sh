#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 3 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${3%/*}") == zfs ]]
cold=$(realpath -e "$1")
build=$(realpath -e "$2")
out=$3
input=$cold/inputs/train-input
archived=$cold/physical/train/score
prior=$cold/independent/predecessor-v1
audit=$build/argument-audit
codec=$cold/inputs/tokenizer.json
state=$input/models/slot3.state
banks=$input/models/primary.words
printf '%s  %s\n' 048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d "$cold/MEMBERS.sha256" |
    sha256sum --strict --quiet -c -
(cd "$build" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$cold" && awk '
    $2 ~ /^\.\/panel\/(regression|near)\// ||
    $2 ~ /^\.\/physical\/train\/score\/(regression|near)\/b\.(native|costs)\.jsonl$/ ||
    $2 ~ /^\.\/independent\/predecessor-v1\/(regression|near)\.json$/ ||
    $2 ~ /^\.\/inputs\/train-input\/(models\/(primary\.words|slot3\.state)|(regression|near)\/(ordered\/text\.bin|reference\.native\.jsonl))$/ ||
    $2 == "./inputs/tokenizer.json" {print}
' MEMBERS.sha256 | sha256sum --strict --quiet -c -)
mkdir "$out"
for split in regression near; do
    cmp "$cold/panel/$split/ordered/text.bin" "$input/$split/ordered/text.bin"
    cmp "$archived/$split/b.native.jsonl" "$input/$split/reference.native.jsonl"
    "$audit" argument-archived "$input/$split/ordered/text.bin" "$state" "$banks" \
        "$archived/$split/b.native.jsonl" "$out/$split-native.json"
done
"$audit" replay "$cold/panel/regression" "$codec" "$state" "$banks" "$archived/regression/b.native.jsonl" \
    "$archived/regression/b.costs.jsonl" "$out/regression.json"
cmp "$out/regression.json" "$prior/regression.json"
jq -e '(.pairs|length)==4 and all(.pairs[]; .follows)' "$out/regression.json" > /dev/null

# Restore the original custody view from the previously independently sealed near members.
jq '{protected_near:(.protected_pairs|keys),
    protected_members:[.members[]|{pair:.pair_id,prompt_id:.original_id,admission,prompt_sha256,
    gold_governors:.governor_addresses,gold_nouns:.site_addresses}]}' "$prior/near.json" > "$out/near-premise.json"
"$audit" near-replay "$cold/panel/near" "$codec" "$out/near-premise.json" "$cold/panel/near/gold.jsonl" \
    "$state" "$banks" "$archived/near/b.native.jsonl" "$out/near.json"
cmp "$out/near.json" "$prior/near.json"
jq -e '(.protected_pairs|length)==24 and (.members|length)==48 and (.lost_protected|length)==0' "$out/near.json" > /dev/null
jq -n --slurpfile reg "$out/regression.json" --slurpfile near "$out/near.json" \
    --slurpfile regraw "$out/regression-native.json" --slurpfile nearraw "$out/near-native.json" '
    {signed_pairs:[$reg[0].pairs[].id],near_pairs:($near[0].protected_pairs|keys),
    near_rows:[$near[0].members[].row_index],native_scalar_fields:($regraw[0].scalar_fields+$nearraw[0].scalar_fields)}
' > "$out/summary.json"
sha256sum "$build/SHA256SUMS" "$cold/MEMBERS.sha256" "$state" "$banks" "$codec" \
    "$archived"/{regression,near}/b.native.jsonl "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
