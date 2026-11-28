#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 11 ]] || {
    echo 'usage: qualify.sh EXPANDED_EXTRACTION PRIOR_4340_PACKET NEW_REVIEW_A NEW_REVIEW_B REPAIR_PINS.json PATCHED_PRIOR_VALIDATION_CODE CLIENT BINDER BASE TOKENIZER FRESH_OUT' >&2
    exit 2
}
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=custody.sh
source "$here/custody.sh"
extract=$(realpath "$1")
packet=$(realpath "$2")
a=$(realpath "$3")
b=$(realpath "$4")
pins=$(realpath "$5")
driver=$(realpath "$6")
chat=$(realpath "$7")
binder=$(realpath "$8")
model=$(realpath "$9")
tokenizer=$(realpath "${10}")
out=$(realpath -m "${11}")

pin "$(jq -er .extraction "$pins")" "$extract/EXTRACTION.sha256"
pin "$(jq -er .prior_packet_members "$pins")" "$packet/MEMBERS.sha256"
pin "$(jq -er .additional_review_a "$pins")" "$a"
pin "$(jq -er .additional_review_b "$pins")" "$b"
if cmp -s "$a" "$b"; then fail 'The additional reviewer files are byte-identical'; fi
(
    cd "$extract"
    sha256 --strict --quiet -c EXTRACTION.sha256
    (cd selection && sha256 --strict --quiet -c SELECTION.sha256)
    (cd selected-semantic && sha256 --strict --quiet -c SEMANTIC.sha256)
)
verify_selection_seal "$extract/selection" "$extract/selection/registration.json"
prior=$packet/validation-selection-v1
verify_prior_selection "$prior"
verify_prior_semantic "$prior"
legacy=$packet/qualification-v1
for name in consensus native pairs evidence; do
    suffix=jsonl
    if [[ $name == evidence ]]; then suffix=json; fi
    member=qualification-v1/$name.$suffix
    pin "$(member_pin "$packet/MEMBERS.sha256" "./$member")" "$packet/$member"
done
old_a=$packet/semantic-a/decisions.jsonl
old_b=$packet/semantic-b/decisions.jsonl
pin cd2fb30947ed46fdc27dec9f553ec3f68162c3a693067ec4990c33ac789d1155 "$old_a"
pin 11d835299d7cc98a94d432a3c507604859e12df481f716b79749acdd9b67bc98 "$old_b"
for name in source rows; do
    cmp <(cat "$prior/selected-semantic/$name.jsonl" "$extract/selected-semantic/additional-$name.jsonl") \
        "$extract/selected-semantic/$name.jsonl"
done
jq -en --slurpfile roster "$extract/selection/additional-identities.jsonl" \
    --slurpfile a "$a" --slurpfile b "$b" '
    ($roster|map(.pair_id)|sort) as $ids |
    ($ids|length)==46 and all([$a,$b][]; map(.pair_id)|sort == $ids)
' >/dev/null

mkdir "$out"
cd "$out"
cat "$old_a" "$a" > review-a.jsonl
cat "$old_b" "$b" > review-b.jsonl
digest() { sha256 < "$1" | cut -d ' ' -f 1; }
jq -n --arg extraction "$(digest "$extract/EXTRACTION.sha256")" \
    --arg review_a "$(digest review-a.jsonl)" --arg review_b "$(digest review-b.jsonl)" \
    '{ $extraction,$review_a,$review_b,
       client:"21461b5af9c82d98f6f0c7f8780ae6f738bd601a52fe8a98edf25a7e07548381",
       binder:"91679e342515d27aa9e904be9712cf581668af2e3bd34d1cc1909d845721327d",
       model:"5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa",
       tokenizer:"a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377" }' > driver-pins.json
cp "$pins" repair-pins.json
bash "$driver/qualify-panel.sh" "$extract" review-a.jsonl review-b.jsonl driver-pins.json \
    "$chat" "$binder" "$model" "$tokenizer" qualification "$extract/selection/review-identities.jsonl"

for name in consensus pairs native; do
    jq -c --slurpfile original "$legacy/$name.jsonl" '
      ($original | map(.pair_id // .case_id) | INDEX(.)) as $ids |
      select($ids[.pair_id // .case_id] != null)
    ' "qualification/$name.jsonl" | cmp - "$legacy/$name.jsonl"
done
jq -n --slurpfile old "$legacy/evidence.json" --slurpfile expanded qualification/evidence.json \
    --slurpfile pairs qualification/pairs.jsonl --slurpfile selection "$extract/selection/prior-selection.json" \
    -f "$here/coverage.jq" > coverage.json
sha256 review-a.jsonl review-b.jsonl driver-pins.json repair-pins.json qualification/SEALED.sha256 coverage.json > SEALED.sha256
