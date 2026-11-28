#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# == 11 ]] || {
    echo 'usage: qualify.sh INVENTORY INVENTORY_SHA PACKET_A PACKET_A_SHA RETURN_A RETURN_A_SHA PACKET_B PACKET_B_SHA RETURN_B RETURN_B_SHA FRESH_DESTINATION' >&2
    exit 2
}
here=$(cd "$(dirname "$0")" && pwd -P)
inventory=$(realpath "$1")
packet_a=$(realpath "$3")
review_a=$(realpath "$5")
packet_b=$(realpath "$7")
review_b=$(realpath "$9")
out=$(realpath -m "${11}")

mkdir "$out"
printf '%s  %s\n' "$2" "$inventory/INVENTORY.sha256" "$4" "$packet_a/PACKET.sha256" "$6" "$review_a" \
  "$8" "$packet_b/PACKET.sha256" "${10}" "$review_b" > "$out/INPUTS.sha256"
sha256sum --quiet --strict -c "$out/INPUTS.sha256"
(cd "$inventory" && sha256sum --quiet --strict -c INVENTORY.sha256)
for packet in "$packet_a" "$packet_b"; do
    (cd "$packet" && sha256sum --quiet --strict -c PACKET.sha256)
    rows_sha=$(sha256sum "$packet/rows.jsonl" | cut -d ' ' -f 1)
    jq -e --arg inventory "$2" --arg rows "$rows_sha" '
      .schema=="uor-r4/directed-relation-review-request/1" and .split=="screen" and
      .inventory_sha256==$inventory and .rows_sha256==$rows and (.reviewer|test("^[a-zA-Z0-9][a-zA-Z0-9._-]*$"))
    ' "$packet/request.json" >/dev/null
done
[[ $(jq -er .reviewer "$packet_a/request.json") != $(jq -er .reviewer "$packet_b/request.json") ]]
[[ $6 != "${10}" ]]
cmp "$packet_a/rows.jsonl" "$packet_b/rows.jsonl"
jq -r .group_id "$packet_a/rows.jsonl" |
    cmp - <(jq -r 'select(.split=="screen").group_id' "$inventory/review.jsonl")

jq -ne --rawfile review_a "$review_a" --rawfile review_b "$review_b" --slurpfile selected "$inventory/review.jsonl" \
   -f "$here/qualify.jq" > "$out/qualification.json"
jq '.summary' "$out/qualification.json" > "$out/counts.json"
jq -c '.accepted[] | .group as $g | .rows[] | {case_id,text:($g.context+" "+.question)}' \
    "$out/qualification.json" > "$out/blind.jsonl"
jq -c '.accepted[] | .group as $g | .rows[] |
  {case_id,question_source_id,question,original_bytes,original_char,span_text,family} +
  ($g|{split,article,paragraph,passage_sha256,passage_bytes})' "$out/qualification.json" > "$out/truth.jsonl"
jq -c '.accepted[] | .decision_a.contrast as $contrast | .group |
  {group_id,split,article,paragraph,passage_sha256,$contrast,
   stratum:(if .stratum=="same-target" then "paraphrase" else .stratum end),
   families:[.families[]|map(.case_id)]}' "$out/qualification.json" > "$out/groups.jsonl"

(cd "$out" && sha256sum INPUTS.sha256 qualification.json counts.json blind.jsonl truth.jsonl groups.jsonl > SUPPLY.sha256)
