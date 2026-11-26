#!/usr/bin/env bash
# Test the adapter and independent grade on previously sealed traces. This never calls a model.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
client_root="$runs/20261126-codex-governor-noun-links/client"
old="$runs/20261125-codex-governing-source-pairs/held/primary"
donor="$runs/20261125-codex-bound-instruction-state"
prior="$donor/ordinary-recipe-final-v1"
codec="$donor/custody-v1/signed/tokenizer.json"
out="$client_root/${1:-fixture-v1}"
test "$(hostname -s)" = wick
[[ $out =~ ^$client_root/fixture-v[1-9][0-9]*$ ]]
test "$(findmnt -no FSTYPE --target "$client_root")" = zfs
umask 077
mkdir "$out"
ref="$old/panel-v1/fresh/reference.json"
correct="$old/capture-controls-v1/correct/fresh/rows.jsonl"
native="$old/capture-baseline-v1/baseline/fresh/rows.jsonl"
stock="$old/capture-stock-v1/stock/fresh/rows.jsonl"
for file in "$ref" "$correct" "$native" "$stock"; do
    dir=${file%%/fresh/*}
    if [[ $dir != "$old/panel-v1" ]]; then dir=${dir%/*}; fi
    rel="./${file#"$dir"/}"
    line=$(awk -v path="$rel" '$2==path' "$dir/SHA256SUMS")
    test -n "$line"
    (cd "$dir" && printf '%s\n' "$line" | sha256sum --strict --quiet -c -)
done
sha256sum "$ref" "$correct" "$native" "$stock" "$codec" "$prior/"{score,raw}.jq \
    "$here/"{plan,common,grade,summary}.jq "$0" > "$out/inputs.sha256"

jq -nce --slurpfile ref "$ref" '
  INDEX($ref[0][]; .id) as $refs |
  first(inputs | . as $a | $refs[$a.id] as $r |
      select($r.admission=="take" and .record.turns[1].stop=="eos" and
        .record.turns[1].role_action_trace.bound_instruction.release_reached and
        .record.turns[1].reply_ids==$r.effective_grounded.nominal.expected))' "$correct" > "$out/take.jsonl"
take_id=$(jq -er .id "$out/take.jsonl")
defer_id=$(jq -er --arg id "$take_id" '
  (first(.[] | select(.id==$id)).admission_pair) as $pair |
  first(.[] | select(.admission=="defer" and .admission_pair==$pair)).id' "$ref")
get_raw() { jq -nce --arg id "$1" 'first(inputs | select(.id==$id))' "$2"; }
get_raw "$defer_id" "$correct" > "$out/defer.jsonl"
get_raw "$defer_id" "$native" > "$out/native.jsonl"
get_raw "$take_id" "$stock" > "$out/stock.jsonl"
get_raw "$defer_id" "$stock" >> "$out/stock.jsonl"
jq -c --arg a "$take_id" --arg b "$defer_id" --slurpfile t "$out/take.jsonl" --slurpfile d "$out/defer.jsonl" '
  INDEX(($t[0],$d[0]);.id) as $captures |
  map(select(.id==$a or .id==$b) |
      $captures[.id].record.turns[1].role_action_trace.bound_instruction.binding as $bound |
      . + {prompt:.stock.prompt,structural_history:.stock.earlier,structural_prompt_sha256:("0"*64),
           distance:"fixture",native_scope:[$bound.affirmed,$bound.excluded,$bound.referenced]})' \
   "$ref" > "$out/reference.json"

jq -nce --slurpfile reference "$out/reference.json" --slurpfile n "$out/native.jsonl" '
  INDEX($reference[0][]; .id) as $rows |
  inputs | $rows[.id] as $row |
  .record.turns[1].role_action_trace.bound_instruction as $bound |
  [$bound.binding.affirmed,$bound.binding.excluded,$bound.binding.referenced] as $winner |
  {selection:($winner | map(.bytes|map_values(.-1))),tokens:($winner|map(.tokens))} as $control |
  ($n[0].record.turns[1].role_action_trace.bound_instruction.selections[0]) as $learned |
  {case_id:.id,pair_id:$row.admission_pair,family:$row.family,admission:$row.admission,distance:"fixture",
   prompt_sha256:$row.structural_prompt_sha256,gold_tokens:($row.native_scope | map(.tokens)),
   gold:[$row.gold_binding.affirmed,$row.gold_binding.excluded,$row.gold_binding.referenced],
   arms:{native:(if .id!=$n[0].id then $control elif $learned.status=="chosen"
      then {selection:($learned.witnesses|map(.bytes|map_values(.-1))),tokens:($learned.witnesses|map(.tokens)),bank_cost:$learned.distance}
      else {selection:null,tokens:null} end),
     governor_only:{selection:null,tokens:null},nouns_only:$control,correct_link:$control,exchanged_link:$control}}' \
   "$out/take.jsonl" "$out/defer.jsonl" > "$out/device.jsonl"
jq -ce -L "$here" --slurpfile device "$out/device.jsonl" -f "$here/plan.jq" \
   "$out/reference.json" > "$out/plan.jsonl"
jq -es 'length==10 and
   all(.[] | select(.arm=="native"); .wire.learned_field=="scope" and (.wire | has("affirmed")|not)) and
   all(.[] | select(.arm=="governor_only"); .wire.affirmed=={start:0,end:0})' \
   "$out/plan.jsonl" >/dev/null
jq -c '.prompt_sha256=("a"*64)' "$out/device.jsonl" > "$out/changed-prompt-device.jsonl"
if jq -ce -L "$here" --slurpfile device "$out/changed-prompt-device.jsonl" -f "$here/plan.jq" \
     "$out/reference.json" > "$out/changed-prompt-plan.jsonl" 2> "$out/changed-prompt.log"; then
    echo "Device output carrying a different prompt digest unexpectedly passed" >&2
    exit 1
fi
rg -q 'source enrollment differs' "$out/changed-prompt.log"

jq -c '.record.turns[1].role_action_trace.bound_instruction |=
  (.status="invalid_binding" | .binding=null |
   .scope={affirmed:{start:1,end:1},excluded:{start:1,end:1},referenced:{start:1,end:1}} |
   .fields.scope=.scope)' "$out/defer.jsonl" > "$out/abstain.jsonl"
jq -c '.record.turns[1].reply_ids[0]=42' "$out/take.jsonl" > "$out/changed-token.jsonl"
jq -c '.record.turns[1].reply_text="wrong visible reply"' "$out/take.jsonl" > "$out/changed-text.jsonl"
jq -c '.record.turns[1].reply_text=""' "$out/defer.jsonl" > "$out/empty-text.jsonl"
jq -sc 'map(if .admission=="take" then .arms.correct_link.tokens[0].start+=1 else . end)[]' \
   "$out/device.jsonl" > "$out/changed-address.jsonl"

grade() {
    local arm=$1 observed=$2 endpoints=${3:-"$out/device.jsonl"} reference=${4:-"$out/reference.json"}
    jq -nce -L "$here" -L "$prior" --arg arm "$arm" \
       --arg model 06a7d1f197870a193699b75dd58ce6269455da3047da87f032de1d45600ccfa2 \
       --arg stock_model 2e16a8ea27b316cce01dac5744c6b02da38e0ee6c0abc3fcb610e381b1c85302 \
       --arg codec_hash a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 \
       --slurpfile reference "$reference" --slurpfile device "$endpoints" --slurpfile codec "$codec" \
       --slurpfile stock "$out/stock.jsonl" -f "$here/grade.jq" "$observed"
}
grade correct_link "$out/take.jsonl" > "$out/take-grade.json"
grade correct_link "$out/defer.jsonl" > "$out/defer-grade.json"
grade native "$out/native.jsonl" > "$out/native-grade.json"
grade governor_only "$out/abstain.jsonl" > "$out/abstain-grade.json"
grade correct_link "$out/take.jsonl" "$out/changed-address.jsonl" > "$out/changed-address-grade.json"
if grade correct_link "$out/changed-token.jsonl" > "$out/changed-token-grade.json" 2> "$out/changed-token.log"; then
    echo "Changed token unexpectedly passed raw grade" >&2
    exit 1
fi
rg -q 'claimed reply IDs or EOS do not match the returned raw steps' "$out/changed-token.log"
for mutation in changed-text empty-text; do
    if grade correct_link "$out/$mutation.jsonl" > "$out/$mutation-grade.json" 2> "$out/$mutation.log"; then
        echo "Changed text unexpectedly passed raw grade for $mutation" >&2
        exit 1
    fi
    rg -q 'returned IDs and stop match while visible target text differs' "$out/$mutation.log"
done
# An abstaining TAKE matches stock rather than the grounded target. Visible
# parity with stock must still be verified even when the target IDs differ.
jq -c --arg id "$defer_id" 'map(if .id==$id then .admission="take" else . end)' \
   "$out/reference.json" > "$out/stock-text-reference.json"
jq -c --arg id "$defer_id" 'if .case_id==$id then .admission="take" else . end' \
   "$out/device.jsonl" > "$out/stock-text-device.jsonl"
jq -c '.record.turns[1].reply_text="wrong visible stock"' "$out/abstain.jsonl" > "$out/stock-text-check.jsonl"
if grade governor_only "$out/stock-text-check.jsonl" "$out/stock-text-device.jsonl" "$out/stock-text-reference.json" \
     > "$out/stock-text-grade.json" 2> "$out/stock-text.log"; then
    echo "Changed stock text unexpectedly passed raw grade" >&2
    exit 1
fi
rg -q 'returned stock IDs and stop match while visible stock text differs' "$out/stock-text.log"
jq -e '.authenticated_take_eos and .correct_route and .source_identity_ok and .returned.first_wrong==null' "$out/take-grade.json"
jq -e '.authenticated_defer_parity and .disposition.zero_bound_emission and .source_identity_ok and
       .returned.correct_prefix_at_cap and (.exact_terminal | not)' "$out/defer-grade.json"
jq -e '.source_identity_ok' "$out/native-grade.json"
jq -e '.disposition.valid_abstention and .source_identity_ok and (.authenticated_defer_parity|not)' "$out/abstain-grade.json"
jq -e '(.source_identity_ok|not) and (.authenticated_take_eos|not)' "$out/changed-address-grade.json"
jq -nce -L "$prior" 'include "score";
  (first_error([7,8,1];[7,1];true)=={index:1,expected_id:8,actual_id:1,reason:"early_eos"}) and
  (first_error([7,8,1];[7,8];true)=={index:2,expected_id:1,actual_id:null,reason:"missing"}) and
  (first_error([7,8,1];[7,8,1];true)==null)'
jq -nce -L "$here" 'include "common"; [range(2;50)] as $ids |
  ({ids:$ids,stop:"max_tokens"} | correct_prefix_at_cap($ids) and correct_prefix_at_cap($ids+[1])) and
  ({ids:($ids | .[4]=0),stop:"max_tokens"} | correct_prefix_at_cap($ids) | not) and
  ({ids:$ids,stop:"eos"} | correct_prefix_at_cap($ids) | not)'
jq -nce -L "$here" --slurpfile take "$out/take-grade.json" --slurpfile defer "$out/defer-grade.json" '
  include "common"; arms[] as $arm | ($take[0],$defer[0]) | .arm=$arm' > "$out/pairs.jsonl"
jq -es -L "$here" --argjson pair_count 1 -f "$here/summary.jq" "$out/pairs.jsonl" > "$out/pair-grade.json"
jq -e 'all(.arms[]; .paired_take_eos_and_defer_stock==1 and
  .all.returned_correct_prefix_at_cap==1 and .all.exact_terminal==1)' "$out/pair-grade.json"
# shellcheck disable=SC2094
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS)
