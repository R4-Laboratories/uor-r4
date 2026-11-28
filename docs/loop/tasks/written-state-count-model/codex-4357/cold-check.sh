#!/usr/bin/env bash
# Replay only the preregistered ordinal-zero opening through the ordinary client.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_VISIBLE_DEVICES=''

if [[ $# != 4 ]]; then
  echo 'usage: cold-check.sh RUN_ROOT VERIFIED_RECEIPT ARCHIVED_SNAPPED_STDOUT NEW_OUTPUT_DIR' >&2
  exit 2
fi

run=$1
verified=$2
archived=$3
out=$4
receipt="$run/pod-members.sha256"
client="$run/bin/uor-chat"
p4=$(realpath -- "$run/p4.d11")
inputs="$run/input/train/inputs.jsonl"
previous="$run/previous-trial"
sentinel="$run/attempt01/trial/sentinel.json"
artifact=e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776
state=368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778
train_sha=5dd411d7dee1335784efc5868adf58a6979da18d4b743809e36b12bd44dc6829
client_expected=4104c436300948e4e1aa3e8775b4bb52c9ec54b13f70c820c3eb19d99226dea4
old_trace_sha=cd96348fc6c98bc55a7122ce011ee1b0e9b201c83c422f30c8c1bfef6429a372
old_witness_sha=e7a6c56b96c30c0dfbd54cdace21f44c821105985aada6810a3bb0dab71d07a0
old_arm=geometry-full-435300-uncentered
sha() { sha256sum -- "$1" | cut -d' ' -f1; }

[[ "$verified" == "$run/receipt-verified" && -f "$verified" ]]
[[ -s "$receipt" && -x "$client" && -s "$sentinel" ]]
mkdir -- "$out"
[[ $(sha "$p4") == "$artifact" && $(sha "$inputs") == "$train_sha" ]]
[[ $(sha "$archived") == "$old_trace_sha" && $(sha "$previous/witnesses.jsonl") == "$old_witness_sha" ]]
client_sha=$(sha "$client")
[[ "$client_sha" == "$client_expected" ]]
receipt_sha=$(sha "$receipt")
sentinel_sha=$(sha "$sentinel")
[[ $(awk '$2 ~ /(^|\/)attempt01\/trial\/sentinel\.json$/ {print $1}' "$receipt") == "$sentinel_sha" ]]
jq -ce --arg arm "$old_arm" 'select(.name == $arm)' \
  "$previous/witnesses.jsonl" > "$out/archived-device.json"

ids=$(jq -ner --slurpfile previous "$previous/sentinel-prefix.json" '
  input.lexical_prefix as $ids
  | if ($previous | length) == 1 and $ids == $previous[0]
       and ($ids | type == "array" and length > 0 and length <= 2048 and .[0] == 0
                    and all(.[]; type == "number" and . == floor and . >= 0 and . < 4096))
    then $ids | join(",") else error("frozen ordinal-zero prefix differs") end
' "$inputs")
jq -es '
  def word($max): type == "number" and . == floor and . >= 0 and . <= $max;
  length == 1 and (.[0]
    | .ordinal == 0 and .original_ids == "1 through 4095 in order"
      and (.carries | type == "array" and length == 3 and all(.[]; word(65535)))
      and (.packed_addresses | type == "array" and length == 3
        and all(.[]; type == "array" and length == 4095 and all(.[]; word(16777215)))))
' "$sentinel" >/dev/null

arms=(snapped transported raw)
families=(geometry midpoint-transported midpoint-raw)
magics=(UORLEX01 UORLEX04 UORLEX05)
head_shas=(
  e5516223485f2332f1e3c4df5a2887daa7d591bb61149fe677bdc6fa5c2dc809
  c8546a8b758d03b9d4db8e3a06bc6c9923c761c93e8848e3a3a4480c821508b0
  e078cc2e5466cddb6d853cd9d44c76e7df516d7392b241191a637cbd615db34a
)
cmp -s -- "$run/frozen/snapped.lex" "$previous/$old_arm.lex"
for index in "${!arms[@]}"; do
  arm=${arms[index]}
  head="$run/frozen/$arm.lex"
  head_sha=$(sha "$head")
  [[ $(wc -c < "$head") -eq 584 && $head_sha == "${head_shas[index]}" ]]
  [[ $(dd if="$head" bs=1 skip=32 count=8 status=none) == "${magics[index]}" ]]
  cmp -s <(dd if="$head" bs=1 count=32 status=none) \
    <(dd if="$run/frozen/snapped.lex" bs=1 count=32 status=none)
  cmp -s <(dd if="$head" bs=1 skip=40 count=512 status=none) \
    <(dd if="$run/frozen/snapped.lex" bs=1 skip=40 count=512 status=none)

  UOR_P1_TIMING=1 /usr/bin/time -f 'wall_seconds=%e maximum_rss_kib=%M' -o "$out/$arm-time.txt" \
    "$client" --discrete "$p4" --prompt-ids "$ids" --max-tokens 1 --p1-trace \
    --p4-lexical "$head" > "$out/$arm-client.json" 2> "$out/$arm-stderr.txt"

  jq -ces --arg arm "$arm" --arg artifact "$artifact" --arg state "$state" --arg head "$head_sha" \
    --arg family "${families[index]}" --argjson index "$index" --slurpfile gpu "$sentinel" \
    --slurpfile archived "$archived" --slurpfile old "$out/archived-device.json" '
    def slots($packed):
      [range(4) as $bank | 64 * $bank + (($packed / pow(64; $bank) | floor) % 64)];
    if length != 1 then error("unexpected ordinary output count") else .[0] end
    | .p4_steps[0] as $step | $step.lexical as $word
    | $word.token as $id
    | if .schema == "uor-r4.uor-chat-discrete-parity/1"
         and .artifact_sha256 == $artifact and .p4_sha256 == $state and .p4_state_sha256 == $state
         and .prompt_ids == $archived[0].prompt_ids and .generated_ids == [$id]
         and .p4_lexical_sha256 == $head and .p4_lexical_addressing == $family
         and (.p4_steps | length) == 1 and $step.token == $id
         and ($id | type == "number" and . == floor and . >= 1 and . <= 4095)
         and ($step | [.context_tokens,.prefix_sha256,.mass_sha256,.prior_winner])
             == ($archived[0].p4_steps[0] | [.context_tokens,.prefix_sha256,.mass_sha256,.prior_winner])
         and $word.carry_code == $gpu[0].carries[$index]
         and $word.addresses == slots($gpu[0].packed_addresses[$index][$id - 1])
         and (if $arm == "snapped" then
                ($old | length) == 1 and $old[0].addressing == "geometry"
                and $word.carry_code == $old[0].carry
                and $id == $old[0].device.winner
                and $word.prior_q48 == $old[0].device.winner_prior_q48
                and $word.score_q48 == $old[0].device.winner_score_q48
                and $word.top_ties == $old[0].device.top_ties
                and $word.addresses == $old[0].device.winner_addresses
              else true end)
      then {arm:$arm, addressing:$family, attachment_sha256:$head,
            winner_original_id:$id, carry:$word.carry_code, slots:$word.addresses,
            prior_q48:$word.prior_q48, score_q48:$word.score_q48, top_ties:$word.top_ties}
      else error("ordinary cold first-step mismatch for " + $arm) end
  ' "$out/$arm-client.json" > "$out/$arm-match.json"
done

cmp -s -- "$out/snapped-client.json" "$archived" || {
  echo 'snapped ordinary stdout differs from archived #4356 stdout' >&2
  exit 1
}
[[ $(sha "$client") == "$client_sha" && $(sha "$receipt") == "$receipt_sha" \
   && $(sha "$sentinel") == "$sentinel_sha" ]]
jq -ces --arg client "$client_sha" --arg artifact "$artifact" --arg state "$state" \
  --arg train "$train_sha" --arg old "$old_trace_sha" --arg receipt "$receipt_sha" \
  --arg sentinel "$sentinel_sha" '
  {status:"PASS", sentinel_ordinal:0, client_sha256:$client, artifact_sha256:$artifact,
   state_sha256:$state, train_inputs_sha256:$train, receipt_sha256:$receipt,
   sentinel_sha256:$sentinel, archived_snapped_stdout_sha256:$old,
   snapped_stdout_byte_exact:true, ordinary_invocations:length, cases:.}
' "$out/snapped-match.json" "$out/transported-match.json" "$out/raw-match.json" > "$out/summary.json"
cat "$out/summary.json"
