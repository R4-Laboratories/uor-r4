#!/usr/bin/env bash
# Reopen the ordinary client for the lowest opaque ID in each held pool.
set -euo pipefail
umask 077
export LC_ALL=C

if [[ $# != 6 ]]; then
  echo 'usage: cold-check.sh CLIENT P4_D11 TOKENIZER BLIND_PACKET TRIAL_DIR NEW_OUTPUT_DIR' >&2
  exit 2
fi

client=$1
p4=$2
tokenizer=$3
packet=$4
trial=$5
out=$6
mkdir "$out"

sha() { sha256sum "$1" | cut -d' ' -f1; }
artifact=$(sha "$p4")
tokenizer_sha=$(sha "$tokenizer")
state=$(jq -er '.state | select(type == "string" and test("^[0-9a-f]{64}$"))' "$trial/fit.json")
test "$artifact" = "$(jq -er '.artifact' "$trial/fit.json")"
client_sha=$(sha "$client")
arms=(zero geometry-435300 ordered-435300 geometry-435301 ordered-435301)

for pool in dev supplement; do
  pool_out="$out/$pool"
  mkdir "$pool_out"
  jq -ces '
    if length > 0 and (map(.id) | unique | length) == length
       and all(.[]; (keys == ["id","messages","prefix"])
           and (.id | type == "string" and test("^[0-9a-f]{24}$"))
           and (.messages | type == "array" and length > 0)
           and (.prefix | type == "array" and length > 0))
    then min_by(.id) else error("invalid blind prompt tape") end
  ' "$packet/$pool/blind.jsonl" > "$pool_out/anchor.json"
  anchor_id=$(jq -er '.id' "$pool_out/anchor.json")
  jq -c '.messages' "$pool_out/anchor.json" > "$pool_out/messages.json"
  jq -ces --arg id "$anchor_id" '
    [.[] | select(.id == $id)]
    | if (map(.arm) | sort) == ["geometry-435300","geometry-435301","ordered-435300","ordered-435301","zero"]
      then . else error("wrong GPU arm roster at cold anchor") end
  ' "$trial/$pool/blind-generation.jsonl" > "$pool_out/gpu.json"

  for arm in "${arms[@]}" frozen; do
    mode=geometry
    [[ $arm != ordered-* ]] || mode=ordered
    sidecar_sha=''
    args=()
    if [[ $arm != frozen ]]; then
      sidecar="$trial/$arm.lex"
      test "$(wc -c < "$sidecar")" -eq 584
      sidecar_sha=$(sha "$sidecar")
      args=(--p4-lexical "$sidecar")
    fi
    UOR_P1_TIMING=1 /usr/bin/time -f 'wall_seconds=%e maximum_rss_kib=%M' -o "$pool_out/$arm-time.txt" \
      "$client" --discrete "$p4" --tokenizer "$tokenizer" --messages "$pool_out/messages.json" \
      --max-tokens 32 --p1-trace "${args[@]}" > "$pool_out/$arm-client.json" 2> "$pool_out/$arm-stderr.txt"

    gpu_arm=$arm
    [[ $arm != frozen ]] || gpu_arm=zero
    jq -ce --arg arm "$arm" --arg gpu_arm "$gpu_arm" --arg mode "$mode" --arg weights "$sidecar_sha" \
      --arg artifact "$artifact" --arg state "$state" --arg tokenizer "$tokenizer_sha" \
      --arg pool "$pool" --slurpfile anchor "$pool_out/anchor.json" --slurpfile gold_free "$pool_out/gpu.json" '
      . as $client | ($gold_free[0] | map(select(.arm == $gpu_arm)) | first) as $gpu
      | def same_commit($i):
          $client.p4_steps[$i] as $a | $gpu.steps[$i] as $b
          | ($a.token == $b.id and $b.id == $gpu.emitted[$i]
             and $a.mass_q48 == $b.prior_q48 and $a.top_ties == $b.top_ties)
          and (if $arm == "frozen" then
                 ($a.lexical == null and $a.prior_winner == null and $a.token == $b.prior_winner)
               else
                 ($a.prior_winner.token == $b.prior_winner and $a.lexical.token == $b.id
                  and $a.lexical.prior_q48 == $b.prior_q48 and $a.lexical.score_q48 == $b.score_q48
                  and $a.lexical.top_ties == $b.top_ties and $a.lexical.addresses == $b.addresses
                  and (if $mode == "ordered" then $a.lexical.carry_code == null
                       else $a.lexical.carry_code == $b.carry_code end))
               end);
      if $client.artifact_sha256 == $artifact and $client.p4_state_sha256 == $state
         and $client.tokenizer_sha256 == $tokenizer
         and $client.protocol == "uor-r4.literal-role-dialogue/2"
         and $client.prompt_ids == $anchor[0].prefix and $client.generated_ids == $gpu.emitted
         and (if $gpu.stop == "eos" then $client.stop == "eos" and $gpu.emitted[-1] == 1
              else $gpu.stop == "cap" and $client.stop == "max_tokens" and ($gpu.emitted | length) == 32 end)
         and ($gpu.emitted | length) > 0 and ($gpu.emitted | length) <= 32
         and ($gpu.steps | length) == ($gpu.emitted | length)
         and ($client.p4_steps | length) == ($gpu.steps | length)
         and (if $arm == "frozen" then $client.p4_lexical_sha256 == null and $client.p4_lexical_addressing == null
              else $client.p4_lexical_sha256 == $weights and $client.p4_lexical_addressing == $mode end)
         and all(range($gpu.steps | length); same_commit(.))
      then {status:"PASS",pool:$pool,arm:$arm,steps:($gpu.steps | length),stop:$gpu.stop}
      else error("cold client mismatch in " + $pool + "/" + $arm) end
    ' "$pool_out/$arm-client.json" > "$pool_out/$arm-match.json"
  done

  jq -e --slurpfile zero "$pool_out/zero-client.json" '
    .generated_ids == $zero[0].generated_ids and .stop == $zero[0].stop
    and .p4_steps == ($zero[0].p4_steps | map(del(.prior_winner,.lexical)))
  ' "$pool_out/frozen-client.json" > "$pool_out/frozen-zero-identical.json"
done

jq -ces --arg client "$client_sha" --arg artifact "$artifact" --arg state "$state" --arg tokenizer "$tokenizer_sha" '
  if length == 12 and all(.[]; .status == "PASS") then
    {status:"PASS", client_sha256:$client, artifact_sha256:$artifact, state_sha256:$state,
     tokenizer_sha256:$tokenizer, ordinary_invocations:length, total_compared_steps:(map(.steps)|add), cases:.}
  else error("incomplete cold check") end
' "$out"/{dev,supplement}/*-match.json > "$out/summary.json"
cat "$out/summary.json"
