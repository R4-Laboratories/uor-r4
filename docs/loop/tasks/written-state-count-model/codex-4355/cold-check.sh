#!/usr/bin/env bash
# Reopen the native client at the frozen TRAIN lexical boundary for each fitted attachment.
set -euo pipefail
umask 077
export LC_ALL=C

if [[ $# != 5 ]]; then
  echo 'usage: cold-check.sh CLIENT P4_D11 TRAIN_INPUT_ROOT TRIAL_DIR NEW_OUTPUT_DIR' >&2
  exit 2
fi

client=$1
p4=$2
train=$3
trial=$4
out=$5
artifact=e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776
state=368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778
split=3b8d07c8112d696d08a91651c2ce0da1ef8acb81723cfcaec01bdea7798ee17c
train_sha=5dd411d7dee1335784efc5868adf58a6979da18d4b743809e36b12bd44dc6829
prefix="$trial/sentinel-prefix.json"
witnesses="$trial/witnesses.jsonl"
arms=(
  zero
  geometry-gold-only-435300 geometry-gold-only-435301
  geometry-full-435300 geometry-full-435301
  ordered-matched-gold-only-435300 ordered-matched-gold-only-435301
  ordered-matched-full-435300 ordered-matched-full-435301
)
expected_arms=$(jq -nc '$ARGS.positional | sort' --args "${arms[@]}")

sha() { sha256sum -- "$1" | cut -d' ' -f1; }

mkdir -- "$out"
if [[ $(sha "$p4") != "$artifact" || $(sha "$train/train/inputs.jsonl") != "$train_sha" ]]; then
  echo 'frozen artifact or TRAIN input checksum differs' >&2
  exit 1
fi
client_sha=$(sha "$client")
[[ -x "$client" ]]
jq -es --arg artifact "$artifact" --arg state "$state" --arg split "$split" '
  length == 1 and (.[0]
    | .artifact == $artifact and .state == $state and .split == $split and .rows == 3891
      and .controls_verified == 32)
' "$trial/run.json" >/dev/null

# Training has already consumed the optional genuine standalone space. Feed the captured
# original IDs at that boundary through the client's regular raw-ID input.
ids=$(jq -ner --slurpfile expected "$prefix" '
  input.lexical_prefix as $ids
  | def id: type == "number" and . == floor and . >= 0 and . < 4096;
  if ($expected | length) == 1 and $expected[0] == $ids
     and ($ids | type == "array" and length > 0 and length <= 2048 and .[0] == 0 and all(.[]; id))
  then $ids | join(",")
  else error("device sentinel differs from frozen TRAIN ordinal zero") end
' "$train/train/inputs.jsonl")
jq -ces --argjson expected "$expected_arms" '
  def int($min; $max):
    type == "number" and . == floor and . >= $min and . <= $max;
  def witness:
    (keys == ["addressing","carry","device","name"])
    and (if (.name | startswith("ordered-matched-"))
         then .addressing == "ordered-matched" else .addressing == "geometry" end)
    and (.carry | int(0;65535))
    and (.device
      | (.winner | int(1;4095))
        and (.winner_prior_q48 | int(0;281474976710656))
        and (.winner_score_q48 | int(-562949953421312;844424930131968))
        and (.top_ties | int(1;4095))
        and (.winner_addresses | type == "array" and length == 4))
    and (.device.winner_addresses as $a | all(range(4); . as $i | $a[$i] | int(64 * $i;64 * $i + 63)));
  if (map(.name) | sort) == $expected
     and all(.[]; witness)
     and ([.[].carry] | unique | length) == 1
  then . else error("invalid TRAIN witness roster or device trace") end
' "$witnesses" > "$out/device.json"

for arm in "${arms[@]}"; do
  head="$trial/$arm.lex"
  if [[ $(wc -c < "$head") -ne 584 ]]; then
    echo "invalid lexical attachment length for $arm" >&2
    exit 1
  fi
  head_sha=$(sha "$head")
  family=geometry
  magic=UORLEX01
  if [[ $arm == ordered-matched-* ]]; then
    family=ordered
    magic=UORLEX03
  fi
  [[ $(dd if="$head" bs=1 skip=32 count=8 status=none) == "$magic" ]]
  UOR_P1_TIMING=1 /usr/bin/time -f 'wall_seconds=%e maximum_rss_kib=%M' -o "$out/$arm-time.txt" \
    "$client" --discrete "$p4" --prompt-ids "$ids" --max-tokens 1 --p1-trace \
    --p4-lexical "$head" > "$out/$arm-client.json" 2> "$out/$arm-stderr.txt"

  jq -ces --arg arm "$arm" --arg artifact "$artifact" --arg state "$state" --arg head "$head_sha" \
    --arg family "$family" --slurpfile prefix "$prefix" --slurpfile witnesses "$out/device.json" '
    ($witnesses[0] | map(select(.name == $arm)) | first) as $gpu
    | def compare($client):
        $client.p4_steps[0] as $step | $step.lexical as $word
        | $client.schema == "uor-r4.uor-chat-discrete-parity/1"
          and $client.artifact_sha256 == $artifact
          and $client.p4_sha256 == $state and $client.p4_state_sha256 == $state
          and $client.prompt_ids == $prefix[0] and $client.generated_ids == [$gpu.device.winner]
          and $client.p4_lexical_sha256 == $head and $client.p4_lexical_addressing == $family
          and ($client.p4_steps | length) == 1
          and $step.context_tokens == ($prefix[0] | length)
          and ($step.prefix_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
          and ($step.mass_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
          and $step.token == $gpu.device.winner and $word.token == $gpu.device.winner
          and $step.mass_q48 == $gpu.device.winner_prior_q48 and $word.prior_q48 == $gpu.device.winner_prior_q48
          and $word.score_q48 == $gpu.device.winner_score_q48
          and $step.top_ties == $gpu.device.top_ties and $word.top_ties == $gpu.device.top_ties
          and $word.addresses == $gpu.device.winner_addresses
          and ($word | has("carry_code"))
          and ($word.carry_code == (if $gpu.addressing == "geometry" then $gpu.carry else null end))
          and (if $arm == "zero" then
                 $word.score_q48 == $step.mass_q48
                 and $step.prior_winner == {token:$step.token,mass_q48:$step.mass_q48,top_ties:$step.top_ties}
               else true end);
    if length == 1 and compare(.[0]) then
      {status:"PASS", arm:$arm, addressing:$gpu.addressing, steps:1, attachment_sha256:$head}
    else error("ordinary cold first-step mismatch for " + $arm) end
  ' "$out/$arm-client.json" > "$out/$arm-match.json"
done

# The fitted readout changes only the election. The prior vector and causal prefix are common.
jq -es --argjson expected "$expected_arms" '
  length == ($expected | length)
  and ([.[].p4_steps[0] | [.mass_sha256,.prefix_sha256,.prior_winner]] | unique | length) == 1
' "$out"/*-client.json >/dev/null

[[ $(sha "$client") == "$client_sha" ]]
jq -ces --arg client "$client_sha" --arg artifact "$artifact" --arg state "$state" \
  --arg split "$split" --arg train "$train_sha" --arg witness "$(sha "$witnesses")" \
  --argjson expected "$expected_arms" '
  if all(.[]; .status == "PASS" and .steps == 1)
     and (map(.arm) | sort) == $expected then
    {status:"PASS", client_sha256:$client, artifact_sha256:$artifact, state_sha256:$state,
     split_sha256:$split, train_inputs_sha256:$train, witnesses_sha256:$witness,
     sentinel_ordinal:0, entry:"ordinary raw original IDs",
     ordinary_invocations:length, total_compared_steps:(map(.steps)|add), cases:.}
  else error("incomplete ordinary cold check") end
' "$out"/*-match.json > "$out/summary.json"
cat "$out/summary.json"
