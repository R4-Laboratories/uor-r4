#!/usr/bin/env bash
# Reopen the ordinary integer client at the frozen TRAIN boundary for all 19 heads.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_VISIBLE_DEVICES=''

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
client_expected=a90aaac01600f7c8c05c73ba8673e53016822e70492bc2d9d2ede1f7d4f6dca4
train_sha=5dd411d7dee1335784efc5868adf58a6979da18d4b743809e36b12bd44dc6829
prefix="$trial/sentinel-prefix.json"
witnesses="$trial/witnesses.jsonl"
arms=(zero historical-geometry-435300 historical-geometry-435301)
for seed in 435300 435301; do
  for addressing in geometry ordered-matched; do
    for policy in gold-only full; do
      for centering in uncentered centered; do
        arms+=("$addressing-$policy-$seed-$centering")
      done
    done
  done
done
expected_arms=$(jq -nc '$ARGS.positional | sort' --args "${arms[@]}")

sha() { sha256sum -- "$1" | cut -d' ' -f1; }

mkdir -- "$out"
if [[ $(sha "$p4") != "$artifact" || $(sha "$train/train/inputs.jsonl") != "$train_sha" ]]; then
  echo 'frozen artifact or TRAIN input checksum differs' >&2
  exit 1
fi
client_sha=$(sha "$client")
if [[ ! -x "$client" || $client_sha != "$client_expected" ]]; then
  echo 'ordinary client differs from the authenticated frozen binary' >&2
  exit 1
fi
jq -es --arg artifact "$artifact" --arg state "$state" --arg split "$split" '
  length == 1 and (.[0]
    | .artifact == $artifact and .state == $state and .split == $split and .rows == 3891
      and .common_rows == 3886 and .controls_verified == 64 and .historical_visits_verified == 249024
      and (.fits | length) == 16 and (.historical_reference_fits | length) == 2)
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
  # Independently authenticate the archived replay heads in the cold roster.
  case $arm in
    zero) expected=5c883501f58310706ac51165df7519e7b9e2435e863f7d71e7bc80ad22bad824 ;;
    historical-geometry-435300) expected=02234649fda6e65e715a7a82faf393922af5b46a41b58cfa0c32ebe1fcbe57d4 ;;
    historical-geometry-435301) expected=dd7983ff762cacc93ce93113fcaa442160d0138023de894a6914643e44fd344d ;;
    geometry-full-435300-uncentered) expected=e5516223485f2332f1e3c4df5a2887daa7d591bb61149fe677bdc6fa5c2dc809 ;;
    geometry-full-435301-uncentered) expected=9e24e72ca30ce4d16060da28182349cfbbac0d395044d166738e441113aa3194 ;;
    geometry-gold-only-435300-uncentered) expected=7df18e1a7abcca3f1f35a3d68f290b3e4faae080263b687cc48c5229b4968896 ;;
    geometry-gold-only-435301-uncentered) expected=74f81ba0125a8b3f48f363eb1eb82647801a6d868a64e88298eb032fd941d876 ;;
    ordered-matched-full-435300-uncentered) expected=1ff45f2a5b3e2d5dd5fbe726d83cc65e03722ea398bebea34a0ffc1c060a1c85 ;;
    ordered-matched-full-435301-uncentered) expected=402a58674728b106db085b1ad8a38b563e38f0e2a6c28b296187a00b65fb093a ;;
    ordered-matched-gold-only-435300-uncentered) expected=7d9e2bfb1d4ad60bc02e30a51a5c292a7474bc66d18d5dc15136d77c64f5d74b ;;
    ordered-matched-gold-only-435301-uncentered) expected=6765c184ee329982a4dc24eca29bd85a3f622e8d71efeb9c8b5c51e3d252685e ;;
    *-centered) expected= ;;
    *) exit 1 ;;
  esac
  if [[ -n $expected && $head_sha != "$expected" ]]; then
    echo "historical head checksum differs for $arm" >&2
    exit 1
  fi
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
