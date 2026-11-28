#!/usr/bin/env bash
# Replay the eight authenticated #4355 ordinary cold witnesses on centered attachments.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_VISIBLE_DEVICES=''

if [[ $# != 4 ]]; then
  echo 'usage: cold-preflight.sh ORIGINAL_PACKET P4_D11 CENTERED_DIR NEW_OUTPUT_DIR' >&2
  exit 2
fi

packet=$(realpath -- "$1")
p4=$(realpath -- "$2")
centered=$(realpath -- "$3")
out=$4
packet_sha=b6603d6dba71299cc3c6351a3404e50bb1a88ee16e6922bb97427d1bea2afe09
artifact_sha=e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776
client_sha=a90aaac01600f7c8c05c73ba8673e53016822e70492bc2d9d2ede1f7d4f6dca4
arms=(
  geometry-full-435300 geometry-full-435301
  geometry-gold-only-435300 geometry-gold-only-435301
  ordered-matched-full-435300 ordered-matched-full-435301
  ordered-matched-gold-only-435300 ordered-matched-gold-only-435301
)
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
require_sha() {
  if [[ $(sha "$1") != "$2" ]]; then
    echo "checksum mismatch for $1" >&2
    exit 1
  fi
}

mkdir -- "$out"
out=$(realpath -- "$out")
require_sha "$packet" "$packet_sha"
require_sha "$p4" "$artifact_sha"
mkdir -- "$out/archive"
prefix=attempt01/trial/sentinel-prefix.json
members=(bin/uor-chat cold01/summary.json "$prefix")
for arm in "${arms[@]}"; do
  members+=("attempt01/trial/$arm.lex" "cold01/$arm-client.json")
done
tar --zstd -xf "$packet" -C "$out/archive" PACKET.sha256 "${members[@]}"
(
  cd "$out/archive"
  for member in "${members[@]}"; do
    awk -v member="$member" '$2 == member {print; count++} END {if (count != 1) exit 1}' PACKET.sha256
  done > "$out/archived-members.sha256"
  sha256sum --strict -c "$out/archived-members.sha256"
) > "$out/archived-verification.txt"

archive=$out/archive
client=$archive/bin/uor-chat
require_sha "$client" "$client_sha"
[[ -x "$client" ]]
archived=$archive/cold01/summary.json
jq -e --arg client "$client_sha" --arg p4 "$artifact_sha" '
  .status == "PASS" and .client_sha256 == $client and .artifact_sha256 == $p4
  and .sentinel_ordinal == 0 and .ordinary_invocations == 9
  and (.cases | length) == 9
' "$archived" >/dev/null
ids=$(jq -er '
  def id: type == "number" and . == floor and . >= 0 and . < 4096;
  if type == "array" and length > 0 and length <= 2048 and .[0] == 0 and all(.[]; id)
  then join(",") else error("invalid archived sentinel prefix") end
' "$archive/$prefix")
expected_arms=$(jq -nc '$ARGS.positional | sort' --args "${arms[@]}")

# Keep the entire trace, including prior mass digest, winner/ties, the selected
# addresses and carry, S2-S4 routing, and the emitted ID. The attachment hash is
# authenticated separately and a P4 with the same bytes may use a different path.
compare() {
  jq -en --slurpfile old "$1" --slurpfile new "$2" --argjson shift "$3" '
    def normalize: del(.artifact,.p4_lexical_sha256,.p4_steps[].lexical.score_q48);
    ($old | length) == 1 and ($new | length) == 1
    and ($old[0] | normalize) == ($new[0] | normalize)
    and all(range($old[0].p4_steps | length); . as $i
      | $new[0].p4_steps[$i].lexical.score_q48 - $old[0].p4_steps[$i].lexical.score_q48 == $shift)
  '
}

check_trace() {
  jq -es --arg digest "$2" --arg family "$3" --arg p4 "$artifact_sha" --argjson cap "$4" \
    --slurpfile prefix "$archive/$prefix" --slurpfile receipt "$archived" '
    def int($min; $max):
      type == "number" and . == floor and . >= $min and . <= $max;
    length == 1 and (.[0] | . as $trace
      | .schema == "uor-r4.uor-chat-discrete-parity/1"
      and .artifact_sha256 == $p4
      and .p4_state_sha256 == $receipt[0].state_sha256
      and .p4_sha256 == $receipt[0].state_sha256
      and .prompt_ids == $prefix[0]
      and .p4_lexical_sha256 == $digest and .p4_lexical_addressing == $family
      and (.p4_steps | length) == (.generated_ids | length)
      and (.generated_ids | length == $cap or (length > 0 and length < $cap and .[-1] == 1))
      and all(range(.p4_steps | length); . as $i
      | $trace.p4_steps[$i] as $step | $step.lexical as $lex
      | $step.context_tokens == ([($prefix[0] | length) + $i, $trace.context] | min)
      and $trace.generated_ids[$i] == $step.token and $step.token == $lex.token
      and ($lex.token | int(1;4095))
      and $step.top_ties == $lex.top_ties and ($lex.top_ties | int(1;4095))
      and $step.mass_q48 == $lex.prior_q48 and ($lex.prior_q48 | int(0;281474976710656))
      and ($step.mass_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      and ($step.prefix_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      and ($lex.score_q48 | int(-562949953421312;844424930131968))
      and ($lex.addresses | length) == 4
      and (all(range(4); . as $i | $lex.addresses[$i] | int(64 * $i; 64 * $i + 63)))
      and ($lex | has("carry_code"))
      and (if $family == "geometry" then ($lex.carry_code | int(0;65535))
           else $lex.carry_code == null end)
      )
    )
  ' "$1" >/dev/null
}

# The controls change a saved original trace, leaving the replay and expected
# shift fixed. Require an ordinary Boolean comparison failure, not a jq error.
first="$archive/cold01/${arms[0]}-client.json"
jq '.p4_steps[0].token as $id
    | (if $id == 4095 then 4094 else $id + 1 end) as $altered
    | .generated_ids = [$altered]
    | .p4_steps[0].token = $altered | .p4_steps[0].lexical.token = $altered' \
  "$first" > "$out/negative-original-id.json"
jq '.p4_steps[0].lexical.score_q48 += 1' "$first" > "$out/negative-original-score.json"
compare "$first" "$first" 0 > "$out/comparator-positive.txt"
for negative in original-id original-score; do
  if compare "$out/negative-$negative.json" "$first" 0 > "$out/comparator-$negative.txt"; then
    echo "comparator accepted changed $negative" >&2
    exit 1
  else
    [[ $? == 1 && $(cat "$out/comparator-$negative.txt") == false ]]
  fi
done

run_client() {
  UOR_P1_TIMING=1 /usr/bin/time -f 'wall_seconds=%e maximum_rss_kib=%M' -o "$out/$3-time.txt" \
    "$client" --discrete "$p4" --prompt-ids "$ids" --max-tokens "$2" --p1-trace \
    --p4-lexical "$1" > "$out/$3-client.json" 2> "$out/$3-stderr.txt"
}

for arm in "${arms[@]}"; do
  old_head="$archive/attempt01/trial/$arm.lex"
  old_trace="$archive/cold01/$arm-client.json"
  new_head="$centered/$arm.lex"
  new_trace="$out/$arm-client.json"
  card="$centered/$arm.json"
  family=geometry
  if [[ $arm == ordered-matched-* ]]; then
    family=ordered
  fi
  old_sha=$(jq -er --arg arm "$arm" '
    [.cases[] | select(.arm == $arm and .steps == 1 and .status == "PASS")]
    | if length == 1 then .[0].attachment_sha256 else error("missing archived case") end
  ' "$archived")
  require_sha "$old_head" "$old_sha"
  check_trace "$old_trace" "$old_sha" "$family" 1

  # Sum the four signed offsets independently. These bounds keep every value
  # and operation in jq's exact integer range (magnitudes below 2^53).
  read -r new_sha shift < <(jq -er --arg old "$old_sha" --arg family "$family" '
    def int($min; $max):
      type == "number" and . == floor and . >= $min and . <= $max;
    (.banks | map(.delta_i16_words)) as $offsets
    | if .source_sha256 == $old
      and .addressing == (if $family == "ordered" then "ordered-matched" else $family end)
      and ([.banks[].bank] == [0,1,2,3])
      and (.centered_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      and ($offsets | length == 4 and all(.[]; int(-65535;65535)))
      and .score_shift_q48 == (($offsets | add) * 4294967296)
      then "\(.centered_sha256) \(.score_shift_q48)"
      else error("invalid centered receipt") end
  ' "$card")
  [[ $(wc -c < "$old_head") == 584 && $(wc -c < "$new_head") == 584 ]]
  require_sha "$new_head" "$new_sha"

  if [[ $arm == geometry-full-435300 ]]; then
    (
      cap=6
      run_client "$old_head" "$cap" continuation-original
      run_client "$new_head" "$cap" continuation-centered
      original="$out/continuation-original-client.json"
      recentered="$out/continuation-centered-client.json"
      check_trace "$original" "$old_sha" "$family" "$cap"
      check_trace "$recentered" "$new_sha" "$family" "$cap"
      compare "$original" "$recentered" "$shift" > "$out/continuation-compare.txt"
      jq -c --arg arm "$arm" --argjson cap "$cap" --argjson shift "$shift" \
        --arg old_trace_sha "$(sha "$original")" --arg new_trace_sha "$(sha "$recentered")" '
        .generated_ids as $ids
        | {arm:$arm, score_shift_q48:$shift, cap:$cap, emitted_count:($ids | length),
           stop:(if ($ids | length) == $cap then "max_tokens" else "eos" end),
           cap_censored:(($ids | length) == $cap),
           eos_id_positions:[$ids | to_entries[] | select(.value == 1) | .key],
           original_trace_sha256:$old_trace_sha, centered_trace_sha256:$new_trace_sha}
      ' "$recentered" > "$out/continuation.json"
    ) &
    continuation_pid=$!
    trap 'wait "$continuation_pid" || true' EXIT
  fi
  run_client "$new_head" 1 "$arm"
  check_trace "$new_trace" "$new_sha" "$family" 1
  if ! compare "$old_trace" "$new_trace" "$shift" > "$out/$arm-compare.txt"; then
    echo "ordinary cold first-step mismatch for $arm" >&2
    exit 1
  fi
  jq -c --arg arm "$arm" --arg old_trace_sha "$(sha "$old_trace")" --arg new_trace_sha "$(sha "$new_trace")" \
    --arg old_sha "$old_sha" --arg new_sha "$new_sha" --argjson shift "$shift" '
    .p4_steps[0].lexical
    | {arm:$arm, winner_original_id:.token, top_ties, score_shift_q48:$shift,
       original_attachment_sha256:$old_sha, centered_attachment_sha256:$new_sha,
       original_trace_sha256:$old_trace_sha, centered_trace_sha256:$new_trace_sha}
  ' "$new_trace" > "$out/$arm-match.json"
done

wait "$continuation_pid"
trap - EXIT
require_sha "$client" "$client_sha"
jq -ces --arg client "$client_sha" --arg artifact "$artifact_sha" --arg packet "$packet_sha" \
  --argjson expected "$expected_arms" --slurpfile continuation "$out/continuation.json" '
  if (map(.arm) | sort) == $expected then
    {status:"PASS", original_packet_sha256:$packet, client_sha256:$client,
     artifact_sha256:$artifact, sentinel_ordinal:0, ordinary_invocations:length,
     changed_original_id_rejected:true, changed_original_score_rejected:true, cases:.,
     self_fed_continuation:$continuation[0]}
  else error("incomplete ordinary cold comparison") end
' "$out"/*-match.json > "$out/summary.json"
cat "$out/summary.json"
