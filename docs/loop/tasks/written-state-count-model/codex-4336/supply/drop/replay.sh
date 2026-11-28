#!/usr/bin/env bash
set -euo pipefail
umask 077
inputs=$(realpath "${1:?eligible study rows}")
chat=$(realpath "${2:?pinned native client}")
tokenizer=$(realpath "${3:?canonical tokenizer}")
balanced=$(realpath "${4:?balanced D11}")
out=$(realpath -m "${5:?fresh result directory}")
sha256sum --strict --quiet -c <<SUMS
f9211c1366636c9bf424338d8ca907dd1053dc5be0fdae8eefa1160052fd8ed3  $inputs
5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa  $balanced
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
SUMS
mkdir "$out"
for mode in baseline native continue stop; do
  flags=()
  if [[ $mode != baseline ]]; then
    request="$out/$mode-request.json"
    jq -ce --arg mode "$mode" '{authority:{prompt_ids,first,second},claim:{first,second,first_action:$mode,second_action:"stop"}}' \
       "$inputs" > "$request"
    flags=(--verified-span-release "$request")
  fi
  jq -j '.text,"\u0000"' "$inputs" | xargs -0 -n1 "$chat" \
     --discrete "$balanced" --tokenizer "$tokenizer" --protocol 2 --seed 0 --temperature 0 \
     --max-tokens 16 --authenticated-copy --native-period-release --native-occurrence-trace "${flags[@]}" --say |
     jq -ceS --arg mode "$mode" '{arm:$mode,client:del(.artifact)}' > "$out/$mode.jsonl"
done
cat "$out"/{baseline,native,continue,stop}.jsonl > "$out/records.jsonl"
jq -ces --slurpfile input "$inputs" '
  $input[0] as $truth | $truth.expected_ids_through_eos as $gold |
  map(
    if .client.prompt_ids != $truth.prompt_ids then error("prompt mismatch") else . end |
    (.client.verified_span_release.steps // []) as $steps |
    ($steps|map(select(.commit=="copy")|.source_pos)) as $actual |
    ([range($truth.first.start;$truth.first.end)] + [range($truth.second.start;$truth.second.end)]) as $addresses |
    .client.generated_ids as $emitted |
    {arm, original_copy_positions:$actual, exact_addresses:($actual==$addresses), eos:(.client.stop=="eos"),
     first_wrong:([range(0;([($emitted|length),($gold|length)]|max)) as $i | select($emitted[$i]!=$gold[$i]) |
        {offset:$i,emitted:$emitted[$i],expected:$gold[$i]}] | first),
     exact_through_eos:($emitted==$gold and .client.stop=="eos")})' "$out/records.jsonl" > "$out/summary.json"
(cd "$out" && sha256sum records.jsonl summary.json > SHA256SUMS)
