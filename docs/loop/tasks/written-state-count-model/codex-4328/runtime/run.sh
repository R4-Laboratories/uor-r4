#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 6 ]]; then
  echo 'usage: run.sh WICK_RUNS OUTPUT_DIR TRACE_CHAT ORIGINAL_CHAT PROMPTS.jsonl PROMPTS_SHA256' >&2
  exit 2
fi
runs=$(realpath "$1")
out=$(realpath -m "$2")
chat=$(realpath "$3")
original=$(realpath "$4")
panel=$(realpath "$5")
panel_sha=$6
[[ $out == "$runs/20261127-codex-ordinary-source/"* ]]
mkdir "$out"

base="$runs/20261127-codex-word-ending/product-qualified-v1/artifact.d11"
balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
paired="$runs/20261127-codex-paired-release/product-v1/paired/artifact.d11"
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
script=$(realpath "$0")
{
  printf '%s  %s\n' \
    93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a "$base" \
    5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$balanced" \
    4a13e77f0320aca1fab33bf92a668e28cef14189afea6ae1f2bc387b968a13e6 "$paired" \
    a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" \
    "$panel_sha" "$panel"
  sha256sum "$chat" "$original" "$script"
} > "$out/INPUTS.sha256"
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
jq -es '
  length == 48 and (map(.case_id) | unique | length == 48) and all(.[];
    keys == ["case_id","prompt"] and (.case_id | test("^c[0-9]{3}$"))
    and (.prompt | type == "string" and length > 0 and (contains("\u0000") | not)))
' "$panel" >/dev/null

record() {
  local executable=$1 artifact=$2 prompt=$3 label=$4 err=$5 timing=$6
  shift 6
  /usr/bin/time -a -o "$timing" -f "$label %e %U %S %M" \
    "$executable" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
    --say "$prompt" --temperature 0 --max-tokens 12 "$@" 2>> "$err"
}

wrap() {
  local row=$1 case_id=$2 artifact_arm=$3 decode_arm=$4 execution=$5
  jq -c --arg case_id "$case_id" --arg artifact_arm "$artifact_arm" \
    --arg decode_arm "$decode_arm" --arg execution "$execution" \
    '{case_id:$case_id,artifact_arm:$artifact_arm,decode_arm:$decode_arm,execution:$execution,client:.}' \
    <<<"$row"
}

check_trace() {
  local mode=$1
  jq -e --arg mode "$mode" '
    . as $row | .prompt_ids as $prompt | .generated_ids as $ids |
    ($prompt | length) as $n | ($prompt + $ids) as $timeline |
    .schema == "uor-r4.uor-chat-discrete-text/1"
      and (.native_occurrence_steps | length) == ($ids | length)
      and (if .stop == "eos" then $ids[-1] == 1 and ($ids[:-1] | index(1) == null)
           else .stop == "max_tokens" and ($ids | length) == 12 and ($ids | index(1) == null) end)
      and all(range(0; $ids|length);
        . as $i | $row.native_occurrence_steps[$i] as $s |
        (($n + $i - $row.context) | [0, .] | max) as $start |
        $s.prefix_start == $start and $s.prefix_len == ($n + $i - $start)
        and $s.previous_id == $timeline[$n + $i - 1]
        and $s.committed_id == $ids[$i]
        and (($s.decision == "selected") == ($s.source_pos != null))
        and (if $s.source_pos == null then $s.source_id == null
             else ($s.source_pos + $start) < $n
               and $s.source_id == $timeline[$s.source_pos + $start]
               and $s.source_id == $s.gate_id end)
        and (if $s.copied then $i == 0 and $mode == "authenticated_copy"
               and $s.source_pos == $s.claim_pos and $s.source_id == $s.claimed_id
               and $s.source_id == $s.committed_id else true end)
        and (if $mode == "no_read" then $s.decision == "disabled" and $s.claim_pos == null
             else $s.decision != "disabled" end))
  ' >/dev/null
}

worker() {
  local artifact_arm=$1 artifact=$2 mode=$3
  local -a options=()
  local clean_bin=$original
  case "$mode" in
    authenticated_copy) options=(--authenticated-copy) ;;
    no_read) options=(--no-native-occurrence); clean_bin=$chat ;;
  esac
  local stem="$out/$artifact_arm.$mode"
  local line id prompt traced clean
  while IFS= read -r line || [[ -n $line ]]; do
    id=$(jq -er .case_id <<<"$line")
    # The sentinel retains any literal trailing newlines through command substitution.
    prompt=$(jq -je .prompt <<<"$line"; printf '\001')
    prompt=${prompt%$'\001'}
    traced=$(record "$chat" "$artifact" "$prompt" "$id traced" "$stem.stderr" "$stem.time" \
      "${options[@]}" --native-occurrence-trace)
    check_trace "$mode" <<<"$traced"
    clean=$(record "$clean_bin" "$artifact" "$prompt" "$id uninstrumented" "$stem.stderr" "$stem.time" \
      "${options[@]}")
    jq -es 'length == 2 and (.[0] | del(.native_occurrence_steps)) == .[1]' \
      <<<"$traced"$'\n'"$clean" >/dev/null
    wrap "$traced" "$id" "$artifact_arm" "$mode" traced >> "$stem.traced.jsonl"
    wrap "$clean" "$id" "$artifact_arm" "$mode" uninstrumented >> "$stem.plain.jsonl"
  done < "$panel"
}

stock() {
  local artifact_arm=$1 artifact=$2
  local stem="$out/$artifact_arm.stock"
  local line id prompt observed
  while IFS= read -r line; do
    id=$(jq -er .case_id <<<"$line")
    case "$id" in c001|c002|c003|c004) ;; *) continue ;; esac
    prompt=$(jq -je .prompt <<<"$line"; printf '\001')
    prompt=${prompt%$'\001'}
    observed=$(record "$chat" "$artifact" "$prompt" "$id traced" "$stem.stderr" "$stem.time" \
      --read-mode no_read --native-occurrence-trace)
    check_trace default <<<"$observed"
    wrap "$observed" "$id" "$artifact_arm" stock_read_mode traced >> "$stem.jsonl"
  done < "$panel"
}

pids=()
for artifact_arm in base balanced paired; do
  artifact=${!artifact_arm}
  for mode in authenticated_copy default no_read; do
    (set -e; worker "$artifact_arm" "$artifact" "$mode") &
    pids+=("$!")
  done
  (set -e; stock "$artifact_arm" "$artifact") &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ $failed == 0 ]]

for artifact_arm in base balanced paired; do
  for mode in default authenticated_copy no_read; do
    cat "$out/$artifact_arm.$mode.traced.jsonl"
  done
done > "$out/traces.jsonl"
for artifact_arm in base balanced paired; do
  for mode in default authenticated_copy no_read; do
    cat "$out/$artifact_arm.$mode.plain.jsonl"
  done
done > "$out/uninstrumented.jsonl"
cat "$out/"*.stock.jsonl > "$out/stock-read-mode.jsonl"
jq -en --slurpfile stock "$out/stock-read-mode.jsonl" --slurpfile traced "$out/traces.jsonl" '
  ($traced | length) == 432 and ($stock | length) == 12 and
  all($stock[]; . as $s |
    any($traced[]; .artifact_arm == $s.artifact_arm and .case_id == $s.case_id
      and .decode_arm == "default" and .client == $s.client))
' >/dev/null
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
(
  cd "$out"
  sha256sum INPUTS.sha256 *.jsonl *.time *.stderr > SHA256SUMS
)
