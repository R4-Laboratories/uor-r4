#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 ]]; then echo 'usage: run.sh WICK_RUNS CENSUS_ROWS NEW_OUTPUT' >&2; exit 2; fi
runs=$(realpath "$1")
rows=$(realpath "$2")
out=$(realpath -m "$3")
mkdir "$out"
model="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
tok="$runs/20261127-codex-ordinary-source/readback-v1/payload/tokenizer.json"
chat="$runs/20261127-codex-ordinary-source/readback-v1/payload/tools/trace-chat"
tape="$runs/20261127-codex-marker-question/donor-v2/answer-first.tokens.u16"
{
  printf '%s  %s\n' \
  5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$model" \
  a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tok" \
  7686a7e51fb34aa091111ee7626c2c43daed47b839d7514ddefd8219d7715e1b "$chat"
  sha256sum "$rows" "$tape" "$(realpath "$0")"
} > "$out/INPUTS.sha256"
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
jq -c 'select(.turn==0 and (.group as $g|[2497,2990,2230,868,1841]|index($g)!=null))' "$rows" > "$out/selected.jsonl"
[[ $(wc -l < "$out/selected.jsonl") == 5 ]]
get_ids() {
  # UORT v1 payload starts at byte 64, all IDs are little-endian u16.
  od --endian=little -An -v -tu2 -j "$((64 + 2*$1))" -N "$((2*($2-$1)))" "$tape" | jq -s -c .
}
while IFS= read -r row; do
  read -r id start stop answer_start answer_end < <(jq -r '[.group,.prompt_offsets[],.answer_offsets[]]|@tsv' <<< "$row")
  [[ $stop == "$answer_start" && $((stop-start)) -le 192 ]]
  prefix=$(get_ids "$start" "$stop")
  answer=$(get_ids "$answer_start" "$answer_end")
  jq -cn --argjson g "$id" --argjson p "$prefix" --argjson a "$answer" '{group:$g,prefix:$p,reference:($a+[1])}' >> "$out/truth.jsonl"
done < "$out/selected.jsonl"
worker() {
  local arm=$1 row id prompt
  local -a flags=()
  case "$arm" in auth) flags=(--authenticated-copy);; disabled) flags=(--no-native-occurrence);; esac
  while IFS= read -r row; do
    id=$(jq -r .group <<< "$row")
    prompt=$(jq -j .prompt <<< "$row"; printf '\001')
    "$chat" --discrete "$model" --tokenizer "$tok" --protocol 2 --temperature 0 --max-tokens 12 \
      --say "${prompt%$'\001'}" --native-occurrence-trace "${flags[@]}" \
      | jq -c --argjson g "$id" --arg arm "$arm" '{group:$g,arm:$arm,client:.}' >> "$out/$arm.jsonl"
  done < "$out/selected.jsonl"
}
pids=()
for arm in native auth disabled; do worker "$arm" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]]
jq -sS --slurpfile truth "$out/truth.jsonl" -f "$(dirname "$0")/grade.jq" "$out"/{native,auth,disabled}.jsonl > "$out/summary.json"
(cd "$out" && sha256sum INPUTS.sha256 *.jsonl summary.json > SHA256SUMS && sha256sum --strict --quiet -c SHA256SUMS)
