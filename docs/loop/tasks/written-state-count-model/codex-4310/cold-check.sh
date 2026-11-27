#!/usr/bin/env bash
# Replay free generations and fixed one-step CUDA witnesses through the ordinary client.
set -euo pipefail
if [ "$#" -ne 6 ]; then
  echo 'usage: cold-check.sh CLIENT P4_D11 HELD_JSONL TRIAL_DIR NEW_COLD_DIR COMPARATOR_JQ' >&2
  exit 2
fi
client=$1
p4=$2
held=$3
trial=$4
out=$5
comparator=$6
mkdir "$out"
gate="$trial/learned.uors4g01"
gate_sha=$(sha256sum "$gate" | cut -d' ' -f1)
test "$gate_sha" = "$(jq -re '.gate_sha256' "$trial/measurement.json")"
test "$(sha256sum "$p4" | cut -d' ' -f1)" = "$(jq -re '.artifact_sha256' "$trial/measurement.json")"
# Both independent domains use the bytewise first preselected identifier.
jq -cs '[.[] | select(.free_run)] | group_by(.domain) | map(min_by(.id))' "$held" > "$out/anchors.json"
jq -cs --slurpfile anchors "$out/anchors.json" '
  [.[] as $g | $anchors[0][] | select(.id==$g.id) |
  {id,domain,prefix,arm:$g.arm,cap:$g.cap,kind:"free",tokens:$g.tokens,steps:$g.steps}]
  | if length==12 then .[] else error("six arms per fixed conversation and article required") end' \
  "$trial/generation.jsonl" > "$out/plan.jsonl"
# The native ordinary operator verifies all 16 fixed DEV prefixes in all six modes.
# Reopen the public executable for one anchor from that set and the actual earlier-EOS case.
jq -cs '[.[] | select(.arm=="learned")] | group_by(.kind) | map(first) |
  if length!=2 then error("two anchor types required") else .[] end |
  {id,domain,prefix,arm,cap:1,kind,tokens:[.step.token_id],steps:[.step]}' \
  "$trial/parity-fixtures.jsonl" >> "$out/plan.jsonl"
jq -c '{id,domain,prefix,arm,cap:1,kind,tokens:[.step.token_id],steps:[.step]}' \
  "$trial/teacher-witnesses.jsonl" >> "$out/plan.jsonl"
run_case() {
  local tag=$1 arm mode ids cap
  local -a args
  arm=$(jq -r '.arm' "$out/$tag-expected.json")
  args=()
  case "$arm" in
    frozen) mode=frozen ;;
    learned) mode=active ;;
    zero) mode=disabled ;;
    role-blind|wrong-address|shuffled-role) mode=$arm ;;
    *) exit 2 ;;
  esac
  if [ "$mode" != frozen ]; then args=(--p4-source-gate "$gate" --p4-source-mode "$mode"); fi
  ids=$(jq -r '.prefix | join(",")' "$out/$tag-expected.json")
  cap=$(jq -r '.cap' "$out/$tag-expected.json")
  UOR_P1_TIMING=1 /usr/bin/time -f 'wall_seconds=%e maximum_rss_kib=%M' \
    "$client" --discrete "$p4" --prompt-ids "$ids" --max-tokens "$cap" --p1-trace "${args[@]}" \
    > "$out/$tag-client.json" 2> "$out/$tag-time.log"
  jq -ce --arg mode "$mode" --slurpfile ref "$out/$tag-expected.json" \
    --slurpfile model "$trial/measurement.json" -f "$comparator" "$out/$tag-client.json" > "$out/$tag-match.json"
}
# Every public client starts cold. Independent cases may overlap on a host with
# sufficient memory. Per-case resource logs record that scheduling choice.
jobs=${S4_COLD_JOBS:-1}
[[ "$jobs" =~ ^[1-4]$ ]]
printf '%s\n' "$jobs" > "$out/parallelism"
i=0
running=()
wait_group() {
  local pid rc=0
  for pid in "${running[@]}"; do wait "$pid" || rc=1; done
  running=()
  return "$rc"
}
while IFS= read -r row; do
  i=$((i+1))
  tag=$(printf '%03d' "$i")
  printf '%s\n' "$row" > "$out/$tag-expected.json"
  run_case "$tag" &
  running+=("$!")
  if [ "${#running[@]}" -eq "$jobs" ]; then wait_group; fi
done < "$out/plan.jsonl"
wait_group
cat "$out"/[0-9][0-9][0-9]-match.json > "$out/matches.jsonl"
jq -ne --argjson n "$i" --slurpfile matched "$out/matches.jsonl" '$n==($matched|length)' > /dev/null
jq -es --slurpfile plan "$out/plan.jsonl" -f "$(dirname "$comparator")/cold-summary.jq" \
  "$out/matches.jsonl" > "$out/summary.json"
cat "$out/summary.json"
