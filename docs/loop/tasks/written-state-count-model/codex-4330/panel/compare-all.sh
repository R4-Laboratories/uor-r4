#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 4 ]]; then
  echo 'usage: compare-all.sh WICK_RUNS BATCH_CLIENT VERIFIED_CONTROLS NEW_OUTPUT_DIR' >&2
  exit 2
fi

runs=$(realpath "$1")
batch=$(realpath "$2")
controls=$(realpath "$3")
output=$(realpath -m "$4")
unit=$(dirname "$(realpath "$0")")
root="$runs/20261127-codex-whole-turn-fit"
panel="$root/panel-v1"
free="$root/free-chat-v1"
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
[[ $output == "$root/"* ]]
(cd "$controls" && sha256sum --quiet --strict -c SEALED.sha256)
for arm in old frozen fitted; do
  cmp "$controls/$arm-batch.canonical.jsonl" "$controls/$arm-fresh.canonical.jsonl"
  [[ $(wc -l < "$controls/$arm-batch.canonical.jsonl") == 103 ]]
done

declare -A artifact pins
artifact[old]="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
artifact[frozen]="$root/product-v1/frozen.d11"
artifact[fitted]="$root/product-v1/fitted.d11"
pins[old]=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
pins[frozen]=6415f6cb9fbf4694b449c08170d17c7e821c168d28e811cfa4e7a3448a333422
pins[fitted]=84680646cce69b2a9f8090071e6419a801d8c6aa1bea7f10130379fbd44fbb24
prompts_sha=9759d2dd35547bc478b5049dc53c6bdf481db39f014330146439360d53f520dc
free_sha=75973980526412ac17ecd2fc424e2cab4439845e07f390ccc5f72d76546c3651

(cd "$panel" && sha256sum --quiet --strict -c SEALED.sha256)
(cd "$free" && sha256sum --quiet --strict -c SEALED.sha256)
mkdir "$output"
for arm in old frozen fitted; do
  bash "$unit/batch.sh" "$batch" "$tokenizer" "$panel/prompts.jsonl" "$prompts_sha" \
    "$arm" "${artifact[$arm]}" "${pins[$arm]}" "$output/$arm"
  jq -n --arg arm "$arm" --slurpfile eval "$panel/eval.jsonl" --slurpfile gold "$panel/gold.jsonl" \
    --slurpfile ordinary "$output/$arm/ordinary.jsonl" -f "$unit/grade.jq" \
    > "$output/$arm.grade.json"
  jq 'del(.rows)' "$output/$arm.grade.json" > "$output/$arm.summary.json"
  bash "$unit/batch.sh" "$batch" "$tokenizer" "$free/prompts.jsonl" "$free_sha" \
    "$arm" "${artifact[$arm]}" "${pins[$arm]}" "$output/free-$arm"
  jq -n --slurpfile sources "$free/sources.jsonl" --slurpfile ordinary "$output/free-$arm/ordinary.jsonl" \
    -f "$unit/free-grade.jq" > "$output/free-$arm.json"
done
jq -n --slurpfile old "$output/old.grade.json" --slurpfile frozen "$output/frozen.grade.json" \
  --slurpfile fitted "$output/fitted.grade.json" -f "$unit/comparison.jq" > "$output/comparison.json"
(
  cd "$output"
  sha256sum old.grade.json frozen.grade.json fitted.grade.json \
    old.summary.json frozen.summary.json fitted.summary.json comparison.json \
    free-old.json free-frozen.json free-fitted.json \
    {old,frozen,fitted,free-old,free-frozen,free-fitted}/SEALED.sha256 > SEALED.sha256
  sha256sum --quiet --strict -c SEALED.sha256
)
