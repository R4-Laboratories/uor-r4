#!/usr/bin/env bash
set -euo pipefail
root=${1:?ZFS study root}; tools=$(cd "$(dirname "$0")" && pwd)
siblings=$(dirname "$root")
for split in fresh dev held prior-dev protect; do
 case "$split" in
 fresh) panel="$root/panel-v1/fresh";;
 dev|held) panel="$siblings/20261127-codex-scope-trace/panel-v1/$split";;
 prior-dev) panel="$siblings/20261127-codex-period-margin/panel-v1/dev";;
 protect) panel="$siblings/20261127-codex-period-margin/panel-v1/protect";;
 esac
 for model in balanced candidate; do
  raw="$root/measured-v1/raw/$split"
  bash "$tools/../product/replay.sh" "$root" "$model" "$split" "$panel/rows.jsonl" "$raw/$model.jsonl" "$raw/$model-reverse.jsonl"
  bash "$tools/qualify.sh" "$root" "$split" "$model" "$root/grade-v1/$split/$model"
 done
 jq -n --arg split "$split" --slurpfile balanced "$root/grade-v1/$split/balanced/rows.jsonl" --slurpfile candidate "$root/grade-v1/$split/candidate/rows.jsonl" -f "$tools/retention.jq" > "$root/grade-v1/$split/identities.json"
done
(cd "$root/grade-v1" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
