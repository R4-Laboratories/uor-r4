#!/usr/bin/env bash
set -euo pipefail

root=$(realpath "${1:?component run root}")
out=${2:?new grade output directory}
archived=${3:-}
here=$(cd "$(dirname "$0")" && pwd)
runs=$(dirname "$root")
prior="$runs/20261127-codex-period-margin"
scope="$runs/20261127-codex-scope-trace"
factorial="$runs/20261127-codex-release-factorial"
initial="$runs/20261127-codex-joint-release"
raw="$root/measured-v1/raw"
(cd "$root/measured-v1" && sha256sum --quiet --strict -c OUTPUTS.sha256)
if [[ -n $archived ]]; then
  archived=$(realpath "$archived")
  (cd "$archived" && sha256sum --quiet --strict -c SHA256SUMS)
fi

mkdir "$out"
for split in train held quartet scope-dev scope-held prior-dev protect; do
  case $split in
    train|held) panel="$initial/panel-v1/$split"; frozen="$initial/measured-v1/raw/$split/frozen" ;;
    quartet) panel="$factorial/panel-v1/fresh"; frozen="$factorial/measured-v1/raw/fresh" ;;
    scope-dev|scope-held) panel="$scope/panel-v1/${split#scope-}"; frozen="$factorial/measured-v1/raw/${split#scope-}" ;;
    prior-dev) panel="$prior/panel-v1/dev"; frozen="$factorial/measured-v1/raw/prior-dev" ;;
    protect) panel="$prior/panel-v1/protect"; frozen="$factorial/measured-v1/raw/protect" ;;
  esac
  previous="$initial/grade-v1/$split"
  if [[ -n $archived ]]; then
    panel_rows="$archived/panels/$split.jsonl"
    panel_keys="$archived/panels/$split.keys.jsonl"
    previous="$archived/previous-grade/$split"
    old_full="$archived/historical/$split/full.jsonl"
    old_zero="$archived/historical/$split/zero.jsonl"
  else
    (cd "$panel" && sha256sum --quiet --strict -c SHA256SUMS)
    (cd "$previous" && sha256sum --quiet --strict -c SHA256SUMS)
    panel_rows="$panel/rows.jsonl"
    panel_keys="$panel/keys.jsonl"
    old_full="$initial/measured-v1/raw/$split/trained/joint.jsonl"
    old_zero="$frozen/candidate.jsonl"
  fi
  zero="$raw/$split/zero-period/first-period.jsonl"
  period="$raw/$split/zero-period/joint.jsonl"
  post_b="$raw/$split/post-full/first-period.jsonl"
  full="$raw/$split/post-full/joint.jsonl"
  cmp "$full" "$old_full"
  cmp "$zero" "$old_zero"

  grade="$out/$split"
  mkdir "$grade"
  sha256sum "$panel_rows" "$panel_keys" "$zero" "$period" "$post_b" "$full" \
    "$previous/balanced.rows.jsonl" "$previous/linked.rows.jsonl" "$old_full" "$old_zero" > "$grade/INPUTS.sha256"
  jq -nc --slurpfile panel "$panel_rows" --slurpfile keys "$panel_keys" \
    --slurpfile zero "$zero" --slurpfile period "$period" --slurpfile post_b "$post_b" --slurpfile full "$full" \
    -f "$here/rows.jq" > "$grade/rows.jsonl"
  jq -sc --arg split "$split" -f "$here/pairs.jq" "$grade/rows.jsonl" > "$grade/pairs.jsonl"
  jq -n --arg split "$split" --slurpfile rows "$grade/rows.jsonl" --slurpfile pairs "$grade/pairs.jsonl" \
    --slurpfile balanced "$previous/balanced.rows.jsonl" --slurpfile linked "$previous/linked.rows.jsonl" \
    -f "$here/summary.jq" > "$grade/summary.json"
  (cd "$grade" && sha256sum INPUTS.sha256 rows.jsonl pairs.jsonl summary.json > SHA256SUMS)
done
jq -s 'map(del(.examples, .by_family)
  | (.previous_balanced, .previous_linked) |= del(.previous_ids)
  | (.previous_balanced.models[], .previous_linked.models[]) |= del(.kept_ids, .lost_ids, .gained_ids))' \
  "$out"/*/summary.json > "$out/summary.json"
(cd "$out" && sha256sum -- */SHA256SUMS summary.json > SHA256SUMS)
