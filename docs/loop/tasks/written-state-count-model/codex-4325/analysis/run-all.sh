#!/usr/bin/env bash
set -euo pipefail
root=${1:?Wick study root}; here=$(cd "$(dirname "$0")" && pwd)
sibling=$(dirname "$root")
old="$sibling/20261127-codex-release-factorial"; scope="$sibling/20261127-codex-scope-trace"; prior="$sibling/20261127-codex-period-margin"
(cd "$root/measured-v1" && sha256sum --strict --quiet -c OUTPUTS.sha256)
(cd "$root" && sha256sum --strict --quiet -c measured-v1/wick.sha256)
(cd "$old/measured-v1" && sha256sum --strict --quiet -c OUTPUTS.sha256)
prepare_model() {
 local model=$1 file
 for split in held train quartet scope-dev scope-held prior-dev protect; do
  case "$split" in
    held|train) panel="$root/panel-v1/$split";;
    quartet) panel="$old/panel-v1/fresh";;
    scope-dev|scope-held) panel="$scope/panel-v1/${split#scope-}";;
    prior-dev) panel="$prior/panel-v1/dev";;
    protect) panel="$prior/panel-v1/protect";;
  esac
  raw="$root/measured-v1/raw/$split/trained"; file=$model
  if [[ $model == candidate || $model == balanced ]]; then
    if [[ $split != train && $split != held ]]; then continue; fi
    raw="$root/measured-v1/raw/$split/frozen"
  fi
  bash "$here/../product/replay.sh" "$root" "$model" "$split" "$panel/rows.jsonl" "$raw/$file.jsonl" "$raw/$file-reverse.jsonl"
 done
}
for model in first-period joint; do
 bash "$here/../product/pack.sh" "$root" "$model" "$root/measured-v1/raw/fit/$model.i16le" "$root/measured-v1/raw/fit/fit.json"
done
# Limit parallel client processes on the shared Wick system.
for model in first-period joint; do prepare_model "$model" > "$root/product-v1/$model/replay.log" 2>&1 & pids+=("$!"); done
rc=0; for pid in "${pids[@]}";do wait "$pid" || rc=1;done; test "$rc" = 0
pids=()
for model in candidate balanced; do prepare_model "$model" > "$root/product-v1/$model.replay.log" 2>&1 & pids+=("$!");done
rc=0; for pid in "${pids[@]}";do wait "$pid" || rc=1;done; test "$rc" = 0
for split in held train quartet scope-dev scope-held prior-dev protect; do
 bash "$here/run-grade.sh" "$root" "$split"
done
d="$root/grade-v1"
jq -s . "$d/held/"{balanced,linked,first-period,joint}.json > "$d/fresh-models.json"
jq -s . "$d/"{quartet,scope-dev,scope-held,prior-dev,protect}/comparison.json > "$d/regressions.json"
jq -n --slurpfile fresh_json "$d/fresh-models.json" --slurpfile old_json "$d/regressions.json" \
  -L "$here" 'include "decision" ; $fresh_json[0] as $fresh | $old_json[0] as $oldcomparisons | decide($fresh;$oldcomparisons)' > "$d/decision.json"
