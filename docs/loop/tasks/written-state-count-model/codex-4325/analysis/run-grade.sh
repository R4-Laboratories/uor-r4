#!/usr/bin/env bash
set -euo pipefail
root=${1:?study root}; split=${2:?panel name}; here=$(cd "$(dirname "$0")" && pwd)
old=$(dirname "$root")/20261127-codex-release-factorial
scope=$(dirname "$root")/20261127-codex-scope-trace
prior=$(dirname "$root")/20261127-codex-period-margin
case "$split" in
 train|held) p="$root/panel-v1/$split"; frozen="$root/measured-v1/raw/$split/frozen";;
 quartet) p="$old/panel-v1/fresh"; frozen="$old/measured-v1/raw/fresh";;
 scope-dev|scope-held) suffix=${split#scope-}; p="$scope/panel-v1/$suffix"; frozen="$old/measured-v1/raw/$suffix";;
 prior-dev) p="$prior/panel-v1/dev"; frozen="$old/measured-v1/raw/prior-dev";;
 protect) p="$prior/panel-v1/protect"; frozen="$old/measured-v1/raw/protect";;
 *) exit 2;;
esac
(cd "$p" && sha256sum --quiet --strict -c SHA256SUMS)
out="$root/grade-v1/$split"; mkdir -p "$(dirname "$out")"; mkdir "$out"
for model in balanced linked first-period joint; do
 file="$root/measured-v1/raw/$split/trained/$model.jsonl"
 if [[ $model == linked ]];then file="$frozen/candidate.jsonl";elif [[ $model == balanced ]];then file="$frozen/balanced.jsonl";fi
 jq -nc --slurpfile panel "$p/rows.jsonl" --slurpfile keys "$p/keys.jsonl" --slurpfile rows "$file" -f "$here/observations.jq" > "$out/$model.rows.jsonl"
 jq -s --arg split "$split" -f "$here/results.jq" "$out/$model.rows.jsonl" > "$out/$model.json"
done
jq -n --arg split "$split" --slurpfile balanced "$out/balanced.rows.jsonl" --slurpfile linked "$out/linked.rows.jsonl" --slurpfile first "$out/first-period.rows.jsonl" --slurpfile joint "$out/joint.rows.jsonl" -f "$here/compare.jq" > "$out/comparison.json"
(cd "$out" && sha256sum *.json *.jsonl > SHA256SUMS)
