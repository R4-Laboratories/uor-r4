#!/usr/bin/env bash
set -euo pipefail
root=${1:?study root}; split=${2:?panel}; model=${3:?candidate or balanced}; destination=${4:?new output directory}
here=$(cd "$(dirname "$0")" && pwd)
siblings=$(dirname "$root")
scope="$siblings/20261127-codex-scope-trace"
prior="$siblings/20261127-codex-period-margin"
old_model=$model
if [[ $model == balanced ]]; then old_model=baseline; fi
case $split in
 fresh) panel="$root/panel-v1/fresh"; old=/dev/null;;
 held) panel="$scope/panel-v1/held"; old="$scope/held-v1/held-out/$old_model/observations.jsonl";;
 dev) panel="$scope/panel-v1/dev"; old="$scope/fit-v1/fit-out/$old_model/dev/observations.jsonl";;
 prior-dev|protect) p=$split; [[ $p != prior-dev ]] || p=dev; panel="$prior/panel-v1/$p"; old="$scope/fit-v1/fit-out/$old_model/$split/observations.jsonl";;
 *) exit 2;;
esac
(cd "$panel" && sha256sum --quiet --strict -c SHA256SUMS)
if [[ $old != /dev/null ]]; then
 manifest="$scope/fit-v1/OUTPUTS.sha256"; [[ $split != held ]] || manifest="$scope/held-v1/OUTPUTS.sha256"
 saved=$(awk -v f="${old#$(dirname "$manifest")/}" '$2==f{print $1}' "$manifest")
 test "$(sha256sum "$old"|cut -d' ' -f1)" = "$saved"
fi
mkdir -p "$(dirname "$destination")"; mkdir "$destination"
traces="$root/measured-v1/raw/$split"
jq -n -c --slurpfile rows "$traces/$model.jsonl" --slurpfile old "$old" --slurpfile keys "$panel/keys.jsonl" --slurpfile panel "$panel/rows.jsonl" -f "$here/rows.jq" > "$destination/rows.jsonl"
jq -s --arg split "$split" -f "$here/counts.jq" "$destination/rows.jsonl" > "$destination/summary.json"
