#!/usr/bin/env bash
set -euo pipefail
root=${1:?Wick run root}
arm=${2:?baseline or candidate}
split=${3:?split}
here=$(cd "$(dirname "$0")" && pwd)
# The private restore packet stages the pinned helpers alongside this script.
audit="$here/audit.jq"
helpers="$here"
if [[ ! -f "$audit" ]]; then
  audit="$here/../../codex-4321/audit.jq"
  helpers="$here/../../codex-4322/analysis"
fi
for filter in "$audit" "$helpers/grade.jq" "$helpers/client.jq" "$helpers/protect.jq"; do test -s "$filter"; done
bin=${SCOPE_BIN:-/home/callard/uor-codex/20261127-scope-product/target/release}
prev="$(dirname "$root")/20261127-codex-period-margin"
case "$split" in
 train|dev|held) panel="$root/panel-v1/$split";;
 protect) panel="$prev/panel-v1/protect";;
 prior-dev) panel="$prev/panel-v1/dev";;
 *) exit 2;;
esac
case "$arm" in
 baseline) artifact="$prev/cold-v1/balanced/artifact.d11";;
 candidate) artifact="$root/product-v1/artifact.d11";;
 position) artifact="$root/position-v1/artifact.d11";;
 *) exit 2;;
esac
raw="$root/fit-v1/fit-out/$arm/$split"
if [[ $split == held ]]; then raw="$root/held-v1/held-out/$arm"; fi
data="$root/qualification-v1/$arm/$split"
mkdir -p "$(dirname "$data")"
mkdir "$data"
rows="$panel/rows.jsonl"; keys="$panel/keys.jsonl"; forward="$raw/observations.jsonl"
(cd "$panel" && sha256sum --quiet --strict -c SHA256SUMS)
jq -ne --slurpfile f "$forward" --slurpfile r "$raw/reverse.jsonl" '($f|length)>0 and $f==($r|reverse)' >/dev/null
jq -n --slurpfile p "$rows" --slurpfile t "$forward" -f "$audit" > "$data/audit.json"
jq -n --slurpfile keys "$keys" --slurpfile rows "$forward" -f "$helpers/grade.jq" > "$data/grade.json"
if [[ $split == train || $split == dev || $split == held ]]; then
  jq --slurpfile keys "$keys" -f "$here/axis.jq" "$data/grade.json" > "$data/axis.json"
fi
"$bin/verify-native-source-choice" "$artifact" "$rows" "$forward" > "$data/cold.jsonl" 2> "$data/cold-receipt.json"
jq -c '{prefix,positions}' "$rows" > "$data/requests.jsonl"
"$bin/serve-native-source-choice" "$artifact" < "$data/requests.jsonl" > "$data/client.jsonl"
jq -n --slurpfile device "$forward" --slurpfile client "$data/client.jsonl" -f "$helpers/client.jq" > "$data/client-receipt.json"
if [[ $split == protect ]]; then
  jq -n --slurpfile prior "$(dirname "$root")/20261127-codex-source-choice/product-qualified-v1/cold.jsonl" --slurpfile measured "$forward" -f "$helpers/protect.jq" > "$data/protected-original.json"
fi
(cd "$data" && sha256sum *.json *.jsonl > SHA256SUMS)
