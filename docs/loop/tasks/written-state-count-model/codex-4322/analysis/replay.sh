#!/usr/bin/env bash
set -euo pipefail

# Run on Wick from copied unit analysis sources. Its audit.jq is the frozen #4321 audit.
analysis=$(cd -- "$(dirname -- "$0")" && pwd)
root=${PERIOD_ROOT:-/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-period-margin}
prior="$(dirname "$root")/20261127-codex-source-choice"
word="$(dirname "$root")/20261127-codex-word-ending"
base="$word/product-qualified-v1/artifact.d11"
bin=${SOURCE_CHOICE_BIN:-/home/callard/uor-codex/20261127-source-choice-product/target/release}
config=${1:?configuration required}
split=${2:?split required}
grading="$root/analysis-v1/$config/$split"
cold="$root/cold-v1/$config/$split"
fit="$root/phase1-v1/output/$config/fit"
case "$split" in
  train) panel="$prior/panel-v1/train"; raw="$root/phase1-v1/output/$config/train" ;;
  dev|protect) panel="$root/panel-v1/$split"; raw="$root/phase1-v1/output/$config/$split" ;;
  held) panel="$root/panel-v1/held"; raw="$root/phase2-v1/held-output/$config" ;;
  *) exit 2 ;;
esac
case "$config" in baseline|balanced|weak|both) ;; *) exit 2 ;; esac

rows="$panel/rows.jsonl"
keys="$panel/keys.jsonl"
forward="$raw/observations.jsonl"
reverse="$raw/reverse.jsonl"
# Fail before grading if the raw return, reference, or checkpoint is incomplete.
test -s "$forward" && test -s "$reverse"
(cd "$panel" && sha256sum --quiet --strict -c SHA256SUMS)
jq -n -e --slurpfile f "$forward" --slurpfile r "$reverse" \
  '($f|length)>0 and ($f==($r|reverse))' >/dev/null
mkdir -p "$grading" "$cold"
jq -n --slurpfile p "$rows" --slurpfile t "$forward" -f "$analysis/audit.jq" > "$grading/audit.json"
jq -n --slurpfile keys "$keys" --slurpfile rows "$forward" -f "$analysis/grade.jq" > "$grading/grade.json"
if [[ $config != baseline ]]; then
  jq --slurpfile base "$root/analysis-v1/baseline/$split/grade.json" \
    -f "$analysis/effects.jq" "$grading/grade.json" > "$grading/effects.json"
fi

if [[ $split == protect ]]; then
  jq -n --slurpfile prior "$prior/product-qualified-v1/cold.jsonl" --slurpfile measured "$forward" \
    -f "$analysis/protect.jq" > "$grading/protected-original.json"
fi

checkpoint="$root/cold-v1/$config"
artifact="$checkpoint/artifact.d11"
fit_sha=$(sha256sum "$fit/train.json" | cut -d' ' -f1)
weights="$fit/weights.i16le"
prior_weights="$prior/measurement-v1/fit/weights.i16le"
test "$(wc -c < "$weights")" -eq 24582
cmp -i 8194:8194 "$prior_weights" "$weights"
if [[ $config == baseline ]]; then cmp "$prior_weights" "$weights"; fi
state_sha=$(sha256sum "$weights" | cut -d' ' -f1)
if [[ ! -f "$artifact" ]]; then
  "$bin/pack-native-source-choice" "$base" "$weights" "$artifact" "$fit_sha" \
    > "$checkpoint/artifact.json"
  sha256sum "$artifact" > "$checkpoint/artifact.sha256"
fi
jq -e --arg fit "$fit_sha" --arg state "$state_sha" '.source_choice.fit_sha256==$fit and .source_choice.state_sha256==$state' \
  "$checkpoint/artifact.json" >/dev/null
"$bin/verify-native-source-choice" "$artifact" "$rows" "$forward" \
  > "$cold/cold.jsonl" 2> "$cold/cold.receipt.json"
jq -c '{prefix,positions}' "$rows" > "$cold/requests.jsonl"
"$bin/serve-native-source-choice" "$artifact" < "$cold/requests.jsonl" > "$cold/client.jsonl"
jq -n --slurpfile device "$forward" --slurpfile client "$cold/client.jsonl" \
  -f "$analysis/client.jq" > "$cold/client.receipt.json"
(cd "$cold" && sha256sum cold.jsonl cold.receipt.json requests.jsonl client.jsonl client.receipt.json > SHA256SUMS)
(cd "$grading" && sha256sum ./*.json > SHA256SUMS)
