#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(realpath "${1:?unit root on Wick}")
client=$(realpath "${2:?pinned client}")
base=$(realpath "${3:?balanced D11}")
tok=$(realpath "${4:?tokenizer}")
out=$(realpath -m "${5:?fresh output}")
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
# The upstream screen already excludes older exposures. Exclude every article
# used in the current study's fresh human panel from this prospective supply.
jq -sc --slurpfile held "$run/data/held-v1/held-truth.jsonl" '
  ($held|map(.article)|map(select(.!=null))|unique) as $forbidden |
  map(select(.article as $a | $forbidden|index($a)==null)) | group_by(.passage_sha256) |
  map(select(length>=2)) | .[][]' "$run/data/held-candidates-v2/candidates.jsonl" > "$out/questions.jsonl"
jq -c '{case_id,text}' "$out/questions.jsonl" > "$out/blind.jsonl"
"$client" --discrete "$base" --tokenizer "$tok" --interval-features "$out/blind.jsonl" > "$out/features.jsonl"
jq -s --slurpfile features "$out/features.jsonl" -f "$here/pairs.jq" "$out/questions.jsonl" > "$out/pairs.json"
jq --slurpfile q "$out/questions.jsonl" '{
  screened_questions:($q|length),candidate_passages:($q|map(.passage_sha256)|unique|length),
  mechanically_supported_pairs:length,distinct_questions:(map([.left,.right])|flatten|unique|length),
  distinct_passages:(map(.passage)|unique|length),distinct_articles:(map(.article)|unique|length),
  both_immediate:map(select(.left_offset==0 and .right_offset==0))|length
}' "$out/pairs.json" > "$out/summary.json"
sha256sum "$run/data/held-candidates-v2/candidates.jsonl" "$run/data/held-v1/held-truth.jsonl" \
    "$client" "$here/count.sh" "$here/pairs.jq" > "$out/INPUTS.sha256"
(cd "$out" && sha256sum questions.jsonl blind.jsonl features.jsonl pairs.json summary.json > OUTPUTS.sha256)
