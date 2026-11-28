#!/usr/bin/env bash
set -euo pipefail
gold=${1:?gold JSONL}
decisions=${2:?device decisions JSONL}
out=${3:?new analysis directory}
mkdir "$out"
here=$(cd "$(dirname "$0")" && pwd)
# One unambiguous official question ID per entire flagged pair.
bad='["5a5ab76c9c0277001abe714b","5a5ad4db9c0277001abe71ce","5a7b67b621c2de001afe9fd6","5ad4c5ed5b96ef001a10a009","5ad4f0d05b96ef001a10a6f2","5ad601975b96ef001a10b023","5ad3f788604f3c001a3ffa23","5ad050c877cf76001a686fc2"]'
jq -cs --argjson bad "$bad" 'map(select(.source_id as $id| $bad|index($id)))|map(.contrast_group)|unique|if length==8 then . else error("flagged ID missing") end' "$gold" > "$out/groups.json"
jq -c --slurpfile bad "$out/groups.json" 'select(.contrast_group as $g | $bad[0]|index($g)==null)' "$gold" > "$out/gold.jsonl"
jq -nc --slurpfile keep "$out/gold.jsonl" '($keep|INDEX(.case_id)) as $index|inputs|select($index[.case_id]!=null)' "$decisions" | jq -nc --slurpfile gold "$out/gold.jsonl" -f "$here/source-grade.jq" > "$out/source.json"
