#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
grade=$1; manifest=$2; out=$3; mkdir "$out"
while IFS=$'\t' read -r id m e mid eid src sha; do
    jq -enc --arg id "$id" --arg mid "$mid" --arg eid "$eid" --arg src "$src" --arg sha "$sha" \
        --argjson m "$m" --argjson e "$e" --slurpfile t "$grade/$id/train.json" \
        --slurpfile r "$grade/$id/regression.json" --slurpfile n "$grade/$id/near.json" \
        '{id:$id,m:$m,e:$e,mid:$mid,eid:$eid,src:$src,sha:$sha,t:$t[0],reg:$r[0].source,near:$n[0].correct,
          regression_ids:$r[0].source_ids,near_ids:$n[0].wins}' >> "$out/records.jsonl"
done < <(tail -n +2 "$manifest")
jq -es -f "$here/elect.jq" "$out/records.jsonl" > "$out/election.json"
(cd "$out" && sha256sum {records.jsonl,election.json} > SHA256SUMS)
