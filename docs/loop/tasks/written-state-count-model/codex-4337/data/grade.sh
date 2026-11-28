#!/usr/bin/env bash
set -euo pipefail
umask 077
gold=$(realpath "${1:?gold JSONL}")
truth=$(realpath "${2:?independent truth JSONL}")
records=$(realpath "${3:?ordinary client JSONL}")
arm=${4:?arm}
out=${5:?new output directory}
visible=${6:-/dev/null}
decisions=$(realpath "${7:?frozen decision-reference.json}")
decision_sha=${8:?published decision-reference SHA-256}
# The qualified arms and panel rosters bind this independent cold replay.
if [[ "$decision_sha" != 6c29462db91865a5081f604d91d03fb8759aecd90795912c7ab69856d2095f0f ]]; then
    echo 'unrecognized published decision reference' >&2
    exit 1
fi
printf '%s  %s\n' "$decision_sha" "$decisions" | sha256sum --strict --quiet -c -
gold_sha=$(sha256sum "$gold" | cut -d ' ' -f1)
truth_sha=$(sha256sum "$truth" | cut -d ' ' -f1)
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
jq -es 'all(.[];(.grade|IN("visible_abstention","false_factual_claim","no_decipherable_claim","empty_reply")) and (.reason|type=="string" and length>0)) and (map(.case_id)| length == (unique|length))' "$visible" > /dev/null
jq -esc --arg arm "$arm" --arg gold_sha "$gold_sha" --arg truth_sha "$truth_sha" \
    --slurpfile gold "$gold" --slurpfile truth "$truth" --slurpfile visible "$visible" \
    --slurpfile decisions "$decisions" -f "$here/client-grade.jq" "$records" > "$out/grade.json"
jq 'del(.rows,.matched_pairs)' "$out/grade.json" > "$out/summary.json"
sha256sum "$gold" "$truth" "$records" "$here/client-grade.jq" "$here/grade.sh" "$visible" "$decisions" > "$out/INPUTS.sha256"
(cd "$out" && sha256sum grade.json summary.json INPUTS.sha256 > OUTPUTS.sha256 && sha256sum --check --strict OUTPUTS.sha256)
