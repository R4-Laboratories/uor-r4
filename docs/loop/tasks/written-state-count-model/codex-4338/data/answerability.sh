#!/usr/bin/env bash
set -euo pipefail
umask 077
runs=$(realpath "${1:?private runs directory}")
assessment=$(realpath "${2:?article split seal registered before TRAIN adjudication}")
out=$(realpath -m "${3:?fresh output directory}")
here=$(cd "$(dirname "$0")" && pwd)
up="$runs/20261127-codex-whole-turn-census"
parent="$runs/20261127-codex-grounded-interval"
echo "1899b7c8e52bc56e08d084e1286855e167d4a5665c13a72538e75e9f1c6c984f  $assessment" | sha256sum --quiet --strict -c -
(cd "$(dirname "$assessment")" && sha256sum --quiet --strict -c "$assessment" && sha256sum --quiet --strict -c ROSTER.sha256)
echo "b8cc4c6ff0e65f731cf909df48c72e1497d72217d431d1d75c23ae50cdd5201e  $up/supply-v1/SEALED.sha256" | sha256sum --quiet --strict -c -
(cd "$up/supply-v1" && sha256sum --quiet --strict -c SEALED.sha256)
official="$up/supply-inputs/squad-train-v2.0.json"
echo "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002  $official" | sha256sum --quiet --strict -c -
mkdir "$out"
jq -c --slurpfile truth "$up/supply-v1/train-truth.jsonl" --slurpfile prompts "$up/supply-v1/train-prompts.jsonl" \
    -f "$here/answerability.jq" "$official" > "$out/rows.jsonl"
jq -es 'length==130 and (map(.passage_sha256)|unique|length)==130 and
    (map(.left.case_id,.right.case_id)|unique|length)==260' "$out/rows.jsonl" >/dev/null
while IFS= read -r row; do
    digest=$(jq -j .context <<< "$row" | sha256sum | cut -d" " -f1)
    test "$digest" = "$(jq -r .passage_sha256 <<< "$row")"
done < "$out/rows.jsonl"
sha256sum "$assessment" "$parent/next-supply-pilot-v1/split.tsv" "$official" "$here/answerability.jq" \
    "$up/supply-v1/train-truth.jsonl" "$up/supply-v1/train-prompts.jsonl" > "$out/INPUTS.sha256"
(cd "$out" && sha256sum rows.jsonl INPUTS.sha256 > SEALED.sha256)
