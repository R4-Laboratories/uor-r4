#!/usr/bin/env bash
set -euo pipefail
umask 077
restore=$(realpath "${1:?authenticated extracted packet}")
replay=$(realpath -m "${2:?fresh replay root}")
unit=$restore/source/docs/loop/tasks/written-state-count-model/codex-4338
parent=$unit/../codex-4337
fixtures=$restore/fixtures
positive_parent=$fixtures/20261127-codex-grounded-interval
official=$fixtures/20261127-codex-whole-turn-census/supply-inputs/squad-train-v2.0.json
assessment=$restore/roster/sealed/SEAL.sha256
mkdir "$replay"
bash "$unit/data/positive.sh" "$positive_parent" "$official" \
    "$parent/supply/pilot-extract.jq" "$replay/positive"
bash "$unit/data/answerability.sh" "$fixtures" "$assessment" "$replay/answer"
for entry in picks.jsonl rows.jsonl SEALED.sha256; do
    cmp "$restore/unit/positive-input/$entry" "$replay/positive/$entry"
done
cmp "$restore/unit/answer-input/rows.jsonl" "$replay/answer/rows.jsonl"
# The captured answerability source manifest records the original absolute paths.
# Compare its expected digests independently of the relocated restore path.
cmp <(cut -d ' ' -f 1 "$restore/unit/answer-input/INPUTS.sha256") \
    <(cut -d ' ' -f 1 "$replay/answer/INPUTS.sha256")

head -n 1 "$assessment" > "$replay/cut-assessment.sha256"
if bash "$unit/data/answerability.sh" "$fixtures" "$replay/cut-assessment.sha256" \
    "$replay/forged-answerability" > "$replay/forged.log" 2>&1; then
    printf 'Accepted a truncated assessment seal\n' >&2
    exit 1
fi
test ! -e "$replay/forged-answerability"
