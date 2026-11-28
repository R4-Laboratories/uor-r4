#!/usr/bin/env bash
# Prepare packed GPU inputs on Wick without accessing the HELD gold tape.
set -euo pipefail
umask 077

if [[ $# != 8 ]]; then
    echo 'usage: prepare.sh CLIENT FIT BASE TOKENIZER FULL_TRAIN_PANEL PROTECTED_PANEL HELD_PANEL FRESH_OUTPUT' >&2
    exit 2
fi
client=$(realpath "$1")
fit=$(realpath "$2")
base=$(realpath "$3")
tokenizer=$(realpath "$4")
train=$(realpath "$5")
protected=$(realpath "$6")
held=$(realpath "$7")
out=$(realpath -m "$8")

(cd "$train" && sha256sum --strict -c SEALED.sha256)
(cd "$protected" && awk '$2 == "protected-input.jsonl"' SEALED.sha256 | sha256sum --strict -c -)
(cd "$held" && awk '$2 == "held-input.jsonl"' SEALED.sha256 | sha256sum --strict -c -)
mkdir "$out"
cd "$out"
mkdir features device
sha256sum "$client" "$fit" "$base" "$tokenizer" "$train/SEALED.sha256" "$protected/SEALED.sha256" "$held/SEALED.sha256" \
    > SOURCES.sha256

"$client" --discrete "$base" --tokenizer "$tokenizer" --interval-features "$train/train-input.jsonl" \
    > features/train.jsonl 2> features/train.stderr
"$client" --discrete "$base" --tokenizer "$tokenizer" --interval-features "$protected/protected-input.jsonl" \
    > features/protected.jsonl 2> features/protected.stderr
"$client" --discrete "$base" --tokenizer "$tokenizer" --interval-features "$held/held-input.jsonl" \
    > features/held.jsonl 2> features/held.stderr

# The Fenced copy cursor retains the frozen period stop. Require the single
# named exclusion from the sealed original-token truth, with its exact span.
case_id=56f7366c711bf01900a44a65
jq -ecn --arg id "$case_id" --slurpfile gold "$train/train-gold.jsonl" \
    --slurpfile truth "$train/train-truth.jsonl" '
    ($gold | map(select(.case_id == $id))) as $label |
    ($truth | map(select(.case_id == $id))) as $row |
    if ($label|length) != 1 or ($row|length) != 1 or ($row[0].source_ids | index(16)) == null
    then error("locked-period case missing from sealed truth or gold")
    else {case_id:$id, span:($label[0] | {start,end}), original_ids:$row[0].source_ids} end
' > features/exclusions.jsonl

"$fit" coverage features/train.jsonl "$train/train-gold.jsonl" > features/train.coverage.json
for arm in geometry count; do
    "$fit" prepare-train features/train.jsonl "$train/train-gold.jsonl" features/exclusions.jsonl \
        "$arm" "device/$arm.train.tape" > "features/$arm.train.inventory.json"
    for split in train protected held; do
        "$fit" prepare-eval "features/$split.jsonl" "$arm" "device/$arm.$split.eval.tape"
    done
done
cmp features/geometry.train.inventory.json features/count.train.inventory.json
"$fit" zero-device device/zero.words
(cd device && sha256sum ./*.train.tape > TRAIN-INPUTS.sha256)
(cd device && sha256sum ./*.eval.tape zero.words > EVAL-INPUTS.sha256)
sha256sum features/*.json features/*.jsonl device/* SOURCES.sha256 > OUTPUTS.sha256
