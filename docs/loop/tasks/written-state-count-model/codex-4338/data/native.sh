#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 8 ]]; then
    echo 'usage: native.sh CLIENT PRIOR_FIT FIT OPENING_BINDER BASE TOKENIZER QUALIFIED_TRAIN FRESH_OUTPUT' >&2
    exit 2
fi
client=$(realpath "$1")
prior_fit=$(realpath "$2")
fit=$(realpath "$3")
binder=$(realpath "$4")
base=$(realpath "$5")
tokenizer=$(realpath "$6")
train=$(realpath "$7")
out=$(realpath -m "$8")
here=$(cd "$(dirname "$0")" && pwd)

(cd "$train" && sha256sum --strict -c SEALED.sha256)
mkdir "$out"
cd "$out"
mkdir device
sha256sum "$client" "$prior_fit" "$fit" "$binder" "$base" "$tokenizer" \
    "$here/native.sh" "$here/native.jq" "$train/SEALED.sha256" > SOURCES.sha256

"$client" --discrete "$base" --tokenizer "$tokenizer" --interval-features "$train/train-input.jsonl" \
    > train.prospective.features.jsonl 2> capture.stderr
"$binder" "$tokenizer" "$train/train-input.jsonl" > train.openings.jsonl
"$prior_fit" coverage train.prospective.features.jsonl "$train/train-gold.jsonl" > coverage.prospective.json
jq -nc --slurpfile train "$train/train-input.jsonl" --slurpfile gold "$train/train-gold.jsonl" \
    --slurpfile truth "$train/train-truth.jsonl" --slurpfile opening train.openings.jsonl \
    --slurpfile capture train.prospective.features.jsonl -f "$here/native.jq" > eligibility.jsonl
jq -c 'select(.retained | not)' eligibility.jsonl > excluded.jsonl
for part in input gold truth; do
    jq -c --slurpfile eligibility eligibility.jsonl \
        '($eligibility | map(select(.retained)) | INDEX(.case_id)) as $keep |
         select($keep[.case_id] != null)' "$train/train-$part.jsonl" > "train-$part.jsonl"
done
jq -c --slurpfile eligibility eligibility.jsonl \
    '($eligibility | map(select(.retained)) | INDEX(.case_id)) as $keep |
     select($keep[.case_id] != null)' train.prospective.features.jsonl > train.features.jsonl
"$prior_fit" coverage train.features.jsonl train-gold.jsonl > coverage.json
"$fit" coverage train.features.jsonl train-gold.jsonl > coverage.rebuilt.json
cmp coverage.json coverage.rebuilt.json
jq -e '(.cases | length) > 0 and all(.cases[];
    .gold_present and .swapped_gold_present != false and .choices >= 2)' coverage.json > /dev/null

: > exclusions.period.jsonl
for arm in geometry count; do
    "$fit" prepare-train train.features.jsonl train-gold.jsonl exclusions.period.jsonl \
        "$arm" "device/$arm.train.tape" > "$arm.train.inventory.json"
    "$fit" prepare-eval train.features.jsonl "$arm" "device/$arm.train.eval.tape"
done
cmp geometry.train.inventory.json count.train.inventory.json
"$fit" zero-device device/zero.words
(cd device && sha256sum ./*.train.tape > TRAIN-INPUTS.sha256
    sha256sum ./*.eval.tape zero.words > EVAL-INPUTS.sha256)

jq -n --slurpfile eligibility eligibility.jsonl --slurpfile coverage coverage.json \
    --slurpfile inventory geometry.train.inventory.json '
    def paired: map(select(.swapped_case_id != null and .case_id < .swapped_case_id));
    def by_tape: group_by(.tape) | map({tape:.[0].tape, rows:length,
      pairs:paired|length});
    {schema:"uor-r4.natural-train-native-qualification/1",
     prospective_rows:($eligibility|length),
     prospective_by_tape:($eligibility|by_tape),
     retained_rows:($eligibility|map(select(.retained))|length),
     retained_by_tape:($eligibility|map(select(.retained))|by_tape),
     excluded_rows:($eligibility|map(select(.retained|not))|length),
     exclusion_reasons:($eligibility|map(.reasons[])|group_by(.)|map({reason:.[0],rows:length})),
     excluded_pairs:($eligibility|paired|map(select(.retained|not))|length),
     pair_exclusion_reasons:($eligibility|paired|
       map((.reasons + .swapped_reasons)|unique[])|
       group_by(.)|map({reason:.[0],pairs:length})),
     choices:($coverage[0].cases|map(.choices)|add),
     all_own_and_reciprocal_targets_covered:true,
     zero_table:($coverage[0]|del(.cases)),
     packer:$inventory[0],
     model_fit:"NOT_RUN",validation_or_locked_input:"NOT_ACCESSED"}' > evidence.json
jq 'del(.schema,.zero_table,.packer.excluded)' evidence.json > summary.json
sha256sum SOURCES.sha256 capture.stderr coverage*.json ./*.jsonl ./*.inventory.json \
    evidence.json summary.json device/* > SEALED.sha256
sha256sum --strict -c SEALED.sha256 >/dev/null
cat summary.json
