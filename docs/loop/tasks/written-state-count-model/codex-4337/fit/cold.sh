#!/usr/bin/env bash
# Run on Wick only after the TRAIN repeats and four device states are frozen.
set -euo pipefail
umask 077

if [[ $# != 12 || ! "${11}" =~ ^[a-f0-9]{64}$ ]]; then
    echo 'usage: cold.sh CLIENT PACK FIT BASE TOKENIZER FULL_TRAIN_PANEL PROTECTED_PANEL HELD_PANEL PREPARATION GPU_RESULTS FIT_SHA256 FRESH_OUTPUT' >&2
    exit 2
fi

client=$(realpath "$1")
pack=$(realpath "$2")
fit=$(realpath "$3")
base=$(realpath "$4")
tokenizer=$(realpath "$5")
train=$(realpath "$6")
protected=$(realpath "$7")
held=$(realpath "$8")
prep=$(realpath "$9")
gpu=$(realpath "${10}")
fit_sha=${11}
out=$(realpath -m "${12}")

(cd "$gpu" && sha256sum --strict -c FROZEN-WEIGHTS.sha256)
(cd "$gpu" && sha256sum --strict -c OUTPUTS.sha256)
(cd "$prep" && sha256sum --strict -c OUTPUTS.sha256)
(cd "$train" && sha256sum --strict -c SEALED.sha256)
(cd "$protected" && sha256sum --strict -c SEALED.sha256)
(cd "$held" && sha256sum --strict -c SEALED.sha256)
mkdir "$out"
cd "$out"

# One fresh read of the complete artifact per arm. Never share an artifact
# across arms, but all three split tapes use the same reopened model.
declare -A blind=(
    [train]="$train/train-input.jsonl"
    [protected]="$protected/protected-input.jsonl"
    [held]="$held/held-input.jsonl"
)
declare -A gold=(
    [train]="$train/train-gold.jsonl"
    [protected]="$protected/protected-gold.jsonl"
    [held]="$held/held-gold.jsonl"
)
cat "${blind[train]}" "${blind[protected]}" "${blind[held]}" > blind.jsonl
jq -es 'map(.case_id) | length == (unique|length)' blind.jsonl > blind-unique.json
sha256sum "$client" "$pack" "$fit" "$base" "$tokenizer" "$gpu/FROZEN-WEIGHTS.sha256" \
    "$gpu/OUTPUTS.sha256" "$prep/OUTPUTS.sha256" > source.sha256

for control in actual zero permuted; do
    for arm in geometry count; do
        name="$arm.$control"
        words="$gpu/$name.words"
        if [[ "$control" == zero ]]; then
            words="$prep/device/zero.words"
        fi
        artifact="$out/$name.d11"
        "$pack" "$arm" "$base" "$words" "$artifact" "$fit_sha" > "$name.pack.json"
        "$client" --discrete "$artifact" --tokenizer "$tokenizer" --interval-features blind.jsonl \
            > "$name.cold.jsonl" 2> "$name.cold.stderr"
        for split in train protected held; do
            prefix="$name.$split"
            jq -c --slurpfile blind "${blind[$split]}" '
                INDEX($blind[]; .case_id) as $ids | select($ids[.case_id] != null)
            ' "$name.cold.jsonl" > "$prefix.cold.jsonl"
            "$fit" verify-device "$prep/features/$split.jsonl" "$prefix.cold.jsonl" "$arm" \
                "$words" "$gpu/$prefix.scores" > "$prefix.parity.json"
            "$fit" replay-device "$prep/features/$split.jsonl" "${gold[$split]}" "$arm" \
                "$words" > "$prefix.selection.json"
        done
    done
done
jq -e -s 'map(.artifact_bytes) | unique | length == 1' ./*.pack.json > equal-artifact-bytes.json
for arm in geometry count; do
    for control in actual permuted zero; do
        for split in train protected held; do
            prefix="$arm.$control.$split"
            jq -c --arg arm "$arm" --arg control "$control" --arg split "$split" \
                --slurpfile parity "$prefix.parity.json" --slurpfile pack "$arm.$control.pack.json" '
                {arm:$arm, control:$control, split:$split, cases:(.cases|length),
                 exact, positive, positive_selected, no_read, false_reads,
                 paired, paired_changes, both_matched_gold_selected:
                    ([.cases[] | select(.both_matched_gold_selected==true)]|length),
                 strictly_positive,minimum_margin, exact_scores:$parity[0].exact_scores,
                 artifact_bytes:$pack[0].artifact_bytes, artifact_sha256:$pack[0].artifact_sha256}
            ' "$prefix.selection.json"
        done
    done
done > summary.jsonl
sha256sum ./*.pack.json ./*.parity.json ./*.selection.json ./*.cold.jsonl \
    blind-unique.json equal-artifact-bytes.json summary.jsonl source.sha256 > OUTPUTS.sha256
