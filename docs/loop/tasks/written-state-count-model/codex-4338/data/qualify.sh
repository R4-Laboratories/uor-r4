#!/usr/bin/env bash
set -euo pipefail
umask 077

unit=$(realpath "${1:?private #4338 TRAIN source and decisions root}")
parent=$(realpath "${2:?private #4337 run root}")
out=$(realpath -m "${3:?fresh qualification directory}")
here=$(cd "$(dirname "$0")" && pwd)
prior=$parent/next-supply-pilot-v1
fit=$parent/data/fit-v1
distinct=$unit/positive-distinct/decisions.jsonl

check_digest() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --quiet --strict -c -
}

check_digest 74e5fc4695bb670cc415acab7ebb315ff334ca1c6a8a091136b94982de493af4 "$unit/positive-input/SEALED.sha256"
check_digest 872d1f72879bb9b6f3c4924f4cab919e496aeddaf1f0782c8e05f4e8233ca63a "$unit/answer-input/SEALED.sha256"
check_digest 948353accf1ec21be9e709719f502570cfd6d8afdb612bcb57c789bc24cb4c3b "$prior/MANIFEST.sha256"
check_digest 3591ebc559da8a889389b80af8905b9486a186b9070a5ae035a2c0934577ba3e "$fit/SEALED.sha256"
check_digest 91f1a4c151e38e7b98eace8edd98d8d5d34ba2811e1bf9843b0fd3035cc35cb7 "$prior/picked-pairs.jsonl"
check_digest 48db8a1f22c2b983b179f3d968c6adcfa1c6e6b023d79a81003f8eab50ddadb4 "$distinct"
for input in positive-input answer-input; do
    (cd "$unit/$input" && sha256sum --quiet --strict -c SEALED.sha256)
done
(cd "$prior" && sha256sum --quiet --strict -c MANIFEST.sha256 SPLIT.sha256 PICKS.sha256)
(cd "$fit" && sha256sum --quiet --strict -c SEALED.sha256)

qualified=$(jq -nc \
    --slurpfile positive "$unit/positive-input/rows.jsonl" \
    --slurpfile picks "$unit/positive-input/picks.jsonl" \
    --slurpfile even "$unit/positive-even/decisions.jsonl" \
    --slurpfile odd "$unit/positive-odd/decisions.jsonl" \
    --slurpfile distinct "$distinct" \
    --slurpfile answer "$unit/answer-input/rows.jsonl" \
    --slurpfile a "$unit/answer-a/decisions.jsonl" \
    --slurpfile b "$unit/answer-b/decisions.jsonl" \
    --slurpfile c "$unit/answer-c/decisions.jsonl" \
    --rawfile split "$prior/split.tsv" \
    --rawfile pilot "$prior/pilot-adjudication.tsv" \
    --slurpfile pilot_picks "$prior/picked-pairs.jsonl" \
    --slurpfile old_exclusions "$fit/exclude.json" \
    --slurpfile old_fit "$fit/train-truth.jsonl" \
    -f "$here/qualify.jq")

mkdir "$out"
for part in input gold truth; do
    jq -c --arg part "$part" '.rows[] | .[$part]' <<< "$qualified" > "$out/train-$part.jsonl"
done
jq -c '.excluded[]' <<< "$qualified" > "$out/excluded.jsonl"
jq '.evidence' <<< "$qualified" > "$out/evidence.json"

# Use stable logical paths so a cold restore can reproduce the manifest bytes.
bind() {
    local digest
    digest=$(sha256sum "$1" | cut -d' ' -f1)
    printf '%s  %s\n' "$digest" "$2"
}
{
    for path in positive-input/SEALED.sha256 answer-input/SEALED.sha256 \
        positive-even/decisions.jsonl positive-odd/decisions.jsonl \
        answer-a/decisions.jsonl answer-b/decisions.jsonl answer-c/decisions.jsonl \
        positive-distinct/decisions.jsonl; do
        bind "$unit/$path" "unit/$path"
    done
    for path in next-supply-pilot-v1/MANIFEST.sha256 next-supply-pilot-v1/picked-pairs.jsonl \
        data/fit-v1/SEALED.sha256; do
        bind "$parent/$path" "parent/$path"
    done
    bind "$here/qualify.sh" tools/qualify.sh
    bind "$here/qualify.jq" tools/qualify.jq
} > "$out/INPUTS.sha256"
(cd "$out" && sha256sum train-input.jsonl train-gold.jsonl train-truth.jsonl excluded.jsonl evidence.json INPUTS.sha256 > SEALED.sha256)
