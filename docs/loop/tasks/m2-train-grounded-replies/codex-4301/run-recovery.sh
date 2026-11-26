#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
(cd recovery && sha256sum --strict --quiet -c SHA256SUMS)
jq -e '.selection=="diagnostic" and .admissible_ids==[]' recovery/initial-election.json > /dev/null
test ! -e fresh-input
mkdir recovery-score
test "$(wc -l < recovery/states/candidates.tsv)" == 128
for part in train regression near; do
    request=-
    if [[ $part != near ]]; then request="train-input/$part/requests.tsv"; fi
    score-rented-v1/margin-score train-input/model/primary.words "train-input/$part/ordered/text.bin" \
        "$request" recovery/states/candidates.tsv "recovery-score/$part"
done
