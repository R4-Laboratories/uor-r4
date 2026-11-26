#!/usr/bin/env bash
set -euo pipefail
manifest=$(realpath -e "$1")
out=$2; mkdir "$out"
for part in train regression near; do
    labels=train-input/$part/requests.tsv
    if [[ $part == near ]]; then labels=-; fi
    scorer-build/margin-score train-input/models/primary.words "train-input/$part/ordered/text.bin" \
        "$labels" "$manifest" "$out/$part"
done
