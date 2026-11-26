#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
(cd fresh-input && sha256sum --strict --quiet -c SHA256SUMS)
test -s fresh-input/FIT.md
mkdir -p held/{bundle,score,probe,controls}
test ! -e held/bundle/models.tsv
while IFS=$'\t' read -r alias id dir hash; do
    state="$dir/$id.state"
    echo "$hash  $state" | sha256sum --strict -c -
    train-input/models/bundle pack ordered "$state" "held/bundle/$alias.bridge"
    mode=$(train-input/models/bundle unpack "held/bundle/$alias.bridge" "held/bundle/$alias.state")
    [[ $mode == ordered ]]
    cmp "$state" "held/bundle/$alias.state"
    printf '%s\t%s.state\n' "$alias" "$alias" >> held/bundle/models.tsv
    lexical-build/lexical probe fresh-input/ordered/text.bin "held/bundle/$alias.state" "held/probe/$alias.jsonl"
done < fresh-input/arms.tsv
scorer-build/margin-score train-input/models/primary.words fresh-input/ordered/text.bin \
    fresh-input/requests.tsv held/bundle/models.tsv held/score/fresh
for arm in gold swapped; do
    train-input/models/textlinks-sm89 correct train-input/models/primary.words \
        fresh-input/ordered/text.bin "fresh-input/$arm-aid.bin" "held/controls/$arm.jsonl"
done
