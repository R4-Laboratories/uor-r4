#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
root=$(pwd -P)
data=$root/fresh-input
bank=$root/train-input/model/primary.words
(cd "$data" && sha256sum --strict --quiet -c SHA256SUMS)
(cd score-rented-v1 && sha256sum --strict --quiet -c SHA256SUMS)
test -s "$data/FIT.md"
mkdir held-raw
mkdir held-raw/{score,readback,controls}
while IFS=$'\t' read -r id sha; do
    case "$id" in recovery-*) file="recovery/states/$id.state" ;; *) file="fit-out/$id.state" ;; esac
    echo "$sha  $file" | sha256sum --strict -c -
    train-input/model/bundle pack ordered "$file" "held-raw/readback/$id.bridge"
    mode=$(train-input/model/bundle unpack "held-raw/readback/$id.bridge" "held-raw/readback/$id.state")
    [[ $mode == ordered ]]
    cmp "$file" "held-raw/readback/$id.state"
    printf '%s\t%s.state\n' "$id" "$id" >> held-raw/readback/states.tsv
done < "$data/states.sha.tsv"
score-rented-v1/margin-score "$bank" "$data/ordered/text.bin" "$data/requests.tsv" \
    held-raw/readback/states.tsv held-raw/score/fresh
for arm in gold swapped; do
    train-input/model/textlinks-sm89 correct "$bank" "$data/ordered/text.bin" "$data/$arm-aid.bin" \
        "held-raw/controls/$arm.jsonl"
done
