#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
root=$(pwd -P)
bank=$root/train-input/model/primary.words
scorer=$root/score-rented-v1/margin-score
data=$root/train-input
(cd release && sha256sum --strict --quiet -c SHA256SUMS)
(cd train-input && sha256sum --strict --quiet -c SHA256SUMS)
(cd score-rented-v1 && sha256sum --strict --quiet -c SHA256SUMS)
test -s release/FIT.md
test ! -e fresh-input
mkdir prefit score
printf 'original\t%s\nmedian\t%s\n' "$data/model/protected.state" "$data/model/ordered.state" > prefit/states.tsv
score_part() {
    local phase=$1 manifest=$2 part=$3 requests=-
    if [[ $part != near ]]; then requests="$data/$part/requests.tsv"; fi
    "$scorer" "$bank" "$data/$part/ordered/text.bin" "$requests" "$manifest" "$phase/$part"
}
for part in train regression near; do
    score_part prefit prefit/states.tsv "$part"
    for id in original median; do
        cmp "prefit/$part/$id.native.jsonl" "release/$part/$id.native.jsonl"
    done
done
date -u +%FT%TZ > prefit/passed.utc
bash fit-code/sources/run.sh fit fit-rented-v2 "$bank" "$data/train/ordered/text.bin" \
    "$data/train/requests.tsv" "$data/model/protected.state" "$data/model/ordered.state" fit-out
for part in train regression near; do score_part score fit-out/candidates.tsv "$part"; done
while IFS=$'\t' read -r id state; do
    cmp "fit-out/$id.native.jsonl" "score/train/$id.native.jsonl"
    [[ $(stat -c %s "fit-out/$state") == 13336 ]]
done < fit-out/candidates.tsv
for part in train regression near; do
    for id in original median; do cmp "prefit/$part/$id.native.jsonl" "score/$part/$id.native.jsonl"; done
done
cat fit-out/complete.json
