#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
root=$(pwd -P)
data=$root/train-input
seed=$data/models/slot3.state
bank=$data/models/primary.words
score=$root/scorer-build/margin-score
lex=$root/lexical-build/lexical
(cd release && sha256sum --strict --quiet -c SHA256SUMS)
(cd train-input && sha256sum --strict --quiet -c SHA256SUMS)
(cd scorer-build && sha256sum --strict --quiet -c SHA256SUMS)
(cd lexical-build && sha256sum --strict --quiet -c SHA256SUMS)
test -s release/FIT.md
test ! -e fresh-input
mkdir prefit
printf 'base\t%s\n' "$seed" > prefit/states.tsv
for part in regression near previous; do
    "$score" "$bank" "$data/$part/ordered/text.bin" - prefit/states.tsv "prefit/$part"
    cmp "prefit/$part/base.native.jsonl" "$data/$part/reference.native.jsonl"
done
date -u +%FT%TZ > prefit/verified.utc
bash m-code/sources/run.sh fit m-build "$bank" "$data/train/ordered/text.bin" \
    "$data/train/requests.tsv" "$seed" fit-out
"$lex" learn "$data/train/ordered/text.bin" "$data/train/retain.tsv" "$seed" e-out
mkdir matrix train-probes
printf 'id\tm\te\tmid\teid\tsource\tsha\n' > matrix/factors.tsv
ordinal=0
while IFS=$'\t' read -r id file; do
    if (( ordinal == 0 )); then
        [[ $id == start ]]
        cmp "$seed" "fit-out/$file"
        mid=b
    else
        mid=m$ordinal
        if cmp -s "$seed" "fit-out/$file"; then echo "Duplicate untreated M seed" >&2; exit 1; fi
    fi
    for e in 0 1 2 3 4; do
        if (( e == 0 )); then eid=b; else eid=e$e; fi
        if (( ordinal == 0 )); then name=$eid; elif (( e == 0 )); then name=$mid; else name=$mid-$eid; fi
        if (( e == 0 )); then
            cp "fit-out/$file" "matrix/$name.state"
        elif (( ordinal == 0 )); then
            cp "e-out/$eid.state" "matrix/$name.state"
        else
            "$lex" cross "$seed" "fit-out/$file" "e-out/$eid.state" "matrix/$name.state"
        fi
        printf '%s\t%s.state\n' "$name" "$name" >> matrix/models.tsv
        hash=$(sha256sum "matrix/$name.state" | cut -d' ' -f1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$ordinal" "$e" "$mid" "$eid" "$id" "$hash" >> matrix/factors.tsv
    done
    ordinal=$((ordinal+1))
done < fit-out/candidates.tsv
cp e-out/wrong.state matrix/wrong.state
for id in b e1 e2 e3 e4 wrong; do
    "$lex" probe "$data/train/ordered/text.bin" "matrix/$id.state" "train-probes/$id.jsonl"
done
bash scripts/score.sh matrix/models.tsv score
while IFS=$'\t' read -r id m e mid eid src hash; do
    if [[ $e == 0 && -f fit-out/$src.native.jsonl ]]; then cmp "fit-out/$src.native.jsonl" "score/train/$id.native.jsonl"; fi
done < <(tail -n +2 matrix/factors.tsv)
cat fit-out/complete.json
