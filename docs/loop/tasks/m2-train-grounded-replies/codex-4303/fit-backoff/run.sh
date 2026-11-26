#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
root=$(pwd -P)
(cd backoff-input && sha256sum --strict --quiet -c SHA256SUMS)
jq -e '.protected_both_gains==0' backoff-input/election.json > /dev/null
test ! -e fresh-input
seed=$root/train-input/models/slot3.state
read -r source digest < backoff-input/source.tsv
echo "$digest  matrix/$source.state" | sha256sum --strict -c -
mkdir projected
/usr/local/cuda/bin/nvcc --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 -I native/lex backoff-input/backoff.cu -o projected/backoff
/usr/local/cuda/bin/cuobjdump --dump-sass projected/backoff > projected/sass
if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|F2I|I2F|DADD|DFMA|DMUL|HFMA|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' projected/sass; then exit 1; fi
projected/backoff "$seed" "matrix/$source.state" projected/deltas
mkdir projected/matrix
head -1 matrix/factors.tsv > projected/matrix/factors.tsv
while IFS=$'\t' read -r source_id path; do
    num=${source_id#projection-}
    m=$((1000+num)); mid=m$m
    sha=$(sha256sum "projected/deltas/$path"|cut -d' ' -f1)
    if awk -F '\t' -v sha="$sha" '$3==0 && $7==sha { f=1 } END{exit !f}' matrix/factors.tsv; then continue; fi
    for e in 0 1 2 3 4; do
        if (( e==0 )); then
            eid=b; id="$mid"; cp "projected/deltas/$path" "projected/matrix/$id.state"
        else
            eid="e$e"; id="$mid-$eid"
            lexical-build/lexical cross "$seed" "projected/deltas/$path" "e-out/$eid.state" "projected/matrix/$id.state"
        fi
        sha=$(sha256sum "projected/matrix/$id.state" | cut -d' ' -f1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$m" "$e" "$mid" "$eid" "$source_id" "$sha" >> projected/matrix/factors.tsv
        printf '%s\t%s.state\n' "$id" "$id" >> projected/matrix/models.tsv
    done
done < projected/deltas/candidates.tsv
if [[ -s projected/matrix/models.tsv ]]; then bash scripts/score.sh projected/matrix/models.tsv projected/score; fi
