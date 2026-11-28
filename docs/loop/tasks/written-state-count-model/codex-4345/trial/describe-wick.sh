#!/usr/bin/env bash
# Derive descriptive output counts after the raw captures and grading seal.
set -euo pipefail
umask 077
[[ $(hostname -s) == wick ]]
self=$(realpath "$0")
tools=$(dirname "$self")
run=$(realpath "${1:?sealed serving directory}")
dest=${2:?fresh descriptive directory}
(
    cd "$run"
    sha256sum --strict --quiet -c RESULTS.sha256
    while read -r _ manifest; do
        (cd "$(dirname "$manifest")"; sha256sum --strict --quiet -c "$(basename "$manifest")")
    done <RESPONSES.sha256
)
mkdir -m 700 "$dest"
dest=$(realpath "$dest")
cd "$dest"
for arm in base off zero geometry.1 geometry.2 ordered.1 ordered.2; do
    responses=()
    for ((i = 0; i < 90; i++)); do responses+=("$run/responses/$arm/open/$i.json"); done
    jq -sc -L "$tools" --slurpfile grades "$run/grades/$arm.writers.jsonl" --arg arm "$arm" \
        'include "describe"; describe($grades) + {arm:$arm}' "${responses[@]}" >> descriptions.jsonl
    if [[ $arm == base ]]; then
        jq -sc -L "$tools" --slurpfile grades "$run/grades/base.writers.jsonl" \
            'include "describe"; eos_ceiling($grades)' "${responses[@]}" >EOS-ONLY-CEILING.json
    fi
done
jq -s --slurpfile ceiling EOS-ONLY-CEILING.json '{schema:"uor-r4.ordinary-termination-descriptive/1",
    arms:., retrospective_exact_reference_eos_only_ceiling:$ceiling[0]}' descriptions.jsonl >DESCRIPTIVE.json
sha256sum "$run/RESULTS.sha256" "$self" "$tools/describe.jq" >TOOLS.sha256
sha256sum DESCRIPTIVE.json descriptions.jsonl EOS-ONLY-CEILING.json TOOLS.sha256 >RESULTS.sha256
sha256sum DESCRIPTIVE.json RESULTS.sha256
