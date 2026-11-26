#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
here=$(cd "$(dirname "$0")" && pwd -P)
mode=${1:?finite, fit or validate}; shift
build=$(realpath -e "${1:?sealed build}"); shift
(cd "$here" && sha256sum --quiet --strict -c SHA256SUMS)
(cd "$build" && sha256sum --quiet --strict -c SHA256SUMS)
cmp "$here/SHA256SUMS" "$build/sources.SHA256SUMS"
fitter="$build/paired-fit-sm89"
case "$mode" in
finite)
    [[ $# == 1 ]]
    out=$1
    "$build/fixture" "$out"
    (
        cd "$out"
        sha256sum fixture.words text.bin original.state median.state damaged.state \
            requests.tsv exchanged.tsv broken.tsv wrong-sign.tsv > fixture-inputs.sha256
        "$fitter" fixture.words text.bin requests.tsv original.state median.state first
        "$fitter" fixture.words text.bin requests.tsv original.state median.state repeat
        diff -r -- first repeat
        chosen=$(jq -r 'select(.phase == "fit" and .path == 0 and .accepted) | .candidate' first/schedule.jsonl | head -n 1)
        [[ $chosen == fit-* ]]
        while IFS=$'\t' read -r _ name; do
            "$build/fixture" original.state "first/$name"
        done < first/candidates.tsv
        for name in original "$chosen"; do
            "$build/switch-competition-sm89" fixture.words text.bin "first/$name.state" requests.tsv "oracle-$name"
            cmp "first/$name.native.jsonl" "oracle-$name/native.jsonl"
            jq -en --arg id "$name" --slurpfile terms first/objectives.jsonl \
                --slurpfile oracle "oracle-$name/requests.jsonl" -f "$here/oracle.jq" > "oracle-$name.json"
        done
        "$fitter" --inspect fixture.words text.bin exchanged.tsv "first/$chosen.state" "first/$chosen.state" exchanged
        if "$fitter" fixture.words text.bin broken.tsv original.state median.state rejected-pair > pair.log 2>&1; then
            echo 'Accepted unmatched pair' >&2; exit 1
        fi
        grep -Fq 'repeated row' pair.log
        [[ ! -e rejected-pair ]]
        if "$fitter" fixture.words text.bin wrong-sign.tsv original.state median.state rejected-sign > sign.log 2>&1; then
            echo 'Accepted wrong polarity' >&2; exit 1
        fi
        grep -Fq 'no two eligible fixed-governor completions' sign.log
        if "$fitter" fixture.words text.bin requests.tsv original.state damaged.state rejected-state > state.log 2>&1; then
            echo 'Accepted immutable-state drift' >&2; exit 1
        fi
        grep -Fq 'state changed outside active signed tail words' state.log
        [[ ! -e rejected-state ]]
        jq -en --arg id "$chosen" --slurpfile terms first/objectives.jsonl --slurpfile wrong exchanged/objectives.jsonl \
            --slurpfile schedule first/schedule.jsonl --slurpfile complete first/complete.json -f "$here/finite.jq" > checks.json
        sha256sum --quiet --strict -c fixture-inputs.sha256
    )
    ;;
fit|validate)
    [[ $# == 6 ]]
    if [[ $mode == validate ]]; then set -- --validate "$@"; fi
    exec "$fitter" "$@"
    ;;
*) exit 64 ;;
esac
