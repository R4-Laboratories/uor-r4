#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
here=$(cd "$(dirname "$0")" && pwd -P)
mode=${1:?finite, fit, inspect or validate}; shift
build=$(realpath -e "${1:?sealed build}"); shift
(cd "$here" && sha256sum --quiet --strict -c SHA256SUMS)
(cd "$build" && sha256sum --quiet --strict -c SHA256SUMS)
cmp "$here/SHA256SUMS" "$build/sources.SHA256SUMS"
fitter="$build/third-fit-sm89"
case "$mode" in
finite)
    [[ $# == 1 ]]
    "$build/fixture" "$1"
    (
        cd "$1"
        sha256sum fixture.words text.bin start.state known.state requests.tsv repeated.tsv > fixture-inputs.sha256
        "$fitter" fixture.words text.bin requests.tsv start.state first
        "$fitter" fixture.words text.bin requests.tsv start.state repeat
        "$fitter" --inspect fixture.words text.bin requests.tsv known.state known
        diff -r -- first repeat
        chosen=$(jq -sr '[.[] | select(.phase == "fit" and .path == "balanced" and .accepted)][0].candidate // empty' first/schedule.jsonl)
        [[ $chosen == menu-* || $chosen == fit-* ]]
        while IFS=$'\t' read -r _ name; do
            "$build/fixture" start.state "first/$name"
        done < first/candidates.tsv
        "$build/fixture" start.state known.state

        compare_oracle() {
            local run=$1
            local candidate=$2
            local oracle="oracle-$run-$candidate"
            "$build/switch-competition-sm89" fixture.words text.bin "$run/$candidate.state" requests.tsv "$oracle"
            cmp "$run/$candidate.native.jsonl" "$oracle/native.jsonl"
            jq -en --arg id "$candidate" --slurpfile terms "$run/objectives.jsonl" \
                --slurpfile oracle "$oracle/requests.jsonl" -f "$here/oracle.jq" > "$oracle/agreement.json"
        }
        compare_oracle first start
        compare_oracle known start
        compare_oracle first "$chosen"

        if "$fitter" fixture.words text.bin repeated.tsv start.state rejected-pair > rejected-pair.log 2>&1; then
            echo 'Accepted repeated physical row' >&2; exit 1
        fi
        grep -Fq 'repeated row' rejected-pair.log
        [[ ! -e rejected-pair ]]
        jq -en --arg picked "$chosen" --slurpfile fit first/objectives.jsonl --slurpfile known known/objectives.jsonl \
            --slurpfile oracle_start oracle-first-start/requests.jsonl --slurpfile schedule first/schedule.jsonl \
            --slurpfile complete first/complete.json -f "$here/finite.jq" > checks.json
        sha256sum --quiet --strict -c fixture-inputs.sha256
        find . -type f -print0 | sort -z | xargs -0 sha256sum > ../third-fit-finite.pending
        mv -- ../third-fit-finite.pending SHA256SUMS
    )
    ;;
fit)
    [[ $# == 5 ]]
    source_hash=$(sha256sum -- "$4")
    if [[ ${source_hash%% *} != e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977 ]]; then
        echo 'Protected starting UORSTT01 digest mismatch' >&2; exit 1
    fi
    exec "$fitter" "$@"
    ;;
inspect|validate)
    [[ $# == 5 ]]
    exec "$fitter" "--$mode" "$@"
    ;;
*) exit 64 ;;
esac
