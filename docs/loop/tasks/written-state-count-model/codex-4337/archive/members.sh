#!/usr/bin/env bash
set -euo pipefail
run=$(realpath "${1:?private experiment root}")
phase=${2:?input or final}
cd "$run"

case "$phase" in
    input|final) ;;
    *) exit 2 ;;
esac

{
    find data/gate-v1 data/gate-v2 data/fit-extra-v1 data/fit-v1 data/held-v1 -type f -print
    find data/held-tool/src -type f -print
    printf '%s\n' data/held-tool/Cargo.toml data/held-tool/Cargo.lock data/no-natural-selection.tsv
    printf '%s\n' data/held-candidates-v1/screen.json data/held-candidates-v2/screen.json
    find fit-gate-v2 fit-gate-v2-unicode fit-gate-v2-final fit-full-coverage-v1 fit-device-v1 \
        -type f ! -path 'fit-full-coverage-v1/train-features.jsonl' \
        ! -path 'fit-device-v1/features/train.jsonl' -print
    printf '%s\n' archive-context-v1/Cargo.toml archive-context-v1/Cargo.lock \
        archive-context-v1/product-base.tar archive-context-v1/product.patch \
        archive-context-v1/PRODUCT-SOURCES.sha256
    find archive-context-v1/provenance -type f -print
    if [[ $phase == final ]]; then
        test -s gpu-v1/measured/OUTPUTS.sha256
        test -s cold-v1/OUTPUTS.sha256
        for split in fresh protected; do
            for arm in geometry count; do
                for control in actual permuted zero; do
                    test -s "ordinary-fitted-v2/$split/$arm.$control/native/REPLAY.sha256"
                    test -s "graded-ordinary-fitted-v2/$split/$arm.$control/native/OUTPUTS.sha256"
                done
                test -s "ordinary-fitted-v2/$split/$arm.actual/stop/REPLAY.sha256"
                test -s "graded-ordinary-fitted-v2/$split/$arm.actual/stop/OUTPUTS.sha256"
            done
        done
        for control in geometry-swap count-swap gold-native gold-stop; do
            test -s "causal-v1/$control/REPLAY.sha256"
            test -s "causal-v1/$control/grades/OUTPUTS.sha256"
        done
        test -s visible-v1/REVIEW.sha256
        test -s fresh-balanced-v1/REPLAY.sha256
        test -s fresh-balanced-v1/grades/OUTPUTS.sha256
        find archive-context-v1/lab-source -type f -print
        find archive-context-v1/collection-bin -type f -print
        find gpu-preflight-v1 -type f ! -name interval-cuda ! -name 'result.tar' -print
        find gpu-v1 -type f ! -name 'result.tar' -print
        find cold-v1 -maxdepth 1 -type f \
            \( -name '*.pack.json' -o -name '*.parity.json' -o -name '*.selection.json' \
                -o -name '*-bytes.json' -o -name 'summary.jsonl' -o -name 'source.sha256' \
                -o -name 'blind-unique.json' -o -name 'OUTPUTS.sha256' \
                -o -name 'geometry.actual.held.cold.jsonl' \) -print
        find ordinary-train-base-v2 ordinary-train-supplied-v2 ordinary-fitted-v1 ordinary-fitted-v2 \
            client-control-smoke-v1 client-delayed-smoke-v1 graded-train-base-v2 \
            grade-train-supplied-native grade-train-supplied-stop graded-ordinary-fitted-v1 \
            graded-ordinary-fitted-v2 transfer-error-v1 causal-v1 fresh-balanced-v1 \
            -type f ! -name '*.d11' -print
        printf '%s\n' visible-v1/{README.md,prepare.sh,finalize.sh,BLIND.sha256,VERDICTS.sha256,REVIEW.sha256} \
            visible-v1/{blind.jsonl,arm-map.jsonl,verdicts.jsonl,visible.json} \
            visible-v1/{first-content.sh,first-content.jsonl} visible-v1/arm-grades/*.jsonl
        printf '%s\n' causal-queue-v2.sh supply-tools/count.sh supply-tools/pairs.jq \
            next-supply-v1/summary.json next-supply-v1/pairs.json \
            next-supply-v1/INPUTS.sha256 next-supply-v1/OUTPUTS.sha256
    fi
} | LC_ALL=C sort -u
