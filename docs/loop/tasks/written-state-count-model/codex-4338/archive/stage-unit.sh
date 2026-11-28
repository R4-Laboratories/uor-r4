#!/usr/bin/env bash
set -euo pipefail
umask 077
mode=${1:?fixtures or final}
runs=$(realpath "${2:?original Wick runs root}")
run=$(realpath "${3:?unit Wick root}")
source=$(realpath "${4:?source tree with unit drivers and parent receipts}")
stage=$(realpath "${5:?completed roster custody root}")
unit=docs/loop/tasks/written-state-count-model/codex-4338
parent=docs/loop/tasks/written-state-count-model/codex-4337
parent_run=$runs/20261127-codex-grounded-interval
supply_run=$runs/20261127-codex-whole-turn-census

fixtures() {
    mkdir "$stage/unit" "$stage/fixtures"
    (
        cd "$source"
        cp -r --parents "$unit/ORDER.md" "$unit/data" "$unit/archive" \
          "$parent/evidence/archive.json" "$parent/evidence/companion.json" \
          "$parent/supply/pilot-extract.jq" "$stage/source/"
    )
    mkdir "$stage/fixtures/20261127-codex-grounded-interval" \
        "$stage/fixtures/20261127-codex-whole-turn-census"
    mkdir "$stage/fixtures/20261127-codex-grounded-interval/data"
    cp -r "$parent_run/data/fit-v1" "$stage/fixtures/20261127-codex-grounded-interval/data/"
    for fixture in next-supply-v1 next-supply-pilot-v1; do
        cp -r "$parent_run/$fixture" "$stage/fixtures/20261127-codex-grounded-interval/"
    done
    cp -r "$supply_run/supply-v1" "$stage/fixtures/20261127-codex-whole-turn-census/"
    mkdir "$stage/fixtures/20261127-codex-whole-turn-census/supply-inputs"
    ln "$stage/roster/source-inputs/20261127-codex-whole-turn-census/supply-inputs/squad-train-v2.0.json" \
       "$stage/fixtures/20261127-codex-whole-turn-census/supply-inputs/squad-train-v2.0.json"
    copy_roots positive-input answer-input
}

copy_roots() (
    cd "$run"
    for root in "$@"; do
        test -d "$root"
        test ! -e "$stage/unit/$root"
        test -z "$(find "$root" ! -type f ! -type d -print -quit)"
        cp -r -- "$root" "$stage/unit/"
    done
)

case "$mode" in
    fixtures) fixtures ;;
    final)
        copy_roots positive-even positive-odd positive-distinct answer-a answer-b answer-c qualified-v1 native-v1 native-v2 matched-v1 tools
        mkdir "$stage/runtime"
        install -m 600 "$parent_run/target/release/uor-chat" "$stage/runtime/uor-chat"
        install -m 600 "$parent_run/fit-target/release/interval-fit" "$stage/runtime/prior-interval-fit"
        install -D -m 600 "$run/native-build-v2/EXECUTABLES.sha256" "$stage/runtime/native-build/EXECUTABLES.sha256"
        install -m 600 "$run/native-build-v2/SOURCES.sha256" "$stage/runtime/native-build/SOURCES.sha256"
        install -m 600 "$run/native-build-v2/fit/target/release/interval-fit" "$stage/runtime/native-fit"
        install -m 600 "$run/native-build-v2/opening/target/release/whole-turn-supervision" "$stage/runtime/native-opening"
        mkdir "$stage/unit/native-rejections-v1"
        for evidence in SEALED.sha256 SOURCES.sha256 summary.json; do
            install -m 600 "$run/native-rejections-v1/$evidence" "$stage/unit/native-rejections-v1/$evidence"
        done
        (
            cd "$run/qualification-ledger-verification-v1"
            mkdir "$stage/unit/qualification-ledger-verification-v1"
            install -m 600 RESULTS.sha256 results.jsonl canonical.stdout \
                positive-*.std* answer-*.std* "$stage/unit/qualification-ledger-verification-v1/"
        )
        for member in train.jsonl train.coverage.json; do
            relative=fit-device-v1/features/$member
            install -D -m 600 "$parent_run/$relative" "$stage/fixtures/20261127-codex-grounded-interval/$relative"
        done
        (
            cd "$source"
            cp -r --parents "$unit/data" "$unit/archive" "$parent/fit/Cargo.toml" \
                "$parent/fit/Cargo.lock" "$parent/fit/src" "$stage/source/"
        )
        ;;
    *) printf 'Unknown staging mode %s\n' "$mode" >&2; exit 2 ;;
esac
