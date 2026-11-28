#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:?gate, fit, or held-tool}
runs=$(realpath "${2:?private Wick runs directory}")
input=$(realpath "${3:?sealed packet or prior #4329 source directory}")
output=$(realpath -m "${4:?fresh output directory}")
here=$(cd "$(dirname "$0")" && pwd)
supply=$runs/20261127-codex-whole-turn-census/supply-v1
authored=$runs/20261127-codex-whole-turn-census/contrasts-v1/train.jsonl
mkdir "$output"

case "$mode" in
    gate)
        previous=$runs/20261127-codex-whole-turn-fit/panel-v1
        jq -nc --slurpfile chosen "$input/selected.json" \
            --slurpfile supply_prompts "$supply/train-prompts.jsonl" \
            --slurpfile supply_truth "$supply/train-truth.jsonl" \
            --slurpfile authored "$authored" \
            --slurpfile protected_prompts "$previous/prompts.jsonl" \
            --slurpfile protected_truth "$previous/gold.jsonl" \
            -f "$here/gate.jq" > "$output/rows.jsonl"
        for split in train protected; do
            for part in input gold truth; do
                jq -c --arg split "$split" --arg part "$part" \
                    'select(.split==$split)|.[$part]' "$output/rows.jsonl" > "$output/$split-$part.jsonl"
            done
        done
        ;;
    fit)
        extra=$runs/20261127-codex-grounded-interval/data/fit-extra-v1
        jq -nc --slurpfile exclude "$input/exclude.json" \
            --slurpfile supply_prompts "$supply/train-prompts.jsonl" \
            --slurpfile supply_truth "$supply/train-truth.jsonl" \
            --slurpfile authored "$authored" \
            --slurpfile extra_input "$extra/held-input.jsonl" \
            --slurpfile extra_gold "$extra/held-gold.jsonl" \
            --slurpfile extra_truth "$extra/held-truth.jsonl" \
            -f "$here/fit.jq" > "$output/rows.jsonl"
        for part in input gold truth; do
            jq -c --arg part "$part" '.[$part]' "$output/rows.jsonl" > "$output/train-$part.jsonl"
        done
        ;;
    held-tool)
        source=$input/docs/loop/tasks/written-state-count-model/codex-4329/supply
        mkdir "$output/src"
        cp "$source/Cargo.lock" "$output/Cargo.lock"
        cp "$source/src/main.rs" "$output/src/main.rs"
        cp "$here/held.rs" "$output/src/held.rs"
        sed "s#../../../../../../crates/uor-r4-tokenizer#$input/crates/uor-r4-tokenizer#" \
            "$source/Cargo.toml" > "$output/Cargo.toml"
        sed -i '1a\mod held;' "$output/src/main.rs"
        sed -i '/^fn main() -> Result<()> {/a\    if std::env::args().nth(1).as_deref() == Some("held") { return held::entry(); }' \
            "$output/src/main.rs"
        (cd "$output" && cargo fmt && cargo build --locked --offline --release)
        ;;
    *)
        printf 'unknown preparation mode %s\n' "$mode" >&2
        exit 2
        ;;
esac
