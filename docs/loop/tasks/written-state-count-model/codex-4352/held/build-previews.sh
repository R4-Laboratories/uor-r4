#!/usr/bin/env bash
# Rebuild the historical text previews in fresh private custody without scoring.
set -euo pipefail
umask 077
if [[ $# != 2 ]]; then
    echo 'usage: build-previews.sh RUNS NEW_PREVIEW_DIR' >&2
    exit 2
fi
runs=$(realpath "$1")
out=$(realpath -m "$2")
unit=$(cd -- "$(dirname -- "$0")" && pwd)
base=$(cd "$unit/../.." && pwd)
a=$runs/20261127-codex-visible-answer/panel
b=$runs/20261127-codex-question-swap/panel/frozen-v2
retained=$runs/20261127-codex-question-swap/panel/coqa-preview-v1.jsonl
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$out-target"}
mkdir "$out"
inputs=("$a/inputs/tokenizer.json" "$a/inputs/coqa-dev-v1.0.json" \
    "$a/frozen-v1/judgments.jsonl" "$a/inputs/d87-bank-prompt-digests.txt")
cargo run --quiet --release --locked --offline \
    --manifest-path "$base/codex-4313/panel/Cargo.toml" -- \
    "${inputs[@]}" --preview "$out/4313-controlled-side-output" > "$out/4313-reproduced.jsonl"
if [[ -f "$retained" ]]; then
    cp "$retained" "$out/4313-retained.jsonl"
    if cmp -s "$out/4313-retained.jsonl" "$out/4313-reproduced.jsonl"; then
        echo 'retained and reproduced #4313 previews match' >&2
    else
        echo 'retained and reproduced #4313 previews differ; both enter exclusions' >&2
    fi
fi
cargo run --quiet --release --locked --offline \
    --manifest-path "$base/codex-4314/panel/Cargo.toml" -- \
    "${inputs[@]}" "$a/frozen-v1/prompts.jsonl" "$b/rows.jsonl" "$b/keys.jsonl" \
    --preview "$out/unused-4314" > "$out/4314-reproduced.jsonl"
cargo run --quiet --release --locked --offline \
    --manifest-path "$base/codex-4315/panel/Cargo.toml" -- \
    preview "$runs" > "$out/4315-reproduced.jsonl"
(
    cd "$out"
    sha256sum *.jsonl > PREVIEWS.sha256
    sha256sum --quiet --strict --check PREVIEWS.sha256
)
