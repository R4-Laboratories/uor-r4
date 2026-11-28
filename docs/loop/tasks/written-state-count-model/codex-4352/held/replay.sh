#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 && $# != 6 ]]; then
    echo 'usage: replay.sh RUNS PREVIEW_DIR NEW_OUTPUT [--coqa-train-as-held FILE SHA256]' >&2
    exit 2
fi
runs=$(realpath "$1")
previews=$(realpath "$2")
out=$(realpath -m "$3")
shift 3
unit=$(cd -- "$(dirname -- "$0")" && pwd)
(
    cd "$previews"
    sha256sum --quiet --strict --check PREVIEWS.sha256
)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$out-target"}
cargo run --quiet --locked --offline --manifest-path "$unit/Cargo.toml" --bin held-coqa-dialogue -- "$runs" "$previews" "$out" "$@"
(
    cd "$out"
    sha256sum blind.jsonl truth.jsonl aggregate.json > MEMBERS.sha256
    sha256sum --quiet --strict --check MEMBERS.sha256
)
