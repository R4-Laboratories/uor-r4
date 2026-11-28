#!/usr/bin/env bash
# Verify an existing #4342 restore and run the source census in a fresh directory.
set -euo pipefail
umask 077

if [[ $# != 3 ]]; then
    echo 'usage: replay.sh VERIFIED_RESTORE TOKENIZER FRESH_OUTPUT' >&2
    exit 2
fi
root=$(realpath "$1")
tokenizer=$(realpath "$2")
out=$(realpath -m "$3")
unit=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
(
    cd "$root"
    printf '%s  %s\n' 7b04b58e4f36f6d1cf38383fc7b937d01bc3bb06a70a50a222a528df00566b4f MEMBERS.sha256 |
        sha256sum --quiet --strict -c -
    sha256sum --quiet --strict -c MEMBERS.sha256
    cd supply
    printf '%s  %s\n' 77fa0cfec83255ec2d22eb28570e45224784437d39c14dd74cc2c68fa557f105 SUPPLY.sha256 |
        sha256sum --quiet --strict -c -
    sha256sum --quiet --strict -c SUPPLY.sha256
)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$out/target"}
cargo run --quiet --locked --manifest-path "$unit/census/Cargo.toml" -- "$root" "$tokenizer" > "$out/audit.json"
jq 'del(.cases,.groups,.same_fact_inverse[].case_id)' "$out/audit.json" > "$out/aggregate.json"
(
    cd "$out"
    sha256sum audit.json aggregate.json > OUTPUTS.sha256
)
