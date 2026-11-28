#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "${1:?source checkout}")
target=$(realpath -m "${2:?private target directory}")
here=$(cd "$(dirname "$0")" && pwd)
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="$target"
# The isolated checkout contains the production dependency crates and the
# registered workspace lock. Its build leaves the parent workspace untouched.
cd "$source_root"
cmp "$here/workspace.toml" Cargo.toml
cmp "$here/Cargo.lock" Cargo.lock
cargo build --offline --locked --release --features p1-count-serving,release-factorial-diagnostic \
    -p uor-r4-integer --bin uor-chat
