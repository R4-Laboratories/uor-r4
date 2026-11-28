#!/usr/bin/env bash
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")" && pwd)
deps=${1:?Wick offline rust dependency directory}
dest=${2:?fresh output}
mkdir -m 700 "$dest"
args=(--edition 2021 -O -D warnings -L "dependency=$deps"
  --extern "uor_r4_tokenizer=$deps/libuor_r4_tokenizer-b13ae4dfc5a6c27b.rlib"
  --extern "sha2=$deps/libsha2-3c93fbf2fec95625.rlib"
  --extern "serde=$deps/libserde-3fa7539275d89649.rlib"
  --extern "serde_json=$deps/libserde_json-81c8b96b6d1e5325.rlib")
rustc=${RUSTC:-$HOME/.cargo/bin/rustc}
"$rustc" "${args[@]}" --test "$here/trial.rs" -o "$dest/test"
"$dest/test"
"$rustc" "${args[@]}" "$here/trial.rs" -o "$dest/trial"
