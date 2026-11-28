#!/usr/bin/env bash
set -euo pipefail
umask 077

runs=${1:?expected preexisting Wick runs directory}
out=${2:?expected new private destination}
driver=$(cd -- "$(dirname -- "$0")" && pwd)
tokenizer=$runs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
donor=$runs/20261127-codex-marker-question/donor-v2
prior=$runs/20261127-codex-visible-answer/donor/pair-01
deps=$runs/20261127-codex-whole-turn-census/target-v1/release/deps
records=$prior/records.jsonl

sha256sum --strict --quiet -c <<SUMS
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
b27d4b14538639095e9ca4828ddb9219aeb521c6fd216eccabe04d1fdff022e6  $records
4ff029e7962b6c58c88014f1baf6adeb4bd5fd1e3d0464cdbbf66d929d10bbd8  $prior/receipt.json
61ba19783fd29b39ee935e1dfa21ae14c5de1caddd3d380c7ed1bedb6b39f6e3  $donor/receipt.json
4e7b8bbc8bdb0582a7a2e6f8a4b68bd0b145ac56152819f0d10effd4043603f6  $donor/source.json
66f847761cbdc2688c8ebbbe1380a154626e65112c807c865990302c3dc0639b  $donor/answer-first.mask.u8
3d55850df31c135698a035a731b8f8fce0110f53d31cc75eeee60d583dece485  $donor/answer-first.tokens.u16
e9b6a5f043e27bd473299e82475b9b62211db6dd712da9537b1eba4d247602ca  $donor/groups.jsonl
d9bc9e45c4fc068ac30841ddf8077a46289bc287714eea7181f710a5d11575ee  $donor/openings.jsonl
SUMS

mkdir -- "$out"
(
  cd -- "$driver"
  sha256sum REGISTRATION.md inventory.rs probe.rs replay-d87.sh
) > "$out/INPUTS.sha256"
rustc="${RUSTC:-$HOME/.cargo/bin/rustc}"
args=(
  --edition 2021 -O -D warnings -L "dependency=$deps"
  --extern "uor_r4_tokenizer=$deps/libuor_r4_tokenizer-b13ae4dfc5a6c27b.rlib"
  --extern "serde=$deps/libserde-3fa7539275d89649.rlib"
  --extern "serde_json=$deps/libserde_json-81c8b96b6d1e5325.rlib"
  "$driver/inventory.rs"
)
"$rustc" "${args[@]}" --test -o "$out/source-checks"
UOR_RELEASE_TEST_TOKENIZER="$tokenizer" "$out/source-checks" --quiet
"$rustc" "${args[@]}" -o "$out/inventory"
"$out/inventory" "$tokenizer" "$donor" "$records" "$out/private-review.jsonl" > "$out/d87.json"
(
  cd -- "$out"
  sha256sum INPUTS.sha256 d87.json private-review.jsonl > SHA256SUMS
  sha256sum --strict -c SHA256SUMS
)
