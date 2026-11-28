#!/usr/bin/env bash
set -euo pipefail
umask 077
repo=$(realpath "${1:?tokenizer workspace used by the pinned ordinary client}")
tasks=$(realpath "${2:?written-state-count-model tasks directory}")
build=$(realpath -m "${3:?new build directory}")
supply=$tasks/codex-4329/supply
fit=$tasks/codex-4337/fit
data=$tasks/codex-4338/data
mkdir "$build"
mkdir -p "$build/opening/src" "$build/fit/src"
sed "s|../../../../../../crates/uor-r4-tokenizer|$repo/crates/uor-r4-tokenizer|" \
    "$supply/Cargo.toml" > "$build/opening/Cargo.toml"
cp "$supply/Cargo.lock" "$build/opening/Cargo.lock"
cp "$data/native-opening.rs" "$build/opening/src/main.rs"
cp "$fit/Cargo.toml" "$fit/Cargo.lock" "$build/fit/"
cp "$fit/src/"*.rs "$build/fit/src/"
(
    cd "$build"
    sha256sum opening/Cargo.* opening/src/*.rs fit/Cargo.* fit/src/*.rs > SOURCES.sha256
    cargo build --locked --offline --release --manifest-path opening/Cargo.toml
    cargo build --locked --offline --release --manifest-path fit/Cargo.toml
    sha256sum opening/target/release/whole-turn-supervision fit/target/release/interval-fit \
        > EXECUTABLES.sha256
)
