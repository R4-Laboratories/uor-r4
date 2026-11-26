#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-third-margin-retain/panel"
predecessor="$runs/20261126-codex-paired-occurrence-margin/panel/generator-build-v3"
method=docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive build version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
build="$root/generator-build-v$version"
mkdir "$build"
(cd "$predecessor" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
source="$build/source/$method"
mkdir -p "$source"
cp -R "$predecessor/source/$method/"{Cargo.toml,Cargo.lock,src} "$source/"
ln -s "$predecessor/source/crates" "$build/source/crates"
(cd "$source" && patch --fuzz=0 -p0 < "$here/generator.patch") > "$build/patch.log" 2>&1
export CARGO_TARGET_DIR="$build/target"
"$HOME/.cargo/bin/cargo" build --locked --offline --release --manifest-path "$source/Cargo.toml" > "$build/build.log" 2>&1
(
    cd "$build"
    sha256sum "source/$method/"{Cargo.toml,Cargo.lock,src/main.rs} target/release/governor-links-panel > SHA256SUMS
    sha256sum "$predecessor/SHA256SUMS" "$predecessor/authority.sha256" \
        "$here/build-wick.sh" "$here/generator.patch" > "$build/authority.sha256"
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
    sha256sum SHA256SUMS authority.sha256
)
