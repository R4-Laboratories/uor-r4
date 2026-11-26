#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-paired-occurrence-margin/panel"
predecessor="$runs/20261126-codex-all-outranker-interval/panel/build-v3"
method=docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel
export_method=docs/loop/tasks/m2-train-grounded-replies/codex-4288/export
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive generator build version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
build="$root/generator-build-v$version"
mkdir "$build"
printf '%s  %s\n' 5afa42a4215f32a29a2dfbbd4d8d15bb5f4c314ee877bd7050868c6ff41ff7bf \
    "$predecessor/SHA256SUMS" | sha256sum --strict --quiet -c -
(cd "$predecessor" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
source="$build/source/$method"
export_source="$build/source/$export_method"
mkdir -p "$source" "$export_source"
cp -R "$predecessor/source/$method/"{Cargo.toml,Cargo.lock,src} "$source/"
cp -R "$predecessor/source/$export_method/"{Cargo.toml,Cargo.lock,src} "$export_source/"
ln -s "$predecessor/source/crates" "$build/source/crates"
(cd "$source" && patch --fuzz=0 -p0 < "$here/generator.patch") > "$build/patch.log" 2>&1
(cd "$export_source" && patch --fuzz=0 -p0 < "$here/contrast-count.patch") > "$build/compare-patch.log" 2>&1
export CARGO_TARGET_DIR="$build/target"
"$HOME/.cargo/bin/cargo" build --locked --offline --release --manifest-path "$source/Cargo.toml" > "$build/build.log" 2>&1
"$HOME/.cargo/bin/cargo" build --locked --offline --release --manifest-path "$export_source/Cargo.toml" > "$build/compare-build.log" 2>&1
(
    cd "$build"
    sha256sum "source/$method/"{Cargo.toml,Cargo.lock,src/main.rs} \
        "source/$export_method/"{Cargo.toml,Cargo.lock,src/main.rs,src/contrast.rs,src/bridge.rs,src/packet.rs} \
        target/release/{governor-links-panel,source-link-text} > SHA256SUMS
    sha256sum "$predecessor/"{SHA256SUMS,authority.sha256} \
        "$here/"{build-generator-wick.sh,generator.patch,contrast-count.patch} > authority.sha256
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
    sha256sum SHA256SUMS authority.sha256
)
