#!/usr/bin/env bash
# Build against the fixed D11 integration snapshot and separate plain user text from annotation.
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-text-governor-links"
parent="$runs/20261125-codex-governing-source-pairs"
oldpanel="$runs/20261126-codex-governor-noun-links/panel/panel-v1"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$parent/models/fold-8/ordered-all.d11"
history="$parent/reserve/history.json"
method=docs/loop/tasks/m2-train-grounded-replies/codex-4288/export
here=$(cd "$(dirname "$0")" && pwd)
test "$(hostname -s)" = wick
test "$(findmnt -no FSTYPE --target "$study")" = zfs
version=${2:?version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
build="$study/export/build-$version"
bin="$build/target/release/source-link-text"
pin(){ printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
case $1 in
build)
    mkdir "$build"
    mkdir -p "$build/source/$method"
    cp -a "$here/"{Cargo.toml,Cargo.lock,src} "$build/source/$method/"
    ln -s "$parent/artifact-build-v3/source/crates" "$build/source/crates"
    (cd "$parent/artifact-build-v3/source" && sha256sum --strict --quiet -c ../source.sha256)
    export CARGO_TARGET_DIR="$build/target"
    "$HOME/.cargo/bin/cargo" build --offline --locked --release --manifest-path "$build/source/$method/Cargo.toml" >"$build/build.log" 2>&1
    (cd "$build" && sha256sum "source/$method/"{Cargo.toml,Cargo.lock,src/main.rs} target/release/source-link-text build.log > SHA256SUMS)
    ;;
train|held)
    cohort=$1
    if [[ $cohort == train ]]; then
        ref="$oldpanel/fresh/reference.json"
        pin b95b52f26debef4dae7197baa2557b228fe60d583c466c8d3d78353271fe88fb "$ref"
    else
        ref="$study/panel/panel-v1/fresh/reference.json"
        pin fcc8031daf05d66a47fda878cbe3015ead90b1b9990dd67a268f316e716edce3 "$ref"
    fi
    out="$study/export/$cohort-${3:?fresh data version}"
    mkdir "$out"
    (cd "$build" && sha256sum --strict --quiet -c SHA256SUMS)
    jq -ce '[.[] | .turns[1] | select(type=="string")] | select(length==256)' "$ref" > "$out/user-only.json"
    "$bin" pack "$out/user-only.json" "$history" "$model" "$codec" "$out/blind"
    "$bin" supervise "$ref" "$out/blind/coordinates.jsonl" "$out/links.bin" "$out/aid.bin"
    sha256sum "$ref" "$build/SHA256SUMS" "$history" "$model" "$codec" > "$out/inputs.sha256"
    (cd "$out" && sha256sum user-only.json blind/{coordinates.jsonl,text.bin} links.bin aid.bin inputs.sha256 > SHA256SUMS)
    ;;
*) exit 64 ;;
esac
