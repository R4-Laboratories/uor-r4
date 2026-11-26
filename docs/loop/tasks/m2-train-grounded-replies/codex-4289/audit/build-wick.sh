#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
r=/mnt/plexpool/uor-r4-corpus/runs
prior="$r/20261126-codex-text-governor-links/parity/source-v2"
root="$r/20261126-codex-far-transport-margin/audit"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?new version}
[[ $version =~ ^v[1-9][0-9]*$ && $(findmnt -no FSTYPE --target "$root") == zfs ]]
d="$root/build-$version"
mkdir "$d"
pkg="$d/staged/codex-4289/audit"
mkdir -p "$pkg" "$d/staged/codex-4288"
ln -s "$prior" "$d/staged/codex-4288/parity"
cp -r "$here/Cargo.toml" "$here/Cargo.lock" "$here/src" "$pkg"
export CARGO_TARGET_DIR="$d/target"
"$HOME/.cargo/bin/cargo" build --release --offline --locked --manifest-path "$pkg/Cargo.toml" >"$d/build.log" 2>&1
(cd "$d" && sha256sum target/release/transport-margin-audit staged/codex-4289/audit/{Cargo.toml,Cargo.lock,src/main.rs} \
    staged/codex-4288/parity/src/packet.rs > SHA256SUMS)
