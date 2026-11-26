#!/usr/bin/env bash
set -euo pipefail
umask 077
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-mixed-transport"
old="$r/20261126-codex-far-transport-margin/audit/method-v2"
packet="$r/20261126-codex-text-governor-links/parity/source-v2/src/packet.rs"
here=$(cd "$(dirname "$0")" && pwd)
v=${1:?new version}; [[ $v =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
d="$s/audit/build-$v"; mkdir -p "$s/audit"; mkdir "$d" "$d/src"
cp "$old/"{Cargo.toml,Cargo.lock} "$d/"
cp "$packet" "$d/src/packet.rs"; cp "$here/replay.rs" "$d/src/main.rs"
(cd "$d" && "$HOME/.cargo/bin/cargo" build --release --offline --locked > build.log 2>&1 &&
 sha256sum target/release/transport-margin-audit Cargo.toml Cargo.lock src/{main,packet}.rs > SHA256SUMS)
