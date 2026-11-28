#!/usr/bin/env bash
set -euo pipefail
umask 077
if [ "$#" -ne 2 ]; then
    echo "usage: setup.sh PRIOR_4333_RESTORE NEW_TOOL_DIRECTORY" >&2
    exit 2
fi
unit=$(cd "$(dirname "$0")" && pwd)
prior=$(realpath "$1")
out=$(realpath -m "$2")
mkdir "$out"
cd "$out"
prior_unit="$prior/payload/source"
mkdir source environment
cp -a "$prior_unit/solver" "$prior_unit/check" source/
patch --directory source --strip 1 < "$unit/rank.patch"
cp "$unit/certificate_replay.rs" source/check/src/
for tool in solver check; do
    cargo build --offline --locked --release --manifest-path "source/$tool/Cargo.toml" --target-dir target
done
for archive in "$prior/payload/tools/"*.deb; do
    dpkg-deb --extract "$archive" environment
done
sha256sum --strict -c <<'SUMS'
07892eb6f7874b9842fb1a4de74fb712f15a7c67f005a9abb028fe4ba5041af4  environment/usr/bin/z3
eeb7d0ea297779044bbeff5b5223a3e8d27855ebffb99ea82c12f2de9db6b18d  environment/usr/bin/highs
15fc7b6a6a542c6d24cc7be9c5f4f17539bf1db5b11025cc7bf345e2011cb2d7  environment/usr/lib/x86_64-linux-gnu/libhighs.so.1
SUMS
