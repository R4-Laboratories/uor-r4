#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 4 ]]; then
    echo "usage: build.sh RELATIVE_4334 SIGNED_4333 NEW_TOOLS solver|check" >&2
    exit 2
fi
unit=$(cd "$(dirname "$0")" && pwd)
relative=$(realpath "$1")
signed=$(realpath "$2")
out=$(realpath -m "$3")
prior_gate="$relative/payload/source/codex-4334/gate"
tool=$4
case "$tool" in
    solver|check) ;;
    *) echo "unknown tool" >&2; exit 2 ;;
esac
(cd "$signed/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
mkdir "$out"
cp -a "$signed/payload/source/solver" "$signed/payload/source/check" "$out"
patch --directory "$out" --strip 1 < "$prior_gate/rank.patch"
cp "$prior_gate/certificate_replay.rs" "$out/check/src/"
patch --directory "$out" --strip 1 < "$unit/edge-controls.patch"
cargo build --offline --locked --release --manifest-path "$out/$tool/Cargo.toml" --target-dir "$out/target"
