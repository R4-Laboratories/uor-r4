#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 2 ]; then
    echo 'usage: stage-sources.sh BUILD_METADATA NEW_SOURCE_STAGE' >&2
    exit 2
fi

build=$(realpath "$1")
stage=$(realpath -m "$2")
unit=docs/loop/tasks/written-state-count-model/codex-4336
repo=https://github.com/UOR-Foundation/uor-r4.git
revision() {
    jq -er ".[\"$1\"] | select(test(\"^[0-9a-f]{40}$\"))" "$build"
}
base=$(revision base_revision)
runtime=$(revision runtime_revision)
panel=$(revision panel_revision)
replay=$(revision replay_revision)
transfer=$(revision transfer_revision)

mkdir "$stage"
git init --quiet --bare "$stage/source.git"
git -C "$stage/source.git" fetch --quiet --no-tags --depth 1 "$repo" "$base" "$runtime" "$panel" "$replay" "$transfer"
mkdir "$stage/bundle"

runtime_files=(
    crates/uor-r4-integer
    crates/uor-r4-native-vector
    crates/uor-r4-tokenizer
    crates/uor-r4-vision
    crates/uor-r4-discrete/src
    crates/uor-r4-discrete/count-runtime
    third_party/arrayref
)
write_delta() {
    local name=$1 rev=$2
    shift 2
    git -C "$stage/source.git" diff --binary "$base" "$rev" -- "$@" > "$stage/bundle/$name.patch"
    mkdir "$stage/$name"
    git -C "$stage/source.git" archive "$rev" "$@" | tar -xf - -C "$stage/$name"
    (
        cd "$stage/$name"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$stage/bundle/$name.sha256"
    )
}
write_delta runtime "$runtime" "${runtime_files[@]}" "$unit/runtime"
write_delta panel "$panel" "$unit/panel"
write_delta replay "$replay" "$unit/replay"
write_delta transfer "$transfer" "$unit/transfer"
