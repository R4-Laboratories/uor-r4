#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
prior=$(realpath -e "${1:?prior methods/scorer source closure}")
src=${2:?unused source directory}
(
    cd "$prior"
    sha256sum --strict -c "$here/predecessor.sha256"
)
mkdir "$src"
mkdir -p "$src/codex-4278/fit"
cp "$prior/textlinks-shared.cuh" "$src/"
cp "$prior/codex-4278/fit/device.cu" "$src/codex-4278/fit/"
cp "$here"/{argument-score.cu,build.sh,finite.rs,finite.jq,finite.sh} "$src/"
(
    cd "$src"
    sha256sum argument-score.cu build.sh codex-4278/fit/device.cu finite.jq finite.rs finite.sh textlinks-shared.cuh > SHA256SUMS
    sha256sum --strict -c SHA256SUMS
)
