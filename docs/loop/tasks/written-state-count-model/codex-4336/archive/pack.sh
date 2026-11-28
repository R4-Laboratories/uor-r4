#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 8 ]; then
    echo 'usage: pack.sh PANEL CAPTURE AUDIT TRANSFER BUILD_METADATA SOURCE_BUNDLE PROVENANCE NEW_ARCHIVE' >&2
    exit 2
fi

panel=$(realpath "$1")
capture=$(realpath "$2")
audit=$(realpath "$3")
transfer=$(realpath "$4")
build=$(realpath "$5")
source_bundle=$(realpath "$6")
provenance=$(realpath "$7")
archive=$(realpath -m "$8")
mkdir "$archive"
payload="$archive/payload"
mkdir "$payload"
cp -r "$panel" "$payload/panel"
cp "$build" "$payload/build.json"
cp -r "$source_bundle" "$payload/source"
cp -r "$provenance" "$payload/provenance"
mkdir "$payload/capture"
cp "$capture/records.jsonl" "$capture/scores.jsonl" "$capture/summary.json" "$payload/capture/"
(
    cd "$payload/capture"
    sha256sum records.jsonl scores.jsonl summary.json > "$payload/REPLAY.sha256"
)
mkdir "$payload/audit"
cp "$audit/summary.json" "$audit/damaged.stderr" "$audit/SHA256SUMS" "$payload/audit/"
mkdir "$payload/transfer"
cp "$transfer/records.jsonl" "$transfer/summary.json" "$transfer/SHA256SUMS" "$payload/transfer/"
(
    cd "$payload"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$archive/MEMBERS.sha256"
    cp "$archive/MEMBERS.sha256" .
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --zstd \
        -cf "$archive/semantic-span.tar.zst" .
)
