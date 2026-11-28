#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(realpath "${1:?completed private custody root}")
out=$(realpath -m "${2:?fresh packet root}")
mkdir "$out"
(
    cd "$stage"
    test ! -e MEMBERS.sha256
    test -z "$(find . ! -type f ! -type d -print -quit)"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$out/MEMBERS.sha256"
    test -s "$out/MEMBERS.sha256"
    install -m 600 "$out/MEMBERS.sha256" .
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --zstd -cf "$out/natural-reciprocal.tar.zst" .
)
(cd "$out" && sha256sum natural-reciprocal.tar.zst MEMBERS.sha256 > PACKET.sha256)
