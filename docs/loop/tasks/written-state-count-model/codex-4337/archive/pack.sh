#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(realpath "${1:?private experiment root}")
members=$(realpath "${2:?relative input paths, one per line}")
out=$(realpath -m "${3:?fresh packet root}")
mkdir "$out"
stage="$out/payload"
mkdir "$stage"
while IFS= read -r entry; do
    [[ -n $entry && $entry != /* && $entry != *..* && $entry != MEMBERS.sha256 ]]
    src=$(realpath "$run/$entry")
    [[ $src == "$run/"* && -f $src && ! -L $run/$entry ]]
    [[ ! -e $stage/$entry ]]
    install -D -m 600 "$src" "$stage/$entry"
done < "$members"
(
    cd "$stage"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$out/MEMBERS.sha256"
    test -s "$out/MEMBERS.sha256"
    install -m 600 "$out/MEMBERS.sha256" .
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --zstd -cf "$out/interval.tar.zst" .
)
(cd "$out" && sha256sum interval.tar.zst MEMBERS.sha256 > PACKET.sha256)
