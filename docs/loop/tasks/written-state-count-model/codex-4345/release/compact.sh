#!/usr/bin/env bash
# Make a minimal portable weight packet and re-open every reconstructed artifact.
set -euo pipefail
umask 077
fit=$(realpath "${1:?sealed fit directory}")
bin=$(realpath "${2:?exact packer}")
base=$(realpath "${3:?frozen base}")
out=$(realpath -m "${4:?fresh private bundle directory}")
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
base_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
test "$(sha "$base")" = "$base_sha"
test "$(sha "$bin")" = b042a06efa9b6fe5820240322e514733c39c9a7ae7cfad14dcd60950d6116814
test "$(sha "$fit/FIT-OUTPUTS.sha256")" = a1e0e65482c642c03f30b83d7d23bad0b93ca82d5d5232342b779886f578879f
(cd "$fit"; sha256sum --strict --quiet -c FIT-OUTPUTS.sha256)
mkdir -m 700 "$out"
mkdir "$out/payload" "$out/cold"
for arm in geometry ordered; do
    for seed in 1 2; do
        run=$arm.$seed
        cp -- "$fit/$run.words" "$fit/$run.fit" "$fit/$run.pack.json" "$out/payload/"
    done
done
(cd "$out/payload"; sha256sum *.words *.fit *.pack.json > CONTENTS.sha256)
(cd "$out/payload"; tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - .) | zstd -q -10 -o "$out/weights.tar.zst"
tar -xf "$out/weights.tar.zst" -C "$out/cold"
cd "$out/cold"
sha256sum --strict --quiet -c CONTENTS.sha256
for arm in geometry ordered; do
    for seed in 1 2; do
        run=$arm.$seed
        "$bin" ordinary-termination "$base" "$base_sha" "$arm" "$run.words" "$run.d11" "$(sha "$run.fit")" > "$run.reopened.json"
        cmp "$run.pack.json" "$run.reopened.json"
        sha256sum "$run.d11" >> REBUILT.sha256
    done
done
(cd "$out"; sha256sum weights.tar.zst cold/CONTENTS.sha256 cold/REBUILT.sha256 > RESTORE.sha256; wc -c weights.tar.zst; sha256sum RESTORE.sha256)
