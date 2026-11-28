#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(realpath "${1:?original Wick run}")
stage=$(realpath "${2:?private stage with committed source already extracted}")
out=$(realpath -m "${3:?fresh packet directory}")
test -d "$stage/source/docs/loop/tasks/written-state-count-model/codex-4340"
test ! -e "$stage/MEMBERS.sha256"
mkdir "$out" "$stage/validation-selection-v1"
extract=$run/validation-selection-v1
cp "$extract/INPUTS.sha256" "$extract/EXTRACTION.sha256" "$stage/validation-selection-v1/"
cp -R "$extract/selection" "$extract/selected-semantic" "$stage/validation-selection-v1/"
cp -R "$run/semantic-a" "$run/semantic-b" "$run/qualification-v1" "$run/fit-input-v1" "$stage/"
cp "$run/panel-pins-v1.json" "$stage/"
(
    cd "$stage"
    test -z "$(find . ! -type f ! -type d -print -quit)"
    test -z "$(find . -name '._*' -print -quit)"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$out/MEMBERS.sha256"
    cp "$out/MEMBERS.sha256" .
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --zstd -cf "$out/transfer.tar.zst" .
)
