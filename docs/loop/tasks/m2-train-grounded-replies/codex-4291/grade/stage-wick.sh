#!/usr/bin/env bash
set -euo pipefail
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin
old="$s/methods-v1/grade/rows.jq"
test "$(hostname -s)" = wick
printf '%s  %s\n' 89f63beb6c83c3ce7125f9fd67fcfa736e9ee3891ff006ef7490c1585349b754 "$old" | sha256sum --quiet --strict -c -
here=$(cd "$(dirname "$0")" && pwd)
target=$here/join.jq
test ! -e "$target"
test "$(grep -Fc '($reference[0]) as $refs |' "$old")" = 1
sed '/^(\$reference\[0\]) as \$refs |/,$d' "$old" > "$target"
cat "$here/arm.jq" >> "$target"
