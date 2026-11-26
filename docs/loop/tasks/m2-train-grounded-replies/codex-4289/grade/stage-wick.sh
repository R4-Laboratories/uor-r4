#!/usr/bin/env bash
# Reuse the audited source identity join from #4288, changing two control names and authenticating the neutral spelling by both address domains.
set -euo pipefail
test "$(hostname -s)" = wick
runs=/mnt/plexpool/uor-r4-corpus/runs
old="$runs/20261126-codex-text-governor-links/methods-v1/grade/rows.jq"
printf '%s  %s\n' eba189d84212d7c085650d863aa33078af19796f8a33ac46b8568e951603db85 "$old" | sha256sum --strict --quiet -c -
output=${1:?new join path}
test ! -e "$output"
sed -e 's/corrupt_noun/baseline/g; s/flipped_sign/positive/g' \
    -e 's/\$corrupt/\$base/g; s/\$flipped/\$pos/g' \
    -e 's/\.tokens==\$p.neutral_rival.tokens/.tokens==$p.neutral_rival.tokens and .bytes==$p.neutral_rival_bytes/' \
    "$old" > "$output"
