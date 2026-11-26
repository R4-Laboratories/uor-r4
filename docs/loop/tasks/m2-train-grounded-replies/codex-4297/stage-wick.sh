#!/usr/bin/env bash
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
from=$(cd "$(dirname "$0")" && pwd); v=${1:?new methods version}
[[ $(hostname -s) == wick && $from == "$s"/incoming-core-v* && $v =~ ^v[1-9][0-9]*$ ]]
dest="$s/methods-$v"; mkdir "$dest" "$dest/grade" "$dest/client"
cp "$from/"{ORDER.md,PRE-RUN.md,bundle.rs,stage-wick.sh,common-wick.sh,core-fixture-wick.sh,measure-wick.sh,grade-batch-wick.sh,held-wick.sh,grade-held-wick.sh,grade-outputs-wick.sh,client-wick.sh} "$dest/"
cp "$from/grade/"*.jq "$dest/grade/"
cp "$from/client/"{*.sh,*.jq,*.patch} "$dest/client/"
source "$dest/common-wick.sh"; ancestry
# Preserve the predecessor's original-index matching. Only the enrolled panel size varies.
sed -e 's/==256/==$row_count/g' -e 's/range(256)/range($row_count)/g' "$prev/grade/join.jq" > "$dest/original-join.jq"
printf '%s\n' '    + (if $split=="contrast" then {selector:$p.contrast.selector} else {} end)' >> "$dest/original-join.jq"
sed -e 's/==256/==$row_count/g' -e 's/==128/==($row_count\/2)/g' "$prev/grade/pairs.jq" > "$dest/original-pairs.jq"
sed 's/==128/==($row_count\/2)/g' "$prev/grade/summary.jq" > "$dest/original-summary.jq"
sha256sum "$prev/SHA256SUMS" "$prev/grade/"{join,pairs,summary}.jq > "$dest/inherited.sha256"
seal "$dest"
