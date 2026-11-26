#!/usr/bin/env bash
# Copy this unit to an unused Wick methods-vN before running this script there.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common-wick.sh"
v=${1:?unused grade-fixture version}; [[ $v =~ ^v[1-9][0-9]*$ ]]
pin 1480c9f725aef1a6f8de7e27fee4190f0dc582bc90dcffcecfc06be6d819c987 "$prev/SHA256SUMS"
verify "$prev"
for f in join pairs summary witness; do
 hash=$(awk -v path="./grade/$f.jq" '$2==path{print $1}' "$prev/SHA256SUMS"); pin "$hash" "$prev/grade/$f.jq"
done
bash "$here/grade/fixture-wick.sh" "$s/grade-fixture-$v"
sha256sum "$prev/SHA256SUMS" "$prev/grade/"{join,pairs,summary,witness}.jq > "$here/previous.sha256"
seal "$here"; sha256sum "$here/SHA256SUMS" "$s/grade-fixture-$v/SHA256SUMS"
