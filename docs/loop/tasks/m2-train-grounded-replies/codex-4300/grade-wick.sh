#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
m=$(realpath -e "${1:?sealed capture}"); panel=$(realpath -e "${2:?sealed panel}"); v=${3:?new grade version}; version "$v"
[[ $m == "$study"/capture-v* && $panel == "$study"/panel/panel-v* ]]
check_dir "$here"; check_dir "$m"; check_dir "$panel"; sha256sum --strict --quiet -c "$m/inputs.sha256"
grep -Fxq "$(sha256sum "$panel/SHA256SUMS")" "$m/inputs.sha256"
out="$study/grade-$v"
bash "$here/grade/reduce.sh" "$m" "$panel" "$out"
sha256sum "$m/SHA256SUMS" "$panel/SHA256SUMS" "$here/SHA256SUMS" > "$out/inputs.sha256"
seal "$out"; sha256sum "$out/SHA256SUMS"
