#!/usr/bin/env bash
# Exhaustive finite operator test. Build only the four constant tables and the oracle.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=${1:?path to discrete_d11.rs}
work=${2:?fresh build directory}
mkdir "$work"
awk '/^const (AXES|COMPOSE|ROOTS|BFS): / {inside=1; seen++} inside {print} /^];$/ {inside=0} END {if (seen!=4 || inside!=0) exit 1}' "$src" > "$work/tables.rs"
cat "$work/tables.rs" "$here/check.rs" > "$work/oracle.rs"
rustc --edition=2021 -C opt-level=1 "$work/oracle.rs" -o "$work/oracle"
"$work/oracle"
