#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 5 ]]; then
    echo "usage: solve.sh RELATIVE_4334 SIGNED_4333 REQUEST_4332 FEATURES NEW_RESULTS" >&2
    exit 2
fi
unit=$(cd "$(dirname "$0")" && pwd)
relative=$(realpath "$1")
signed=$(realpath "$2")
request=$(realpath "$3")
features=$(realpath "$4")
out=$(realpath -m "$5")
tools="$relative/tools"
parent_source="$relative/payload/source/codex-4334"
mkdir "$out"
export LD_LIBRARY_PATH="$tools/environment/usr/lib/x86_64-linux-gnu"
export SIGNED_GATE_HIGHS="$tools/environment/usr/bin/highs"
z3="$tools/environment/usr/bin/z3"
solver="$tools/target/release/signed-site-gate"
old_key="$request/panel/truth/key.jsonl"
fresh_key="$signed/panel/truth/key.jsonl"

(cd "$relative/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
(cd "$features" && sha256sum --strict --quiet -c SCAN.sha256)
bash "$unit/build.sh" "$relative" "$signed" "$out/tools" solver
solve_frame() (
    panel=$1
    if [[ "$panel" == edge ]]; then
        solver="$out/tools/target/release/signed-site-gate"
    fi
    dest="$out/$panel"
    mkdir "$dest" "$dest/scratch"
    cd "$dest/scratch"
    export SIGNED_GATE_SCRATCH="$dest/scratch"
    earlier="$features/diagnostic-$panel.jsonl"
    fresh="$features/fresh-$panel.jsonl"
    "$solver" --rank "$z3" "$fresh" "$fresh_key" > "$dest/rank.jsonl"
    "$solver" "$z3" "$earlier" "$old_key" > "$dest/old.jsonl"
    "$solver" "$z3" "$fresh" "$fresh_key" > "$dest/fresh.jsonl"
    "$solver" "$z3" "$earlier" "$old_key" "$fresh" "$fresh_key" > "$dest/joint.jsonl"
    for mask in word passage passage-word supplied; do
        jq -c --arg mask "$mask" --slurpfile truth "$fresh_key" \
            -f "$parent_source/rivals/select.jq" "$fresh" > "$dest/scratch/$mask.jsonl"
        "$solver" --rank "$z3" "$dest/scratch/$mask.jsonl" "$fresh_key" > "$dest/$mask.jsonl"
    done
)
solve_frame edge &
edge_pid=$!
solve_frame clamp &
clamp_pid=$!
status=0
wait "$edge_pid" || status=$?
wait "$clamp_pid" || status=$?
if [[ "$status" != 0 ]]; then exit "$status"; fi
(cd "$out" && sha256sum {edge,clamp}/*.jsonl > SHA256SUMS)
