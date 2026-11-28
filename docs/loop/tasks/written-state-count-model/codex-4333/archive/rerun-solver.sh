#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 3 ]; then
    echo "usage: rerun-solver.sh RESTORE PRIOR_4332_RESTORE NEW_RESULTS" >&2
    exit 2
fi
restore=$(realpath "$1")
prior=$(realpath "$2")
results=$(realpath -m "$3")
mkdir "$results"
cd "$results"
packages="$restore/payload/tools"
mkdir environment scratch
for package in "$packages/"*.deb; do
    dpkg-deb --extract "$package" environment
done
export LD_LIBRARY_PATH="$results/environment/usr/lib/x86_64-linux-gnu"
export SIGNED_GATE_HIGHS="$results/environment/usr/bin/highs"
export SIGNED_GATE_SCRATCH="$results/scratch"
z3="$results/environment/usr/bin/z3"
sha256sum --strict -c <<SUMS
07892eb6f7874b9842fb1a4de74fb712f15a7c67f005a9abb028fe4ba5041af4  $z3
eeb7d0ea297779044bbeff5b5223a3e8d27855ebffb99ea82c12f2de9db6b18d  $SIGNED_GATE_HIGHS
15fc7b6a6a542c6d24cc7be9c5f4f17539bf1db5b11025cc7bf345e2011cb2d7  $LD_LIBRARY_PATH/libhighs.so.1
SUMS
solver="$restore/target/release/signed-site-gate"
verify="$restore/target/release/verify-signed-verdicts"
old_signed="$restore/transport/diagnostic-signed.jsonl"
old_key="$prior/panel/truth/key.jsonl"
fresh_signed="$restore/transport/fresh-signed.jsonl"
fresh_key="$restore/panel/truth/key.jsonl"
omitted=$(jq -er -s 'map(.id) | unique | if length == 1 then .[0] else error("ambiguous omitted row") end' \
    "$restore/payload/solver/fresh-omit-row-v3.solver.jsonl")

"$solver" "$z3" "$old_signed" "$old_key" > old.jsonl
"$solver" "$z3" "$fresh_signed" "$fresh_key" > fresh.jsonl
"$solver" "$z3" "$old_signed" "$old_key" "$fresh_signed" "$fresh_key" > joint.jsonl
"$solver" --panel-positive-only "$z3" "$fresh_signed" "$fresh_key" > fresh-positive-only.jsonl
"$solver" --panel-omit-row "$omitted" "$z3" "$fresh_signed" "$fresh_key" > fresh-omit-row.jsonl

"$verify" "$old_signed" "$old_key" old.jsonl
"$verify" "$fresh_signed" "$fresh_key" fresh.jsonl
"$verify" "$restore/transport/joint-signed.jsonl" "$restore/transport/joint-keys.jsonl" joint.jsonl
"$verify" "$fresh_signed" "$fresh_key" fresh-positive-only.jsonl --positive
"$verify" "$fresh_signed" "$fresh_key" fresh-omit-row.jsonl --omit "$omitted"
for name in old fresh joint fresh-positive-only fresh-omit-row; do
    columns='[.offsets,.scope,.margin,.arm,.level,.id,.requirement_count,.verdict.status]'
    cmp <(jq -c "$columns" "$restore/payload/solver/$name-v3.solver.jsonl") <(jq -c "$columns" "$name.jsonl")
done
