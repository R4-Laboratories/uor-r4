#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 5 ]]; then
    echo 'usage: grade.sh VERIFIED_INTAKE PHYSICAL_ATTEMPT RECOUNT_BIN ASSIGNMENT_BIN FRESH_OUTPUT' >&2
    exit 2
fi
intake=$(cd "$1" && pwd)
attempt=$(cd "$2" && pwd)
recount=$3
assignment=$4
test -x "$recount"
test -x "$assignment"
mkdir -m 700 -- "$5"
out=$(cd "$5" && pwd)
(cd "$intake" && sha256sum --strict -c intake.sha256) > "$out/intake.log"
test -f "$attempt/COMPLETE"
(cd "$attempt" && sha256sum --strict -c members.sha256) > "$out/physical.log"
i=$intake/inputs
args=("$i/train.jsonl" "$i/probe-final.sm89.cubin" "$i/historical.json"
      "$i/base.uorbfy03" "$i/reset.uorbfy03" "$i/full.uorbfy03")
"$recount" "${args[@]}" "$attempt/rows.jsonl" "$attempt/instrument.jsonl" > "$out/recount.json"
"$assignment" "$attempt/rows.jsonl" > "$out/assignment.json"
# Flip one measured integer in a temporary copy of the first real article.
# All hashes of the three intact physical checkpoints still match.
scratch=$(mktemp -d)
trap 'rm -f "$scratch/changed.jsonl"; rmdir "$scratch"' EXIT
{
    head -n 1 "$attempt/rows.jsonl" | jq -c '.cuda_raw_coarse_q15[0] += 1'
    tail -n +2 "$attempt/rows.jsonl"
} > "$scratch/changed.jsonl"
if "$recount" "${args[@]}" "$scratch/changed.jsonl" "$attempt/instrument.jsonl" \
    > "$out/changed.stdout" 2> "$out/changed.stderr"; then
    echo 'recount accepted corrupted physical score' >&2
    exit 1
fi
test -s "$out/changed.stderr"
cd "$out"
sha256sum recount.json assignment.json physical.log intake.log changed.stdout changed.stderr > members.sha256
