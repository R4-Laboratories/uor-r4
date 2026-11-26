#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
d=$(realpath -e "${1:?finite payload}")
binary=$(realpath -e "${2:?rented SM89 build}")
out=${3:?unused output}; mkdir "$out"; out=$(realpath -e "$out")
printf 'probe\t%s\n' "$d/fixture.state" > "$out/models.tsv"
for arm in a b; do
    "$binary" "$d/fixture.words" "$d/text.bin" "$d/requests.tsv" "$out/models.tsv" "$out/$arm"
    cmp "$out/$arm/probe.native.jsonl" "$d/original.jsonl"
done
diff -r "$out/a" "$out/b"
jq -ne --slurpfile read "$out/a/probe.costs.jsonl" --slurpfile old "$d/old.requests.jsonl" \
    --slurpfile native "$out/a/probe.native.jsonl" -f "$d/finite.jq" > "$out/checks.json"
cat "$out/checks.json"
