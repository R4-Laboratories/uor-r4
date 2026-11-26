#!/usr/bin/env bash
# Verify physical fixture checkpoints from the separate Rust margin oracle, then falsify the observer.
set -euo pipefail
umask 077
r=/mnt/plexpool/uor-r4-corpus/runs; s="$r/20261126-codex-mixed-transport"
prev="$r/20261126-codex-far-transport-margin"
f=$(realpath "${1:?physical fixture}"); build=$(realpath "${2:?audit build}"); out=${3:?new audit fixture root}
[[ $(hostname -s) == wick && $f == "$s"/fit/finite-v* && $out == "$s"/audit/finite-v* && $build == "$s"/audit/build-v* ]]
(cd "$f" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$build" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$out"; v="$build/target/release/transport-margin-audit"
"$prev/audit/build-v2/target/release/transport-margin-audit" contrastive "$f/text.bin" "$f/labels.bin" "$f/parent.state" "$f/legacy.state" "$f/legacy-final.state" "$out/oracle.jsonl" "$out/legacy-reconstructed.json"
jq -es --slurpfile physical "$f/mixed/trace.jsonl" -f "$prev/methods-v1/audit/compare.jq" "$out/oracle.jsonl" > "$out/signed-checks.json"
for arm in mixed all-far; do "$v" "$f/text.bin" "$f/$arm.txt" "$f/parent.state" "$out/oracle.jsonl" "$f/$arm" "$out/$arm.json"; done
mkdir "$out/tampered"
for file in "$f"/mixed/step-*.state "$f/mixed/"{quality,vote-summary}.jsonl; do ln -s "$file" "$out/tampered/"; done
jq -c 'if .step==0 and .sign==0 and .slot==0 and .axis==0 then .near+=1 else . end' "$f/mixed/votes.jsonl" > "$out/tampered/votes.jsonl"
if "$v" "$f/text.bin" "$f/mixed.txt" "$f/parent.state" "$out/oracle.jsonl" "$out/tampered" "$out/should-not-validate.json" 2> "$out/rejection.txt"; then exit 1; fi
grep -q 'vote evidence mismatch' "$out/rejection.txt"
(cd "$out" && sha256sum {mixed,all-far,legacy-reconstructed,signed-checks}.json oracle.jsonl rejection.txt tampered/votes.jsonl > SHA256SUMS)
