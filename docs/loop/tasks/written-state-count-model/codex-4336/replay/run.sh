#!/usr/bin/env bash
set -euo pipefail
checker=$(realpath "${1:?independent #4335 replay binary}")
artifact=$(realpath "${2:?balanced artifact}")
records=$(realpath "${3:?captured client JSON lines}")
out=$(realpath -m "${4:?fresh output}")
mkdir "$out"
jq -ce 'select((.client.verified_span_release.steps // []) | length > 0) | .client' \
  "$records" > "$out/observations.jsonl"
"$checker" "$artifact" < "$out/observations.jsonl" > "$out/summary.json"
# Mutate the recorded ordinary proposal after an intervened commit. The final
# emitted ID, span and grade stay intact, so a grade-only checker would pass.
jq -ce 'select(.verified_span_release.steps | length > 1) |
  select(.verified_span_release.steps[0].committed_id != .verified_span_release.steps[0].ordinary_id) |
  .verified_span_release.steps[1].ordinary_id += 1' "$out/observations.jsonl" > "$out/damaged.jsonl"
test -s "$out/damaged.jsonl"
if "$checker" "$artifact" < "$out/damaged.jsonl" > "$out/damaged.stdout" 2> "$out/damaged.stderr"; then
  echo "replay accepted a forged next ordinary proposal" >&2
  exit 1
fi
grep -Fq 'ordinary proposal or self-fed ID mismatch at offset 1' "$out/damaged.stderr"
(cd "$out" && sha256sum observations.jsonl summary.json damaged.jsonl damaged.stderr > SHA256SUMS)
