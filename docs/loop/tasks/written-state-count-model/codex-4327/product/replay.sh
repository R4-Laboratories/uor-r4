#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then
  echo 'usage: replay.sh STUDY_ROOT HELD.jsonl RAW_HELD_TRAINED_DIR' >&2
  exit 2
fi
root=$(realpath "$1")
panel=$(realpath "$2")
gpu=$(realpath "$3")
bin=$(realpath "${PAIRED_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}")
serve="$bin/serve-native-source-choice"
verify="$bin/verify-native-source-choice"
rows=$(wc -l < "$panel")
mkdir "$root/product-replay-v1"
for arm in rowwise paired; do
  case "$arm" in
    rowwise) slot=first-period ;;
    paired) slot=joint ;;
  esac
  forward="$gpu/$slot.jsonl"
  reverse="$gpu/$slot-reverse.jsonl"
  packed="$root/product-v1/$arm"
  (cd "$packed" && sha256sum --strict --quiet -c SHA256SUMS)
  artifact="$packed/artifact.d11"
  expected_sha=$(jq -er .artifact_sha256 "$packed/artifact.json")
  dir="$root/product-replay-v1/$arm"
  mkdir "$dir"
  {
    printf '%s  %s\n' "$expected_sha" "$artifact"
    sha256sum "$panel" "$forward" "$reverse" "$serve" "$verify"
  } > "$dir/INPUTS.sha256"
  sha256sum --strict --quiet -c "$dir/INPUTS.sha256"
  jq -c '{prefix,positions}' "$panel" > "$dir/requests.jsonl"
  /usr/bin/time -o "$dir/client.time" -f 'wall_seconds=%e peak_rss_kib=%M' \
    "$serve" "$artifact" < "$dir/requests.jsonl" > "$dir/client.jsonl"
  "$verify" --release-factorial "$artifact" "$panel" "$forward" "$reverse" \
    "$dir/client.jsonl" > "$dir/cold.jsonl" 2> "$dir/cold-receipt.json"
  jq -e --arg artifact "$expected_sha" --argjson rows "$rows" '
    .artifact_sha256 == $artifact and
    .matched_client_rows == $rows and .matched_rows_per_device_direction == $rows
  ' "$dir/cold-receipt.json" >/dev/null
  sha256sum --strict --quiet -c "$dir/INPUTS.sha256"
  (
    cd "$dir"
    sha256sum INPUTS.sha256 requests.jsonl client.jsonl client.time cold.jsonl cold-receipt.json > SHA256SUMS
  )
done
