#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then
  echo 'usage: replay.sh OUTPUT_ROOT HELD.jsonl DEVICE_HELD_DIR' >&2
  exit 2
fi
out=$(realpath "$1")
panel=$(realpath "$2")
device=$(realpath "$3")
bin=${COMPONENT_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}
bin=$(realpath "$bin")
serve="$bin/serve-native-source-choice"
verify="$bin/verify-native-source-choice"
panel_sha=62a4099f172eea49a8fa8e5231744a45f81bcca15023271ffc8a0254cd76fe44

mkdir "$out/product-replay-v1"
for arm in zero period post-b full; do
  case "$arm" in
    zero) stem=zero-period/first-period ;;
    period) stem=zero-period/joint ;;
    post-b) stem=post-full/first-period ;;
    full) stem=post-full/joint ;;
  esac
  packed="$out/product-v1/$arm"
  artifact="$packed/artifact.d11"
  artifact_sha=$(jq -er .artifact_sha256 "$packed/artifact.json")
  forward="$device/$stem.jsonl"
  reverse="$device/$stem-reverse.jsonl"
  dir="$out/product-replay-v1/$arm"
  mkdir "$dir"
  (cd "$packed" && sha256sum --strict --quiet -c SHA256SUMS)
  {
    printf '%s  %s\n' "$artifact_sha" "$artifact" "$panel_sha" "$panel"
    sha256sum "$forward" "$reverse" "$serve" "$verify"
  } > "$dir/INPUTS.sha256"
  sha256sum --strict --quiet -c "$dir/INPUTS.sha256"
  jq -c '{prefix,positions}' "$panel" > "$dir/requests.jsonl"
  /usr/bin/time -o "$dir/client.time" -f 'wall_seconds=%e peak_rss_kib=%M' \
    "$serve" "$artifact" < "$dir/requests.jsonl" > "$dir/client.jsonl"
  "$verify" --release-factorial "$artifact" "$panel" "$forward" "$reverse" \
    "$dir/client.jsonl" > "$dir/cold.jsonl" 2> "$dir/cold-receipt.json"
  jq -e --arg artifact "$artifact_sha" '
    .artifact_sha256 == $artifact and
    .matched_client_rows == 512 and .matched_rows_per_device_direction == 512
  ' "$dir/cold-receipt.json" >/dev/null
  sha256sum --strict --quiet -c "$dir/INPUTS.sha256"
  (
    cd "$dir"
    sha256sum INPUTS.sha256 requests.jsonl client.jsonl client.time cold.jsonl cold-receipt.json > SHA256SUMS
  )
done
