#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo 'usage: replay.sh RUN_ROOT candidate|balanced SPLIT PANEL.jsonl DEVICE_FORWARD.jsonl DEVICE_REVERSE.jsonl' >&2
  exit 2
fi
root=$(realpath "$1")
model=$2
split=$3
panel=$(realpath "$4")
device_forward=$(realpath "$5")
device_reverse=$(realpath "$6")
bin=${FACTORIAL_PRODUCT_BIN:-/home/callard/uor-codex/20261127-factorial-product/target/release}
verify=$(realpath "$bin/verify-native-release-factorial")
serve=$(realpath "$bin/serve-native-source-choice")
runs=/mnt/plexpool/uor-r4-corpus/runs

case "$model" in
  candidate)
    artifact=${FACTORIAL_CANDIDATE_ARTIFACT:-$runs/20261127-codex-scope-trace/product-v1/artifact.d11}
    artifact_sha=35c7761650990cec81ec96b96e7c3683c936633dad1b073a40c168b71e85dd28
    ;;
  balanced)
    artifact=${FACTORIAL_BALANCED_ARTIFACT:-$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11}
    artifact_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
    ;;
  *) echo "unknown frozen model $model" >&2; exit 2 ;;
esac
if [[ ! $split =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "invalid split $split" >&2
  exit 2
fi
artifact=$(realpath "$artifact")
data="$root/product-replay-v1/$model/$split"
mkdir -p "$(dirname "$data")"
mkdir "$data"

sha256sum "$artifact" "$panel" "$device_forward" "$device_reverse" "$verify" "$serve" > "$data/INPUTS.sha256"
read -r actual_sha _ < "$data/INPUTS.sha256"
if [[ $actual_sha != "$artifact_sha" ]]; then
  echo "frozen $model artifact digest mismatch" >&2
  exit 1
fi
jq -c '{prefix,positions}' "$panel" > "$data/requests.jsonl"
"$serve" "$artifact" < "$data/requests.jsonl" > "$data/client.jsonl"
"$verify" "$artifact" "$panel" "$device_forward" "$device_reverse" "$data/client.jsonl" \
  > "$data/cold.jsonl" 2> "$data/cold-receipt.json"
jq -e --arg expected "$artifact_sha" '.artifact_sha256 == $expected' "$data/cold-receipt.json" >/dev/null
sha256sum --quiet --strict -c "$data/INPUTS.sha256"
(cd "$data" && sha256sum INPUTS.sha256 client.jsonl cold-receipt.json cold.jsonl requests.jsonl > SHA256SUMS)
