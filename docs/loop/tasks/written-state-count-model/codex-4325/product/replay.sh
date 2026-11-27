#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo 'usage: replay.sh RUN_ROOT candidate|balanced|first-period|joint SPLIT PANEL.jsonl DEVICE_FORWARD.jsonl DEVICE_REVERSE.jsonl' >&2
  exit 2
fi
root=$(realpath "$1")
model=$2
split=$3
panel=$(realpath "$4")
device_forward=$(realpath "$5")
device_reverse=$(realpath "$6")
bin=${JOINT_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}
verify=$(realpath "$bin/verify-native-source-choice")
serve=$(realpath "$bin/serve-native-source-choice")
runs=/mnt/plexpool/uor-r4-corpus/runs

case "$model" in
  candidate)
    artifact=${JOINT_CANDIDATE_ARTIFACT:-$runs/20261127-codex-scope-trace/product-v1/artifact.d11}
    artifact_sha=35c7761650990cec81ec96b96e7c3683c936633dad1b073a40c168b71e85dd28
    ;;
  balanced)
    artifact=${JOINT_BALANCED_ARTIFACT:-$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11}
    artifact_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
    ;;
  first-period|joint)
    packed="$root/product-v1/$model"
    artifact="$packed/artifact.d11"
    (cd "$packed" && sha256sum --quiet --strict -c SHA256SUMS)
    artifact_sha=$(jq -er '.artifact_sha256 | select(test("^[a-f0-9]{64}$"))' "$packed/artifact.json")
    ;;
  *) echo "unknown model $model" >&2; exit 2 ;;
esac
if [[ ! $split =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "invalid split $split" >&2
  exit 2
fi
artifact=$(realpath "$artifact")
data="$root/product-replay-v1/$model/$split"
mkdir -p "$(dirname "$data")"
mkdir "$data"

{
  printf '%s  %s\n' "$artifact_sha" "$artifact"
  sha256sum "$panel" "$device_forward" "$device_reverse" "$verify" "$serve"
} > "$data/INPUTS.sha256"
sha256sum --quiet --strict -c "$data/INPUTS.sha256"
jq -c '{prefix,positions}' "$panel" > "$data/requests.jsonl"
/usr/bin/time -o "$data/client.time" -f 'wall_seconds=%e peak_rss_kib=%M' \
  "$serve" "$artifact" < "$data/requests.jsonl" > "$data/client.jsonl"
"$verify" --release-factorial "$artifact" "$panel" "$device_forward" "$device_reverse" \
  "$data/client.jsonl" > "$data/cold.jsonl" 2> "$data/cold-receipt.json"
jq -e --arg expected "$artifact_sha" '.artifact_sha256 == $expected' "$data/cold-receipt.json" >/dev/null
sha256sum --quiet --strict -c "$data/INPUTS.sha256"
(cd "$data" && sha256sum INPUTS.sha256 client.jsonl client.time cold-receipt.json cold.jsonl requests.jsonl > SHA256SUMS)
