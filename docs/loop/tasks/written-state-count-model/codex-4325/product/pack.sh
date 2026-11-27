#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 && $# -ne 5 ]]; then
  echo 'usage: pack.sh RUN_ROOT first-period|joint WEIGHTS.i16le FIT.json [BASE.d11]' >&2
  exit 2
fi
root=$(realpath "$1")
arm=$2
weights=$(realpath "$3")
fit=$(realpath "$4")
runs=/mnt/plexpool/uor-r4-corpus/runs
base=$(realpath "${5:-$runs/20261127-codex-word-ending/product-qualified-v1/artifact.d11}")
bin=${JOINT_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}
pack=$(realpath "$bin/pack-native-source-choice")

case "$arm" in
  first-period|joint) ;;
  *) echo "unknown fit $arm" >&2; exit 2 ;;
esac
linked_sha=bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91
base_sha=93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a
weights_sha=$(jq -er --arg arm "$arm" --arg linked "$linked_sha" '
  select(.linked_weights_sha256 == $linked) | .outputs[$arm].weights_sha256 |
  select(type == "string" and test("^[a-f0-9]{64}$"))
' "$fit")
fit_sha=$(sha256sum < "$fit")
fit_sha=${fit_sha%% *}
if [[ $(stat -c %s "$weights") != 49158 ]]; then
  echo "wrong size for the $arm native state" >&2
  exit 1
fi
data="$root/product-v1/$arm"
mkdir -p "$(dirname "$data")"
mkdir "$data"
{
  printf '%s  %s\n' "$base_sha" "$base" "$weights_sha" "$weights" "$fit_sha" "$fit"
  sha256sum "$pack"
} > "$data/INPUTS.sha256"
sha256sum --quiet --strict -c "$data/INPUTS.sha256"
[[ $(jq -er --arg arm "$arm" '.outputs[$arm].weights_sha256' "$fit") == "$weights_sha" ]]
"$pack" "$base" "$weights" "$data/artifact.d11" "$fit_sha" > "$data/artifact.json"
jq -e --arg fit "$fit_sha" --arg weights "$weights_sha" --arg base "$base_sha" '
  .source_choice.state_sha256 == $weights and
  .source_choice.fit_sha256 == $fit and
  .source_choice.base_sha256 == $base and
  .source_choice.state_bytes == 49158
' "$data/artifact.json" >/dev/null
sha256sum --quiet --strict -c "$data/INPUTS.sha256"
(cd "$data" && sha256sum INPUTS.sha256 artifact.json artifact.d11 > SHA256SUMS)
