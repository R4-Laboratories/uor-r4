#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then
  echo 'usage: pack.sh STUDY_ROOT RAW_FIT_DIR FROZEN_BASE.d11' >&2
  exit 2
fi
root=$(realpath "$1")
fit=$(realpath "$2")
base=$(realpath "$3")
bin=${PAIRED_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}
pack=$(realpath "$bin/pack-native-source-choice")
base_sha=93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a
linked_sha=bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91

sha() { sha256sum < "$1" | cut -d' ' -f1; }
check_sha() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
check_sha "$base_sha" "$base"
(cd "$fit" && sha256sum --strict --quiet -c sealed-weights.sha256)
fit_sha=$(sha "$fit/fit.json")
jq -e --arg linked "$linked_sha" '.pass == true and .linked_weights_sha256 == $linked and .state_bytes == 49158' "$fit/fit.json" >/dev/null
mkdir "$root/product-v1"
for arm in rowwise paired; do
  state="$fit/$arm.i16le"
  state_sha=$(sha "$state")
  expected_state=$(awk -v file="$arm.i16le" '$2 == file {print $1}' "$fit/sealed-weights.sha256")
  [[ $state_sha == "$expected_state" && $(stat -c %s "$state") == 49158 ]]
  [[ $state_sha == "$(jq -er --arg arm "$arm" '.outputs[$arm].weights_sha256' "$fit/fit.json")" ]]
  [[ $(head -c 40966 "$state" | sha256sum | cut -d' ' -f1) == "$linked_sha" ]]
  out="$root/product-v1/$arm"
  mkdir "$out"
  {
    printf '%s  %s\n' "$base_sha" "$base" "$state_sha" "$state" "$fit_sha" "$fit/fit.json"
    sha256sum "$pack" "$fit/sealed-weights.sha256"
  } > "$out/INPUTS.sha256"
  "$pack" "$base" "$state" "$out/artifact.d11" "$fit_sha" > "$out/artifact.json"
  jq -e --arg fit "$fit_sha" --arg state "$state_sha" --arg base "$base_sha" '
    .schema == "uor-r4.lut-discrete-native-source-choice/1" and
    .artifact_bytes == 1202201720 and
    .source_choice.state_sha256 == $state and .source_choice.fit_sha256 == $fit and
    .source_choice.base_sha256 == $base and .source_choice.state_bytes == 49158
  ' "$out/artifact.json" >/dev/null
  sha256sum --strict --quiet -c "$out/INPUTS.sha256"
  (
    cd "$out"
    check_sha "$(jq -er .artifact_sha256 artifact.json)" artifact.d11
    sha256sum INPUTS.sha256 artifact.d11 artifact.json > SHA256SUMS
  )
done
