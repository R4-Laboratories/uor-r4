#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 4 ]]; then
  echo 'usage: pack.sh OUTPUT_ROOT COMPONENTS_DIR ORIGINAL_FIT.json BASE.d11' >&2
  exit 2
fi
out=$(realpath "$1")
components=$(realpath "$2")
original_fit=$(realpath "$3")
base=$(realpath "$4")
pack=$(realpath "${COMPONENT_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}/pack-native-source-choice")

base_sha=93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a
linked_sha=bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91
original_fit_sha=50097ca583531e370d2d5a4cdf93bdfccf26fbad19bc9de1aa29554ccef6a3ee
full_sha=2216c08d143fa15de31a7c31e009c791b5e56ad845f35bd89327afdbdcc29525
full_artifact=277cd792af0ef2ba49515737c4382201baa4ce52f679f50ee0c18372cc40eba7
sha() { sha256sum < "$1" | cut -d' ' -f1; }
check_sha() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
check_sha "$base_sha" "$base"
check_sha "$original_fit_sha" "$original_fit"
jq -e --arg linked "$linked_sha" --arg full "$full_sha" '
  .linked_weights_sha256 == $linked and .outputs.joint.weights_sha256 == $full
' "$original_fit" >/dev/null
(cd "$components" && sha256sum --strict --quiet -c SHA256SUMS)
check_sha "$full_sha" "$components/full.i16le"
component_sha=$(sha "$components/SHA256SUMS")

mkdir "$out/product-v1"
for arm in zero period post-b full; do
  state="$components/$arm.i16le"
  state_sha=$(sha "$state")
  expected_state_sha=$(awk -v file="$arm.i16le" '$2 == file { print $1 }' "$components/SHA256SUMS")
  [[ $state_sha == "$expected_state_sha" && $(stat -c %s "$state") == 49158 ]]
  [[ $(head -c 40966 "$state" | sha256sum | cut -d' ' -f1) == "$linked_sha" ]]
  fit_sha=$component_sha
  if [[ $arm == full ]]; then fit_sha=$original_fit_sha; fi

  dir="$out/product-v1/$arm"
  mkdir "$dir"
  {
    printf '%s  %s\n' "$base_sha" "$base" "$state_sha" "$state" \
      "$original_fit_sha" "$original_fit" "$component_sha" "$components/SHA256SUMS"
    sha256sum "$pack"
  } > "$dir/INPUTS.sha256"
  "$pack" "$base" "$state" "$dir/artifact.d11" "$fit_sha" > "$dir/artifact.json"
  jq -e --arg fit "$fit_sha" --arg state "$state_sha" --arg base "$base_sha" '
    .schema == "uor-r4.lut-discrete-native-source-choice/1" and
    .artifact_bytes == 1202201720 and
    .source_choice.state_sha256 == $state and .source_choice.fit_sha256 == $fit and
    .source_choice.base_sha256 == $base and .source_choice.state_bytes == 49158
  ' "$dir/artifact.json" >/dev/null
  artifact_sha=$(jq -er .artifact_sha256 "$dir/artifact.json")
  if [[ $arm == full ]]; then [[ $artifact_sha == "$full_artifact" ]]; fi
  sha256sum --strict --quiet -c "$dir/INPUTS.sha256"
  (
    cd "$dir"
    check_sha "$artifact_sha" artifact.d11
    sha256sum INPUTS.sha256 artifact.d11 artifact.json > SHA256SUMS
  )
done
