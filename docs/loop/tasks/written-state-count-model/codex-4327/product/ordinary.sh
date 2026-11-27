#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 4 ]]; then
  echo 'usage: ordinary.sh STUDY_ROOT ORDINARY_PROBE TOKENIZER.json FROZEN_BASE.d11' >&2
  exit 2
fi
root=$(realpath "$1")
probe=$(realpath "$2")
vocab=$(realpath "$3")
base=$(realpath "$4")
bin=$(realpath "${PAIRED_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}")
product=$(cd "$(dirname "$0")" && pwd)
panel="$product/ordinary/dialogues.json"
chat="$bin/uor-chat"
grounded="$bin/serve-native-source-choice"
base_sha=93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a
out="$root/product-ordinary-v1"
mkdir "$out"
(cd "$product/ordinary" && sha256sum --strict --quiet -c SEALED.sha256)
{
  printf '%s  %s\n' "$base_sha" "$base"
  sha256sum "$probe" "$chat" "$grounded" "$vocab" "$panel" "$product/ordinary/SEALED.sha256"
} > "$out/INPUTS.sha256"
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
for arm in base rowwise paired; do
  if [[ $arm == base ]]; then
    artifact="$base"
    artifact_sha="$base_sha"
  else
    packed="$root/product-v1/$arm"
    artifact="$packed/artifact.d11"
    (cd "$packed" && sha256sum --strict --quiet -c SHA256SUMS)
    artifact_sha=$(jq -er .artifact_sha256 "$packed/artifact.json")
  fi
  /usr/bin/time -o "$out/$arm.time" -f 'wall_seconds=%e peak_rss_kib=%M' \
    "$probe" "$chat" "$grounded" "$artifact" "$vocab" "$panel" "$artifact_sha" "$arm" > "$out/$arm.jsonl"
  jq -esc '[.[] | select(.kind == "turn")] | length == 22' "$out/$arm.jsonl" >/dev/null
done

jq -n --slurpfile base "$out/base.jsonl" --slurpfile rowwise "$out/rowwise.jsonl" \
  --slurpfile paired "$out/paired.jsonl" '
  def turns: map(select(.kind == "turn") | del(.artifact_sha256, .model_schema));
  def details: {
    summaries: map(select(.kind == "summary")),
    refusals: map(select(.kind == "interface")),
    raw_tokens: (turns | map(.emitted | length) | add),
    scored_tokens: (turns | map(if .eos then .eos_position + 1 else .emitted | length end) | add)
  };
  {
    base: ($base | details), rowwise: ($rowwise | details), paired: ($paired | details),
    same_turn_records: (($base|turns) == ($rowwise|turns) and ($rowwise|turns) == ($paired|turns))
  }
' > "$out/summary.json"
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
(
  cd "$out"
  sha256sum INPUTS.sha256 base.jsonl rowwise.jsonl paired.jsonl base.time rowwise.time paired.time summary.json > SHA256SUMS
)
