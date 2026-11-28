#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 4 ]]; then
  echo 'usage: readback.sh WICK_RUNS NEW_READBACK PACKET.tar.zst PACKET_SHA256' >&2
  exit 2
fi
runs=$(realpath "$1")
out=$(realpath -m "$2")
packet=$(realpath "$3")
packet_sha=$4
[[ $out == "$runs/20261127-codex-ordinary-source/"* ]]
printf '%s  %s\n' "$packet_sha" "$packet" | sha256sum --strict --quiet -c -
mkdir "$out" "$out/payload"
tar --zstd -xf "$packet" -C "$out/payload"
payload="$out/payload"
(
  cd "$payload"
  sha256sum --strict --quiet -c PAYLOAD.sha256
)
artifact="$runs/20261127-codex-word-ending/product-qualified-v1/artifact.d11"
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
cmp "$payload/tokenizer.json" "$tokenizer"

"$payload/tools/ordinary-source-panel" "$payload/tokenizer.json" "$out/encoded"
for file in prompts.jsonl truth.jsonl; do
  cmp "$payload/panel-v1/$file" "$out/encoded/$file"
done
(cd "$payload/panel-v1" && sha256sum --strict --quiet -c SEALED.sha256)
runtime="$payload/runtime-v1"
"$payload/tools/ordinary-source-grade" "$out/encoded/truth.jsonl" \
  "$runtime/traces.jsonl" "$runtime/uninstrumented.jsonl" "$out/graded"
for file in rows.jsonl summary.json; do
  cmp "$payload/grade-v1/$file" "$out/graded/$file"
done
bash "$payload/source/controls/run.sh" "$artifact" \
  93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a base \
  "$payload/tools/verify-native-copy" "$payload/tools/verify-native-occurrence" \
  "$out/encoded/truth.jsonl" "$runtime/traces.jsonl" "$out/controls"
for file in requests.jsonl copy.jsonl copy.json occurrence.jsonl occurrence.json grade.jsonl summary.json; do
  cmp "$payload/controls-v1/$file" "$out/controls/$file"
done
jq -sS --slurpfile prompts "$out/encoded/prompts.jsonl" \
  -f "$payload/source/analysis/matched.jq" "$out/controls/grade.jsonl" > "$out/matched.json"
cmp "$payload/source/evidence/matched.json" "$out/matched.json"
cmp "$payload/matched-v1/matched.json" "$out/matched.json"
release="$payload/release-v1"
jq -sS --slurpfile ordinary "$runtime/traces.jsonl" --slurpfile truth "$out/encoded/truth.jsonl" \
  -f "$payload/source/release/grade.jq" "$release/"{zero,rowwise,paired}.jsonl > "$out/release-grade.json"
jq -S 'del(.rows)' "$out/release-grade.json" > "$out/release-summary.json"
jq -S --slurpfile truth "$out/encoded/truth.jsonl" -f "$payload/source/release/instruction.jq" \
  "$out/release-grade.json" > "$out/release-scope.json"
cmp "$payload/release-grade-v1/grade.json" "$out/release-grade.json"
cmp "$payload/release-grade-v1/summary.json" "$out/release-summary.json"
cmp "$payload/release-grade-v1/scope.json" "$out/release-scope.json"
bash "$payload/source/controls/replay-sample.sh" "$runs" "$payload/tools/trace-chat" \
  "$out/encoded/prompts.jsonl" "$runtime/traces.jsonl" "$out/ordinary-sample"
bash "$payload/source/controls/release-sample.sh" "$runs" "$payload/tools/release-chat" \
  "$out/encoded/prompts.jsonl" "$out/encoded/truth.jsonl" "$release" "$out/release-sample"
(
  cd "$out"
  sha256sum payload/PAYLOAD.sha256 encoded/{prompts,truth}.jsonl \
    graded/{rows.jsonl,summary.json} ordinary-sample/{observed,expected}.jsonl \
    release-{grade,summary,scope}.json release-sample/{cases.json,observed.jsonl,expected.jsonl} \
    controls/{requests,copy,occurrence,grade}.jsonl controls/summary.json matched.json > readback.sha256
  sha256sum --strict --quiet -c readback.sha256
)
