#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
# Run from the leased job root with a sealed release from Wick.
for dir in runtime build-v1 finite-v1 input; do
    (cd "$dir" && sha256sum --strict --quiet -c SHA256SUMS)
done
test -s input/FIT.md
test -s input/PRE-REVIEW.md
test ! -e raw
mkdir -p raw/controls raw/ordinary
runtime/bundle pack ordered runtime/slot3.state raw/ordinary/state.bridge
test "$(runtime/bundle unpack raw/ordinary/state.bridge raw/ordinary/state.readback)" = ordered
cmp raw/ordinary/state.bridge runtime/protected.bridge
cmp raw/ordinary/state.readback runtime/slot3.state
for part in regression near fresh; do
    text="input/$part/text.bin"
    runtime/textlinks-sm89 learned runtime/primary.words "$text" raw/ordinary/state.readback "raw/ordinary/$part.native.jsonl"
    if [[ $part != fresh ]]; then
        cmp "input/$part/original.native.jsonl" "raw/ordinary/$part.native.jsonl"
    fi
    build-v1/argument-score runtime/primary.words "$text" raw/ordinary/state.readback "input/$part/masks.tsv" "raw/$part"
    cp "raw/ordinary/$part.native.jsonl" "raw/$part/ordinary.jsonl"
    cmp "raw/$part/baseline.jsonl" "raw/$part/native-reference.jsonl"
    cmp "raw/$part/baseline.jsonl" "raw/$part/ordinary.jsonl"
    cmp "raw/$part/state.readback" runtime/slot3.state
done
# These two separate invocations run after the source reads. Both files are already encoded.
for which in gold swapped; do
    runtime/textlinks-sm89 correct runtime/primary.words input/fresh/text.bin \
        "input/controls/$which-aid.bin" "raw/controls/$which.jsonl"
done
(find raw -type f -print0 | sort -z | xargs -0 sha256sum) > raw.members
(cd raw && jq -e -s 'length==256' fresh/baseline.jsonl && jq -e -s 'length==256' near/baseline.jsonl &&
    jq -e -s 'length==8' regression/baseline.jsonl && jq -e -s 'length==256' controls/swapped.jsonl)
