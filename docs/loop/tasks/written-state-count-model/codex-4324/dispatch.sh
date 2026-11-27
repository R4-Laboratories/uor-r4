#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "${1:?isolated pod run directory}"
go=${2:?authorization digest}
test "$(sha256sum go.json | cut -d' ' -f1)" = "$go"
jq -e '.authorized==true and .verdict=="FIT"' go.json >/dev/null
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -er .uuid go.json)"
test "$(sha256sum static.sha256 | cut -d' ' -f1)" = "$(jq -er .static_sha256 go.json)"
test "$(sha256sum fit-verdict.md | cut -d' ' -f1)" = "$(jq -er .fit_sha256 go.json)"
grep -Fxq "TEST FITNESS: FIT" fit-verdict.md
grep -Fq "$(jq -er .uuid go.json)" fit-verdict.md
grep -Fq "$(jq -er .static_sha256 go.json)" fit-verdict.md
sha256sum --strict --quiet -c static.sha256
test "$(cat source.commit)" = "$(jq -er .source_commit go.json)"
export CUDA_DISABLE_PTX_JIT=1
mkdir raw
trap 'echo $? > raw/exit' EXIT
date -u +%FT%T.%NZ > raw/started
for split in fresh dev held prior-dev protect; do
 ./release-factorial-device score scope.cubin e69ffddfd7682113109fba31a43ba644d833cf90e3cc69830e1edf85d56171e7 original.cubin e6981880ef98f5e3ea5327e12e84ca3f184acbfa7c742525af31e31b86721402 copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv "$split.jsonl" candidate.i16le bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91 balanced.i16le 45d517f31074aa8061307a36f5d522e3af4bb464b434e5e6ff1779f4dd82c05e "raw/$split"
done
date -u +%FT%T.%NZ > raw/completed
