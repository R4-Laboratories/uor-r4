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
sha() { sha256sum "$1" | cut -d' ' -f1; }
image_sha=43c00194967c4ad875b642274a9e9eede1251dd63c9be67ea293f1012409b0e7
common=(copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv)
test "$(sha full.i16le)" = 2216c08d143fa15de31a7c31e009c791b5e56ad845f35bd89327afdbdcc29525
mkdir raw
trap 'echo $? > raw/exit' EXIT
date -u +%FT%T.%NZ > raw/started
for split in held train quartet scope-dev scope-held prior-dev protect; do
 mkdir "raw/$split"
 for grp in zero-period post-full; do
  case "$grp" in
   zero-period) left=zero.i16le; right=period.i16le;;
   post-full) left=post-b.i16le; right=full.i16le;;
  esac
  ./joint-release-device score joint-image/joint.cubin "$image_sha" "${common[@]}" "$split.jsonl" "$left" "$(sha "$left")" "$right" "$(sha "$right")" "raw/$split/$grp"
 done
done
date -u +%FT%T.%NZ > raw/completed
