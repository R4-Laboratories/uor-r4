#!/usr/bin/env bash
# Invoke on the leased GPU from the frozen working directory.
set -euo pipefail
cd "${1:?working directory}"
expected=${2:?GO file SHA-256}
umask 077
finish() { rc=$?; echo "$rc" > qualified.exit.tmp; mv qualified.exit.tmp qualified.exit; }
trap finish EXIT
test "$(sha256sum go.json | cut -d' ' -f1)" = "$expected"
test "$(sha256sum approved.sha256 | cut -d' ' -f1)" = "$(jq -r .approved_sha256 go.json)"
sha256sum --quiet --strict -c approved.sha256
jq -e '.authorized and .test_fitness=="FIT"' go.json >/dev/null
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid go.json)"
export CUDA_DISABLE_PTX_JIT=1
date -u +%FT%T.%NZ > qualified.started
./copy-id-device score image-v2/copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv qualified.jsonl measured
date -u +%FT%T.%NZ > qualified.completed
(cd measured && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > measured.sha256
