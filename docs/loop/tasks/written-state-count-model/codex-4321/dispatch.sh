#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "${1:?isolated experiment root}"
trap 'echo $? > qualified.exit' EXIT
test ! -e fit && test ! -e measured
test "$(sha256sum go.json | cut -d' ' -f1)" = "${2:?authorized GO digest}"
test "$(sha256sum approved.sha256 | cut -d' ' -f1)" = "$(jq -r .approved_sha256 go.json)"
sha256sum --strict --quiet -c approved.sha256
jq -e '.authorized and .test_fitness=="FIT"' go.json >/dev/null
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid go.json)"
export CUDA_DISABLE_PTX_JIT=1
image=image/choice.cubin
sha=$(jq -r .choice_cubin_sha256 go.json)
date -u +%FT%T.%NZ > fit.started
./choice-device train "$image" "$sha" base-full-v1 train.jsonl fit
date -u +%FT%T.%NZ > fit.completed
test "$(wc -c < fit/weights.i16le)" -eq 24582
sha256sum fit/weights.i16le fit/train.json > fit.sha256
date -u +%FT%T.%NZ > held.started
./choice-device score "$image" "$sha" copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv held.jsonl fit/weights.i16le measured
date -u +%FT%T.%NZ > held.completed
sha256sum --quiet --strict -c fit.sha256
(cd measured && sha256sum observations.jsonl reverse.jsonl replay.json) > measured.sha256
