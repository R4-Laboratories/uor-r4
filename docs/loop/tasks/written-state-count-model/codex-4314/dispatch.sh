#!/usr/bin/env bash
# Invoke under the named RTX 4090 lease, from the staged experiment directory.
set -euo pipefail
umask 077
expected=${1:?GO SHA-256}
test "$(sha256sum go.json | cut -d' ' -f1)" = "$expected"
test "$(sha256sum approved.sha256 | cut -d' ' -f1)" = "$(jq -r .approved_sha256 go.json)"
sha256sum --quiet --strict -c approved.sha256
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid go.json)"
jq -e '.authorized and .fit=="TEST FITNESS: FIT"' go.json >/dev/null
jq -e '.status=="pass" and (.controls|length)>=10' finite/finite.json >/dev/null
export CUDA_DISABLE_PTX_JIT=1
finish() { rc=$?; printf '%s\n' "$rc" > dispatch.exit.tmp; mv dispatch.exit.tmp dispatch.exit; }
trap finish EXIT
date -u +%FT%T.%NZ > fit.started
./fit-native-opening fit opening.cubin base-v1 donor-v2/colon-targets.jsonl panel-v2/train/rows.jsonl weights
(cd weights && sha256sum colon.tsv marker.tsv receipts.jsonl > SHA256SUMS)
date -u +%FT%T.%NZ > fit.completed
./fit-native-opening score opening.cubin base-v1 panel-v2/held/rows.jsonl weights held
./unchanged-lexical answer-first/artifact.d11 panel-v2/held/lexical-steps.jsonl held/later-lexical.jsonl
(cd held && sha256sum colon.jsonl marker.jsonl later-lexical.jsonl > SHA256SUMS)
date -u +%FT%T.%NZ > held.completed
