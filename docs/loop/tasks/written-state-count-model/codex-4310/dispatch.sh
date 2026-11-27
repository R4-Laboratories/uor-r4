#!/usr/bin/env bash
set -euo pipefail
cd "${1:?pass staged workspace}"
out=${2:?pass a new attempt directory}
uuid=${3:?pass qualified 4090 device UUID}
mkdir "$out"
exec > >(tee "$out/dispatch.log") 2>&1
trap 'code=$?; date -u +%FT%TZ > "$out/end.utc"; echo "$code" > "$out/exit.code"' EXIT
date -u +%FT%TZ > "$out/start.utc"
p4=inputs/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11
r=inputs/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16
dev=inputs/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/dev-tune.tokens.u16
finite=prep/finite-36bacaff7
sha256sum -c "$finite/artifacts.sha256"
jq -e '.status=="PASS" and .cubin_sha256=="b2038aa9f4363a05290a8399cd84805838417e7187e7e28055c5a3a930133db2"' "$finite/finite.log"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "$uuid"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
sha256sum -c prep/approved.sha256
printf '%s  %s\n' \
  e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776 "$p4" \
  5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a "$r" \
  75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a "$dev" \
  e5b1c14eb96686b7fdee3f78bd70634f79b114e2912d5c53aca635c791edcc52 panel/fit.jsonl \
  d8630bf60f06cc65ff4b32732cf5b1b5aba2a2837008408a207e32b950d24ef0 panel/held.jsonl \
  20081180a3d4881661b7dfd0336498e4c0adb256ded264a7c6adc20c65a2dbfd panel/receipt.json \
  0b873d1afe4dfe855f18042f6973af32cb896866a85e603a68e274c0e16d5e0a bin/uor-chat | sha256sum -c -
nvidia-smi --query-gpu=timestamp,uuid,pstate,power.draw,utilization.gpu,memory.used --format=csv -l 1 > "$out/gpu.csv" &
sampler=$!
trap 'code=$?; kill "$sampler" 2>/dev/null || :; wait "$sampler" 2>/dev/null || :; date -u +%FT%TZ > "$out/end.utc"; echo "$code" > "$out/exit.code"' EXIT
export CUDA_DISABLE_PTX_JIT=1
/usr/bin/time -v -o "$out/resources.txt" bin/s4-output-runner run "$p4" "$r" "$dev" panel/fit.jsonl panel/held.jsonl "$out/trial"
jq -e '.status=="PASS" and .held_panel.records==256 and .held_panel.targets==17803' "$out/trial/measurement.json"
