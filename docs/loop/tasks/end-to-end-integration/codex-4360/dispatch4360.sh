#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /root/c4360
out=${1:?fresh absolute attempt directory}
mkdir "$out"
exec 2>"$out/diagnostics.log"
sha256sum -c dispatch.sha256 > "$out/intake.txt"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv > "$out/device.csv"
bin=run/device-v2
kernel=run/probe-final.sm89.cubin
model=assets/model.uorbfy03
"$bin" --finite "$kernel" > "$out/finite.json"
"$bin" --historical "$kernel" "$model" assets/historical.json > "$out/historical.json"
"$bin" --parity "$kernel" "$model" assets/six.jsonl 1 32 > "$out/parity.json"
"$bin" "$kernel" "$model" assets/pairs.jsonl 1 32 > "$out/rows.jsonl"
test "$(wc -l < "$out/rows.jsonl")" -eq 64
(cd "$out" && sha256sum intake.txt device.csv diagnostics.log finite.json historical.json parity.json rows.jsonl > members.sha256)
touch "$out/COMPLETE"
