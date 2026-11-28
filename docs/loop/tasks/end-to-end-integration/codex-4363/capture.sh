#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 2 ]]; then
    echo 'usage: capture.sh VERIFIED_INTAKE FRESH_ATTEMPT' >&2
    exit 2
fi
intake=$(cd "$1" && pwd)
mkdir -m 700 -- "$2"
cd "$2"
out=$PWD
cd "$intake"
export CUDA_DISABLE_PTX_JIT=1
sha256sum --strict -c intake.sha256 > "$out/intake.log"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv > "$out/device.csv"
test "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)" = 'NVIDIA GeForce RTX 4090, 8.9'
# The instrument runs the planted controls and the historical fixture before
# opening the TRAIN output file. It refuses reuse of an output path.
bin/margin inputs/probe-final.sm89.cubin inputs/historical.json inputs/train.jsonl \
    inputs/base.uorbfy03 inputs/reset.uorbfy03 inputs/full.uorbfy03 "$out/rows.jsonl" \
    > "$out/instrument.jsonl" 2> "$out/stderr.log"
test "$(wc -l < "$out/rows.jsonl")" -eq 768
cd "$out"
sha256sum device.csv intake.log instrument.jsonl rows.jsonl stderr.log > members.sha256
sha256sum --strict -c members.sha256 > verified.log
touch COMPLETE
