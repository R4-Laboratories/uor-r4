#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 11 ]]; then
    echo 'usage: replay-cold.sh READOUT READOUT_SHA ORIGINAL_CLI INTAKE_INPUTS CHECKPOINTS TRAIN_RECEIPT_SHA TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA GPU_ROWS GPU_ROWS_SHA NEW_OUTPUT' >&2
    exit 2
fi

readout=$1
readout_sha=$2
cli=$3
inputs=$4
checkpoints=$5
receipt_sha=$6
training_cubin_sha=$7
finite_receipt_sha=$8
gpu_rows=$9
gpu_rows_sha=${10}
output=${11}
printf '%s  %s\n' "$readout_sha" "$readout" | sha256sum --strict --status -c -
exec "$readout" cold "$cli" "$inputs/tokenizer.json" "$inputs/role-and-source.jsonl" \
    "$inputs/train.jsonl" "$checkpoints/report.json" "$receipt_sha" \
    "$training_cubin_sha" "$finite_receipt_sha" "$inputs/base.uorbfy03" \
    "$checkpoints/sham.uorbfy03" "$checkpoints/active.uorbfy03" "$gpu_rows" "$gpu_rows_sha" "$output"
