#!/usr/bin/env bash
set -euo pipefail
cd "${1:?staged run directory}"
trap 'echo $? > finite.exit' EXIT
if [[ -d image ]]; then
    (cd image && sha256sum -c SHA256SUMS)
else
    bash docs/loop/tasks/written-state-count-model/codex-4315/compute/build-image.sh common/headers image
fi
(cd base-full-v1 && sha256sum -c SHA256SUMS)
sha256sum -c finite-input.sha256
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > gpu.csv
start=$(date +%s%N)
./driver fixture image/occurrence.cubin base-full-v1 product-inputs-v1/token-classes.u8 finite
echo "$(( $(date +%s%N) - start ))" > finite.nanoseconds
./common/baseline score common/baseline.cubin base-v1 finite/fixture-inputs.jsonl common/baseline-state finite-baseline
./driver score image/occurrence.cubin base-full-v1 product-inputs-v1/token-classes.u8 finite/fixture-inputs.jsonl finite-baseline/marker.jsonl finite finite-ordinary
