#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -lt 3 ]]; then
    echo 'usage: run-stage.sh INTAKE FRESH_ATTEMPT finite|train [FINITE_RECEIPT]' >&2
    exit 2
fi
intake=$(cd "$1" && pwd)
mkdir -m 700 -- "$2"
cd "$2"
attempt=$PWD
cd "$intake"
export CUDA_DISABLE_PTX_JIT=1
test "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)" = 'NVIDIA GeForce RTX 4090, 8.9'
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv > "$attempt/device.csv"
sha256sum --strict -c intake.sha256 > "$attempt/intake.log"
case "$3" in
    finite)
        test "$#" -eq 3
        bin/readout gpu-calibration inputs/serving.sm89.cubin inputs/historical.json inputs/base.uorbfy03 \
            > "$attempt/serving-finite.json" 2> "$attempt/serving.stderr"
        bin/checker finite inputs/original-train.sm89.cubin > "$attempt/finite.json" 2> "$attempt/stderr.log"
        jq -e 'type == "object"' "$attempt/finite.json" > /dev/null
        ;;
    train)
        test "$#" -eq 4
        # The binary binds the receipt, concrete GPU UUID, cubins, roster and parent.
        bin/checker one-update inputs/base.uorbfy03 inputs/train.jsonl "$4" \
            inputs/original-train.sm89.cubin "$attempt/checkpoints" \
            > "$attempt/train.log" 2> "$attempt/stderr.log"
        test -s "$attempt/checkpoints/sham.uorbfy03"
        test -s "$attempt/checkpoints/active.uorbfy03"
        test -s "$attempt/checkpoints/report.json"
        ;;
    *) echo 'unknown stage' >&2; exit 2 ;;
esac
cd "$attempt"
find . -type f ! -name members.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > ../"$(basename "$attempt")".sha256
mv ../"$(basename "$attempt")".sha256 members.sha256
sha256sum --strict -c members.sha256 > verified.log
touch COMPLETE
