#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /root/c4361
mkdir /workspace/attempt01
sha256sum -c finite.sha256 > /workspace/attempt01/pins.txt
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv >/workspace/attempt01/device.csv
export CUDA_DISABLE_PTX_JIT=1
bin/train --finite /workspace/attempt01/finite.json > /workspace/attempt01/stdout.log 2>/workspace/attempt01/stderr.log
cd /workspace/attempt01
sha256sum finite.json device.csv stdout.log stderr.log pins.txt >members.sha256
touch COMPLETE
