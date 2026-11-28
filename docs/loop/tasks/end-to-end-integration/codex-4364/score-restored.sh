#!/bin/bash
set -euo pipefail
umask 077
export CUDA_DISABLE_PTX_JIT=1
cd /root/c4364
sha256sum --strict --status -c intake.sha256
printf "%s  bin/readout-v2\n" b7871a9caea7a7a72b72acff3192784bf51a40875df476e184ebdf338d161ee1 | sha256sum --strict --status -c -
ck=/workspace/attempt02/checkpoints
report=$ck/report.json
attempt=/workspace/attempt04
mkdir -m700 "$attempt"
report_sha=$(sha256sum "$report" | cut -d' ' -f1)
finite_sha=def40d1eca5803436b6890c034ffd378ff24c552a8ad6d2e6cc26a76016db53c
kernel_sha=9b8185fd00e3d7896569476b1160102dc0305ddc24fe29d63cb0bd231113d533
test "$report_sha" = "$1"
(cd /workspace/attempt02 && sha256sum --strict --status -c members.sha256)
bash verify-finite.sh /workspace/attempt01 "$finite_sha"
bin/readout-v2 gpu inputs/serving.sm89.cubin inputs/historical.json inputs/train.jsonl "$report" "$report_sha" "$kernel_sha" "$finite_sha" inputs/base.uorbfy03 "$ck/sham.uorbfy03" "$ck/active.uorbfy03" "$attempt/readout" > "$attempt/stdout" 2> "$attempt/stderr"
cd "$attempt"
sha256sum readout/{calibration.json,report.json,rows.jsonl} stdout stderr > members.sha256
sha256sum --strict -c members.sha256 > verified.log
touch COMPLETE
