#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /root/c4361
out=/workspace/attempt03
mkdir "$out"
exec 2>"$out/errors.log"
sha256sum -c calibration.sha256 >"$out/intake.txt"
checkpoints=/workspace/attempt02/checkpoints
test -f /workspace/attempt02/COMPLETE
(cd /workspace/attempt02 && sha256sum -c members.sha256 >"$out/training-pins.txt")
export CUDA_DISABLE_PTX_JIT=1
kernel=bin/probe-final.sm89.cubin
kernel_sha=8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713
for kind in frozen reset full; do
    case "$kind" in
    frozen) model=inputs/base.uorbfy03; sha=20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589 ;;
    *) model="$checkpoints/$kind.uorbfy03"; sha=$(sha256sum "$model" | cut -d' ' -f1);;
    esac
    bin/evaluate evaluate "$kernel" "$kernel_sha" "$model" "$sha" inputs/first3.jsonl cc747479b693456e2d4e6b200fd0ac6b79dce16218050ee70165760c9a2fdb12 inputs/tokenizer.json "$out/$kind.rows.jsonl" >"$out/$kind.receipt.json"
done
(cd "$out" && sha256sum *.json *.jsonl *.txt errors.log >members.sha256)
touch "$out/COMPLETE"
