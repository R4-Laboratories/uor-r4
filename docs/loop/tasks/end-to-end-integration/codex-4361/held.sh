#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /root/c4361
out=/workspace/attempt04
mkdir "$out"
exec 2>"$out/errors.log"
sha256sum -c held.sha256 > "$out/intake.txt"
export CUDA_DISABLE_PTX_JIT=1
kernel=bin/probe-final.sm89.cubin
kernel_sha=8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713
held_sha=6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee
train_sha=f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a
ckpt=/workspace/attempt02/checkpoints
(cd /workspace/attempt02 && sha256sum -c members.sha256 >"$out/training-pins.txt")
for kind in frozen reset full; do
    case "$kind" in
    frozen) model=inputs/base.uorbfy03; sha=20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589 ;;
    *) model="$ckpt/$kind.uorbfy03"; sha=$(sha256sum "$model" | cut -d' ' -f1);;
    esac
    bin/evaluate evaluate "$kernel" "$kernel_sha" "$model" "$sha" inputs/held.jsonl "$held_sha" inputs/tokenizer.json "$out/$kind.rows.jsonl" >"$out/$kind.receipt.json"
done
bin/evaluate counts bin/counts.sm89.cubin 948c85b8522d72e04eb80d7217a6b397a72fc4b9329e5eed31706b43f5ee27be inputs/train.jsonl "$train_sha" inputs/held.jsonl "$held_sha" "$out/counts.rows.jsonl" >"$out/counts.receipt.json"
(cd "$out" && sha256sum *.json *.jsonl *.txt errors.log >members.sha256)
touch "$out/COMPLETE"
