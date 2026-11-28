#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /root/c4361
out=/workspace/attempt02
mkdir "$out"
sha256sum -c training.sha256 > "$out/intake.txt"
export CUDA_DISABLE_PTX_JIT=1
bin/historical --historical bin/probe-final.sm89.cubin inputs/base.uorbfy03 inputs/historical.json > "$out/historical.json"
bin/train --train inputs/base.uorbfy03 inputs/train.jsonl f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a "$out/checkpoints" /workspace/attempt01/finite.json > "$out/stdout.log" 2> "$out/stderr.log"
(cd "$out" && sha256sum historical.json intake.txt stdout.log stderr.log checkpoints/* > members.sha256)
touch "$out/COMPLETE"
