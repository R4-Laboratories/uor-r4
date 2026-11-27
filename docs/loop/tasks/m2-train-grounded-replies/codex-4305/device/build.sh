#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
src=$(realpath -e "${1:?source closure}")
out=${2:?unused build directory}
cuda=${CUDA_BIN_DIR:-/usr/local/cuda/bin}
(cd "$src" && sha256sum --strict -c SHA256SUMS)
mkdir "$out"
out=$(realpath -e "$out")
cp "$src/SHA256SUMS" "$out/source.sha256"
scan='(^|[^[:alnum:]_])(softmax|log_softmax|softplus|sigmoid|tanh|gelu|matmul|gemm|cublas|nvrtc)([^[:alnum:]_]|$)'
if grep -Ein "$scan" "$src"/argument-score.cu "$src"/textlinks-shared.cuh "$src"/codex-4278/fit/device.cu > "$out/source-scan.txt"; then
    cat "$out/source-scan.txt" >&2
    exit 1
fi
"$cuda/nvcc" --version > "$out/nvcc.txt"
"$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 -I "$src" \
    "$src/argument-score.cu" -o "$out/argument-score" > "$out/compile.log" 2>&1 || {
    cat "$out/compile.log" >&2
    exit 1
}
"$cuda/cuobjdump" --list-elf "$out/argument-score" > "$out/elf.txt"
"$cuda/cuobjdump" --list-ptx "$out/argument-score" > "$out/ptx.txt" 2>&1 || true
grep -q sm_89 "$out/elf.txt"
grep -Fq 'No PTX file found to extract' "$out/ptx.txt"
"$cuda/cuobjdump" --dump-sass "$out/argument-score" > "$out/device.sass"
if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|F2F|F2I|I2F|DADD|DFMA|DMUL|DSETP|HFMA|HFMA2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$out/device.sass"; then
    exit 1
fi
(cd "$out" && sha256sum argument-score compile.log device.sass elf.txt nvcc.txt ptx.txt source-scan.txt source.sha256 > SHA256SUMS)
sha256sum "$out/SHA256SUMS" "$out/argument-score"
