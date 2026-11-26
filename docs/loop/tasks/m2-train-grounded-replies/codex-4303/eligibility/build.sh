#!/usr/bin/env bash
set -euo pipefail
src=$(realpath -e "${1:?checked source closure}")
out=${2:?new build directory}
cuda=${CUDA_BIN_DIR:-/usr/local/cuda/bin}
mkdir "$out"
(cd "$src" && sha256sum --quiet --strict -c SHA256SUMS)
"$cuda/nvcc" --version > "$out/nvcc.txt"
"$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 -I "$src" "$src/lexical.cu" -o "$out/lexical" > "$out/compile.log" 2>&1 || { cat "$out/compile.log"; exit 1; }
"$cuda/cuobjdump" --list-elf "$out/lexical" > "$out/elf"
"$cuda/cuobjdump" --list-ptx "$out/lexical" > "$out/ptx" 2>&1 || true
grep -q sm_89 "$out/elf"; grep -q 'No PTX file found to extract' "$out/ptx"
"$cuda/cuobjdump" --dump-sass "$out/lexical" > "$out/sass"
if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|F2F|F2I|I2F|DADD|DFMA|DMUL|DSETP|HFMA|HFMA2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$out/sass"; then exit 1; fi
(cd "$out" && find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum > .pending && mv .pending SHA256SUMS)
