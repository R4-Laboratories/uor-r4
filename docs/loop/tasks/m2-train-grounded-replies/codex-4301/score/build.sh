#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
src=$(realpath -e "${1:?source closure}")
out=${2:?unused build dir}
cuda=${CUDA_BIN_DIR:-/usr/local/cuda/bin}
mkdir "$out"; out=$(realpath -e "$out")
test "$(grep -c '^int main(int argc, char \*\*argv) {' "$src/switch-competition.cu")" = 1
sed '/^int main(int argc, char \*\*argv) {/,$d' "$src/switch-competition.cu" > "$out/competition-shared.cuh"
"$cuda/nvcc" --version > "$out/nvcc.txt"
"$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 -I "$out" -I "$src" \
    "$src/score.cu" -o "$out/margin-score" > "$out/compile.log" 2>&1 || { cat "$out/compile.log" >&2; exit 1; }
"$cuda/cuobjdump" --list-elf "$out/margin-score" > "$out/elf.txt"
"$cuda/cuobjdump" --list-ptx "$out/margin-score" > "$out/ptx.txt" 2>&1 || true
grep -q sm_89 "$out/elf.txt"; grep -Fq 'No PTX file found to extract' "$out/ptx.txt"
"$cuda/cuobjdump" --dump-sass "$out/margin-score" > "$out/device.sass"
if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|F2F|F2I|I2F|DADD|DFMA|DMUL|DSETP|HFMA|HFMA2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$out/device.sass"; then exit 1; fi
(cd "$out" && find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum > .pending && mv .pending SHA256SUMS)
sha256sum "$out/SHA256SUMS" "$out/margin-score"
