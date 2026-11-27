#!/usr/bin/env bash
set -euo pipefail
out=${1:?new output directory}
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
nvcc -ptx -std=c++17 -arch=compute_89 -o "$out/ending.ptx" "$here/src/ending.cu"
ptxas --verbose --gpu-name=sm_89 "$out/ending.ptx" -o "$out/ending.cubin" > "$out/ptxas.log" 2>&1
cuobjdump --dump-sass "$out/ending.cubin" > "$out/ending.sass"
sed 's@//.*@@' "$out/ending.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2
  exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/ending.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2
  exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' ending_commit ending_pack ending_score ending_train) "$out/entries"
(cd "$out" && sha256sum ending.ptx ending.cubin ending.sass ptxas.log entries forbidden > SHA256SUMS)
