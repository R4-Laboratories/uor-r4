#!/usr/bin/env bash
set -euo pipefail
tables=$(cd "${1:?directory containing frozen d11_tables.h}" && pwd)
out=${2:?new output directory}
here=$(cd "$(dirname "$0")" && pwd)
core="$here/../../../../../labs/m1-d11-artifact-scorer/instrument/src/kernel_core.h"
check_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}
check_sha f42c487aeedde3eb41b7378a45ec1a8e7e7cf6b3902c2aa3ade6df580cd6da16 "$core"
check_sha 8a8747f12df80ea7ef5c5259363adacaa3ed09f93a4fcaaa5f41b897a98e93a2 "$tables/d11_tables.h"
mkdir "$out"
cp "$core" "$out/kernel_core.h"
nvcc -ptx -std=c++17 -arch=compute_89 -I "$out" -I "$tables" -o "$out/completion.ptx" "$here/src/completion.cu"
ptxas --verbose --gpu-name=sm_89 "$out/completion.ptx" -o "$out/completion.cubin" > "$out/ptxas.log" 2>&1
cuobjdump --dump-sass "$out/completion.cubin" > "$out/completion.sass"
sed 's@//.*@@' "$out/completion.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2; exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/completion.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2; exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' postperiod_select postperiod_trace) "$out/entries"
(cd "$out" && sha256sum kernel_core.h completion.ptx completion.cubin completion.sass > SHA256SUMS)
