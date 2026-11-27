#!/usr/bin/env bash
# First argument is the OUT_DIR of this source tree's compiled native scorer.
set -euo pipefail
tables=${1:?scorer build output directory}
out=${2:?new image directory}
here=$(cd "$(dirname "$0")" && pwd)
core="$here/../../../../../labs/m1-d11-artifact-scorer/instrument/src"
test -s "$tables/d11_tables.h"
mkdir "$out"
nvcc -ptx -std=c++17 -arch=compute_89 -I "$tables" -I "$core" -o "$out/occurrence.ptx" "$here/src/occurrence.cu"
ptxas --verbose --gpu-name=sm_89 "$out/occurrence.ptx" -o "$out/occurrence.cubin" > "$out/ptxas.log" 2>&1
cuobjdump --dump-sass "$out/occurrence.cubin" > "$out/occurrence.sass"
# Audit executable instructions, ignoring comments and source paths.
sed 's@//.*@@' "$out/occurrence.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2; exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/occurrence.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2; exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' fit_occurrences publish_weights score_occurrences) "$out/entries"
(cd "$out" && sha256sum occurrence.ptx occurrence.cubin occurrence.sass > SHA256SUMS)
