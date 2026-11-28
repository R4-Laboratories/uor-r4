#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
tables=$(realpath "${1:?D11 generated header directory}")
out=${2:?new image directory}
here=$(cd "$(dirname "$0")" && pwd)
core="$here/../../../../../labs/m1-d11-artifact-scorer/instrument/src"
cuda_bin=${CUDA_BIN:-/usr/local/cuda/bin}
test -s "$tables/d11_tables.h"
mkdir "$out"
"$cuda_bin/nvcc" -ptx -std=c++17 -arch=compute_89 -I "$tables" -I "$core" -o "$out/whole-turn.ptx" "$here/src/whole_turn.cu"
"$cuda_bin/ptxas" --verbose --gpu-name=sm_89 "$out/whole-turn.ptx" -o "$out/whole-turn.cubin" > "$out/ptxas.log" 2>&1
"$cuda_bin/cuobjdump" --dump-sass "$out/whole-turn.cubin" > "$out/whole-turn.sass"
sed 's@//.*@@' "$out/whole-turn.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2
  exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/whole-turn.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2
  exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' fit_occurrences fit_whole_turn publish_weights score_occurrences score_whole_turn) "$out/entries"
sha256sum "$tables/d11_tables.h" "$core/kernel_core.h" "$here/src/whole_turn.cu" "$here/../../codex-4315/compute/src/occurrence.cu" > "$out/source.sha256"
(cd "$out" && sha256sum whole-turn.ptx whole-turn.cubin whole-turn.sass ptxas.log entries > SHA256SUMS)
