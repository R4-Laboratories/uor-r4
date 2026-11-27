#!/usr/bin/env bash
set -euo pipefail
arm=${1:?balanced, weak, or both}
out=${2:?new output directory}
case "$arm" in
  balanced) balanced=1; weak=0 ;;
  weak) balanced=0; weak=1 ;;
  both) balanced=1; weak=1 ;;
  *) echo "Unknown fit arm: $arm" >&2; exit 1 ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
frozen="$here/../../codex-4321/device/src/choice.cu"
predecessor="$here/../../codex-4319/device/src/ending.cu"
printf '%s  %s\n' \
  20b840717372b56e95fc1ca26ab12a046721a767f90aaee55064a5f6fbda1363 "$frozen" \
  b4c2802e0eb06f6d54302e1f5a682de1b38bda1e53236b02520e243da095c831 "$predecessor" |
  sha256sum -c -

mkdir "$out"
awk '/^struct Update / { exit } { print }' "$predecessor" > "$out/instruction.h"
{
  printf '#define CHOICE_BALANCED %s\n#define CHOICE_WEAK %s\n' "$balanced" "$weak"
  awk '/^\/\/ Each stage is independent/ { exit } { print }' "$frozen"
  cat "$here/margin.cu"
  sed -n '/^extern "C" __global__ void choice_score(/,$p' "$frozen"
} > "$out/choice.cu"

nvcc -ptx -std=c++17 -arch=compute_89 -I "$out" -o "$out/choice.ptx" "$out/choice.cu"
ptxas --verbose --gpu-name=sm_89 "$out/choice.ptx" -o "$out/choice.cubin" > "$out/ptxas.log" 2>&1
cuobjdump --dump-sass "$out/choice.cubin" > "$out/choice.sass"
sed 's@//.*@@' "$out/choice.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2
  exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/choice.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2
  exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' choice_commit choice_score choice_train) "$out/entries"

nvcc -std=c++17 -lcuda -o "$out/finite" "$here/finite.cpp"
(cd "$out" && sha256sum instruction.h choice.cu choice.ptx choice.cubin choice.sass ptxas.log entries forbidden finite > SHA256SUMS)
