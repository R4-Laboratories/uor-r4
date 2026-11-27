#!/usr/bin/env bash
set -euo pipefail
out=${1:?new output directory}
here=$(cd "$(dirname "$0")" && pwd)
frozen="$here/../../codex-4321/device/src/choice.cu"
predecessor="$here/../../codex-4319/device/src/ending.cu"
linked="$here/../../codex-4323/device/src/scope.cu"
release="$here/../../codex-4325/device/src/joint.cu"
printf '%s  %s\n' \
  20b840717372b56e95fc1ca26ab12a046721a767f90aaee55064a5f6fbda1363 "$frozen" \
  b4c2802e0eb06f6d54302e1f5a682de1b38bda1e53236b02520e243da095c831 "$predecessor" \
  026f88214a8552f2c371e47946994f847c6634c3031bf4eb2328da40c5f5071e "$linked" \
  f61ddbd931bba59df2af1d28eb9c346d4a3d9a0a65d3cc2e82c4acaf8a8c16da "$release" | sha256sum -c -

mkdir "$out"
awk '/^struct Update / { exit } { print }' "$predecessor" > "$out/instruction.h"
{
  awk '/^extern "C" __global__ void choice_score\(/ { exit } { print }' "$frozen"
  awk '/^extern "C" __global__ void choice_score\(/ { exit } { print }' "$linked"
  awk '/One worker applies signed unit updates/ { exit } { print }' "$release"
  cat "$here/src/paired.cu"
  sed -n '/^__device__ static Action action(/,$p' "$frozen"
} > "$out/paired.cu"

nvcc -ptx -std=c++17 -arch=compute_89 -I "$out" -o "$out/paired.ptx" "$out/paired.cu"
ptxas --verbose --gpu-name=sm_89 "$out/paired.ptx" -o "$out/paired.cubin" > "$out/ptxas.log" 2>&1
cuobjdump --dump-sass "$out/paired.cubin" > "$out/paired.sass"
sed 's@//.*@@' "$out/paired.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2
  exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/paired.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2
  exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' choice_commit choice_score choice_train paired_fit) "$out/entries"

(cd "$out" && sha256sum instruction.h paired.cu paired.ptx paired.cubin paired.sass ptxas.log entries forbidden > SHA256SUMS)
