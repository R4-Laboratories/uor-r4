#!/usr/bin/env bash
set -euo pipefail
out=${1:?new output directory}
here=$(cd "$(dirname "$0")" && pwd)
frozen="$here/../../codex-4321/device/src/choice.cu"
predecessor="$here/../../codex-4319/device/src/ending.cu"
linked="$here/../../codex-4323/device/src/scope.cu"
printf '%s  %s\n' \
  20b840717372b56e95fc1ca26ab12a046721a767f90aaee55064a5f6fbda1363 "$frozen" \
  b4c2802e0eb06f6d54302e1f5a682de1b38bda1e53236b02520e243da095c831 "$predecessor" \
  026f88214a8552f2c371e47946994f847c6634c3031bf4eb2328da40c5f5071e "$linked" | sha256sum -c -

mkdir "$out"
awk '/^struct Update / { exit } { print }' "$predecessor" > "$out/instruction.h"
{
  awk '/^extern "C" __global__ void choice_score\(/ { exit } { print }' "$frozen"
  awk '/^extern "C" __global__ void choice_score\(/ { exit } { print }' "$linked"
  cat "$here/src/joint.cu"
  sed -n '/^__device__ static Action action(/,$p' "$frozen"
} > "$out/joint.cu"

nvcc -ptx -std=c++17 -arch=compute_89 -I "$out" -o "$out/joint.ptx" "$out/joint.cu"
ptxas --verbose --gpu-name=sm_89 "$out/joint.ptx" -o "$out/joint.cubin" > "$out/ptxas.log" 2>&1
cuobjdump --dump-sass "$out/joint.cubin" > "$out/joint.sass"
sed 's@//.*@@' "$out/joint.ptx" > "$out/instructions.ptx"
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' "$out/instructions.ptx"; then
  echo "float or tensor PTX" >&2
  exit 1
fi
if awk '/\/\*[0-9a-fA-F]+\*\//{sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, ""); sub(/^@!?P[T0-9]+[[:space:]]+/, ""); print $1}' "$out/joint.sass" |
    grep -Ei '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA)' > "$out/forbidden"; then
  cat "$out/forbidden" >&2
  exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' "$out/instructions.ptx" | awk '{print $2}' | sort > "$out/entries"
diff -u <(printf '%s\n' choice_commit choice_score choice_train joint_fit) "$out/entries"

(cd "$out" && sha256sum instruction.h joint.cu joint.ptx joint.cubin joint.sass ptxas.log entries forbidden > SHA256SUMS)
