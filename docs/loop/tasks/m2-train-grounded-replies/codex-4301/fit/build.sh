#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 1 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
out=$1
cuda_bin=${CUDA_BIN:-/usr/local/cuda/bin}
(cd "$here" && sha256sum --quiet --strict -c SHA256SUMS)
mkdir -- "$out"
"$cuda_bin/nvcc" --version > "$out/nvcc.txt"
rustc --version > "$out/rustc.txt"
for entry in paired-fit switch-competition; do
    "$cuda_bin/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        "$here/$entry.cu" -o "$out/$entry-sm89" > "$out/$entry.build.log" 2>&1
    "$cuda_bin/cuobjdump" --list-elf "$out/$entry-sm89" > "$out/$entry.elf"
    "$cuda_bin/cuobjdump" --list-ptx "$out/$entry-sm89" > "$out/$entry.ptx" 2>&1 || true
    grep -q sm_89 "$out/$entry.elf"
    grep -Fq 'No PTX file found to extract' "$out/$entry.ptx"
    "$cuda_bin/cuobjdump" --dump-sass "$out/$entry-sm89" > "$out/$entry.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|F2F|F2I|I2F|DADD|DFMA|DMUL|DSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$out/$entry.sass"; then
        echo 'Banned device instruction' >&2
        exit 1
    fi
done
for kernel in _ZN9textlinks11attachments _ZN9textlinks7predict; do
    for entry in paired-fit switch-competition; do
        awk -v kernel="$kernel" '/Function : / { keep = index($0, kernel) != 0 } keep { print }' "$out/$entry.sass" \
            > "$out/$entry-$kernel.sass"
        test -s "$out/$entry-$kernel.sass"
    done
    cmp "$out/paired-fit-$kernel.sass" "$out/switch-competition-$kernel.sass"
done
rustc --edition 2021 -D warnings "$here/fixture.rs" -o "$out/fixture"
cp -- "$here/SHA256SUMS" "$out/sources.SHA256SUMS"
(cd "$out" && find . -type f -print0 | sort -z | xargs -0 sha256sum > ../"${out##*/}".pending)
mv -- "$(dirname "$out")/${out##*/}.pending" "$out/SHA256SUMS"
