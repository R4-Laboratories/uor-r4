#!/usr/bin/env bash
# Run after the recorded FIT, pre-run approval, and assignment of an owned device.
set -euo pipefail
umask 077
export LC_ALL=C

if [[ $# != 3 || ! "$1" =~ ^[0-9]{2,3}$ ]]; then
    echo 'usage: launch.sh ARCH_NUMBER SEALED_INPUTS FRESH_OUTPUT' >&2
    exit 2
fi
architecture=$1
inputs=$(realpath "$2")
out=$(realpath -m "$3")
source_dir=$(cd "$(dirname "$0")" && pwd)
(cd "$inputs" && sha256sum --strict -c TRAIN-INPUTS.sha256)
mkdir "$out"
cd "$out"

nvcc --version > nvcc-version.txt
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv > gpu.txt
sha256sum "$source_dir/interval.cu" "$source_dir/launch.sh" > source.sha256
nvcc -std=c++17 -O3 --generate-code "arch=compute_$architecture,code=sm_$architecture" \
    "$source_dir/interval.cu" -o interval-cuda
sha256sum interval-cuda > executable.sha256

# Scan the actual embedded device image and the compiler's corresponding PTX
# listing before TRAIN. The deployed executable carries only native cubins.
nvcc -ptx -std=c++17 -O3 -arch="compute_$architecture" "$source_dir/interval.cu" -o interval.ptx
cuobjdump --dump-ptx interval-cuda > embedded.ptx 2> embedded-ptx.log
if grep -Eq '\.(entry|target|version)([[:space:]]|$)' embedded.ptx; then
    echo 'executable contains an embedded PTX target' >&2
    exit 1
fi
cuobjdump --extract-elf all interval-cuda > extract-elf.log 2>&1
shopt -s nullglob
cubins=( *.cubin )
if [[ ${#cubins[@]} == 0 ]]; then
    echo 'expected a native device image' >&2
    exit 1
fi
: > interval.sass
: > cubin-symbols.txt
for image in "${cubins[@]}"; do
    printf '\nImage: %s\n' "$image" >> interval.sass
    cuobjdump --dump-sass "$image" >> interval.sass
    printf '\nImage: %s\n' "$image" >> cubin-symbols.txt
    cuobjdump --dump-elf-symbols "$image" >> cubin-symbols.txt
done
sed 's@//.*@@' interval.ptx > instructions.ptx
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' instructions.ptx; then
    echo 'float or tensor PTX' >&2
    exit 1
fi
awk '/\/\*[0-9a-fA-F]+\*\// {
    sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, "")
    sub(/^@!?P[T0-9]+[[:space:]]+/, "")
    if (NF) print $1
}' interval.sass | sort -u > sass-opcodes.txt
test -s sass-opcodes.txt
if grep -Eiq '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA|TCGEN)' sass-opcodes.txt; then
    echo 'float, transcendental or tensor SASS' >&2
    exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' instructions.ptx |
    awk '{print $2}' | c++filt | cut -d '(' -f 1 | sort > ptx-entries.txt
diff -u <(printf '%s\n' fit replay) ptx-entries.txt
awk '$1 == "STT_FUNC" && $3 == "STO_ENTRY" {print $4}' cubin-symbols.txt |
    c++filt | cut -d '(' -f 1 | sort > cubin-entries.txt
diff -u ptx-entries.txt cubin-entries.txt
readelf -d interval-cuda > dynamic.txt
if grep -Eiq 'NEEDED.*(nvrtc|cublas|cudnn)' dynamic.txt; then
    echo 'prohibited runtime dependency' >&2
    exit 1
fi
sha256sum "${cubins[@]}" interval.sass interval.ptx instructions.ptx sass-opcodes.txt ptx-entries.txt \
    cubin-symbols.txt cubin-entries.txt embedded.ptx embedded-ptx.log extract-elf.log dynamic.txt > DEVICE-IMAGE.sha256

# Complete every frozen TRAIN-only update before any transfer tape is evaluated.
mkdir repeat
for arm in geometry count; do
    for control in actual permuted; do
        prefix="$arm.$control"
        ./interval-cuda train "$inputs/$arm.train.tape" "$control" "$prefix.words" "$prefix.stats.json"
        ./interval-cuda train "$inputs/$arm.train.tape" "$control" "repeat/$prefix.words" "repeat/$prefix.stats.json"
        cmp "$prefix.words" "repeat/$prefix.words"
        cmp "$prefix.stats.json" "repeat/$prefix.stats.json"
    done
done
sha256sum repeat/*.words repeat/*.stats.json > REPEAT.sha256
sha256sum geometry.actual.words geometry.permuted.words count.actual.words count.permuted.words \
    > FROZEN-WEIGHTS.sha256

(cd "$inputs" && sha256sum --strict -c EVAL-INPUTS.sha256)
for split in train protected held; do
    for arm in geometry count; do
        for control in actual permuted zero; do
            prefix="$arm.$control"
            words="$prefix.words"
            if [[ "$control" == zero ]]; then
                words="$inputs/zero.words"
            fi
            ./interval-cuda eval "$inputs/$arm.$split.eval.tape" "$words" "$prefix.$split.scores"
        done
    done
done
sha256sum ./*.stats.json ./*.scores ./*.words FROZEN-WEIGHTS.sha256 REPEAT.sha256 executable.sha256 gpu.txt \
    nvcc-version.txt source.sha256 DEVICE-IMAGE.sha256 > OUTPUTS.sha256
