#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
export PATH="/usr/local/cuda/bin:$PATH"

if [[ $# != 2 || $(hostname -s) != wick ]]; then
    echo 'usage on Wick: finite-wick.sh GPU_UUID FRESH_OUTPUT_DIRECTORY' >&2
    exit 2
fi
gpu=$1
out=$(realpath -m "$2")
source_dir=$(cd "$(dirname "$0")" && pwd)
test "$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader)" = 8.9
mkdir "$out"
cd "$out"

sha256sum "$source_dir/termination.cu" "$source_dir/src/main.rs" "$source_dir/finite-wick.sh" > source.sha256
nvcc --version > nvcc.txt
rustc --version > rustc.txt
nvcc -std=c++17 -O3 --generate-code arch=compute_89,code=sm_89 "$source_dir/termination.cu" -o termination-cuda
nvcc -std=c++17 -O3 -arch=compute_89 -ptx "$source_dir/termination.cu" -o termination.ptx
rustc --edition 2024 -D warnings -O "$source_dir/src/main.rs" -o termination-oracle

cuobjdump --dump-ptx termination-cuda > embedded.ptx 2> embedded-ptx.log
if grep -Eq '\.(entry|target|version)([[:space:]]|$)' embedded.ptx; then
    echo 'embedded PTX survived the native-only build' >&2
    exit 1
fi
cuobjdump --extract-elf all termination-cuda > extract-elf.log 2>&1
shopt -s nullglob
cubins=( *.cubin )
test "${#cubins[@]}" -gt 0
for image in "${cubins[@]}"; do
    cuobjdump --dump-sass "$image"
done > termination.sass
for image in "${cubins[@]}"; do
    cuobjdump --dump-elf-symbols "$image"
done > cubin-symbols.txt
sed 's@//.*@@' termination.ptx > instructions.ptx
if grep -Eiq '\.(f16|f16x2|bf16|bf16x2|tf32|f32|f64|mma|wmma|wgmma|tcgen05|tensor)([. ;,]|$)|ex2\.approx' instructions.ptx; then
    echo 'prohibited arithmetic in PTX' >&2
    exit 1
fi
awk '/\/\*[0-9a-fA-F]+\*\// {
    sub(/^.*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, "")
    sub(/^@!?P[T0-9]+[[:space:]]+/, "")
    if (NF) print $1
}' termination.sass | sort -u > sass-opcodes.txt
test -s sass-opcodes.txt
if grep -Eiq '^(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DMNMX|HADD2|HMUL2|HFMA2|HSET2|HSETP2|HMNMX2|MUFU|RRO|[HIBDQG]?MMA|WMMA|TCGEN)' sass-opcodes.txt; then
    echo 'prohibited arithmetic in SASS' >&2
    exit 1
fi
grep -oE '\.entry[[:space:]]+[[:alnum:]_]+' instructions.ptx |
    awk '{print $2}' | sort > ptx-entries.txt
diff -u <(printf '%s\n' fit_termination replay_termination) ptx-entries.txt
awk '$1 == "STT_FUNC" && $3 == "STO_ENTRY" {print $4}' cubin-symbols.txt |
    sort > cubin-entries.txt
diff -u ptx-entries.txt cubin-entries.txt
readelf -d termination-cuda > dynamic.txt
if grep -Eiq 'NEEDED.*(nvrtc|cublas|cudnn)' dynamic.txt; then
    echo 'prohibited runtime dependency' >&2
    exit 1
fi
sha256sum termination-cuda termination-oracle "${cubins[@]}" termination.sass termination.ptx \
    sass-opcodes.txt cubin-entries.txt ptx-entries.txt dynamic.txt > device-image.sha256

lock="$HOME/uor-codex/20261118-diagnostics/wick-codex-gpu.lock"
test -f "$lock"
exec 8<>"$lock"
flock -n 8 || { echo 'shared Wick GPU lock busy' >&2; exit 75; }
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DISABLE_PTX_JIT=1
nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,compute_cap,memory.free --format=csv > device.csv

mkdir primary repeat
for directory in primary repeat; do
    ./termination-oracle fixture "$directory"
    for seed in 1 2; do
        for variant in actual flipped; do
            prefix="$directory/$seed.$variant"
            ./termination-cuda train "$directory/finite.train" "$seed" "$variant" "$prefix.words" "$prefix.json"
            ./termination-cuda replay "$directory/finite.replay" "$prefix.words" "$prefix.scores"
        done
    done
    for fixture in zero extreme; do
        ./termination-cuda replay "$directory/finite.replay" "$directory/$fixture.words" "$directory/$fixture.scores"
    done
    ./termination-cuda finite-saturation "$directory/saturation.words" "$directory/saturation.csv"
    ./termination-oracle verify-fixture "$directory" | tee "$directory/parity.txt"
done
diff -r primary repeat

refuse() {
    local label=$1
    local error=$2
    shift 2
    if "$@" > "$label.stdout" 2> "$label.stderr"; then
        echo "unexpected pass on $label" >&2
        exit 1
    fi
    grep -Fxq "$error" "$label.stderr"
}
refuse target 'invalid ordinary ID, target flag, or reserved byte' \
    ./termination-cuda replay primary/label.replay primary/zero.words rejected.scores
refuse foreign_bank 'feature key outside its declared bank' \
    ./termination-cuda replay primary/bank.replay primary/zero.words rejected.scores
refuse truncated 'truncated or unreadable input' \
    ./termination-cuda replay primary/truncated.replay primary/zero.words rejected.scores
refuse train_tape 'wrong tape kind for this command' \
    ./termination-cuda replay primary/finite.train primary/zero.words rejected.scores
refuse score_tape 'wrong tape kind for this command' \
    ./termination-cuda train primary/finite.replay 1 actual rejected.words rejected.json
test ! -e rejected.scores && test ! -e rejected.words && test ! -e rejected.json
sha256sum primary/* repeat/* *.stderr device.csv source.sha256 device-image.sha256 > results.sha256
sha256sum --check --strict results.sha256 > verify.sha256
