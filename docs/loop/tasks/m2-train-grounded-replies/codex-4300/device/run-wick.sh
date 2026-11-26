#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
base=/mnt/plexpool/uor-r4-corpus/runs
root="$base/20261126-codex-switch-source-competition/device"
reader="$base/20261126-codex-text-governor-links/learner/build-v2"
frozen="$base/20261125-codex-within-prompt-scope-margin/device-build-v1"
lock="$frozen/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "$0")" && pwd)

verify() (cd "$1" && sha256sum --quiet --strict -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
verify_inherited() {
    printf '%s  %s\n' \
        1a6483ba0f4c3feeae7c846b706e1674172c93f1692d49a90269531468146a88 "$reader/SHA256SUMS" \
        579ddc99d890b8c7a0c9be603bce73b76504db841ac64db81e6d4b5e865ace1e "$frozen/SHA256SUMS" \
        | sha256sum --quiet --strict -c -
    verify "$reader"
    verify "$frozen"
    sha256sum --quiet --strict -c "$reader/dependencies.sha256"
}
kernel_sass() {
    awk -v symbol="$1" '/Function : / { keep = index($0, symbol) != 0 } keep { print }' "$2"
}
valid_version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }

mode=${1:?build or finite}
version=${2:?new output version}
valid_version "$version"
case "$mode" in
build)
    test "$#" -eq 2
    verify_inherited
    mkdir -p "$root"
    test "$(findmnt -no FSTYPE --target "$root")" = zfs
    build="$root/build-$version"
    mkdir "$build" "$build/source"
    cp -- "$here/"{switch-competition.cu,fixture.rs,finite.jq,run-wick.sh,README.md} "$build/source"
    predecessor="$reader/source/learner.cu"
    test "$(grep -c '^int main(int argc, char \*\*argv) {$' "$predecessor")" -eq 1
    sed '/^int main(int argc, char \*\*argv) {$/,$d' "$predecessor" > "$build/source/textlinks-shared.cuh"
    sha256sum "$reader/SHA256SUMS" "$predecessor" "$frozen/SHA256SUMS" "$lock" \
        "$frozen/source/codex-4278/fit/device.cu" > "$build/dependencies.sha256"
    cuda=/usr/local/cuda/bin
    binary="$build/switch-competition-sm89"
    "$cuda/nvcc" --version > "$build/nvcc.txt"
    "$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$build/source/switch-competition.cu" -o "$binary" > "$build/build.log" 2>&1
    "$cuda/cuobjdump" --list-elf "$binary" > "$build/gpu-elf.txt"
    "$cuda/cuobjdump" --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    "$cuda/cuobjdump" --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|F2F|F2I|I2F|DADD|DFMA|DMUL|DSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in competition device code' >&2
        exit 1
    fi
    for kernel in _ZN9textlinks11attachments _ZN9textlinks7predict; do
        kernel_sass "$kernel" "$build/device.sass" > "$build/$kernel.sass"
        test -s "$build/$kernel.sass"
        cmp <(kernel_sass "$kernel" "$reader/device.sass") "$build/$kernel.sass"
    done
    rustc --version > "$build/rustc.txt"
    rustc --edition 2021 -D warnings "$build/source/fixture.rs" -o "$build/fixture" >> "$build/build.log" 2>&1
    seal "$build"
    ;;
finite|finite-inner)
    test "$#" -eq 3
    build_version=$3
    valid_version "$build_version"
    build="$root/build-$build_version"
    out="$root/finite-$version"
    verify_inherited
    verify "$build"
    sha256sum --quiet --strict -c "$build/dependencies.sha256"
    if [[ $mode == finite ]]; then
        exec bash "$lock" "$out-lease" bash "$build/source/run-wick.sh" finite-inner "$version" "$build_version"
    fi
    test -e "/proc/$$/fd/9"
    "$build/fixture" "$out"
    (
        cd "$out"
        original="$reader/textlinks-sm89"
        binary="$build/switch-competition-sm89"
        sha256sum fixture.words text.bin fixture.state requests.tsv fixture-text.txt > fixture-inputs.sha256
        "$original" learned fixture.words text.bin fixture.state original.jsonl
        for arm in first repeat; do
            "$binary" fixture.words text.bin fixture.state requests.tsv "$arm"
        done
        cmp original.jsonl first/native.jsonl
        diff -r first repeat
        sha256sum --quiet --strict -c fixture-inputs.sha256
        printf '1\t0\t2\t3\t4\n' > sign.tsv
        printf '3\t1\t0\t0\t1\n' > row.tsv
        printf '1\t1\t4\t0\t1\n' > word.tsv
        printf '1\t1\t0\t0\t8\n' > site.tsv
        printf '1\t1\t0\t0\t0\n' > duplicate-site.tsv
        printf '1 1 0 0 1\n' > spaces.tsv
        printf '1\t1\t0\t0\t1\textra\n' > trailing.tsv
        for bad in sign row word site duplicate-site spaces trailing; do
            if "$binary" fixture.words text.bin fixture.state "$bad.tsv" "unexpected-$bad" 2> "$bad.stderr"; then
                echo "Accepted malformed requests: $bad" >&2
                exit 1
            fi
            test ! -e "unexpected-$bad"
            grep -Fq 'invalid request at TSV line 1' "$bad.stderr"
        done
        jq -en --rawfile prompt fixture-text.txt --slurpfile native original.jsonl \
            --slurpfile rows first/requests.jsonl -f "$build/source/finite.jq" > checks.json
        sha256sum "$build/SHA256SUMS" "$reader/SHA256SUMS" "$frozen/SHA256SUMS" > dependencies.sha256
    )
    seal "$out"
    ;;
*) exit 64 ;;
esac
