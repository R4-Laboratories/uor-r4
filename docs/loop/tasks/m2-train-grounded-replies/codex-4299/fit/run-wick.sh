#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-fixed-assignment-median/fit"
previous="$runs/20261126-codex-all-outranker-interval/fit/build-v2"
margin="$runs/20261126-codex-far-transport-margin/fit/build-v3"
frozen="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1"
lock="$frozen/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root/"* ]]

verify() (cd "$1" && sha256sum --quiet --strict -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
verify_inherited() {
    printf '%s  %s\n' \
        acdfcdcb2864caa9c225bdb494e60ab8caf1e079099f65a9023cf526cb92f1bc "$previous/SHA256SUMS" \
        925b91687a70740ca2b097c5a59f056c8701a503d68261e544c09ce1fbf4bd0e "$margin/SHA256SUMS" \
        579ddc99d890b8c7a0c9be603bce73b76504db841ac64db81e6d4b5e865ace1e "$frozen/SHA256SUMS" \
        405c949c10a91ff33c31504a7ec6094d3688b11d072ad2f6973b4e284a2c8d84 "$lock" \
        | sha256sum --quiet --strict -c
    for dependency in "$previous" "$margin" "$frozen"; do
        verify "$dependency"
    done
}
valid_version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
kernel_sass() {
    awk -v symbol="$1" '/Function : / { keep = index($0, symbol) != 0 } keep { print }' "$2"
}

mode=${1:?build or finite}
version=${2:?new output version}
valid_version "$version"
case "$mode" in
build)
    test "$#" -eq 2
    verify_inherited
    build="$root/build-$version"
    mkdir "$build" "$build/source"
    cp -- "$here/"{median-fit.cu,fixture.rs,finite.jq,run-wick.sh,README.md} "$build/source"
    cp -- "$previous/source/"{textlinks-shared.cuh,margin-shared.cuh} "$build/source"
    sha256sum "$previous/SHA256SUMS" "$margin/SHA256SUMS" "$frozen/SHA256SUMS" "$lock" > "$build/dependencies.sha256"
    binary="$build/median-fit-sm89"
    cuda=/usr/local/cuda/bin
    "$cuda/nvcc" --version > "$build/nvcc.txt"
    "$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$build/source/median-fit.cu" -o "$binary" > "$build/build.log" 2>&1
    "$cuda/cuobjdump" --list-elf "$binary" > "$build/gpu-elf.txt"
    "$cuda/cuobjdump" --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    "$cuda/cuobjdump" --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in median fit device code' >&2
        exit 1
    fi
    for kernel in _ZN10margin_fit6enroll _ZN10margin_fit7measure \
                  _ZN9textlinks11attachments _ZN9textlinks7predict; do
        kernel_sass "$kernel" "$build/device.sass" > "$build/$kernel.sass"
        cmp "$previous/$kernel.sass" "$build/$kernel.sass"
    done
    for kernel in 6gather 6assign 13lower_medians 13apply_medians 12score_states; do
        symbol="_ZN10median_fit$kernel"
        kernel_sass "$symbol" "$build/device.sass" > "$build/$symbol.sass"
        test -s "$build/$symbol.sass"
    done
    rustc --version > "$build/rustc.txt"
    rustc --edition 2021 -D warnings "$build/source/fixture.rs" -o "$build/median-fixture" >> "$build/build.log" 2>&1
    seal "$build"
    ;;
finite|finite-inner)
    test "$#" -eq 3
    build_version=${3:?build version}
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
    "$build/median-fixture" "$out"
    (
        cd "$out"
        fitter="$build/median-fit-sm89"
        "$fitter" --fixture endpoint.bin ordered.bin labels.bin protected.state fit
        "$fitter" --fixture endpoint.bin ordered.bin labels.bin protected.state repeated
        diff -qr fit repeated
        if "$fitter" --fixture endpoint.bin changed-short.bin labels.bin protected.state invalid 2>short-path-error.txt; then
            echo 'The fitter accepted a changed short-gap transport' >&2
            exit 1
        fi
        grep -Fx 'median-fit-sm89: unmatched geometry packets' short-path-error.txt
        test ! -e invalid
        for arm in endpoint ordered; do
            cmp protected.state "fit/$arm/base.state"
            cmp "$arm-full.state" "fit/$arm/full.state"
            for i in 00 01 08; do
                cmp "$arm-slot-$i.state" "fit/$arm/slot-$i.state"
            done
            for i in 02 03 04 05 06 07 09 10 11 12 13 14 15; do
                cmp protected.state "fit/$arm/slot-$i.state"
            done
        done
        "$margin/transport-margin-sm89" diagnose endpoint.bin labels.bin protected.state archived-assignment.jsonl
        sha256sum "$build/SHA256SUMS" "$margin/SHA256SUMS" "$lock" > inputs.sha256
        jq -en -f "$build/source/finite.jq" \
            --slurpfile assignments fit/assignments.jsonl --slurpfile archived archived-assignment.jsonl \
            --slurpfile endpoint_points fit/endpoint/coordinates.jsonl --slurpfile ordered_points fit/ordered/coordinates.jsonl \
            --slurpfile endpoint_medians fit/endpoint/medians.jsonl --slurpfile ordered_medians fit/ordered/medians.jsonl \
            --slurpfile endpoint_scores fit/endpoint/scores.jsonl --slurpfile ordered_scores fit/ordered/scores.jsonl > checks.json
    )
    seal "$out"
    ;;
*) exit 64 ;;
esac
