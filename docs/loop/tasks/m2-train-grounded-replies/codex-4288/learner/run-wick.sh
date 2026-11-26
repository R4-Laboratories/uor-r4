#!/usr/bin/env bash
# This entry point can build and execute synthetic fixtures under the Wick lease.
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
base=/mnt/plexpool/uor-r4-corpus/runs
root="$base/20261126-codex-text-governor-links/learner"
pinned="$base/20261125-codex-within-prompt-scope-margin/device-build-v1"
lock="$pinned/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "$0")" && pwd)
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0

verify() (cd "$1" && sha256sum --quiet --strict -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
valid_version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
build_paths() {
    valid_version "$1"
    build="$root/build-$1"
    binary="$build/textlinks-sm89"
}
verify_build() {
    verify "$pinned"
    verify "$build"
    sha256sum --quiet --strict -c "$build/dependencies.sha256"
}

mode=${1:?build or finite}
version=${2:?new output version}
valid_version "$version"
case "$mode" in
build)
    build_paths "$version"
    verify "$pinned"
    mkdir -p "$root"
    mkdir "$build"
    mkdir "$build/source"
    cp -- "$here/"{learner.cu,fixture.rs,fixture.jq,run-wick.sh,README.md} "$build/source"
    /usr/local/cuda/bin/nvcc --version > "$build/nvcc.txt"
    /usr/local/cuda/bin/nvcc --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$pinned/source" "$build/source/learner.cu" -o "$binary" > "$build/build.log" 2>&1
    /usr/local/cuda/bin/cuobjdump --list-elf "$binary" > "$build/gpu-elf.txt"
    /usr/local/cuda/bin/cuobjdump --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q 'sm_89' "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    /usr/local/cuda/bin/cuobjdump --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in learner device code' >&2
        exit 1
    fi
    command -v rustc > "$build/rust-path.txt"
    rustc --version > "$build/rustc.txt"
    rustc --edition 2021 -D warnings "$build/source/fixture.rs" -o "$build/fixture" >> "$build/build.log" 2>&1
    sha256sum "$pinned/SHA256SUMS" "$lock" "$pinned/source/codex-4278/fit/device.cu" \
        > "$build/dependencies.sha256"
    seal "$build"
    ;;
finite|finite-inner)
    build_paths "${3:?build version}"
    verify_build
    out="$root/finite-$version"
    if [[ $mode == finite ]]; then
        bash "$lock" "$out-lease" bash "$build/source/run-wick.sh" finite-inner "$version" "$3"
    else
        test -e "/proc/$$/fd/9"
        "$build/fixture" "$out"
        (
            cd "$out"
            gpu() { "$binary" "$1" fixture.words "${@:2}"; }
            for variant in trained swapped endpoints; do
                subcommand=train
                if [[ $variant != trained ]]; then subcommand="train-$variant"; fi
                gpu "$subcommand" train.bin train-labels.bin "$variant.state" "$variant.live.jsonl"
                cmp "expect-$variant" "$variant.state"
                gpu learned train.bin "$variant.state" "$variant.cold.jsonl"
                cmp "$variant.live.jsonl" "$variant.cold.jsonl"
                gpu learned predict.bin "$variant.state" "$variant.jsonl"
                gpu learned predict.bin "$variant.state" "$variant.repeat.jsonl"
                cmp "$variant.jsonl" "$variant.repeat.jsonl"
            done
            cmp -s trained.state swapped.state && exit 1
            cmp -s trained.state endpoints.state && exit 1
            gpu native predict.bin native.jsonl
            gpu unfitted predict.bin unfitted.jsonl
            gpu governor_gap predict.bin trained.state governor_gap.jsonl
            gpu correct predict.bin aid.bin correct.jsonl
            gpu correct predict.bin aid-permuted.bin correct.permuted.jsonl
            cmp correct.jsonl correct.permuted.jsonl
            gpu correct predict.bin aid-wrong-edge.bin wrong_edge.jsonl
            gpu correct predict.bin aid-wrong-key.bin wrong_key.jsonl
            gpu exchanged predict.bin aid.bin exchanged.jsonl
            gpu exchanged predict.bin aid-permuted.bin exchanged.permuted.jsonl
            cmp exchanged.jsonl exchanged.permuted.jsonl
            for negative in dangling collision; do
                if gpu correct predict.bin "aid-$negative.bin" "$negative.jsonl" 2> "$negative.stderr"; then
                    echo "Accepted invalid intervention $negative" >&2
                    exit 1
                fi
                test ! -e "$negative.jsonl"
            done
            grep -Fq 'dangling' dangling.stderr
            grep -Fq 'colliding' collision.stderr
            cp predict.bin trailing.bin
            printf '\x01' >> trailing.bin
            if gpu learned trailing.bin trained.state trailing.jsonl 2> trailing.stderr; then
                echo 'Accepted a trailing packet byte' >&2
                exit 1
            fi
            grep -Fq 'trailing packet data' trailing.stderr
            jq -e -n -f "$build/source/fixture.jq" \
                --slurpfile trained trained.jsonl --slurpfile swapped swapped.jsonl \
                --slurpfile endpoints endpoints.jsonl --slurpfile native native.jsonl \
                --slurpfile unfitted unfitted.jsonl --slurpfile gap governor_gap.jsonl \
                --slurpfile correct correct.jsonl --slurpfile exchanged exchanged.jsonl \
                --slurpfile wrong_edge wrong_edge.jsonl --slurpfile wrong_key wrong_key.jsonl \
                > checks.json
            sha256sum "$build/SHA256SUMS" "$pinned/SHA256SUMS" > inputs.sha256
        )
        seal "$out"
    fi
    ;;
*) exit 64 ;;
esac
