#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
base=/mnt/plexpool/uor-r4-corpus/runs
root="$base/20261126-codex-mixed-transport/fit"
previous="$base/20261126-codex-far-transport-margin/fit"
released="$previous/build-v3"
reader="$base/20261126-codex-text-governor-links/learner/build-v2"
frozen="$base/20261125-codex-within-prompt-scope-margin/device-build-v1"
lock="$frozen/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "$0")" && pwd)
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0

verify() (cd "$1" && sha256sum --quiet --strict -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
verify_inherited() {
    printf '%s  %s\n' \
        925b91687a70740ca2b097c5a59f056c8701a503d68261e544c09ce1fbf4bd0e "$released/SHA256SUMS" \
        1e618952d13e9745c7812139793fad210969ea320e5d473c3cc3cdac627a0a0b "$previous/finite-v4/SHA256SUMS" \
        1a6483ba0f4c3feeae7c846b706e1674172c93f1692d49a90269531468146a88 "$reader/SHA256SUMS" \
        579ddc99d890b8c7a0c9be603bce73b76504db841ac64db81e6d4b5e865ace1e "$frozen/SHA256SUMS" \
        | sha256sum --quiet --strict -c
    for dependency in "$released" "$previous/finite-v4" "$reader" "$frozen"; do
        verify "$dependency"
    done
}
valid_version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
kernel_sass() {
    awk -v symbol="_ZN10margin_fit$1" '/Function : / { keep = index($0, symbol) != 0 } keep { print }' "$2"
}

mode=${1:?build or finite}
version=${2:?new output version}
valid_version "$version"
case "$mode" in
build)
    test "$#" -eq 2
    verify_inherited
    mkdir -p "$root"
    build="$root/build-$version"
    mkdir "$build"
    mkdir "$build/source"
    cp -- "$here/"{mixed.cu,observe.cuh,margin-capture.patch,fixture.rs,finite.jq,run-wick.sh,README.md} "$build/source"
    cp -- "$released/source/"{margin.cu,textlinks-shared.cuh} "$build/source"
    patch --fuzz=0 -o "$build/source/margin-observed.cu" "$build/source/margin.cu" \
        "$build/source/margin-capture.patch" > "$build/patch.log"
    test "$(grep -c '^int main(int argc, char \*\*argv) {$' "$build/source/margin-observed.cu")" -eq 1
    sed '/^int main(int argc, char \*\*argv) {$/,$d' "$build/source/margin-observed.cu" > "$build/source/margin-capture.cuh"
    sha256sum "$released/SHA256SUMS" "$previous/finite-v4/SHA256SUMS" "$reader/SHA256SUMS" \
        "$frozen/SHA256SUMS" "$lock" > "$build/dependencies.sha256"
    binary="$build/transport-mixed-sm89"
    /usr/local/cuda/bin/nvcc --version > "$build/nvcc.txt"
    /usr/local/cuda/bin/nvcc --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$build/source/mixed.cu" -o "$binary" > "$build/build.log" 2>&1
    /usr/local/cuda/bin/cuobjdump --list-elf "$binary" > "$build/gpu-elf.txt"
    /usr/local/cuda/bin/cuobjdump --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    /usr/local/cuda/bin/cuobjdump --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in transport fit device code' >&2
        exit 1
    fi
    for kernel in 6enroll 7measure 6update; do
        section="$build/$kernel.sass"
        kernel_sass "$kernel" "$build/device.sass" > "$section"
        test -s "$section"
        cmp <(kernel_sass "$kernel" "$released/device.sass") "$section"
    done
    rustc --version > "$build/rustc.txt"
    rustc --edition 2021 -D warnings "$build/source/fixture.rs" -o "$build/fixture" >> "$build/build.log" 2>&1
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
        bash "$lock" "$out-lease" bash "$build/source/run-wick.sh" finite-inner "$version" "$build_version"
        exit
    fi
    test -e "/proc/$$/fd/9"
    "$build/fixture" "$out"
    (
        cd "$out"
        binary="$build/transport-mixed-sm89"
        teacher="$released/transport-margin-sm89"
        blind="$reader/textlinks-sm89"
        words="$previous/finite-v4/fixture.words"
        mkdir cold
        for invalid in short malformed; do
            if "$binary" text.bin labels.bin parent.state "$invalid.txt" "unexpected-$invalid" 2> "$invalid.stderr"; then
                echo 'Accepted malformed distance metadata' >&2
                exit 1
            fi
            test ! -e "unexpected-$invalid"
            grep -q '^transport-mixed-sm89: distance' "$invalid.stderr"
        done
        "$teacher" train contrastive text.bin labels.bin parent.state legacy.state legacy.jsonl legacy-final.state \
            2> legacy.stderr
        for run in all-far mixed mixed-repeat; do
            metadata=${run%-repeat}
            "$binary" text.bin labels.bin parent.state "$metadata.txt" "$run" 2> "$run.stderr"
            cmp legacy.stderr "$run.stderr"
            cmp legacy.jsonl "$run/trace.jsonl"
            chosen=$(jq -er .chosen_step "$run/trace.jsonl" | sort -u)
            cmp legacy.state "$run/step-$chosen.state"
            cmp legacy-final.state "$run/step-8.state"
            cmp parent.state "$run/step-0.state"
            cmp expected-step-1.state "$run/step-1.state"
        done
        diff -r mixed mixed-repeat
        for step in {0..8}; do
            candidate="mixed/step-$step.state"
            cmp "$candidate" "all-far/step-$step.state"
            # The sole mutable center in this fixture occupies bytes [6160,6224).
            # The other sign is all strict wins, so its active bank remains byte-identical as well.
            cmp -n 6160 parent.state "$candidate"
            cmp -i 6224 parent.state "$candidate"
            "$teacher" diagnose text.bin labels.bin "$candidate" "cold/step-$step.jsonl" 2> "cold/step-$step.stderr"
            jq -c --argjson step "$step" '[.row,.sign,.history[$step]]' legacy.jsonl > "cold/step-$step.expected.jsonl"
            jq -c '[.row,.sign,.before]' "cold/step-$step.jsonl" > "cold/step-$step.actual.jsonl"
            cmp "cold/step-$step.expected.jsonl" "cold/step-$step.actual.jsonl"
            "$blind" learned "$words" text.bin "$candidate" "cold/step-$step.reader.jsonl"
            "$blind" learned "$words" text.bin "mixed-repeat/step-$step.state" "cold/step-$step.reader-repeat.jsonl"
            cmp "cold/step-$step.reader.jsonl" "cold/step-$step.reader-repeat.jsonl"
        done
        jq -en -f "$build/source/finite.jq" --slurpfile mixed mixed/votes.jsonl \
            --slurpfile all all-far/votes.jsonl --slurpfile summary mixed/vote-summary.jsonl \
            --slurpfile quality mixed/quality.jsonl --slurpfile events mixed/trace.jsonl > checks.json
        sha256sum "$build/SHA256SUMS" "$released/SHA256SUMS" "$previous/finite-v4/SHA256SUMS" \
            "$reader/SHA256SUMS" > inputs.sha256
    )
    seal "$out"
    ;;
*) exit 64 ;;
esac
