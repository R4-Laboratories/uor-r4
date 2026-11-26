#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
base=/mnt/plexpool/uor-r4-corpus/runs
root="$base/20261126-codex-protected-transport/fit"
previous="$base/20261126-codex-far-transport-margin/fit"
margin="$previous/build-v3"
reader="$base/20261126-codex-text-governor-links/learner/build-v2"
frozen="$base/20261125-codex-within-prompt-scope-margin/device-build-v1"
lock="$frozen/source/codex-4281/fit/device-wick.sh"
fixture_words="$previous/finite-v4/fixture.words"
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
        925b91687a70740ca2b097c5a59f056c8701a503d68261e544c09ce1fbf4bd0e "$margin/SHA256SUMS" \
        1e618952d13e9745c7812139793fad210969ea320e5d473c3cc3cdac627a0a0b "$previous/finite-v4/SHA256SUMS" \
        1a6483ba0f4c3feeae7c846b706e1674172c93f1692d49a90269531468146a88 "$reader/SHA256SUMS" \
        579ddc99d890b8c7a0c9be603bce73b76504db841ac64db81e6d4b5e865ace1e "$frozen/SHA256SUMS" \
        | sha256sum --quiet --strict -c
    for dependency in "$margin" "$previous/finite-v4" "$reader" "$frozen"; do verify "$dependency"; done
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
    mkdir -p "$root"
    build="$root/build-$version"
    mkdir "$build" "$build/source"
    cp -- "$here/"{proposals.cu,fixture.rs,finite.jq,run-wick.sh,README.md} "$build/source"
    cp -- "$margin/source/textlinks-shared.cuh" "$build/source"
    test "$(grep -c '^int main(int argc, char \*\*argv) {$' "$margin/source/margin.cu")" -eq 1
    sed '/^int main(int argc, char \*\*argv) {$/,$d' "$margin/source/margin.cu" > "$build/source/margin-shared.cuh"
    sha256sum "$margin/SHA256SUMS" "$previous/finite-v4/SHA256SUMS" "$reader/SHA256SUMS" \
        "$frozen/SHA256SUMS" "$lock" > "$build/dependencies.sha256"
    binary="$build/transport-proposals-sm89"
    cuda=/usr/local/cuda/bin
    "$cuda/nvcc" --version > "$build/nvcc.txt"
    "$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$build/source/proposals.cu" -o "$binary" > "$build/build.log" 2>&1
    "$cuda/cuobjdump" --list-elf "$binary" > "$build/gpu-elf.txt"
    "$cuda/cuobjdump" --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    "$cuda/cuobjdump" --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in proposal device code' >&2
        exit 1
    fi
    for kernel in _ZN10margin_fit6enroll _ZN10margin_fit7measure _ZN10margin_fit6update \
                  _ZN9textlinks11attachments _ZN9textlinks7predict; do
        kernel_sass "$kernel" "$build/device.sass" > "$build/$kernel.sass"
        test -s "$build/$kernel.sass"
        cmp <(kernel_sass "$kernel" "$margin/device.sass") "$build/$kernel.sass"
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
        exec bash "$lock" "$out-lease" bash "$build/source/run-wick.sh" finite-inner "$version" "$build_version"
    fi
    test -e "/proc/$$/fd/9"
    "$build/fixture" "$out"
    (
        cd "$out"
        proposer="$build/transport-proposals-sm89"
        original_rank="$reader/textlinks-sm89"
        original_fit="$margin/transport-margin-sm89"
        args=(--fixture 2 "$fixture_words" text.bin labels.bin parent.state)
        if "$proposer" "${args[@]}" parent.state parent.state bad-premise 2> bad-premise.stderr; then
            echo 'Accepted wrong first simultaneous checkpoint' >&2
            exit 1
        fi
        test ! -e bad-premise
        grep -Fq 'physical origin update disagrees' bad-premise.stderr
        for run in first repeated; do
            "$proposer" "${args[@]}" expected-step-1.state parent.state "$run" 2> "$run.stderr"
        done
        diff -r first repeated
        "$proposer" "${args[@]}" expected-step-1.state first/5.state second 2> second.stderr
        cmp first/menu.jsonl second/menu.jsonl
        cmp parent.state first/base.state
        cmp first/5.state second/base.state
        for id in {0..5}; do cmp "expected-$id.state" "first/$id.state"; done
        for id in 2 3; do cmp second/base.state "second/$id.state"; done
        cmp expected-saturated-center.state second/5.state
        mkdir official
        for run in first second; do
            cmp expected-step-1.state "$run/step1-replayed.state"
            for id in base {0..5}; do
                stem="official/$run-$id"
                "$original_rank" learned "$fixture_words" text.bin "$run/$id.state" "$stem.reader.jsonl" 2> "$stem.reader.stderr"
                cmp "$stem.reader.jsonl" "$run/$id.reader.jsonl"
                "$original_fit" diagnose text.bin labels.bin "$run/$id.state" "$stem.signed.jsonl" 2> "$stem.signed.stderr"
                cmp <(jq -c '{row,sign,governor_word,gold_site,eligible_for_fit,eligible_rivals,before}' "$stem.signed.jsonl") \
                    "$run/$id.edges.jsonl"
            done
        done
        jq -en -f "$build/source/finite.jq" --slurpfile menu first/menu.jsonl \
            --slurpfile summary first/menu-summary.json --slurpfile first first/scores.jsonl \
            --slurpfile second second/scores.jsonl --slurpfile parent first/base.reader.jsonl \
            --slurpfile safe first/0.reader.jsonl --slurpfile harmful first/1.reader.jsonl > checks.json
        sha256sum "$build/SHA256SUMS" "$margin/SHA256SUMS" "$reader/SHA256SUMS" > inputs.sha256
    )
    seal "$out"
    ;;
*) exit 64 ;;
esac
