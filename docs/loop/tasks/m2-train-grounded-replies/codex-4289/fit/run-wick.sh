#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
base=/mnt/plexpool/uor-r4-corpus/runs
root="$base/20261126-codex-far-transport-margin/fit"
parent="$base/20261126-codex-text-governor-links/learner/build-v2"
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
valid_version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
verify_dependencies() {
    printf '%s  %s\n' '1a6483ba0f4c3feeae7c846b706e1674172c93f1692d49a90269531468146a88' "$parent/SHA256SUMS" \
        | sha256sum --quiet --strict -c
    verify "$parent"
    verify "$frozen"
    sha256sum --quiet --strict -c "$parent/dependencies.sha256"
}

mode=${1:?build or finite}
version=${2:?new output version}
valid_version "$version"
case "$mode" in
build)
    build="$root/build-$version"
    binary="$build/transport-margin-sm89"
    verify_dependencies
    mkdir -p "$root"
    mkdir "$build"
    mkdir "$build/source"
    cp -- "$here/"{margin.cu,fixture.rs,fixture.jq,run-wick.sh,README.md} "$build/source"
    predecessor="$parent/source/learner.cu"
    test "$(grep -c '^int main(int argc, char \*\*argv) {$' "$predecessor")" -eq 1
    # Keep every operator and ABI definition verbatim. Exclude the program entry point.
    sed '/^int main(int argc, char \*\*argv) {$/,$d' "$predecessor" > "$build/source/textlinks-shared.cuh"
    sha256sum "$parent/SHA256SUMS" "$predecessor" "$frozen/SHA256SUMS" "$lock" \
        "$frozen/source/codex-4278/fit/device.cu" > "$build/dependencies.sha256"
    /usr/local/cuda/bin/nvcc --version > "$build/nvcc.txt"
    /usr/local/cuda/bin/nvcc --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$build/source/margin.cu" -o "$binary" > "$build/build.log" 2>&1
    /usr/local/cuda/bin/cuobjdump --list-elf "$binary" > "$build/gpu-elf.txt"
    /usr/local/cuda/bin/cuobjdump --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    /usr/local/cuda/bin/cuobjdump --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in transport fit device code' >&2
        exit 1
    fi
    rustc --version > "$build/rustc.txt"
    rustc --edition 2021 -D warnings "$build/source/fixture.rs" -o "$build/fixture" >> "$build/build.log" 2>&1
    seal "$build"
    ;;
finite|finite-inner)
    build_version=${3:?build version}
    valid_version "$build_version"
    build="$root/build-$build_version"
    binary="$build/transport-margin-sm89"
    out="$root/finite-$version"
    verify_dependencies
    verify "$build"
    sha256sum --quiet --strict -c "$build/dependencies.sha256"
    if [[ $mode == finite ]]; then
        bash "$lock" "$out-lease" bash "$build/source/run-wick.sh" finite-inner "$version" "$build_version"
    else
        test -e "/proc/$$/fd/9"
        "$build/fixture" "$out"
        (
            cd "$out"
            "$binary" diagnose fixture.bin fixture-labels.bin parent.state parent.jsonl
            "$binary" diagnose tie-only.bin tie-only-labels.bin parent.state tie-only.jsonl
            if "$binary" train contrastive tie-only.bin tie-only-labels.bin parent.state \
                unexpected.state unexpected.jsonl 2> tie-only.stderr; then
                echo 'Accepted a tie-only training premise' >&2
                exit 1
            fi
            test ! -e unexpected.state
            test ! -e unexpected.jsonl
            grep -Fq 'no admissible strictly losing edge' tie-only.stderr
            "$binary" train contrastive pocket.bin pocket-labels.bin parent.state \
                pocket.state pocket.jsonl pocket.final.state
            cmp pocket.state expect-pocket.state
            cmp pocket.final.state parent.state
            cmp <(dd if=repeated-site-source.txt bs=1 skip=6 count=4 status=none) \
                <(dd if=repeated-site-source.txt bs=1 skip=12 count=4 status=none)
            "$parent/textlinks-sm89" learned fixture.words fixture.bin parent.state parent-reader.jsonl
            for arm in contrastive positive; do
                for suffix in "" .repeat; do
                    "$binary" train "$arm" fixture.bin fixture-labels.bin parent.state \
                        "$arm$suffix.state" "$arm$suffix.jsonl" "$arm$suffix.final.state"
                done
                cmp "$arm.state" "expect-$arm.state"
                cmp "$arm.state" "$arm.final.state"
                cmp "$arm.state" "$arm.repeat.state"
                cmp "$arm.jsonl" "$arm.repeat.jsonl"
                cmp "$arm.final.state" "$arm.repeat.final.state"
                "$binary" diagnose fixture.bin fixture-labels.bin "$arm.state" "$arm.cold.jsonl"
                diff <(jq -c '[.row,.sign,.after,.candidates | .]' "$arm.jsonl" | jq -c '.[3] |= map(del(.before))') \
                    <(jq -c '[.row,.sign,.before,.candidates | .] | .[3] |= map(del(.before))' "$arm.cold.jsonl")
                "$parent/textlinks-sm89" learned fixture.words fixture.bin "$arm.state" "$arm-reader.jsonl"
                "$parent/textlinks-sm89" learned fixture.words fixture.bin "$arm.repeat.state" "$arm-reader.repeat.jsonl"
                cmp "$arm-reader.jsonl" "$arm-reader.repeat.jsonl"
            done
            jq -e -n -f "$build/source/fixture.jq" --slurpfile base parent.jsonl \
                --slurpfile contrast contrastive.jsonl --slurpfile positive positive.jsonl \
                --slurpfile pocket pocket.jsonl \
                --slurpfile base_reader parent-reader.jsonl --slurpfile contrast_reader contrastive-reader.jsonl \
                --slurpfile positive_reader positive-reader.jsonl > checks.json
            sha256sum "$build/SHA256SUMS" "$parent/SHA256SUMS" "$frozen/SHA256SUMS" > inputs.sha256
        )
        seal "$out"
    fi
    ;;
*) exit 64 ;;
esac
