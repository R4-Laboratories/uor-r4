#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-all-outranker-interval/fit"
previous="$runs/20261126-codex-far-transport-margin/fit"
margin="$previous/build-v3"
reader="$runs/20261126-codex-text-governor-links/learner/build-v2"
frozen="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1"
proposals="$runs/20261126-codex-protected-transport/fit/build-v1"
lock="$frozen/source/codex-4281/fit/device-wick.sh"
fixture_words="$previous/finite-v4/fixture.words"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root/"* ]]

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
        39690592125983e5a459f491b26d452ed3bc24fe51a4d4b47297492bb0ff3da9 "$proposals/SHA256SUMS" \
        | sha256sum --quiet --strict -c
    for dependency in "$margin" "$previous/finite-v4" "$reader" "$frozen" "$proposals"; do
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
    cp -- "$here/"{bridge-fit.cu,fixture.rs,finite.jq,run-wick.sh,README.md} "$build/source"
    cp -- "$margin/source/textlinks-shared.cuh" "$build/source"
    test "$(grep -c '^int main(int argc, char \*\*argv) {$' "$margin/source/margin.cu")" -eq 1
    sed '/^int main(int argc, char \*\*argv) {$/,$d' "$margin/source/margin.cu" > "$build/source/margin-shared.cuh"
    sha256sum "$margin/SHA256SUMS" "$previous/finite-v4/SHA256SUMS" "$reader/SHA256SUMS" \
        "$frozen/SHA256SUMS" "$proposals/SHA256SUMS" "$fixture_words" "$lock" > "$build/dependencies.sha256"
    binary="$build/bridge-fit-sm89"
    cuda=/usr/local/cuda/bin
    "$cuda/nvcc" --version > "$build/nvcc.txt"
    "$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$build/source/bridge-fit.cu" -o "$binary" > "$build/build.log" 2>&1
    "$cuda/cuobjdump" --list-elf "$binary" > "$build/gpu-elf.txt"
    "$cuda/cuobjdump" --list-ptx "$binary" > "$build/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$build/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$build/gpu-ptx.txt"
    "$cuda/cuobjdump" --dump-sass "$binary" > "$build/device.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$build/device.sass"; then
        echo 'Forbidden instruction in bridge fit device code' >&2
        exit 1
    fi
    for kernel in _ZN10margin_fit6enroll _ZN10margin_fit7measure \
                  _ZN9textlinks11attachments _ZN9textlinks7predict; do
        kernel_sass "$kernel" "$build/device.sass" > "$build/$kernel.sass"
        test -s "$build/$kernel.sass"
        cmp "$proposals/$kernel.sass" "$build/$kernel.sass"
    done
    rustc --version > "$build/rustc.txt"
    rustc --edition 2021 -D warnings "$build/source/fixture.rs" -o "$build/bridge-fixture" >> "$build/build.log" 2>&1
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
    "$build/bridge-fixture" "$out"
    (
        cd "$out"
        proposer="$build/bridge-fit-sm89"
        original_reader="$reader/textlinks-sm89"
        original_signed="$margin/transport-margin-sm89"
        for arm in strongest all; do
            "$proposer" --fixture 2 "$arm" "$fixture_words" text.bin labels.bin parent.state "$arm"
        done
        "$proposer" --fixture 2 ordered-all "$fixture_words" text.bin labels.bin parent.state ordered-all
        diff -qr --exclude complete.json all ordered-all
        "$proposer" --fixture 2 all "$fixture_words" text.bin labels.bin parent.state repeated
        diff -qr all repeated
        "$proposer" --fixture 2 strongest "$fixture_words" text.bin labels.bin strongest/1.state remine
        "$proposer" --fixture 2 all "$fixture_words" text.bin labels.bin all/1.state zero
        "$proposer" --fixture 2 all "$fixture_words" boundary-text.bin labels.bin boundary-parent.state boundary
        cmp parent.state all/base.state
        cmp parent.state all/28.state
        cmp expected-strongest-step2.state strongest/1.state
        cmp expected-all-step2.state all/1.state
        cmp expected-all-step2.state all/65.state
        cmp expected-remine-step2.state remine/1.state
        cmp expected-saturated.state boundary/32.state
        cmp expected-saturated.state boundary/35.state
        mkdir official
        for sample in strongest/base strongest/1 all/1 remine/base remine/1 zero/base boundary/35; do
            text=text.bin
            if [[ $sample == boundary/* ]]; then text=boundary-text.bin; fi
            stem="official/${sample/\//-}"
            "$original_reader" learned "$fixture_words" "$text" "$sample.state" "$stem.reader.jsonl"
            cmp "$stem.reader.jsonl" "$sample.reader.jsonl"
            "$original_signed" diagnose "$text" labels.bin "$sample.state" "$stem.signed.jsonl"
            cmp <(jq -c '{row,sign,governor_word,gold_site,eligible_for_fit,eligible_rivals,before,
                           candidates:[.candidates[]|[.site,.before.cost,.before.center]]}' "$stem.signed.jsonl") \
                <(jq -c '{row,sign,governor_word,gold_site,eligible_for_fit,eligible_rivals,before,candidates}' \
                        "$sample.edges.jsonl")
        done
        "$original_reader" correct "$fixture_words" text.bin aid.bin official/correct.reader.jsonl
        "$original_reader" exchanged "$fixture_words" text.bin aid.bin official/exchanged.reader.jsonl
        jq -en -f "$build/source/finite.jq" --slurpfile menu all/menu.jsonl \
            --slurpfile single strongest/scores.jsonl --slurpfile all all/scores.jsonl \
            --slurpfile base_edges all/base.edges.jsonl \
            --slurpfile single_votes strongest/directions.jsonl --slurpfile all_votes all/directions.jsonl \
            --slurpfile base all/base.reader.jsonl --slurpfile single_step strongest/1.reader.jsonl \
            --slurpfile all_step all/1.reader.jsonl --slurpfile correct official/correct.reader.jsonl \
            --slurpfile exchanged official/exchanged.reader.jsonl --slurpfile remine remine/scores.jsonl \
            --slurpfile remine_votes remine/directions.jsonl --slurpfile remine_edges remine/base.edges.jsonl \
            --slurpfile zero_votes zero/directions.jsonl --slurpfile zero zero/scores.jsonl \
            --slurpfile boundary boundary/scores.jsonl > checks.json
        sha256sum "$build/SHA256SUMS" "$margin/SHA256SUMS" "$reader/SHA256SUMS" "$lock" > inputs.sha256
    )
    seal "$out"
    ;;
*) exit 64 ;;
esac
