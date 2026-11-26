#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 1 && $1 =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-protected-transport/fit"
stage="$root/vast-stage-$1"
[[ $(findmnt -no FSTYPE --target "$root") == zfs ]]
wick="$root/build-v1"
old="$runs/20261126-codex-far-transport-margin/fit/build-v3"
old_fixture="$runs/20261126-codex-far-transport-margin/fit/finite-v4"
frozen="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1"
rank="$runs/20261126-codex-text-governor-links/learner/build-v2"
mixed="$runs/20261126-codex-mixed-transport"
train="$mixed/panel/panel-v1/mixed"
baseline="$mixed/measurement-v1/train/mixed"
banks="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"

verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum > .pending
    mv .pending SHA256SUMS
)
kernel_sass() {
    awk -v symbol="$1" '/Function : / { keep = index($0, symbol) != 0 } keep { print }' "$2"
}
make_provider_source() {
    awk '
        $0 == "void require_lease() {" {
            ++found; skipping = 1; print "#include \"vast-admission.cuh\""; next
        }
        skipping && $0 == "}" { skipping = 0; next }
        !skipping { print }
        END { if (found != 1 || skipping) exit 1 }
    ' "$1" > "$2"
}

pin 39690592125983e5a459f491b26d452ed3bc24fe51a4d4b47297492bb0ff3da9 "$wick/SHA256SUMS"
pin 925b91687a70740ca2b097c5a59f056c8701a503d68261e544c09ce1fbf4bd0e "$old/SHA256SUMS"
pin 1e618952d13e9745c7812139793fad210969ea320e5d473c3cc3cdac627a0a0b "$old_fixture/SHA256SUMS"
pin 579ddc99d890b8c7a0c9be603bce73b76504db841ac64db81e6d4b5e865ace1e "$frozen/SHA256SUMS"
pin 1a6483ba0f4c3feeae7c846b706e1674172c93f1692d49a90269531468146a88 "$rank/SHA256SUMS"
pin c4eb44e57970bf0695ecc0e4e358bd61c96c4c84cfab2f48c0f0e745889d0dc5 "$baseline/first/step-0.state"
pin 668bc7cd01eff93a788e98d5e8b543add426cf0a1ae1402b72512d125e2923ca "$baseline/first/step-1.state"
for dependency in "$wick" "$old" "$old_fixture" "$frozen" "$rank" "$mixed/measurement-v1/train"; do
    verify "$dependency"
done
(
    cd "$train/.."
    awk '$2 == "./mixed/blind/text.bin" || $2 == "./mixed/links.bin" || $2 == "./mixed/distance.txt" {
        print; ++found
    } END { if (found != 3) exit 1 }' SHA256SUMS | sha256sum --strict --quiet -c -
)

mkdir "$stage"
payload="$stage/payload"
build_source="$stage/source"
mkdir "$payload" "$payload/build" "$payload/build/diagnostic" "$payload/build/source" \
    "$payload/reader" "$payload/train" "$build_source"
cp "$here/"{stage-wick.sh,run-pod.sh,vast-admission.cuh,README.md} "$build_source/"
cp "$wick/source/"{proposals.cu,textlinks-shared.cuh} "$build_source/"
make_provider_source "$wick/source/margin-shared.cuh" "$build_source/margin-shared.cuh"
make_provider_source "$old/source/margin.cu" "$build_source/margin-provider.cu"
cp "$here/run-pod.sh" "$payload/"
cp "$rank/"{textlinks-sm89,SHA256SUMS} "$payload/reader/"
cp "$wick/source/finite.jq" "$payload/build/source/"
"$wick/fixture" "$payload/fixture"
cp "$old_fixture/fixture.words" "$payload/fixture/"
awk '
    $0 == "    \"$build/fixture\" \"$out\"" { copying = 1; ++start; next }
    copying && $0 == "    seal \"$out\"" { copying = 0; ++finish; next }
    copying { print }
    END { if (start != 1 || finish != 1 || copying) exit 1 }
' "$wick/source/run-wick.sh" > "$payload/build/finite-calls.sh"

for kind in proposals diagnostic; do
    if [[ $kind == proposals ]]; then
        source_file="$build_source/proposals.cu"; reference="$wick/device.sass"
        target="$payload/build"; binary="$target/transport-proposals-sm89"
        extra=(_ZN9proposals10apply_move _ZN9proposals13summarize_far)
    else
        source_file="$build_source/margin-provider.cu"; reference="$old/device.sass"
        target="$payload/build/diagnostic"; binary="$target/transport-margin-sm89"
        extra=()
    fi
    cuda=/usr/local/cuda/bin
    "$cuda/nvcc" --version > "$target/nvcc.txt"
    cmp "$wick/nvcc.txt" "$target/nvcc.txt"
    "$cuda/nvcc" --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$frozen/source" "$source_file" -o "$binary" > "$target/build.log" 2>&1
    "$cuda/cuobjdump" --list-elf "$binary" > "$target/gpu-elf.txt"
    "$cuda/cuobjdump" --list-ptx "$binary" > "$target/gpu-ptx.txt" 2>&1 || true
    grep -q sm_89 "$target/gpu-elf.txt"
    grep -Fq 'No PTX file found to extract' "$target/gpu-ptx.txt"
    readelf --version-info "$binary" > "$target/elf-versions.txt"
    "$cuda/cuobjdump" --dump-sass "$binary" > "$stage/$kind.sass"
    if grep -E '(^|[[:space:]])(FADD|FFMA|FMUL|FSEL|FSET|FSETP|HFMA|HFMA2|HMUL2|HMMA|IMMA|MMA|MUFU)([[:space:].]|$)' "$stage/$kind.sass"; then
        exit 1
    fi
    for symbol in _ZN10margin_fit6enroll _ZN10margin_fit7measure _ZN10margin_fit6update \
                  _ZN9textlinks11attachments _ZN9textlinks7predict "${extra[@]}"; do
        expected="$stage/$kind-$symbol.wick.sass"; actual="$stage/$kind-$symbol.vast.sass"
        kernel_sass "$symbol" "$reference" > "$expected"
        kernel_sass "$symbol" "$stage/$kind.sass" > "$actual"
        test -s "$expected"; cmp "$expected" "$actual"
        printf '%s  %s\n' "$(sha256sum "$actual" | cut -d' ' -f1)" "$symbol"
    done > "$target/identical-wick-kernels.txt"
done
readelf --version-info "$payload/reader/textlinks-sm89" > "$payload/reader/elf-versions.txt"
if (exec 20>&-; CUDA_VISIBLE_DEVICES=0 "$payload/build/transport-proposals-sm89" --fixture 2 \
        "$payload/fixture/fixture.words" "$payload/fixture/text.bin" "$payload/fixture/labels.bin" \
        "$payload/fixture/parent.state" "$payload/fixture/expected-step-1.state" "$payload/fixture/parent.state" \
        "$stage/unexpected-unleased") > "$stage/unleased.stdout" 2> "$stage/unleased.stderr"; then
    echo 'Unleased provider binary accepted a host staging call' >&2
    exit 1
fi
[[ ! -e $stage/unexpected-unleased ]]
grep -Fq 'run under the inherited uor-pod GPU 0 lease on descriptor 20' "$stage/unleased.stderr"
seal "$payload/build/diagnostic"
seal "$payload/build"
# Check the sealed locations expected by the extracted finite invocations.
for required in build/transport-proposals-sm89 build/diagnostic/transport-margin-sm89 \
                build/source/finite.jq build/SHA256SUMS build/diagnostic/SHA256SUMS \
                reader/textlinks-sm89 reader/SHA256SUMS fixture/fixture.words; do
    test -s "$payload/$required"
done

cp "$train/blind/text.bin" "$train/links.bin" "$train/distance.txt" "$payload/train/"
cp "$banks" "$payload/train/"
cp "$baseline/first/step-0.state" "$baseline/first/step-1.state" "$payload/train/"
cp "$baseline/step-0.raw.jsonl" "$baseline/initial.jsonl" "$payload/train/"
sha256sum "$wick/SHA256SUMS" "$old/SHA256SUMS" "$rank/SHA256SUMS" "$train/../SHA256SUMS" \
    "$mixed/measurement-v1/train/SHA256SUMS" "$banks" > "$stage/source-inputs.sha256"
seal "$payload"
(cd "$stage" && tar -czf payload.tar.gz payload && sha256sum payload.tar.gz > payload.tar.sha256)
seal "$stage"
