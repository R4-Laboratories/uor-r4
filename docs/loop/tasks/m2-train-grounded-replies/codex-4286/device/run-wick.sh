#!/usr/bin/env bash
# Build and finite checks are separate from the preregistered held collection.
set -euo pipefail
test "$(hostname -s)" = wick
base=/mnt/plexpool/uor-r4-corpus/runs
root="$base/20261126-codex-governor-noun-links/device"
historical="$base/20261125-codex-governing-source-pairs"
previous_rust="$historical/artifact-build-v3"
previous_cuda="$base/20261125-codex-within-prompt-scope-margin/device-build-v1"
fit="$historical/fit-v1/primary.words"
artifact="$historical/models/fold-8/ordered-all.d11"
codec="$base/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
lock="$previous_cuda/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "$0")" && pwd)
method=docs/loop/tasks/m2-train-grounded-replies/codex-4286/device
arms=(native governor_only nouns_only correct_link exchanged_link)
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
sha() { sha256sum "$1" | cut -d' ' -f1; }
verify() ( cd "$1" && sha256sum --quiet --strict -c SHA256SUMS; )
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)

build_paths() {
    [[ $1 =~ ^v[1-9][0-9]*$ ]]
    build="$root/build-$1"
    rust="$build/target/release/governor-noun-links"
    gpu="$build/governor-noun-links-sm89"
}
verify_build() {
    verify "$build"
    sha256sum --quiet --strict -c "$build/dependencies.sha256"
    (cd "$build/source" && sha256sum --quiet --strict -c ../source.sha256)
    (cd "$previous_rust/source" && sha256sum --quiet --strict -c ../source.sha256)
    verify "$previous_cuda"
}
rank_arm() {
    local fit_input=$1 prepared=$2 output=$3 arm=$4
    /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$output/$arm.time" \
        "$gpu" "$fit_input" "$prepared/geometry.bin" "$prepared/$arm.bin" "$output/$arm.rank"
    "$gpu" "$fit_input" "$prepared/geometry.bin" "$prepared/$arm.bin" "$output/$arm.repeat.rank"
    cmp "$output/$arm.rank" "$output/$arm.repeat.rank"
    if test -f "$prepared/$arm.permuted.bin"; then
        "$gpu" "$fit_input" "$prepared/geometry.bin" "$prepared/$arm.permuted.bin" "$output/$arm.permuted.rank"
        cmp "$output/$arm.rank" "$output/$arm.permuted.rank"
    fi
}
rank_all() {
    local arm
    for arm in "${arms[@]}"; do rank_arm "$@" "$arm"; done
}
released() {
    [[ $release == "$root/"* && $prepared == "$root/prepared-"* ]]
    test "$(sha "$release")" = "$release_sha"
    sha256sum --quiet --strict -c "$release"
    for needed in "$build/SHA256SUMS" "$prepared/SHA256SUMS"; do
        grep -Fxq "$(sha256sum "$needed")" "$release"
    done
    local finite
    finite=$(awk -v prefix="$root/finite-" 'index($2, prefix) == 1 && substr($2, length(prefix) + 1) ~ /^v[1-9][0-9]*\/SHA256SUMS$/ { print $2 }' "$release")
    test -f "$finite"
    verify "$(dirname "$finite")"
    grep -Fxq "$(sha256sum "$build/SHA256SUMS")" "$(dirname "$finite")/inputs.sha256"
    verify_build
    verify "$prepared"
    grep -Fxq "$(sha256sum "$build/SHA256SUMS")" "$prepared/inputs.sha256"
    sha256sum --quiet --strict -c "$prepared/inputs.sha256"
}

mode=${1:?build, finite, prepare, or score}
version=${2:?fresh output version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
case "$mode" in
build)
    build_paths "$version"
    mkdir "$build"
    mkdir -p "$build/source/$method/src"
    cp -- "$here/"{Cargo.toml,Cargo.lock,links.cu,run-wick.sh,README.md} "$build/source/$method"
    cp -- "$here"/src/*.rs "$build/source/$method/src"
    ln -s "$previous_rust/source/crates" "$build/source/crates"
    (cd "$build/source" && find "$method" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > ../source.sha256)
    (cd "$previous_rust/source" && sha256sum --quiet --strict -c ../source.sha256)
    verify "$previous_cuda"
    /usr/local/cuda/bin/nvcc --version > "$build/compiler.txt"
    CARGO_TARGET_DIR="$build/target" "$HOME/.cargo/bin/cargo" build --offline --locked --release \
        --manifest-path "$build/source/$method/Cargo.toml" > "$build/build.log" 2>&1
    /usr/local/cuda/bin/nvcc --std=c++17 -O3 -gencode arch=compute_89,code=sm_89 \
        -I "$previous_cuda/source" "$build/source/$method/links.cu" -o "$gpu" >> "$build/build.log" 2>&1
    "$(/usr/bin/realpath /usr/local/cuda/bin/cuobjdump)" --list-elf "$gpu" > "$build/gpu-elf.txt"
    grep -q 'sm_89' "$build/gpu-elf.txt"
    sha256sum "$previous_rust/source.sha256" "$previous_cuda/SHA256SUMS" > "$build/dependencies.sha256"
    (cd "$build" && sha256sum source.sha256 compiler.txt build.log gpu-elf.txt dependencies.sha256 \
        governor-noun-links-sm89 target/release/governor-noun-links > SHA256SUMS)
    ;;
prepare)
    build_paths "${3:?built version}"
    reference=$(realpath "${4:?frozen reference}")
    verify_build
    prepared="$root/prepared-$version"
    "$rust" prepare "$reference" "$artifact" "$codec" "$fit" "$prepared"
    sha256sum "$reference" "$artifact" "$codec" "$fit" "$build/SHA256SUMS" > "$prepared/inputs.sha256"
    seal "$prepared"
    ;;
finite)
    build_paths "${3:?built version}"
    bash "$lock" "$root/finite-$version-lease" bash "$build/source/$method/run-wick.sh" finite-inner "$version" "$3"
    ;;
finite-inner)
    test -e "/proc/$$/fd/9"
    build_paths "${3:?built version}"
    verify_build
    out="$root/finite-$version"
    "$rust" fixture "$out"
    mkdir "$out/scored"
    rank_all "$out/fit.words" "$out/prepared" "$out/scored"
    rank_arm "$out/fit.words" "$out/prepared" "$out/scored" wrong_governor
    rank_arm "$out/fit.words" "$out/prepared" "$out/scored" flipped_governor_keys
    cmp "$out/scored/flipped_governor_keys.rank" "$out/scored/exchanged_link.rank"
    "$rust" grade-fixture "$out"
    # Appending one invalid byte exercises strict geometry ABI consumption.
    cp "$out/prepared/geometry.bin" "$out/bad-geometry.bin"
    printf '\x01' >> "$out/bad-geometry.bin"
    if "$gpu" "$out/fit.words" "$out/bad-geometry.bin" "$out/prepared/native.bin" "$out/invalid.rank" \
        > "$out/invalid.stdout" 2> "$out/invalid.stderr"; then
        echo 'Malformed finite packet was accepted' >&2
        exit 1
    fi
    grep -q 'trailing packet data' "$out/invalid.stderr"
    sha256sum "$build/SHA256SUMS" "$previous_cuda/SHA256SUMS" > "$out/inputs.sha256"
    seal "$out"
    ;;
score|score-inner)
    build_paths "${3:?built version}"
    prepared=$(realpath "${4:?prepared packet directory}")
    release=$(realpath "${5:?released input manifest}")
    release_sha=${6:?released manifest SHA-256}
    released
    out="$root/score-$version"
    if [[ $mode == score ]]; then
        bash "$lock" "$out-lease" bash "$build/source/$method/run-wick.sh" score-inner "$version" "${@:3}"
    else
        test -e "/proc/$$/fd/9"
        mkdir "$out"
        rank_all "$fit" "$prepared" "$out"
        "$rust" grade "$prepared" "$artifact" "$fit" "$out"
        sha256sum "$release" "$build/SHA256SUMS" "$prepared/SHA256SUMS" "$artifact" "$fit" > "$out/inputs.sha256"
        seal "$out"
    fi
    ;;
*) exit 64 ;;
esac
