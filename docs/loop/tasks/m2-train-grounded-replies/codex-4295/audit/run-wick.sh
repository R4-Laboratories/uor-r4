#!/usr/bin/env bash
# Offline readback of recorded native rank outputs. Never dispatches CUDA.
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-intervening-source"
audit="$study/audit"
old="$runs/20261126-codex-protected-transport"
mixed="$runs/20261126-codex-mixed-transport/panel/panel-v1"
pooled="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
prior="$runs/20261126-codex-text-governor-links/parity/source-v2"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$audit"/source-v* ]]
mode=${1:?build, fixture, antecedent or captured}
version=${2:?new audit output version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -path './target/*' ! -name '.pending' -print0 | sort -z | xargs -0 -r sha256sum > .pending
    mv .pending SHA256SUMS
)
member() (
    cd "$1"
    awk -v path="./$2" '$2 == path || $2 == substr(path, 3) {print; ++found} END {if(found != 1) exit 1}' SHA256SUMS |
        sha256sum --strict --quiet -c -
)
if [[ $mode == build ]]; then
    [[ $# == 2 ]]
    out="$audit/build-$version"
    mkdir "$out" "$out/src"
    printf '%s  %s\n' \
        7ebd2863ca70cbd935b5789f4488adfc35c5b050ca8a5b834f7d98fbb38f0bc3 "$prior/Cargo.toml" \
        49247dbb6be992bc6a719a74a97eee96c942ebd1faebbc504a5d2bdc3e23e6e2 "$prior/Cargo.lock" \
        b125d17182c8751dcaa06bf1d3b268544b6b0757800c546b806fa69f08ff51a9 "$prior/src/packet.rs" \
        037cf53fe613a8f9e0935cc78012ba5882add9ad14fdc1d4c901f2f64d4fc48a "$prior/src/replay.rs" > "$out/prior.sha256"
    sha256sum --strict --quiet -c "$out/prior.sha256"
    cp "$prior"/{Cargo.toml,Cargo.lock} "$out/"
    cp "$prior"/src/{packet,replay}.rs "$out/src/"
    cp "$here"/{main,source,tests}.rs "$out/src/"
    cp "$here"/{run-wick.sh,replay.patch} "$out/"
    (
        cd "$out"
        patch --batch --fuzz=0 -p1 < replay.patch > patch.log
        "$HOME/.cargo/bin/rustc" --version > rustc.txt
        lscpu > cpu.txt
        RUSTFLAGS='-D warnings' "$HOME/.cargo/bin/cargo" test --release --offline --locked > tests.log 2>&1
        RUSTFLAGS='-D warnings' "$HOME/.cargo/bin/cargo" build --release --offline --locked > build.log 2>&1
        cp target/release/textlinks-trace-audit occurrence-audit
    )
    seal "$out"
    exit
fi
build=${3:?sealed audit build version}
[[ $build =~ ^v[1-9][0-9]*$ ]]
work="$audit/build-$build"
verify "$work"
out="$audit/$mode-$version"
mkdir "$out"
check_antecedents() {
    pin fe7485ee9e0a7073a7530e71aadf7eafad95a77ce8e3ca3bdbac32feb19c90bb "$old/measurement-v1/SHA256SUMS"
    pin 9d23a96d98dcb1667627e18009415bca0b110febb0ee6285031d9d29bb52c17f "$old/far-residuals-v2/summary.json"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$old/measurement-v1/states/protected.state"
    pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$mixed/SHA256SUMS"
    verify "$panel"; verify "$mixed"; verify "$old/far-residuals-v2"; verify "$old/measurement-v1"
    cmp "$old/measurement-v1/train/protected/device/240.state" "$old/measurement-v1/states/protected.state"
    cmp "$old/measurement-v1/train/root/device/234.state" "$old/measurement-v1/train/protected/device/base.state"
    sha256sum "$work/SHA256SUMS" "$old/measurement-v1/SHA256SUMS" "$old/far-residuals-v2/SHA256SUMS" \
        "$mixed/SHA256SUMS" "$panel/SHA256SUMS" "$pooled"
}
status=0
case "$mode" in
fixture)
    [[ $# == 4 ]]
    physical=$(realpath "$4")
    [[ $physical == "$study/device/finite-v"* ]]
    verify "$physical"
    sha256sum "$work/SHA256SUMS" "$physical/SHA256SUMS" > "$out/inputs.sha256"
    "$work/occurrence-audit" fixture "$physical" "$physical/first" "$out" > "$out/stdout.log" 2> "$out/stderr.log" || status=$?
    ;;
antecedent|captured)
    panel=$(realpath "${4:?panel directory}")
    [[ $panel == "$study/panel/panel-v"* ]]
    check_antecedents > "$out/inputs.sha256"
    if [[ $mode == captured ]]; then
        [[ $# == 5 ]]
        physical=$(realpath "$5")
        [[ $physical == "$study/"* ]]
        verify "$physical"
        for arm in neutral winner control; do
            member "$physical" "raw/$arm.jsonl"
            member "$physical" "raw/$arm.ranks.jsonl"
        done
        sha256sum "$physical/SHA256SUMS" >> "$out/inputs.sha256"
        physical="$physical/raw"
    else
        [[ $# == 4 ]]
        physical=-
    fi
    "$work/occurrence-audit" "$mode" "$mixed/mixed" "$old" "$panel" "$pooled" "$physical" "$out" > "$out/stdout.log" 2> "$out/stderr.log" || status=$?
    ;;
*) exit 64 ;;
esac
printf '%s\n' "$status" > "$out/exit-code.txt"
seal "$out"
exit "$status"
