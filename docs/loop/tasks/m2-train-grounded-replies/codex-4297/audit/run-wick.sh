#!/usr/bin/env bash
# Audits sealed captures with the scalar integer reader. Never dispatches CUDA.
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-all-outranker-interval"
audit="$study/audit"
old="$runs/20261126-codex-protected-transport"
omission="$runs/20261126-codex-intervening-source"
mixed="$runs/20261126-codex-mixed-transport/panel/panel-v1"
pooled="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
prior="$runs/20261126-codex-text-governor-links/parity/source-v2"
fixture_words="$runs/20261126-codex-far-transport-margin/fit/finite-v4/fixture.words"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$audit"/source-v* ]]
mode=${1:?build, antecedent, fixture, gate-fixture, panel, train or held}
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
    cp "$here"/*.rs "$out/src/"
    cp "$here/run-wick.sh" "$out/"
    (
        cd "$out"
        "$HOME/.cargo/bin/rustc" --version > rustc.txt
        lscpu > cpu.txt
        RUSTFLAGS='-D warnings' "$HOME/.cargo/bin/cargo" test --release --offline --locked > tests.log 2>&1
        RUSTFLAGS='-D warnings' "$HOME/.cargo/bin/cargo" build --release --offline --locked > build.log 2>&1
        cp target/release/textlinks-trace-audit bridge-audit
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
bin="$work/bridge-audit"
old_measurement="$old/measurement-v1"
case "$mode" in
antecedent)
    [[ $# == 3 ]]
    pin fe7485ee9e0a7073a7530e71aadf7eafad95a77ce8e3ca3bdbac32feb19c90bb "$old_measurement/SHA256SUMS"
    pin ae9471af842a490d39a9b114fd096e194cb61ba9158c495adbbcddae35278fb8 "$omission/measurement-v1/SHA256SUMS"
    pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$mixed/SHA256SUMS"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$old_measurement/states/protected.state"
    verify "$old_measurement"; verify "$omission/measurement-v1"; verify "$mixed"
    sha256sum "$work/SHA256SUMS" "$old_measurement/SHA256SUMS" "$omission/measurement-v1/SHA256SUMS" "$mixed/SHA256SUMS" "$pooled" > "$out/inputs.sha256"
    args=(antecedent "$mixed/mixed" "$old" "$omission" "$pooled")
    ;;
fixture)
    [[ $# == 5 ]]
    build_gpu=$(realpath "$4"); finite=$(realpath "$5"); lease="$finite-lease"
    [[ $build_gpu == "$study"/fit/build-v* && $finite == "$study"/fit/finite-v* && -e $lease/SHA256SUMS ]]
    verify "$build_gpu"; verify "$finite"; verify "$lease"
    [[ $(cat "$lease/exit-code.txt") == 0 ]]
    sha256sum --strict --quiet -c "$finite/inputs.sha256"
    sha256sum "$work/SHA256SUMS" "$build_gpu/SHA256SUMS" "$finite/SHA256SUMS" "$lease/SHA256SUMS" "$fixture_words" > "$out/inputs.sha256"
    args=(fixture "$fixture_words" "$finite")
    ;;
gate-fixture)
    [[ $# == 4 ]]
    core=$(realpath "$4")
    [[ $core == "$study"/finite-core-v* ]]
    verify "$core"
    sha256sum "$work/SHA256SUMS" "$core/SHA256SUMS" > "$out/inputs.sha256"
    args=(gate-fixture "$core")
    ;;
panel)
    [[ $# -ge 4 ]]
    panel=$(realpath "$4")
    [[ $panel == "$study"/panel/panel-v* ]]
    verify "$panel"; verify "$mixed"
    shift 4
    sha256sum "$work/SHA256SUMS" "$mixed/SHA256SUMS" "$panel/SHA256SUMS" "$@" > "$out/inputs.sha256"
    args=(panel "$panel" "$mixed/mixed" "$out/readback.json" "$@")
    ;;
train)
    [[ $# == 5 ]]
    panel=$(realpath "$4"); physical=$(realpath "$5")
    [[ $panel == "$study"/panel/panel-v* && $physical == "$study"/measurement-v* ]]
    verify "$panel"; verify "$physical"
    for arm in strongest all ordered-all; do
        for round in 1 2; do
            verify "$physical/train/$arm/round-$round/device"
            verify "$physical/train/$arm/round-$round/grade"
        done
    done
    sha256sum "$work/SHA256SUMS" "$old_measurement/SHA256SUMS" "$panel/SHA256SUMS" "$physical/SHA256SUMS" "$pooled" > "$out/inputs.sha256"
    args=(train "$old_measurement" "$pooled" "$panel" "$physical")
    ;;
held)
    [[ $# == 7 ]]
    panel=$(realpath "$4"); training=$(realpath "$5"); held=$(realpath "$6"); grade=$(realpath "$7")
    [[ $panel == "$study"/panel/panel-v* && $training == "$study"/measurement-v* && $held == "$study"/held-v* && $grade == "$study"/grade-v* ]]
    verify "$panel"; verify "$training"; verify "$held"; verify "$grade"
    sha256sum --strict --quiet -c "$held/inputs.sha256" "$grade/inputs.sha256"
    far_state="$runs/20261126-codex-mixed-transport/measurement-v1/states/far_only.state"
    pin d156060e1dd2a2b9765fb43567b06590d2e1acc1094a4fd19274179982e55eb7 "$far_state"
    sha256sum "$work/SHA256SUMS" "$old_measurement/SHA256SUMS" "$panel/SHA256SUMS" "$training/SHA256SUMS" \
        "$held/SHA256SUMS" "$grade/SHA256SUMS" "$pooled" "$far_state" > "$out/inputs.sha256"
    args=(held "$pooled" "$old_measurement/states/protected.state" "$far_state" "$panel" "$training" "$held" "$grade")
    ;;
*) exit 64 ;;
esac
status=0
if [[ $mode != panel ]]; then args+=("$out/readback.json"); fi
"$bin" "${args[@]}" > "$out/stdout.log" 2> "$out/stderr.log" || status=$?
printf '%s\n' "$status" > "$out/exit-code.txt"
seal "$out"
exit "$status"
