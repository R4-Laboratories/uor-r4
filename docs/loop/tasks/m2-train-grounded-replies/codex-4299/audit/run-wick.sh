#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-fixed-assignment-median"
audit="$study/audit"
predecessor="$runs/20261126-codex-all-outranker-interval"
historic="$runs/20261126-codex-protected-transport/measurement-v1"
parent_bin="$historic/states/protected.state"
pool="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
old_src="$runs/20261126-codex-text-governor-links/parity/source-v2"
old_audit="$predecessor/audit/build-v5"
far="$runs/20261126-codex-mixed-transport/measurement-v1/states/far_only.state"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$audit"/source-v* ]]
version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -path './target/*' ! -name .pending -print0 | sort -z | xargs -0 -r sha256sum > .pending
    mv .pending SHA256SUMS
    find . -type f ! -path './target/*' -exec chmod u-w {} +
)
action=${1:?build, premise, fixture, panel, gate-fixture, train or held}
trial=${2:?new output version}
version "$trial"
out="$audit/$action-$trial"
if [[ $action == build ]]; then
    test "$#" -eq 2
    pin 527d4b485091a3b25305519079a77da7f7f7944bbf6498fbedd0bb78dea616a0 "$old_audit/SHA256SUMS"
    verify "$old_audit"
    mkdir "$out" "$out/src"
    for file in controls gate identity panel replay signed packet; do
        cp "$old_audit/src/$file.rs" "$out/src/"
    done
    for file in Cargo.toml Cargo.lock; do
        cmp "$old_audit/$file" "$old_src/$file"
        cp "$old_audit/$file" "$out/"
    done
    cp "$here/"*.rs "$out/src/"
    cp "$here/run-wick.sh" "$out/"
    (cd "$out"
        "$HOME/.cargo/bin/rustc" --version > rustc.txt
        RUSTFLAGS='-D warnings' "$HOME/.cargo/bin/cargo" test --release --offline --locked > tests.log 2>&1
        RUSTFLAGS='-D warnings' "$HOME/.cargo/bin/cargo" build --release --offline --locked > build.log 2>&1
        cp target/release/textlinks-trace-audit median-audit
    )
    sha256sum "$old_audit/SHA256SUMS" "$here/"*.rs "$here/run-wick.sh" > "$out/inputs.sha256"
    seal "$out"
    exit
fi
version "${3:?sealed Rust auditor version}"
build="$audit/build-$3"
verify "$build"
stable="$predecessor/panel/panel-v1"
pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$parent_bin"
pin fe7485ee9e0a7073a7530e71aadf7eafad95a77ce8e3ca3bdbac32feb19c90bb "$historic/SHA256SUMS"
pin 92704b9d3ffb8201245a26fe05c5648e29360d0a383dd0db9ccb1389f845eb4b "$stable/SHA256SUMS"

manifest=("$build/SHA256SUMS")
case $action in
premise)
    verify "$historic"
    verify "$stable"
    panel=${4:-"$stable"}
    if [[ $panel != "$stable" ]]; then [[ $panel == "$study"/panel/panel-v* ]]; verify "$panel"; fi
    manifest+=("$historic/SHA256SUMS" "$stable/SHA256SUMS" "$panel/SHA256SUMS" "$parent_bin" "$pool")
    args=(premise "$pool" "$parent_bin" "$panel" "$stable" "$historic" "$out/readback.json")
    ;;
fixture)
    [[ $# == 8 ]]
    endpoint=$(realpath "$4"); ordered=$(realpath "$5"); labels=$(realpath "$6"); state=$(realpath "$7"); device=$(realpath "$8")
    for fixture in "$endpoint" "$ordered" "$labels" "$state" "$device"; do
        [[ $fixture == "$study"/fit/* ]]
    done
    finite=${device%/fit}
    [[ $finite == "$study"/fit/finite-v* && $device == "$finite/fit" ]]
    lease="$finite-lease"
    verify "$finite"; verify "$lease"
    [[ $(cat "$lease/exit-code.txt") == 0 ]]
    sha256sum --strict --quiet -c "$finite/inputs.sha256"
    manifest+=("$finite/SHA256SUMS" "$lease/SHA256SUMS" "$endpoint" "$ordered" "$labels" "$state")
    args=(fixture "$endpoint" "$ordered" "$labels" "$state" "$device" "$out/readback.json")
    ;;
panel)
    panel=$(realpath "${4:?fresh panel}")
    [[ $panel == "$study"/panel/panel-v* ]]
    verify "$panel"; verify "$stable"
    shift 4
    manifest+=("$panel/SHA256SUMS" "$stable/SHA256SUMS" "$stable/held/reference.json" "$stable/contrast/reference.json" "$@")
    args=(panel "$panel" "$stable/mixed" "$out/readback.json" "$stable/held/reference.json" "$stable/contrast/reference.json" "$@")
    ;;
gate-fixture)
    core=$(realpath "${4:-"$predecessor/finite-core-v3"}")
    [[ $core == "$predecessor"/finite-core-v* || $core == "$study"/finite-core-v* ]]
    verify "$core"
    manifest+=("$core/SHA256SUMS")
    args=(gate-fixture "$core" "$out/readback.json")
    ;;
train)
    [[ $# == 5 ]]
    panel=$(realpath "$4"); physical=$(realpath "$5")
    [[ $panel == "$study"/panel/panel-v* && $physical == "$study"/train-v* ]]
    verify "$panel"; verify "$physical"; verify "$historic"
    manifest+=("$panel/SHA256SUMS" "$physical/SHA256SUMS" "$historic/SHA256SUMS" "$pool")
    args=(train "$pool" "$parent_bin" "$panel" "$historic" "$physical" "$out/readback.json")
    ;;
held)
    [[ $# == 7 || $# == 8 ]]
    panel=$(realpath "$4"); train=$(realpath "$5"); physical=$(realpath "$6"); graded=$(realpath "$7")
    verified="$audit/train-${8:-$trial}"
    [[ $panel == "$study"/panel/panel-v* && $train == "$study"/train-v* && $physical == "$study"/held-v* && $graded == "$study"/grade-v* ]]
    for path in "$panel" "$train" "$physical" "$graded" "$verified"; do verify "$path"; done
    pin d156060e1dd2a2b9765fb43567b06590d2e1acc1094a4fd19274179982e55eb7 "$far"
    grep -Fxq "$(sha256sum "$train/SHA256SUMS")" "$verified/inputs.sha256"
    manifest+=("$panel/SHA256SUMS" "$train/SHA256SUMS" "$physical/SHA256SUMS" "$graded/SHA256SUMS" "$verified/SHA256SUMS" "$far" "$pool")
    args=(held "$pool" "$parent_bin" "$far" "$panel" "$train" "$verified/readback.json" "$physical" "$graded" "$out/readback.json")
    ;;
*) exit 64 ;;
esac
mkdir "$out"
sha256sum "${manifest[@]}" > "$out/inputs.sha256"
status=0
"$build/median-audit" "${args[@]}" > "$out/stdout.log" 2> "$out/stderr.log" || status=$?
printf '%s\n' "$status" > "$out/exit-code.txt"
seal "$out"
exit "$status"
