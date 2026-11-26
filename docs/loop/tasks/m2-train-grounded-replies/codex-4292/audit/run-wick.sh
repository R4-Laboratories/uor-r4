#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
joint="$runs/20261126-codex-text-governor-links/parity/source-v2"
signed="$runs/20261126-codex-far-transport-margin/audit/method-v2/src/main.rs"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
mode=${1:?mode}
version=${2:?new receipt version}
if [[ $mode == batch ]]; then
    [[ $version =~ ^v[1-9][0-9]*(-(root|protected|free))?$ ]]
else
    [[ $version =~ ^v[1-9][0-9]*$ ]]
fi
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -path './target/*' ! -name '.pending' -print0 | sort -z | xargs -0 sha256sum > .pending
    mv .pending SHA256SUMS
)
mkdir -p "$s/audit"
case "$mode" in
build)
    [[ $# == 2 ]]
    d="$s/audit/build-$version"
    mkdir "$d" "$d/src"
    printf '%s  %s\n' \
        7ebd2863ca70cbd935b5789f4488adfc35c5b050ca8a5b834f7d98fbb38f0bc3 "$joint/Cargo.toml" \
        49247dbb6be992bc6a719a74a97eee96c942ebd1faebbc504a5d2bdc3e23e6e2 "$joint/Cargo.lock" \
        b125d17182c8751dcaa06bf1d3b268544b6b0757800c546b806fa69f08ff51a9 "$joint/src/packet.rs" \
        037cf53fe613a8f9e0935cc78012ba5882add9ad14fdc1d4c901f2f64d4fc48a "$joint/src/replay.rs" \
        fec06a65bda41c80d6b32fa036da5408f4853f1cd4208ad7b40cb6cf9f90cf7e "$signed" > "$d/original-source.sha256"
    sha256sum --strict --quiet -c "$d/original-source.sha256"
    cp "$joint"/{Cargo.toml,Cargo.lock} "$d/"
    cp "$joint"/src/{packet,replay}.rs "$d/src/"
    cp "$here/replay.rs" "$d/src/main.rs"
    cp "$here"/{run-wick.sh,signed-tail.rs,election.jq} "$d/"
    {
        printf 'use crate::packet;\nuse packet::{Row, Sign, EDGE};\nuse serde_json::{json, Value};\nuse std::{collections::BTreeSet, fs, path::Path};\n'
        sed -n '/^type Result/,/^fn addr(/{ /^fn addr(/!p; }' "$signed"
        sed -n '/^fn brief(/,/^fn hinge_and_wins/{ /^fn hinge_and_wins/!p; }' "$signed"
        cat "$d/signed-tail.rs"
    } > "$d/src/signed.rs"
    (
        cd "$d"
        "$HOME/.cargo/bin/rustc" --version > rustc.txt
        RUSTFLAGS="-D warnings" "$HOME/.cargo/bin/cargo" build --release --offline --locked > build.log 2>&1
        cp target/release/textlinks-trace-audit protected-transport-audit
    )
    seal "$d"
    ;;
batch|held)
    bv=${3:?audit build version}
    [[ $bv =~ ^v[1-9][0-9]*$ ]]
    d="$s/audit/build-$bv"
    verify "$d"
    shift 3
    out="$s/audit/$mode-$version"
    mkdir "$out"
    input=()
    for f in "$@"; do
        if [[ $f == - ]]; then continue; fi
        if [[ -d $f ]]; then
            if [[ -f $f/SHA256SUMS ]]; then
                verify "$f"
                input+=("$f/SHA256SUMS")
            else
                for name in menu.jsonl menu-summary.json complete.json scores.jsonl step1-replayed.state; do
                    input+=("$f/$name")
                done
            fi
        else
            input+=("$f")
        fi
    done
    sha256sum "$d/SHA256SUMS" "${input[@]}" > "$out/inputs.sha256"
    "$d/protected-transport-audit" "$mode" "$@" "$out/readback.json"
    seal "$out"
    ;;
*)
    exit 64
    ;;
esac
