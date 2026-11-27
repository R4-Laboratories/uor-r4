#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 2 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${2%/*}") == zfs ]]
cold=$(realpath -e "$1")
out=$2
here=$(cd "$(dirname "$0")" && pwd -P)
old=$cold/independent/build
printf '%s  %s\n' 048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d "$cold/MEMBERS.sha256" \
    e0d8b534043b1f63e758eb3e2e18ad9201d7eaf8546ea6c6447ce10eed41ae8c "$old/SHA256SUMS" |
    sha256sum --strict --quiet -c -
(cd "$old" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$cold" && awk '$2 ~ /^\.\/independent\/build\/tokenizer\//' MEMBERS.sha256 | sha256sum --strict --quiet -c -)
mkdir "$out"
for name in Cargo.toml Cargo.lock main.rs packet.rs enrollment.rs identities.rs ranking.rs rival.rs controls.rs slot3.rs margin.rs; do
    install -m 400 "$old/$name" "$out/$name"
done
for name in route.rs study.rs evidence.rs; do
    install -m 400 "$here/$name" "$out/$name"
done
sed -e '/^mod controls;/a mod study;' \
    -e '/include!("identities.rs");/a\    include!("evidence.rs");' \
    -e '/include!("margin.rs");/a\    include!("route.rs");' \
    -e '/^fn run(args: \&\[PathBuf\]) -> Result<()>/a\    if let Some(result) = study::dispatch(args) { return result; }' \
    "$old/paired-main.rs" > "$out/paired-main.rs"
ln -s "$old/tokenizer" "$out/tokenizer"
sha256sum "$cold/MEMBERS.sha256" "$old/SHA256SUMS" "$here"/{build-wick.sh,route.rs,study.rs,evidence.rs} > "$out/input.sha256"
export CARGO_TARGET_DIR="$out/target"
(cd "$out" && cargo test --locked --offline --release --bin paired-audit > tests.log 2>&1)
(cd "$out" && cargo build --locked --offline --release --bin paired-audit > build.log 2>&1)
install -m 500 "$out/target/release/paired-audit" "$out/argument-audit"
(cd "$out" && sha256sum Cargo.toml Cargo.lock *.rs *.log input.sha256 argument-audit > SHA256SUMS)
