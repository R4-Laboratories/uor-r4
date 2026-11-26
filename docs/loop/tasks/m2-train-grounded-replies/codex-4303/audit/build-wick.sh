#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
inherited=${1:?sealed paired-audit build-v3}
out=${2:?new output directory}
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $# == 2 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
printf '%s  %s\n' d844f1de2efc2334adc027339755319cdcdda34cd816d16a6b59529a329e85c7 \
    "$inherited/SHA256SUMS" | sha256sum --strict --quiet -c -
(cd "$inherited" && sha256sum --strict --quiet -c SHA256SUMS && sha256sum --strict --quiet -c input.sha256)
mkdir "$out"
for name in Cargo.toml Cargo.lock main.rs packet.rs enrollment.rs identities.rs ranking.rs rival.rs controls.rs; do
    install -m 400 "$inherited/$name" "$out/$name"
done
install -m 400 "$here/slot3.rs" "$out/slot3.rs"
install -m 400 "$here/margin.rs" "$out/margin.rs"
sed -e '/^mod controls;/a mod slot3;' \
    -e '/include!("rival.rs");/a\    include!("margin.rs");' \
    -e '/^fn run(args: \&\[PathBuf\]) -> Result<()>/a\    if let Some(result) = slot3::dispatch(args) { return result; }' \
    "$inherited/paired-main.rs" > "$out/paired-main.rs"
ln -s "$(readlink -f "$inherited/tokenizer")" "$out/tokenizer"
sha256sum "$inherited/SHA256SUMS" "$inherited/input.sha256" "$here/slot3.rs" "$here/margin.rs" "$here/build-wick.sh" > "$out/input.sha256"
export CARGO_TARGET_DIR="$out/target"
(cd "$out" && cargo test --locked --offline --release --bin paired-audit > tests.log 2>&1)
(cd "$out" && cargo build --locked --offline --release --bin paired-audit > build.log 2>&1)
install -m 500 "$out/target/release/paired-audit" "$out/slot3-audit"
(cd "$out" && sha256sum Cargo.toml Cargo.lock *.rs *.log input.sha256 slot3-audit > SHA256SUMS)
