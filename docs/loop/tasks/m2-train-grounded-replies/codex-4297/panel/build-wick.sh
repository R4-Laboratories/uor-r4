#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-all-outranker-interval/panel"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive build version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ && $here == "$root"/methods-v* ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs ]]
build="$root/build-v$version"
method=docs/loop/tasks/m2-train-grounded-replies/codex-4288/export
generator_method=docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel
assets="$runs/20261125-codex-governing-source-pairs/artifact-build-v3"
original="$runs/20261126-codex-text-governor-links/export/build-v2"
generator="$runs/20261126-codex-governor-noun-links/panel/source-v1/$generator_method"
packet="$runs/20261126-codex-protected-transport/audit/build-v2/src/packet.rs"
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 57bdafd1e0171a3493a93a66079437ca8101530d95d5096c0e2115233ac3817c "$assets/source.sha256"
pin b125d17182c8751dcaa06bf1d3b268544b6b0757800c546b806fa69f08ff51a9 "$packet"
pin 8a51e68330ebec2966329cd0538c9e276e9e3d09dd77fc5bf5fe00cf13fedafd "$original/source/$method/src/main.rs"
pin e945d1090b289085f36c9adfa0d06334b5fa0fafbc51972c2901d5bba7016aef "$generator/src/main.rs"
(cd "$assets/source" && sha256sum --strict --quiet -c ../source.sha256)
(cd "$original" && sha256sum --strict --quiet -c SHA256SUMS)

mkdir "$build"
target="$build/source/$method"
contrast="$build/source/$generator_method"
mkdir -p "$target" "$contrast"
cp -a "$original/source/$method/"{Cargo.toml,Cargo.lock,src} "$target/"
cp -a "$generator/"{Cargo.toml,Cargo.lock,src} "$contrast/"
ln -s "$assets/source/crates" "$build/source/crates"
cp "$packet" "$target/src/packet.rs"
cp "$here/"{bridge.rs,contrast.rs} "$target/src/"
(cd "$target" && patch --fuzz=0 -p0 < "$here/exporter.patch")
(cd "$contrast" && patch --fuzz=0 -p0 < "$here/contrast-generator.patch")
export CARGO_TARGET_DIR="$build/target"
"$HOME/.cargo/bin/cargo" build --offline --locked --release --manifest-path "$target/Cargo.toml" > "$build/build.log" 2>&1
"$HOME/.cargo/bin/cargo" build --offline --locked --release --manifest-path "$contrast/Cargo.toml" > "$build/contrast-build.log" 2>&1
(
    cd "$build"
    sha256sum "source/$method/"{Cargo.toml,Cargo.lock,src/main.rs,src/bridge.rs,src/contrast.rs,src/packet.rs} \
        "source/$generator_method/"{Cargo.toml,Cargo.lock,src/main.rs} \
        target/release/{source-link-text,governor-links-panel} > SHA256SUMS
    sha256sum "$here/"{bridge.rs,contrast.rs,exporter.patch,contrast-generator.patch,build-wick.sh} "$packet" "$generator/src/main.rs" \
        "$original/SHA256SUMS" "$assets/source.sha256" \
        "$assets/source/crates/uor-r4-integer/src/"{role_action.rs,role_action/instruction.rs,discrete_d11.rs} > authority.sha256
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
