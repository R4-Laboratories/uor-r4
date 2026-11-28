#!/usr/bin/env bash
set -euo pipefail
umask 077
runs=$(realpath "${1:?Wick runs root}")
repo=$(realpath "${2:?same checkout or tokenizer workspace}")
unit=$(realpath "${3:?path to codex-4338 source}")
build=$(realpath -m "${4:?new build directory}")
out=$(realpath -m "${5:?new roster directory}")
tasks=$unit/..
supply=$tasks/codex-4329/supply
mkdir -p "$build/src"
sed "s|../../../../../../crates/uor-r4-tokenizer|$repo/crates/uor-r4-tokenizer|" "$supply/Cargo.toml" > "$build/Cargo.toml"
cp "$supply/Cargo.lock" "$build/Cargo.lock"
cp "$unit/holdout/roster.rs" "$build/src/roster.rs"
{
    # Reuse source-authentication, tokenization, and existing historical pins.
    sed '/^fn main() -> Result<()>/,$d' "$supply/src/main.rs"
    cat <<'RS'
mod roster;
mod history;
fn main() -> Result<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [runs, inputs, output] = args.as_slice() else {
        return Err(bad_input("usage: roster RUNS OFFICIAL_INPUTS NEW_OUTPUT"));
    };
    roster::run(runs, inputs, output)
}
RS
} > "$build/src/main.rs"
sed -n '/^const POOLS: /,/^\];$/p' "$tasks/codex-4337/data/held.rs" |
    sed 's/^const POOLS/pub const POOLS/' > "$build/src/history.rs"
test -s "$build/src/history.rs"
cargo build --locked --release --manifest-path "$build/Cargo.toml"
"$build/target/release/whole-turn-supervision" "$runs" "$runs/20261127-codex-whole-turn-census/supply-inputs" "$out"
(
    cd "$out"
    sha256sum articles.jsonl prior-positive-split.tsv roster.jsonl summary.json > ROSTER.sha256
    sha256sum -c ROSTER.sha256
    date -u +%FT%TZ > sealed-at
    sha256sum ROSTER.sha256 sealed-at > SEAL.sha256
    chmod -R u-w,go-rwx .
    sha256sum SEAL.sha256
)
