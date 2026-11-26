#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-governor-noun-links/panel"
prior="$runs/20261125-codex-governing-source-pairs"
codec="$runs/20261125-codex-instruction-admission-phase-archive/payload/signed/tokenizer.json"
model="$prior/models/fold-8/ordered-all.d11"
history="$prior/reserve/history.json"
capture="$prior/held/primary/capture-stock-v1/stock/fresh/rows.jsonl"
grammar="$root/grammar-v4.json"

version=${1:?unused positive panel version}
[[ $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs ]]
[[ $here == "$root/"* ]]
out="$root/panel-v$version"
partial="$out.partial.$$"
[[ ! -e $out && ! -e $partial ]]

check_hash() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -
}
source_root=$(cd "$here/../../../../../.." && pwd)
integer="$source_root/crates/uor-r4-integer/src"
query="$integer/role_action/instruction/learned/query.rs"
bound="$integer/role_action/instruction/bound.rs"
dialogue="$source_root/crates/uor-r4-tokenizer/src/dialogue.rs"
check_hash 022d12038ab8b86341db50b7a35196d57d71184ca0fc96fb59e9edc2b9e7a96d "$source_root/Cargo.toml"
check_hash 091ab58c5b25798191c1c6cdb41fa27566893531461a2b0f229f38e33266954e "$source_root/Cargo.lock"
check_hash 9fb82a1c4e0059d60eed928f33eb6ef49e23ea0a32365d51047d5db3ef173080 "$query"
check_hash 1af4c748066ab2feee98e33ea1f55097633e82fc5e8cd6c6cb59789aa7e5055a "$bound"
check_hash ed0642683bed0e40209fbea82d22016952825dbe2db21461c06f02756f2d1c7a "$integer/role_action.rs"
check_hash 7047c2323dd7c9277732adbf2899578af5661187451c7210d6af5fc17fab5b30 "$dialogue"
check_hash a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
check_hash 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$model"
check_hash 45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389 "$history"
check_hash 0d7db70bd8f89bf3a5ac634a543a077fb864c211211cbedc7091107328aa5902 "$capture"
check_hash c5b9fdb9c514e9ce9a9a8310520611f8defdbe1b17bb36107d589b554a022fea "$grammar"

export CARGO_TARGET_DIR="$root/target"
cargo build --release --locked --offline --manifest-path "$here/Cargo.toml"
binary="$CARGO_TARGET_DIR/release/governor-links-panel"
mkdir "$partial"
head -n 1 "$capture" | jq -ce '.record.turns[0] | {user,prompt_ids,reply_ids,stop,reply_text}' > "$partial/history.json"
cmp "$history" "$partial/history.json"
"$binary" "$codec" "$model" "$partial/history.json" "$grammar" "$partial/fresh" | tee "$partial/fixture.txt"
jq -e -f "$here/enrollment.jq" "$partial/fresh/reference.json" > "$partial/enrollment.json"
sha256sum "$here/"{Cargo.toml,Cargo.lock,src/main.rs,prepare-wick.sh,enrollment.jq} \
    "$source_root/"{Cargo.toml,Cargo.lock} "$query" "$bound" "$integer/role_action.rs" "$dialogue" \
    "$binary" "$codec" "$model" "$history" "$capture" "$grammar" > "$partial/authority.sha256"
(
    cd "$partial"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,fresh/reference.json,enrollment.json,authority.sha256}
