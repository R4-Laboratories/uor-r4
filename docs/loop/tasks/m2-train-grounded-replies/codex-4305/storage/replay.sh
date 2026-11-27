#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 2 ]]
root=$(realpath -e "$1")
mkdir -- "$2"
out=$(realpath -e "$2")
case "$out/" in "$root/"*) echo "replay output must be outside the packet" >&2; exit 64 ;; esac
methods="$root/methods"
seed="$root/inputs/seed"
archived="$root/inputs/archived"
panel="$root/panel"
physical="$root/physical"
raw="$physical/raw"
expected="$root/expected"
state="$seed/slot3.state"
bank="$seed/primary.words"
codec="$seed/tokenizer.json"
verify() (cd "$1" && sha256sum --strict --quiet -c "${2:-SHA256SUMS}")
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }

test -z "$(find "$root" ! -type f ! -type d -print -quit)"
(
    cd "$root"
    find . -type f ! -path './MEMBERS.sha256' -print | sort > "$out/actual-members.txt"
    cut -d' ' -f3- MEMBERS.sha256 > "$out/declared-members.txt"
)
cmp "$out/actual-members.txt" "$out/declared-members.txt"
verify "$root" MEMBERS.sha256
for dir in "$seed" "$root/inputs/finite" "$methods"/{device,audit,grade,readout} \
    "$physical"/{build,finite} "$expected"/{grade,readout,archived,independent-finite,independent-panel,independent-physical,independent-identities} \
    "$root/custody/device-records"; do
    verify "$dir"
done
verify "$physical" raw.members
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$seed/ordered-all.d11"
pin e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977 "$state"
pin ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$bank"
cmp "$methods/device/SHA256SUMS" "$physical/build/source.sha256"
cmp "$state" "$seed/restored.state"
cmp "$seed/protected.bridge" "$raw/ordinary/state.bridge"
cmp "$state" "$raw/ordinary/state.readback"
[[ $("$seed/bundle" unpack "$seed/protected.bridge" "$out/ordered.state") == ordered ]]
cmp "$state" "$out/ordered.state"
for split in near regression; do
    for name in ordered/text.bin ordered/coordinates.jsonl reference.json; do
        cmp "$panel/$split/$name" "$archived/$split/$name"
    done
    cmp "$seed/previous/$split/original.jsonl" "$archived/captures/$split.original.jsonl"
done
cmp "$panel/near/gold.jsonl" "$archived/near/gold.jsonl"
cmp "$panel/near/protected-near.tsv" "$seed/previous/near/protected-near.tsv"
for ext in jsonl tsv; do cmp "$panel/regression/requests.$ext" "$archived/regression/requests.$ext"; done
printf 'Verified all members, protected input capture identities, and unpacked UORSTT01 bytes.\n'

bash "$methods/storage/panel-replay.sh" "$root" "$out/panel"
bash "$methods/grade/grade.sh" "$panel" "$raw" "$out/grade"
cmp "$expected/grade/SHA256SUMS" "$out/grade/SHA256SUMS"
verify "$out/grade"
bash "$methods/readout/readout.sh" "$out/grade" "$raw" "$out/readout"
cmp "$expected/readout/SHA256SUMS" "$out/readout/SHA256SUMS"
verify "$out/readout"
printf 'Regraded every arm, enrolled original guard and both supplied-address controls.\n'

mkdir "$out/cargo-home"
(
    cd "$methods/audit"
    export CARGO_HOME="$out/cargo-home"
    cargo test --locked --offline --release --bin paired-audit --target-dir "$out/target"
    cargo build --locked --offline --release --bin paired-audit --target-dir "$out/target"
) > "$out/audit-build.log" 2>&1 || { tail -80 "$out/audit-build.log" >&2; exit 1; }
audit="$out/target/release/paired-audit"
mkdir "$out/independent"
independent="$out/independent"
for split in near regression; do
    "$audit" argument-archived "$archived/$split/ordered/text.bin" "$state" "$bank" \
        "$archived/captures/$split.original.jsonl" "$independent/$split-native.json"
    cmp "$independent/$split-native.json" "$expected/archived/$split-native.json"
done
"$audit" replay "$archived/regression" "$codec" "$state" "$bank" \
    "$archived/captures/regression.original.jsonl" "$archived/captures/regression.costs.jsonl" \
    "$independent/regression-guards.json"
cmp "$independent/regression-guards.json" "$expected/archived/regression.json"
jq -e '(.pairs | length) == 4 and all(.pairs[]; .follows)' "$independent/regression-guards.json" >/dev/null
jq '{protected_near:(.protected_pairs|keys),
    protected_members:[.members[]|{pair:.pair_id,prompt_id:.original_id,admission,prompt_sha256,
    gold_governors:.governor_addresses,gold_nouns:.site_addresses}]}' \
    "$expected/archived/near.json" > "$independent/near-premise.json"
cmp "$independent/near-premise.json" "$expected/archived/near-premise.json"
"$audit" near-replay "$archived/near" "$codec" "$independent/near-premise.json" "$archived/near/gold.jsonl" \
    "$state" "$bank" "$archived/captures/near.original.jsonl" "$independent/near-guards.json"
cmp "$independent/near-guards.json" "$expected/archived/near.json"
jq -e '(.protected_pairs|length)==24 and (.members|length)==48 and (.lost_protected|length)==0' \
    "$independent/near-guards.json" >/dev/null

"$audit" argument-census "$panel/fresh" "$codec" "$independent/geometry.json"
"$audit" geometry "$panel/fresh" "$codec" "$panel/fresh/endpoint/text.bin" "$independent/paired-geometry.json"
"$audit" control-census "$panel/fresh" "$codec" "$independent/control-census.json"
for name in geometry paired-geometry; do
    cmp "$independent/$name.json" "$expected/independent-panel/$name.json"
done
cmp "$independent/control-census.json" "$expected/independent-panel/controls.json"
"$audit" physical-controls "$panel/fresh" "$codec" "$raw/controls/gold.jsonl" \
    "$raw/controls/swapped.jsonl" "$independent/controls.json"
cmp "$independent/controls.json" "$expected/independent-physical/controls.json"
jq -e '.physical.signed_rows == [256,0] and .physical.full_rows == [256,0]' "$independent/controls.json" >/dev/null

for split in regression near fresh; do
    cmp "$raw/$split/baseline.jsonl" "$raw/$split/native-reference.jsonl"
    cmp "$raw/$split/baseline.jsonl" "$raw/$split/ordinary.jsonl"
    cmp "$raw/$split/baseline.jsonl" "$raw/ordinary/$split.native.jsonl"
    cmp "$raw/$split/state.readback" "$state"
    original=()
    if [[ $split != fresh ]]; then original+=("$archived/captures/$split.original.jsonl"); fi
    "$audit" argument-captured "$panel/$split/ordered/text.bin" "$panel/$split/masks.tsv" \
        "$state" "$bank" "$raw/$split" "${original[@]}" "$independent/$split.json"
    cmp "$independent/$split.json" "$expected/independent-physical/$split.json"
done
jq -s '
    {rows:(map(.rows)|add),
     native_scalar_fields:([.[].arms[].native_scalar_fields]|add),
     route_scalar_fields:([.[].arms[].route_scalar_fields]|add),
     archived_baselines_equal:([.[]|select(.archived_baseline_equal == true)]|length),
     original_off_equal:all(.[]; .native_off_equal),
     split:([.[] | {rows,masked_rows,arms}] )}
' "$independent/"{regression,near,fresh}.json > "$independent/summary.json"
cmp "$independent/summary.json" "$expected/independent-physical/summary.json"
printf 'Rebuilt the offline Rust verifier and replayed every original ordinal, route and governor multiplicity.\n'

fixture="$root/inputs/finite"
finite="$physical/finite"
rustc --edition 2021 -D warnings "$methods/device/finite.rs" -o "$out/generate-finite"
"$out/generate-finite" "$out/fixture"
verify "$out/fixture" "$fixture/SHA256SUMS"
cmp "$finite/baseline.jsonl" "$finite/native-reference.jsonl"
cmp "$finite/state.readback" "$fixture/fixture.state"
jq -ne \
    --slurpfile baseline "$finite/baseline.jsonl" --slurpfile rb "$finite/routes.baseline.jsonl" \
    --slurpfile g "$finite/G.jsonl" --slurpfile rg "$finite/routes.G.jsonl" \
    --slurpfile n "$finite/N.jsonl" --slurpfile rn "$finite/routes.N.jsonl" \
    --slurpfile gn "$finite/GN.jsonl" --slurpfile rgn "$finite/routes.GN.jsonl" \
    --slurpfile wrong_g "$finite/wrong-G.jsonl" --slurpfile wrong_n "$finite/wrong-N.jsonl" \
    --slurpfile gn_wrong_g "$finite/GN-wrong-G.jsonl" --slurpfile gn_wrong_n "$finite/GN-wrong-N.jsonl" \
    --slurpfile gn_wrong_both "$finite/GN-wrong-both.jsonl" \
    --slurpfile gold "$finite/gold.jsonl" --slurpfile swap "$finite/swapped.jsonl" \
    -f "$methods/device/finite.jq" > "$independent/finite.json"
cmp "$independent/finite.json" "$finite/finite.json"
"$audit" argument-captured "$fixture/fixture.text" "$fixture/masks.tsv" "$fixture/fixture.state" \
    "$fixture/fixture.bank" "$finite" "$independent/finite-routes.json"
cmp "$independent/finite-routes.json" "$expected/independent-finite/native.json"
printf 'Rebuilt the finite inputs and matched the captured physical assertions and independent finite routes.\n'

(
    cd "$out"
    find independent panel grade readout fixture -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS
)
