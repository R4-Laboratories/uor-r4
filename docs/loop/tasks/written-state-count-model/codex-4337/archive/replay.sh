#!/usr/bin/env bash
set -euo pipefail
umask 077

restored=$(realpath "${1:?restored packet directory}")
balanced=$(realpath "${2:?pinned balanced D11}")
tokenizer=$(realpath "${3:?pinned tokenizer JSON}")
out=$(realpath -m "${4:?fresh replay directory}")
context=$restored/archive-context-v1
held=$restored/data/held-v1
gpu=$restored/gpu-v1/measured
cold=$restored/cold-v1
prep=$restored/fit-device-v1
destination=$out/source

check_identity() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -
}
pins=$context/provenance/identities.json
check_identity "$(jq -er .balanced_sha256 "$pins")" "$balanced"
check_identity "$(jq -er .tokenizer_sha256 "$pins")" "$tokenizer"
runtime_rustc=${INTERVAL_RUNTIME_RUSTC:-/usr/bin/rustc}
runtime_cargo=${INTERVAL_RUNTIME_CARGO:-/usr/bin/cargo}
cmp "$context/provenance/runtime-rustc.txt" <("$runtime_rustc" -Vv)
cmp "$context/provenance/rustc.txt" <(rustc -Vv)

mkdir "$out" "$destination"
base=$(jq -er .base_revision "$context/provenance/build.json")
[[ $(git get-tar-commit-id < "$context/product-base.tar") == "$base" ]]
tar --no-same-owner --no-same-permissions -xf "$context/product-base.tar" -C "$destination"
install -m 600 "$context/Cargo.toml" "$context/Cargo.lock" "$destination/"
(
    cd "$destination"
    git apply "$context/product.patch"
    sha256sum --strict --quiet -c "$context/PRODUCT-SOURCES.sha256"
)
RUSTC="$runtime_rustc" "$runtime_cargo" build --locked --offline --release --manifest-path "$destination/Cargo.toml" \
    --target-dir "$out/runtime-target" -p uor-r4-integer --features serving-wasm \
    --bin uor-chat --bin pack-native-interval > "$out/runtime-build.log" 2>&1
cargo build --locked --offline --release --manifest-path "$context/lab-source/fit/Cargo.toml" \
    --target-dir "$out/fit-target" > "$out/fit-build.log" 2>&1

fit=$out/fit-target/release/interval-fit
check_identity "$(jq -er .client_sha256 "$pins")" "$context/collection-bin/uor-chat"
check_identity "$(jq -er .pack_sha256 "$pins")" "$context/collection-bin/pack-native-interval"
mkdir "$out/collection-bin"
install -m 700 "$context/collection-bin/uor-chat" "$context/collection-bin/pack-native-interval" "$out/collection-bin/"

recorded=$cold/geometry.actual.pack.json
words=$gpu/geometry.actual.words
fit_sha=$(jq -er .interval.fit_sha256 "$recorded")
jq -c . "$cold/geometry.actual.held.cold.jsonl" > "$out/held.reference.jsonl"

# Choose by the sealed case identities and reciprocal distinct targets, before
# consulting emitted outcomes. Replay the partner question over the same passage.
jq -nc --slurpfile gold "$held/held-gold.jsonl" --slurpfile blind "$held/held-input.jsonl" '
    ($gold|INDEX(.case_id)) as $targets |
    ($blind|INDEX(.case_id)) as $prompts |
    [($gold|sort_by(.case_id))[] | select(.start!=null and .swapped_case_id!=null) |
       . as $original | $targets[.swapped_case_id] as $partner |
       select($partner.start!=null and $partner.start!=$original.start and $partner.swapped_case_id==$original.case_id) |
       {original:.case_id,peer:$partner.case_id}][0] as $pair |
    if $pair==null or $prompts[$pair.peer].text==$prompts[$pair.original].text
    then error("missing changed reciprocal question") else $pair end
    ' > "$out/paired-case.json"
peer=$(jq -er .peer "$out/paired-case.json")
jq -ce --arg peer "$peer" 'select(.case_id==$peer)' "$held/held-input.jsonl" > "$out/one.jsonl"
reference=$restored/ordinary-fitted-v2/fresh/geometry.actual/native/records.jsonl
jq -ceS --arg peer "$peer" 'select(.case_id==$peer)' "$reference" > "$out/ordinary.reference.jsonl"

for build in captured rebuilt; do
    bin=$out/collection-bin
    if [[ $build == rebuilt ]]; then bin=$out/runtime-target/release; fi
    capture=$out/$build
    mkdir "$capture"
    artifact=$capture/geometry.actual.d11
    client=$bin/uor-chat
    pack=$bin/pack-native-interval
    "$pack" geometry "$balanced" "$words" "$artifact" "$fit_sha" > "$capture/pack.json"
    cmp "$capture/pack.json" "$recorded"
    "$client" --discrete "$artifact" --tokenizer "$tokenizer" \
        --interval-features "$held/held-input.jsonl" > "$capture/held.cold.jsonl"
    jq -c . "$capture/held.cold.jsonl" > "$capture/held.canonical.jsonl"
    cmp "$capture/held.canonical.jsonl" "$out/held.reference.jsonl"
    "$fit" verify-device "$prep/features/held.jsonl" "$capture/held.canonical.jsonl" geometry \
        "$words" "$gpu/geometry.actual.held.scores" > "$capture/parity.json"
    cmp "$capture/parity.json" "$cold/geometry.actual.held.parity.json"
    bash "$context/lab-source/client/capture.sh" "$client" "$artifact" "$tokenizer" \
        "$out/one.jsonl" "$capture/ordinary" --native-occurrence-trace
    cmp "$capture/ordinary/records.jsonl" "$out/ordinary.reference.jsonl"
done
(
    cd "$out"
    sha256sum {captured,rebuilt}/{pack.json,held.canonical.jsonl,parity.json,ordinary/records.jsonl} \
        runtime-target/release/{uor-chat,pack-native-interval} fit-target/release/interval-fit > REPLAY.sha256
)
