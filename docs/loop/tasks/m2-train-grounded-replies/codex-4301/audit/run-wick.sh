#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-paired-occurrence-margin"
prior="$runs/20261126-codex-switch-source-competition"
median="$runs/20261126-codex-fixed-assignment-median"
decoder="$prior/audit/build-v5"
assets="$runs/20261125-codex-bound-instruction-state"
codec="$assets/custody-v1/signed/tokenizer.json"
audit="$study/audit"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$audit"/source-v* ]]
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .pending -print0 | sort -z | xargs -0 -r sha256sum > .pending
    mv .pending SHA256SUMS
    find . -type f -exec chmod u-w {} +
)
mode=${1:?build, train-census, fresh-census, archived-near, score-finite, or fit-finite}
version=${2:?unique version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
out="$audit/$mode-$version"
case "$mode" in
build)
    [[ $# == 2 ]]
    pin 9c6f634ef58752c914f47ee153cd23e39ddef943f19c773fa2c06e64360ddf01 "$decoder/SHA256SUMS"
    verify "$decoder"
    sha256sum --strict --quiet -c "$decoder/tokenizer-source.sha256"
    mkdir "$out"
    cp "$decoder/"{Cargo.toml,Cargo.lock,packet.rs,enrollment.rs,ranking.rs} "$out/"
    ln -s "$(realpath "$decoder/tokenizer")" "$out/tokenizer"
    new_sources=(paired-main.rs identities.rs rival.rs controls.rs run-wick.sh)
    for file in "${new_sources[@]}"; do cp "$here/$file" "$out/"; done
    cat >> "$out/Cargo.toml" <<'CARGO'

[[bin]]
name = "paired-audit"
path = "paired-main.rs"
CARGO
    cp "$decoder/main.rs" "$out/"
    sha256sum "$decoder/SHA256SUMS" "${new_sources[@]/#/$here/}" > "$out/input.sha256"
    (cd "$out"
        export CARGO_TARGET_DIR="$audit/target-$version" RUSTFLAGS='-D warnings'
        cargo test --bin paired-audit --release --locked --offline > tests.log 2>&1
        cargo build --bin paired-audit --release --locked --offline > build.log 2>&1
        cp "$CARGO_TARGET_DIR/release/paired-audit" .
    )
    seal "$out"
    ;;
train-census)
    [[ $# == 3 && $3 =~ ^v[1-9][0-9]*$ ]]
    build="$audit/build-$3"
    verify "$build"
    panel="$study/panel"
    premise="$median/audit/premise-v1"
    canonical_near="$median/panel/panel-v1/mixed"
    canonical_fit="$prior/panel/panel-v1/train-derived"
    canonical_reg="$prior/matched-train/panel-v2"
    pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
    pin 65afdefbb0f1e13cb7db0aaf95f3b248f18fbfa41917e590f75c7ee3acecb4a1 "$premise/SHA256SUMS"
    pin 80bbfaa53532c8e113000445285721c1e52d6f060b51a880a7c17681980c4a93 "$median/panel/panel-v1/SHA256SUMS"
    pin 8f46a07da62c7243cf2d5965c460ce2ba3f6484c1fa7266b77ecb7a4f0f09d92 "$prior/panel/panel-v1/SHA256SUMS"
    pin 099ac234e8d8dd17a12cc04639d2719a3d410163e38e3b517e062bfa9370a370 "$canonical_reg/SHA256SUMS"
    for path in "$premise" "$median/panel/panel-v1" "$prior/panel/panel-v1" "$canonical_reg"; do verify "$path"; done
    (cd "$panel" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
    mkdir "$out"
    for f in reference.json ordered/text.bin ordered/coordinates.jsonl; do
        cmp "$canonical_near/$f" "$panel/near/$f"
    done
    for f in reference.json ordered/text.bin ordered/coordinates.jsonl rows.jsonl requests.jsonl requests.tsv; do
        cmp "$canonical_fit/$f" "$panel/train/$f"
        cmp "$canonical_reg/$f" "$panel/regression/$f"
    done
    cmp <(cut -f2- "$panel/train/fit.tsv") "$panel/train/requests.tsv"
    cmp <(cut -f2- "$panel/regression/regression-index.tsv") "$panel/regression/requests.tsv"
    for f in "$panel/train/fit.tsv" "$panel/regression/regression-index.tsv"; do
        awk -F '\t' 'NF != 6 || $1 != int((NR - 1) / 2) || $2 != NR - 1 {exit 1}' "$f"
    done
    cmp <(jq -r '[.row_index, .governors[], .sites[]] | @tsv' "$panel/near/gold.jsonl") "$panel/near/guard.tsv"
    "$build/paired-audit" census "$panel/train" "$codec" "$out/train.json"
    "$build/paired-audit" census "$panel/regression" "$codec" "$out/regression.json"
    jq -se 'length == 8 and all(group_by(.pair_id)[]; length == 2 and
        (map(.sign)|unique|length) == 1 and (map(.selector)|sort) == ["first","later"]) and
        all(.[]; .governor.tokens.start - .requested.tokens.end ==
            (if .selector == "first" then 25 else 18 end) and
            .governor.tokens.start - .rival.tokens.end == (if .selector == "first" then 18 else 25 end))' \
        "$panel/regression/requests.jsonl" >/dev/null
    "$build/paired-audit" near-census "$panel/near" "$codec" "$premise/readback.json" \
        "$panel/near/gold.jsonl" "$out/near.json"
    sha256sum "$build/SHA256SUMS" "$panel/"{training.sha256,training-authority.sha256} "$codec" \
        "$premise/SHA256SUMS" "$median/panel/panel-v1/SHA256SUMS" "$prior/panel/panel-v1/SHA256SUMS" \
        "$canonical_reg/SHA256SUMS" "$here/run-wick.sh" > "$out/input.sha256"
    seal "$out"
    ;;
fresh-census)
    [[ $# == 3 && $3 =~ ^v[1-9][0-9]*$ ]]
    build="$audit/build-$3"
    panel="$study/panel/fresh"
    verify "$build"
    verify "$panel"
    sha256sum --strict --quiet -c "$panel/authority.sha256"
    pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
    cmp <(cut -f2- "$panel/pair-index.tsv") "$panel/requests.tsv"
    awk -F '\t' 'NF != 6 || $1 != int((NR - 1) / 2) || $2 != NR - 1 {exit 1}' "$panel/pair-index.tsv"
    jq -e '.comparison_rows == 1032 and (.overlaps | length == 8 and all(.[]; length == 0))' \
        "$panel/provenance.json" >/dev/null
    mkdir "$out"
    "$build/paired-audit" census "$panel" "$codec" "$out/enrollment.json"
    "$build/paired-audit" geometry "$panel" "$codec" "$panel/endpoint/text.bin" "$out/geometry.json"
    "$build/paired-audit" control-census "$panel" "$codec" "$out/controls.json"
    jq -e '.rows == 256 and .selector_pairs == 128 and .sign_requests == [128,128] and
        (.all_registered_strata | length == 16 and all(.[]; . == 16))' "$out/enrollment.json" >/dev/null
    jq -e '.rows == 256 and .selector_attachments_with_same_endpoint_and_changed_ordered == 256' \
        "$out/geometry.json" >/dev/null
    sha256sum "$build/SHA256SUMS" "$panel/"{SHA256SUMS,authority.sha256} "$codec" "$here/run-wick.sh" > "$out/input.sha256"
    seal "$out"
    ;;
archived-near)
    [[ $# == 4 && $3 =~ ^v[1-9][0-9]*$ && $4 =~ ^v[1-9][0-9]*$ ]]
    build="$audit/build-$3"
    census="$audit/train-census-$4"
    panel="$study/panel"
    premise="$median/audit/premise-v1"
    physical="$median/train-v1"
    original="$median/methods-v2/inputs/protected.state"
    refit="$physical/states/ordered.state"
    banks="$median/methods-v2/inputs/primary.words"
    for path in "$build" "$census" "$premise" "$physical"; do verify "$path"; done
    sha256sum --strict --quiet -c "$census/input.sha256"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$original"
    pin 0edd2cb3ee9251b9fa40cd829b52e3aff2fc477bf05d515c787cea9435af01ea "$refit"
    pin ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$banks"
    (cd "$panel" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
    mkdir "$out"
    common=(near-replay "$panel/near" "$codec" "$premise/readback.json" "$panel/near/gold.jsonl")
    "$build/paired-audit" "${common[@]}" "$original" "$banks" "$physical/ordered/base.raw.jsonl" "$out/original.json"
    "$build/paired-audit" "${common[@]}" "$refit" "$banks" "$physical/ordered/full.raw.jsonl" "$out/median.json"
    jq -e '.lost_protected | length == 8' "$out/original.json" >/dev/null
    jq -e '.lost_protected == []' "$out/median.json" >/dev/null
    sha256sum "$build/SHA256SUMS" "$census/SHA256SUMS" "$physical/SHA256SUMS" "$original" "$refit" "$banks" \
        > "$out/input.sha256"
    seal "$out"
    ;;
score-finite)
    [[ $# == 4 && $3 =~ ^v[1-9][0-9]*$ ]]
    build="$audit/build-$3"
    returned=$(realpath -e "$4")
    [[ $returned == "$study"/rented-score-fixture-v* ]]
    verify "$build"
    verify "$returned"
    score_input="$returned/score-code/fixture"
    score="$returned/score-finite-v1"
    mkdir "$out"
    diff -r "$score/a" "$score/b"
    "$build/paired-audit" fixture "$score_input/text.bin" "$score_input/fixture.state" \
        "$score_input/fixture.words" "$score_input/requests.tsv" "$score/a/probe.native.jsonl" \
        "$score/a/probe.costs.jsonl" "$out/competition.json"
    jq -e '.pass and .requests == 9 and .alternate_governor_thirds > 0 and
        .absent_groups > 0 and .tied_groups > 0' "$score/checks.json" >/dev/null
    sha256sum "$build/SHA256SUMS" "$returned/SHA256SUMS" "$here/run-wick.sh" > "$out/input.sha256"
    seal "$out"
    ;;
fit-finite)
    [[ $# == 3 ]]
    returned=$(realpath -e "$3")
    [[ $returned == "$study"/rented-finite-v* ]]
    verify "$returned"
    verify "$decoder"
    learning="$returned/fit-finite-v1"
    mkdir "$out"
    rustc --edition=2021 -D warnings "$here/finite-state.rs" -o "$out/finite-state"
    selected=$(jq -r 'select(.phase == "fit" and .path == 0 and .accepted) | .candidate' \
        "$learning/first/schedule.jsonl" | sed -n '1p')
    [[ $selected =~ ^fit-[0-9]+-[0-9]+$ ]]
    diff -r "$learning/first" "$learning/repeat"
    jq -e 'length == 5 and all(.[]; . == true)' "$learning/checks.json" >/dev/null
    for candidate in original "$selected"; do
        cmp "$learning/first/$candidate.native.jsonl" "$learning/oracle-$candidate/native.jsonl"
        "$out/finite-state" "$learning/first/original.state" "$learning/first/$candidate.state" "$out/$candidate.view"
        "$decoder/switch-audit" fixture "$learning/text.bin" "$out/$candidate.view" \
            "$learning/fixture.words" "$learning/requests.tsv" "$learning/oracle-$candidate" "$out/$candidate.json"
        jq -e '. == true' "$learning/oracle-$candidate.json" >/dev/null
    done
    jq -se 'length == 4 and all(.[]; .margin < 0 and .joint_margin < 0 and .pair_both_win == false)' \
        <(jq -c 'select(.candidate == "original")' "$learning/exchanged/objectives.jsonl") >/dev/null
    sha256sum "$decoder/SHA256SUMS" "$returned/SHA256SUMS" \
        "$here/finite-state.rs" "$here/run-wick.sh" > "$out/input.sha256"
    seal "$out"
    ;;
*) exit 64 ;;
esac
