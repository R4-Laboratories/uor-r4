#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-intervening-source/panel"
previous="$runs/20261126-codex-protected-transport"
measurement="$previous/measurement-v1"
residual="$previous/far-residuals-v1"
mixed="$runs/20261126-codex-mixed-transport/panel/panel-v1"
assets="$runs/20261125-codex-governing-source-pairs/artifact-build-v3"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
packet="$previous/audit/build-v2/src/packet.rs"
old="$measurement/train/root/device/234"
current="$measurement/train/protected/device/240"
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:?build or prepare}
version=${2:?unused version, e.g. v1}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$runs") == zfs ]]
[[ $version =~ ^v[1-9][0-9]*$ ]]
mkdir -p "$study"

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
member() (
    cd "$1"
    awk -v name="$2" '$2 == name {print; ++found} END {if (found != 1) exit 1}' SHA256SUMS |
        sha256sum --strict --quiet -c -
)

case "$mode" in
build)
    [[ $# == 2 ]]
    build="$study/build-$version"
    pin b125d17182c8751dcaa06bf1d3b268544b6b0757800c546b806fa69f08ff51a9 "$packet"
    pin 57bdafd1e0171a3493a93a66079437ca8101530d95d5096c0e2115233ac3817c "$assets/source.sha256"
    (cd "$assets/source" && sha256sum --strict --quiet -c ../source.sha256)
    mkdir "$build"
    cp "$here/"{Cargo.toml,Cargo.lock,enroll.rs,prepare-wick.sh} "$build/"
    cp "$packet" "$build/packet.rs"
    ln -s "$assets/source/crates/uor-r4-tokenizer" "$build/tokenizer"
    (cd "$build" &&
        "$HOME/.cargo/bin/cargo" build --offline --locked --release --target-dir "$build/target" > build.log 2>&1 &&
        sha256sum Cargo.toml Cargo.lock enroll.rs prepare-wick.sh packet.rs > method.sha256 &&
        sha256sum method.sha256 target/release/original-occurrence-panel > SHA256SUMS &&
        sha256sum --strict --quiet -c method.sha256 SHA256SUMS)
    ;;
prepare)
    [[ $# == 3 && $3 =~ ^v[1-9][0-9]*$ ]]
    build="$study/build-$3"
    out="$study/panel-$version"
    [[ ! -e $out ]]
    partial=$(mktemp -d "$study/panel-$version.partial.XXXXXXXX")
    (cd "$build" && sha256sum --strict --quiet -c method.sha256 SHA256SUMS)
    pin fe7485ee9e0a7073a7530e71aadf7eafad95a77ce8e3ca3bdbac32feb19c90bb "$measurement/SHA256SUMS"
    pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$mixed/SHA256SUMS"
    pin f72d72523f8faaabb7b0e53997df5b7a9bb059f5e068a1102eb96bb594cc9bb8 "$residual/SHA256SUMS"
    pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
    member "$residual" summary.json
    for f in reference.json blind/coordinates.jsonl blind/text.bin links.bin distance.txt; do
        member "$mixed" "./mixed/$f"
    done
    for prefix in "$old" "$current"; do
        for ext in state reader.jsonl edges.jsonl; do
            member "$measurement" "./${prefix#"$measurement"/}.$ext"
        done
    done
    for f in selection.json states/protected-first.state states/protected.state; do
        member "$measurement" "./$f"
    done
    cmp "$old.state" "$measurement/states/protected-first.state"
    cmp "$current.state" "$measurement/states/protected.state"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$current.state"
    jq -e '.protected_first.id == 234 and .protected.id == 240' "$measurement/selection.json" >/dev/null

    panel() {
        "$build/target/release/original-occurrence-panel" "$1" "$mixed/mixed" \
            "$residual/summary.json" "$old" "$current" "$codec" "$2"
    }
    panel prepare "$partial"
    panel check "$partial/masks.tsv"
    IFS=$'\t' read -r row gold < <(
        jq -r -s '.[0] | [.row, .gold_tuple_sites[.failed_sign]] | @tsv' "$partial/rows.jsonl"
    )
    negative() {
        local name=$1 field=$2 replacement=$3
        awk -v row="$row" -v field="$field" -v value="$replacement" \
            'BEGIN {OFS="\t"} NR==row+1 {$field=value} {print}' "$partial/masks.tsv" > "$partial/$name.tsv"
        if panel check "$partial/$name.tsv" >"$partial/$name.stderr" 2>&1; then
            printf 'Invalid %s mask was accepted\n' "$name" >&2
            exit 1
        fi
        rg -Fq "row $row misaddressed or noneligible" "$partial/$name.stderr"
    }
    negative misaddressed-winner 2 128
    negative ineligible-control 3 128
    negative gold-control 3 "$gold"

    jq -sce '{
        prompts:length, unique_pairs:(map(.pair_id)|unique|length),
        fixed_background_rows:(256-length),
        failed_signs:(group_by(.failed_sign)|map({sign:.[0].failed_sign,rows:length})),
        zero_controls:map(select(.control_site == -1)|.row),
        control_surfaces:(map(select(.control != null) | .control.surface)|group_by(.)|
            map({surface:.[0],count:length})),
        state_234_and_240_signed_tie_lists_match:all(.[]; .prior_234_strongest_signed_rivals==.strongest_signed_rivals)
    }' "$partial/rows.jsonl" > "$partial/receipt.json"
    cp "$build/method.sha256" "$partial/method.sha256"
    sha256sum "$mixed/SHA256SUMS" "$mixed/mixed/"{reference.json,blind/coordinates.jsonl,blind/text.bin,links.bin,distance.txt} \
        "$residual/"{SHA256SUMS,summary.json} "$measurement/"{SHA256SUMS,selection.json} \
        "$old."{state,reader.jsonl,edges.jsonl} "$current."{state,reader.jsonl,edges.jsonl} \
        "$codec" "$assets/source.sha256" "$packet" > "$partial/inputs.sha256"
    (cd "$partial" &&
        sha256sum masks.tsv rows.jsonl receipt.json method.sha256 inputs.sha256 {misaddressed-winner,ineligible-control,gold-control}.{tsv,stderr} > SHA256SUMS &&
        sha256sum --strict --quiet -c SHA256SUMS inputs.sha256)
    chmod -R a-w "$partial"
    mv -T --no-clobber "$partial" "$out"
    test ! -e "$partial"
    sha256sum "$out/"{SHA256SUMS,masks.tsv,rows.jsonl,receipt.json}
    cat "$out/receipt.json"
    ;;
*)
    exit 64
    ;;
esac
