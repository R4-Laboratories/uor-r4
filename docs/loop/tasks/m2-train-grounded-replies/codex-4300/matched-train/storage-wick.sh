#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/archive-lib.sh"
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition
storage="$study/matched-train/storage"
mode=${1:?pack or local}
if [[ $mode != replay ]]; then version "${2:?archive version}"; fi
archive="$storage/archive-$2"

retain() {
    local dir=$1 hash leaf
    verify "$dir"
    while read -r hash leaf; do
        leaf=${leaf#./}
        [[ $hash =~ ^[0-9a-f]{64}$ ]]
        copy "$dir/$leaf"
    done < "$dir/SHA256SUMS"
    copy "$dir/SHA256SUMS"
}

copy() {
    local file=$1 member=${1#"$study"/}
    [[ $file == "$study/"* ]]
    require_regular "$study" "$member"
    mkdir -p -- "$snapshot/$(dirname -- "$member")"
    cp -p --reflink=auto "$file" "$snapshot/$member"
}

case "$mode" in
pack)
    [[ $# == 2 && $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$storage") == zfs && $here == "$storage"/methods-v* ]]
    primary="$study/storage/archive-v1"
    primary_receipt="$primary/local-v1/receipt.json"
    pin ba5e0c8a7452ec8ad64465df6c4ed517076f25b34e061ee756f419e78114f682 "$primary/packet/SHA256SUMS"
    verify "$primary/packet"
    verify "$primary/local-v1"
    pin d58a0bf85983fbfd4b2abb21eb41a0d80bfac58600b85659d03ff07d3104bde9 "$primary/local-v1/replayed/receipt.json"

    panel="$study/matched-train/panel-v1"
    physical="$study/matched-device/capture-layout-v1"
    replay="$study/matched-device/replay-v1"
    grade="$study/layout-grade-v1"
    comparison="$study/train-comparison-v1"
    release="$study/matched-device/authorization-v1"
    axes="$study/audit/selector-axes-v1"
    for dir in "$panel" "$physical" "$physical-lease" "$replay" "$grade" "$release" "$comparison" "$axes"; do verify "$dir"; done
    for dir in "$physical" "$replay" "$grade" "$axes"; do sha256sum --quiet --strict -c "$dir/inputs.sha256"; done
    for file in "$panel/SHA256SUMS" "$physical/SHA256SUMS" "$physical-lease/SHA256SUMS"; do
        listed "$replay/inputs.sha256" "$file"
    done
    listed "$grade/inputs.sha256" "$replay/rows.jsonl"
    listed "$grade/inputs.sha256" "$study/grade-v1/historical.rows.jsonl"
    listed "$grade/inputs.sha256" "$comparison/compare-train.jq"
    cmp "$physical/SHA256SUMS" "$physical-lease/raw.SHA256SUMS"
    [[ $(cat "$physical-lease/exit-code.txt") == 0 ]]
    jq -es --arg digest "$(digest "$physical/SHA256SUMS")" '
        . as [$earlier,$later] |
        $later.provider=="vast" and $later.phase=="layout" and
        $later.instance_id==$earlier.instance_id and $later.gpu_uuid==$earlier.gpu_uuid and
        $later.binary_sha256==$earlier.binary_sha256 and
        $later.raw_manifest_sha256==$digest and $later.payload_manifest_sha256==$digest
    ' "$study/capture-v1-lease/provider.json" "$physical-lease/provider.json" >/dev/null
    audit_manifest=$(awk -v prefix="$study/audit/build-" '
        index($2,prefix)==1 && $2 ~ /\/SHA256SUMS$/ {print $2; n++} END {if(n!=1)exit 1}
    ' "$replay/inputs.sha256")
    release_fit=$(awk -v prefix="$study/audit/layout-FIT-" '
        index($2,prefix)==1 && $2 ~ /\/SHA256SUMS$/ {print $2; n++} END {if(n!=1)exit 1}
    ' "$release/layout-auth/inputs.sha256")
    axes_manifest=$(awk -v prefix="$study/audit/build-" '
        index($2,prefix)==1 && $2 ~ /\/SHA256SUMS$/ {print $2; n++} END {if(n!=1)exit 1}
    ' "$axes/inputs.sha256")
    mkdir "$archive"
    snapshot="$archive/snapshot"; mkdir "$snapshot"
    for dir in "$panel" "$physical" "$physical-lease" "$replay" "$grade" "$comparison" "$release" \
        "$study/matched-device/"{recipe-v1,package-v1,import-v1} "$study/layout-registration-v1" \
        "$study/audit/"{layout-census-v1,layout-replay-v1} "$axes" \
        "${release_fit%/*}" "${audit_manifest%/*}" "${axes_manifest%/*}" "$here"; do
        retain "$dir"
    done
    while read -r hash file; do
        if [[ $file == "$study"/matched-train/methods-v*/* ]]; then
            pin "$hash" "$file"
            copy "$file"
        fi
    done < "$panel/authority.sha256"
    while read -r hash file; do
        if [[ $file == "$study"/audit/recipe-v*/axes-wick.sh ]]; then
            pin "$hash" "$file"
            copy "$file"
        fi
    done < "$axes/inputs.sha256"
    jq -n --arg primary_archive_sha256 "$(jq -er .archive_sha256 "$primary_receipt")" \
        --arg primary_packet_sha256 "$(digest "$primary/packet/SHA256SUMS")" \
        --arg primary_receipt_sha256 "$(digest "$primary_receipt")" \
        --arg audit "${audit_manifest#"$study"/}" --arg method "${here#"$study"/}" \
        --arg axes "${axes_manifest#"$study"/}" '
      {schema:"uor-r4.codex-4300.layout-cold/1",primary_archive_sha256:$primary_archive_sha256,
       primary_packet_sha256:$primary_packet_sha256,primary_receipt_sha256:$primary_receipt_sha256,
       audit:($audit|sub("/SHA256SUMS$";"")),axes:($axes|sub("/SHA256SUMS$";"")),method:$method}
    ' > "$snapshot/layout.json"
    (cd "$snapshot" && find . -type f -printf '%P\n' | sort) > "$archive/members.txt"
    write_packet "$snapshot" "$archive/members.txt" "$archive/packet"
    sha256sum "$archive/packet/"{SHA256SUMS,measurement.tar.zst}
    ;;
local)
    [[ $# == 3 && $(hostname -s) == wick && $here == "$storage"/methods-v* ]]; version "$3"
    out="$archive/local-$3"; mkdir "$out"
    extract_packet "$archive/packet" "$out/extracted"
    primary="$study/storage/archive-v1"
    pin "$(jq -er .primary_packet_sha256 "$out/extracted/layout.json")" "$primary/packet/SHA256SUMS"
    pin "$(jq -er .primary_archive_sha256 "$out/extracted/layout.json")" "$primary/packet/measurement.tar.zst"
    pin "$(jq -er .primary_receipt_sha256 "$out/extracted/layout.json")" "$primary/local-v1/receipt.json"
    extract_packet "$primary/packet" "$out/primary"
    recipe=$(jq -er '.method + "/storage-wick.sh"' "$out/extracted/layout.json")
    require_regular "$out/extracted" "$recipe"
    bash "$out/extracted/$recipe" replay "$out/extracted" "$out/primary" "$out/replayed"
    jq --arg archive_sha256 "$(digest "$archive/packet/measurement.tar.zst")" \
        --arg packet_sha256 "$(digest "$archive/packet/SHA256SUMS")" \
        --arg members_sha256 "$(digest "$archive/packet/MEMBERS.sha256")" \
        --argjson members "$(wc -l < "$archive/packet/MEMBERS.sha256")" \
        --argjson archive_bytes "$(stat -c %s "$archive/packet/measurement.tar.zst")" \
        --arg replay_sha256 "$(digest "$out/replayed/SHA256SUMS")" '
        .+{archive_sha256:$archive_sha256,packet_sha256:$packet_sha256,members_sha256:$members_sha256,
           members:$members,archive_bytes:$archive_bytes,replay_sha256:$replay_sha256}
    ' "$out/replayed/receipt.json" > "$out/receipt.json"
    (cd "$out" && sha256sum receipt.json replayed/SHA256SUMS > SHA256SUMS)
    cat "$out/receipt.json"
    ;;
replay)
    # The outer local command has verified both archive member lists and restored canonical bytes.
    [[ $# == 4 ]]
    root=$(realpath -e "$2"); primary=$(realpath -e "$3"); out=$4
    [[ $here == "$root/"* ]]
    layout="$root/layout.json"; previous="$primary/layout.json"
    jq -e '.schema=="uor-r4.codex-4300.layout-cold/1"' "$layout" >/dev/null
    primary_grade=$(jq -er .grade "$previous"); methods=$(jq -er .methods "$previous")
    audit=$(jq -er .audit "$layout")
    axes=$(jq -er .axes "$layout")
    require_regular "$root" "$audit/switch-audit"
    require_regular "$root" "$axes/switch-audit"
    require_regular "$primary" "$methods/grade/row.jq"
    require_regular "$primary" "$primary_grade/historical.rows.jsonl"
    panel="$root/matched-train/panel-v1"
    capture="$root/matched-device/capture-layout-v1"
    expected="$root/matched-device/replay-v1"
    ancestry="$primary/ancestry"
    before="$ancestry/methods-v2/inputs/protected.state"
    after="$ancestry/train-v1/states/ordered.state"
    bank="$ancestry/methods-v2/inputs/primary.words"
    codec="$ancestry/tokenizer.json"
    cmp "$capture/SHA256SUMS" "$capture-lease/raw.SHA256SUMS"
    (cd "$capture" && sha256sum --strict --quiet -c "$capture-lease/raw.SHA256SUMS")
    mkdir "$out"
    "$root/$audit/switch-audit" census "$panel" "$codec" "$out/census.json"
    "$root/$audit/switch-audit" replay "$panel" "$codec" "$before" "$bank" "$capture/before" - "$out/before.json"
    "$root/$audit/switch-audit" replay "$panel" "$codec" "$after" "$bank" "$capture/after" - "$out/after.json"
    "$root/$axes/switch-audit" geometry "$panel" "$codec" "$panel/endpoint/text.bin" "$out/axes.json"
    jq -nc --arg cohort matched-train --slurpfile meta "$panel/requests.jsonl" --slurpfile rows "$panel/rows.jsonl" \
        --slurpfile before "$capture/before/requests.jsonl" --slurpfile after "$capture/after/requests.jsonl" \
        -f "$primary/$methods/grade/row.jq" > "$out/rows.jsonl"
    jq -n --slurpfile historical "$primary/$primary_grade/historical.rows.jsonl" --slurpfile reads "$out/rows.jsonl" \
        -f "$root/train-comparison-v1/compare-train.jq" > "$out/summary.json"
    for file in census.json before.json after.json rows.jsonl; do cmp "$expected/$file" "$out/$file"; done
    for file in before.json after.json; do cmp "$root/audit/layout-replay-v1/$file" "$out/$file"; done
    cmp "$root/layout-grade-v1/summary.json" "$out/summary.json"
    cmp "$root/audit/selector-axes-v1/layout.json" "$out/axes.json"
    jq -n --arg primary "$(jq -er .primary_archive_sha256 "$layout")" \
        --arg summary "$(digest "$out/summary.json")" '
        {schema:"uor-r4.codex-4300.layout-zfs-store/1",pass:true,rows:8,pairs:4,
         independent_replays:2,identical_outputs:6,primary_archive_sha256:$primary,summary_sha256:$summary}
    ' > "$out/receipt.json"
    (cd "$out" && sha256sum census.json before.json after.json axes.json rows.jsonl summary.json receipt.json > SHA256SUMS)
    ;;
*) exit 64 ;;
esac
