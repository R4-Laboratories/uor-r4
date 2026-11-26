#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/archive-lib.sh"
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-fixed-assignment-median"
storage="$study/storage"
repo=caseyallard/uor-r4-store
hf="$HOME/.local/bin/hf"
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs &&
    ${here%/*} == "$storage" && ${here##*/} =~ ^methods-v[1-9][0-9]*$ ]]

action=${1:?fixture, pack, local, publish, or download}
av=${2:?new archive, fixture, or download version}
version "$av"
archive="$storage/archive-$av"
upload="$archive/upload"
tools=${here#"$study"/}

read_layout() {
    jq -er --arg key "$1" --arg re "$2" '.[$key] | select(type=="string" and test($re))' "$layout"
}
read_audit_build() {
    local manifest=$1
    awk -v prefix="$study/audit/build-" '
        index($2,prefix)==1 && $2 ~ /\/SHA256SUMS$/ { print $2; n++ }
        END { if (n!=1) exit 1 }' "$manifest"
}

declare -A included=() known=()
declare -a seals=()
add_file() {
    local path=$1 rel=${1#"$study"/}
    [[ $path == "$study/"* ]]
    if [[ ${included[$rel]+yes} ]]; then return; fi
    require_regular "$study" "$rel"
    included["$rel"]=1
    case ${path##*/} in
        SHA256SUMS|inputs.sha256|unique-inputs.sha256|selection.sha256|authority.sha256|operator.sha256|dispatch.sha256|artifacts.sha256|dependencies.sha256|ancestor.sha256|antecedents.sha256)
            seals+=("$path") ;;
    esac
}
add_dir() { add_file "$1/SHA256SUMS"; }
follow_seals() {
    local i manifest hash name file base
    for ((i = 0; i < ${#seals[@]}; ++i)); do
        manifest=${seals[i]}
        base=${manifest%/*}
        if [[ $manifest == "$study"/report-v*/inputs.sha256 ]]; then base=$study; fi
        while read -r hash name; do
            [[ $hash =~ ^[0-9a-f]{64}$ ]]
            if [[ $name == /* ]]; then
                file=$(realpath -e -- "$name")
            else
                name=${name#./}; safe_member "$name"
                file="$base/$name"
            fi
            [[ $file == "$runs/"* ]]
            if [[ ${known[$file]+yes} ]]; then
                [[ ${known[$file]} == "$hash" ]]
            else
                pin "$hash" "$file"
                known["$file"]=$hash
            fi
            # Retain study dependencies. External paths stay in the originating
            # receipt. Only material needed for cold replay is copied below.
            if [[ $file == "$study/"* ]]; then add_file "$file"; fi
        done < "$manifest"
    done
}
copy_ancestry() {
    local source=$1 name=$2 expected=$3
    safe_member "$name"
    pin "$expected" "$source"
    mkdir -p "$(dirname "$archive/ancestry/$name")"
    cp -p --reflink=auto "$source" "$archive/ancestry/$name"
    add_file "$archive/ancestry/$name"
    sha256sum "$source" >> "$archive/ancestry/source.sha256"
}
cold_restore() {
    local packet=$1 out=$2 layout_av=$3
    extract_packet "$packet" "$out/extracted"
    bash "$out/extracted/$tools/restore-grade-wick.sh" "$out/extracted" \
        "$out/extracted/storage/archive-$layout_av/layout.json" "$out/replayed"
    sha256sum "$packet/SHA256SUMS" "$packet/measurement.tar.zst" "$out/replayed/receipt.json" > "$out/readback.sha256"
}

check_receipt() {
    jq -e --arg repo "$repo" '
        .schema=="uor-r4.codex-4299.store/1" and .repo==$repo and
        (.archive_version|test("^v[1-9][0-9]*$")) and (.revision|test("^[0-9a-f]{40}$")) and
        ([(.archive_sha256),(.packet_sha256),(.members_sha256),(.cold_replay_sha256)] |
            all(.[]; test("^[0-9a-f]{64}$"))) and
        .path==("results/codex/fixed-assignment-median-20261126-"+.archive_sha256[:12])' "$1" >/dev/null
}
download() {
    local out=$1 receipt=$2 revision prefix packet layout_av
    check_receipt "$receipt"
    revision=$(jq -er .revision "$receipt"); prefix=$(jq -er .path "$receipt")
    layout_av=$(jq -er .archive_version "$receipt")
    "$hf" download "$repo" --repo-type dataset --revision "$revision" --include "$prefix/*" \
        --local-dir "$out/download" --force-download --quiet > "$out/download.log"
    packet="$out/download/$prefix"
    pin "$(jq -er .archive_sha256 "$receipt")" "$packet/measurement.tar.zst"
    pin "$(jq -er .packet_sha256 "$receipt")" "$packet/SHA256SUMS"
    pin "$(jq -er .members_sha256 "$receipt")" "$packet/MEMBERS.sha256"
    [[ $(jq -er .members "$receipt") == "$(wc -l < "$packet/MEMBERS.sha256")" ]]
    cold_restore "$packet" "$out" "$layout_av"
    pin "$(jq -er .cold_replay_sha256 "$receipt")" "$out/replayed/receipt.json"
}

case $action in
fixture)
    [[ $# == 2 ]]
    bash "$here/fixture-wick.sh" "$storage/fixture-$av"
    ;;
pack)
    [[ $# == 3 ]]
    layout=$(realpath -e "$3")
    jq -e '.schema=="uor-r4.codex-4299.archive-layout/1" and
        (.additional_seals|type)=="array" and all(.additional_seals[];type=="string")' "$layout" >/dev/null
    methods="$study/$(read_layout methods '^methods-v[1-9][0-9]*$')"
    panel="$study/$(read_layout panel '^panel/panel-v[1-9][0-9]*$')"
    train="$study/$(read_layout train '^train-v[1-9][0-9]*$')"
    held="$study/$(read_layout held '^held-v[1-9][0-9]*$')"
    grade="$study/$(read_layout grade '^grade-v[1-9][0-9]*$')"
    release="$study/$(read_layout release '^release-v[1-9][0-9]*$')"
    permit="$study/$(read_layout permit '^permit-v[1-9][0-9]*$')"
    train_audit="$study/$(read_layout train_audit '^audit/train-v[1-9][0-9]*$')"
    held_audit="$study/$(read_layout held_audit '^audit/held-v[1-9][0-9]*$')"
    adapter="$study/$(read_layout adapter '^client/methods-v[1-9][0-9]*$')"
    fixture="$study/$(read_layout client_fixture '^client/fixture-v[1-9][0-9]*$')"
    cv=$(read_layout writer_version '^v[1-9][0-9]*$')
    verify "$release"; verify "$permit"
    cmp "$release/panel-path" <(printf '%s\n' "$panel")
    cmp "$release/client-path" <(printf '%s\n' "$adapter")
    for file in "$panel/SHA256SUMS" "$methods/SHA256SUMS" "$adapter/SHA256SUMS"; do
        listed "$release/inputs.sha256" "$file"
    done
    listed "$train/inputs.sha256" "$release/inputs.sha256"
    for file in "$train/selection.json" "$train/selection.sha256" "$train/SHA256SUMS" "$methods/SHA256SUMS" \
        "$permit/verified.json" "$train_audit/SHA256SUMS"; do
        listed "$permit/inputs.sha256" "$file"
    done
    jq -e '.pass==true' "$permit/verified.json" >/dev/null
    for file in "$permit/inputs.sha256" "$train/SHA256SUMS" "$panel/SHA256SUMS" "$methods/SHA256SUMS"; do
        listed "$held/inputs.sha256" "$file"
    done
    for file in "$train/SHA256SUMS" "$held/SHA256SUMS" "$panel/SHA256SUMS" "$methods/SHA256SUMS"; do
        listed "$grade/inputs.sha256" "$file"
    done
    [[ $(cat "$train-lease/exit-code.txt") == 0 && $(cat "$held-lease/exit-code.txt") == 0 ]]
    jq -e '.pass==true and .assignments==512' "$train_audit/readback.json" >/dev/null
    jq -e '.pass==true' "$held_audit/readback.json" >/dev/null
    for file in "$train/SHA256SUMS" "$held/SHA256SUMS" "$grade/SHA256SUMS" "$panel/SHA256SUMS" "$train_audit/SHA256SUMS"; do
        listed "$held_audit/inputs.sha256" "$file"
    done
    train_build=$(read_audit_build "$train_audit/inputs.sha256")
    held_build=$(read_audit_build "$held_audit/inputs.sha256")
    readarray -t extra < <(jq -er '.additional_seals[]' "$layout")
    for dir in "${extra[@]}"; do
        [[ $dir == audit/* || $dir == fit/* || $dir =~ ^(core-fixture|report|report-methods)-v[1-9][0-9]*$ ||
            $dir =~ ^analysis/(train|held)-v[1-9][0-9]*$ ]]
        safe_member "$dir"
        add_dir "$study/$dir"
    done

    writer_status=$(jq -er 'if .pass and (.passing|length)>0 then "CAPTURED"
        elif (.pass|not) and .passing==[] then "NOT_RUN" else error("invalid gate result") end' "$grade/summary.json")
    case $writer_status in
    CAPTURED)
        for dir in capture writer writer-"$cv"-lease tokens; do
            if [[ $dir == *-lease ]]; then add_dir "$study/client/$dir"; else add_dir "$study/client/$dir-$cv"; fi
        done
        listed "$study/client/tokens-$cv/inputs.sha256" "$grade/summary.json"
        ;;
    NOT_RUN)
        add_dir "$study/client/refusal-$cv"
        listed "$study/client/refusal-$cv/inputs.sha256" "$grade/summary.json"
        jq -e '.status=="NOT_RUN" and .passing==[]' "$study/client/refusal-$cv/receipt.json" >/dev/null
        ;;
    esac
    mkdir "$archive"
    jq --arg train_build "${train_build#"$study"/}" --arg held_build "${held_build#"$study"/}" \
        --arg writer_status "$writer_status" '.+{train_build:$train_build,held_build:$held_build,writer_status:$writer_status}' \
        "$layout" > "$archive/layout.json"
    add_file "$archive/layout.json"
    historical="$runs/20261126-codex-protected-transport/measurement-v1"
    copy_ancestry "$historical/selection.json" historical/selection.json \
        a8ecc6b2552de1c423a6075674292fbfa92dd42b4f2c1f08211863e1db3db46e
    copy_ancestry "$historical/premise.json" historical/premise.json \
        5543f2c9da1a7841773f2a7b3e74b0893bd331844456284f25ba08579572dadc
    if [[ $writer_status == CAPTURED ]]; then
        donor="$runs/20261125-codex-bound-instruction-state"
        prior="$donor/ordinary-recipe-final-v1"
        copy_ancestry "$prior/raw.jq" client/raw.jq 7b98ba6438c951edef3acc7938d8d47de3d8f008d0f2d4657d9e13dcd4174433
        copy_ancestry "$prior/score.jq" client/score.jq f98ff93e98a38873f747b0fa9df57f45ee93929ef7cb0caea69cb7e2c2311f98
        copy_ancestry "$donor/custody-v1/signed/tokenizer.json" client/tokenizer.json \
            a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
    fi
    add_file "$archive/ancestry/source.sha256"
    add_dir "$here"
    for dir in "$train" "$held" "$train-lease" "$held-lease" "$grade" "$methods" "$panel" \
        "$release" "$permit" "$train_audit" "$held_audit" "$adapter" "$fixture"; do add_dir "$dir"; done
    follow_seals
    printf '%s\n' "${!included[@]}" | sort > "$archive/members.txt"
    write_packet "$study" "$archive/members.txt" "$upload"
    jq -n --arg version "$av" --arg archive_sha "$(digest "$upload/measurement.tar.zst")" \
        --arg packet "$(digest "$upload/SHA256SUMS")" --arg members_sha "$(digest "$upload/MEMBERS.sha256")" \
        --argjson members "$(wc -l < "$archive/members.txt")" \
        --argjson bytes "$(stat -c %s "$upload/measurement.tar.zst")" '
        {archive_version:$version,archive_sha256:$archive_sha,packet_sha256:$packet,
         members_sha256:$members_sha,members:$members,archive_bytes:$bytes}' > "$archive/pack.json"
    chmod a-w "$archive/"{layout.json,members.txt,pack.json}
    cat "$archive/pack.json"
    ;;
local)
    rv=${3:?new local replay version}; [[ $# == 3 ]]; version "$rv"
    out="$archive/local-$rv"; mkdir "$out"
    cold_restore "$upload" "$out" "$av"
    cat "$out/replayed/receipt.json"
    ;;
publish)
    rv=${3:?new immutable readback version}; lv=${4:?successful local replay version}
    [[ $# == 4 ]]; version "$rv"; version "$lv"
    local_restore="$archive/local-$lv"
    sha256sum --strict --quiet -c "$local_restore/readback.sha256"
    listed "$local_restore/readback.sha256" "$upload/measurement.tar.zst"
    jq -e '.pass and .disagreements==0 and .held_lanes==26 and
        .train_elections==2 and .independent_train and .independent_held' "$local_restore/replayed/receipt.json" >/dev/null
    verify "$upload"
    out="$archive/published-$rv"; mkdir "$out"
    "$hf" datasets info "$repo" --json | jq -e 'select(.private==true)|{private}' > "$out/privacy.json"
    archive_sha=$(digest "$upload/measurement.tar.zst")
    prefix="results/codex/fixed-assignment-median-20261126-${archive_sha:0:12}"
    "$hf" upload "$repo" "$upload" "$prefix" --repo-type dataset \
        --commit-message '[codex] Preserve fixed-assignment median evidence' --format json > "$out/upload.json"
    revision=$(jq -er --arg prefix "https://huggingface.co/datasets/$repo/commit/" '
        .url|select(startswith($prefix))|ltrimstr($prefix)|select(test("^[0-9a-f]{40}$"))' "$out/upload.json")
    jq --arg repo "$repo" --arg path "$prefix" --arg revision "$revision" \
        --arg cold "$(digest "$local_restore/replayed/receipt.json")" '
        .+{schema:"uor-r4.codex-4299.store/1",repo:$repo,path:$path,revision:$revision,cold_replay_sha256:$cold}' \
        "$archive/pack.json" > "$out/upload-receipt.json"
    download "$out" "$out/upload-receipt.json"
    cmp "$local_restore/replayed/receipt.json" "$out/replayed/receipt.json"
    jq '.+{verified_immutable_readback:true}' "$out/upload-receipt.json" > "$out/receipt.json"
    cat "$out/receipt.json"
    ;;
download)
    [[ $# == 3 ]]
    receipt=$(realpath -e "$3")
    out="$storage/download-$av"; mkdir "$out"
    download "$out" "$receipt"
    cat "$out/replayed/receipt.json"
    ;;
*) exit 64 ;;
esac
