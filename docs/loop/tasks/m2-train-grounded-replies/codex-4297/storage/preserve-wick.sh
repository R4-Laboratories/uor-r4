#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/archive-lib.sh"
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-all-outranker-interval"
storage="$study/storage"
repo=caseyallard/uor-r4-store
hf="$HOME/.local/bin/hf"
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs &&
    $here == "$storage"/methods-v* && ${here##*/} =~ ^methods-v[1-9][0-9]*$ ]]

action=${1:?pack, local, publish, download, or fixture}
av=${2:?new archive, download or fixture version}
version "$av"
archive="$storage/archive-$av"
upload="$archive/upload"
metadata="$archive/layout.json"
lib_rel=${here#"$study"/}

one_ref() {
    local file=$1 pattern=$2
    awk -v pattern="$pattern" '$2 ~ pattern {print $2; count++} END {if (count!=1) exit 1}' "$file"
}
under_study() {
    [[ $1 == "$study/"* ]] && safe_member "${1#"$study"/}"
}
trim_study() { under_study "$1" || return; printf '%s\n' "${1#"$study"/}"; }

# Keep the closure of selected study seals. External inputs are copied only
# when they are named by a reviewed study manifest or explicitly used for replay.
declare -A included=() checked_hash=()
declare -a manifests=()
members_expected=0
add_file() {
    local file=$1 rel
    rel=$(trim_study "$file")
    if [[ ${included[$rel]+yes} ]]; then return; fi
    require_regular "$study" "$rel"
    included["$rel"]=1
    members_expected=$((members_expected + 1))
    case ${file##*/} in
        SHA256SUMS|inputs.sha256|authority.sha256|dependencies.sha256|inherited.sha256|selection.sha256|previous.sha256|prior.sha256|readback.sha256)
            # Mirrored external seals remain provenance. Their prior study owns
            # any additional ancestors and details its own restoration process.
            if [[ $file != "$archive"/ancestry/* ]]; then manifests+=("$file"); fi
            ;;
    esac
}
add_dir() {
    local dir=$1
    add_file "$dir/SHA256SUMS"
    if [[ -f "$dir/authority.sha256" ]]; then add_file "$dir/authority.sha256"; fi
}
mirror() {
    local file=$1 rel copy
    [[ $file == "$runs/"* && $file != "$study/"* ]]
    rel=${file#"$runs"/}
    require_regular "$runs" "$rel"
    copy="$archive/ancestry/$rel"
    if [[ ! -f $copy ]]; then
        mkdir -p "$(dirname "$copy")"
        cp -p --reflink=auto "$file" "$copy"
    fi
    add_file "$copy"
}
check_ref() {
    local hash=$1 file=$2
    [[ $hash =~ ^[0-9a-f]{64}$ && $file == "$runs/"* ]]
    if [[ ${checked_hash[$file]+yes} ]]; then
        [[ ${checked_hash[$file]} == "$hash" ]]
    else
        pin "$hash" "$file"
        checked_hash["$file"]=$hash
    fi
    if [[ $file == "$study/"* ]]; then
        add_file "$file"
        if [[ ${file##*/} == SHA256SUMS && -f "${file%/*}/authority.sha256" ]]; then
            add_file "${file%/*}/authority.sha256"
        fi
    else
        mirror "$file"
    fi
}
follow_seals() {
    local i file hash name
    for ((i = 0; i < ${#manifests[@]}; ++i)); do
        file=${manifests[i]}
        while read -r hash name; do
            [[ $hash =~ ^[0-9a-f]{64}$ ]]
            if [[ $name != /* ]]; then
                name=${name#./}; safe_member "$name"
                name="${file%/*}/$name"
            fi
            check_ref "$hash" "$name"
        done < "$file"
    done
}
cold_restore() {
    local packet=$1 out=$2 layout_ver=$3
    extract_packet "$packet" "$out/extracted"
    bash "$out/extracted/$lib_rel/restore-grade-wick.sh" "$out/extracted" \
        "$out/extracted/storage/archive-$layout_ver/layout.json" "$out/replayed"
    sha256sum "$packet/SHA256SUMS" "$packet/measurement.tar.zst" "$out/replayed/receipt.json" > "$out/readback.sha256"
}
check_receipt() {
    local receipt=$1
    jq -e --arg repo "$repo" '
       .schema=="uor-r4.codex-4297.store/1" and .repo==$repo and
       (.archive_version|test("^v[1-9][0-9]*$")) and (.revision|test("^[0-9a-f]{40}$")) and
       ([(.archive_sha256),(.packet_sha256),(.members_sha256),(.cold_replay_sha256)]|all(.[]; test("^[0-9a-f]{64}$"))) and
       .path==("results/codex/all-outranker-interval-20261126-"+.archive_sha256[:12])' "$receipt" >/dev/null
}
download() {
    local out=$1 receipt=$2 retrieved revision prefix layout_ver
    check_receipt "$receipt"
    revision=$(jq -er .revision "$receipt"); prefix=$(jq -er .path "$receipt"); layout_ver=$(jq -er .archive_version "$receipt")
    "$hf" download "$repo" --repo-type dataset --revision "$revision" --include "$prefix/*" \
        --local-dir "$out/download" --force-download --quiet > "$out/download.log"
    retrieved="$out/download/$prefix"
    pin "$(jq -er .archive_sha256 "$receipt")" "$retrieved/measurement.tar.zst"
    pin "$(jq -er .packet_sha256 "$receipt")" "$retrieved/SHA256SUMS"
    pin "$(jq -er .members_sha256 "$receipt")" "$retrieved/MEMBERS.sha256"
    cold_restore "$retrieved" "$out" "$layout_ver"
    pin "$(jq -er .cold_replay_sha256 "$receipt")" "$out/replayed/receipt.json"
    [[ $(jq -er .members "$receipt") == "$(wc -l < "$retrieved/MEMBERS.sha256")" ]]
}

case $action in
pack)
    mv=${3:?measurement version}; hv=${4:?held version}; gv=${5:?grade version}; cv=${6:?client or refusal version}
    hav=${7:?independent held audit version}
    for v in "$mv" "$hv" "$gv" "$cv" "$hav"; do version "$v"; done
    shift 7
    m="$study/measurement-$mv"; raw="$study/held-$hv"; grade="$study/grade-$gv"
    for dir in "$m" "$raw" "$grade" "$m-lease" "$raw-lease"; do verify "$dir"; done
    [[ $(cat "$m-lease/exit-code.txt") == 0 && $(cat "$raw-lease/exit-code.txt") == 0 ]]
    release_manifest=$(one_ref "$m/inputs.sha256" "^$study/release-v[1-9][0-9]*/inputs[.]sha256$")
    release=${release_manifest%/*}
    train_methods_manifest=$(one_ref "$release_manifest" "^$study/methods-v[1-9][0-9]*/SHA256SUMS$")
    train_methods=${train_methods_manifest%/*}
    methods_manifest=$(one_ref "$grade/inputs.sha256" "^$study/methods-v[1-9][0-9]*/SHA256SUMS$")
    methods=${methods_manifest%/*}
    panel=$(cat "$release/panel-path")
    [[ $panel == "$study"/panel/panel-v* ]]
    fit=$(dirname "$(cat "$release/fit-path")"); [[ $fit == "$study"/fit/build-v* ]]
    packer=$(cat "$release/packer-path"); [[ $packer == "$study"/finite-core-v*/bundle ]]
    permit=$(one_ref "$raw/inputs.sha256" "^$study/train-verified-v[1-9][0-9]*/inputs[.]sha256$")
    [[ $(jq -er '.verified and .batches==6|tostring' "${permit%/*}/verified.json") == true ]]
    audit_manifest=$(one_ref "$permit" "^$study/audit/train-v[1-9][0-9]*/SHA256SUMS$")
    audit=${audit_manifest%/*}
    audit_build_manifest=$(one_ref "$audit/inputs.sha256" "^$study/audit/build-v[1-9][0-9]*/SHA256SUMS$")
    audit_build=${audit_build_manifest%/*}
    held_audit="$study/audit/held-$hav"
    held_build_manifest=$(one_ref "$held_audit/inputs.sha256" "^$study/audit/build-v[1-9][0-9]*/SHA256SUMS$")
    held_build=${held_build_manifest%/*}
    for dir in "$methods" "$train_methods" "$panel" "$fit" "${packer%/*}" "$held_audit" "$held_build"; do verify "$dir"; done
    jq -e '.pass==true' "$held_audit/readback.json" >/dev/null
    for file in "$m/SHA256SUMS" "$panel/SHA256SUMS"; do
        listed "$raw/inputs.sha256" "$file"
        listed "$grade/inputs.sha256" "$file"
    done
    listed "$raw/inputs.sha256" "$train_methods_manifest"
    listed "$permit" "$train_methods_manifest"
    for file in grade/{candidate,witness,elect,gate,contrast}.jq original-{join,pairs,summary}.jq; do
        cmp "$train_methods/$file" "$methods/$file"
    done
    listed "$grade/inputs.sha256" "$raw/SHA256SUMS"
    listed "$permit" "$m/SHA256SUMS"
    for file in "$m/SHA256SUMS" "$raw/SHA256SUMS" "$grade/SHA256SUMS" "$panel/SHA256SUMS"; do
        listed "$held_audit/inputs.sha256" "$file"
    done
    artifacts=()
    case $(jq -er '.pass|select(type=="boolean")|tostring' "$grade/summary.json") in
        true)
            writer=CAPTURED
            artifacts=("$study/client/"{capture,writer,tokens}"-$cv")
            listed "$study/client/tokens-$cv/inputs.sha256" "$grade/summary.json"
            ;;
        false)
            writer=NOT_RUN
            artifacts=("$study/client/refusal-$cv")
            jq -e '.status=="NOT_RUN" and .passing==[]' "${artifacts[0]}/receipt.json" >/dev/null
            listed "${artifacts[0]}/inputs.sha256" "$grade/summary.json"
            ;;
        *) exit 65 ;;
    esac
    failed_grade=
    if [[ $gv != v1 && -d "$study/grade-v1" && ! -e "$study/grade-v1/SHA256SUMS" ]]; then
        failed_grade=grade-v1
    fi
    mkdir "$archive"
    jq -n --arg m "$(trim_study "$m")" --arg held "$(trim_study "$raw")" --arg grade "$(trim_study "$grade")" \
        --arg methods "$(trim_study "$methods")" --arg train_methods "$(trim_study "$train_methods")" \
        --arg panel "$(trim_study "$panel")" --arg failed "$failed_grade" \
        --arg release "$(trim_study "$release")" --arg fit "$(trim_study "$fit")" \
        --arg permit "$(trim_study "${permit%/*}")" --arg packer "$(trim_study "$packer")" \
        --arg audit "$(trim_study "$audit")" --arg audit_build "$(trim_study "$audit_build")" \
        --arg held_audit "$(trim_study "$held_audit")" --arg held_build "$(trim_study "$held_build")" \
        --arg client "$cv" --arg status "$writer" \
        '{schema:"uor-r4.codex-4297.archive-layout/1",measurement:$m,held:$held,grade:$grade,
         methods:$methods,train_methods:$train_methods,panel:$panel,release:$release,fit:$fit,
         failed_partial_grade:(if $failed=="" then null else $failed end),train_permit:$permit,packer:$packer,
         train_audit:$audit,audit_build:$audit_build,held_audit:$held_audit,held_audit_build:$held_build,
         client:{version:$client,status:$status}}' > "$metadata"
    add_file "$metadata"
    add_dir "$here"
    for dir in "$m" "$m-lease" "$raw" "$raw-lease" "$grade" "$methods" "$panel" "$fit" "${packer%/*}" "${artifacts[@]}"; do
        add_dir "$dir"
    done
    add_dir "$held_audit"
    if [[ -n $failed_grade ]]; then
        while IFS= read -r -d '' file; do add_file "$file"; done < <(find "$study/$failed_grade" -type f -print0)
    fi
    add_file "$release_manifest"
    add_file "$permit"
    # Preserve the first failed physical fixture receipt and its exact method.
    add_dir "$study/fit/build-v1"
    add_dir "$study/fit/finite-v1-lease"
    add_file "$study/fit/finite-v1/all/1.reader.jsonl"
    add_file "$study/fit/finite-v1/strongest/1.reader.jsonl"
    add_file "$study/fit/finite-v1/all/base.reader.jsonl"
    for entry in "$@"; do
        safe_member "$entry"
        [[ $entry != storage && $entry != storage/* ]]
        if [[ -d "$study/$entry" ]]; then add_dir "$study/$entry"; else add_file "$study/$entry"; fi
    done
    # These two cold-readback dependencies are public to the study even when
    # the release lists only the parent manifests.
    bank="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
    typed="$runs/20261126-codex-text-governor-links/parity/build-v2/release/textlinks-trace-audit"
    inherited_cuda="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4278/fit/device.cu"
    for file in "$bank" "$typed" "$inherited_cuda"; do mirror "$file"; done
    follow_seals
    printf '%s\n' "${!included[@]}" | sort > "$archive/members.txt"
    [[ $members_expected == "$(wc -l < "$archive/members.txt")" ]]
    write_packet "$study" "$archive/members.txt" "$upload"
    source_bytes=$(cd "$study" && xargs -d '\n' stat -c %s < "$archive/members.txt" | awk '{sum+=$1} END {printf "%.0f\n",sum}')
    jq -n --arg version "$av" --arg sha "$(digest "$upload/measurement.tar.zst")" \
        --arg packet "$(digest "$upload/SHA256SUMS")" --arg members_sha "$(digest "$upload/MEMBERS.sha256")" \
        --argjson members "$members_expected" --argjson archive "$(stat -c %s "$upload/measurement.tar.zst")" \
        --argjson source_bytes "$source_bytes" --argjson allocated_bytes "$(du -s -B1 "$archive" | cut -f1)" \
        --argjson uploaded "$(du -bc "$upload/"* | awk 'END{print $1}')" '
        {archive_version:$version,archive_sha256:$sha,packet_sha256:$packet,members_sha256:$members_sha,
         members:$members,source_member_bytes:$source_bytes,archive_bytes:$archive,
         upload_bytes:$uploaded,pack_zfs_allocated_bytes:$allocated_bytes}' > "$archive/pack.json"
    chmod a-w "$archive/"{layout.json,members.txt,pack.json}
    cat "$archive/pack.json"
    ;;
local)
    rv=${3:?new restore version}; [[ $# == 3 ]]; version "$rv"
    out="$archive/local-$rv"; mkdir "$out"
    cold_restore "$upload" "$out" "$av"
    cat "$out/replayed/receipt.json"
    ;;
publish)
    rv=${3:?new publication readback version}; lv=${4:?successful local restore version}
    [[ $# == 4 ]]; version "$rv"; version "$lv"
    local_restore="$archive/local-$lv"
    sha256sum --strict --quiet -c "$local_restore/readback.sha256"
    listed "$local_restore/readback.sha256" "$upload/measurement.tar.zst"
    jq -e '.pass and .disagreements==0 and .held_sets==2 and .train_elections==6' "$local_restore/replayed/receipt.json" >/dev/null
    verify "$upload"
    out="$archive/published-$rv"; mkdir "$out"
    "$hf" datasets info "$repo" --json | jq -e 'select(.private==true)|{private}' > "$out/privacy.json"
    archive_sha=$(digest "$upload/measurement.tar.zst")
    prefix="results/codex/all-outranker-interval-20261126-${archive_sha:0:12}"
    "$hf" upload "$repo" "$upload" "$prefix" --repo-type dataset \
        --commit-message '[codex] Preserve matched original-occurrence transport evidence' --format json > "$out/upload.json"
    revision=$(jq -er --arg prefix "https://huggingface.co/datasets/$repo/commit/" '
        .url|select(startswith($prefix))|ltrimstr($prefix)|select(test("^[0-9a-f]{40}$"))' "$out/upload.json")
    jq --arg repo "$repo" --arg path "$prefix" --arg revision "$revision" \
        --arg cold "$(digest "$local_restore/replayed/receipt.json")" '
        .+{schema:"uor-r4.codex-4297.store/1",repo:$repo,path:$path,revision:$revision,cold_replay_sha256:$cold}' \
        "$archive/pack.json" > "$out/upload-receipt.json"
    download "$out" "$out/upload-receipt.json"
    cmp "$local_restore/replayed/receipt.json" "$out/replayed/receipt.json"
    jq '.+{verified_immutable_readback:true}' "$out/upload-receipt.json" > "$out/receipt.json"
    cat "$out/receipt.json"
    ;;
download)
    [[ $# == 3 ]]
    receipt=$(realpath -e "${3:?immutable upload receipt}")
    out="$storage/download-$av"; mkdir "$out"
    download "$out" "$receipt"
    cat "$out/replayed/receipt.json"
    ;;
fixture)
    [[ $# == 2 ]]
    bash "$here/fixture-wick.sh" "$storage/fixture-$av"
    ;;
*) exit 64 ;;
esac
