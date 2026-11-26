#!/usr/bin/env bash
# Preserve the retired provider's unique evidence and restore any confirmed copies from Wick.
set -euo pipefail
umask 077
export LC_ALL=C

s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport
storage="$s/storage"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$storage"/vast-methods-v* ]]
sha(){ sha256sum "$1" | cut -d' ' -f1; }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
version_ok(){ [[ $1 =~ ^v[1-9][0-9]*$ ]]; }

mode=${1:?pack or restore}
v=${2:?supplement version}
version_ok "$v"
bundle="$storage/vast-archive-$v"
packet="$bundle/upload"
mirror="$s/fit/vast-capture-v2"
stage="$s/fit/vast-stage-v3"

append_file(){
    local file=$1
    [[ $file == "$s/"* && -f $file && ! -L $file ]]
    printf '%s\n' "${file#"$s"/}" >> "$bundle/members.pending"
}
append_tree(){
    local dir=$1 file
    if [[ -f $dir/SHA256SUMS ]]; then verify "$dir"; fi
    while IFS= read -r -d '' file; do append_file "$file"; done < <(find "$dir" -type f -print0)
}
append_or_share(){
    local file=$1 paired=$2
    if [[ -f $paired ]] && cmp -s "$file" "$paired"; then
        printf '%s\t%s\t%s\n' "$(sha "$file")" "${file#"$s"/}" "${paired#"$s"/}" >> "$bundle/shared.tsv"
    else append_file "$file"; fi
}
append_capture(){
    local file raw=$1 wick=$2
    verify "$raw"; verify "$wick"
    while IFS= read -r -d '' file; do
        if [[ $file == "$raw/"* && $file != "$raw/SHA256SUMS" ]]; then
            append_or_share "$file" "$wick/${file#"$raw"/}"
        else append_file "$file"; fi
    done < <(find "$mirror" -type f -print0 | sort -z)
}
append_grade(){
    local file dir=$1 wick_grade=$2
    verify "$dir"; verify "$wick_grade"
    while IFS= read -r -d '' file; do
        case "$file" in
            "$dir/records/"*|"$dir/candidates.jsonl"|"$dir/election.json")
                append_or_share "$file" "$wick_grade/${file#"$dir"/}" ;;
            *) append_file "$file" ;;
        esac
    done < <(find "$dir" -type f -print0 | sort -z)
}

case "$mode" in
pack)
    mv=${3:?primary measurement version}; pv=${4:?completed primary archive version}
    [[ $# == 4 ]]; version_ok "$mv"; version_ok "$pv"
    primary="$storage/archive-$pv/upload"
    wick="$s/measurement-$mv/train/root/device"
    raw="$mirror/train/root/device"
    verify "$primary"; verify "$mirror"; verify "$mirror/train"; verify "$mirror/finite"
    verify "$mirror/receipts/train"; verify "$mirror/receipts/finite"
    verify "$s/fit/vast-grade-v1"; verify "$s/audit/vast-train-v1"
    verify "$stage"
    test -s "$s/fit/vast-controller-v1/admission.txt"
    test -s "$s/fit/vast-controller-v1/jobs-and-reaper.tar.gz"
    mkdir "$bundle" "$packet"
    : > "$bundle/shared.tsv"
    append_capture "$raw" "$wick"
    raw_shared=$(wc -l < "$bundle/shared.tsv")
    append_grade "$s/fit/vast-grade-v1" "$s/measurement-$mv/train/root/grade"
    all_shared=$(wc -l < "$bundle/shared.tsv")
    while IFS=$'\t' read -r hash _ primary_member; do
        grep -Fxq "$hash  $primary_member" "$primary/MEMBERS.sha256"
    done < "$bundle/shared.tsv"
    for dir in fit/vast-controller-v1 fit/vast-inspection-v1 fit/vast-permit-v1 fit/hardware-comparison-v1 \
        audit/batch-v13 audit/vast-finite-v11 audit/vast-finite-v12 audit/vast-train-v1; do
        append_tree "$s/$dir"
    done
    # Retain the exact launched transfer. Rebuild its extracted payload from the saved tarball.
    while IFS= read -r -d '' file; do
        case "$file" in "$stage/payload/"*) continue ;; esac
        append_file "$file"
    done < <(find "$stage" -type f -print0)
    append_file "$here/preserve-vast-wick.sh"
    jq -n --arg archive_sha "$(sha "$primary/measurement.tar.zst")" \
        --arg primary_archive "archive-$pv" --arg measurement "measurement-$mv" \
        --arg wick "$(sha "$wick/SHA256SUMS")" --arg vast "$(sha "$raw/SHA256SUMS")" \
        --argjson shared "$raw_shared" --argjson all "$all_shared" --argjson raw "$(wc -l < "$raw/SHA256SUMS")" '
        {primary_archive:$primary_archive, primary_archive_sha256:$archive_sha, primary_measurement:$measurement,
        primary_raw_manifest_sha256:$wick, mirror_raw_manifest_sha256:$vast,
        mirror_raw_files:$raw, shared_raw_files:$shared, copied_raw_files:($raw-$shared), shared_grade_files:($all-$shared),
        all_raw_files_identical:($raw==$shared and $wick==$vast)}' > "$bundle/primary-link.json"
    append_file "$bundle/primary-link.json"; append_file "$bundle/shared.tsv"
    sort -u "$bundle/members.pending" > "$bundle/members.txt"
    (
        cd "$s"
        xargs -d '\n' sha256sum < "$bundle/members.txt" > "$packet/MEMBERS.sha256"
        tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
            -cf - -T "$bundle/members.txt" | zstd -q -15 -T1 -o "$packet/supplement.tar.zst"
    )
    (cd "$packet" && sha256sum MEMBERS.sha256 supplement.tar.zst > SHA256SUMS)
    jq -n --arg sha "$(sha "$packet/supplement.tar.zst")" --argjson members "$(wc -l < "$bundle/members.txt")" \
        --argjson bytes "$(du -bc "$packet/"* | awk 'END{print $1}')" --slurpfile link "$bundle/primary-link.json" \
        '$link[0] + {archive_sha256:$sha,archive_members:$members,upload_bytes:$bytes}' > "$bundle/pack.json"
    chmod a-w "$packet" "$packet/"* "$bundle/"{shared.tsv,primary-link.json,members.txt,pack.json}
    cat "$bundle/pack.json"
    ;;
restore)
    rv=${3:?new restore version}; restored=${4:?verified extracted primary archive}
    [[ $# == 4 || $# == 5 ]]; version_ok "$rv"
    if [[ $# == 5 ]]; then
        packet=$(realpath "$5")
        [[ $packet == "$bundle"/download-v*/download/results/codex/protected-transport-vast-* ]]
    fi
    verify "$packet"
    [[ $restored == "$storage"/archive-v*/local-v*/extracted || $restored == "$storage"/archive-v*/restore-v*/extracted ]]
    out="$bundle/restore-$rv"; root="$out/extracted"
    mkdir "$out" "$root"
    tar --zstd -xf "$packet/supplement.tar.zst" -C "$root"
    (cd "$root" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
    meta="$root/${bundle#"$s"/}/primary-link.json"
    archive=$(jq -er '.primary_archive|select(test("^archive-v[1-9][0-9]*$"))' "$meta")
    measurement=$(jq -er '.primary_measurement|select(test("^measurement-v[1-9][0-9]*$"))' "$meta")
    [[ $restored == "$storage/$archive/"* && $(sha "$storage/$archive/upload/measurement.tar.zst") == "$(jq -r .primary_archive_sha256 "$meta")" ]]
    verify "$restored/$measurement/train/root/device"
    verify "$restored/$measurement/train/root/grade"
    [[ $(sha "$restored/$measurement/train/root/device/SHA256SUMS") == "$(jq -r .primary_raw_manifest_sha256 "$meta")" ]]
    while IFS=$'\t' read -r hash member primary; do
        case "$member" in
            fit/vast-capture-v2/train/root/device/*) [[ $primary == "$measurement/train/root/device/"* ]] ;;
            fit/vast-grade-v1/*) [[ $primary == "$measurement/train/root/grade/"* ]] ;;
            *) exit 1 ;;
        esac
        [[ $(sha "$restored/$primary") == "$hash" && ! -e $root/$member ]]
        mkdir -p "$(dirname "$root/$member")"
        cp --reflink=auto "$restored/$primary" "$root/$member"
    done < "$root/${bundle#"$s"/}/shared.tsv"
    capture="$root/fit/vast-capture-v2"
    for dir in "$capture" "$capture/train" "$capture/train/root/device" "$capture/finite" "$capture/receipts/train" \
        "$capture/receipts/finite" "$root/fit/vast-grade-v1" "$root/fit/hardware-comparison-v1" "$root/fit/vast-controller-v1"; do
        verify "$dir"
    done
    staged="$root/${stage#"$s"/}"
    (cd "$staged" && sha256sum --strict --quiet -c payload.tar.sha256)
    tar -xzf "$staged/payload.tar.gz" -C "$staged"
    verify "$staged"
    jq -ne --slurpfile input "$meta" --arg manifest "$(sha "$capture/train/root/device/SHA256SUMS")" '
        if $manifest==$input[0].mirror_raw_manifest_sha256 then
        {pass:true,mirror_raw_manifest_sha256:$manifest,reconstructed_raw_files:$input[0].shared_raw_files,
         reconstructed_grade_files:$input[0].shared_grade_files,
         identical_raw_batches:$input[0].all_raw_files_identical} else error("restored raw manifest changed") end' \
        > "$out/receipt.json"
    cat "$out/receipt.json"
    ;;
*) exit 64 ;;
esac
