#!/usr/bin/env bash
# Preserve sealed raw evidence and verify the compact private archive before optional publication.
set -euo pipefail
umask 077
export LC_ALL=C

s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport
storage="$s/storage"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$storage"/methods-v* ]]
sha(){ sha256sum "$1" | cut -d' ' -f1; }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
check_version(){ [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
match_pin(){ grep -Fxq "$(sha256sum "$1")" "$2"; }

mode=${1:?pack, restore-local, or publish}
version=${2:?new archive version}
check_version "$version"
bundle="$storage/archive-$version"
packet="$bundle/upload"

append_file(){
    local file=$1
    [[ $file == "$s/"* && -f $file && ! -L $file ]]
    printf '%s\n' "${file#"$s"/}" >> "$bundle/members.pending"
}
append_seal(){
    local dir=$1 relative
    verify "$dir"
    append_file "$dir/SHA256SUMS"
    while IFS= read -r relative; do
        relative=${relative#./}
        case "$dir/$relative" in
            "$panel/held/blind/text.bin"|"$panel/held/structural-links/geometry.bin") continue ;;
        esac
        append_file "$dir/$relative"
    done < <(cut -d' ' -f3- "$dir/SHA256SUMS")
}
restore(){
    local source=$1 out=$2
    verify "$source"
    mkdir "$out/extracted"
    tar --zstd -xf "$source/measurement.tar.zst" -C "$out/extracted"
    (cd "$out/extracted" && sha256sum --strict --quiet -c "$source/MEMBERS.sha256")
    bash "$out/extracted/${here#"$s"/}/restore-grade-wick.sh" "$out/extracted" "$out/replayed"
    sha256sum "$source/SHA256SUMS" "$source/measurement.tar.zst" "$out/replayed/receipt.json" > "$out/verification.sha256"
}

case "$mode" in
pack)
    mv=${3:?measurement version}; tv=${4:?transfer version or -}; gv=${5:?grade version or -}; cv=${6:--}
    [[ $# -le 6 ]]; check_version "$mv"
    m="$s/measurement-$mv"; verify "$m"
    premise=$(jq -er '.pass|select(type=="boolean")|tostring' "$m/premise.json")
    manifest=$(cut -d' ' -f3- "$m/inputs.sha256"); release=${manifest%/inputs.sha256}
    [[ $manifest == "$s"/release-v*/inputs.sha256 ]]
    sha256sum --strict --quiet -c "$m/inputs.sha256" "$manifest"
    methods_manifest=$(awk -v prefix="$s/methods-" 'index($2,prefix)==1 && $2~/\/SHA256SUMS$/ {print $2}' "$manifest")
    methods=${methods_manifest%/SHA256SUMS}
    [[ $methods == "$s"/methods-v* && $methods != *$'\n'* ]]
    panel=$(cat "$release/panel-path.txt"); [[ $panel == "$s"/panel/panel-v* ]]
    verify "$panel"; sha256sum --strict --quiet -c "$panel/authority.sha256"
    match_pin "$panel/SHA256SUMS" "$manifest"
    build=$(dirname "$(cat "$release/gpu-path.txt")")
    [[ $build == "$s"/fit/build-v* ]]
    audit_manifest=$(awk -v prefix="$s/audit/build-" 'index($2,prefix)==1 && $2~/\/SHA256SUMS$/ {print $2}' "$manifest")
    audit_build=${audit_manifest%/SHA256SUMS}
    [[ $audit_build == "$s"/audit/build-v* && $audit_build != *$'\n'* ]]
    t=-; g=-; client=-; tokens=-; permit=-
    if [[ $premise == true ]]; then
        check_version "$tv"; check_version "$gv"
        t="$s/transfer-$tv"; g="$s/grade-$gv"
        verify "$t"; verify "$g"
        sha256sum --strict --quiet -c "$m/selection.sha256" "$t/inputs.sha256" "$g/inputs.sha256"
        match_pin "$m/SHA256SUMS" "$t/inputs.sha256"
        match_pin "$panel/SHA256SUMS" "$t/inputs.sha256"
        match_pin "$t/SHA256SUMS" "$g/inputs.sha256"
        match_pin "$m/SHA256SUMS" "$g/inputs.sha256"
        match_pin "$methods_manifest" "$g/inputs.sha256"
        permit=$(awk -v prefix="$s/train-verified-v" 'index($2,prefix)==1 && $2~/\/inputs.sha256$/ {print $2}' "$t/inputs.sha256")
        [[ $permit == "$s"/train-verified-v*/inputs.sha256 && $permit != *$'\n'* ]]
        sha256sum --strict --quiet -c "$permit"
        match_pin "$m/SHA256SUMS" "$permit"
        jq -e '.pass and .compared_unique_batches==(.batches|length)' "${permit%/*}/verified.json" >/dev/null
        match_pin "${permit%/*}/verified.json" "$permit"
        pass=$(jq -er '.pass|select(type=="boolean")|tostring' "$g/summary.json")
        if [[ $pass == true ]]; then
            check_version "$cv"
            client="$s/client/capture-$cv"; tokens="$s/client/token-$cv"
            verify "$client"; verify "$tokens"
            sha256sum --strict --quiet -c "$tokens/inputs.sha256"
            match_pin "$g/summary.json" "$tokens/inputs.sha256"
            match_pin "$client/SHA256SUMS" "$tokens/inputs.sha256"
        else [[ $cv == - ]]; fi
    else
        [[ $tv == - && $gv == - && $cv == - && ! -e $m/selection.json && ! -e $m/states ]]
    fi

    mkdir "$bundle" "$packet"
    append_file "$here/preserve-wick.sh"; append_file "$here/restore-grade-wick.sh"
    append_seal "$m"; append_seal "$methods"; append_seal "$panel"; append_seal "$build"; append_seal "$audit_build"
    append_seal "$m-lease"
    append_file "$manifest"
    # Keep released local receipts, and keep their manifests' members when present.
    while IFS= read -r origin; do
        case "$origin" in
            "$panel/held/blind/text.bin") ;;
            "$s"/methods-v*/*|"$panel"/*|"$build"/*|"$audit_build"/*) ;;
            "$s"/*/SHA256SUMS) append_seal "${origin%/*}" ;;
            "$s"/*/readback.sha256)
                (cd "${origin%/*}" && sha256sum --strict --quiet -c "$origin")
                append_file "$origin"
                while IFS= read -r child; do append_file "${origin%/*}/${child#./}"; done < <(cut -d' ' -f3- "$origin")
                ;;
            "$s"/*) append_file "$origin" ;;
        esac
    done < <(cut -d' ' -f3- "$manifest")
    for origin in "$panel/authority.sha256" "$methods/previous.sha256"; do
        while IFS= read -r input; do
            case "$input" in "$s"/panel/private/*|"$s"/panel/methods-v*/*) append_file "$input" ;; esac
        done < <(cut -d' ' -f3- "$origin")
    done
    while IFS= read -r -d '' input; do append_file "$input"; done < <(
        find "$s/panel/private" -maxdepth 1 -type f -print0)
    antecedents="$s/panel/antecedents-v1"
    sha256sum --strict --quiet -c "$antecedents/inputs.sha256"
    match_pin "$panel/SHA256SUMS" "$antecedents/inputs.sha256"
    append_seal "$antecedents"
    append_file "$s/panel/methods-v5/verify-antecedents-wick.sh"
    if [[ $premise == true ]]; then
        append_seal "$t"; append_seal "$t-lease"; append_seal "$g"
        append_file "$permit"
        permit_dir=${permit%/*}
        # Include the authorization, every batch's independent replay and their checked input lists.
        while IFS= read -r -d '' input; do append_file "$input"; done < <(find "$permit_dir" -type f -print0)
        while IFS= read -r input; do
            case "$input" in
                "$s"/audit/*/SHA256SUMS) append_seal "${input%/*}" ;;
                "$s"/audit/train-v*/inputs.sha256)
                    append_seal "${input%/*}"
                    while IFS= read -r audit_input; do
                        case "$audit_input" in
                            "$s"/fitness/*/*.sh|"$s"/fitness/*/*.jq) append_file "$audit_input" ;;
                        esac
                    done < <(cut -d' ' -f3- "$input")
                    ;;
            esac
        done < <(cut -d' ' -f3- "$permit")
        if [[ $client != - ]]; then append_seal "$client"; append_seal "$tokens"; fi
        for report in "$s"/report-v*; do
            [[ -f "$report/inputs.sha256" ]] || continue
            match_pin "$m/SHA256SUMS" "$report/inputs.sha256" || continue
            match_pin "$t/SHA256SUMS" "$report/inputs.sha256" || continue
            match_pin "$g/SHA256SUMS" "$report/inputs.sha256" || continue
            sha256sum --strict --quiet -c "$report/inputs.sha256"
            append_seal "$report"
        done
    fi
    jq -n --arg prefix "$s/" --arg m "$m" --arg release "$release" --arg methods "$methods" \
        --arg panel "$panel" --arg build "$audit_build" --arg transfer "$t" --arg grade "$g" \
        --arg permit "$permit" --arg client "$client" --arg tokens "$tokens" --argjson premise "$premise" '
        def entry: if .=="-" then null else ltrimstr($prefix) end;
        {schema:1,premise:$premise,measurement:($m|entry),release:($release|entry),methods:($methods|entry),
        panel:($panel|entry),audit_build:($build|entry),transfer:($transfer|entry),grade:($grade|entry),
        train_permit:($permit|entry),client:($client|entry),tokens:($tokens|entry)}' > "$bundle/archive.json"
    append_file "$bundle/archive.json"
    sort -u "$bundle/members.pending" > "$bundle/members.txt"
    (
        cd "$s"
        xargs -d '\n' sha256sum < "$bundle/members.txt" > "$packet/MEMBERS.sha256"
        tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
            -cf - -T "$bundle/members.txt" | zstd -q -15 -T1 -o "$packet/measurement.tar.zst"
    )
    (cd "$packet" && sha256sum MEMBERS.sha256 measurement.tar.zst > SHA256SUMS)
    jq -n --arg hash "$(sha "$packet/measurement.tar.zst")" --argjson count "$(wc -l < "$bundle/members.txt")" \
        --argjson bytes "$(du -bc "$packet/"* | awk 'END{print $1}')" '
        {archive_sha256:$hash,members:$count,upload_bytes:$bytes}' > "$bundle/pack.json"
    chmod a-w "$packet" "$packet/"* "$bundle/"{archive.json,members.txt,pack.json}
    cat "$bundle/pack.json"
    ;;
restore-local)
    rv=${3:?new restore version}; [[ $# == 3 ]]; check_version "$rv"
    out="$bundle/local-$rv"; mkdir "$out"; restore "$packet" "$out"
    ;;
publish)
    rv=${3:?new remote restore version}; lv=${4:?passed local restore version}
    [[ $# == 4 ]]; check_version "$rv"; check_version "$lv"
    verify "$packet"
    local_restore="$bundle/local-$lv"
    sha256sum --strict --quiet -c "$local_restore/verification.sha256"
    match_pin "$packet/measurement.tar.zst" "$local_restore/verification.sha256"
    jq -e '.pass and .substantive_disagreements==0' "$local_restore/replayed/receipt.json" >/dev/null
    out="$bundle/restore-$rv"; mkdir "$out"
    repo=caseyallard/uor-r4-store
    archive=$(sha "$packet/measurement.tar.zst")
    prefix="results/codex/protected-transport-20261126-${archive:0:12}"
    hf="$HOME/.local/bin/hf"
    "$hf" datasets info "$repo" --json | jq -e 'select(.private==true)|{private}' > "$out/privacy.json"
    "$hf" upload "$repo" "$packet" "$prefix" --repo-type dataset \
        --commit-message '[codex] Preserve the protected integer transport measurements' --format json > "$out/upload.json"
    revision=$(jq -er --arg prefix "https://huggingface.co/datasets/$repo/commit/" '
        .url|select(startswith($prefix))|ltrimstr($prefix)|select(test("^[0-9a-f]{40}$"))' "$out/upload.json")
    "$hf" download "$repo" --repo-type dataset --revision "$revision" --include "$prefix/*" \
        --local-dir "$out/download" --quiet > "$out/download.log"
    retrieved="$out/download/$prefix"
    for member in SHA256SUMS MEMBERS.sha256 measurement.tar.zst; do cmp "$packet/$member" "$retrieved/$member"; done
    restore "$retrieved" "$out"
    jq -n --arg repo "$repo" --arg path "$prefix" --arg revision "$revision" \
        --slurpfile pack "$bundle/pack.json" '$pack[0] + {repo:$repo,path:$path,revision:$revision}' > "$out/receipt.json"
    cat "$out/receipt.json"
    ;;
*) exit 64 ;;
esac
