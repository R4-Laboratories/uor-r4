#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
prior="$runs/20261126-codex-fixed-assignment-median"
device="$study/device"
reader="$runs/20261126-codex-text-governor-links/learner/build-v2"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$device"/pod-* ]]
before="$prior/methods-v2/inputs/protected.state"
after="$prior/train-v1/states/ordered.state"
bank="$prior/methods-v2/inputs/primary.words"
archive=("$prior/held-v1/contrast/"{ordered_base,ordered_full}.raw.jsonl "$prior/train-v1/ordered/"{base,full}.raw.jsonl)

sha() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { [[ $1 =~ ^[0-9a-f]{64}$ && $(sha "$2") == "$1" ]]; }
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
listed() { grep -Fxq "$(sha256sum -- "$2")" "$1"; }

mode=${1:?package, authorize, import-finite or import-capture}
v=${2:?new version}
[[ $v =~ ^v[1-9][0-9]*$ ]]
case "$mode" in
package)
    [[ $# == 5 ]]
    panel=$(realpath -e "$3"); build=$(realpath -e "$4"); fixture=$(realpath -e "$5")
    [[ $panel == "$study"/panel/panel-v* && $build == "$device"/build-v* && $fixture == "$device"/host-finite-v* ]]
    for directory in "$panel" "$build" "$fixture" "$reader" "$prior/held-v1" "$prior/train-v1"; do verify "$directory"; done
    pin 2bd85f7280e6ffec0026f686e64f4376ee6f74aa6d0924816ce56a6291430da0 "$reader/textlinks-sm89"
    pin 7cd831ddec22535ecc94783cef2dae34db1d71a0428e4b40bfe8d5a43b4437d1 "$build/switch-competition-sm89"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$before"
    pin 0edd2cb3ee9251b9fa40cd829b52e3aff2fc477bf05d515c787cea9435af01ea "$after"
    pin ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$bank"
    folder="$device/pod-payloads-$v"; mkdir "$folder"
    for phase in finite research; do
        p="$folder/$phase"
        mkdir "$p" "$p/bin"
        printf '%s\n' "$phase" > "$p/phase"
        cp -- "$here/pod-run.sh" "$p/"
        cp -- "$build/switch-competition-sm89" "$p/bin/"
        cp -- "$build/SHA256SUMS" "$p/wick-build.SHA256SUMS"
    done
    f="$folder/finite"; mkdir "$f/fixture" "$f/fixture/rejected"
    cp -- "$reader/textlinks-sm89" "$f/bin/"
    cp -- "$fixture/"{fixture.words,text.bin,fixture.state,requests.tsv,fixture-text.txt} "$f/fixture/"
    cp -- "$fixture/"{sign,row,word,site,duplicate-site,spaces,trailing,empty}.tsv "$f/fixture/rejected/"
    cp -- "$build/source/finite.jq" "$f/fixture/"
    cp -- "$fixture/SHA256SUMS" "$f/wick-fixture.SHA256SUMS"
    seal "$f"
    tar -C "$folder" -cf - finite | gzip -3 > "$folder/finite.tar.gz"
    sha256sum "$f/SHA256SUMS" "$folder/finite.tar.gz"

    p="$folder/research"; mkdir "$p/cohorts" "$p/states"
    cp -- "$before" "$p/states/before"; cp -- "$after" "$p/states/after"; cp -- "$bank" "$p/states/bank"
    cp -- "$panel/SHA256SUMS" "$p/wick-panel.SHA256SUMS"
    for cohort in historical train-original train-derived; do
        source_dir="$panel/$cohort"; destination="$p/cohorts/$cohort"
        mkdir "$destination"
        cp -- "$source_dir/ordered/text.bin" "$source_dir/requests.tsv" "$destination/"
        wc -l < "$source_dir/rows.jsonl" | tr -d ' ' > "$destination/rows.txt"
    done
    cp -- "${archive[0]}" "$p/cohorts/historical/before.archive.jsonl"
    cp -- "${archive[1]}" "$p/cohorts/historical/after.archive.jsonl"
    cp -- "${archive[2]}" "$p/cohorts/train-original/before.archive.jsonl"
    cp -- "${archive[3]}" "$p/cohorts/train-original/after.archive.jsonl"
    seal "$p"
    tar -C "$folder" -cf - research | gzip -3 > "$folder/research.tar.gz"
    sha256sum "$p/SHA256SUMS" "$folder/research.tar.gz"
    sha256sum "$panel/SHA256SUMS" "$build/SHA256SUMS" "$reader/SHA256SUMS" "$fixture/SHA256SUMS" \
        "$prior/held-v1/SHA256SUMS" "$prior/train-v1/SHA256SUMS" "$here/"{pod-run.sh,pod-stage.sh} > "$folder/inputs.sha256"
    seal "$folder"
    ;;
authorize)
    [[ $# == 5 ]]
    package=$(realpath -e "$3"); release=$(realpath -e "$4")
    [[ $package == "$device"/pod-payloads-v* && $release == "$study"/release-v* ]]
    verify "$package"; verify "$release"; pin "$5" "$release/SHA256SUMS"
    sha256sum --strict --quiet -c "$release/inputs.sha256"
    grep -Fq 'TEST FITNESS: FIT' "$release/FIT.md"
    panel=$(jq -er .panel "$release/paths.json"); binary=$(jq -er .binary "$release/paths.json")
    [[ $panel == "$study"/panel/panel-v* && $binary == "$device"/build-v*/switch-competition-sm89 ]]
    payload="$package/research"
    verify "$payload"
    cmp "$binary" "$payload/bin/switch-competition-sm89"
    cmp "$before" "$payload/states/before"
    cmp "$after" "$payload/states/after"
    cmp "$bank" "$payload/states/bank"
    cmp "$release/ordered.unpacked" "$payload/states/after"
    cmp "$panel/SHA256SUMS" "$payload/wick-panel.SHA256SUMS"
    for file in "$panel/SHA256SUMS" "$binary" "$before" "$after" "$bank"; do listed "$release/inputs.sha256" "$file"; done
    methods=
    while IFS= read -r candidate; do
        if listed "$release/inputs.sha256" "$candidate" && [[ $(sha "$candidate") == 91f298ab1af6ca606e8f11305ff97c9799b3228c0caac8320fc6560cbf1c7299 ]]; then
            methods="$candidate"
            break
        fi
    done < <(find "$study" -maxdepth 2 -path "$study/methods-v*/SHA256SUMS" -type f | sort)
    [[ -n $methods ]]
    dir="$device/pod-authorization-$v"; mkdir "$dir" "$dir/auth"
    auth="$dir/auth"
    cp -- "$release/"{FIT.md,ordered.unpacked} "$auth/"
    cp -- "$release/SHA256SUMS" "$auth/release.SHA256SUMS"
    sha "$payload/SHA256SUMS" > "$auth/payload-sha256.txt"
    sha256sum "$release/SHA256SUMS" "$methods" "$panel/SHA256SUMS" "$binary" "$bank" "$before" "$after" \
        "${archive[@]}" > "$auth/inputs.sha256"
    seal "$auth"
    tar -C "$dir" -czf "$dir/authorization.tar.gz" auth
    seal "$dir"
    sha256sum "$auth/SHA256SUMS" "$dir/authorization.tar.gz" "$release/SHA256SUMS"
    ;;
import-finite|import-capture)
    [[ $# == 5 ]]
    tarball=$(realpath -e "$3")
    output_sha=$4; payload_sha=$5
    output_type=${mode#import-}
    name="$output_type-$v"
    dest="$device/$name"
    if [[ $output_type == capture ]]; then dest="$study/$name"; fi
    [[ ! -e $dest ]]
    tar -tzf "$tarball" | awk -v name="$name" '
        $0 !~ ("^" name "/") || $0 ~ /(^|\/)\.\.(\/|$)/ {bad=1} END {exit bad}'
    tar -tvzf "$tarball" | awk '$0 !~ /^[-d]/ {bad=1} END {exit bad}'
    imported="$device/pod-import-$name"; mkdir "$imported"
    tar -xzf "$tarball" -C "$imported" --no-same-owner --no-same-permissions
    source_dir="$imported/$name"
    pin "$output_sha" "$source_dir/SHA256SUMS"; verify "$source_dir"
    [[ $(cat "$source_dir/payload-sha256.txt") == "$payload_sha" &&
       $(cat "$source_dir/phase.txt") == "$output_type" &&
       $(cat "$source_dir/binary-sha256.txt") == 7cd831ddec22535ecc94783cef2dae34db1d71a0428e4b40bfe8d5a43b4437d1 ]]
    if [[ $output_type == finite ]]; then
        jq -en --rawfile prompt "$source_dir/fixture-text.txt" --slurpfile native "$source_dir/original.jsonl" \
            --slurpfile rows "$source_dir/first/requests.jsonl" -f "$device/build-v1/source/finite.jq" > "$imported/checks.json"
        cmp "$source_dir/original.jsonl" "$source_dir/first/native.jsonl"
        diff -r "$source_dir/first" "$source_dir/repeat"
    else
        sha256sum --strict --quiet -c "$source_dir/inputs.sha256"
        cmp "$source_dir/historical/before/native.jsonl" "${archive[0]}"
        cmp "$source_dir/historical/after/native.jsonl" "${archive[1]}"
        cmp "$source_dir/train-original/before/native.jsonl" "${archive[2]}"
        cmp "$source_dir/train-original/after/native.jsonl" "${archive[3]}"
    fi
    mv -T -- "$source_dir" "$dest"
    sha256sum "$tarball" "$dest/SHA256SUMS" > "$imported/inputs.sha256"
    seal "$imported"
    sha256sum "$dest/SHA256SUMS" "$imported/SHA256SUMS"
    ;;
*) exit 64 ;;
esac
