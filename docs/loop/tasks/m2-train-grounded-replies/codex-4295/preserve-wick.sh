#!/usr/bin/env bash
# Reuse the private study store; publish only a locally regraded packet.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common-wick.sh"
action=${1:?pack, local, publish or download}; pv=${2:?packet version}; version "$pv"
packet="$study/packet-$pv"; upload="$packet/upload"
safe_member(){
    [[ $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ && $1 != *..* && -e "$study/$1" && ! -L "$study/$1" ]]
}
restore(){
    local src=$1 out=$2 restored meta name
    verify "$src"; mkdir "$out/extracted"
    tar --zstd -xf "$src/measurement.tar.zst" -C "$out/extracted"
    restored="$out/extracted"
    (cd "$restored" && sha256sum --strict --quiet -c "$src/MEMBERS.sha256")
    meta="$restored/packet-$pv/layout.json"
    local m p g methods audit
    m=$(jq -er .measurement "$meta"); p=$(jq -er .panel "$meta"); g=$(jq -er .grade "$meta")
    methods=$(jq -er .methods "$meta"); audit=$(jq -er .audit "$meta")
    for name in "$m" "$p" "$g" "$methods" "$audit"; do verify "$restored/$name"; done
    bash "$restored/$methods/grade-wick.sh" "$restored/$m" "$restored/$p" "$out/regraded"
    local n=0
    while IFS= read -r name; do
        case "$name" in ./inputs.sha256) continue ;; esac
        cmp "$restored/$g/$name" "$out/regraded/$name"; n=$((n+1))
    done < <(cut -d' ' -f3- "$restored/$g/SHA256SUMS")
    jq -n --arg measurement "$(digest "$restored/$m/SHA256SUMS")" --arg audit "$(digest "$restored/$audit/SHA256SUMS")" \
        --arg archive "$(digest "$src/measurement.tar.zst")" --arg grade "$(digest "$restored/$g/summary.json")" \
        --argjson files "$(wc -l < "$src/MEMBERS.sha256")" --argjson n "$n" \
        '{verified:true,archive_sha256:$archive,verified_members:$files,measurement_sha256:$measurement,
          independent_audit_sha256:$audit,grade_summary_sha256:$grade,regraded_files:$n,disagreements:0}' > "$out/replay.json"
}
case $action in
pack)
    shift 2
    methods=${1:?methods}; release=${2:?release}; m=${3:?measurement}; p=${4:?panel}; g=${5:?grade}; audit=${6:?audit}; shift 6
    [[ $methods == methods-v* && $release == release-v* && $m == measurement-v* && $p == panel/panel-v* && $g == grade-v* && $audit == audit/* ]]
    for dir in "$methods" "$release" "$m" "$p" "$g" "$audit"; do safe_member "$dir"; done
    for dir in "$methods" "$m" "$p" "$g" "$audit" "$m-lease"; do verify "$study/$dir"; done
    [[ $(cat "$study/$m-lease/exit-code.txt") == 0 ]]
    [[ $(cat "$study/$audit/exit-code.txt") == 0 ]]
    jq -e '.pass==true' "$study/$audit/readback.json" >/dev/null
    sha256sum --quiet --strict -c "$study/$release/inputs.sha256" "$study/$audit/inputs.sha256"
    for f in "$study/$m/SHA256SUMS" "$study/$p/SHA256SUMS"; do
        grep -Fxq "$(sha256sum "$f")" "$study/$audit/inputs.sha256"
    done
    grep -Fxq "$(sha256sum "$study/$release/inputs.sha256")" "$study/$m/inputs.sha256"
    for file in "$study/$m/SHA256SUMS" "$study/$p/SHA256SUMS"; do grep -Fxq "$(sha256sum "$file")" "$study/$g/inputs.sha256"; done
    mkdir "$packet" "$upload"
    jq -n --arg method "$methods" --arg release "$release" --arg m "$m" --arg p "$p" --arg g "$g" --arg audit "$audit" \
        '{methods:$method,release:$release,measurement:$m,panel:$p,grade:$g,audit:$audit}' > "$packet/layout.json"
    cp "$old/far-residuals-v1/summary.json" "$packet/preceding-residuals.json"
    for entry in "$methods" "$release" "$m" "$m-lease" "$p" "$g" "$audit" "${here#"$study"/}" \
            "packet-$pv/layout.json" "packet-$pv/preceding-residuals.json" "$@"; do
        safe_member "$entry"
        if [[ $entry == */SHA256SUMS ]]; then
            verify "$study/${entry%/*}"; echo "$entry"
            while IFS= read -r leaf; do
                child="${entry%/*}/${leaf#./}"; safe_member "$child"; echo "$child"
            done < <(cut -d' ' -f3- "$study/$entry")
        elif [[ -f "$study/$entry" ]]; then echo "$entry"
        else (cd "$study" && find "$entry" -type f); fi
    done | LC_ALL=C sort -u > "$packet/members.txt"
    (cd "$study" && while IFS= read -r file; do sha256sum "$file"; done < "$packet/members.txt") > "$upload/MEMBERS.sha256"
    tar -C "$study" --sort=name --mtime=@0 --no-recursion --owner=0 --group=0 --numeric-owner -cf - -T "$packet/members.txt" |
        zstd -q -14 -T1 -o "$upload/measurement.tar.zst"
    (cd "$upload" && sha256sum MEMBERS.sha256 measurement.tar.zst > SHA256SUMS)
    jq -n --arg sha "$(digest "$upload/measurement.tar.zst")" --argjson members "$(wc -l < "$packet/members.txt")" \
        --argjson bytes "$(du -bc "$upload/"*|tail -1|cut -f1)" \
        '{archive_sha256:$sha,members:$members,uploaded_bytes:$bytes}' > "$packet/pack.json"
    ;;
local)
    v=${3:?new restore version}; version "$v"; out="$packet/local-$v"; mkdir "$out"; restore "$upload" "$out"
    ;;
publish|download)
    v=${3:?new restore version}; version "$v"
    hf="$HOME/.local/bin/hf"; repo=caseyallard/uor-r4-store
    if [[ $action == publish ]]; then
        lv=${4:?successful local replay}; version "$lv"; reference="$packet/local-$lv/replay.json"
        jq -e --arg sha "$(digest "$upload/measurement.tar.zst")" '.verified and .archive_sha256==$sha and .disagreements==0' "$reference" >/dev/null
        out="$packet/remote-$v"; mkdir "$out"
        "$hf" datasets info "$repo" --json | jq -e 'select(.private==true)|{private}' > "$out/privacy.json"
        hash=$(digest "$upload/measurement.tar.zst"); prefix="results/codex/intervening-source-20261126-${hash:0:12}"
        "$hf" upload "$repo" "$upload" "$prefix" --repo-type dataset --commit-message '[codex] Preserve the original noun occurrence omission test' --format json > "$out/commit.json"
        revision=$(jq -er --arg pre "https://huggingface.co/datasets/$repo/commit/" '.url|select(startswith($pre))|ltrimstr($pre)|select(test("^[0-9a-f]{40}$"))' "$out/commit.json")
    else
        receipt=$(realpath "${4:?immutable remote receipt}"); reference=$(dirname "$receipt")/replay.json
        prefix=$(jq -er .path "$receipt"); revision=$(jq -er .revision "$receipt")
        [[ $(jq -er .repo "$receipt") == "$repo" && $revision =~ ^[0-9a-f]{40}$ && $prefix == results/codex/intervening-source-* ]]
        out="$packet/download-$v"; mkdir "$out"
    fi
    "$hf" download "$repo" --repo-type dataset --revision "$revision" --include "$prefix/*" --local-dir "$out/download" --quiet > "$out/download.log"
    restored="$out/download/$prefix"
    for file in MEMBERS.sha256 measurement.tar.zst SHA256SUMS; do cmp "$upload/$file" "$restored/$file"; done
    restore "$restored" "$out"; cmp "$reference" "$out/replay.json"
    jq --arg repo "$repo" --arg path "$prefix" --arg rev "$revision" '.+{repo:$repo,path:$path,revision:$rev}' "$packet/pack.json" > "$out/receipt.json"
    ;;
*) exit 64 ;;
esac
