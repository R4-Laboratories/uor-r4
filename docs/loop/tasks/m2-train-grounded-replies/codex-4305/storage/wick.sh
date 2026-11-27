#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-governor-argument-sites"
prior="$runs/20261126-codex-third-margin-retain/storage/cold-v1/packet"
storage="$study/storage"
here=$(cd "$(dirname "$0")" && pwd -P)
mode=${1:?stage, seal or cold}
version=${2:?archive version}
[[ $version =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
[[ $(findmnt -n -o FSTYPE --target "$storage") == zfs && $here == "$storage"/methods-v* ]]
archive="$storage/archive-$version"
packet="$archive/packet"
cold="$storage/cold-$version"
tarball="$archive/governor-argument-sites-$version.tar.zst"
digest() { sha256sum -- "$1" | cut -d' ' -f1; }
verify() (cd "$1" && sha256sum --strict --quiet -c "${2:-SHA256SUMS}")
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }

case "$mode" in
stage)
    [[ $# == 3 && $3 =~ ^build-v[1-9][0-9]*$ ]]
    audit="$study/audit/$3"
    verify "$audit"
    for source in seed-v1 device/source device/fixture physical-finite-v1 physical-finite-v1/build-v1 physical-finite-v1/finite-v1 audit/archived-v1; do
        verify "$study/$source"
    done
    pin 048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d "$prior/MEMBERS.sha256"
    (
        cd "$prior"
        awk '$2 ~ /^\.\/panel\/(near|regression)\// ||
             $2 ~ /^\.\/physical\/train\/score\/(near|regression)\/b\.(native|costs)\.jsonl$/ ||
             $2 ~ /^\.\/independent\/build\/tokenizer\// {print}' MEMBERS.sha256 |
            sha256sum --strict --quiet -c -
    )
    mkdir "$archive" "$packet"
    mkdir "$packet/"{methods,inputs,panel,physical,expected,custody}
    cp -a -- "$here" "$packet/methods/storage"
    cp -a -- "$study/seed-v1" "$packet/inputs/seed"
    cp -a -- "$study/device/source" "$packet/methods/device"
    cp -a -- "$study/device/fixture" "$packet/inputs/finite"
    cp -a -- "$study/physical-finite-v1/build-v1" "$packet/physical/build"
    cp -a -- "$study/physical-finite-v1/finite-v1" "$packet/physical/finite"
    cp -a -- "$study/physical-finite-v1/SHA256SUMS" "$packet/custody/physical-finite-origin.sha256"
    cp -a -- "$study/audit/$3/SHA256SUMS" "$packet/custody/audit-build-origin.sha256"
    cp -a -- "$study/audit/archived-v1" "$packet/expected/archived"
    mkdir "$packet/inputs/archived" "$packet/inputs/archived/captures"
    for split in near regression; do
        cp -a -- "$prior/panel/$split" "$packet/inputs/archived/"
        cp -a -- "$prior/physical/train/score/$split/b.native.jsonl" "$packet/inputs/archived/captures/$split.original.jsonl"
        cmp "$packet/inputs/archived/captures/$split.original.jsonl" "$packet/inputs/seed/previous/$split/original.jsonl"
    done
    cp -a -- "$prior/physical/train/score/regression/b.costs.jsonl" "$packet/inputs/archived/captures/regression.costs.jsonl"
    mkdir "$packet/methods/audit"
    while IFS= read -r file; do
        [[ $file =~ ^[a-zA-Z0-9_.-]+$ ]]
        cp -a -- "$audit/$file" "$packet/methods/audit/"
    done < <(awk '{print $2}' "$audit/SHA256SUMS")
    cp -a -- "$audit/SHA256SUMS" "$packet/methods/audit/"
    cp -a -- "$(realpath -e "$audit/tokenizer")" "$packet/methods/audit/tokenizer"
    cp -a -- "$study/audit/"{archived-wick,build-wick,finite-wick,panel-wick,physical-wick}.sh "$packet/methods/audit/"
    verify "$packet/methods/audit"
    (
        cd "$packet/methods/audit"
        mkdir .cargo
        cargo vendor --locked --offline vendor > .cargo/config.toml 2> "$archive/vendor.log"
        grep -Fx 'directory = "vendor"' .cargo/config.toml
    )
    pin d4a722c9d688f816515fdd60df6e362c05fae992f7ece3646c08379720500b32 "$study/panel/PANELS.sha256"
    bash "$here/panel-stage.sh" "$study/panel" "$packet"
    date -u +%FT%TZ > "$archive/staged-utc.txt"
    ;;
seal)
    [[ $# == 5 && $3 =~ ^grade-v[1-9][0-9]*$ && $4 =~ ^grade-held-v[1-9][0-9]*$ &&
        $5 =~ ^[0-9a-f]{64}$ && -f $archive/staged-utc.txt && ! -e $packet/MEMBERS.sha256 && ! -e $tarball ]]
    for source in "$3" "$4" analysis-results-v1 readout-v1 supply-ops-v1 audit/finite-v1 audit/panel-v1 \
        audit/physical-v1 audit/identities-v1 device-records-v1 pre-v1 release-scripts-v1; do
        verify "$study/$source"
    done
    verify "$study/physical-v1" raw.members
    pin c6a8b5cd7ca5b2f204773a5e044bb836082a349ffbf5d9a1821bcb667a6b6b65 "$study/physical-v1/raw.members"
    pin 67f0c31d31c619f6174e97cc3c27a1440ffd5ed4de01380ce809d43aa47eafb2 "$study/audit/physical-v1/SHA256SUMS"
    pin "$5" "$study/$3/SHA256SUMS"
    pin 611a11a8d1aac772fe65314bf527844c1b57b723d3fc46573b8ee821d78f406c "$study/$4/SHA256SUMS"
    pin 1823c3363951b7bfabcbfb9fe0464f9770d2fab2a9bc31514b7341a4f16c82da "$study/readout-v1/SHA256SUMS"
    pin 55be252fb30c5e641f65a96aa00142eba6f337193d8b5b7efb45ef0541ad3db0 "$study/audit/identities-v1/SHA256SUMS"
    pin 617645c0c446ade640140a27da04800106bc2153a0c9b8ce9892ada6667beb74 "$study/device-records-v1/SHA256SUMS"
    cp -a -- "$here/"{README.md,wick.sh,replay.sh,panel-stage.sh,panel-replay.sh} "$packet/methods/storage/"
    cp -a -- "$study/$3" "$packet/methods/grade"
    cp -a -- "$study/analysis-results-v1" "$packet/methods/readout"
    cp -a -- "$study/supply-ops-v1" "$packet/methods/supply"
    cp -a -- "$study/$4" "$packet/expected/grade"
    cp -a -- "$study/readout-v1" "$packet/expected/readout"
    cp -a -- "$study/audit/finite-v1" "$packet/expected/independent-finite"
    cp -a -- "$study/audit/panel-v1" "$packet/expected/independent-panel"
    cp -a -- "$study/audit/physical-v1" "$packet/expected/independent-physical"
    cp -a -- "$study/audit/identities-v1" "$packet/expected/independent-identities"
    cp -a -- "$study/audit/"{identities-wick.sh,identities.jq} "$packet/methods/audit/"
    cp -a -- "$study/physical-v1/raw" "$packet/physical/raw"
    cp -a -- "$study/physical-v1/raw.members" "$packet/physical/raw.members"
    cp -a -- "$study/device-records-v1" "$packet/custody/device-records"
    cp -a -- "$study/audit/FIT.md" "$packet/custody/audit-FIT.md"
    cp -a -- "$study/pre-v1" "$packet/custody/pre"
    cp -a -- "$study/release-scripts-v1" "$packet/methods/release"
    if [[ -f $study/physical-v1/SHA256SUMS ]]; then
        cp -a -- "$study/physical-v1/SHA256SUMS" "$packet/custody/physical-origin.sha256"
    fi
    test -z "$(find "$packet" ! -type f ! -type d -print -quit)"
    (cd "$packet" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$archive/MEMBERS.pending")
    mv -- "$archive/MEMBERS.pending" "$packet/MEMBERS.sha256"
    verify "$packet" MEMBERS.sha256
    date -u +%FT%TZ > "$archive/sealed-utc.txt"
    /usr/bin/time -f '%e %U %S %M' -o "$archive/pack.time" \
        tar -C "$archive" --sort=name --format=posix --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --pax-option=delete=atime,delete=ctime -I 'zstd -T2 -5' -cf "$tarball" packet
    jq -n --arg created "$(cat "$archive/sealed-utc.txt")" \
        --arg archive "${tarball##*/}" --arg archive_sha "$(digest "$tarball")" \
        --arg members_sha "$(digest "$packet/MEMBERS.sha256")" \
        --argjson archive_bytes "$(stat -c %s "$tarball")" \
        --argjson logical_bytes "$(find "$packet" -type f -printf '%s\n' | awk '{n+=$1} END {printf "%.0f\n", n}')" \
        --argjson members "$(wc -l < "$packet/MEMBERS.sha256")" \
        --arg pack_time "$(cat "$archive/pack.time")" \
        '{created_utc:$created,archive:$archive,archive_sha256:$archive_sha,archive_bytes:$archive_bytes,
          logical_bytes:$logical_bytes,members_sha256:$members_sha,members:$members,pack_time:$pack_time}' > "$archive/receipt.json"
    (cd "$archive" && sha256sum "${tarball##*/}" receipt.json pack.time > SHA256SUMS && sha256sum --strict --quiet -c SHA256SUMS)
    cat "$archive/receipt.json"
    ;;
cold)
    [[ $# == 2 ]]
    verify "$archive"
    mkdir "$cold"
    /usr/bin/time -f '%e %U %S %M' -o "$cold/extract.time" \
        tar -I zstd --no-same-owner --no-same-permissions -xf "$tarball" -C "$cold"
    restored="$cold/packet"
    pin "$(jq -er .members_sha256 "$archive/receipt.json")" "$restored/MEMBERS.sha256"
    /usr/bin/time -f '%e %U %S %M' -o "$cold/replay.time" \
        bash "$restored/methods/storage/replay.sh" "$restored" "$cold/verification" > "$cold/replay.log" 2>&1
    jq -n --arg receipt "$(digest "$archive/receipt.json")" --arg utc "$(date -u +%FT%TZ)" \
        --arg outputs "$(digest "$cold/verification/SHA256SUMS")" --arg extract_time "$(cat "$cold/extract.time")" \
        --arg replay_time "$(cat "$cold/replay.time")" \
        '{pass:true,verified_utc:$utc,archive_receipt_sha256:$receipt,outputs_sha256:$outputs,
          extract_time:$extract_time,replay_time:$replay_time}' > "$cold/receipt.json"
    (cd "$cold" && sha256sum receipt.json extract.time replay.time replay.log verification/SHA256SUMS > SHA256SUMS)
    cat "$cold/receipt.json"
    sha256sum "$cold/SHA256SUMS"
    ;;
*) exit 64 ;;
esac
