#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-third-margin-retain"
storage="$study/storage"
here=$(cd "$(dirname "$0")" && pwd -P)
mode=${1:?stage, train, seal or cold}
version=${2:?archive version}
[[ $version =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
[[ $(findmnt -n -o FSTYPE --target "$storage") == zfs && $here == "$storage"/methods-v* ]]
archive="$storage/archive-$version"
packet="$archive/packet"
cold="$storage/cold-$version"
tarball="$archive/third-margin-retain-$version.tar.zst"
verify() (cd "$1" && sha256sum --strict --quiet -c "${2:-SHA256SUMS}")
digest() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }

case "$mode" in
stage)
    [[ $# == 2 ]]
    for d in methods-v2 grade-ops panel/train panel/fresh shipment/train-input shipment/native/lex \
        shipment/native/scorer/source physical-e-finite-v1 physical-m-finite-v1 fit/portable-v1/sources; do
        verify "$study/$d"
    done
    verify "$study/panel" guards.sha256
    verify "$study/panel" training.sha256
    verify "$study/fit/portable-v1" receipt.sha256
    for name in rows.jq pairs.jq near.jq controls.jq metrics.jq elect.jq election.sh state.sh; do
        cmp "$study/methods-v2/grade/$name" "$study/grade-ops/grade/$name"
    done
    codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
    projection="$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11"
    pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
    pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$projection"

    mkdir "$archive" "$packet"
    mkdir "$packet/"{methods,inputs,panel,physical,expected,independent,custody}
    cp -a -- "$here" "$packet/methods/storage"
    cp -a -- "$study/methods-v2" "$packet/methods/frozen"
    cp -a -- "$study/grade-ops/grade" "$packet/methods/grade"
    cp -a -- "$study/grade-ops/SHA256SUMS" "$packet/methods/grade-ops.sha256"
    cp -a -- "$study/fit/portable-v1/sources" "$packet/methods/margin-fit"
    cp -a -- "$study/shipment/native/lex" "$packet/methods/eligibility"
    cp -a -- "$study/shipment/native/scorer/source" "$packet/methods/scorer"
    cp -a -- "$study/shipment/scripts" "$packet/methods/device-run"
    cp -a -- "$study/panel/methods-v7" "$packet/methods/panel"
    mkdir "$packet/methods/panel/generator-source"
    cp -a -- "$study/panel/generator-build-v1/source/docs" "$packet/methods/panel/generator-source/"
    cp -a -- "$study/panel/generator-build-v1/"{authority.sha256,patch.log,build.log,SHA256SUMS} "$packet/methods/panel/"
    cp -a -- "$study/shipment/train-input" "$packet/inputs/train-input"
    cp -a -- "$codec" "$projection" "$packet/inputs/"
    cp -a -- "$study/panel/"{train,fresh,regression,near,training.sha256,guards.sha256,history.json} "$packet/panel/"
    cp -a -- "$study/physical-e-finite-v1" "$packet/physical/fit-finite-e"
    cp -a -- "$study/physical-m-finite-v1" "$packet/physical/fit-finite-m"
    cmp "$packet/inputs/train-input/train/ordered/text.bin" "$packet/panel/train/ordered/text.bin"
    cmp "$packet/inputs/train-input/train/requests.tsv" "$packet/panel/train/requests.tsv"
    cmp "$packet/inputs/train-input/train/retain.tsv" "$packet/panel/train/retain.tsv"
    for name in regression near; do
        cmp "$packet/inputs/train-input/$name/ordered/text.bin" "$packet/panel/$name/ordered/text.bin"
    done
    pin e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977 "$packet/inputs/train-input/models/slot3.state"
    date -u +%FT%TZ > "$archive/staged-utc.txt"
    ;;
train)
    [[ $# == 2 && -f $archive/staged-utc.txt && ! -e $packet/physical/train && ! -e $packet/MEMBERS.sha256 ]]
    src="$study/physical-train-v1"
    verify "$src"
    mkdir "$packet/physical/train"
    cp -a -- "$src/"{fit-out,e-out,matrix,train-probes,score,prefit,release,scripts} "$packet/physical/train/"
    cp -a -- "$src/SHA256SUMS" "$packet/custody/physical-train.members.sha256"
    for part in regression near previous; do
        cmp "$packet/inputs/train-input/$part/reference.native.jsonl" "$packet/physical/train/prefit/$part/base.native.jsonl"
    done
    cmp "$packet/inputs/train-input/models/slot3.state" "$packet/physical/train/fit-out/start.state"
    date -u +%FT%TZ > "$archive/training-staged-utc.txt"
    ;;
seal)
    [[ $# == 3 && -f $archive/training-staged-utc.txt && ! -e $packet/MEMBERS.sha256 && ! -e $tarball ]]
    build=${3:?independent audit build directory}
    [[ $build =~ ^build-v[1-9][0-9]*$ ]]
    verify "$study/audit/$build"
    for d in election-v1 physical-held-v1 controls-v1 device-records-v1; do
        verify "$study/$d"
    done
    for d in e-finite-v2 panel-v2 predecessor-v1 protected-v1; do
        verify "$study/audit/$d"
        cp -a -- "$study/audit/$d" "$packet/independent/"
    done
    for d in m-finite-v3 train-fit-v1 held-fit-v1 physical-train-factors-v1 train-finalists-v1 train-all-v1 physical-held-v1; do
        verify "$study/audit/$d"
        cp -a -- "$study/audit/$d" "$packet/independent/"
    done
    cp -a -- "$study/audit/source-v5" "$packet/methods/independent"
    mkdir "$packet/independent/build"
    awk '{print $2}' "$study/audit/$build/SHA256SUMS" | while IFS= read -r file; do
        [[ $file =~ ^[a-zA-Z0-9_.-]+$ ]]
        cp -a -- "$study/audit/$build/$file" "$packet/independent/build/"
    done
    cp -a -- "$study/audit/$build/SHA256SUMS" "$packet/independent/build/"
    cp -a -- "$(realpath -e "$study/audit/$build/tokenizer")" "$packet/independent/build/tokenizer"
    cp -a -- "$study/election-v1" "$packet/expected/election"
    cp -a -- "$study/controls-v1" "$packet/expected/controls"
    cp -a -- "$study/grade-held-v1" "$packet/expected/held"
    verify "$study/panel/held-reduction-v1"
    cp -a -- "$study/panel/held-reduction-v1" "$packet/expected/held-reduction"
    cp -a -- "$study/panel/held-methods-v3" "$packet/methods/held-reduction"
    cp -a -- "$study/physical-held-v1/held" "$packet/physical/held"
    cp -a -- "$study/physical-held-v1/SHA256SUMS" "$packet/custody/physical-held.members.sha256"
    cp -a -- "$study/device-records-v1" "$packet/custody/device-records"

    projected="$study/physical-backoff-v1"
    if [[ -d $projected ]]; then
        verify "$projected"
        mkdir "$packet/physical/backoff"
        cp -a -- "$projected/projected" "$packet/physical/backoff/"
        cp -a -- "$projected/SHA256SUMS" "$packet/custody/physical-backoff.members.sha256"
        verify "$study/shipment/backoff-input"
        cp -a -- "$study/shipment/backoff-input" "$packet/inputs/"
    fi
    if [[ -d $study/shipment/fresh-input ]]; then
        verify "$study/shipment/fresh-input"
        cp -a -- "$study/shipment/fresh-input" "$packet/inputs/"
        cmp "$packet/inputs/fresh-input/arms.tsv" "$packet/expected/election/arms.tsv"
    fi
    test -z "$(find "$packet" ! -type f ! -type d -print -quit)"
    (cd "$packet" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$archive/MEMBERS.pending")
    mv -- "$archive/MEMBERS.pending" "$packet/MEMBERS.sha256"
    verify "$packet" MEMBERS.sha256
    date -u +%FT%TZ > "$archive/sealed-utc.txt"
    /usr/bin/time -f '%e %U %S %M' -o "$archive/pack.time" \
        tar --sort=name --format=posix -C "$archive" -I 'zstd -T2 -5' -cf "$tarball" packet
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
        bash "$restored/methods/storage/replay.sh" "$restored" "$cold/verification" > "$cold/replay.log"
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
