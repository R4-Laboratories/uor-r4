#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-paired-occurrence-margin"
storage="$study/storage"
here=$(cd "$(dirname "$0")" && pwd -P)
mode=${1:?stage, seal or cold}
version=${2:?new archive version}
[[ $# == 2 && $version =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
[[ $(findmnt -n -o FSTYPE --target "$storage") == zfs && $here == "$storage"/pack-v* ]]
archive="$storage/archive-$version"
packet="$archive/packet"
cold="$storage/cold-$version"
tarball="$archive/paired-occurrence-$version.tar.zst"
train="$study/rented-train-v1"
held="$study/rented-fresh-v1"
election="$study/election-v1"
audit="$study/audit"
panel="$study/panel"
verify() (cd "$1" && sha256sum --strict --quiet -c "${2:-SHA256SUMS}")
digest() { sha256sum -- "$1" | cut -d' ' -f1; }

case "$mode" in
stage)
    for sealed in "$train" "$election" "$panel/fresh" "$audit/build-v3" \
        "$study/fit/portable-v1/sources" "$study/fit/recovery/lineage-v3" \
        "$study/fit/recovery/crossovers-v1" "$study/rented-finite-v1" \
        "$study/rented-score-fixture-v1" "$study/shipment/train-input" "$study/shipment/release" \
        "$study/shipment/score-code/source"; do
        verify "$sealed"
    done
    verify "$panel" training.sha256
    mkdir -p "$archive"
    mkdir "$packet" "$packet/"{methods,models,panel,primary,expected,physical,independent,custody,supplemental}
    cp -- "$here/replay.sh" "$packet/methods/replay.sh"
    mkdir "$packet/methods/pack"
    cp -- "$here/"{wick.sh,semantic.sh,README.md} "$packet/methods/pack/"
    cp -a -- "$here/grade" "$packet/methods/grade"
    for file in elect.jq election.sh grade.sh near.jq pairs.jq rows.jq; do
        cmp "$packet/methods/grade/$file" "$study/analysis-code-v1/grade/$file"
    done
    cmp "$packet/methods/grade/all.sh" "$study/grade-ops-v1/all.sh"
    cp -a -- "$study/fit/portable-v1/sources" "$packet/methods/fit"
    cp -a -- "$study/fit/recovery/lineage-v3/source" "$packet/methods/lineage"
    cp -a -- "$study/fit/recovery/crossovers-v1/source" "$packet/methods/recovery"
    cp -a -- "$study/shipment/score-code/source" "$packet/methods/scorer"
    cp -a -- "$panel/methods-v8" "$packet/methods/panel"
    cp -a -- "$panel/generator-build-v3/source/docs" "$packet/methods/panel/generated-source"
    cp -- "$panel/generator-build-v3/"{authority.sha256,patch.log,build.log,compare-patch.log,compare-build.log} "$packet/methods/panel/"
    cp -- "$study/analysis-code-v1/"{run-train.sh,run-fresh.sh} "$packet/methods/"

    cp -a -- "$study/shipment/train-input/model/." "$packet/models/"
    codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
    printf '%s  %s\n' a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec" | sha256sum --strict --quiet -c -
    cp -- "$codec" "$packet/models/"
    cmp "$packet/models/protected.state" "$train/fit-out/original.state"
    cmp "$packet/models/ordered.state" "$train/fit-out/median.state"
    cp -a -- "$panel/"{train,regression,near,fresh,training.sha256,training-authority.sha256,history.json} "$packet/panel/"
    cp -a -- "$train/"{fit-out,score,prefit,release,run-train.sh} "$packet/primary/"
    cp -- "$train/SHA256SUMS" "$packet/custody/rented-train.members.sha256"
    cp -a -- "$election/." "$packet/expected/"
    mkdir "$packet/independent/build-v3"
    find "$audit/build-v3" -maxdepth 1 -type f -exec cp -p -- '{}' "$packet/independent/build-v3/" \;
    cp -a -- "$(realpath -e "$audit/build-v3/tokenizer")" "$packet/independent/build-v3/tokenizer"
    for dir in archived-near-v1 train-census-v3 train-baseline-v1 train-finalists-v1 fresh-census-v2 \
        train-fit-v1 score-finite-v1 fit-finite-v2 election-finite-v2; do
        verify "$audit/$dir"
        cp -a -- "$audit/$dir" "$packet/independent/"
    done
    cp -a -- "$study/fit/recovery/lineage-v3" "$packet/independent/fit-lineage"
    cp -a -- "$study/fit/recovery/crossovers-v1" "$packet/supplemental/recovery-prepared"
    cp -a -- "$study/rented-finite-v1" "$packet/physical/fit-finite"
    cp -a -- "$study/rented-score-fixture-v1" "$packet/physical/scorer-finite"
    cp -a -- "$study/shipment/release" "$packet/custody/train-release"
    cp -- "$study/shipment/train-input/SHA256SUMS" "$packet/custody/rented-train-input.members.sha256"
    if grep -q '^recovery-' "$election/all-states.tsv"; then
        recovery="$study/rented-recovery-v1"
        verify "$recovery"
        mkdir "$packet/recovery"
        cp -a -- "$recovery/scores" "$packet/recovery/"
        cp -a -- "$packet/supplemental/recovery-prepared/states" "$packet/recovery/"
        cp -- "$recovery/SHA256SUMS" "$packet/custody/rented-recovery.members.sha256"
    fi
    date -u +%FT%TZ > "$archive/staged-utc.txt"
    ;;

seal)
    [[ -e "$archive/staged-utc.txt" && ! -e "$packet/MEMBERS.sha256" && ! -e "$tarball" ]]
    for sealed in "$held" "$study/controls-v1" "$audit/fresh-replay-v1" "$study/device-records-v1" \
        "$panel/fresh-reduction-v1" "$study/physical-v1"; do
        verify "$sealed"
    done
    mkdir "$packet/held" "$packet/expected/fresh"
    cp -a -- "$held/held-raw/." "$packet/held/"
    cp -- "$held/SHA256SUMS" "$packet/custody/rented-fresh.members.sha256"
    if [[ -e "$held/run-fresh.sh" ]]; then cp -- "$held/run-fresh.sh" "$packet/custody/rented-run-fresh.sh"; fi
    cp -a -- "$study/device-records-v1" "$packet/custody/device-records"
    cp -- "$study/physical-v1/SHA256SUMS" "$packet/custody/whole-rental.members.sha256"
    cp -a -- "$panel/fresh-reduction-v1" "$packet/expected/enriched-fresh"
    cp -a -- "$panel/reduction-methods-v1" "$packet/methods/fresh-reduction"
    cp -a -- "$audit/fresh-replay-v1" "$packet/independent/fresh-replay"
    cp -- "$study/controls-v1/controls.json" "$packet/expected/controls.json"
    cp -a -- "$study/controls-v1" "$packet/physical/controls"
    if [[ -d "$held/fresh-input" ]]; then
        verify "$held/fresh-input"
        cp -a -- "$held/fresh-input" "$packet/custody/fresh-release"
    fi
    [[ $(wc -l < "$packet/held/readback/states.tsv") -eq 4 ]]
    while IFS=$'\t' read -r state payload; do
        [[ $state =~ ^[a-zA-Z0-9_-]+$ && $payload == "$state.state" ]]
        mkdir "$packet/expected/fresh/$state"
        cp -- "$study/grades-fresh-v1/$state/"{fresh.json,fresh.rows.jsonl} "$packet/expected/fresh/$state/"
        origin="$packet/primary/fit-out/$payload"
        if [[ $state == recovery-* ]]; then origin="$packet/recovery/states/$payload"; fi
        cmp "$origin" "$packet/held/readback/$payload"
    done < "$packet/held/readback/states.tsv"
    test -z "$(find "$packet" ! -type f ! -type d -print -quit)"
    (cd "$packet" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$archive/MEMBERS.pending")
    mv -- "$archive/MEMBERS.pending" "$packet/MEMBERS.sha256"
    verify "$packet" MEMBERS.sha256
    date -u +%FT%TZ > "$archive/sealed-utc.txt"
    tar --sort=name --format=posix -C "$archive" -I 'zstd -T2 -5' -cf "$tarball" packet
    (
        cd "$archive"
        sha256sum packet/MEMBERS.sha256 "${tarball##*/}" > SHA256SUMS
        sha256sum --strict --quiet -c SHA256SUMS
    )
    jq -n --arg created "$(cat "$archive/sealed-utc.txt")" \
        --arg archive "${tarball##*/}" --arg archive_sha "$(digest "$tarball")" \
        --arg members_sha "$(digest "$packet/MEMBERS.sha256")" \
        --argjson archive_bytes "$(stat -c %s "$tarball")" \
        --argjson members "$(wc -l < "$packet/MEMBERS.sha256")" \
        '{created_utc:$created,archive:$archive,archive_sha256:$archive_sha,archive_bytes:$archive_bytes,
          members_sha256:$members_sha,members:$members}' > "$archive/receipt.json"
    cat "$archive/receipt.json"
    ;;

cold)
    verify "$archive"
    mkdir "$cold"
    tar -I zstd --no-same-owner --no-same-permissions -xf "$tarball" -C "$cold"
    restored="$cold/packet"
    cmp "$restored/MEMBERS.sha256" "$packet/MEMBERS.sha256"
    bash "$restored/methods/replay.sh" "$restored" "$cold/verification" > "$cold/replay.log"
    bash "$restored/methods/pack/semantic.sh" "$restored" "$cold/independent" > "$cold/independent.log"
    jq -n --arg receipt "$(digest "$archive/receipt.json")" \
        --arg utc "$(date -u +%FT%TZ)" \
        --arg election "$(digest "$cold/verification/election/election.json")" \
        --arg controls "$(digest "$cold/verification/controls.json")" \
        --arg independent "$(digest "$cold/independent/SHA256SUMS")" \
        '{pass:true,verified_utc:$utc,archive_receipt_sha256:$receipt,election_sha256:$election,
          controls_sha256:$controls,independent_sha256:$independent}' > "$cold/receipt.json"
    (cd "$cold" && sha256sum receipt.json replay.log independent.log independent/SHA256SUMS > SHA256SUMS)
    cat "$cold/receipt.json"
    sha256sum "$cold/SHA256SUMS"
    ;;

*) exit 64 ;;
esac
