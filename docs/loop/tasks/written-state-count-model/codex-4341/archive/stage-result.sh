#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 2 ]]; then
    echo 'usage: stage-result.sh FRESHLY_STAGED_REPAIR FINAL_RUN.json' >&2
    exit 2
fi
stage=$(realpath "$1")
run=$(realpath "$2")
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/../repair/custody.sh"
test -s "$stage/roster/ROSTER.sha256"
test ! -e "$stage/result"

jq -e '
    .schema=="uor-r4-natural-reciprocal-repair-final-run/1" and .trial=="finished" and
    (.source_status|IN("negative","pass","instrument_inconclusive")) and
    (.writer_status|IN("pass","fail","unresolved","instrument_inconclusive")) and
    (.locked_evaluation|IN("NOT_RUN","SEPARATELY_REGISTERED")) and
    (.diagnostics|type)=="array" and
    ((.diagnostics|map(.name)|length)==(.diagnostics|map(.name)|unique|length))
' "$run" >/dev/null

root() { jq -er --arg role "$1" '.roots[$role] | strings | select(startswith("/"))' "$run"; }
receipt() { jq -er --arg role "$1" '.pins[$role]' "$run"; }
verify() {
    local role=$1 manifest=$2 directory
    directory=$(root "$role")
    pin "$(receipt "$role")" "$directory/$manifest"
    (cd "$directory" && sha256 --strict --quiet -c "$manifest")
}

verify device OUTPUTS.sha256
verify replay OUTPUTS.sha256
verify cold OUTPUTS.sha256
verify source EVIDENCE.sha256
verify source_grade SOURCE-GRADE.sha256
verify lease LEASE.sha256
device=$(root device)
replay=$(root replay)
cold=$(root cold)
source_evidence=$(root source)
writer=$(root writer)
lease=$(root lease)

test "$(find "$replay" -maxdepth 1 -type f -name '*.scores' | wc -l)" -eq 24
test "$(find "$cold" -maxdepth 1 -type f -name '*.d11' | wc -l)" -eq 12
test "$(find "$device" -maxdepth 1 -type f -name '*.words' | wc -l)" -eq 8
test "$(find "$device/repeat" -maxdepth 1 -type f -name '*.words' | wc -l)" -eq 8
test -d "$writer"
test -d "$lease"
test "$(cat "$device/../job.exit")" = 0
test "$(cat "$replay/../replay.exit")" = 0
test -s "$source_evidence/evidence.json"
jq -e '(.models|length)==12' "$source_evidence/evidence.json" >/dev/null
test "$(jq -er .source_status "$run")" = "$(jq -er .source_status "$(root source_grade)/source-report.json")"

writer_status=$(jq -er .writer_status "$run")
if [[ $writer_status != instrument_inconclusive ]]; then
    verify writer WRITER-MATRIX.sha256
    verify writer_grade WRITER-GRADE.sha256
    (cd "$writer" && sha256 --strict --quiet -c WRITER-OPERATIONS.sha256)
    test "$writer_status" = "$(jq -er .writer_status "$(root writer_grade)/writer-report.json")"
    test "$(cat "$writer/matrix.exit")" = 0
    projection=$(jq -er '.writer_projection|select(IN("projected-deterministic","projected-adjudicated"))' "$run")
    (cd "$writer/$projection" && sha256 --strict --quiet -c WRITER.sha256)
    for mix in original natural; do
        for key in count geometry; do
            for fit in actual permuted zero; do
                name=$mix.$key.$fit
                test "$(cat "$writer/dispositions/$name.exit")" = 0
                (cd "$writer/capture/$name" && sha256 --strict --quiet -c CAPTURE.sha256)
                (cd "$writer/verified/$name" && sha256 --strict --quiet -c TRACE.sha256)
                (cd "$writer/visible/$name" && sha256 --strict --quiet -c VISIBLE.sha256)
            done
        done
    done
fi

copy_tree() {
    local from=$1 to=$2
    mkdir "$to"
    tar --exclude='._*' -cf - -C "$from" . | tar -xf - -C "$to"
}
mkdir "$stage/result" "$stage/result/cold" "$stage/result/device-job" "$stage/result/replay-job"
for role in device replay source source_grade writer lease; do
    copy_tree "$(root "$role")" "$stage/result/$role"
done
cp "$device/../"{job.exit,job.stdout,job.stderr,job.wall_seconds,started-utc,finished-utc} "$stage/result/device-job/"
cp "$device/../../device-summary.jsonl" "$stage/result/device-job/"
cp "$replay/../"{replay.exit,replay.sh,replay.stdout,replay.stderr} "$stage/result/replay-job/"
if [[ $writer_status != instrument_inconclusive ]]; then
    copy_tree "$(root writer_grade)" "$stage/result/writer_grade"
fi
# The cold manifest continues to bind the 12 exact rebuilt artifacts. Copy
# every other receipt and trace without altering that manifest.
(
    cd "$cold"
    find . -type f ! -name '*.d11' ! -name '._*' -print0 |
        tar --null -T - -cf - |
        tar -xf - -C "$stage/result/cold"
)
while IFS=$'\t' read -r name directory manifest digest; do
    [[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && $directory == /* ]]
    if [[ -n $manifest || -n $digest ]]; then
        [[ $manifest =~ ^[A-Z][A-Z0-9-]*\.sha256$ ]]
        pin "$digest" "$directory/$manifest"
        (cd "$directory" && sha256 --strict --quiet -c "$manifest")
    fi
    test ! -e "$stage/diagnostics/$name"
    copy_tree "$directory" "$stage/diagnostics/$name"
done < <(jq -r '.diagnostics[] | [.name,.path,.manifest // "",.sha256 // ""] | @tsv' "$run")
cp "$run" "$stage/result/FINAL-RUN.json"
