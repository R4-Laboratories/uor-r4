#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 ]]; then
    echo 'usage: regrade.sh RESTORED_RESULT VERIFIED_SCORER FRESH_OUT' >&2
    exit 2
fi
packet=$(realpath "$1")
tool=$(realpath "$2")
out=$(realpath -m "$3")
result=$packet/result
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/../repair/custody.sh"
(cd "$packet" && sha256 --strict --quiet -c MEMBERS.sha256)
run=$result/FINAL-RUN.json
jq -e '.trial=="finished"' "$run" >/dev/null
pin "$(jq -er .pins.source "$run")" "$result/source/EVIDENCE.sha256"
pin "$(jq -er .pins.source_grade "$run")" "$result/source_grade/SOURCE-GRADE.sha256"
roster=$packet/roster/roster.json
evidence=$result/source/evidence.json
roster_sha=$(member_pin "$packet/roster/ROSTER.sha256" roster.json)
evidence_sha=$(member_pin "$result/source/EVIDENCE.sha256" ./evidence.json)
mkdir "$out"
"$tool" source "$roster" "$roster_sha" "$evidence" "$evidence_sha" > "$out/source-report.json"
cmp "$result/source_grade/source-report.json" "$out/source-report.json"
if [[ $(jq -er .writer_status "$run") != instrument_inconclusive ]]; then
    pin "$(jq -er .pins.writer_grade "$run")" "$result/writer_grade/WRITER-GRADE.sha256"
    projection=$(jq -er '.writer_projection|select(IN("projected-deterministic","projected-adjudicated"))' "$run")
    (
        cd "$result/writer/$projection"
        sha256 --strict --quiet -c WRITER.sha256
    )
    writer=$result/writer/$projection/writer.json
    writer_sha=$(sha256 < "$writer" | cut -d' ' -f1)
    "$tool" writer "$roster" "$roster_sha" "$evidence" "$evidence_sha" "$writer" "$writer_sha" \
        > "$out/writer-report.json"
    cmp "$result/writer_grade/writer-report.json" "$out/writer-report.json"
fi
