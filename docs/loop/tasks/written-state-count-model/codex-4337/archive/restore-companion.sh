#!/usr/bin/env bash
set -euo pipefail
umask 077

packet=$(realpath "${1:?downloaded companion packet}")
receipt=$(realpath "${2:?companion JSON receipt}")
root=$(realpath -m "${3:?fresh companion restore root}")
here=$(cd "$(dirname "$0")" && pwd)
bash "$here/restore.sh" "$packet" "$receipt" "$root"

(
    cd "$root/next-supply-pilot-v1"
    for seal in SPLIT.sha256 PICKS.sha256 SOURCE.sha256 TRAIN-NEGATIVES.sha256 \
        MANIFEST.sha256 FOLLOW-THROUGH.sha256; do
        sha256sum --strict --quiet -c "$seal"
    done
)
(
    cd "$root/companion-source-v1"
    sha256sum --strict --quiet -c PILOT-SOURCES.sha256
)
validation=$(jq -er .grade_validation.path "$receipt")
[[ "$validation" == regrade-v3/* && "$validation" != *..* ]]
printf '%s  %s\n' "$(jq -er .grade_validation.sha256 "$receipt")" "$root/$validation" \
    | sha256sum --strict --quiet -c -
printf '%s  %s\n' "$(jq -er .grade_validation.manifest_sha256 "$receipt")" \
    "$root/regrade-v3/OUTPUTS.sha256" | sha256sum --strict --quiet -c -
(cd "$root/regrade-v3/source" && sha256sum --strict --quiet -c SOURCES.sha256)
(cd "$root/regrade-v3/negative-controls" && sha256sum --strict --quiet -c REJECTIONS.sha256)
