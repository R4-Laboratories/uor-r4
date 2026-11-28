#!/usr/bin/env bash
# Assemble a frozen scorer roster and separate label-blind evaluation inputs.
set -euo pipefail
umask 077
if [[ $# != 9 ]]; then
    echo 'usage: assemble-roster.sh TOOL {4340|4341} EXTRACTION EXTRACT_SHA QUALIFICATION_ROOT QUALIFICATION_SHA CONTROLS CONTROLS_SHA NEW_OUT' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=../repair/custody.sh
source "$here/../repair/custody.sh"
tool=$(realpath "$1")
study=$2
extraction=$(realpath "$3")
extraction_sha=$4
qualified_root=$(realpath "$5")
qualification_sha=$6
controls=$(realpath "$7")
controls_sha=$8
out=$(realpath -m "$9")

pin "$extraction_sha" "$extraction/EXTRACTION.sha256"
pin "$(member_pin "$extraction/EXTRACTION.sha256" selection/SELECTION.sha256)"     "$extraction/selection/SELECTION.sha256"
(cd "$extraction/selection" && sha256 --strict --quiet -c SELECTION.sha256)
pin "$qualification_sha" "$qualified_root/SEALED.sha256"
pin "$controls_sha" "$controls/CONTROLS.sha256"
(cd "$controls" && sha256 --strict --quiet -c CONTROLS.sha256)
case "$study" in
    4340)
        verify_prior_selection "$extraction"
        qual=$qualified_root
        selection=$extraction/selection/selection.json
        registration=$here/../../codex-4340/validation/REGISTRATION.md
        study_id=original4340
        ;;
    4341)
        qual=$qualified_root/qualification
        selection=$extraction/selection/prior-selection.json
        registration=$here/../REGISTRATION.json
        cmp -s "$registration" "$extraction/selection/registration.json"
        verify_selection_seal "$extraction/selection" "$registration"
        pin "$(member_pin "$qualified_root/SEALED.sha256" qualification/SEALED.sha256)"             "$qual/SEALED.sha256"
        pin "$(member_pin "$qualified_root/SEALED.sha256" driver-pins.json)"             "$qualified_root/driver-pins.json"
        test "$(jq -er .extraction "$qualified_root/driver-pins.json")" = "$extraction_sha"
        study_id=expanded4341
        ;;
    *) fail 'Unregistered study';;
esac
(cd "$qual" && sha256 --strict --quiet -c SEALED.sha256)
digest() { sha256 < "$1" | cut -d' ' -f1; }

mkdir "$out"
cd "$out"
"$tool" validation "$study" "$selection" "$extraction/selection/review-identities.jsonl"     "$qual" > titles.json
for role in natural original protected; do
    p=$controls/$role
    "$tool" controls "$role" "$p/input.jsonl" "$p/gold.jsonl" "$p/truth.jsonl"         "$p/openings.jsonl" "$p/features.jsonl" > "$role.json"
done
jq -n --arg study "$study_id" --arg reg "$(digest "$registration")"     --arg selection "$(digest "$extraction/selection/review-identities.jsonl")"     --arg qualification "$qualification_sha" --arg extraction "$extraction_sha"     --slurpfile titles titles.json --slurpfile original original.json     --slurpfile natural natural.json --slurpfile protected protected.json '
    {schema:"uor-r4.reciprocal-transfer-validation-roster/1",$study,registration_sha256:$reg,
     selection_sha256:$selection,judgments_sha256:$qualification,source_auth_sha256:$extraction,
     tokenizer_sha256:"a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
     protected_reference_sha256:"da8667c7f7659ac0553bf0ff84ab9d2f836f1e22a103a19a399993b847e839b7",
     titles:$titles[0],training:[{mixture:"original",sources:$original[0]},
       {mixture:"natural",sources:$natural[0]}],protected:$protected[0]}
' > roster.json
roster_sha=$(digest roster.json)
"$tool" coverage roster.json "$roster_sha" > coverage.json

mkdir -p blind/validation blind/protected
jq -c --slurpfile pairs "$qual/pairs.jsonl" '
    ($pairs | map(select(.qualified) | .left_case_id,.right_case_id) | INDEX(.)) as $needed |
    select($needed[.case_id] != null)
' "$qual/input.jsonl" > blind/validation/input.jsonl
jq -c --slurpfile input blind/validation/input.jsonl '
    ($input | INDEX(.case_id)) as $needed | select($needed[.case_id] != null)
' "$qual/features.jsonl" > blind/validation/features.jsonl
jq -ce --slurpfile input blind/validation/input.jsonl --slurpfile q "$qual/pairs.jsonl" -n '
    ($input | map(.case_id)) as $ids |
    [$q[] | select(.qualified) | (.left_case_id, .right_case_id)] == $ids and
    ($ids|unique|length)==($ids|length)
' >/dev/null
jq -ce --slurpfile pairs "$qual/pairs.jsonl" --slurpfile input blind/validation/input.jsonl -n '
    ($input | INDEX(.case_id)) as $blind |
    $pairs[] | select(.qualified) |
    {request_id:.left_case_id,case_id:.right_case_id},
    {request_id:.right_case_id,case_id:.left_case_id} |
    . + {text:$blind[.case_id].text}
' > swap-requests.jsonl
jq -c '{case_id,text}' swap-requests.jsonl > blind/validation/swaps.jsonl
for part in input features; do
    cp "$controls/protected/$part.jsonl" "blind/protected/$part.jsonl"
done
# Private writer adjudication projection. This is kept out of blind/.
pin "$(member_pin "$extraction/EXTRACTION.sha256" selected-semantic/SEMANTIC.sha256)"     "$extraction/selected-semantic/SEMANTIC.sha256"
pin "$(member_pin "$extraction/selected-semantic/SEMANTIC.sha256" ./source.jsonl)"     "$extraction/selected-semantic/source.jsonl"
jq -c --slurpfile input blind/validation/input.jsonl '
    ($input|INDEX(.case_id)) as $blind |
    . as $row | (.left,.right) | select($blind[.case_id] != null) |
    {case_id,passage:$row.context,question}
' "$extraction/selected-semantic/source.jsonl" > writer-contexts.jsonl
jq -r .case_id writer-contexts.jsonl | cmp - <(jq -r .case_id blind/validation/input.jsonl)

(cd blind && sha256 ./*/*.jsonl > BLIND.sha256)
sha256 roster.json coverage.json blind/BLIND.sha256 swap-requests.jsonl writer-contexts.jsonl > ROSTER.sha256
