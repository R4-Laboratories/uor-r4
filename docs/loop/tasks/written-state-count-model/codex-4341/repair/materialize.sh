#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 7 ]] || {
    echo 'usage: materialize.sh PRIOR_4340_EXTRACTION FROZEN_SELECTION REGISTRATION.json SEALED_4338 OFFICIAL_DEV.json PRIOR_VALIDATION_CODE FRESH_EXTRACTION' >&2
    exit 2
}
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=custody.sh
source "$here/custody.sh"
prior=$(realpath "$1")
selection=$(realpath "$2")
registered=$(realpath "$3")
sealed=$(realpath "$4")
official=$(realpath "$5")
driver=$(realpath "$6")
out=$(realpath -m "$7")

verify_prior_selection "$prior"
verify_selection_seal "$selection" "$registered"
# Reproduce the registered selection before accessing semantic source.
jq -c --slurpfile old "$prior/selection/review-identities.jsonl" -f "$here/expand.jq" \
    "$prior/selection/selection.json" | cmp - "$selection/additional-identities.jsonl"
cat "$prior/selection/review-identities.jsonl" "$selection/additional-identities.jsonl" |
    cmp - "$selection/review-identities.jsonl"

pin ee89888a4ef1201902d1eeb836a8252e9bf4ff740be446a4ed79fc1e930178e0 "$driver/extract-selected.jq"
pin 80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8 "$official"
pin 1899b7c8e52bc56e08d084e1286855e167d4a5665c13a72538e75e9f1c6c984f "$sealed/SEAL.sha256"
pin "$(member_pin "$sealed/SEAL.sha256" ROSTER.sha256)" "$sealed/ROSTER.sha256"
pin "$(member_pin "$sealed/ROSTER.sha256" roster.jsonl)" "$sealed/roster.jsonl"
verify_prior_semantic "$prior"

mkdir "$out"
cd "$out"
mkdir selection selected-semantic
cp "$prior/selection/selection.json" selection/prior-selection.json
cp "$selection/"{additional-identities,review-identities}.jsonl selection/
cp "$registered" selection/registration.json
(cd selection && sha256 ./*.json ./*.jsonl > SELECTION.sha256)

jq -c --slurpfile picks selection/additional-identities.jsonl '
  ($picks | INDEX(.passage_sha256)) as $wanted |
  select(.split == "validation" and $wanted[.passage_sha256] != null)
' "$sealed/roster.jsonl" > selected-semantic/additional-answerability-source.jsonl

jq -c --arg family answerability --slurpfile picks selection/additional-identities.jsonl \
    --slurpfile source selected-semantic/additional-answerability-source.jsonl \
    -f "$driver/extract-selected.jq" "$official" > selected-semantic/additional-source.jsonl

# Check the original full-paragraph identity in custody. Reviewers see only the
# window given to the client. Full paragraph bytes never enter a review file.
jq -r --slurpfile selected selected-semantic/additional-source.jsonl '
  (.data | INDEX(.title)) as $by_title | $selected[] |
  [.passage_sha256,($by_title[.article].paragraphs[.paragraph].context | @base64)] | @tsv
' "$official" | while IFS=$'\t' read -r expected encoded; do
    digest=$(printf '%s' "$encoded" | openssl base64 -d -A | sha256)
    [[ ${digest%% *} == "$expected" ]] || fail "Original paragraph identity mismatch: $expected"
done

jq -ec --slurpfile picks selection/additional-identities.jsonl '
  map({article,passage_sha256,family,rank,pair_id,left:.left.case_id,right:.right.case_id}) == $picks
' -s selected-semantic/additional-source.jsonl >/dev/null
jq -c 'del(.left.prefix,.right.prefix,.left.user_bounds,.right.user_bounds)' \
    selected-semantic/additional-source.jsonl > selected-semantic/additional-rows.jsonl

cat "$prior/selected-semantic/source.jsonl" selected-semantic/additional-source.jsonl > selected-semantic/source.jsonl
cat "$prior/selected-semantic/rows.jsonl" selected-semantic/additional-rows.jsonl > selected-semantic/rows.jsonl
jq -ec --slurpfile picks selection/review-identities.jsonl '
  map({article,passage_sha256,family,rank,pair_id,left:.left.case_id,right:.right.case_id}) as $rows |
  ($rows[:67]|sort_by(.pair_id)) == ($picks[:67]|sort_by(.pair_id)) and $rows[67:] == $picks[67:]
' -s selected-semantic/source.jsonl >/dev/null

# Relative logical names make these receipts stable after restoring to a fresh path.
record() {
    local digest
    digest=$(sha256 < "$1")
    printf '%s  %s\n' "${digest%% *}" "$2"
}
{
    record "$prior/EXTRACTION.sha256" prior-4340-extraction
    record "$sealed/SEAL.sha256" parent-4338-seal
    record "$sealed/ROSTER.sha256" parent-4338-roster-seal
    record "$official" official-dev
    record "$driver/extract-selected.jq" extract-selected-jq
    for script in custody.sh expand.jq select.sh materialize.sh; do
        record "$here/$script" "$script"
    done
} > SOURCE-PINS.sha256
(cd selected-semantic && sha256 ./*.jsonl > SEMANTIC.sha256)
sha256 SOURCE-PINS.sha256 selection/SELECTION.sha256 selected-semantic/SEMANTIC.sha256 > EXTRACTION.sha256
# Hand each reviewer only selected-semantic/additional-rows.jsonl.
