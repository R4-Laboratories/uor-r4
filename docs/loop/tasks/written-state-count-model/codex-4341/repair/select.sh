#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 2 ]] || { echo 'usage: select.sh PRIOR_4340_EXTRACTION FRESH_SELECTION' >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=custody.sh
source "$here/custody.sh"
prior=$(cd "$1" && pwd -P)
verify_prior_selection "$prior"
mkdir "$2"
cd "$2"
cp "$prior/selection/selection.json" prior-selection.json
cp "$prior/selection/review-identities.jsonl" prior-identities.jsonl
jq -c --slurpfile old prior-identities.jsonl -f "$here/expand.jq" prior-selection.json > additional-identities.jsonl
cat prior-identities.jsonl additional-identities.jsonl > review-identities.jsonl
sha256 ./*.json ./*.jsonl > SELECTION.sha256
sha256 additional-identities.jsonl review-identities.jsonl
