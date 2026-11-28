# shellcheck shell=bash
# Shared byte-level custody checks. The selector calls only verify_prior_selection.
if command -v gsha256sum >/dev/null 2>&1; then
    sha256() { gsha256sum "$@"; }
else
    sha256() { sha256sum "$@"; }
fi

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

pin() {
    [[ $1 =~ ^[0-9a-f]{64}$ ]] || fail 'Malformed SHA-256 pin'
    local digest
    digest=$(sha256 - < "$2")
    [[ ${digest%% *} == "$1" ]] || fail "SHA-256 mismatch for $2"
}

member_pin() {
    local manifest=$1 member=$2
    awk -v member="$member" '
        $2 == member && NF == 2 { digest = $1; count++ }
        END { if (count != 1) exit 1; print digest }
    ' "$manifest"
}

verify_prior_selection() {
    local prior=$1 manifest=$1/selection/SELECTION.sha256
    pin ae7a726d7d8880ee6f6db02575922320e61c8773b2d181e4bc328a44305cee52 "$prior/EXTRACTION.sha256"
    pin "$(member_pin "$prior/EXTRACTION.sha256" selection/SELECTION.sha256)" "$manifest"
    pin "$(member_pin "$manifest" ./selection.json)" "$prior/selection/selection.json"
    pin "$(member_pin "$manifest" ./review-identities.jsonl)" "$prior/selection/review-identities.jsonl"
}

verify_prior_semantic() {
    local prior=$1 manifest=$1/selected-semantic/SEMANTIC.sha256
    pin "$(member_pin "$prior/EXTRACTION.sha256" selected-semantic/SEMANTIC.sha256)" "$manifest"
    pin "$(member_pin "$manifest" ./source.jsonl)" "$prior/selected-semantic/source.jsonl"
    pin 2a3baa08fe9b692e22c6b153b949e8b40f0c3d0a251f2df547743744ef9ee3d3 "$prior/selected-semantic/rows.jsonl"
    pin "$(member_pin "$manifest" ./rows.jsonl)" "$prior/selected-semantic/rows.jsonl"
}

verify_selection_seal() {
    local selection=$1 registered=$2
    pin "$(jq -er .additional_identities_sha256 "$registered")" "$selection/additional-identities.jsonl"
    pin "$(jq -er .expanded_identities_sha256 "$registered")" "$selection/review-identities.jsonl"
}
