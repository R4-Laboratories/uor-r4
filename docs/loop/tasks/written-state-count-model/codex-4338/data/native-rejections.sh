#!/usr/bin/env bash
set -euo pipefail
umask 077
fit=$(realpath "${1:?pinned interval-fit}")
train=$(realpath "${2:?qualified TRAIN input}")
native=$(realpath "${3:?native capture corresponding to TRAIN input}")
out=$(realpath -m "${4:?fresh rejection evidence directory}")
here=$(cd "$(dirname "$0")" && pwd)
(cd "$train" && sha256sum --quiet --strict -c SEALED.sha256)
(cd "$native" && sha256sum --quiet --strict -c SEALED.sha256)
mkdir "$out"
cd "$out"
positive=$(jq -ers 'first(.[]|select(.start != null))|.case_id' "$train/train-gold.jsonl")
no_read=$(jq -ers 'first(.[]|select(.start == null))|.case_id' "$train/train-gold.jsonl")
facts="$native/train.prospective.features.jsonl"

eligibility() {
    jq -nc --slurpfile train "$train/train-input.jsonl" --slurpfile gold "$1" \
        --slurpfile truth "$2" --slurpfile opening "$native/train.openings.jsonl" \
        --slurpfile capture "$facts" -f "$here/native.jq"
}
reject() {
    local label=$1 phrase=$2
    shift 2
    if "$@" > "$label.stdout" 2> "$label.stderr"; then
        echo "$label was incorrectly accepted" >&2
        exit 1
    fi
    grep -Fq -- "$phrase" "$label.stderr"
}

# Keep the original tokens and change only the byte position of the annotation.
jq -c --arg id "$positive" 'if .case_id == $id then .original_bytes[0] += 1 else . end' \
    "$train/train-truth.jsonl" > changed-bytes.jsonl
reject changed-bytes 'original source IDs differ at own positions' \
    eligibility "$train/train-gold.jsonl" changed-bytes.jsonl

# Keep the native candidate stream and gold nulls but attach source IDs to NoRead.
jq -c --arg id "$no_read" 'if .case_id == $id then .source_ids = [21] else . end' \
    "$train/train-truth.jsonl" > forged-no-read.jsonl
reject forged-no-read 'original source IDs differ at own positions' \
    eligibility "$train/train-gold.jsonl" forged-no-read.jsonl

# Bindings must carry the first-class NoRead choice for each arm.
jq -c --arg id "$no_read" 'if .case_id == $id and .arm == "count" then del(.no_read_keys) else . end' \
    "$facts" > missing-no-read.jsonl
reject missing-no-read 'missing field `no_read_keys`' \
    "$fit" coverage missing-no-read.jsonl "$train/train-gold.jsonl"

printf '%s\n' '{"changed_original_bytes":"REJECTED","forged_no_read_source":"REJECTED","missing_native_no_read":"REJECTED"}' \
    > summary.json
sha256sum "$fit" "$here/native.jq" "$here/native-rejections.sh" "$train/SEALED.sha256" \
    "$native/SEALED.sha256" > SOURCES.sha256
sha256sum ./*.jsonl ./*.stdout ./*.stderr SOURCES.sha256 summary.json > SEALED.sha256
sha256sum --quiet --strict -c SEALED.sha256
cat summary.json
