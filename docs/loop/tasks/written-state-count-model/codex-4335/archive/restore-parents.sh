#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 2 ]; then
    echo "usage: restore-parents.sh WICK_RUNS NEW_PARENT_RESTORE" >&2
    exit 2
fi
runs=$(realpath "$1")
parents=$(realpath -m "$2")
studies=$(cd "$(dirname "$0")/../.." && pwd)
mkdir "$parents"

for study in 4332 4333 4334; do
    source="$studies/codex-$study"
    receipt="$source/evidence/archive.json"
    repository=$(jq -er .repository "$receipt")
    revision=$(jq -er .revision "$receipt")
    packet=$(jq -er .path "$receipt")
    digest=$(jq -er .sha256 "$receipt")
    manifest=$(jq -er .members_manifest_sha256 "$receipt")
    hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
        jq -e --arg revision "$revision" '.private == true and .sha == $revision' >/dev/null
    hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
        --force-download --local-dir "$parents/download-$study" >"$parents/download-$study.log" 2>&1

    inputs=()
    case "$study" in
        4333) inputs+=("$parents/4332") ;;
        4334) inputs+=("$parents/4333" "$parents/4332") ;;
    esac
    if ! bash "$source/archive/restore.sh" "$runs" "$parents/$study" \
        "$parents/download-$study/$packet" "$digest" "${inputs[@]}" \
        >"$parents/restore-$study.log" 2>&1; then
        printf 'Parent %s restore failed. See %s\n' "$study" "$parents/restore-$study.log" >&2
        exit 1
    fi
    printf '%s  %s\n' "$manifest" "$parents/$study/payload/MEMBERS.sha256" | sha256sum --strict -c
done
