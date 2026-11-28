#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 13 ]]; then
    echo 'usage: capture.sh STAGE BLIND_SHA TRUTH_SHA GROUPS_SHA BASE TOKENIZER TOKENIZER_SHA OLD_CLIENT OLD_CLIENT_SHA NEW_CLIENT NEW_CLIENT_SHA PROBE FRESH_OUTPUT' >&2
    exit 2
fi
stage=$(realpath "$1")
blind_sha=$2
truth_sha=$3
groups_sha=$4
base=$(realpath "$5")
tokenizer=$(realpath "$6")
tokenizer_sha=$7
old_client=$(realpath "$8")
old_client_sha=$9
new_client=$(realpath "${10}")
new_client_sha=${11}
probe=$(realpath "${12}")
out=$(realpath -m "${13}")
base_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
source_dir=$(cd "$(dirname "$0")" && pwd)

pin() {
    [[ $1 =~ ^[0-9a-f]{64}$ ]]
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -
}

pin "$blind_sha" "$stage/blind.jsonl"
pin "$truth_sha" "$stage/truth.jsonl"
pin "$groups_sha" "$stage/groups.jsonl"
pin "$base_sha" "$base"
pin "$tokenizer_sha" "$tokenizer"
pin "$old_client_sha" "$old_client"
pin "$new_client_sha" "$new_client"
[[ "$old_client_sha" != "$new_client_sha" ]]
(
    cd "$source_dir"
    sha256sum --strict --quiet -c SOURCE.sha256
)

mkdir "$out"
"$probe" preflight "$stage/blind.jsonl" "$blind_sha" "$base" "$tokenizer" "$tokenizer_sha" \
    > "$out/admission.json"
jq -c '.admitted[]' "$out/admission.json" > "$out/admitted.jsonl"
# Both clients receive exactly the same label-blind prompts and bare artifact.
"$old_client" --discrete "$base" --tokenizer "$tokenizer" \
    --interval-features "$out/admitted.jsonl" > "$out/old.jsonl"
"$new_client" --discrete "$base" --tokenizer "$tokenizer" \
    --interval-features "$out/admitted.jsonl" > "$out/new-legacy.jsonl"
cmp "$out/old.jsonl" "$out/new-legacy.jsonl"
"$new_client" --discrete "$base" --tokenizer "$tokenizer" \
    --interval-relation-features "$out/admitted.jsonl" > "$out/new.jsonl"

old_sha=$(sha256sum "$out/old.jsonl" | cut -d' ' -f1)
new_sha=$(sha256sum "$out/new.jsonl" | cut -d' ' -f1)
"$probe" "$stage" "$blind_sha" "$truth_sha" "$groups_sha" "$base" "$tokenizer" "$tokenizer_sha" \
    "$out/old.jsonl" "$old_sha" "$out/new.jsonl" "$new_sha" > "$out/probe.json"
pin "$blind_sha" "$stage/blind.jsonl"
pin "$truth_sha" "$stage/truth.jsonl"
pin "$groups_sha" "$stage/groups.jsonl"
jq -n --arg blind_sha "$blind_sha" --arg truth_sha "$truth_sha" --arg groups_sha "$groups_sha" \
    --arg base_sha "$base_sha" --arg tokenizer_sha "$tokenizer_sha" \
    --arg old_client_sha "$old_client_sha" --arg new_client_sha "$new_client_sha" \
    --arg probe_sha "$(sha256sum "$probe" | cut -d' ' -f1)" \
    --arg source_sha "$(sha256sum "$source_dir/SOURCE.sha256" | cut -d' ' -f1)" \
    '{$blind_sha,$truth_sha,$groups_sha,$base_sha,$tokenizer_sha,$old_client_sha,$new_client_sha,$probe_sha,$source_sha}' \
    > "$out/inputs.json"
(
    cd "$out"
    sha256sum ./inputs.json ./admission.json ./admitted.jsonl ./old.jsonl ./new-legacy.jsonl \
        ./new.jsonl ./probe.json > PROBE.sha256
)
