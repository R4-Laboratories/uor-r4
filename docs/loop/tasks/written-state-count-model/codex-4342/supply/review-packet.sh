#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# == 5 ]] || { echo 'usage: review-packet.sh INVENTORY INVENTORY_SEAL_SHA256 SPLIT REVIEWER FRESH_OUTPUT' >&2; exit 2; }
inventory=$(realpath "$1")
inventory_sha=$2
split=$3
reviewer=$4
out=$(realpath -m "$5")
[[ $split == screen || $split == construction ]]
[[ $reviewer =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
printf '%s  %s\n' "$inventory_sha" "$inventory/INVENTORY.sha256" | sha256sum --quiet --strict -c -
(cd "$inventory" && sha256sum --quiet --strict -c INVENTORY.sha256)
mkdir "$out"
jq -c --arg split "$split" 'select(.split==$split)|
  {group_id,split,stratum,article,paragraph,passage_sha256,context,
   families:[.families[]|map({case_id,question,original_char,original_bytes,span_text})]}' \
  "$inventory/review.jsonl" > "$out/rows.jsonl"
jq -n --arg reviewer "$reviewer" --arg split "$split" --arg inventory_sha256 "$inventory_sha" \
   --arg rows_sha256 "$(sha256sum "$out/rows.jsonl" | cut -d ' ' -f 1)" \
   '{schema:"uor-r4/directed-relation-review-request/1",$reviewer,$split,$inventory_sha256,$rows_sha256}' > "$out/request.json"
(cd "$out" && sha256sum rows.jsonl request.json > PACKET.sha256)
