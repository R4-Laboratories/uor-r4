#!/usr/bin/env bash
# Reopen the balanced artifact and recheck all labels, candidate keys and comparisons.
set -euo pipefail
[[ $# == 4 ]] || { echo 'usage: regrade.sh VERIFIED_RESTORE BALANCED_D11 TOKENIZER FRESH_OUTPUT' >&2; exit 2; }
restored=$(realpath "$1")
base=$(realpath "$2")
tokenizer=$(realpath "$3")
out=$(realpath -m "$4")
mkdir "$out"
cd "$restored"
sha256sum --quiet --strict -c MEMBERS.sha256
input=probe/inputs.json
get() { jq -er --arg key "$1" '.[$key]' "$input"; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin "$(get probe_sha)" tools/structural-probe
pin "$(get base_sha)" "$base"
pin "$(get tokenizer_sha)" "$tokenizer"
tools/structural-probe preflight supply/blind.jsonl "$(get blind_sha)" "$base" "$tokenizer" "$(get tokenizer_sha)" > "$out/admission.json"
cmp "$out/admission.json" probe/admission.json
tools/structural-probe supply "$(get blind_sha)" "$(get truth_sha)" "$(get groups_sha)" \
    "$base" "$tokenizer" "$(get tokenizer_sha)" \
    probe/old.jsonl "$(sha256sum probe/old.jsonl | cut -d' ' -f1)" \
    probe/new.jsonl "$(sha256sum probe/new.jsonl | cut -d' ' -f1)" > "$out/probe.json"
cmp "$out/probe.json" probe/probe.json
