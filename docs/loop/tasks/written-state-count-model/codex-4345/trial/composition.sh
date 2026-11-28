#!/usr/bin/env bash
# Read only invented prompts, fabricated tables and synthetic source control.
set -euo pipefail
umask 077
r=$(realpath "${1:?private root}")
out=${2:?fresh output}
old=$r/client/bin-final/uor-chat
new=$r/client/bin-composition/uor-chat
tok=$(dirname "$r")/20261127-codex-visible-answer/panel/inputs/tokenizer.json
base=$(dirname "$r")/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
mkdir "$out"
cd "$out"
jq -c '{prompt_ids,start:4,end:5,ids:.prompt_ids[4:5]}' "$r/client/finite-final/base.json" > controlled.json
artifact=$r/client/finite-final/positive.geometry.d11
flags=(--discrete "$artifact" --protocol 2 --tokenizer "$tok" --say 'Which bird did Mira describe?' --max-tokens 4)
for v in old new; do
    "${!v}" "${flags[@]}" --ordinary-termination verified-end --interval-supplied-source controlled.json > "$v.json"
    "${!v}" "${flags[@]}" --ordinary-termination off --interval-supplied-source controlled.json > "$v.off.json"
    "${!v}" --discrete "$base" --ordinary-termination-capture "$r/client/finite-final/blind.jsonl" > "$v.keys.jsonl"
done
cmp old.keys.jsonl new.keys.jsonl
cmp old.off.json new.off.json
jq -e --slurpfile old old.json '
    .generated_ids==$old[0].generated_ids and .stop=="eos" and
    (.native_interval.steps | any(.event=="copy" and .committed_id!=1 and (has("final_committed_id")|not))) and
    (.native_interval.steps | any(.event=="verified_end" and .committed_id!=1 and .final_committed_id==1 and .commit_phase=="before_ordinary_termination"))
' new.json >/dev/null
sha256sum "$old" "$new" controlled.json *.jsonl old.json new.json old.off.json new.off.json > VERIFIED.sha256
jq -c '{generated_ids,stop,steps:[.native_interval.steps[]|{event,committed_id,final_committed_id,commit_phase}]}' new.json
