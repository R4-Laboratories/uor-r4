#!/usr/bin/env bash
# Fabricated integer tables and prompts exercise the actual ordinary artifact loader.
set -euo pipefail
umask 077
r=${1:?private run root}
bin=${2:?new binaries directory}
out=${3:?fresh output}
parent=$(dirname "$r")
base=$parent/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
tok=$parent/20261127-codex-visible-answer/panel/inputs/tokenizer.json
base_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
mkdir "$out"
cd "$out"
dd if=/dev/zero of=zero.words bs=8192 count=1 status=none
# LE 0x0101 = +257, all 4,096 signed words.
LC_ALL=C tr '\000' '\001' <zero.words >positive.words
fit=$(sha256sum zero.words | cut -d' ' -f1)
for item in zero.geometry positive.geometry positive.ordered; do
    "$bin/pack-native-interval" ordinary-termination "$base" "$base_sha" "${item#*.}" \
        "${item%%.*}.words" "$item.d11" "$fit" >"$item.pack.json"
done
say() {
    local executable=$1 artifact=$2
    shift 2
    "$executable" --discrete "$artifact" --protocol 2 --tokenizer "$tok" \
        --say 'Which bird did Mira describe?' --max-tokens 4 "$@"
}
say "$r/client/base-chat" "$base" >legacy.json
say "$bin/uor-chat" "$base" --ordinary-termination-trace >base.json
for item in zero.geometry positive.geometry positive.ordered; do
    for mode in off generic verified-end; do
        say "$bin/uor-chat" "$item.d11" --ordinary-termination "$mode" --ordinary-termination-trace >"$item.$mode.json"
    done
done
for item in base zero.geometry.off zero.geometry.generic zero.geometry.verified-end \
    positive.geometry.off positive.geometry.verified-end positive.ordered.off positive.ordered.verified-end; do
    jq -e --slurpfile other legacy.json '.prompt_ids==$other[0].prompt_ids and .generated_ids==$other[0].generated_ids' "$item.json" >/dev/null
done
for item in positive.geometry positive.ordered; do
    jq -e '.generated_ids==[1] and .stop=="eos" and .ordinary_termination.steps[0].score==4112
       and .ordinary_termination.steps[0].ordinary_id!=1 and .ordinary_termination.steps[0].promoted' "$item.generic.json" >/dev/null
done
jq -c '{prefix:.prompt_ids}' base.json >blind.jsonl
"$bin/uor-chat" --discrete "$base" --ordinary-termination-capture blind.jsonl >keys.jsonl
jq -e --slurpfile cap keys.jsonl '.ordinary_termination.steps[0] as $s | $s.geometry_keys==$cap[0].geometry_keys
    and $s.ordered_keys==$cap[0].ordered_keys and $s.ordinary_id==$cap[0].ordinary_id
    and $s.writer.read_code==$cap[0].read_code and $s.writer.recurrence_code==$cap[0].recurrence_code' base.json >/dev/null
jq -c '.+{target:1}' blind.jsonl >label.jsonl
if "$bin/uor-chat" --discrete "$base" --ordinary-termination-capture label.jsonl >bad.stdout 2>bad.stderr; then exit 1; fi
test ! -s bad.stdout
grep -q 'unknown field' bad.stderr
sha256sum *.d11 *.json *.jsonl *.words >FILES.sha256
