#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 5 ]; then
    echo "usage: capture.sh WICK_RUNS COLD_4333 COLD_4330 D26_ARTIFACT NEW_OUTPUT" >&2
    exit 2
fi
runs=$(realpath "$1")
signed=$(realpath "$2")
core=$(realpath "$3")
contextual=$(realpath "$4")
out=$(realpath -m "$5")
balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
frozen="$core/reconstructed/frozen.d11"
panel="$signed/payload/panel-v1"
chat="$core/tools/uor-chat"
tokenizer="$core/tools/tokenizer.json"

sha256sum --strict --quiet -c <<SUMS
641e9ed7c21ed0d7f12da5601615bc0f38dc07bbec859bd706462d38cc5ed693  $signed/payload/MEMBERS.sha256
d565eb65bb16c178a7f5eb96770b9595d1790e03cc3cfe80efa81c81afea9305  $core/SHA256SUMS
87cfb3ca166a313c983a9e217ae3474ad92ce433aae4daf4fe2006fbd1d305d3  $chat
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa  $balanced
6415f6cb9fbf4694b449c08170d17c7e821c168d28e811cfa4e7a3448a333422  $frozen
b17352679c64a96bf657fad7640d4bb05ea3de0ecd08dadd13e9d2756e979bc4  $contextual
SUMS
(cd "$signed/payload" && awk '$2 ~ /^\.\/panel-v1\/blind\//' MEMBERS.sha256 | sha256sum --strict --quiet -c)
prompts="$panel/blind/prompts.jsonl"
jq -es '
    length == 18 and (map(.case_id) | unique | length) == 18
    and all(.[]; (.case_id | test("^[a-f0-9]{64}$")) and
        (.text | type=="string" and length>0 and (contains("\u0000") | not)))
' "$prompts" >/dev/null

mkdir "$out"
for arm in balanced frozen contextual; do
    mkdir "$out/$arm"
    case "$arm" in
        balanced) artifact="$balanced"; options=(--authenticated-copy --native-occurrence-trace --native-period-release) ;;
        frozen) artifact="$frozen"; options=(--authenticated-copy --native-occurrence-trace --native-period-release) ;;
        contextual) artifact="$contextual"; options=(--read-mode enabled --contextual-source-trace) ;;
    esac
    jq -j '.case_id, "\u0000", .text, "\u0000"' "$prompts" |
        xargs -0 -r -n 2 -P 2 bash -ceu '
            set -o pipefail
            chat=$1; tokenizer=$2; artifact=$3; dest=$4
            shift 4
            option_count=$#
            case_id=${@: -2:1}
            text=${@: -1}
            "$chat" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
                --say "$text" --temperature 0 --seed 0 --max-tokens 16 "${@:1:option_count-2}" |
                jq -ceS "del(.artifact)" > "$dest/$case_id.json"
        ' ordinary "$chat" "$tokenizer" "$artifact" "$out/$arm" "${options[@]}"
    while IFS= read -r case_id; do
        jq -ceS --arg case_id "$case_id" --arg arm "$arm" \
            '{case_id:$case_id, arm:$arm, client:.}' "$out/$arm/$case_id.json"
    done < <(jq -er .case_id "$prompts") > "$out/$arm.jsonl"
done
(cd "$out" && sha256sum balanced.jsonl frozen.jsonl contextual.jsonl > SHA256SUMS)
