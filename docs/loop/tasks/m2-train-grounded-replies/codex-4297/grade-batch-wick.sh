#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?grade or one}; batch=$(realpath "${2:?batch}"); arm=${3:?loss arm}
[[ $batch == "$s"/measurement-v*/train/*/round-[12] ]]
raw="$batch/device"; g="$batch/grade"
case $mode in
one)
    id=${4:?id}; if [[ $id == base ]]; then number=null; else [[ $id =~ ^[0-9]+$ ]]; number=$id; fi
    join_rows mixed "$train" learned "$raw/$id.reader.jsonl" | pair_rows mixed |
        jq -ce --slurpfile initial "$old_pairs" --slurpfile pin "$reference" \
        --slurpfile score <(jq -ce --argjson id "$number" 'select(.id==$id)' "$raw/scores.jsonl") \
        -f "$here/grade/candidate.jq" > "$g/records/$id.json"
    ;;
grade)
    test -f "$raw/complete.json"; mkdir "$g" "$g/records"
    { echo base; seq 0 67; } | xargs -P 4 -n 1 bash "$0" one "$batch" "$arm"
    { cat "$g/records/base.json"; for id in $(seq 0 67); do cat "$g/records/$id.json"; done; } > "$g/candidates.jsonl"
    jq -es --arg arm "$arm" --slurpfile menu "$raw/menu.jsonl" -f "$here/grade/elect.jq" "$g/candidates.jsonl" > "$g/election.json"
    seal "$g"
    ;;
*) exit 64 ;;
esac
