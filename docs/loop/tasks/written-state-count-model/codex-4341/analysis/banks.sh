#!/usr/bin/env bash
# Descriptive readback restricted to the sealed natural geometry actual state.
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 2 ]]; then
    echo 'usage: banks.sh WICK_REPAIR_RUN_ROOT FRESH_OUT' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=../repair/custody.sh
source "$here/../repair/custody.sh"
root=$(realpath "$1")
out=$(realpath -m "$2")
train=$root/device-return-v2/job/measured
score=$root/score-return-v1/scored
train_cold=$root/cold-train-v1
cold=$root/cold-evidence-v1
margins=$root/margins-v1
roster=$root/scorer-build.RrPj7Zas/roster-v2/roster.json
arm=natural.geometry.actual
pin e15559f77bfda763197c132aaa3517a5a14c13581f48e46512c716a2cdca89af "$train/$arm.words"
pin 2544bee3b8a57d1cfc2b4466cbba4648a2fa22a07016ef4f29fa4f681393e967 "$train/OUTPUTS.sha256"
pin 77e609758aef4155c04c4de549fc5ab2e8866df4c720c2717763df896f4f0897 "$score/OUTPUTS.sha256"
pin 4cdefd4a56276fce28cf489ed4373f51f3d697c9846df45c4a4323400340c0a7 "$train_cold/OUTPUTS.sha256"
pin 759094b13da43cf9df88d368d298f6e9dc5f3e0fac4a93ef2bac3a95d630378c "$cold/EVIDENCE.sha256"
pin 21d0e453e1390e69cc65d099ed2bd2c6ad34584c2c7f36f6d198c6437472749f "$margins/ANALYSIS.sha256"
pin 407293ea282aeb1722263bc321b3edb699f9e262eed8f1a010094f0f0477a811 "$roster"
check_member() {
    local parent=$1 manifest=$2 file=$3 digest
    digest=$(awk -v name="$file" '$2 == name || $2 == "./"name {value=$1; n++}
        END {if(n!=1) exit 1; print value}' "$parent/$manifest")
    pin "$digest" "$parent/$file"
}
check_member "$train" OUTPUTS.sha256 "$arm.scores"
check_member "$score" OUTPUTS.sha256 "$arm.validation.scores"
check_member "$train_cold" OUTPUTS.sha256 "$arm.cold.jsonl"
check_member "$cold" EVIDENCE.sha256 "$arm.validation.cold.jsonl"
for panel in train validation; do
    check_member "$margins" ANALYSIS.sha256 "natural.geometry.$panel.json"
done
mkdir "$out"
cd "$out"
to_json_words() {
    local width=$1 input=$2
    od -An -v -td"$width" --endian=little "$input" |
        jq -Rs 'split("\n")|map(split(" ")[])|map(select(length>0)|tonumber)'
}
to_json_words 2 "$train/$arm.words" > weights.json
for panel in train validation; do
    if [[ "$panel" == train ]]; then
        raw=$train/$arm.scores
        trace=$train_cold/$arm.cold.jsonl
    else
        raw=$score/$arm.validation.scores
        trace=$cold/$arm.validation.cold.jsonl
    fi
    to_json_words 4 "$raw" > "$panel.device.json"
    jq -n --arg panel "$panel" --slurpfile weights weights.json --slurpfile cold "$trace"         --slurpfile device "$panel.device.json" --slurpfile roster "$roster"         --slurpfile margins "$margins/natural.geometry.$panel.json" -f "$here/banks.jq" > "$panel.json"
done
jq -s '{scope:"Descriptive frozen natural geometry actual score reconstruction",source_claim:"No new fit or generalization result",
        panels:map(del(.rows))}' train.json validation.json > summary.json
sha256 "$here/banks.jq" "$here/banks.sh" "$roster" "$train/$arm.words"     "$margins/ANALYSIS.sha256" "$train/OUTPUTS.sha256" "$score/OUTPUTS.sha256"     "$train_cold/OUTPUTS.sha256" "$cold/EVIDENCE.sha256" > INPUTS.sha256
sha256 INPUTS.sha256 weights.json ./*.device.json train.json validation.json summary.json > BANKS.sha256
