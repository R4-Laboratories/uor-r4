#!/usr/bin/env bash
# Reopen frozen D11 artifacts on validation and protected prompts, compare
# every signed score to independent CUDA replay, and assemble gate evidence.
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 17 ]]; then
    echo 'usage: cold-evidence.sh TOOL FIT CLIENT TOKENIZER ROSTER ROSTER_SHA EVAL EVAL_SHA TRAIN_COLD TRAIN_COLD_SHA TRAIN_RETURN TRAIN_RETURN_SHA TRAIN_CAPSULE MATCHED EVAL_RETURN EVAL_RETURN_SHA NEW_OUT' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=../repair/custody.sh
source "$here/../repair/custody.sh"
tool=$(realpath "$1")
fit=$(realpath "$2")
client=$(realpath "$3")
tokenizer=$(realpath "$4")
roster=$(realpath "$5")
roster_pin=$6
evaldir=$(realpath "$7")
eval_pin=$8
train_cold=$(realpath "$9")
train_cold_pin=${10}
train_return=$(realpath "${11}")
train_pin=${12}
capsule=$(realpath "${13}")
matched=$(realpath "${14}")
scored=$(realpath "${15}")
scored_pin=${16}
out=$(realpath -m "${17}")
trainer=$here/../../codex-4340/fit

fit_sha=7cd8f52ad299622eef75579364ec5c31d9d98313051248df85565c74a2cc6eb5
client_sha=21461b5af9c82d98f6f0c7f8780ae6f738bd601a52fe8a98edf25a7e07548381
tokenizer_sha=a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
capsule_sha=09c93ef41bbe46385f40cb13aa1bf67906ed3b95494ad55ccc8f4545f0be48cf
matched_sha=ffd24cf73b009a3a38d70ea95b8e9f0db2b1dbbb19cdb1ee2062b7d0c24a8c7d
pin "$fit_sha" "$fit"
pin "$client_sha" "$client"
pin "$tokenizer_sha" "$tokenizer"
pin "$capsule_sha" "$capsule/INPUTS.sha256"
pin "$matched_sha" "$matched/SEALED.sha256"
pin "$roster_pin" "$roster/ROSTER.sha256"
pin "$eval_pin" "$evaldir/EVAL.sha256"
pin "$train_cold_pin" "$train_cold/OUTPUTS.sha256"
pin "$train_pin" "$train_return/OUTPUTS.sha256"
pin "$scored_pin" "$scored/OUTPUTS.sha256"
(cd "$capsule" && sha256 --quiet --strict -c INPUTS.sha256 && sha256 --quiet --strict -c TRAIN-INPUTS.sha256 && sha256 --quiet --strict -c EVAL-INPUTS.sha256)
(cd "$matched" && sha256 --quiet --strict -c SEALED.sha256)
(cd "$roster" && sha256 --quiet --strict -c ROSTER.sha256 && (cd blind && sha256 --quiet --strict -c BLIND.sha256))
(cd "$evaldir" && sha256 --quiet --strict -c EVAL.sha256)
(cd "$train_cold" && sha256 --quiet --strict -c OUTPUTS.sha256)
(cd "$scored" && sha256 --quiet --strict -c OUTPUTS.sha256)
"$trainer/check-return.sh" "$train_return" "$capsule_sha" >/dev/null
jq -e '.coverage == "pass"' "$roster/coverage.json" >/dev/null
digest() { sha256 < "$1" | cut -d' ' -f1; }
frozen=$(digest "$train_return/FROZEN-WEIGHTS.sha256")
test "$frozen" = "$(cat "$scored/frozen-source-pin")"
test "$frozen" = "$(cat "$train_cold/fit-id.txt")"
test "$eval_pin" = "$(cat "$scored/panel-pin")"
printf '%s  ROSTER.sha256\n' "$roster_pin" | cmp - "$evaldir/roster-pin.txt"
printf '%s  interval-fit\n' "$fit_sha" | cmp - "$evaldir/fit-pin.txt"

mkdir "$out"
cd "$out"
sha256 "$tool" "$fit" "$client" "$tokenizer" "$roster/ROSTER.sha256" "$evaldir/EVAL.sha256"     "$capsule/INPUTS.sha256" "$matched/SEALED.sha256" "$train_cold/OUTPUTS.sha256"     "$train_return/OUTPUTS.sha256" "$scored/OUTPUTS.sha256" > source-pins.sha256

for mix in original natural; do
    for arm in count geometry; do
        for kind in actual permuted zero; do
            name=$mix.$arm.$kind
            artifact=$train_cold/$name.d11
            packed=$train_cold/$name.pack.json
            words=$train_return/$name.words
            repeat=$train_return/repeat/$name.words
            fit_input=$capsule/$mix/$arm.train.tape
            primary_stats=$train_return/$name.stats.json
            repeat_stats=$train_return/repeat/$name.stats.json
            if [[ "$kind" == zero ]]; then
                words=$capsule/zero.words
                repeat=$words
                fit_input=$words
                primary_stats=$words
                repeat_stats=$words
            fi
            state_sha=$(digest "$words")
            artifact_sha=$(member_pin "$train_cold/OUTPUTS.sha256" "./$name.d11")
            jq -e --arg state "$state_sha" --arg artifact "$artifact_sha" --arg arm "$arm" '
                .artifact_sha256==$artifact and .schema=="uor-r4.lut-discrete-native-interval/1" and
                .interval.arm==$arm and .interval.state_sha256==$state and .interval.state_bytes==8192
            ' "$packed" >/dev/null

            for panel in validation protected; do
                b=$roster/blind/$panel
                "$client" --discrete "$artifact" --tokenizer "$tokenizer"                     --interval-features "$b/input.jsonl" > "$name.$panel.cold.jsonl"                     2> "$name.$panel.cold.stderr"
                "$fit" verify-device "$b/features.jsonl" "$name.$panel.cold.jsonl" "$arm"                     "$words" "$scored/$name.$panel.scores" > "$name.$panel.parity.json"
                "$tool" panel "$arm" "$b/input.jsonl" "$name.$panel.cold.jsonl"                     "$scored/$name.$panel.scores" > "$name.$panel.panel.json"
            done
            "$tool" panel "$arm" "$train_cold/$mix.blind.jsonl" "$train_cold/$name.cold.jsonl"                 "$train_return/$name.scores" > "$name.train.panel.json"
            "$client" --discrete "$artifact" --tokenizer "$tokenizer"                 --interval-features "$roster/blind/validation/swaps.jsonl" > "$name.swaps.cold.jsonl"                 2> "$name.swaps.cold.stderr"
            "$tool" swaps "$roster/swap-requests.jsonl" "$name.swaps.cold.jsonl" > "$name.swaps.json"
            sha256 "$train_cold/$name.parity.json" "$name.validation.parity.json"                 "$name.protected.parity.json" "$scored/$name.validation.scores"                 "$scored/$name.protected.scores" "$train_return/$name.scores"                 "$name.validation.cold.jsonl" "$name.protected.cold.jsonl"                 "$name.swaps.cold.jsonl" "$packed" > "$name.parity.sha256"
            jq -n --arg mixture "$mix" --arg key "$arm" --arg kind "$kind" --arg artifact "$artifact_sha"                 --arg state "$state_sha" --arg replay_state "$(digest "$repeat")"                 --arg verified_device "$(digest "$name.parity.sha256")" --arg fit_input "$(digest "$fit_input")"                 --arg fit_output "$(digest "$primary_stats")" --arg replay_output "$(digest "$repeat_stats")"                 --slurpfile validation "$name.validation.panel.json" --slurpfile train "$name.train.panel.json"                 --slurpfile protected "$name.protected.panel.json" --slurpfile swaps "$name.swaps.json" '
                {arm:{$mixture,$key,fit:$kind},artifact_sha256:$artifact,state_sha256:$state,
                 verified_device_sha256:$verified_device,fit_input_sha256:$fit_input,fit_replay_state_sha256:$replay_state,
                 fit_output_sha256:$fit_output,fit_replay_output_sha256:$replay_output,validation:$validation[0],
                 train:$train[0],protected:$protected[0],swaps:$swaps[0]}
            ' > "$name.evidence.json"
        done
    done
done
roster_json_pin=$(member_pin "$roster/ROSTER.sha256" roster.json)
jq -s --arg roster "$roster_json_pin" --arg client "$client_sha" --arg tokenizer "$tokenizer_sha"     --slurpfile source "$roster/roster.json" '
    {schema:"uor-r4.reciprocal-transfer-validation-evidence/1",roster_sha256:$roster,ordinary_client_sha256:$client,
     base_writer_sha256:"5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa",tokenizer_sha256:$tokenizer,
     protected_reference_sha256:$source[0].protected_reference_sha256,models:.}
' ./*.evidence.json > evidence.json
sha256 ./*.json ./*.jsonl ./*.stderr source-pins.sha256 ./*.parity.sha256 > EVIDENCE.sha256
sha256 --quiet --strict -c EVIDENCE.sha256
