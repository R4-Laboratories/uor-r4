#!/usr/bin/env bash
# Wick. Reopen the complete D11 artifacts on the exact matched blind TRAIN inputs.
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 12 ]]; then
    echo 'usage: cold-train.sh RETURN INPUT_CAPSULE INPUT_PIN MATCHED FIT CLIENT PACK BASE TOKENIZER ORIGINAL_BLIND NATURAL_BLIND FRESH_OUTPUT' >&2
    exit 2
fi
returned=$(realpath "$1")
capsule=$(realpath "$2")
pin=$3
matched=$(realpath "$4")
fit=$(realpath "$5")
client=$(realpath "$6")
pack=$(realpath "$7")
base=$(realpath "$8")
tokenizer=$(realpath "$9")
original_blind=$(realpath "${10}")
natural_blind=$(realpath "${11}")
out=$(realpath -m "${12}")
here=$(cd "$(dirname "$0")" && pwd)

verify_pin() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --quiet --strict -c -
}
verify_pin "$pin" "$capsule/INPUTS.sha256"
(cd "$capsule" && sha256sum --quiet --strict -c INPUTS.sha256 && sha256sum --quiet --strict -c EVAL-INPUTS.sha256)
verify_pin ffd24cf73b009a3a38d70ea95b8e9f0db2b1dbbb19cdb1ee2062b7d0c24a8c7d "$matched/SEALED.sha256"
(cd "$matched" && sha256sum --quiet --strict -c SEALED.sha256)
verify_pin 7cd8f52ad299622eef75579364ec5c31d9d98313051248df85565c74a2cc6eb5 "$fit"
verify_pin 21461b5af9c82d98f6f0c7f8780ae6f738bd601a52fe8a98edf25a7e07548381 "$client"
verify_pin 1ddddd477cddc06b848e2add89289bf7d0f8da1d4ab6c6b9c2bf45fbb431f13d "$pack"
verify_pin 5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$base"
verify_pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer"
verify_pin 3ce8bb80a99496f1a7cac3b7df39568ecdce99831046f9f7925a732c4bc7db84 "$original_blind"
verify_pin a9dd7b4165ce503f60eb805a87e4d99a8a630306b82e6cd4aacdccadaabf16a1 "$natural_blind"
mkdir "$out"
cd "$out"
"$here/check-return.sh" "$returned" "$pin" > device-summary.jsonl
fit_id=$(sha256sum "$returned/FROZEN-WEIGHTS.sha256" | cut -d' ' -f1)
printf '%s\n' "$fit_id" > fit-id.txt
sha256sum "$fit" "$client" "$pack" "$base" "$tokenizer" "$original_blind" "$natural_blind" \
    "$matched/SEALED.sha256" "$returned/OUTPUTS.sha256" "$capsule/INPUTS.sha256" > source.sha256

for mix in original natural; do
    case "$mix" in
        original) blind=$original_blind; candidates=25141 ;;
        natural) blind=$natural_blind; candidates=42776 ;;
    esac
    jq -cs --rawfile order "$matched/$mix/case-ids" '
        INDEX(.case_id) as $by | $order|split("\n")[]|select(length>0) | $by[.] // error("missing opening")
    ' "$blind" > "$mix.blind.jsonl"
    jq -r '.case_id' "$mix.blind.jsonl" | diff -u "$matched/$mix/case-ids" -
    for arm in geometry count; do
        for control in actual permuted zero; do
            name="$mix.$arm.$control"
            words="$returned/$name.words"
            if [[ "$control" == zero ]]; then
                words="$capsule/zero.words"
            fi
            "$pack" "$arm" "$base" "$words" "$name.d11" "$fit_id" > "$name.pack.json"
            "$client" --discrete "$name.d11" --tokenizer "$tokenizer" --interval-features "$mix.blind.jsonl" \
                > "$name.cold.jsonl" 2> "$name.cold.stderr"
            "$fit" verify-device "$matched/$mix/train.features.jsonl" "$name.cold.jsonl" "$arm" \
                "$words" "$returned/$name.scores" > "$name.parity.json"
            jq -e --argjson candidates "$candidates" '.cases==264 and .exact_scores==$candidates and .banks==16 and .payload_bytes==8192' \
                "$name.parity.json" >/dev/null
            "$fit" replay-device "$matched/$mix/train.features.jsonl" "$matched/$mix/train-gold.jsonl" "$arm" \
                "$words" > "$name.selection.json"
        done
    done
done
jq -es 'length==12 and (map(.artifact_bytes) | unique)==[1202185782]' ./*.pack.json > equal-artifact-bytes.json

for mix in original natural; do
    for arm in geometry count; do
        for control in actual permuted zero; do
            name="$mix.$arm.$control"
            jq -c --arg mix "$mix" --arg arm "$arm" --arg control "$control" \
               --slurpfile parity "$name.parity.json" --slurpfile pack "$name.pack.json" '
                {mixture:$mix, arm:$arm, control:$control, cases:(.cases|length),
                 exact, positive, positive_selected, no_read, false_reads,
                 paired_rows: .paired, paired_changed_rows: .paired_changes,
                 both_matched_gold_selected_pairs: (([.cases[]|select(.both_matched_gold_selected==true)]|length)/2),
                 strictly_positive, minimum_margin, exact_scores:$parity[0].exact_scores,
                 artifact_bytes:$pack[0].artifact_bytes, artifact_sha256:$pack[0].artifact_sha256}
            ' "$name.selection.json"
        done
    done
done > train-selection-summary.jsonl
sha256sum ./*.d11 ./*.json ./*.jsonl ./*.stderr fit-id.txt source.sha256 > OUTPUTS.sha256
