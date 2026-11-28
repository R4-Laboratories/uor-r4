#!/usr/bin/env bash
set -euo pipefail
umask 077
if [ "$#" -ne 6 ]; then
    echo "usage: run.sh TOOL_DIRECTORY PRIOR_4333_RESTORE PRIOR_4332_RESTORE FEATURE_DIRECTORY CUT NEW_RESULT_DIRECTORY" >&2
    exit 2
fi
unit=$(cd "$(dirname "$0")" && pwd)
tools=$(realpath "$1")
prior=$(realpath "$2")
older=$(realpath "$3")
features=$(realpath "$4")
cut=$5
case "$cut" in
    zero|separator|tail32) ;;
    *) echo "unregistered cut: $cut" >&2; exit 2 ;;
esac
out=$(realpath -m "$6")
mkdir "$out"
cd "$out"
mkdir scratch
export LD_LIBRARY_PATH="$tools/environment/usr/lib/x86_64-linux-gnu"
export SIGNED_GATE_HIGHS="$tools/environment/usr/bin/highs"
export SIGNED_GATE_SCRATCH="$out/scratch"
z3="$tools/environment/usr/bin/z3"
solver="$tools/target/release/signed-site-gate"
verify="$tools/target/release/verify-signed-verdicts"
earlier="$features/diagnostic-$cut.jsonl"
fresh="$features/fresh-$cut.jsonl"
earlier_key="$older/panel/truth/key.jsonl"
fresh_key="$prior/panel/truth/key.jsonl"
original="$prior/transport/fresh-signed.jsonl"
certificate="$unit/../../codex-4333/evidence/source-ranking.json"

cmp "$features/diagnostic-zero.jsonl" "$prior/transport/diagnostic-signed.jsonl"
cmp "$features/fresh-zero.jsonl" "$original"
for panel in diagnostic fresh; do
    cmp <(jq -cS 'del(.candidates[] | .real_signed,.fixed_signed,.swapped_signed,.no_question_signed)' "$features/$panel-zero.jsonl") \
        <(jq -cS 'del(.candidates[] | .real_signed,.fixed_signed,.swapped_signed,.no_question_signed)' "$features/$panel-$cut.jsonl")
done
input_hash() {
    local hash
    hash=$(sha256sum < "$1")
    printf '%s  %s\n' "${hash%% *}" "$2"
}
{
    input_hash "$earlier" earlier-features
    input_hash "$earlier_key" earlier-key
    input_hash "$fresh" fresh-features
    input_hash "$fresh_key" fresh-key
    input_hash "$certificate" prior-source-ranking
} > inputs.sha256
"$verify" --prior-certificate "$original" "$fresh" "$fresh_key" "$certificate" > prior-certificate.json
"$solver" --rank "$z3" "$fresh" "$fresh_key" > rank.jsonl
"$verify" "$fresh" "$fresh_key" rank.jsonl --rank > rank-check.json
jq -es --slurpfile capture "$fresh" -f "$unit/proofs.jq" rank.jsonl > rank-proofs.json

cat "$earlier" "$fresh" > joint-features.jsonl
cat "$earlier_key" "$fresh_key" > joint-key.jsonl
"$solver" "$z3" "$earlier" "$earlier_key" > old.jsonl
"$verify" "$earlier" "$earlier_key" old.jsonl > old-check.json
"$solver" "$z3" "$fresh" "$fresh_key" > fresh.jsonl
"$verify" "$fresh" "$fresh_key" fresh.jsonl > fresh-check.json
"$solver" "$z3" "$earlier" "$earlier_key" "$fresh" "$fresh_key" > joint.jsonl
"$verify" joint-features.jsonl joint-key.jsonl joint.jsonl > joint-check.json

sha256sum inputs.sha256 prior-certificate.json rank-proofs.json rank{.jsonl,-check.json} old{.jsonl,-check.json} \
    fresh{.jsonl,-check.json} joint{.jsonl,-check.json} > SHA256SUMS
