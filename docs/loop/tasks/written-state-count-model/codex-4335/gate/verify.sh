#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 6 ]]; then
    echo "usage: verify.sh RELATIVE_4334 SIGNED_4333 REQUEST_4332 FEATURES ARCHIVED_RESULTS NEW_VERIFY" >&2
    exit 2
fi
unit=$(cd "$(dirname "$0")" && pwd)
relative=$(realpath "$1")
signed=$(realpath "$2")
request=$(realpath "$3")
features=$(realpath "$4")
results=$(realpath "$5")
out=$(realpath -m "$6")
parent_source="$relative/payload/source/codex-4334"
prior_gate="$relative/payload/gate/tail32"
prior_verify="$relative/tools/target/release/verify-signed-verdicts"
old_key="$request/panel/truth/key.jsonl"
fresh_key="$signed/panel/truth/key.jsonl"
mkdir "$out"
scratch="$out/scratch"
mkdir "$scratch"

(cd "$relative/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
(cd "$features" && sha256sum --strict --quiet -c SCAN.sha256)
(cd "$results" && sha256sum --strict --quiet -c SHA256SUMS)
bash "$unit/build.sh" "$relative" "$signed" "$out/tools" check
cat "$old_key" "$fresh_key" > "$scratch/joint-key.jsonl"

for panel in diagnostic fresh; do
    reference="$relative/transport/$panel-tail32.jsonl"
    jq -cS --arg kind features -f "$unit/rotate.jq" "$reference" > "$scratch/rotated.jsonl"
    jq -cS . "$features/$panel-clamp.jsonl" > "$scratch/clamp.jsonl"
    cmp "$scratch/rotated.jsonl" "$scratch/clamp.jsonl"
    for frame in tail32 edge; do
        tape="$features/$panel-$frame.jsonl"
        if [[ "$frame" == tail32 ]]; then tape="$reference"; fi
        jq -cS 'del(.candidates[] | .real_signed,.fixed_signed,.swapped_signed,.no_question_signed)' \
            "$tape" > "$scratch/identity-$frame.jsonl"
    done
    cmp "$scratch/identity-tail32.jsonl" "$scratch/identity-edge.jsonl"
    if cmp -s "$features/$panel-clamp.jsonl" "$features/$panel-edge.jsonl"; then
        echo "edge and clamp feature tapes are identical" >&2
        exit 1
    fi
done

for frame in edge clamp; do
    verify="$prior_verify"
    if [[ "$frame" == edge ]]; then
        verify="$out/tools/target/release/verify-signed-verdicts"
    fi
    dest="$out/$frame"
    mkdir "$dest"
    earlier="$features/diagnostic-$frame.jsonl"
    fresh="$features/fresh-$frame.jsonl"
    cat "$earlier" "$fresh" > "$scratch/joint-features.jsonl"
    for mask in word passage passage-word supplied; do
        jq -c --arg mask "$mask" --slurpfile truth "$fresh_key" \
            -f "$parent_source/rivals/select.jq" "$fresh" > "$scratch/$mask.jsonl"
    done
    for design in rank old fresh joint word passage passage-word supplied; do
        verdicts="$results/$frame/$design.jsonl"
        options=()
        reference="$prior_gate/$design.jsonl"
        reference_check="$prior_gate/$design-check.json"
        case "$design" in
            rank) tape="$fresh"; key="$fresh_key"; options=(--rank) ;;
            old) tape="$earlier"; key="$old_key" ;;
            fresh) tape="$fresh"; key="$fresh_key" ;;
            joint) tape="$scratch/joint-features.jsonl"; key="$scratch/joint-key.jsonl" ;;
            *)
                tape="$scratch/$design.jsonl"
                key="$fresh_key"
                options=(--rank)
                reference="$relative/payload/rivals/tail32/$design/verdicts.jsonl"
                reference_check="$relative/payload/rivals/tail32/$design/verified.json"
                ;;
        esac
        "$verify" "$tape" "$key" "$verdicts" "${options[@]}" > "$dest/$design-check.json"
        jq -e '(.validated.unknown // 0) == 0' "$dest/$design-check.json" > /dev/null
        if [[ "$frame" == clamp ]]; then
            jq -c --arg kind verdicts -f "$unit/rotate.jq" "$reference" > "$scratch/rotated-verdicts.jsonl"
            "$verify" "$tape" "$key" "$scratch/rotated-verdicts.jsonl" "${options[@]}" \
                > "$dest/$design-tail32-substitution.json"
            cmp "$reference_check" "$dest/$design-tail32-substitution.json"
            for origin in reference verdicts; do
                path="$reference"
                if [[ "$origin" == verdicts ]]; then path="$verdicts"; fi
                jq -cS '.verdict |= {status} | del(.actual_witness_pass)' \
                    "$path" > "$scratch/status-$origin.jsonl"
            done
            cmp "$scratch/status-reference.jsonl" "$scratch/status-verdicts.jsonl"
        fi
    done
    jq -es --slurpfile capture "$fresh" -f "$parent_source/gate/proofs.jq" \
        "$results/$frame/rank.jsonl" > "$dest/rank-proofs.json"
    jq -nc --slurpfile earlier "$earlier" --slurpfile recent "$fresh" \
        --slurpfile k0 "$old_key" --slurpfile k1 "$fresh_key" \
        -f "$parent_source/rivals/tuple-overlap.jq" > "$dest/tuple-overlap.jsonl"
    for mask in rank passage-word supplied; do
        tape="$scratch/$mask.jsonl"
        if [[ "$mask" == rank ]]; then tape="$fresh"; fi
        jq -ce -f "$unit/proof-support.jq" "$results/$frame/$mask.jsonl" \
            > "$dest/$mask-proof-support.json"
        jq -nec --slurpfile rows "$tape" --slurpfile keys "$fresh_key" \
            -f "$unit/short-certificate.jq" > "$dest/$mask-short-certificate.json"
        if jq -e '.proof != null' "$dest/$mask-short-certificate.json" > /dev/null; then
            jq -ce --slurpfile short "$dest/$mask-short-certificate.json" '
                if .level=="panel_positive_only" and .scope=="supported_gold_vs_sites"
                   and .arm=="actual" and .offsets=="zero" and .margin==1
                then .verdict = $short[0].proof else . end
            ' "$results/$frame/$mask.jsonl" > "$scratch/short-verdicts.jsonl"
            "$verify" "$tape" "$fresh_key" "$scratch/short-verdicts.jsonl" --rank \
                > "$dest/$mask-short-check.json"
        fi
    done
done

jq -ces '
    map(select(.level=="row"))
    | group_by([.id,.offsets,.scope,.margin])
    | map(
        map(select(.arm=="actual" or .arm=="fixed")) as $both
        | select(($both | map(.verdict.status) | unique | length) == 2)
        | $both
    )
' "$results/edge/old.jsonl" > "$out/edge-single-row-counterexamples.json"
if "$prior_verify" "$features/diagnostic-edge.jsonl" "$old_key" "$results/edge/old.jsonl" \
    > "$scratch/prior-invariant.stdout" 2> "$out/prior-invariant.stderr"; then
    echo "expected a counterexample to the additive single-row gauge invariant" >&2
    exit 1
fi
case $(<"$out/prior-invariant.stderr") in
    *"signed gauge single-row invariant failed"*) ;;
    *) cat "$out/prior-invariant.stderr" >&2; exit 1 ;;
esac

# Both predecessors use the same 13 supported addresses and the same margin.
jq -ce --slurpfile keys "$fresh_key" '
    select(.level=="panel_positive_only" and .scope=="supported_gold_vs_sites"
           and .arm=="actual" and .offsets=="zero" and .margin==1)
    | .case_ids = [$keys[] | select(.target != null) | .case_id]
' "$prior_gate/rank.jsonl" > "$scratch/tail32-source-certificate.json"
"$prior_verify" --prior-certificate "$signed/transport/fresh-signed.jsonl" "$features/fresh-edge.jsonl" \
    "$fresh_key" "$relative/payload/source/codex-4333/evidence/source-ranking.json" \
    > "$out/absolute-proof-in-edge.json"
"$prior_verify" --prior-certificate "$relative/transport/fresh-tail32.jsonl" "$features/fresh-edge.jsonl" \
    "$fresh_key" "$scratch/tail32-source-certificate.json" > "$out/tail32-proof-in-edge.json"

# A coordinate substitution must change the proof, not only its input name.
jq -c --arg kind verdicts -f "$unit/rotate.jq" "$prior_gate/rank.jsonl" > "$scratch/rotated-rank.jsonl"
if "$prior_verify" "$features/fresh-edge.jsonl" "$fresh_key" "$scratch/rotated-rank.jsonl" --rank \
    > "$scratch/wrong-frame.stdout" 2> "$out/wrong-frame.stderr"; then
    echo "the independent checker accepted the clamped certificate stream on edge features" >&2
    exit 1
fi
case $(<"$out/wrong-frame.stderr") in
    *"certificate differs from independent coefficients or offsets"*) ;;
    *) cat "$out/wrong-frame.stderr" >&2; exit 1 ;;
esac

(cd "$out" && sha256sum *.json *.stderr {edge,clamp}/*.json {edge,clamp}/*.jsonl > SHA256SUMS)
