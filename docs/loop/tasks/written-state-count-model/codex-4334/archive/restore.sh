#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 6 ]; then
    echo "usage: restore.sh WICK_RUNS NEW_RESTORE PACKET EXPECTED_SHA256 SIGNED_4333_RESTORE REQUEST_4332_RESTORE" >&2
    exit 2
fi
runs=$(realpath "$1")
restore=$(realpath -m "$2")
packet=$(realpath "$3")
signed=$(realpath "$5")
request=$(realpath "$6")
sha256sum --strict -c <<SUMS
$4  $packet
641e9ed7c21ed0d7f12da5601615bc0f38dc07bbec859bd706462d38cc5ed693  $signed/payload/MEMBERS.sha256
b5b192da4e77b12b78327c95176400107c8f361c3c66562b6521e425896f8121  $request/payload/MEMBERS.sha256
SUMS
(cd "$signed/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
(cd "$request/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
(cd "$runs" && sha256sum --strict --quiet -c "$request/payload/WICK-ARTIFACTS.sha256")
diff -qr "$signed/payload/panel-v1" "$signed/panel"
for tape in blind truth; do
    diff -qr "$request/payload/panel-v2/$tape" "$request/panel/$tape"
done
cmp "$request/payload/gate-v1/capture.jsonl" "$request/capture.jsonl"
(cd "$signed/transport" && sha256sum --strict --quiet -c "$signed/payload/transport/SHA256SUMS")

mkdir "$restore"
mkdir "$restore/payload"
payload="$restore/payload"
tar -I zstd -xf "$packet" -C "$payload"
(cd "$payload" && sha256sum --strict --quiet -c MEMBERS.sha256)

source="$payload/source/codex-4334"
bash "$source/transport/run.sh" "$runs" "$signed" "$request" "$restore/transport"
cmp "$payload/transport/SCAN.sha256" "$restore/transport/SCAN.sha256"
(cd "$restore/transport" && sha256sum --strict --quiet -c "$payload/transport/SCAN.sha256")
cmp "$payload/transport/operator-check.json" "$restore/transport/operator-check.json"
for panel in diagnostic fresh; do
    for kind in cuts separator tail32; do
        name="$panel-$kind.jsonl"
        cmp "$payload/transport/$name" "$restore/transport/$name"
    done
    jq -n --slurpfile cuts "$restore/transport/$panel-cuts.jsonl" \
        --slurpfile zero "$restore/transport/$panel-zero.jsonl" \
        --slurpfile separator "$restore/transport/$panel-separator.jsonl" \
        --slurpfile tail32 "$restore/transport/$panel-tail32.jsonl" \
        -f "$source/transport/summary.jq" > "$restore/transport/$panel-summary.json"
    cmp "$payload/transport/$panel-summary.json" "$restore/transport/$panel-summary.json"
done
bash "$source/gate/setup.sh" "$signed" "$restore/tools"
verify="$restore/tools/target/release/verify-signed-verdicts"
certificate="$payload/source/codex-4333/evidence/source-ranking.json"
earlier_key="$request/panel/truth/key.jsonl"
fresh_key="$signed/panel/truth/key.jsonl"

mkdir "$restore/verified"
cat "$earlier_key" "$fresh_key" > "$restore/verified/joint-key.jsonl"
for cut in zero separator tail32; do
    original="$payload/gate/$cut"
    checked="$restore/verified/$cut"
    mkdir "$checked"
    (cd "$original" && sha256sum --strict --quiet -c SHA256SUMS)
    earlier="$restore/transport/diagnostic-$cut.jsonl"
    fresh="$restore/transport/fresh-$cut.jsonl"
    cat "$earlier" "$fresh" > "$checked/joint-features.jsonl"

    "$verify" --prior-certificate "$signed/transport/fresh-signed.jsonl" \
        "$fresh" "$fresh_key" "$certificate" > "$checked/prior-certificate.json"
    "$verify" "$fresh" "$fresh_key" "$original/rank.jsonl" --rank > "$checked/rank-check.json"
    jq -es --slurpfile capture "$fresh" -f "$source/gate/proofs.jq" \
        "$original/rank.jsonl" > "$checked/rank-proofs.json"
    "$verify" "$earlier" "$earlier_key" "$original/old.jsonl" > "$checked/old-check.json"
    "$verify" "$fresh" "$fresh_key" "$original/fresh.jsonl" > "$checked/fresh-check.json"
    "$verify" "$checked/joint-features.jsonl" "$restore/verified/joint-key.jsonl" \
        "$original/joint.jsonl" > "$checked/joint-check.json"
    for receipt in prior-certificate.json rank-proofs.json {rank,old,fresh,joint}-check.json; do
        cmp "$original/$receipt" "$checked/$receipt"
    done

    jq -nc --slurpfile earlier "$earlier" --slurpfile recent "$fresh" \
        --slurpfile k0 "$earlier_key" --slurpfile k1 "$fresh_key" \
        -f "$source/rivals/tuple-overlap.jq" > "$checked/tuple-overlap.jsonl"
    cmp "$payload/rivals/$cut/tuple-overlap.jsonl" "$checked/tuple-overlap.jsonl"
    for mask in word passage passage-word supplied; do
        rival="$payload/rivals/$cut/$mask"
        jq -c --arg mask "$mask" --slurpfile truth "$fresh_key" \
            -f "$source/rivals/select.jq" "$fresh" > "$checked/rivals-$mask.jsonl"
        "$verify" "$checked/rivals-$mask.jsonl" "$fresh_key" "$rival/verdicts.jsonl" --rank \
            > "$checked/rivals-$mask.json"
        cmp "$rival/verified.json" "$checked/rivals-$mask.json"
        jq -ce --arg mask "$mask" \
            'select(.level=="panel_positive_only" and .scope=="supported_gold_vs_sites" and .arm=="actual")
            | {mask:$mask,offsets,margin,comparisons:.requirement_count,status:.verdict.status,
               box_maximum:.verdict.box_maximum,weighted_minimum:.verdict.weighted_minimum}' \
            "$rival/verdicts.jsonl"
    done > "$checked/rivals-summaries.jsonl"
    cmp "$payload/rivals/$cut/summaries.jsonl" "$checked/rivals-summaries.jsonl"
done

wrong_cut="$restore/verified/wrong-cut.stderr"
if "$verify" "$restore/transport/fresh-separator.jsonl" "$fresh_key" \
    "$payload/gate/zero/rank.jsonl" --rank > "$restore/verified/wrong-cut.stdout" 2> "$wrong_cut"; then
    echo "the independent checker accepted the zero-cut certificate on separator features" >&2
    exit 1
fi
case $(<"$wrong_cut") in
    *"certificate differs from independent coefficients or offsets"*) ;;
    *) cat "$wrong_cut" >&2; exit 1 ;;
esac
printf '%s\n' 'Rebuilt the feature tapes, verified the sealed verdicts, and rejected the wrong-cut witness.'
