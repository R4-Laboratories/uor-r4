#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
v=${1:?new report fixture version}; version "$v"; out="$study/report-fixture-$v"; mkdir "$out"
jq -nc -f "$here/grade/fixtures.jq" > "$out/fixture.json"
for field in rows meta before after; do jq -c --arg key "$field" '.[$key][]' "$out/fixture.json" > "$out/$field.jsonl"; done
grade() {
    jq -nc --arg cohort fixture --slurpfile rows "$out/rows.jsonl" --slurpfile meta "$1" \
        --slurpfile before "$out/before.jsonl" --slurpfile after "$out/after.jsonl" -f "$here/grade/row.jq"
}
grade "$out/meta.jsonl" > "$out/moves.json"
jq -e '.before.local=="win" and .before.joint=="loss" and .before.stage=="joint-only reversal" and
       .after.local=="loss" and .after.stage=="local loss" and .after.native.choice=="third site" and
       .before.requested_tuple.signed_governors[0].text=="V" and .after.native.signed_site.text=="B"' "$out/moves.json" >/dev/null
jq -c '.rival.bytes=.requested.bytes' "$out/meta.jsonl" > "$out/wrong-address.jsonl"
if grade "$out/wrong-address.jsonl" > "$out/wrong.json" 2>"$out/wrong.log"; then exit 1; fi
grep -Fq 'original governor/noun/spelling enrollment mismatch' "$out/wrong.log"
jq -n '{pass:true,local_loss:true,joint_only_reversal:true,third_site:true,same_word_wrong_address_rejected:true}' > "$out/receipt.json"
sha256sum "$here/SHA256SUMS" > "$out/inputs.sha256"
seal "$out"; sha256sum "$out/SHA256SUMS"
