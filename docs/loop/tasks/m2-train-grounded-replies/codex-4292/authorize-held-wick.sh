#!/usr/bin/env bash
# Bind each independent TRAIN replay to the corresponding physical batch before opening held ranks.
source "$(dirname "$0")/common-wick.sh"
mv=${1:?measurement version}; av=${2:?complete independent audit version}; pv=${3:?unused permit version}
[[ $mv =~ ^v[1-9][0-9]*$ && $av =~ ^v[1-9][0-9]*$ && $pv =~ ^v[1-9][0-9]*$ ]]
m="$s/measurement-$mv"; a="$s/audit/train-$av"; out="$s/train-verified-$pv"
verify "$a"; verify "$m"; verify "$here"; sha256sum --strict --quiet -c "$a/inputs.sha256"
for f in "$m/SHA256SUMS" "$m/selection.sha256" "$m/selection.json"; do
 grep -Fxq "$(sha256sum "$f")" "$a/inputs.sha256"
done
# A replay that lists only the first round cannot authorize a held read.
unique=$(for batch in root protected free; do if [[ -d $m/train/$batch/device ]]; then echo "$batch"; fi; done | jq -Rn '[inputs]|sort')
jq -e --argjson expected "$unique" '.pass and (.batches|sort)==$expected and .compared_unique_batches==($expected|length)' "$a/verified.json" >/dev/null
members=()
for batch in $(jq -r '.[]' <<<"$unique"); do
 d="$s/audit/batch-$av-$batch"
 verify "$d"
 jq -e '.pass and .candidate_count>0 and .comparison_rows_per_state==256 and .joint_rows_verified==((.candidate_count+1)*256) and
        .signed_edges_verified==(.joint_rows_verified*2)' "$d/readback.json" >/dev/null
 members+=("$d/SHA256SUMS" "$m/train/$batch/device/SHA256SUMS" "$m/train/$batch/grade/SHA256SUMS")
done
mkdir "$out"; cp "$a/verified.json" "$out/verified.json"
sha256sum "$out/verified.json" "$a/SHA256SUMS" "$a/inputs.sha256" "$m/SHA256SUMS" "$m/selection.sha256" "$m/selection.json" \
 "$here/SHA256SUMS" "$here/authorize-held-wick.sh" "${members[@]}" > "$out/inputs.sha256"
sha256sum --strict --quiet -c "$out/inputs.sha256"; sha256sum "$out/inputs.sha256"
