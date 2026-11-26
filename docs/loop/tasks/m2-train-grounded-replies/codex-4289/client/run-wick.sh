#!/usr/bin/env bash
# Use the authenticated ordinary planner, trace grader and frozen writer already qualified at #4288.
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-far-transport-margin"
old="$runs/20261126-codex-text-governor-links/client/methods-v1"
raw="$runs/20261126-codex-text-governor-links/client/adapter-v1/audit-tokens.jq"
client="$s/client"
version=${2:?version}; [[ $version =~ ^v[1-9][0-9]*$ ]]
methods="$client/methods-$version"
test "$(hostname -s)" = wick
test "$(findmnt -no FSTYPE --target "$s")" = zfs
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
seal(){ (cd "$1" && find . -type f ! -name .pending -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > .pending && mv .pending SHA256SUMS); }
case $1 in
stage)
  printf '%s  %s\n' 8adf36f3fec0bbc7aa47230f391101c2d9d78d9ef718f21716c1fefd39c4ff68 "$old/SHA256SUMS" | sha256sum --strict --quiet -c -
  verify "$old"
  mkdir "$methods"
  for f in common.jq fixture-wick.sh run-wick.sh; do
    sed -e 's/20261126-codex-text-governor-links/20261126-codex-far-transport-margin/g' \
        -e 's/governor_gap/baseline/g' \
        -e 's@frozen_panel="\$study/panel/panel-v1"@frozen_panel="$study/panel/panel-v1/held"@' \
        -e 's@verify "\$frozen_panel"@verify "$frozen_panel/.."@' \
        -e 's@"\$frozen_panel/SHA256SUMS"@"$frozen_panel/../SHA256SUMS"@g' "$old/$f" > "$methods/$f"
  done
  cp "$old/"{grade,plan,summary}.jq "$methods/"
  sha256sum "$old/SHA256SUMS" "$raw" > "$methods/input.sha256"
  seal "$methods"
  ;;
finite)
  verify "$methods"
  bash "$methods/fixture-wick.sh" "fixture-$version"
  verify "$client/fixture-$version"
  jq -es 'length==10 and (map(.arm)|unique)==["baseline","correct_link","exchanged_link","learned","native"]' \
    "$client/fixture-$version/plan.jsonl"
  jq -ce -f "$raw" "$client/fixture-$version/"{take,defer}-grade.json | jq -es 'length==2 and .[0].returned.exact_through_eos and .[1].returned.correct_prefix_at_cap'
  ;;
capture)
  gate=$(realpath "${3:?approved gate}")
  sha=${4:?approved sha}
  label=${5:?new capture-vN}
  [[ $gate =~ ^$s/grade-v[1-9][0-9]*/summary\.json$ && $label =~ ^capture-v[1-9][0-9]*$ ]]
  verify "$(dirname "$gate")"; verify "$methods"; verify "$client/fixture-$version"
  printf '%s  %s\n' "$sha" "$gate" | sha256sum --strict --quiet -c -
  jq -e '.pass==true and .pairs==128 and all(.gate[];.)' "$gate" >/dev/null
  endpoints="$(dirname "$gate")/endpoints.jsonl"
  bash "$methods/run-wick.sh" capture "$gate" "$sha" "$endpoints" "$label"
  verify "$client/$label"
  out="$client/token-${label#capture-}"; mkdir "$out"
  jq -ce -f "$raw" "$client/$label/grade/"*.jsonl > "$out/tokens.jsonl"
  jq -es 'if length!=1280 then error("missing real client runs") else group_by(.arm)|map({
    arm:.[0].arm,returned_exact_eos:map(select(.returned.exact_through_eos))|length,
    returned_cap:map(select(.returned.correct_prefix_at_cap))|length,
    wrong_token:map(.returned.first_differing_id)|group_by(.)|map({first:.[0],count:length})}) end' \
    "$out/tokens.jsonl" > "$out/summary.json"
  sha256sum "$gate" "$endpoints" "$raw" "$client/$label/SHA256SUMS" > "$out/inputs.sha256"
  seal "$out"
  ;;
*) exit 64 ;;
esac
