#!/usr/bin/env bash
# Expand the five verified #4289 ordinary lanes to the six registered source arms.
set -euo pipefail
umask 077
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-mixed-transport"
prior="$r/20261126-codex-far-transport-margin/client/methods-v1"
audit="$r/20261126-codex-text-governor-links/client/adapter-v1/audit-tokens.jq"
test "$(hostname -s)" = wick
test "$(findmnt -no FSTYPE --target "$s")" = zfs
client="$s/client"; m="$client/methods-v1"
seal(){ (cd "$1"; test ! -f SHA256SUMS; find . -type f ! -name .pending -print0|LC_ALL=C sort -z|xargs -0 sha256sum > .pending; mv .pending SHA256SUMS); }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
case ${1:?stage, finite, or capture} in
stage)
  printf '%s  %s\n' f5f68a5fd572d9873b520e11eac8578cb8d307356e1a80eea0107ec2df01ed50 "$prior/SHA256SUMS" | sha256sum --strict --quiet -c -
  verify "$prior"; mkdir "$m"
  sed 's/20261126-codex-far-transport-margin/20261126-codex-mixed-transport/g;
    s/baseline learned correct_link/baseline far_only retained correct_link/g; s/\$i<20/\$i<24/g' "$prior/run-wick.sh" > "$m/run-wick.sh"
  sed 's/"baseline","learned","correct_link"/"baseline","far_only","retained","correct_link"/' "$prior/common.jq" > "$m/common.jq"
  sed 's/20261126-codex-far-transport-margin/20261126-codex-mixed-transport/g; s/learned:\$control,correct_link/far_only:\$control,retained:\$control,correct_link/; s/length==10/length==12/' "$prior/fixture-wick.sh" > "$m/fixture-wick.sh"
  cp "$prior/"{plan,grade,summary}.jq "$m/"
  sha256sum "$prior/SHA256SUMS" "$audit" > "$m/inputs.sha256"; seal "$m"
  ;;
finite)
  verify "$m"; bash "$m/fixture-wick.sh" fixture-v1; verify "$client/fixture-v1"
  jq -es 'length==12 and (map(.arm)|unique)==["baseline","correct_link","exchanged_link","far_only","native","retained"]' "$client/fixture-v1/plan.jsonl"
  jq -ce -f "$audit" "$client/fixture-v1/"{take,defer}-grade.json | jq -es 'length==2 and .[0].returned.exact_through_eos and .[1].returned.correct_prefix_at_cap'
  ;;
capture)
  gate=$(realpath "${2:?gate path}"); hash=${3:?gate SHA256}; label=${4:?new capture-vN}
  [[ $gate =~ ^$s/grade-v[1-9][0-9]*/summary\.json$ && $label =~ ^capture-v[1-9][0-9]*$ ]]
  verify "$(dirname "$gate")"; verify "$m"; (cd "$m" && sha256sum --strict --quiet -c inputs.sha256); verify "$client/fixture-v1"
  printf '%s  %s\n' "$hash" "$gate" | sha256sum --strict --quiet -c -
  jq -e '.pass==true and .pairs==128 and all(.gate[];.)' "$gate" >/dev/null
  bash "$m/run-wick.sh" capture "$gate" "$hash" "$(dirname "$gate")/endpoints.jsonl" "$label"
  out="$client/token-${label#capture-}"; mkdir "$out"
  verify "$client/$label"
  jq -ce -f "$audit" "$client/$label/grade/"*.jsonl > "$out/tokens.jsonl"
  jq -es 'if length!=1536 or (group_by(.arm)|length)!=6 then error("missing client read") else
    group_by(.arm)|map({arm:.[0].arm,events:length,first_eos:map(select(.first.exact_through_eos))|length,
    returned_eos:map(select(.returned.exact_through_eos))|length,returned_exact_pairs:group_by(.pair_id)|map(select(length==2 and all(.[];.returned.exact_through_eos)))|length,
    correct_prefix_at_cap:map(select(.returned.correct_prefix_at_cap))|length,
    first_divergences:(map(.first.first_differing_id)|group_by(.)|map({first:.[0],n:length})),
    returned_divergences:(map(.returned.first_differing_id)|group_by(.)|map({first:.[0],n:length}))}) end' "$out/tokens.jsonl" > "$out/summary.json"
  sha256sum "$gate" "$client/$label/SHA256SUMS" "$audit" > "$out/inputs.sha256"; seal "$out"
  ;;
*) exit 64 ;;
esac
