#!/usr/bin/env bash
set -euo pipefail
umask 077
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-protected-transport"
previous="$r/20261126-codex-mixed-transport/client/methods-v1"
token_audit="$r/20261126-codex-text-governor-links/client/adapter-v1/audit-tokens.jq"
c="$s/client"; m="$c/methods-v1"
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$c") == zfs ]]
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
seal(){ (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .pending -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >.pending && mv .pending SHA256SUMS); }
case ${1:?stage, finite or capture} in
stage)
  printf '%s  %s\n' 7d810248d92a306e2d3320007dfd2b3fe1d7647238276c8467914b7459580cdf "$previous/SHA256SUMS" | sha256sum --strict --quiet -c -
  verify "$previous"
  mkdir "$m"
  sed 's/20261126-codex-mixed-transport/20261126-codex-protected-transport/g
       s/far_only retained correct_link/far_only free protected correct_link/g
       s/\$i<24/\$i<28/g' "$previous/run-wick.sh" > "$m/run-wick.sh"
  sed 's/"far_only","retained"/"far_only","free","protected"/' "$previous/common.jq" > "$m/common.jq"
  sed 's/20261126-codex-mixed-transport/20261126-codex-protected-transport/g
       s/far_only:\$control,retained:\$control/far_only:\$control,free:\$control,protected:\$control/g
       s/length==12/length==14/g' "$previous/fixture-wick.sh" > "$m/fixture-wick.sh"
  cp "$previous/plan.jq" "$previous/grade.jq" "$previous/summary.jq" "$m/"
  sha256sum "$previous/SHA256SUMS" "$token_audit" > "$m/inputs.sha256"; seal "$m"
  ;;
finite)
  verify "$m"; bash "$m/fixture-wick.sh" fixture-v1; verify "$c/fixture-v1"
  jq -es 'length==14 and (map(.arm)|unique)==["baseline","correct_link","exchanged_link","far_only","free","native","protected"]' "$c/fixture-v1/plan.jsonl" >/dev/null
  jq -ce -f "$token_audit" "$c/fixture-v1/"{take,defer}-grade.json |
    jq -es 'length==2 and .[0].returned.exact_through_eos and .[1].returned.correct_prefix_at_cap' >/dev/null
  ;;
capture)
  gate=$(realpath "${2:?gate path}"); digest=${3:?gate SHA}; label=${4:?capture-vN}
  [[ $gate =~ ^$s/grade-v[1-9][0-9]*/summary\.json$ && $label =~ ^capture-v[1-9][0-9]*$ ]]
  verify "$m"; verify "$c/fixture-v1"; verify "$(dirname "$gate")"
  printf '%s  %s\n' "$digest" "$gate" | sha256sum --strict --quiet -c -
  jq -e '.pass==true and .pairs==128 and all(.gate[];.)' "$gate" >/dev/null
  bash "$m/run-wick.sh" capture "$gate" "$digest" "$(dirname "$gate")/endpoints.jsonl" "$label"
  verify "$c/$label"; out="$c/token-${label#capture-}"; mkdir "$out"
  jq -ce -f "$token_audit" "$c/$label/grade/"*.jsonl > "$out/tokens.jsonl"
  jq -es 'if length!=1792 or (group_by(.arm)|length)!=7 then error("missing ordinary reply") else
    group_by(.arm)|map({arm:.[0].arm,events:length,first_eos:map(select(.first.exact_through_eos))|length,
    returned_eos:map(select(.returned.exact_through_eos))|length,exact_pairs:group_by(.pair_id)|map(select(length==2 and all(.[];.returned.exact_through_eos)))|length,
    correct_prefix_at_cap:map(select(.returned.correct_prefix_at_cap))|length,
    first_divergences:map(.first.first_differing_id)|group_by(.)|map({id:.[0],n:length}),
    returned_divergences:map(.returned.first_differing_id)|group_by(.)|map({id:.[0],n:length})}) end' "$out/tokens.jsonl" > "$out/summary.json"
  sha256sum "$gate" "$c/$label/SHA256SUMS" "$token_audit" > "$out/inputs.sha256"; seal "$out"
  ;;
*) exit 64 ;;
esac
