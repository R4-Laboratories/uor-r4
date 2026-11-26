#!/usr/bin/env bash
# Additional novelty check against the original #4283 primary and working reserve copies.
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-protected-transport/panel"
archive="$runs/20261125-codex-governing-source-pairs"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive check version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ && $here == "$root"/methods-v* ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs ]]
sealed="$root/panel-v1"
design="$root/private/held-v4.json"
ref="$sealed/held/reference.json"
primary="$archive/held/primary"
held="$primary/panel-v1/fresh/reference.json"
regress="$primary/panel-v1/regress/reference.json"
original="$primary/panel-design-v1/fresh/reference.json"
reserve="$archive/reserve/sandbox-v6/reference.json"
out="$root/antecedents-v$version"
mkdir "$out"
printf '%s  %s\n' \
    22213dcba2081598d51e08cb573ca5a721ad26dcb04c20d4fd906a5cbae511cc "$sealed/SHA256SUMS" \
    ff71ba32ec6f99a3d98de5d2ad4dd1bed60e0e71350d89f8404819e209bf2eb1 "$design" \
    62067f2a1762ca85385910cb89d0b233cae8a1fa60c3e3c8aa58b395b0a50f8f "$held" \
    ef98f04867309e77558b0bfed5ee569727a94f51dcf37d33e8cf50136302848a "$regress" \
    f96029e315babc6a65823eafc6dfd13034b0705b04147b183a955d625cb50bfd "$original" \
    9229e98dc330b73eca6e6564c627c8149664ed3c87de913b127a8a5cd077b634 "$reserve" > "$out/inputs.sha256"
sha256sum --strict --quiet -c "$out/inputs.sha256"
(cd "$sealed" && sha256sum --strict --quiet -c SHA256SUMS)
jq -e --slurpfile design "$design" --slurpfile history "$sealed/history.json" \
    --slurpfile prior <(jq -c '.' "$held" "$regress" "$original" "$reserve") '
    def require($ok;$message): if $ok then . else error($message) end;
    def disjoint($x;$y): ($x - $y) == $x;
    def words: ascii_downcase | scan("[a-z]+");
    . as $fresh | ($prior | add) as $old |
    require(length==256 and ($prior|map(length))==[256,16,256,256]; "panel census changed") |
    require(all($old[]; (.turns | length)==2 and (.second_suffix|length)>0); "prior encoded turns missing") |
    require(disjoint([$fresh[].id]; [$old[].id]); "case ID overlap") |
    require(disjoint([$fresh[].family]; [$old[].family]); "wording family overlap") |
    require(disjoint([$fresh[].fact, $fresh[].fact_spans[].value]; [$old[] | .fact, .fact_spans[]?.value]); "fact overlap") |
    require(disjoint([$fresh[].turns[1]]; [$old[].turns[]]); "authored text overlap") |
    require(disjoint([$fresh[].second_suffix]; [$old[].second_suffix]); "encoded user suffix overlap") |
    require(disjoint([$fresh[].prompt]; [$old[] | .stock.prompt //
        (if .structural_history then .structural_history.prompt + .structural_history.ids + .second_suffix
         else .first_prompt + $history[0].reply_ids + .second_suffix end)]); "encoded complete prompt overlap") |
    require(disjoint([$design[0].wordings[] | .reference, .rival]; [$old[].turns[] | words] | unique);
        "new target noun appears in earlier primary or reserve user text") |
    {pass:true, held_rows:length, prior_reference_records:($old|length), prior_reference_sizes:($prior|map(length)),
    overlap:{ids:0,families:0,facts:0,authored_text:0,encoded_suffix:0,encoded_prompt:0,nouns:0}}
' "$ref" > "$out/checks.json"
(cd "$out" && sha256sum inputs.sha256 checks.json > SHA256SUMS && sha256sum --strict --quiet -c SHA256SUMS)
chmod -R a-w "$out"
cat "$out/checks.json"
