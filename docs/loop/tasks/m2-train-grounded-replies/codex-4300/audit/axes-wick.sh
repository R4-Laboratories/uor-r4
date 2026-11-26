#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
audit="$study/audit"
here=$(cd "$(dirname "$0")" && pwd)
build=$(realpath -e "${1:?sealed auditor build with selector support}")
layout=$(realpath -e "${2:?sealed supplementary panel}")
out=$(realpath -m "${3:?new axes receipt}")
prior="$runs/20261126-codex-fixed-assignment-median/panel/panel-v1"
primary="$study/panel/panel-v1"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
[[ $# == 3 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs &&
   $here == "$audit"/recipe-v* && $build == "$audit"/build-v* &&
   $layout == "$study"/matched-train/panel-v* && $out == "$audit"/selector-axes-v* ]]
for folder in "$build" "$layout" "$prior" "$primary"; do
    (cd "$folder" && sha256sum --strict --quiet -c SHA256SUMS)
done
mkdir "$out"
"$build/switch-audit" geometry "$primary/historical" "$codec" "$prior/contrast/blind/text.bin" "$out/historical.json"
"$build/switch-audit" geometry "$layout" "$codec" "$layout/endpoint/text.bin" "$out/layout.json"
jq -ne --slurpfile all "$out/historical.json" --slurpfile supplementary "$out/layout.json" '
    def counts($items):
        {links:($items|length),union:([$items[].changed_ordered_axes[]]|unique),
         per_axis:([$items[].changed_ordered_axes[]]|group_by(.)|map({axis:.[0],links:length})),
         by_gap:($items|group_by(.bpe_gap)|map({gap:.[0].bpe_gap,links:length,
                   support:([.[].changed_ordered_axes[]]|unique)}))};
    [$all[0].selector_support[]|select(.historical_lost)] as $h |
    $supplementary[0].selector_support as $l |
    if ($h|length)!=8 or ($l|length)!=8 or
       any(($h+$l)[];(.bpe_gap!=25 and .bpe_gap!=18) or
         (.changed_ordered_axes|length)==0 or any(.changed_ordered_axes[];.<20 or .>31))
    then error("selector support must cover eight corresponding original links per cohort") else . end |
    counts($h) as $a | counts($l) as $b |
    {historical:$a,train_layout:$b,
      common_axes:($a.union-($a.union-$b.union)),
      support_sets_disjoint:(($a.union-($a.union-$b.union))|length==0)}
' > "$out/summary.json"
sha256sum "$0" "$build/SHA256SUMS" "$primary/SHA256SUMS" "$layout/SHA256SUMS" "$prior/SHA256SUMS" "$codec" > "$out/inputs.sha256"
(
    cd "$out"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
