#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/archive-lib.sh"
root=${1:?verified archive extraction root}; out=${2:?new replay directory}
[[ $# == 2 && $root == /* && $here == "$root/"* ]]
layout="$root/layout.json"
jq -e '.schema=="uor-r4.codex-4300.cold-replay/1"' "$layout" >/dev/null
directory() {
    local selected
    selected=$(jq -er --arg key "$1" '.[$key] | select(type=="string")' "$layout")
    require_regular "$root" "$selected/SHA256SUMS"
    verify "$root/$selected"
    printf '%s\n' "$root/$selected"
}
release=$(directory release); capture=$(directory capture); panel=$(directory panel)
audit=$(directory audit); audit_build=$(directory audit_build); methods=$(directory methods)
grade=$(directory grade)
venue=$(jq -er '.compute_origin.capture.provider // "synthetic"' "$layout")
if [[ $venue == vast ]]; then
    finite=$(directory finite)
    binary=$(jq -er .device_binary "$layout")
    require_regular "$root" "$binary"
    for phase in finite capture; do
        if [[ $phase == finite ]]; then payload=$finite; else payload=$capture; fi
        verify "$payload-lease"
        test "$(cat "$payload-lease/exit-code.txt")" = 0
        (cd "$payload" && sha256sum --strict --quiet -c "$payload-lease/raw.SHA256SUMS")
        jq -en --arg phase "$phase" --slurpfile layout "$layout" \
            --arg binary "$(digest "$root/$binary")" --arg manifest "$(digest "$payload/SHA256SUMS")" \
            --arg raw "$(digest "$payload-lease/raw.SHA256SUMS")" \
            --slurpfile origin "$payload-lease/provider.json" '
            $origin[0] == $layout[0].compute_origin[$phase] and
            $origin[0].binary_sha256 == $binary and $origin[0].payload_manifest_sha256 == $manifest and
            $origin[0].raw_manifest_sha256 == $raw
        ' >/dev/null
    done
fi
ancestry="$root/ancestry"
(cd "$ancestry" && sha256sum --strict --quiet -c inputs.sha256)
before="$ancestry/methods-v2/inputs/protected.state"
after="$ancestry/train-v1/states/ordered.state"
banks="$ancestry/methods-v2/inputs/primary.words"
codec="$ancestry/tokenizer.json"
cmp "$after" "$release/ordered.unpacked"
mkdir "$out"
bash "$methods/grade/reduce.sh" "$capture" "$panel" "$out/grade"
mkdir "$out/audit"

compare() {
    cmp "$1/$3" "$2/$3"
    printf '%s\n' "$3"
}
for cohort in historical train-original train-derived; do
    compare "$grade" "$out/grade" "$cohort.rows.jsonl"
    for arm in before after; do
        state=$before; hist=ordered_base; train=base
        if [[ $arm == after ]]; then state=$after; hist=ordered_full; train=full; fi
        original=-
        if [[ $cohort == historical ]]; then original="$ancestry/held-v1/contrast/$hist.raw.jsonl"; fi
        if [[ $cohort == train-original ]]; then original="$ancestry/train-v1/ordered/$train.raw.jsonl"; fi
        name="$cohort-$arm.json"
        "$audit_build/switch-audit" replay "$panel/$cohort" "$codec" "$state" "$banks" \
            "$capture/$cohort/$arm" "$original" "$out/audit/$name"
        compare "$audit" "$out/audit" "$name"
    done
done > "$out/compared.txt"
for result in summary.json four-switches.json four-switches.tsv; do
    compare "$grade" "$out/grade" "$result"
done >> "$out/compared.txt"
[[ $(wc -l < "$out/compared.txt") == 12 ]]
jq -n --arg compared_sha256 "$(digest "$out/compared.txt")" --arg summary_sha256 "$(digest "$out/grade/summary.json")" \
    --arg venue "$venue" '
    {pass:true,compute_venue:$venue,cohorts:3,states_per_cohort:2,independent_replays:6,identical_outputs:12,
     compared_sha256:$compared_sha256,summary_sha256:$summary_sha256}
' > "$out/receipt.json"
(
    cd "$out"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS
)
