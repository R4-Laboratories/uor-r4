#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

[[ $# == 1 ]]
out=$1
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$(dirname "$out")") == zfs ]]
here=$(cd "$(dirname "$0")" && pwd)
previous=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-paired-occurrence-margin
packet="$previous/panel/fresh"
physical="$previous/rented-fresh-v1/held-raw"
native="$physical/score/fresh/median-bilateral-3.native.jsonl"
costs="$physical/score/fresh/median-bilateral-3.costs.jsonl"
mkdir "$out"

jq -enc --slurpfile geometry "$packet/rows.jsonl" --slurpfile gold "$packet/gold.jsonl" \
    --slurpfile correct "$physical/controls/gold.jsonl" --slurpfile swap "$physical/controls/swapped.jsonl" \
    -f "$here/held-controls.jq" > "$out/controls.json"
if jq -enc --slurpfile geometry "$packet/rows.jsonl" --slurpfile gold "$packet/gold.jsonl" \
    --slurpfile correct "$physical/controls/swapped.jsonl" --slurpfile swap "$physical/controls/gold.jsonl" \
    -f "$here/held-controls.jq" > /dev/null 2> "$out/reversed-controls.rejection.txt"; then
    echo "an intentionally reversed control passed" >&2
    exit 1
fi
rg -F 'supplied-control capture disagrees' "$out/reversed-controls.rejection.txt" >/dev/null

# The precedent does not have a matching lexical probe. For this fixture only,
# encode the known local sign admission in dummy distance pairs. The archived
# source rankings, coordinate maps, and control reads are used byte for byte.
jq -enc --slurpfile geometry "$packet/rows.jsonl" --slurpfile meta "$packet/requests.jsonl" \
    --slurpfile raw "$costs" '
    range(256) as $i | $raw[$i] as $c | $meta[$i] as $m |
    (if $m.sign == 1 then [$m.governor,$m.opposite_governor] else [$m.opposite_governor,$m.governor] end) as $w |
    {row:$i,words:([$geometry[$i].coordinate_map.words[]|[8,9,0,0]] |
        .[$w[1].index] = [9,8,0,0] |
        .[$c.word] = (if (($c.sites[$c.requested].local != null) == ($c.sign == 1))
            then [8,9,0,0] else [9,8,0,0] end))}
' > "$out/fixture-probe.jsonl"
jq -enc --slurpfile gold "$packet/gold.jsonl" --slurpfile meta "$packet/requests.jsonl" \
    --slurpfile reference "$packet/reference.json" '
    range(256) as $i | $gold[$i] as $g | $meta[$i] as $m | $reference[0][$i] as $r |
    {row_index:$i,pair_id:$g.pair_id,case_id:$g.case_id,
     affirmative:(if $m.sign == 1 then $m.governor else $m.opposite_governor end),
     requested_gap_bpe:$r.governors.referent.gap,requested_role:$r.requested_role,requested_horizon:$r.requested_horizon}
' > "$out/fixture-registered.jsonl"
jq -enc --slurpfile geometry "$packet/rows.jsonl" --slurpfile meta "$packet/requests.jsonl" \
    --slurpfile gold "$packet/gold.jsonl" --slurpfile registered "$out/fixture-registered.jsonl" \
    --slurpfile reads "$native" --slurpfile costs "$costs" --slurpfile probe "$out/fixture-probe.jsonl" \
    --slurpfile baseline_probe "$out/fixture-probe.jsonl" \
    --slurpfile review /dev/null -f "$here/held-row.jq" > "$out/rows.jsonl"

# Same raw observations in every lane must give zero gains, losses, and
# interactions. Then give just the combined arm one paired win to exercise the
# interaction and identity joined counts independently of the headline.
jq -c 'if .row < 2 then .source=true else . end' "$out/rows.jsonl" > "$out/combined.fixture.jsonl"
jq -enc --slurpfile controls "$out/controls.json" --slurpfile baseline "$out/rows.jsonl" \
    --slurpfile margin "$out/rows.jsonl" --slurpfile eligibility "$out/rows.jsonl" \
    --slurpfile combined "$out/combined.fixture.jsonl" --slurpfile wrong "$out/rows.jsonl" \
    -f "$here/held-summary.jq" > "$out/summary.json"
jq -e '
    .controls.gold_exact_original_pairs == 128 and .controls.swapped_exact_wrong_original_pairs == 128 and
    .arms.baseline.overall.exact_signed_pairs == 0 and
    .arms.baseline.overall.pinned_third.status == {"loss":224,"missing-target":32} and
    .arms.baseline.overall.pinned_target_free_third.deciding_key == {"loss/lexical":128,"loss/transport":96} and
    .arms.baseline.overall.route_attribution["tuple-tie"] > 0 and
    .effects.margin.paired.source.gains == [] and .effects.eligibility.paired.source.gains == [] and
    .effects.full_factorial.paired.source.gains == ["courtroom-inventory-minutes/apiary-and-bay/2"] and
    .paired_interaction.source.interaction == 1 and
    .paired_interaction.source.combined_only == ["courtroom-inventory-minutes/apiary-and-bay/2"]
' "$out/summary.json" > "$out/fixture-pass.json"
(
    cd "$out"
    sha256sum controls.json rows.jsonl summary.json fixture-pass.json reversed-controls.rejection.txt > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS
)
