# Independently recount the raw JSONL visits. Invoke with jq -s and
# --slurpfile reported epochs.json. All arithmetic here uses exact integers
# below 2^53; no reported result supplies any numerator or denominator.
def strict: .margin_q48 > 0;
def elections($kind):
    {
        strict: (map(select(.[$kind] | strict)) | length),
        ties_won: (map(select(.[$kind].margin_q48 == 0 and .[$kind].rival > .gold_id)) | length),
        ties_lost: (map(select(.[$kind].margin_q48 == 0 and .[$kind].rival < .gold_id)) | length)
    };
def census:
    {
        rows: length,
        ceiling: elections("ceiling"),
        zero: elections("zero"),
        before: elections("before"),
        old_rival_strict: (map(select(.old_rival_margin_q48 > 0)) | length),
        old_rival_ties_won: (map(select(.old_rival_margin_q48 == 0 and .before.rival > .gold_id)) | length),
        old_rival_ties_lost: (map(select(.old_rival_margin_q48 == 0 and .before.rival < .gold_id)) | length),
        immediate: elections("after"),
        checkpoint: elections("checkpoint"),
        retained: (map(select((.after | strict) and (.checkpoint | strict))) | length),
        later_gains: (map(select((.after | strict | not) and (.checkpoint | strict))) | length),
        immediate_gains: (map(select((.before | strict | not) and (.after | strict))) | length),
        clipped_rows: (map(select(.clipped_words != 0)) | length),
        clipped_words: (map(.clipped_words) | add // 0),
        aliased_rows: (map(select(.alias_count != 0)) | length),
        first_losses: {
            structural: (map(select(.ceiling | strict | not)) | length),
            clipped_old_rival: (map(select((.ceiling | strict) and .old_rival_margin_q48 <= 0 and .clipped_words > 0)) | length),
            old_rival: (map(select((.ceiling | strict) and .old_rival_margin_q48 <= 0 and .clipped_words == 0)) | length),
            replacement: (map(select((.ceiling | strict) and .old_rival_margin_q48 > 0 and (.after | strict | not))) | length),
            epoch_end: (map(select((.after | strict) and (.checkpoint | strict | not))) | length)
        }
    };
def pairs:
    group_by(.row.ordinal)
    | if length != 3891 then error("incomplete ordinal domain") else . end
    | map(
        if (map(.mode) | sort) != ["geometry", "ordered-matched"] then error("missing or duplicated arm")
        else {geometry: (map(select(.mode == "geometry"))[0].row), ordered: (map(select(.mode == "ordered-matched"))[0].row)}
        end
        | if (.geometry | [.ordinal, .group, .gold_id, .zero]) != (.ordered | [.ordinal, .group, .gold_id, .zero])
            then error("ordinal, group, gold or frozen score drift")
          else . end
    );
def mode_arm($pairs; $common; $mode):
    {all: ($pairs | map(.[$mode]) | census), common_attainable: ($common | map(.[$mode]) | census)};
def take_seed($seed; $pairs):
    ($pairs | map(select((.geometry.ceiling | strict) and (.ordered.ceiling | strict)))) as $common
    | {
        seed: $seed,
        geometry: mode_arm($pairs; $common; "geometry"),
        ordered: mode_arm($pairs; $common; "ordered"),
        common_groups: ($common | map(.geometry.group) | unique | length)
    }
    | .geometry.common_attainable as $g
    | .ordered.common_attainable as $o
    | .meets_endpoints = (
        $g.rows > 0
        and 20 * ($g.checkpoint.strict - $g.zero.strict) >= $g.rows
        and 20 * ($g.checkpoint.strict - $o.checkpoint.strict) >= $g.rows
    )
    | .meets_retention = ($g.immediate.strict > 0 and 5 * $g.retained >= 4 * $g.immediate.strict);
def clean_report:
    map({
        epoch, advance,
        seeds: (.seeds | map({
            seed,
            geometry: (.geometry | {all, common_attainable}),
            ordered: (.ordered | {all, common_attainable}),
            common_groups: .common_attainable.groups,
            meets_endpoints: .common_attainable.meets_endpoints,
            meets_retention: .common_attainable.meets_retention
        }))
    });

if any(group_by([.mode, .row.ordinal])[];
    (map(.row | [.group, .gold_id, .zero, .ceiling, .prior_rank, .alias_count]) | unique | length) != 1)
then error("frozen population or support drift across epochs or seeds") else . end
| group_by(.epoch)
| if map(.[0].epoch) != [1,2,3,4,5,6,7,8] then error("missing or unexpected epoch") else . end
| map(
    .[0].epoch as $e
    | group_by(.seed)
    | if map(.[0].seed) != [435300,435301] then error("missing or unexpected seed") else . end
    | map(.[0].seed as $seed | pairs | take_seed($seed; .))
    | {epoch: $e, seeds: ., advance: (if $e == 8 then all(.[]; .meets_endpoints and .meets_retention) else null end)}
)
| . as $independent
| if $independent != ($reported[0] | clean_report) then error("raw visit recount disagrees with the reported grade")
  else $independent end
