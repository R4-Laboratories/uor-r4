def require($condition; $reason):
    if $condition then . else error($reason) end;

def selector_edits($first; $later):
    [$first.text | match("\\bfirst\\b";"g") | .offset as $at |
        select($later.text == ($first.text[:$at] + "later" + $first.text[$at + 5:])) |
        $at];

def roles:
    group_by(.role) |
    map({role:.[0].role, requests:length, rows:(map(.row_index)|unique|length)});

def count:
    {
        rows:length,
        with_rival:(map(select(.eligible_requests > 0)) | length),
        without_rival:(map(select(.eligible_requests == 0)) | length),
        requests:(map(.eligible_requests) | add),
        signs:([.[].requests[]] | roles),
        pairs:(map(.pair_id) | unique | length)
    };

. as $rows |
require(($rows|length) == $expected; "incomplete physical census") |
require(($rows | group_by(.pair_id) | all(.[]; length == 2)); "incomplete pair endpoint") |
{
    cohort:$cohort,
    population:($rows | count),
    by_family:($rows | group_by(.family) | map({family:.[0].family} + count)),
    by_distance:($rows | group_by(.distance) | map({distance:.[0].distance} + count)),
    by_stratum:($rows | group_by([.family,.fact,.distance,.governor_order,.admission]) |
        map({family:.[0].family,fact:.[0].fact,distance:.[0].distance,
             governor_order:.[0].governor_order,admission:.[0].admission} + count)),
    first_later_edits:($rows | group_by(.pair_id) | map(
        .[0] as $a | .[1] as $b |
        select((selector_edits($a;$b) | length) > 0 or (selector_edits($b;$a) | length) > 0) |
        .[0].pair_id)),
    historical_losses:($rows | map(select(.historical_lost)) | group_by(.pair_id) |
        map({pair_id:.[0].pair_id,cases:map({row_index,case_id,selector,sign:.requests[0].sign})}))
} |
require((.first_later_edits|length) == (if $cohort=="train-original" then 0 else $expected/2 end);
    "the actual text differs from its registered selector enrollment") |
require((.historical_losses|length) == (if $cohort=="historical" then 4 else 0 end);
    "wrong historical loss enrollment")
