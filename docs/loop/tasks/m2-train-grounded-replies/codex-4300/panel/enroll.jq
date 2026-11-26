def require($condition; $reason):
    if $condition then . else error($reason) end;

def native_bytes:
    .bytes |= (.start -= 1 | .end -= 1);

def raw_address($bytes; $tokens):
    {bytes:$bytes, tokens:{start:$tokens[0], end:$tokens[1]}};

def original_index($addresses; $address):
    [
        $addresses | to_entries[] | select(.value == $address) | .key
    ] | require(length == 1; "original address absent or duplicated") | .[0];

def address($addresses; $index; $row):
    $addresses[$index] as $span |
    $span + {
        index:$index,
        text:$row.turns[1][$span.bytes.start:$span.bytes.end],
        bpe_ids:$row.prompt[$span.tokens.start:$span.tokens.end]
    } |
    require((.text | length) == (.bytes.end - .bytes.start)
        and (.bpe_ids | length) == (.tokens.end - .tokens.start)
        and (.text | test("^[A-Za-z]+$")); "invalid original word slice");

def indices_for($row; $coordinates; $role):
    original_index($coordinates.words;
        raw_address($row.governors[$role].bytes; $row.governors[$role].tokens));

def overlaps($one; $two):
    $one.start < $two.end and $two.start < $one.end;

def pair_identity($row; $index):
    {
        row_index:$index, pair_id:$row.pair_id, case_id:$row.id, family:$row.family,
        fact:$row.fact, distance:$row.distance, governor_order:$row.governor_order,
        admission:$row.admission, selector:($row.contrast.selector // null),
        historical_lost:($lost[0] | any(.[]; .pair_id == $row.pair_id))
    };

def enroll($row; $coords; $index; $position):
    ["affirmative", "excluded"] as $roles |
    $roles[$position] as $role |
    $roles[1 - $position] as $other_role |
    (indices_for($row; $coords; $role)) as $governor_index |
    (indices_for($row; $coords; $other_role)) as $other_governor_index |
    (address($coords.words; $governor_index; $row)) as $governor |
    (address($coords.words; $other_governor_index; $row)) as $other_governor |
    ($row.native_scope | map(. | native_bytes |
        . as $span | original_index($coords.sites; $span))) as $target |
    $target[$position] as $requested_index |
    (address($coords.sites; $requested_index; $row)) as $requested |
    (address($coords.sites; $target[1 - $position]; $row)) as $opposite |
    (address($coords.sites; $target[2]; $row)) as $heading |
    ($coords.tuples | map(.[$position]) | group_by(.) |
        map({key:(.[0]|tostring),value:length}) | from_entries) as $counts |
    (if $cohort == "train-original" then
        [$coords.sites | to_entries[] | .key]
     else
        [original_index($coords.sites; $row.neutral_rival | native_bytes)]
     end) as $candidate_indices |
    require(
        $governor_index != $other_governor_index
        and any($coords.tuples[]; . == $target)
        and $row.governors[$role].noun == $requested.bytes
        and $row.governors[$role].native_noun == $requested.tokens
        and ($row.governors[$role].native | native_bytes) ==
            ($governor | {bytes,tokens});
        "signed reference does not identify its physical original tuple and governor"
    ) |
    [$candidate_indices[] as $rival_index |
        $coords.sites[$rival_index] as $candidate |
        select($rival_index != $requested_index
            and ($counts[$rival_index|tostring] // 0) > 0
            and (overlaps($candidate.bytes; $governor.bytes) | not)
            and $row.turns[1][$candidate.bytes.start:$candidate.bytes.end] == $requested.text) |
        (address($coords.sites; $rival_index; $row)) as $rival |
        pair_identity($row; $index) + {
            sign:(1 - 2 * $position), role:$role,
            governor_word_index:$governor_index, requested_site_index:$requested_index,
            rival_site_index:$rival_index,
            governor:$governor, requested:$requested, rival:$rival,
            opposite_governor:$other_governor, opposite:$opposite, heading:$heading,
            requested_fixed_tuple_count:$counts[$requested_index|tostring],
            rival_fixed_tuple_count:$counts[$rival_index|tostring]
        }
    ] |
    require($cohort == "train-original" or length == 1;
        "registered selector rival is missing, ineligible or not an exact surface match");

require(length == ($coordinates|length); "coordinate length mismatch") |
to_entries[] | .key as $index | .value as $row | $coordinates[$index] as $coords |
require($coords.ordinal == $index and $coords.id == $row.id
    and $coords.prompt_sha256 == $row.structural_prompt_sha256
    and ($coords.tuples|length) == $row.scope_tuple_count
    and ($row.turns[1] | explode | length > 0 and all(.[]; . >= 32 and . <= 126));
    "row, original prompt digest or original tuple census mismatch") |
(if $cohort == "train-original" then [0,1]
 elif $row.admission == "take" then [0]
 elif $row.admission == "defer" then [1]
 else error("invalid selector sign") end) as $positions |
[$positions[] | enroll($row; $coords; $index; .)[]] as $requests |
pair_identity($row; $index) + {
    text:$row.turns[1], prompt_sha256:$coords.prompt_sha256,
    coordinate_map:($coords | {words,sites,tuples}),
    eligible_requests:($requests | length), requests:$requests
}
