def require($ok; $problem): if $ok then . else error($problem) end;

def unique_index($coordinates; $address):
    [$coordinates | to_entries[] | select(.value == $address) | .key] |
    require(length == 1; "protected physical address is absent or duplicated") | .[0];

def raw_native:
    .bytes |= (.start -= 1 | .end -= 1);

($mask | split("\n") | .[1:] | map(select(length > 0) | split("\t"))) as $pairs |
require(($pairs | length) == 24 and ($pairs | map(.[0]) | unique | length) == 24 and
    all($pairs[]; .[1:] == ["true","true","true","true"]);
    "predecessor table is missing one of the 24 protected near pair identities") |
($pairs | map(.[0])) as $keys |
require(length == 256 and ($coordinates | length) == 256;
    "predecessor mixed packet cardinality differs") |
[to_entries[] |
    .key as $row_index | .value as $reference | $coordinates[$row_index] as $map |
    select($keys | index($reference.pair_id) != null) |
    require($reference.distance == "near" and $row_index == $map.ordinal and
       $reference.id == $map.id and $reference.structural_prompt_sha256 == $map.prompt_sha256;
       "protected reference is not the original near row of the packet") |
    [ ["affirmative","excluded"][] as $role |
        $reference.governors[$role] as $governor |
        ($governor.native | raw_native) as $address |
        require($address.bytes == $governor.bytes and
            [$address.tokens.start,$address.tokens.end] == $governor.tokens;
            "protected governor byte/BPE mismatch") |
        unique_index($map.words; $address)
    ] as $governors |
    [$reference.native_scope[] | unique_index($map.sites; raw_native)] as $sites |
    require(($sites | unique | length) == 3 and ($governors | unique | length) == 2 and
        any($map.tuples[]; . == $sites);
        "protected full tuple or its original governors are ineligible") |
    {row_index:$row_index, pair_id:$reference.pair_id, admission:$reference.admission,
     governors:$governors, sites:$sites}
] |
require(length == 48 and (group_by(.pair_id) | length) == 24 and
    all(group_by(.pair_id)[]; map(.admission) | sort == ["defer","take"]);
    "both opposed original prompts were not found for every protected pair") |
.[] | del(.admission)
