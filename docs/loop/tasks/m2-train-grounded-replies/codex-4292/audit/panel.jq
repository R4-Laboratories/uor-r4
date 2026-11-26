def require($ok; $why): if $ok then . else error($why) end;
def bounds: [.start,.end];
def span($coordinates): .[$coordinates.start:$coordinates.end];
def sorted_governors(f):
    [.governors.affirmative,.governors.excluded] | map(f) | sort_by(.start);
def split_words($text; $a; $b):
    [$text[:$a.start],$text[$a.end:$b.start],$text[$b.end:]];
def verify_spans($text; $ids; $gov):
    ($gov.native_noun|bounds) as $noun |
    require($gov.bytes.start>=0 and $gov.bytes.end>$gov.bytes.start and $gov.bytes.end<=($text|length) and
            $gov.tokens[0]>=0 and $gov.tokens[1]>$gov.tokens[0] and $gov.tokens[1]<=($ids|length) and
            ([($noun[0]-$gov.tokens[1]),($gov.tokens[0]-$noun[1]),0]|max)==$gov.gap;
            "unresolvable governor or claimed BPE gap");

. as $rows |
require(length==256 and (map(.id)|unique|length)==256 and
    all(.[];(.turns[1]|length)==(.turns[1]|utf8bytelength)); "expected unique ASCII original turns") |
require(all(.[]; .structural_history == {prompt:$history[0].prompt_ids,ids:$history[0].reply_ids,
      stop:"eos",text:$history[0].reply_text} and
     .turns[0]==$history[0].user and .prompt==($history[0].prompt_ids+$history[0].reply_ids+.second_suffix));
    "history or original absolute tokens changed") |
(group_by(.pair_id) | map(
    require(length==2 and (map(.admission)|sort)==["defer","take"] and
       (map(.distance)|unique|length)==1 and (map(.family)|unique|length)==1;
       "unpaired endpoint") |
    map(. as $r | .turns[1] as $text |
        verify_spans($text; .prompt; .governors.affirmative) |
        verify_spans($text; .prompt; .governors.excluded) |
        (sorted_governors(.bytes)) as [$g,$h] |
        (sorted_governors(.tokens|{start:.[0],end:.[1]})) as [$i,$j] |
        require($g.end<=$h.start and $i.end<=$j.start; "overlapping governor coordinates") |
        {family,distance,pair_id,original_text:split_words($text;$g;$h),
         original_tokens:split_words(.prompt;$i;$j),
         governors:[($text|span($g)),($text|span($h))],
         governor_ids:[(.prompt|span($i)),(.prompt|span($j))]}) |
    require(.[0].original_text==.[1].original_text and .[0].original_tokens==.[1].original_tokens;
        "opposed prompts change outside original governor addresses") |
    . as $pair |
    require(all(range(2); . as $index |
        $pair[0].governors[$index]!=$pair[1].governors[$index] and
        $pair[0].governor_ids[$index]!=$pair[1].governor_ids[$index]);
        "an opposed governor did not change") |
    .[0] | {family,distance,pair_id}
)) as $pairs |
require(($pairs|length)==128 and ($pairs|map(.family)|unique|length)==4 and
    all($pairs|group_by(.distance)[];length==64); "unexpected held panel distribution") |
($rows | group_by(.distance) | map(
  {key:.[0].distance,value:(map(.governors.affirmative.gap,.governors.excluded.gap) |
     {min:min,max:max})}) | from_entries) as $gaps |
require($gaps.near.max<$gaps.far.min;"near/far token distances overlap") |
{pass:true,endpoints:256,pairs:128,families:4,absolute_governor_gaps:$gaps,bytes_and_bpe_fixed_outside_swapped_governors:true}
