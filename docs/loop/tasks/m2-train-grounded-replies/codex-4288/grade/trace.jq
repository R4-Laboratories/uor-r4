def words($text;$x): [$x[] | if .==null then null else $text[.start:.end] end];
def witness($r;$p):
  $r.arms.learned as $a | $p.turns[1] as $t | if ($t|length)!=($t|utf8bytelength) then error("snapshot requires ASCII") else . end |
  {id:$r.case_id,pair:$r.pair_id,family:$r.family,distance:$r.distance,admission:$r.admission,
   error:$a.first_source_error,ties:$a.ties,eligible:$a.eligible,
   governors:words($t;[$p.governors.affirmative.bytes,$p.governors.excluded.bytes]),
   read_governors:words($t;($a.governors|map(.bytes))),
   gold:words($t;$r.gold),selected:words($t;($a.selection // [null,null,null])),
   gold_bpe:$r.gold_tokens,read_bpe:$a.tokens,
   neutral_bpe:$p.neutral_rival.tokens,score:$a.best,next_score:$a.rival};
($reference[0]|INDEX(.case_id)) as $idx |
[.[] | witness(.;$idx[.case_id])] as $rows |
{by_distance:($rows|group_by(.distance)|map({distance:.[0].distance,counts:(group_by(.error)|map({error:.[0].error,rows:length})),
     first_wrong_noun_surfaces:(map(select(.error=="affirmed_noun" or .error=="excluded_noun") |
          if .error=="affirmed_noun" then .selected[0] else .selected[1] end)|
          group_by(.)|map({term:.[0],rows:length})|sort_by(-.rows,.term)|.[:10])})),
 examples:($rows|group_by([.distance,.error])|map(sort_by(.id)|first))}
