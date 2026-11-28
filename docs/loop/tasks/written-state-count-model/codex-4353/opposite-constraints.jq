# jq -cn --slurpfile supplement supplement/held-ranks.jsonl --slurpfile dev dev/held-ranks.jsonl -f held-opposites.jq
def require($ok;$why): if $ok then . else error($why) end;
def count(f): map(select(f))|length;
def selected: .arm=="zero" or .arm=="geometry-435300" or .arm=="geometry-435301";
def degree($es;$v): [$es[]|select(.a==$v or .b==$v)]|length;
def numbers:
  map(select(.a < .b)) | group_by([.a,.b])
  | map(max_by(.incompatibility));
def greedy:
  . as $edges
  | sort_by((degree($edges;.a) + degree($edges;.b)),.a,.b)
  | reduce .[] as $e ({used:{},size:0};
      if .used[$e.a] or .used[$e.b] then .
      else .used[$e.a]=true | .used[$e.b]=true | .size+=1 end)
  | .size;
def constraints($pool):
  map(select(selected))
  | group_by(.id)
  | map(. as $trace
     | require((map(.arm)|sort)==["geometry-435300","geometry-435301","zero"]
          and all(.[]; .gold_lexical_id==$trace[0].gold_lexical_id
            and .score.gold_prior_q48==$trace[0].score.gold_prior_q48
            and .score.gold_addresses==$trace[0].score.gold_addresses);
          "missing trace or gold address drift")
     | [.[] | .score as $s | .gold_lexical_id as $gold
        | require(($s.rival>0 and $s.rival<4096 and $s.rival!=$gold)
          and ($s.gold_addresses|length)==4 and ($s.rival_addresses|length)==4
          and all(range(0;4); . as $i |
              all([$s.gold_addresses[$i],$s.rival_addresses[$i]][]; .>=64*$i and .<64*($i+1)
                 and floor==.)); "invalid original ID or address")
        | {row:($pool+"/"+.id),pool:$pool,rival:$s.rival,
           lower_q48:($s.rival_prior_q48-$s.gold_prior_q48+(if $s.rival<$gold then 1 else 0 end)),
           vector:[range(0;4) as $i | $s.gold_addresses[$i] as $g | $s.rival_addresses[$i] as $r
             | select($g!=$r)|[$g,$r]]}]
     | group_by(.rival)
     | map(require((. as $siblings | all(.[];.==$siblings[0]));"rival prior or address drift")|first)
   )|flatten(1)
   | map(. + {lower:(.lower_q48/4294967296|ceil),
      vector_key:(.vector|tojson),opposite:(.vector|map(reverse)|tojson)});
def compare($c):
  INDEX(($c|group_by(.vector_key))[];.[0].vector_key) as $by
  | [ $c[]|select((.vector|length)>0) | . as $a | ($by[.opposite] // [])[]
    | . as $b | select($a.row<$b.row and ($a.lower+$b.lower)>0)
    | {a:$a.row,b:$b.row,incompatibility:($a.lower+$b.lower),
       group:([$a.pool,$b.pool]|sort|unique|join("+"))} ]
  | numbers;
def summarize($rows;$c;$p):
  {rows:($rows|length),
   distinct_row_rival_constraints:($c|length),
   zero_vector_impossible_constraints:($c|count((.vector|length)==0 and .lower>0)),
   zero_vector_impossible_rows:($c|map(select((.vector|length)==0 and .lower>0)|.row)|unique|length),
   opposite_incompatible_unique_row_pairs:($p|length),
   max_certified_pair_conflict_words:($p|map(.incompatibility)|max),
   certified_disjoint_row_pairs:($p|greedy)};
($supplement|constraints("supplement")) as $s | ($dev|constraints("dev")) as $d
| ($s+$d) as $constraints
| compare($constraints) as $all
| {supplement:summarize(($supplement|map(.id)|unique);$s;($all|map(select(.group=="supplement")))),
   dev:summarize(($dev|map(.id)|unique);$d;($all|map(select(.group=="dev")))),
   cross_pool_unique_pairs:($all|count(.group=="dev+supplement")),
   combined:summarize(($constraints|map(.row)|unique);$constraints;$all)}
