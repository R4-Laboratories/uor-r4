# Integer readback of the existing NATURAL.geometry.actual table. Equal
# relative terms certify invariance for any weights in this table layout.
def insist($ok; $why): if $ok then . else error($why) end;
def sign: if .>0 then "positive" elif .<0 then "negative" else "zero" end;
def word_score($choice;$bank_start;$bank_end):
  [$choice.keys[$bank_start:$bank_end][] | $weights[0][.]] | add;
def valid_keys($keys):
  ($keys|length)==16 and all(range(0;16) as $bank | $keys[$bank];
    type=="number" and .==floor) and
  all(range(0;16) as $bank | [$keys[$bank],$bank]; .[0]>=256*.[1] and .[0]<256*(.[1]+1));
def terms($own;$mate):
  [range(0;16) as $bank | if $own.keys[$bank]==$mate.keys[$bank] then null
    else [$own.keys[$bank],$mate.keys[$bank]] end];
def same_term($one;$two): $one==(if $two==null then null else $two|reverse end);

def raw($cold):
  [{span:null,keys:$cold.no_read_keys}] +
  [$cold.candidates[] | {span:{start,end},keys}] |
  insist(all(.[]; valid_keys(.keys)); "invalid original native word address") |
  map(. as $choice | . + {shared:word_score($choice;0;8),route:word_score($choice;8;16)})
  | map(.+{score:(.shared+.route)});

def reconstruct($cold;$gpu):
  raw($cold) as $choices |
  ($choices|map(.score)) as $scores |
  ($scores|max) as $max |
  ($scores|index($max)) as $picked |
  insist($scores==([$cold.scores.selection.no_read_score]+$cold.scores.candidates) and
         $cold.scores.selection.span==$choices[$picked].span and
         $cold.scores.selection.best_candidate_score==($scores[1:]|max) and
         ($gpu|length)==(1+($choices|length)) and $gpu==([$picked]+$scores);
         "CUDA/cold/full sixteen-bank reconstruction diverged for ($cold.case_id)") |
  $choices;

def pairs:
  if $panel=="validation" then
    $roster[0].titles[] as $title | $title.selected[] | select(.qualification.stage=="qualified") |
    {title:$title.name,family:$title.family,cases:.qualification.opening.cases}
  elif $panel=="train" then
    $roster[0].training[] | select(.mixture=="natural") | .sources[] | select((.openings|length)==2) |
    [.openings[].cases[0]] as $cases |
    {title,family:(if all($cases[];.target.kind=="source") then "two_answer" else "answerability" end),$cases}
  else error("unregistered panel")
  end;

def side($own;$other;$rows;$prior):
  $rows[$own.id] as $choices | ($own.target.span//null) as $target | ($other.target.span//null) as $mate |
  ([$choices[] | select(.span==$target)] | insist(length==1;"absent target") | .[0]) as $a |
  ([$choices[] | select(.span==$mate)] | insist(length==1;"absent mate") | .[0]) as $b |
  ($choices|map(select(.span!=$target))) as $rivals |
  ($rivals|map(.score)|max) as $best |
  ($rivals|map(select(.score==$best))) as $best_rivals |
  {id:$own.id, own_kind:$own.target.kind, full_margin:($a.score-$best),
   mate_margin:($a.score-$b.score), shared_margin:($a.shared-$b.shared),
   route_margin:($a.route-$b.route), term:terms($a;$b),
   best_rival_kinds:($best_rivals|map(if .span==$mate then "mate" elif .span==null then "other_no_read" else "other_interval" end)|unique),
   best_rival_count:($best_rivals|length)} |
  insist(.full_margin==$prior[$own.id].margin_over_all and .mate_margin==$prior[$own.id].margin_over_mate;
         "per-case margin differs from pre-existing readback");

def family_summary:
  def n(f): map(select(f))|length;
  def signs(f): group_by(f|sign)|map({sign:(.[0]|f|sign),pairs:length});
  def bank_signs($bank):
    {bank:$bank,changed_pairs:n(.changed_banks|index($bank)!=null),
     question_effect:signs(.bank_effects[$bank])};
  group_by(.family)|map(
    {family:.[0].family,pairs:length,sides:(2*length),strict_vs_mate:n(.strict_vs_mate),strict_vs_all:n(.strict_vs_all),
     lost_to_other_or_tie:n(.strict_vs_mate and (.strict_vs_all|not)),
     targets_same_relative_keys:n((.changed_banks|length)==0),
     shared_same_relative_keys:n((.changed_shared|length)==0),
     route_same_relative_keys:n((.changed_route|length)==0),
     route_changed_but_tied:n((.changed_route|length)>0 and .route_effect==0),
     full_effect_sign:signs(.effect),shared_effect_sign:signs(.shared_effect),route_effect_sign:signs(.route_effect),
     both_positive_shared_only:n(all(.sides[];.shared_margin>0)),
     both_positive_route_only:n(all(.sides[];.route_margin>0)),
     side_routes:([.[].sides[]]|group_by(.own_kind)|map(
          {target:.[0].own_kind,n:length,complete_own_mate_aliases:n(all(.term[];.==null)),
           wrong_vs_mate:n(.mate_margin<0),tied_vs_mate:n(.mate_margin==0),
           mate_defeated_but_other_interval:n(.mate_margin>0 and .full_margin<=0 and (.best_rival_kinds|index("other_interval")!=null)),
           mate_defeated_but_other_noread:n(.mate_margin>0 and .full_margin<=0 and (.best_rival_kinds|index("other_no_read")!=null)),
           mate_defeated_but_tied_for_win:n(.mate_margin>0 and .full_margin==0)})),
        banks:[range(0;16) as $bank | bank_signs($bank)]}
  );

insist(($weights|length)==1 and ($weights[0]|length)==4096; "wrong frozen i16 payload") |
([$cold[] | select(.arm=="geometry")]) as $geometry |
insist(($geometry|length)*2==($cold|length) and all($geometry[]; .scores!=null)
  and all($cold[]|select(.arm=="count"); .scores==null); "cold arms differ") |
(reduce $geometry[] as $row ({offset:0,rows:{},counts:{}};
   (2+($row.candidates|length)) as $width | .offset as $off |
   insist(.rows[$row.case_id]==null; "duplicate cold case") |
   .rows[$row.case_id]=reconstruct($row;$device[0][$off:($off+$width)]) |
   .offset+=$width)) as $reconstructed |
insist($reconstructed.offset==($device[0]|length); "trailing CUDA score words") |
($margins[0].rows|map(.sides[])|INDEX(.id)) as $prior |
[pairs | .cases as $cases | [side($cases[0];$cases[1];$reconstructed.rows;$prior),
                              side($cases[1];$cases[0];$reconstructed.rows;$prior)] as $rows |
  ([range(0;16) as $bank | select(same_term($rows[0].term[$bank];$rows[1].term[$bank])|not) | $bank]) as $changed |
  {title,family,sides:$rows,strict_vs_mate:all($rows[];.mate_margin>0),strict_vs_all:all($rows[];.full_margin>0),
   changed_banks:$changed,changed_shared:[$changed[]|select(.<8)],changed_route:[$changed[]|select(.>=8)],
   bank_effects:[range(0;16) as $bank | [$rows[].term[$bank] | if .==null then 0 else $weights[0][.[0]]-$weights[0][.[1]] end]|add],
   effect:($rows|map(.mate_margin)|add),shared_effect:($rows|map(.shared_margin)|add),route_effect:($rows|map(.route_margin)|add)} |
  insist(.effect==(.shared_effect+.route_effect) and .effect==(.bank_effects|add); "incorrect bank decomposition")] as $pairs |
insist(($prior|length)==2*($pairs|length); "missing pre-existing per-case margins") |
{panel:$panel,variant:"natural.geometry.actual",all_cases_rebuilt:($geometry|length),
 all_candidate_scores_rebuilt:($reconstructed.offset-($geometry|length)),paired_cases:(2*($pairs|length)),
 by_family:($pairs|family_summary),rows:$pairs}
