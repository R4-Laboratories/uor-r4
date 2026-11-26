def require($test;$why): if $test then . else error($why) end;
def tally: sort | group_by(.) | map({key:.[0],value:length}) | from_entries;
def sig($s): [$s.local,$s.joint,$s.native.choice,$s.native.governor_correct];
def row_sig: [sig(.before),sig(.after)];
def source($s): [$s.native.signed_site.bytes,$s.native.signed_site.tokens];
def paired($r):
    ($r|sort_by(.selector)) as [$f,$l] |
    require(($r|length)==2 and $f.selector=="first" and $l.selector=="later" and
        $f.sign==$l.sign and $f.family==$l.family and $f.fact==$l.fact and
        $f.historical_lost==$l.historical_lost and $f.governor==$l.governor and
        $f.requested==$l.rival and $l.requested==$f.rival;"pair was not a fixed-address selector edit") |
    def observation($a;$b):
        {correct:($a.native.choice=="requested" and $b.native.choice=="requested"),
         consensus_correct:($a.native.consensus_choice=="requested" and $b.native.consensus_choice=="requested"),
         with_governor:($a.native.choice=="requested" and $b.native.choice=="requested" and
                         $a.native.governor_correct==true and $b.native.governor_correct==true),
         distinct:($a.native.signed_site!=null and $b.native.signed_site!=null and source($a)!=source($b))};
    {pair_id:$f.pair_id,role:$f.role,sign:$f.sign,family:$f.family,fact:$f.fact,historical_lost:$f.historical_lost,
     rows:[$f.row_index,$l.row_index],before:observation($f.before;$l.before),after:observation($f.after;$l.after),
     signature:[($f|row_sig),($l|row_sig)]};
def matched($array):
    $array|map(select(.selector!=null)) | group_by(.pair_id) | map(paired(.));
def counts($p): {pairs:($p|length),
    before_correct:([$p[]|select(.before.correct)]|length),after_correct:([$p[]|select(.after.correct)]|length),
    before_with_governor:([$p[]|select(.before.with_governor)]|length),after_with_governor:([$p[]|select(.after.with_governor)]|length),
    before_consensus_correct:([$p[]|select(.before.consensus_correct)]|length),after_consensus_correct:([$p[]|select(.after.consensus_correct)]|length),
    before_distinct:([$p[]|select(.before.distinct)]|length),after_distinct:([$p[]|select(.after.distinct)]|length),
    lost:([$p[]|select(.before.correct and (.after.correct|not))]|length),
    retained:([$p[]|select(.before.correct and .after.correct)]|length),
    gained:([$p[]|select((.before.correct|not) and .after.correct)]|length)};
def byrow($r):
    {requests:($r|length),rows:([$r[].row_index]|unique|length),families:([$r[].family]|unique),
     before:{stage:([$r[].before.stage]|tally),local:([$r[].before.local]|tally),joint:([$r[].before.joint]|tally),native:([$r[].before.native.choice]|tally)},
     after:{stage:([$r[].after.stage]|tally),local:([$r[].after.local]|tally),joint:([$r[].after.joint]|tally),native:([$r[].after.native.choice]|tally)}};
def recurrence($r;$target):
    [$r[]|. as $x|select(any($target[]; row_sig==($x|row_sig)))] as $matching |
    [$matching[]|. as $x|select(any($target[]; .sign==$x.sign and row_sig==($x|row_sig)))] as $signed |
    {requests:($r|length),matching_requests:($matching|length),matching_unique_rows:([$matching[].row_index]|unique|length),
     matching_requests_same_sign:($signed|length),examples:([$matching[:4][]|{row_index,request,family,role,requested,rival,signature:row_sig}])};
matched($historical) as $hp | matched($original) as $op | matched($derived) as $dp |
[$hp[]|select(.historical_lost)] as $losses |
[$historical[]|select(.historical_lost)] as $target |
require(($target|length)==8 and ($losses|length)==4 and all($losses[];.before.correct and (.after.correct|not)) and
        ($hp|length)==32 and ([$hp[]|select(.before.correct and (.after.correct|not))]|length)==4;
        "unrestricted native read failed to reproduce the four historical switches") |
[$dp[]|select(.before.correct and (.after.correct|not))] as $dl |
[$dl[]|. as $x|select(any($losses[];.signature==$x.signature))] as $same |
{historical:{all:counts($hp),lost_four:$losses,rows:byrow($target)},
 original_train:{enrolled_prompts:$original_prompts,by_row:byrow($original),matched_selectors:counts($op),
                 row_signature_recurrence:recurrence($original;$target),
                 by_rival_position:($original|group_by(.rival.index==.heading.index)|map({
                     rival_is_original_heading:(.[0].rival.index==.[0].heading.index),
                     before_after:byrow(.),row_signature_recurrence:recurrence(.;$target)}))},
 train_derived:{enrolled_prompts:$derived_prompts,all:counts($dp),rows:byrow($derived),
      by_family:($dp|group_by(.family)|map({family:.[0].family,counts:counts(.)})),
      paired_losses:$dl,matching_full_pair_signature:$same,
      matching_full_pair_signature_same_sign:[$same[]|. as $x|select(any($losses[]; .sign==$x.sign and .signature==$x.signature))],
      row_signature_recurrence:recurrence($derived;$target)},
 witnesses:[$target[]|{pair_id,selector,role,request,row_index,signature:row_sig}]}
