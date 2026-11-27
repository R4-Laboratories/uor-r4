# Run the frozen #4321 independent provenance audit first.
def ck($ok;$why): if $ok then . else error($why) end;
def key: [.pair_id,.side] | @json;
def n(f): map(select(f)) | length;
def exact($arm):
  . as $row | .arms[$arm] |
  .terminal and ([.steps[].committed_id] == $row.expected);
def live:
  . as $row | .contrast+2 as $index |
  .arms.learned as $run |
  ($run.steps[$index]) as $step |
  (([$run.steps[0:$index][].committed_id] == $row.expected[0:$index]) and $step!=null) as $reached |
  {reached:$reached,active:($reached and $step.stage==$row.contrast),
   selected:$step.selected_source,
   correct:($reached and $step.stage==$row.contrast and
            $step.committed_id==$row.expected[$index] and
            $step.selected_source==(if $row.side==0 then "selected_eos" else "verified_copy" end))};
def hist(f): group_by(f) | map({value:(.[0]|f),n:length});
def distribution:
  sort as $s | ($s|length) as $size |
  {n:$size,min:$s[0],median_low:$s[(($size-1)/2|floor)],median_high:$s[($size/2|floor)],max:$s[-1],counts:hist(.)};
def paired($keys):
  group_by(.pair_id) | map(sort_by(.side) | .[0] as $short | .[1] as $long |
    $keys[$short|key] as $a | $keys[$long|key] as $b |
    ck(length==2 and $short.side==0 and $long.side==1 and
       $a!=null and $b!=null and $a.contrast==$short.contrast and
       $b.contrast==$long.contrast and $a.context==$b.context and
       $a.variant==$b.variant and $a.directive!=$b.directive;"wording join") |
    {pair_id:$short.pair_id,contrast:$short.contrast,family:$short.family,
     context:$a.context,facts:$short.ids,wording:[$a.directive,$b.directive],
     variant:$a.variant,
     short_score:$short.common.own_score,long_score:$long.common.own_score,
     short_margin:-$short.common.own_score,long_margin:$long.common.own_score,
     directed:($short.common.own_score<0 and $long.common.own_score>0),
     inverted:($short.common.own_score>0 and $long.common.own_score<0),
     short_exact:($short|exact("learned")),long_exact:($long|exact("learned")),
     exact:all(.[];exact("learned")),swapped:all(.[];exact("swapped")),
     first_wrong:(map(. as $row | .arms.learned.first_mismatch as $mis |
       if $mis==null then null else
       {position:$mis,expected:$row.expected[$mis-1],committed:.arms.learned.steps[$mis-1].committed_id}
       end)),
     end_source:(map(.arms.learned |
       if .terminal then .steps[-1].selected_source else "open_cap" end)),
     actual_actions:(map(live))
    });
def summarize:
  {pairs:length,unique_fact_contexts:(map(.context)|unique|length),
   unique_ordered_fact_ids:(map(.facts)|unique|length),
   unique_instruction_pairs:(map(.wording)|unique|length),
   unique_unpunctuated_instruction_pairs:(map(.wording|map(rtrimstr(".")))|unique|length),
   punctuation:hist(.wording[0]|endswith(".")),
   unique_directives:(map(.wording[])|unique|length),
   directed:n(.directed),inverted:n(.inverted),exact:n(.exact),swapped:n(.swapped),
   short_exact:n(.short_exact),long_exact:n(.long_exact),
   selected_eos:(map(.end_source[])|n(.=="selected_eos")),
   native_eos:(map(.end_source[])|n(.=="native")),
   open_cap:(map(.end_source[])|n(.=="open_cap")),
   margin_pass_short:n(.short_margin>=8),margin_pass_long:n(.long_margin>=8),
   margin_pass_both:n(.short_margin>=8 and .long_margin>=8),
   directed_below_margin:n(.directed and (.short_margin<8 or .long_margin<8)),
   score_short:(map(.short_score)|distribution),score_long:(map(.long_score)|distribution),
   reached_both:n(all(.actual_actions[];.reached)),
   active_both:n(all(.actual_actions[];.active)),
   actual_both_directed:n(all(.actual_actions[];.correct)),
   actual_actions:hist(.actual_actions),
   paired_margins:hist([.short_margin,.long_margin]),
   wrong_short:(map(.first_wrong[0])|hist(.)),
   wrong_long:(map(.first_wrong[1])|hist(.))};
ck(($keys|length)==($rows|length);"key cardinality") |
(INDEX($keys[];key)) as $ref | ck(($ref|length)==($keys|length);"duplicate keys") |
($rows|paired($ref)) as $pairs |
{overall:($pairs|summarize),
 contrasts:($pairs|group_by(.contrast)|
   map({key:(.[0].contrast|tostring),value:summarize})|from_entries),
 wordings:($pairs|group_by([.contrast,.family,.variant,.wording])|
   map({contrast:.[0].contrast,family:.[0].family,variant:.[0].variant,
        wording:.[0].wording,counts:summarize})),
 pairs:$pairs}
