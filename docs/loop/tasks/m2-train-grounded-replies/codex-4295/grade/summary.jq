def insist($ok;$why): if $ok then . else error($why) end;
def hist(f): group_by(f)|map({value:(.[0]|f),n:length});
def selected: {joint_selected,joint_governors,joint_links,joint_ties};
def counts:
  . as $x | group_by(.pair_id) as $pairs |
  {prompts:length,exact_prompts:map(select(.full))|length,exact_tuples:map(select(.exact_tuple))|length,
   pairs:($pairs|length),complete_pairs:([$pairs[]|select(all(.[];.full))]|length),
   one_side:([$pairs[]|select((map(select(.full))|length)==1)]|length),
   all_governors:map(select(.governors_correct))|length,correct_headings:map(select(.heading_correct))|length,
   gold_ranks:hist(.gold_rank),gold_tie_counts:hist(.gold_ties),
   gold_unique_first:map(select(.gold_rank==1 and .gold_ties==1))|length,
   gold_tied_first:map(select(.gold_rank==1 and .gold_ties>1))|length,
   gold_below_first:map(select(.gold_rank>1))|length,joint_ties:map(select(.joint_ties>1))|length,
   first_source_errors:hist(.first_source_error),
   failed_role_joint_surfaces:hist(if .joint_selected==null then null else .joint_selected[.failed_sign].word end),
   strongest_remaining_wrong_surfaces:hist(.strongest_remaining_wrong|map(.word)),
   matched_controls:map(select(.matched_control!=null))|length};
def public_arm:
  {gold_rank,gold_ties,signed_winners,strongest_remaining_wrong,joint_selected,joint_governors,
   joint_ties,heading_correct,governors_correct,full,first_source_error};
[$neutral,$winner,$control] as $arms |
insist(all($arms[]; length==48 and (group_by(.pair_id)|length)==24 and
  all(group_by(.pair_id)[];length==2 and (map(.admission)|sort)==["defer","take"])) and
  ($arms|map(map(.row))|unique|length)==1 and ($arms|map(map(.case_id))|unique|length)==1 and
  ($neutral|map(.pair_id)|unique)==($residual[0].residual_pair_ids|sort);
  "expected exactly the authenticated 24 opposed pairs and 48 same rows in each arm") |
[range(48) as $i|$neutral[$i] as $a|$winner[$i] as $b|$control[$i] as $c|
 insist(($a.full|not) and $a.governors_correct and $a.heading_correct and $a.joint_ties==1 and
   $a.joint_selected[$a.failed_sign].site==$a.old_winner.site and ($a.signed_winners|length)==1 and
   $a.signed_winners[0].site==$a.old_winner.site and $a.strongest_remaining_wrong[0].site==$a.old_winner.site and
   $b.omitted==$a.old_winner and $c.omitted==$a.matched_control and
   (($a|selected)==($c|selected)) and $a.gold==$b.gold and $a.gold==$c.gold;
   "the previous source or matched nonwinner control changed") |
 {row:$a.row,case_id:$a.case_id,pair_id:$a.pair_id,admission:$a.admission,family:$a.family,failed_sign:$a.failed_sign,
  governor:$a.governor,gold:$a.gold,old_winner:$a.old_winner,matched_control:$a.matched_control,
  neutral:($a|public_arm),winner:($b|public_arm),control:($c|public_arm)}] as $rows |
{arm:{neutral:($neutral|counts),winner:($winner|counts),control:($control|counts)},
 by_family:($rows|group_by(.family)|map(. as $g|{family:.[0].family,pairs:(group_by(.pair_id)|length),
  winning_omission_exact_prompts:map(select(.winner.full))|length,winning_omission_full_pairs:
    ([group_by(.pair_id)[]|select(all(.[];.winner.full))]|length),
  gold_rank_changes:hist([.neutral.gold_rank,.winner.gold_rank]),original_winner:hist(.old_winner.word),
  new_signed_winners:hist(.winner.signed_winners|map(.word)),
  new_joint_winner:hist(if .winner.joint_selected==null then null else .winner.joint_selected[.failed_sign].word end)})),
 identities:($rows|group_by(.pair_id)|map({pair_id:.[0].pair_id,matched_both:all(.[];.matched_control!=null),
  neutral:all(.[];.neutral.full),winner:all(.[];.winner.full),control:all(.[];.control.full)})),
 matched_pairs:([$rows|group_by(.pair_id)[]|select(all(.[];.matched_control!=null))]|length),
 recovered_matched_pairs:([$rows|group_by(.pair_id)[]|select(all(.[];.matched_control!=null and .winner.full and (.control.full|not)))]|length),
 remaining_when_winner_omitted:($rows|map(select(.winner.full|not))|
  hist(if .winner.gold_rank>1 then "other signed occurrence outranks gold"
       elif .winner.gold_ties>1 then "gold tied at lowest signed cost"
       else "gold unique at fixed governor, joint read still incomplete" end)),
 per_prompt:$rows}
