#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive fixture version}
[[ $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs && $here == "$study/"* ]]
out="$study/grade-fixture-v$version"
partial="$out.partial.$$"
[[ ! -e $out && ! -e $partial ]]
mkdir -p "$partial/methods" "$partial/release"
cp "$here/"{fixture-wick.sh,loss-cases.jq,loss-summary.jq,release.jq,source-scores.jq} "$partial/methods/"
cd "$partial"

# All addresses, costs, IDs, and text below are fixture values.
jq -nc '
  def span($start;$end): {start:$start,end:$end};
  def occurrence($start;$end;$token):
    {bytes:span($start;$end),tokens:span($token;$token+1)};
  def native: .bytes.start += 1 | .bytes.end += 1;
  def verb($word;$noun):
    {bytes:$word.bytes,tokens:[$word.tokens.start,$word.tokens.end],
     noun:$noun.bytes,native_noun:$noun.tokens};
  [occurrence(0;6;10),occurrence(21;27;13)] as $verbs |
  [occurrence(7;13;11),occurrence(14;20;12),occurrence(28;34;14),occurrence(35;39;15)] as $sites |
  [range(256) as $i |
   {case_id:("case-"+($i|tostring)), digest:("fixture-prompt-"+($i|tostring)),
    pair_id:("pair-"+(($i/2|floor)|tostring)), family:("family-"+(($i/32|floor)|tostring)),
    admission:(if $i%2==0 then "defer" else "take" end),distance:"far",
    turns:["Hello.", "accept ballot ballot reject docket path "+($i|tostring)],
    governors:{affirmative:verb($verbs[0];$sites[0]), excluded:verb($verbs[1];$sites[2])},
    native_scope:[$sites[0],$sites[2],$sites[3]] | map(native),
    neutral_rival_bytes:$sites[1].bytes,neutral_rival:($sites[1]|native),
    words:$verbs,sites:$sites, ordinal:$i}]
' > fixture.json
jq -c 'map(.structural_prompt_sha256 = .digest | del(.digest,.words,.sites,.ordinal))' fixture.json > reference.json
jq -c '.[] | {id:.case_id,prompt_sha256:.digest,ordinal,words,sites}' fixture.json > coordinates.jsonl
jq -c '
  def address: {bytes:[.bytes.start,.bytes.end],tokens:[.tokens.start,.tokens.end]};
  def point($cost): {margin:($cost-3),cost:$cost,center:0};
  def outcome($m): if $m<0 then "loss" elif $m==0 then "tie" else "win" end;
  def minimum($field):
    sort_by(.[$field].cost,.site) | .[0] as $pivot |
    [ .[] | select(.[$field].cost == $pivot[$field].cost) | .site ] as $tied |
    $pivot[$field].margin as $m |
    {rival_site:$pivot.site,rival_ties:($tied|length),rivals:$tied,gold_cost:3,rival_cost:$pivot[$field].cost,
     margin:$m,loss:([0,1-$m]|max),outcome:outcome($m)};
  .[] as $r | range(2) as $sign |
  if $sign==0 and $r.ordinal==0 then [[2,4,7],[4,2,7]]
  elif $sign==0 and $r.ordinal==1 then [[2,4,7],[4,3,7]]
  elif $sign==0 and $r.ordinal==2 then [[2,2,7],[5,4,7]]
  elif $sign==0 and $r.ordinal==3 then [[5,6,7],[2,5,7]]
  elif $sign==0 and $r.ordinal==4 then [[5,6,7],[3,5,7]]
  elif $sign==1 and ($r.ordinal==6 or $r.ordinal==8) then [[6,4,2],[6,4,5]]
  else [[5,6,7],[5,6,7]] end as $costs |
  (if $sign==0 then 0 else 2 end) as $gold |
  [$r.sites[1],$r.sites[3],$r.sites[2-$gold]] as $wrong |
  [1,3,2-$gold] as $site_ids |
  [range(3) as $i | {site:$site_ids[$i],address:($wrong[$i]|address),
     before:point($costs[0][$i]),after:point($costs[1][$i])}] as $candidates |
  ($candidates|minimum("before")) as $old |
  ($candidates|minimum("after")) as $new |
  ($r.ordinal!=8 or $sign!=1) as $admitted |
  {row:$r.ordinal, sign:$sign, governor_word:$sign, gold_site:$gold,
   governor:($r.words[$sign]|address), gold:($r.sites[$gold]|address),
   eligible_for_fit:$admitted, governor_admitted:$admitted,gold_in_slot:true, eligible_rivals:3,
   before:$old,after:$new,candidates:$candidates,
   initial_rivals_after:[$candidates[] | select(.before.cost==$old.rival_cost)],
   chosen_step:2,flip:($old.outcome+"_to_"+$new.outcome)}
' fixture.json > raw.jsonl

join() {
    jq -nce --slurpfile reference reference.json --slurpfile coords coordinates.jsonl \
        -f methods/loss-cases.jq "$@"
}
summarize() {
    jq -es -f methods/loss-summary.jq "$@"
}
join raw.jsonl > edges.jsonl
summarize edges.jsonl > loss.json
jq -es '
  def row($r;$s): map(select(.row==$r and .sign==$s)) | first;
  length == 512 and
  (row(0;0) | .transition=="loss_to_loss" and .old_rival.word=="ballot" and
    .old_rival.bytes==[14,20] and .old_rival.tokens==[12,13] and .gold.bytes==[7,13] and
    .new_rival.word=="path" and .new_rival.bytes==[35,39] and .new_rival.tokens==[15,16] and
    .old_kind=="neutral_same_spelling_occurrence" and .new_kind=="other_word" and
    .beats_all_old_rivals and .old_pivot_after==1 and .new_margin == -1 and (.current_pivot_equals_old|not)) and
  (row(1;0) | .transition=="loss_to_tie" and .beats_all_old_rivals and
    .old_pivot_after==1 and .new_margin==0 and .new_rival.word=="path") and
  (row(2;0) | .transition=="loss_to_win" and .old_tie_count==2 and
    .beats_all_old_rivals and .new_margin==1) and
  (row(6;1) | .transition=="loss_to_win" and .old_kind=="other_signed_noun") and
  (row(8;1) | .transition=="loss_to_win" and (.eligible|not))
' edges.jsonl > joined.ok
jq -e '
  .events==512 and .eligible==511 and .unadmitted==1 and .eligible_losses==4 and .loss_to_win==2 and
  .before_outcomes=={loss:4,win:507} and .after_outcomes=={loss:2,tie:2,win:507} and
  .transitions=={loss:{loss:1,tie:1,win:2},tie:{loss:0,tie:0,win:0},win:{loss:1,tie:1,win:505}} and
  .strict_losses_that_beat_all_original==4 and .beat_old_but_still_wrong==1 and .beat_old_but_now_tied==1 and
  .initial_minimum_ties==1 and .all_four_signed_edges_won_before==124 and .all_four_signed_edges_won_after==124 and
  .by_sign.affirmed.eligible_losses==3 and .by_sign.excluded.eligible_losses==1 and
  .original_loss_categories=={neutral_same_spelling_occurrence:{loss:1,tie:1,win:1},other_signed_noun:{win:1}}
' loss.json > loss.ok

if jq -c 'if .row==0 and .sign==0 then .governor.bytes[0] += 1 else . end' raw.jsonl |
    join > changed-address.stdout 2> changed-address.stderr; then
    echo 'Altered original governor address passed the join' >&2
    exit 1
fi
rg -Fq 'unauthenticated governor or target address' changed-address.stderr
if head -n 511 edges.jsonl | summarize > incomplete-census.stdout 2> incomplete-census.stderr; then
    echo 'Incomplete signed edge census passed the summary' >&2
    exit 1
fi
rg -Fq 'signed edge census incomplete' incomplete-census.stderr

jq -c 'if .eligible and .before=="loss" then
  .before="tie" | .old_margin=0 | .old_hinge=1 | .transition=("tie_to_"+.after)
  else . end' edges.jsonl | summarize > no-genuine-loss.json
jq -e '.eligible_losses==0 and .loss_to_win==0 and .unadmitted==1' no-genuine-loss.json > no-genuine-loss.ok
jq -c 'if .eligible and .transition=="loss_to_win" then
  .after="tie" | .new_margin=0 | .new_hinge=1 | .transition="loss_to_tie"
  else . end' edges.jsonl | summarize > no-loss-flip.json
jq -e '.eligible_losses==4 and .loss_to_win==0' no-loss-flip.json > no-loss-flip.ok

source_fixture() {
    jq -nc --arg test "$1" '
      def winner($exact;$full;$earlier):
        {exact:$exact,full:$full,earlier:$earlier,both_governors:true,neutral:false,swapped:false,
         ties:1,eligible:1,first_source_error:(if $exact then "none" else "scope" end)};
      range(128) as $pair | range(2) as $side | ($pair%64) as $i | ($pair<64) as $far |
      (if $test=="far_gain_flat" then (if $far then 4 else 14 end)
       elif $test=="near_retention_drop" then (if $far then 13 else 5 end)
       elif ($test=="inherited_seven" or $test=="one_sided") and $far then 11
       elif $far then 12 else 6 end) as $limit |
      ($i<$limit or ($test=="one_sided" and $far and $i>=11 and $i<19 and $side==0)) as $exact |
      ($exact and (if ($test=="missing_far_full" and $far) or ($test=="missing_near_full" and ($far|not))
                  then $side==0 else true end)) as $full |
      ($i < (if $test=="native_seven" then (if $far then 6 else 5 end)
             elif $test=="control_eight" then 5
             elif $test=="far_gain_flat" and $far then 3 else 4 end)) as $native |
      ($i < (if $test=="unfitted_seven" then (if $far then 6 else 5 end)
             elif $test=="control_eight" then 5
             elif $test=="far_gain_flat" and $far then 3 else 4 end)) as $unfitted |
      ($i<(if $far then 4 else 6 end)) as $baseline |
      (($test!="missing_far_correct" or ($far|not)) and ($test!="missing_near_correct" or $far)) as $correct |
      ($test=="one_exchanged_exact" and $pair==0 and $side==0) as $exchanged |
      {case_id:("source-case-"+(($pair*2+$side)|tostring)),pair_id:("pair-"+($pair|tostring)),
       admission:(if $side==0 then "defer" else "take" end),
       distance:(if $far then "far" else "near" end),family:("family-"+(($i/16|floor)|tostring)),
       arms:{baseline:winner($baseline;false;$baseline),
         learned:winner($exact;$full;$exact),native:winner($native;false;$native),
         unfitted:winner($unfitted;false;$unfitted),positive:winner(false;false;false),
         governor_gap:winner(false;false;false),correct_link:winner($correct;$correct;true),
         exchanged_link:winner($exchanged;false;false)}}
    '
}
release_case() {
    local name=$1 expected_failure=$2 training=${3:-loss.json}
    source_fixture "$name" > "release/$name.endpoints.jsonl"
    jq -es --arg split held -f methods/source-scores.jq "release/$name.endpoints.jsonl" > "release/$name.scores.json"
    jq -e --slurpfile training "$training" --arg reference_sha synthetic-reference --arg writer_sha synthetic-writer \
        --arg source_sha synthetic-raw -f methods/release.jq "release/$name.scores.json" > "release/$name.json"
    jq -e --arg gate "$expected_failure" '
      if $gate=="" then .pass and all(.gate[];.)
      else (.pass|not) and ([.gate | to_entries[] | select(.value|not) | .key] == [$gate])
      end
    ' "release/$name.json" > "release/$name.ok"
}
release_case threshold_eight ''
release_case control_eight ''
release_case inherited_seven gain_over_inherited
release_case native_seven net_over_native_and_unfitted
release_case unfitted_seven net_over_native_and_unfitted
release_case one_sided gain_over_inherited
release_case far_gain_flat far_gain_over_inherited
release_case near_retention_drop near_retention
release_case missing_far_full full_links_at_both_distances
release_case missing_near_full full_links_at_both_distances
release_case missing_far_correct supplied_controls
release_case missing_near_correct supplied_controls
release_case one_exchanged_exact supplied_controls
release_case no_genuine_loss genuine_training_loss_flip no-genuine-loss.json
release_case no_loss_flip genuine_training_loss_flip no-loss-flip.json

jq -e '.arms.learned.exact_pairs-.arms.baseline.exact_pairs==8 and
      .by_distance.near.learned.exact_pairs==.by_distance.near.baseline.exact_pairs' \
    release/threshold_eight.json > threshold.ok
jq -e '.arms.learned.exact_pairs-.arms.native.exact_pairs==8 and
      .arms.learned.exact_pairs-.arms.unfitted.exact_pairs==8' release/control_eight.json > control-threshold.ok
jq -e '.arms.learned.exact_pairs-.arms.baseline.exact_pairs==7 and
      .arms.learned.exact_rows-.arms.baseline.exact_rows>=16' release/one_sided.json > one-sided.ok
jq -e '.arms.correct_link.earlier_pairs==128 and .arms.correct_link.exact_pairs==64 and
      .by_distance.far.correct_link.exact_pairs==0' release/missing_far_correct.json > far-control.ok
jq -e '.arms.correct_link.earlier_pairs==128 and .arms.correct_link.exact_pairs==64 and
      .by_distance.near.correct_link.exact_pairs==0' release/missing_near_correct.json > near-control.ok
jq -e '.arms.exchanged_link.exact_pairs==0 and .arms.exchanged_link.exact_rows==1' \
    release/one_exchanged_exact.json > exchange-control.ok

jq -n --slurpfile loss loss.json --argjson raw_count "$(wc -l < raw.jsonl)" \
  --argjson joined_count "$(wc -l < edges.jsonl)" '[inputs | {case:input_filename,pass,gate}] as $cases |
  {synthetic:true,raw_signed_events:$raw_count,joined_signed_events:$joined_count,
   training_transition_counts:$loss[0].transitions,release_cases:($cases|length),
   passing:([$cases[] | select(.pass)]|length),cases:$cases}
' release/{threshold_eight,control_eight,inherited_seven,native_seven,unfitted_seven,one_sided,far_gain_flat,near_retention_drop,missing_far_full,missing_near_full,missing_far_correct,missing_near_correct,one_exchanged_exact,no_genuine_loss,no_loss_flip}.json > fixture-receipt.json
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum --strict --quiet -c SHA256SUMS
chmod -R a-w .
cd "$study"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,fixture-receipt.json,loss.json}
