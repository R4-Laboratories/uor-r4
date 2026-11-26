def need($ok; $why): if $ok then . else error($why) end;
def cost_order:
  {"requested_better":"win","rival_better":"loss","equal":"tie",
   "both_missing":"neither","requested_missing":"requested-ineligible","rival_missing":"rival-ineligible"}[.];
def source_choice:
  {"requested":"requested","rival":"same-word rival","third":"third site","tied":"unresolved","absent":"unresolved"}[.];
def equivalent($audit; $report; $sign):
  ($audit.local|cost_order)==$report.local and ($audit.joint|cost_order)==$report.joint and
  ($audit.native|source_choice)==$report.native.choice and
  $audit.native_site==$report.native.signed_site.index and
  $audit.native_consensus_site==$report.native.consensus_signed_site.index and
  $audit.native_governor_correct==$report.native.governor_correct and
  $audit.native_min_sites==$report.native.min_sites[$sign] and
  $audit.requested_local_cost==$report.requested_local.cost and $audit.rival_local_cost==$report.rival_local.cost and
  $audit.requested_joint_cost==$report.requested_tuple.cost and $audit.rival_joint_cost==$report.rival_tuple.cost and
  $audit.requested_ties==$report.requested_tuple.ties and $audit.rival_ties==$report.rival_tuple.ties;
def checked($audit; $report):
  $audit.request==$report.request and $audit.row==$report.row_index and
  $audit.case_id==$report.case_id and $audit.pair_id==$report.pair_id and
  $audit.sign==$report.sign and $audit.family==$report.family and $audit.selector==$report.selector and
  $audit.historical_lost==$report.historical_lost;
$old[0].classifications as $baseline | $new[0].classifications as $refit |
need(($baseline|length)==($refit|length) and ($baseline|length)==($grade|length);"graded request count changed") |
range(0;($grade|length)) as $i | $grade[$i] as $scored |
(if $scored.sign==1 then 0 else 1 end) as $sign |
need(checked($baseline[$i];$scored) and checked($refit[$i];$scored) and
  equivalent($baseline[$i];$scored.before;$sign) and equivalent($refit[$i];$scored.after;$sign);
  "grade disagrees with independently replayed signed cost or original source at request "+($i|tostring)) |
true
