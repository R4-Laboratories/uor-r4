# Compact public evidence; raw prompts, human replies and token streams stay private.
def histogram: group_by(.) | map({value:.[0],n:length});
def cohort:
  {records,targets,learned_minus_control_brier,arms:(.counts|with_entries(.value |= (
    . + {mean_brier:(.brier_q32 / .n_targets / 4294967296)} |
    {n_targets,top1,mean_brier,brier_q32,prll_bits_q12,role_positive,admitted,active,top1_changed,gained,lost,
     present_targets,present_top1,present_brier_q32,absent_targets,absent_top1,absent_brier_q32,
     eos_targets,eos_top1,zero_weight,no_wrong_address,conditioned_pooled_weight_diff,conditioned_pooled_score_diff}
  )))};
$measurement[0] as $m | $paired[0] as $p | $grade[0] as $gr |
[$gen[] | select(.arm=="learned" and .domain=="conversation") as $g |
  ($held[]|select(.id==$g.id)) as $row |
  {gold:$g.gold[0], emitted:$g.tokens[0], proposed:$g.steps[0].pick_successor, source:$g.steps[0].s4_source.source_token,
   rival:$g.steps[0].s4_source.rival_token, in_prompt:($row.prefix|index($g.gold[0])!=null),
   transfer:$g.steps[0].s4_source.transfer_q48}] as $opening |
{schema:"uor-r4.s4-ordinary-output/1", device_source:"36bacaff778bc10af3c38c0831f2e66e36c2f3ff",
 weights:($m|{gate_sha256,state_sha256,artifact_sha256,train_identity_sha256,cubin_sha256,
              support:.complete_fit.support,weights:.complete_fit.weights}),
 inputs:($m|{fit_panel,held_panel,r_fit,d87_dev_tune}),
 timing_ms:$m.timing_ms, parity:$m.raw_ordinary_parity, generated_steps:$m.generated_steps,
 primary_gate_passed:$p.primary_gate_passed, teacher:($p.cohorts|map_values(cohort)),
 generation:$gr.by_domain_and_chat_band,
 alignment:([$gr.details[]|{band,arm,first_wrong:.first_wrong.token_number,first_difference_from_frozen:.first_diff_from_frozen.token_number}]
   |group_by([.band,.arm])|map({band:.[0].band,arm:.[0].arm,first_wrong_positions:(map(.first_wrong)|histogram),
      first_model_difference_positions:(map(.first_difference_from_frozen|select(.!=null))|histogram)})),
 opening_diagnostic:{rows:($opening|length),emitted_ids:($opening|map(.emitted)|histogram),
   correct_first:($opening|map(select(.emitted==.gold))|length),
   gold_in_prompt:($opening|map(select(.in_prompt))|length),
   proposed_gold:($opening|map(select(.proposed==.gold))|length),
   authenticated_gold:($opening|map(select(.source==.gold))|length),
   selected_rival_gold:($opening|map(select(.rival==.gold))|length),
   nonzero_transfers:($opening|map(select(.transfer>0))|length)},
 private_grade_sha256:$grade_sha, raw_manifest_sha256:$manifest}
