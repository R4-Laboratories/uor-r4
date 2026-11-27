# Cohorts have been sealed and the per-model summaries independently derived from wire token IDs.
def decide($fresh;$oldcomparisons):
($fresh|map(.crossed.contexts.exact))as $q |
($fresh|map(.crossed.equal_length_contexts.exact))as $eq |
($fresh[3].crossed_families|map(.family as $family |.stats.contexts.exact as $j |
 select(all($fresh[:3][]; (.crossed_families[]|select(.family==$family)|.stats.contexts.exact)<$j))|.family)) as $families |
([$oldcomparisons[]|.pair_identities[]|select(.to=="joint" and (.cohort=="all" or .cohort=="selector_edge"))|(.lost|length)]|add)as $loss |
{fresh_quartets:$q,fresh_equal_length_quartets:$eq,improved_families:$families,
 more_quartets:($q[3]>($q[:3]|max)),equal_length_gain:($eq[3]>($eq[:3]|max)),multiple_families:(($families|length)>=2),
 retained_all:($loss==0),prior_pair_identity_losses_summed_by_model_and_split:$loss,
 direct_lengths_preserved:(all($fresh[3].direct_by_length[]; . as $j | all($fresh[:2][].direct_by_length[]|select(.contrast==$j.contrast); .stats.exact_short<=$j.stats.exact_short and .stats.exact_long<=$j.stats.exact_long))),
 fresh_sides_preserved:(all(["scoped","direct"][]; . as $slice | $fresh[3][$slice] as $j | all($fresh[:2][]; .[$slice].exact_short<=$j.exact_short and .[$slice].exact_long<=$j.exact_long)))} |
 .+{joint_advantage:(.more_quartets and .equal_length_gain and .multiple_families)} |
 .+{advance_to_source_discovery:(.joint_advantage and .retained_all and .fresh_sides_preserved and .direct_lengths_preserved)};
