# Cross-check every reconstructed difference with the observed four CUDA states.
def require(ok; why): if ok then . else error(why) end;
def coefficients($zero; $period; $after; $full):
 {frozen_gpu:$zero, period_component:($period - $zero), post_b_component:($after - $zero), full_gpu:$full};
($pairs|INDEX(.[];.pair_id)) as $by_pair |
[$geom[]|. as $g | $by_pair[$g.id] as $p |
 require($p != null and $p.axis==(if $g.source_axis=="archive" then "direct-count-control" else $g.source_axis end)
   and ([$p.members[] | .pair_id + ":" + (.side|tostring)] == [$g.stop_row,$g.copy_row]); "feature pair missing from replay") |
 (if $p.axis=="selector-swap" then "period" else "after_b" end) as $stage |
 ($p.models|map_values(.[$stage].copy_minus_stop)) as $v |
 require($g.copy_minus_stop == coefficients($v.zero;$v.period_only;$v.post_b_only;$v.full); "feature contributions differ from GPU pair scores") | .id] as $checked |
([$zero[0],$period[0],$after[0],$full[0]] | map(INDEX(.[];.context_id))) as $parts |
[$contexts[]|. as $g|[$parts[]|.[$g.context_id].copy_minus_stop] as $v |
 require(all($v[];type=="number") and $g.copy_diagonal_minus_stop_diagonal == coefficients($v[0];$v[1];$v[2];$v[3]); "feature contributions differ from GPU quartet scores") | .context_id] as $crossed |
require(($checked|length)==($checked|unique|length) and ($crossed|length)==($crossed|unique|length); "duplicate comparison") |
{split:$split, compared_pairs:($checked|length), compared_quartets:($crossed|length), all_components_exact:true}
