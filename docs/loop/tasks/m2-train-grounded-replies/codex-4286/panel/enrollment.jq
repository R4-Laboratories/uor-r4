def invariant($ok; $reason): if $ok then . else error($reason) end;
def counts: group_by(.) | map({key:.[0],value:length}) | from_entries;
def range: {min:min,max:max};

. as $rows |
invariant(length==256; "expected 256 enrolled prompts") |
invariant((group_by(.id) | length)==256; "case IDs are not distinct") |
invariant((group_by(.pair_id) | length==128 and all(.[];
          (map(.admission)|sort) == ["defer","take"] and
          (map([.family,.fact,.distance,.governor_order,.placement_name,.requested_role,.requested_horizon]) | unique | length)==1));
          "the TAKE/DEFER matched blocks changed") |
group_by(.family) |
invariant(length==4 and all(.[]; length==64); "family enrollment changed") |
map({family:.[0].family, rows:length, distance:
  (group_by(.distance) |
  invariant(length==2 and all(.[]; length==32); "near and far must cross in every family") |
  map({distance:.[0].distance, rows:length,
    admissions:(map(.admission)|counts),
    neutral_order:(map(.placement_name)|counts),
    governor_order:(map(.governor_order)|counts),
    fact_first:(map(.requested_role)|counts),
    scopes:(map(.scope_tuple_count)|range),
    nominal_reply_ids:(map(.effective_grounded.nominal.expected|length)|range),
    affirmative_gap:(map(.governors.affirmative.gap)|range),
    excluded_gap:(map(.governors.excluded.gap)|range),
    affirmative_visible:(map(.governors.affirmative.visible_tokens)|range),
    excluded_visible:(map(.governors.excluded.visible_tokens)|range)
  }))}) |
invariant(all(.[].distance[]; .admissions=={defer:16,take:16} and .neutral_order=={after:16,before:16}
    and .governor_order=={after:16,before:16} and .fact_first=={home:16,job:16}
    and (if .distance=="near" then .affirmative_gap.max<6 and .excluded_gap.max<6
      and .affirmative_visible.min>0 and .excluded_visible.min>0
    else .affirmative_gap.min>=8 and .excluded_gap.min>=8
      and .affirmative_visible.max==0 and .excluded_visible.max==0 end));
    "distance strata lost matched visibility or balance") |
{rows:($rows|length),matched_pairs:128,paired_dimensions:["distance","governor_order","neutral_order","requested_role"],families:.}
