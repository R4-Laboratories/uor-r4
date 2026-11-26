def require($ok;$message): if $ok then . else error($message) end;
def released_arm: ["baseline","correct_link","exchanged_link","far_only","free","native","protected"];
def empty: {start:0,end:0};

. as $plan |
($refs[0]|INDEX(.id)) as $ref |
($devices|INDEX(.case_id)) as $reader |
require(length==14 and all(group_by(.id)[]; length==7 and (map(.arm)|sort)==released_arm);
    "seven identical-arm cohorts are required") |
require(all($plan[]; . as $p | $ref[$p.id] as $r | $reader[$p.id] as $d | $p.wire as $wire |
    ($d.arms[$p.arm].selection // [empty,empty,empty]) as $span |
    $wire.prompt_sha256==$r.structural_prompt_sha256 and $wire.prompt_sha256==$d.prompt_sha256 and
    $wire.first_role==$r.gold_binding.first_role and $wire.pending==$r.gold_binding.pending and
    $p.selection==$d.arms[$p.arm].selection and
    (if .arm=="native" then $wire.learned_field=="scope" and ($wire|has("affirmed")|not)
     else [$wire.affirmed,$wire.excluded,$wire.referenced]==$span and ($wire|has("learned_field")|not) end));
    "role, pending action, prompt or original span changed on the ordinary client") |
($take[0]) as $eos | ($defer[0]) as $cap |
require(all($eos.first,$eos.returned;
  .ids==$eos.target.ids and .stop=="eos" and .exact_through_eos and .first_wrong==null);
  "expected original IDs through observed EOS") |
require(all($cap.first,$cap.returned;
   .ids==$cap.target.ids and (.ids|length)==48 and .stop=="max_tokens" and
   .exact_through_eos==false and .correct_prefix_at_cap and .first_wrong==null);
   "correct 48-ID prefix must not earn EOS") |
{pass:true,plans:length,arms:released_arm,same_role_and_pending:true,
 original_span_wires:true,authentic_eos:true,cap_is_not_eos:true}
