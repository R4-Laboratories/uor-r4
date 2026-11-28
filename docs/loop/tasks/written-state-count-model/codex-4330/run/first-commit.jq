# jq -nc --arg arm ... --slurpfile reads slim-decisions.jsonl --slurpfile gold gold.jsonl -f first-commit.jq ordinary.jsonl
($reads|INDEX(.case_id)) as $lookup | ($gold|INDEX(.case_id)) as $truth |
[inputs as $row | $row.case_id as $id |
 $row.client.native_occurrence_steps[0] as $s | $lookup[$id][$arm] as $p |
 if $row.arm!=$arm or ($lookup[$id]|type)!="object" or $row.client.prompt_ids==null or
    $s.prefix_start!=0 or $s.source_pos!=$p or $s.read_pos!=$p or
    $s.committed_id!=$row.client.generated_ids[0] or
    ($p==null and ($s.claim_pos!=null or $s.copied or $s.committed_id!=$s.native_proposal)) or
    ($p!=null and ($s.source_id!=$row.client.prompt_ids[$p] or $s.claim_pos!=$p or $s.claimed_id!=$s.source_id)) or
    ($s.copied and ($p==null or $s.committed_id!=$s.source_id)) or
    (($s.copied|not) and $s.committed_id!=$s.native_proposal)
 then error("ordinary first decision/commit differs from CUDA and native fallback: "+$id) else
  {id:$id,cohort:$truth[$id].cohort,split:$truth[$id].split,supported:$truth[$id].supported,
   selected:($p!=null),copied:$s.copied,correct:($p!=null and $p==$truth[$id].source.pos)}
 end] as $records |
if ($records|length)!=($gold|length) or ($records|map(.id)|unique|length)!=($gold|length) then error("incomplete join") else
{arm:$arm,total:($records|length),by_cohort:($records|group_by([.cohort,.split])|map({
 cohort:.[0].cohort,split:.[0].split,total:length,selected:(map(select(.selected))|length),
 copied:(map(select(.copied))|length),correct_source_copied:(map(select(.copied and .correct))|length),
 false_reads:(map(select((.supported|not) and .selected))|length),
 unsupported_actual_copies:(map(select((.supported|not) and .copied))|length)}))}
end
