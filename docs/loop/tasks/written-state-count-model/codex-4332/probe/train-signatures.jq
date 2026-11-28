# Usage: jq -cn --slurpfile gold GOLD --slurpfile captures CAPTURE -f train-signatures.jq
# The frozen candidate sums and NoRead words bound the two signed slot
# inequalities. Report authored TRAIN support flips at one locked original
# address. No model fitting or quality scoring enters this audit.
($gold | INDEX(.case_id)) as $truth |
[
  $captures[] as $row |
  $truth[$row.case_id] as $key |
  select($key.cohort=="authored" and $key.split=="train"
    and $key.supported and ($key.paired_ids|length)==1
    and $truth[$key.paired_ids[0]].supported==false
    and $truth[$key.paired_ids[0]].split=="train") |
  {
    case_id:$key.case_id,
    negative_case_id:$key.paired_ids[0],
    contrast_group:$key.contrast_group,
    contrast_type:$key.contrast_type,
    source:{pos:$key.source.pos,id:$key.source.id}
  }
] as $pairs |
($captures | INDEX(.case_id)) as $captured |
$pairs | map(
  . as $pair |
  $captured[$pair.case_id] as $positive |
  $captured[$pair.negative_case_id] as $negative |
  def at_source:
    [.candidates[] | select(.pos==$pair.source.pos and .id==$pair.source.id)] |
    if length==1 then .[0] else error("missing original source in TRAIN capture") end;
  if $positive.expanded_frozen.no_read_score!=$negative.expanded_frozen.no_read_score then
    error("frozen threshold varies across rows")
  else
    ($positive|at_source) as $a |
    ($negative|at_source) as $b |
    if $a.recurrence_code!=$b.recurrence_code then
      error("causal recurrence differs at the locked TRAIN address")
    else
      $positive.expanded_frozen.no_read_score as $no_read |
      $pair + {
        supported:$a, unsupported:$b,
        supported_query_code:$positive.query_code,
        unsupported_query_code:$negative.query_code,
        no_read:$no_read,
        feasibility:[
          1,8 | . as $margin |
          ([(-4096), ($no_read + $margin - $a.score)] | max) as $lower |
          ([4096, ($no_read - $margin - $b.score)] | min) as $upper |
          {
            margin:$margin,
            lower:$lower,
            upper:$upper,
            feasible:($lower<=4096 and $upper>=(-4096)
              and ($a.real_mask!=$b.real_mask or $lower<=$upper))
          }
        ]
      }
    end
  end
) | .[]
