# Compare authentic ordinary output on the same 73 positives. Null first_wrong
# means no differing committed ID; the separate exact flag checks source and EOS.
def positives:
  map(select(.negative_group==null)) |
  if length==73 and (map(.case_id)|unique|length)==73 then . else error("incomplete positive panel") end;
def signature:
  if .first_wrong==null then "ids_match"
  elif .first_wrong.actual==null then "missing_id"
  elif .first_wrong.expected==null then "extra_id" else "substitution" end;
def profile:
  {exact_natural_eos:(map(select(.exact_through_natural_eos))|length),
   canonical_initial_space_missed:(map(select(.canonical_initial_space_missed))|length),
   first_wrong_offsets:(group_by(.first_wrong.offset)|map({offset:.[0].first_wrong.offset,count:length})),
   first_wrong_types:(group_by(signature)|map({type:(.[0]|signature),count:length})),
   first_wrong_id_pairs:(map(select(.first_wrong!=null))|group_by([.first_wrong.offset,.first_wrong.expected,.first_wrong.actual])|
       map(.[0].first_wrong+{count:length})|sort_by(-.count,.offset,.expected,.actual))};
def movement($a;$b):
  if $a.first_wrong==null and $b.first_wrong==null then "both_id_sequences_match"
  elif $a.first_wrong==null then "became_id_exact"
  elif $b.first_wrong==null then "lost_id_exactness"
  elif $a.first_wrong.offset>$b.first_wrong.offset then "later"
  elif $a.first_wrong.offset<$b.first_wrong.offset then "earlier" else "same_offset" end;
(positives) as $new | ($control[0]|positives) as $base |
if ($new|map([.case_id,.opening_sha256])|sort)!=($base|map([.case_id,.opening_sha256])|sort) then error("different openings") else . end |
($base|INDEX(.case_id)) as $old |
[$new[] | . as $a | $old[.case_id] as $b | {case_id,change:movement($a;$b),natural:.first_wrong,control:$b.first_wrong,
 natural_eos:.exact_through_natural_eos,control_eos:$b.exact_through_natural_eos}] as $pairs |
{positive_prompts:($new|length),proposed:($new|profile),control:($base|profile),
 changes:($pairs|group_by(.change)|map({change:.[0].change,count:length})),paired:$pairs}
