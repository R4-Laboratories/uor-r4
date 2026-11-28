def require($ok;$message): if $ok then . else error($message) end;
def addresses($row): $row.candidates | map([.start,.end]);
# ROLE_COPY_STOPS in the pinned ordinary client, enforced by both interval
# enumeration and the exact copy writer.
def locked_ids: [16,1,0,201];
def first_difference($own;$peer):
  first(range(0;([($own|length),($peer|length)]|max)) |
    select($own[.] != $peer[.]) | {offset:.,own_id:$own[.],swapped_id:$peer[.]}) // null;
def has_gold($case;$gold):
  $gold.start == null or any($case.candidates[]; .start == $gold.start and .end == $gold.end);
def rebind($opening;$gold;$truth):
  if $gold.start == null then $gold.end == null and $truth.source_ids == null
  else
    ($gold.start | type) == "number" and ($gold.end | type) == "number" and
    $gold.start >= $opening.user[0] and $gold.end <= $opening.user[1] and
    $gold.start < $gold.end and
    ($opening.ids[$gold.start:$gold.end] == $truth.source_ids) and
    ($truth.original_bytes | type) == "array" and ($truth.original_bytes | length) == 2 and
    ([$opening.byte_bounds[$gold.start - $opening.user[0]][0],
      $opening.byte_bounds[$gold.end - 1 - $opening.user[0]][1]]
      == ($truth.original_bytes | map(. + 1)))
  end;

($train | map(.case_id)) as $ids |
require(($ids | length) > 0 and ($ids | unique | length) == ($ids | length) and
  all([$gold,$truth,$opening][]; map(.case_id) == $ids); "inconsistent TRAIN roster") |
($capture | map(select(.arm == "geometry"))) as $geometry |
($capture | map(select(.arm == "count"))) as $count |
require(($capture | length) == 2 * ($ids | length) and
  ($geometry | map(.case_id)) == $ids and ($count | map(.case_id)) == $ids;
  "inconsistent native case or arm roster") |
($gold | INDEX(.case_id)) as $g |
($truth | INDEX(.case_id)) as $t |
($geometry | INDEX(.case_id)) as $f |
($count | INDEX(.case_id)) as $c |
($opening | INDEX(.case_id)) as $o |
[$gold[] | . as $label | .case_id as $id | .swapped_case_id as $mate |
  ($f[$id]) as $here | ($f[$mate // $id]) as $peer |
  ($o[$id].ids[$here.fact[0]:$here.fact[1]]) as $original_facts |
  ($o[$mate // $id].ids[$peer.fact[0]:$peer.fact[1]]) as $swapped_facts |
  ([($t[$id].source_ids // [])[] as $token | select(locked_ids | index($token) != null) | $token]
    | unique) as $locked_source_ids |
  require($here.user == $o[$id].user and $here.user == $c[$id].user and
    $here.fact == $c[$id].fact and $here.question == $c[$id].question and
    addresses($here) == addresses($c[$id]); "opening or native arms disagree: " + $id) |
  require(rebind($o[$id];$label;$t[$id]); "original source IDs differ at own positions: " + $id) |
  require($mate == null or ($mate != $id and $g[$mate].swapped_case_id == $id and
    $g[$mate].group == $label.group and $t[$id].passage_sha256 != null and
    $t[$mate].passage_sha256 == $t[$id].passage_sha256);
    "invalid reciprocal identity: " + $id) |
  {case_id:$id,swapped_case_id:$mate,group:$label.group,tape:$t[$id].tape,
   own_fact:$here.fact, swapped_fact:$peer.fact,
   own_question:$here.question,swapped_question:$peer.question,
   fact_token_difference:first_difference($original_facts;$swapped_facts),
   locked_source_ids:$locked_source_ids,
   reasons:[
     if $mate != null and addresses($here) != addresses($peer)
     then "different_candidate_intervals" else empty end,
     if $mate != null and $here.fact != $peer.fact
     then "different_fact_bounds" else empty end,
     if $mate != null and $original_facts != $swapped_facts
     then "different_fact_token_ids" else empty end,
     if ($locked_source_ids|length) > 0 then "source_contains_locked_copy_stop" else empty end,
     if has_gold($here;$label) | not then "own_gold_absent" else empty end,
     if $mate != null and (has_gold($here;$g[$mate]) | not)
     then "swapped_gold_absent" else empty end,
     if $mate != null and (rebind($o[$mate];$label;$t[$id]) | not)
     then "source_ids_differ_in_swapped_opening" else empty end,
     if ($here.candidates | length) == 0 then "no_source_candidate_rival" else empty end
   ]}] |
INDEX(.case_id) as $claims |
$ids[] as $id | $claims[$id] |
  ($claims[.swapped_case_id // $id].reasons) as $pair_reasons |
  . + {retained:((.reasons | length) == 0 and ($pair_reasons | length) == 0),
       swapped_reasons:(if .swapped_case_id == null then [] else $pair_reasons end)}
