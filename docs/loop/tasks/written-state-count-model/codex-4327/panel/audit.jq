def require(ok; why): if ok then . else error(why) end;
def row_key: [.pair_id, .side] | @json;
def normalized: gsub("[[:space:]]+"; " ") | sub("^ "; "") | sub(" $"; "") | ascii_downcase;
def overlap($a; $b): (($a | unique) - (($a | unique) - ($b | unique))) | length;
def axis:
  map(select(.one_word_clause_ordinal != null)) as $scoped |
  ($scoped | group_by(.selector_swap_id)) as $selector |
  ($scoped | group_by(.pair_id)) as $clause |
  ($scoped | group_by(.context_id)) as $quartet |
  require(($selector | length) == 128 and ($clause | length) == 128 and ($quartet | length) == 64;
    "wrong number of scoped matched units") |
  require(all($selector[];
      length == 2 and (map(.side)|sort) == [0,1]
      and (map(.one_word_clause_ordinal)|unique|length) == 1
      and (map(.operative_clause_ordinal)|sort) == [1,2]);
    "selector pairs are not minimal") |
  require(all($clause[];
      length == 2 and (map(.side)|sort) == [0,1]
      and (map(.operative_clause_ordinal)|unique|length) == 1
      and (map(.one_word_clause_ordinal)|sort) == [1,2]);
    "clause pairs are not minimal") |
  require(all($quartet[];
      length == 4 and (map([.operative_clause_ordinal,.one_word_clause_ordinal])|sort) == [[1,1],[1,2],[2,1],[2,2]]
      and (map([.context,.question,.family,.wording_author])|unique|length) == 1
      and all(.[]; .side == (if .operative_clause_ordinal == .one_word_clause_ordinal then 0 else 1 end)));
    "invalid matched quartet") |
  {selector_pairs:($selector|length),clause_pairs:($clause|length),quartets:($quartet|length)};

($held_keys | INDEX(.[]; row_key)) as $keys |
require(($held_rows | length) == 512 and ($keys | length) == 512
    and ($held_rows | map(row_key) | unique | length) == 512; "nonunique or incorrect row/key count") |
require(all($held_rows[]; . as $r | $keys[row_key] as $k |
   $k != null and $k.row_id == $r.row_id and $k.source_row_id == $r.row_id
   and $k.family == $r.family and $k.contrast == $r.contrast
   and $k.prefix_tokens == ($r.prefix | length)
   and $k.expected_word_count == ($r.side + $r.contrast + 1)
   and $r.expected == ([$r.ids[0],16] + $r.ids[1:($r.side + $r.contrast + 1)] + [1])
   and all(range(4); . as $i | $r.prefix[$r.positions[$i]] == $r.ids[$i]
       and ($r.prefix | map(select(. == $r.ids[$i])) | length) == 1)
   and ($r.prefix | length) + 5 < 192);
   "encoded values, expected sequence or key identity drift") |
($held_keys | axis) as $axes |
($held_keys | map(select(.pair_axis == "direct-count-control")) | group_by(.contrast)) as $direct |
require(($direct | map(length)) == [128,128] and all($direct[] | group_by(.pair_id)[];
    length == 2 and (map(.side) | sort) == [0,1]
    and (map([.context,.question,.family,.wording_author]) | unique | length) == 1);
  "direct roles or counts drift") |
($held_rows | group_by(.pair_id)) as $original_pairs |
require(all($original_pairs[]; (map([.positions,.ids])|unique|length) == 1);
  "original pair moved a source") |
($held_rows | map(.prefix | @json)) as $prefixes |
($held_keys | map(.user | normalized)) as $users |
require(($prefixes | unique | length) == 512 and ($users | unique | length) == 512; "held prompt duplicate") |
{source_ids:overlap([$held_rows[].ids[]];[$prior_rows[]|.ids[]?, .target?, .next_id? | select(. != null)]),
 prefixes:overlap($prefixes;[$prior_rows[].prefix | @json]),
 pair_ids:overlap([$held_rows[].pair_id];[$prior_rows[].pair_id]),
 prompts:overlap($users;[$prior_keys[].user | normalized]),
 contexts:overlap([$held_keys[].context | normalized];[$prior_keys[].context | normalized]),
 directives:overlap([$held_keys[].directive | normalized];[$prior_keys[].directive | normalized])} as $overlap |
require(all($overlap[]; . == 0); "archived exact overlap: " + ($overlap | @json)) |
{held_rows:($held_rows|length), audited_prior_rows:($prior_rows|length), exact_overlap:$overlap, matched:$axes,
 direct_one_two_pairs:($direct[0]|length/2), direct_two_three_pairs:($direct[1]|length/2),
 authors:($held_keys|group_by(.wording_author)|map({author:.[0].wording_author,rows:length,families:(map(.family)|unique|length)}))}
