def must($condition; $failure):
  if $condition then . else error($failure) end;
def clean:
  ascii_downcase | gsub("[[:space:]]+"; " ") | sub("^ "; "") | sub(" $"; "");
def wording:
  ascii_downcase | [scan("[a-z0-9]+")] | join(" ");
def row_id:
  .pair_id + ":" + (.side | tostring);
def index_set($values):
  reduce $values[] as $value ({}; .[$value | tostring] = true);
def overlaps($left; $right):
  index_set($right) as $set | [$left[] | select($set[. | tostring])];
def join_tape($rows; $keys):
  must(($rows | length) == ($keys | length); "row/key count differs")
  | [range(0; $rows | length) as $i
     | { row: $rows[$i], key: $keys[$i] }
     | must((.row | row_id) == .row.row_id
            and .row.row_id == .key.row_id
            and .row.pair_id == .key.pair_id
            and .row.contrast == .key.contrast
            and .row.side == .key.side
            and .row.family == .key.family;
            "row/key identity differs")
     | (.row.side + .row.contrast) as $extra
     | (.row.prefix | length) as $length
     | (.row.prefix | indices([201,201])) as $fences
     | must(($fences | length) == 1
            and $length == .key.prefix_tokens and $length + 5 < 192
            and .row.prefix[-6:] == [201,35,560,652,714,28]
            and .row.expected == ([.row.ids[0],16] + .row.ids[1:1+$extra] + [1])
            and .key.expected_word_count == $extra + 1
            and (.row.ids | unique | length) == 4;
            "invalid prompt length, source set or reply")
     | . as $item
     | must(all(range(0;4); . as $i
                | $item.row.positions[$i] < $fences[0]
                and $item.row.prefix[$item.row.positions[$i]] == $item.row.ids[$i]
                and ([$item.row.prefix[] | select(. == $item.row.ids[$i])] | length) == 1);
            "invalid original source occurrence")];
def matched:
  .[0] as $first
  | all(.[]; .row.positions == $first.row.positions
            and .row.ids == $first.row.ids and .key.context == $first.key.context
            and .key.question == $first.key.question
            and .key.fact_role_order == $first.key.fact_role_order);
def check_pair:
  (.[0].key.prefix_tokens == .[1].key.prefix_tokens) as $equal
  | must(length == 2 and (map(.row.side) == [0,1]) and matched
       and .[0].row.contrast == .[1].row.contrast
       and .[0].key.directive != .[1].key.directive
       and all(.[]; .key.equal_length_pair == $equal);
       "invalid paired comparison");
def check_context:
  . as $items
  | must(matched
        and all(.[]; .key.equal_length_context == ([$items[].key.prefix_tokens] | unique | length == 1))
        and ([.[].key.directive] | unique | length) == length
        and ([$items[].row.family] | unique | length) == 1;
         "invalid matched context")
  | if length == 4 then
      must((map([.key.operative_clause_ordinal, .key.one_word_clause_ordinal]) | sort) ==
            [[1,1],[1,2],[2,1],[2,2]]
            and all(.[]; .row.contrast == 0
                         and (.row.side == 0) == (.key.operative_clause_ordinal == .key.one_word_clause_ordinal));
           "invalid quartet crossing")
      | must((group_by(.key.selector_swap_id)
             | all(.[]; length == 2 and ([.[].row.side] | sort) == [0,1]
                          and (map(.key.equal_length_selector_pair) | unique) ==
                              [.[0].key.prefix_tokens == .[1].key.prefix_tokens]));
            "invalid alternate selector edges")
    else
      must(length == 2 and all(.[]; .key.operative_clause_ordinal == null
                                  and .key.one_word_clause_ordinal == null
                                  and .key.selector_swap_id == null
                                  and .key.equal_length_selector_pair == null);
           "invalid direct control")
    end;
def assert_fresh($items):
  ($items | group_by(.key.context_id) | map(check_context)) as $contexts
  | ($contexts | map(select(length == 4))) as $quartets
  | ($contexts | map(select(length == 2))) as $direct
  | must(($items|length) == 512 and ($quartets|length) == 64 and ($direct|length) == 128
         and ([$quartets[] | .[0].row.family] | unique | length) == 8
         and ([$direct[] | .[0].row.family] | unique | length) == 4
         and ($direct | map(select(.[0].row.contrast == 1)) | length) == 64
         and ([$items[].row.row_id] | unique | length) == 512
         and ([$items[].row.prefix] | unique | length) == 512
         and ([$items[].key.user | clean] | unique | length) == 512;
         "fresh census or identity changed")
  | {rows: ($items|length), quartets: ($quartets|length),
     equal_length_quartets: ([$quartets[] | select(.[0].key.equal_length_context)] | length),
     equal_length_selector_edges: ([$quartets[][] | select(.key.equal_length_selector_pair)] | length / 2),
     equal_length_order_edges: ([$quartets[][] | select(.key.equal_length_pair)] | length / 2),
     direct_1v2_pairs: ([$direct[] | select(.[0].row.contrast == 0)] | length),
     direct_2v3_pairs: ([$direct[] | select(.[0].row.contrast == 1)] | length),
     equal_length_direct_pairs: ([$direct[] | select(.[0].key.equal_length_context)] | length)};
def overlap_counts($left_rows; $left_keys; $right_rows; $right_keys):
  {prefixes: (overlaps([$left_rows[].prefix]; [$right_rows[].prefix]) | length),
   pair_ids: (overlaps([$left_rows[].pair_id]; [$right_rows[].pair_id]) | length),
   prompts: (overlaps([$left_keys[].user | clean]; [$right_keys[].user | clean]) | length),
   contexts: (overlaps([$left_keys[].context | clean]; [$right_keys[].context | clean]) | length),
   directives: (overlaps([$left_keys[].directive | wording]; [$right_keys[].directive | wording]) | length),
   source_ids: (overlaps([$left_rows[].ids[]] | unique;
                         [$right_rows[] | (.ids[]?, .target?, .next_id?) | select(. != null)] | unique) | length)};
def original_rows($rows):
  reduce $rows[] as $row ({}; .[$row | row_id] = $row);
def original_keys($keys):
  reduce $keys[] as $key ({}; .[$key | row_id] = $key);

join_tape($train_rows; $train_keys) as $train
| join_tape($held_rows; $held_keys) as $held
| ($train | map(select(.key.cohort == "fresh"))) as $fresh
| ($train | map(select(.key.cohort != "fresh"))) as $rehearsal
| [$train,$held | group_by(.row.pair_id)[] | check_pair] as $pairs
| assert_fresh($fresh) as $train_census
| assert_fresh($held) as $held_census
| overlap_counts([$fresh[].row]; [$fresh[].key]; $held_rows; $held_keys) as $between
| overlap_counts([$fresh[].row]; [$fresh[].key]; $old_rows; $old_keys) as $train_archive
| overlap_counts($held_rows; $held_keys; $old_rows; $old_keys) as $held_archive
| overlap_counts($train_rows; $train_keys; $blocked_rows; $blocked_keys) as $blocked
| must(all([$between, $train_archive, $held_archive][]; all(.[]; . == 0))
       and ($blocked.prefixes + $blocked.pair_ids + $blocked.prompts + $blocked.contexts == 0);
       "unexpected archive or split overlap")
| original_rows($source_4322_rows) as $old_4322_rows
| original_rows($source_4323_rows) as $old_4323_rows
| original_keys($source_4322_keys) as $old_4322_keys
| original_keys($source_4323_keys) as $old_4323_keys
| must(all($rehearsal[];
           . as $record | .key.source_row_id as $id
           | if .key.cohort == "rehearsal-4322" then
               [$old_4322_rows[$id],$old_4322_keys[$id]]
             elif .key.cohort == "rehearsal-4323" then
               [$old_4323_rows[$id],$old_4323_keys[$id]]
             else error("unknown rehearsal cohort")
             end
           | .[0] == ($record.row | del(.row_id))
             and .[1] != null
             and all(.[1] | to_entries[]; . as $entry | $record.key[$entry.key] == $entry.value)
             and $record.key.source_split == "train"
             and $record.key.source_row_id == $record.row.row_id);
       "rehearsal diverges from original TRAIN row or key")
| ($rehearsal | group_by(.key.cohort) | map({cohort: .[0].key.cohort, rows: length})) as $rehearsal_census
| must(($train|length) == 768
       and ([$train[] | select(.row.contrast == 1)] | length) == 224
       and $rehearsal_census == [{cohort:"rehearsal-4322",rows:192},{cohort:"rehearsal-4323",rows:64}];
       "combined fit tape budget changed")
| {schema: "codex-4325.panel-audit/1", train: $train_census, held: $held_census,
   old_rows_checked: ($old_rows|length), excluded_rows_checked: ($blocked_rows|length),
   combined_train_rows: ($train|length), train_contrast_1_pairs: 112,
   rehearsal: $rehearsal_census, fresh_train_held_overlap: $between,
   fresh_train_archive_overlap: $train_archive, held_archive_overlap: $held_archive,
   complete_fit_tape_vs_excluded: $blocked, matched_pairs_validated: ($pairs|length)}
