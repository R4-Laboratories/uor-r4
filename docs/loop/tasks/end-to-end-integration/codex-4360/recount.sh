#!/bin/sh
# Independent recount of the paired preference endpoint and native raw outputs.
set -eu
if [ "$#" -ne 3 ]; then
    printf 'usage: %s FROZEN_PAIRS.jsonl ARCHIVED_GPU_RAW.jsonl TOKENIZER.json\n' "$0" >&2
    exit 2
fi
expected_pin=8640f5f72de45f2c8a0c6fb8990830691da3e5602cf87ac506ff584b3f4182b4
observed_pin=$(shasum -a 256 < "$1" | cut -d ' ' -f 1)
if [ "$observed_pin" != "$expected_pin" ]; then
    printf 'frozen pair tape mismatch: %s\n' "$observed_pin" >&2
    exit 1
fi
expected_tokenizer_pin=a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
observed_tokenizer_pin=$(shasum -a 256 < "$3" | cut -d ' ' -f 1)
if [ "$observed_tokenizer_pin" != "$expected_tokenizer_pin" ]; then
    printf 'Gate 0 tokenizer mismatch: %s\n' "$observed_tokenizer_pin" >&2
    exit 1
fi
exec jq -ne --slurpfile tokenizer "$3" --slurpfile frozen "$1" --slurpfile gpu "$2" '
    def check($condition; $message): if $condition then . else error($message) end;
    def count(filter): map(select(filter)) | length;
    def sign($a; $b): if $a > $b then 1 elif $a < $b then -1 else 0 end;
    def mass: .target_mass_q48 |
        if type == "number" and floor == . and . >= 0 and . <= 281474976710656
        then . else error("invalid Q48 mass") end;
    def prefs:
        {own: count(. == 1), other: count(. == -1), tied: count(. == 0)};
    def arm($row; $rival):
        check(.own.token == $row.expected[0] and .other.token == $rival.expected[0];
              "wrong original target identity") |
        {own: (.own | mass), other: (.other | mass)} |
        . + {preference: sign(.own; .other)};
    def native($run; $gold):
        $run.emitted as $ids | $run.own as $own |
        ($ids | length) as $n |
        check($n >= 1 and $n <= 32 and
              all($ids[]; type == "number" and floor == . and . >= 0 and . < 4096);
              "invalid emitted IDs or length") |
        check(all($ids[0:-1][]; . > 1) and
              ($ids[-1] == 0 or $ids[-1] == 1 or $n == 32);
              "emission after end or short cap") |
        check($run.first.greedy_token == $ids[0]; "first trace and emission differ") |
        ($run.first.greedy_coarse == $own.cell) as $admitted |
        $own.diagnostic_within_cell_rank as $rank |
        check($rank >= 1 and $rank <= 64 and $rank == ($rank | floor) and
              $admitted == $own.admitted and
              ($ids[0] == $gold[0]) == ($admitted and $rank == 1);
              "admission or rank inconsistent with greedy routing") |
        ([range(0; $n) | select($ids[.] != $gold[.])] | first) as $wrong |
        check($wrong == $run.first_error; "first native disagreement mismatch") |
        ($ids[-1] == 1) as $eos |
        ($eos and $ids == $gold) as $exact |
        check($exact == $run.exact and $eos == $run.hit_eos;
              "native natural EOS or exact receipt mismatch") |
        {first: $ids[0], admitted: $admitted, rank: $rank, top_one: ($ids[0] == $gold[0]),
         wrong: $wrong, eos: $eos, exact: $exact,
         early_eos: ($eos and $n < ($gold | length)), invalid_bos: ($ids[-1] == 0),
         cap: ($ids[-1] > 1), censored: ($wrong == null and ($exact | not))};
    def top_five($vocab; $special; $byte_by_cp):
        group_by(.) | map({id: .[0], count: length}) | sort_by(-.count, .id) | .[0:5] |
        map(.id as $id | $vocab[$id | tostring] as $piece |
            ($special[$id | tostring]) as $literal |
            if $literal != null then . + {vocab_piece: $piece, decoded: $literal}
            else ($piece | explode | map($byte_by_cp[. | tostring])) as $bytes |
                check(all($bytes[]; . != null); "invalid byte-level vocab character") |
                . + {vocab_piece: $piece, decoded_bytes: $bytes,
                     decoded: (if all($bytes[]; . < 128) then $bytes | implode else null end)}
            end);
    ([range(33; 127), range(161; 173), range(174; 256)]) as $literal_bytes |
    ([range(0; 256)] - $literal_bytes) as $encoded_bytes |
    ([$literal_bytes[] | {key: tostring, value: .}] +
     [$encoded_bytes | to_entries[] | {key: ((.key + 256) | tostring), value: .value}] | from_entries) as $byte_by_cp |
    ($tokenizer[0].added_tokens | map({key: (.id | tostring), value: .content}) | from_entries) as $special |
    ($tokenizer[0].model.vocab | to_entries | map({key: (.value | tostring), value: .key}) | from_entries) as $vocab |
    ([range(1; 65)]) as $ids |
    check(($frozen | map(.pair_id) | sort) == $ids and
          ($gpu | map(.pair_id) | sort) == $ids; "pair IDs missing or reused") |
    ($gpu | INDEX(.pair_id)) as $by_id |
    [$frozen[] as $p | $by_id[$p.pair_id | tostring] as $g |
        check(all(["a", "b"][]; . as $side |
              $p[$side].prefix == $g[$side].row.prefix and $p[$side].expected == $g[$side].row.expected and
              $p[$side].expected[-1] == 1 and all($p[$side].expected[0:-1][]; . > 1));
              "raw input tape or natural gold EOS differs") |
        ($g.a.native | arm($p.a; $p.b)) as $an |
        ($g.b.native | arm($p.b; $p.a)) as $bn |
        ($g.a.swap | arm($p.a; $p.b)) as $as |
        ($g.b.swap | arm($p.b; $p.a)) as $bs |
        {signs: [$an.preference, $bn.preference, $as.preference, $bs.preference],
         native_sum: sign(($an.own + $bn.own); ($an.other + $bn.other)),
         swap_sum: sign(($as.own + $bs.own); ($as.other + $bs.other)),
         swap_ids: [$g.a.swap.emitted[0], $g.b.swap.emitted[0]],
         native_a: native($g.a.native; $p.a.expected),
         native_b: native($g.b.native; $p.b.expected)}
    ] as $pairs |
    ($pairs | map(.native_a, .native_b)) as $native |
    ($native | map(.rank) | sort) as $ranks |
    {pairs: ($pairs | length),
     native_replies: ($native | length),
     native_dual_own_preference: ($pairs | count(.signs[0:2] == [1, 1])),
     swap_dual_donor_preference: ($pairs | count(.signs[2:4] == [-1, -1])),
     strict_four_signs: ($pairs | count(.signs == [1, 1, -1, -1])),
     native_row_preference: ($pairs | map(.signs[0:2][]) | prefs),
     swap_row_preference: ($pairs | map(.signs[2:4][]) | prefs),
     native_assignment_sum: ($pairs | map(.native_sum) | prefs),
     swap_assignment_sum: ($pairs | map(.swap_sum) | prefs),
     native_to_swap_sum_reversal: ($pairs | count(.native_sum == 1 and .swap_sum == -1)),
     native_top_five_first_ids: ($native | map(.first) | top_five($vocab; $special; $byte_by_cp)),
     swap_top_five_first_ids: ($pairs | map(.swap_ids[]) | top_five($vocab; $special; $byte_by_cp)),
     swaps_changed_first_original_id: ($pairs | count(.native_a.first != .swap_ids[0]) + count(.native_b.first != .swap_ids[1])),
     native_admitted_rank_top_one: ($native | count(.admitted and .rank == 1)),
     native_admitted_rank_top_eight: ($native | count(.admitted and .rank <= 8)),
     native_admitted: ($native | count(.admitted)),
     native_both_admitted: ($pairs | count(.native_a.admitted and .native_b.admitted)),
     native_target_rank_histogram: ($ranks | group_by(.) | map({key: (.[0] | tostring), value: length}) | from_entries),
     native_target_rank_middle: [$ranks[63], $ranks[64]],
     native_forced_top_one: ($native | count(.rank == 1)),
     native_forced_top_eight: ($native | count(.rank <= 8)),
     native_greedy_top_one: ($native | count(.top_one)),
     native_first_wrong: ($native | map(.wrong) | group_by(.) | map({key: (.[0] | tostring), value: length}) | from_entries),
     native_natural_eos: ($native | count(.eos)),
     native_exact: ($native | count(.exact)),
     native_early_eos: ($native | count(.early_eos)),
     native_invalid_bos: ($native | count(.invalid_bos)),
     native_cap: ($native | count(.cap)),
     native_censored: ($native | count(.censored))}
'
