# Independent raw-visit recount. Counts and score comparisons use integers
# under 2^53. jq 1.8+ must preserve the literal u64 conversation group IDs.
def count(test): map(select(test)) | length;
def strict: .margin_q48 > 0;
def same_election($decision; $gold):
    ($decision.rival > 0 and $decision.rival <= 4095 and $decision.rival != $gold)
    and ($decision.rank > 0 and $decision.rank <= 4095)
    and (($decision.rank == 1) == ($decision.margin_q48 > 0 or ($decision.margin_q48 == 0 and $decision.rival > $gold)));

def valid_row:
    . as $r
    | .gold >= 1 and .gold <= 4095
      and all([.zero, .old_geometry, .ceiling, .visit.before, .visit.after, .checkpoint][]; same_election(.; $r.gold))
      and (.checkpoint.margin_q48 <= .ceiling.margin_q48)
      and (.visit.after.margin_q48 <= .ceiling.margin_q48)
      and (.visit.before.margin_q48 <= .visit.after.margin_q48 and .visit.before.rank >= .visit.after.rank)
      and (if .policy == "full" then (.visit.after | strict) == (.ceiling | strict) else true end)
      and (if .visit.outcome == 1 then
              (.visit.before | strict | not) and (.visit.after | strict)
              and (.visit.radius >= 1 and .visit.radius <= 65535)
              and (.visit.changed_words >= 1)
              and (.visit.l1 >= .visit.changed_words and .visit.l1 <= (.visit.changed_words * .visit.radius))
          elif .visit.outcome == 0 or .visit.outcome == 2 then
              .visit.before == .visit.after and ((.visit.before | strict) == (.visit.outcome == 0))
              and (.visit | [.radius,.changed_words,.l1,.saturated_words] == [0,0,0,0])
          else false end);
def movement:
    {
      rows:length,
      strict_before:count(.visit.before | strict),
      immediate:count(.visit.after | strict),
      updated:count(.visit.outcome == 1),
      infeasible:count(.visit.outcome == 2),
      retained:count((.visit.after | strict) and (.checkpoint | strict)),
      erased:count((.visit.after | strict) and (.checkpoint | strict | not)),
      later_gains:count((.visit.after | strict | not) and (.checkpoint | strict)),
      checkpoint_strict:count(.checkpoint | strict),
      ties_won:count(.checkpoint.margin_q48 == 0 and .checkpoint.rival > .gold),
      ties_lost:count(.checkpoint.margin_q48 == 0 and .checkpoint.rival < .gold),
      changed_words:(map(.visit.changed_words) | add // 0),
      l1:(map(.visit.l1) | add // 0),
      directional_saturations:(map(.visit.saturated_words) | add // 0),
      max_radius:(map(.visit.radius) | max // 0)
    };
def pair:
    sort_by(.addressing)
    | if map(.addressing) != ["geometry","ordered-matched"] then error("missing or duplicated addressing arm") else . end
    | {g:.[0],o:.[1]}
    | if (.g | [.ordinal,.group,.gold,.zero,.old_geometry,.common]) != (.o | [.ordinal,.group,.gold,.zero,.old_geometry,.common])
         or (.g.common != ((.g.ceiling | strict) and (.o.ceiling | strict)))
      then error("row pair or common-attainable mask disagreement") else . end;
def paired:
    map(select(.g.common))
    | {
      rows:length,
      groups:(map(.g.group) | unique | length),
      zero_strict:count(.g.zero | strict),
      old_strict:count(.g.old_geometry | strict),
      geometry_strict:count(.g.checkpoint | strict),
      ordered_strict:count(.o.checkpoint | strict),
      geometry_immediate:count(.g.visit.after | strict),
      geometry_retained:count((.g.visit.after | strict) and (.g.checkpoint | strict)),
      ordered_immediate:count(.o.visit.after | strict),
      ordered_retained:count((.o.visit.after | strict) and (.o.checkpoint | strict))
    }
    | . as $counts
    | .meets_endpoints = (.rows > 0 and all([.zero_strict,.old_strict,.ordered_strict][]; 20 * ($counts.geometry_strict - .) >= $counts.rows))
    | .meets_retention = (.geometry_immediate > 0 and 5 * .geometry_retained >= 4 * .geometry_immediate)
    | .pass = (.meets_endpoints and .meets_retention);
def observed:
    .paired |= del(.geometry_minus_old,.geometry_minus_zero,.geometry_minus_ordered,.geometry_retention,.ordered_retention,.resamples);
def recount:
    .[0] as $key
    | group_by(.ordinal)
    | if length != 3891 then error("incomplete visit domain") else . end
    | map(pair)
    | {
        epoch:$key.epoch,
        seed:$key.seed,
        policy:$key.policy,
        geometry:(map(.g) | movement),
        ordered:(map(.o) | movement),
        paired:paired
      };

if any(.[]; (valid_row | not) or (.visit | .changed_words < 0 or .saturated_words < 0)
    or (. as $r | (if .policy == "gold-only" then 4 else 104 end) as $cap | $r.visit.changed_words > $cap or $r.visit.saturated_words > $cap))
then error("invalid visit outcome, decision, movement or full-policy ceiling mismatch") else . end
| if any(group_by([.addressing,.ordinal])[]; (map([.group,.gold,.zero,.ceiling,.common]) | unique | length) != 1)
  then error("frozen electorate or feasibility drift across seeds, epochs or policies") else . end
| if any(group_by([.seed,.epoch,.ordinal])[]; (map(.old_geometry) | unique | length) != 1)
  then error("historical geometric control drift across addressing or policies") else . end
| group_by([.epoch,.seed,.policy])
| if map(.[0] | [.epoch,.seed,.policy]) != [range(1;9) as $e | (435300,435301) as $s | ("full","gold-only") | [$e,$s,.]]
  then error("missing or unexpected epoch, seed or policy") else . end
| map(recount)
| . as $epochs
| if $epochs != ($reported[0] | map(observed) | sort_by(.epoch,.seed,.policy))
  then error("raw counts differ from reported epochs") else . end
| ($epochs | map(select(.epoch == 8)) | group_by(.policy) | map({key:(.[0].policy|gsub("-";"_")),value:all(.[]; .paired.pass)}) | from_entries) as $advance
| if $advance != $run[0].two_seed_full_gate or $run[0].controls_verified != 32
  then error("gate or history count disagrees with reported run") else {epochs:$epochs,two_seed_full_gate:$advance} end
