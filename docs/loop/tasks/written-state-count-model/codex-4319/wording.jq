# Group repeated facts by the actual pair of instruction token sequences.
# Usage: jq -n --slurpfile obs observations.jsonl -f wording.jq
def tail: .prefix[:-6] as $user | $user[(([$user|indices(33)[]]|last)+1):];
def exact($arm): [.arms[$arm].steps[].committed_id]==.expected and .arms[$arm].terminal;
def check(ok; msg): if ok then . else error(msg) end;
[$obs | group_by(.pair_id)[] | sort_by(.side) |
 check(length==2 and .[0].side==0 and .[1].side==1;"invalid instruction pair") |
 {family:.[0].family,stop_instruction:(.[0]|tail),period_instruction:(.[1]|tail),
 stop_score:.[0].score,period_score:.[1].score,
 native_pair:(all(.[];exact("native"))),learned_pair:(all(.[];exact("learned"))),
 swapped_pair:(all(.[];exact("swapped"))),oracle_pair:(all(.[];exact("oracle"))),
 stop_exact:(.[0]|exact("learned")),period_exact:(.[1]|exact("learned"))}] |
 group_by([.family,.stop_instruction,.period_instruction]) |
 map(check((map(.stop_score)|unique|length)==1 and (map(.period_score)|unique|length)==1;"fact-dependent instruction score") |
 {family:.[0].family,pairs:length,stop_ids:.[0].stop_instruction,period_ids:.[0].period_instruction,
 stop_score:.[0].stop_score,period_score:.[0].period_score,
 native_pairs:map(select(.native_pair))|length,learned_pairs:map(select(.learned_pair))|length,
 swapped_pairs:map(select(.swapped_pair))|length,oracle_pairs:map(select(.oracle_pair))|length,
 learned_exact_stop:map(select(.stop_exact))|length,learned_exact_period:map(select(.period_exact))|length})
