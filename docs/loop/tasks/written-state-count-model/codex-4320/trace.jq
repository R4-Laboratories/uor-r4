# Read and rank attribution from actual post-period commits.
def tally(xs): group_by(xs)|map({value:(.[0]|xs),n:length})|sort_by(-.n);
def unique_n(f): map(f)|unique|length;
def same_rows(f):group_by(.pair_id)|map(select(length==2 and ((map(f)|unique|length)==1)))|length;
def count_if(f):map(select(f))|length;
def summary: {
 rows:length, read_codes:unique_n(.read),recurrence_codes:unique_n(.rec),unique_joint:unique_n([.read,.rec]),
 paired_same_read:same_rows(.read),paired_same_rec:same_rows(.rec),paired_same_gate:same_rows(.gate),paired_same_emission:same_rows(.emission),
 pointer_predecessor_id:tally(.pred[1]),pointer_successor_id:tally(.succ[1]),pointer_region:tally(.region),
 gate_ids:tally(.gate),writer_ids:tally(.emission),gate_and_output:tally([.gate,.emission]),
 pointer_wins:count_if(.gate==.succ[1]),writer_preserves_gate:count_if(.gate==.emission),
 sensitivity_winners:tally(.sensitivity),gate_terms:tally(.gateprobe.terms)[:8],
 gate_distance_zero:count_if(.gateprobe.distance==0),g_distance_zero:count_if(.gprobe.distance==0),
 gate_writer_rank:tally(.gateprobe.writer_rank),g_writer_rank:tally(.gprobe.writer_rank),top_eight:tally(.shortlist|map(.id))[:6]};
[.[] as $o | $o.third_trace as $t | ($o.prefix|length) as $end |
 ($t.pointer_predecessor[0]) as $pos | ($o.prefix|indices([201,201])|last) as $fence |
 {pair_id:$o.pair_id,side:$o.side,marked:($o.prefix[$end-7]==16),
 read:$t.read_code,rec:$t.recurrence_code,gate:$t.gate_id,emission:$t.writer_id,
 pred:$t.pointer_predecessor,succ:$t.pointer_successor,sensitivity:$t.sensitivity,
 region:(if $pos==null then "absent" elif $pos==$end-7 then "instruction_final_token" elif $pos>=$end then "generated" elif $pos>=$end-6 then "marker" elif $pos==$fence-1 then "fact_final_period_before_blank_line" elif $pos>=$fence+2 then "question_or_instruction" else "fact_or_fence" end),
 gateprobe:([$t.candidates[]|select(.id==$t.gate_id)]|first),gprobe:([$t.candidates[]|select(.id==73)]|first),shortlist:$t.shortlist}] |
 {all:summary, by_input_period:(group_by(.marked)|map({key:(.[0].marked|tostring),value:summary})|from_entries)}
