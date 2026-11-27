def rng(f):map(f)|map(select(.!=null))|if length>0 then {min:min,max:max} else null end;
def freq(f):group_by(f)|map({value:(.[0]|f),n:length})|sort_by(-.n);
def c(id):[.third_trace.candidates[]|select(.id==id)][0];
def gr(id): [.attribution.gate_positive_ranks[]|select(.[0]==id)|.[1]][0];
{rows:length,query_only:freq(.attribution.query_only_winner),minimum_ties:freq(.attribution.writer_minimum_ties),
margin:rng(.attribution.gate_score_margin),g_gate_rank:rng(gr(73)),eos_gate_rank:rng(gr(1)),
score_gate:rng(c(201).score),g_score:rng(c(73).score),eos_score:rng(c(1).score),
g_distance:freq(c(73).distance),eos_distance:freq(c(1).distance),query_count_LF:freq(c(201).terms[0]),
next_gate_available:freq(.next_id as $id|c($id).in_pool),next_gate_rank:rng(.next_id as $id|gr($id)),next_admitted:freq(.next_id as $id|c($id).admitted),next_distance:rng(.next_id as $id|c($id).distance),
eos_admitted:freq(c(1).admitted),recurrence_sums:(map(.third_trace.recurrence_sum)|unique|length),
paired_same_frame:(group_by(.pair_id)|map(select((map(.third_trace.recurrent_frames)|unique|length)==1))|length),
paired_same_raw_recurrence:(group_by(.pair_id)|map(select((map(.third_trace.recurrence)|unique|length)==1))|length),
pooled:freq(.third_trace.sensitivity),pointer_locations:freq(.attribution.pointer_location)}
