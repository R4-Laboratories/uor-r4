# Private courtesy export of measured rows. Use jq -nr with four --slurpfile inputs.
def key: .pair_id + "/" + (.side|tostring);
def index_rows($x): reduce $x[] as $r ({}; .[$r|key]=$r);
def mass: if .pool_member then .mass else 0 end;
index_rows($panel) as $inputs | index_rows($labels) as $label |
(["checkpoint","cohort","group_id","pair_id","side","question","reference_answer","first_content_id",
  "prior_rank","gate_rank","gate_rank_with_prior_tail","same_id_rank_under_other_question","space_gate_rank",
  "gold_eligible_mass","same_id_eligible_mass_under_other_question","space_eligible_mass","gold_in_gate_pool",
  "read_code","other_read_code","recurrence_code","other_recurrence_code",
  "successor_position","successor_id","gate_winner","writer_id","gold_writer_rank","space_writer_rank"] | @csv),
([["reasoning-first",$reasoning],["answer-first",$answer]][] as [$arm,$rows] |
    index_rows($rows) as $tr |
    ($rows|sort_by(.cohort,.pair_id,.side))[] as $r |
    ($r|key) as $id |
    $tr[$r.pair_id + "/" + ((1-$r.side)|tostring)] as $other |
    if ($r.target!=$inputs[$id].target or $r.target!=$other.probes.rival.id or $label[$id].gold[0]!=$r.target)
    then error("panel/trace/label mismatch at "+$id)
    else
       [$arm,$r.cohort,$inputs[$id].group_id,$r.pair_id,$r.side,$label[$id].question,$label[$id].reply,$r.target,
        $r.probes.gold.prior_rank,$r.probes.gold.gate_rank,$r.probes.gold.gate_full_rank,$other.probes.rival.gate_full_rank,$r.probes.space.gate_rank,
        ($r.probes.gold|mass),($other.probes.rival|mass),($r.probes.space|mass),$r.probes.gold.pool_member,
        $r.read_code,$other.read_code,$r.recurrence_code,$other.recurrence_code,$r.source_position,$r.source_token,
        $r.gate_selected,$r.emitted,$r.probes.gold.writer_rank,$r.probes.space.writer_rank] | @csv
    end)
