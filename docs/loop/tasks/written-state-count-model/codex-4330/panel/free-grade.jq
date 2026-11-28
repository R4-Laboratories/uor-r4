($sources | INDEX(.case_id)) as $ids |
if ($ordinary | map(.case_id) | sort) != ($ids | keys) then
  error("free chat did not replay the eight sealed original prompts")
else
  [
    $ordinary[] | . as $r | $r.client as $c |
    if $c.prompt_ids != $ids[$r.case_id].prompt_ids
       or (($c.stop == "eos") != ($c.generated_ids[-1] == 1))
       or ($c.stop == "max_tokens" and ($c.generated_ids|length) != 16)
    then error("free chat prompt or trace changed for " + $r.case_id)
    else {case_id,arm,request_group:$ids[$r.case_id].kind,
      emitted:($c.generated_ids|length),stop:$c.stop,
      first_id:$c.generated_ids[0],source_claim:($c.native_occurrence_steps[0].claim_pos != null),
      authenticated_copy:$c.native_occurrence_steps[0].copied, generated_ids:$c.generated_ids}
    end
  ] as $rows |
  {schema:"uor-r4.whole-turn-free-chat-description/1",arm:$rows[0].arm,
    rows:$rows,
    eos:([$rows[]|select(.stop == "eos")]|length),
    capped:([$rows[]|select(.stop == "max_tokens")]|length),
    copied:([$rows[]|select(.authenticated_copy)]|length),
    distinct_id_sequences:($rows|map(.generated_ids)|unique|length)}
end
