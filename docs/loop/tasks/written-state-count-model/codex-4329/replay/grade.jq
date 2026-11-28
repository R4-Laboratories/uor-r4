($truth|map({key:(.group|tostring),value:.})|from_entries) as $key
| if length!=15 or (map([.arm,.group])|unique|length)!=15 then error("case census") else . end
| map(. as $r | $key[.group|tostring] as $t | .client as $c | $c.native_occurrence_steps as $steps
   | if $c.prompt_ids!=$t.prefix or ($steps|map(.committed_id))!=$c.generated_ids or
        ($c.stop=="eos")!=($c.generated_ids[-1]==1) then error("prefix, commits, or EOS") else . end
   | {group,arm,first_expected:$t.reference[:4],first_emitted:$c.generated_ids[:4],reference_length:($t.reference|length),
      stop:$c.stop,exact:($t.reference==$c.generated_ids),decision:$steps[0].decision,
      first_address:$steps[0].source_pos,first_claim:$steps[0].claim_pos,first_copied:$steps[0].copied,
      second_address:$steps[1].source_pos,second_claim:$steps[1].claim_pos,second_copied:$steps[1].copied,
      first_wrong:(first(range(0;([$c.generated_ids,$t.reference]|map(length)|max)) as $i |
        select($c.generated_ids[$i]!=$t.reference[$i]) | {step:($i+1),actual:$c.generated_ids[$i],expected:$t.reference[$i]}) // null)})
