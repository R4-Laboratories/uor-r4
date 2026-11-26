# Appended to the pinned original-address verifier when staging the Wick methods.
($reference[0]) as $refs |
insist(($refs|length)==256 and ($coords|length)==256 and ($physical|length)==256 and
       ($refs|map(.id)|unique|length)==256; "incomplete source collection") |
range(256) as $i | $refs[$i] as $p | $coords[$i] as $c |
insist($c.ordinal==$i and $p.id==$c.id and $p.structural_prompt_sha256==$c.prompt_sha256;
       "physical and original case identity differ") |
{ordinal:$i,case_id:$p.id,pair_id:$p.pair_id,admission:$p.admission,family:$p.family,distance:$p.distance,
 block:(if $split=="mixed" or $split=="far" then (if $i<128 then "common" else "replacement" end) else "evaluation" end),
 prompt_sha256:$c.prompt_sha256,gold:[$p.gold_binding.affirmed,$p.gold_binding.excluded,$p.gold_binding.referenced],
 gold_tokens:($p.native_scope|map(.tokens)),arm:rank($kind;$physical[$i];$c;$p)}
