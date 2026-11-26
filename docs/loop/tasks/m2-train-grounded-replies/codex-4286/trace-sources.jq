# Describe sources the physical device chose, using the sealed original user bytes.
def surface($text; $span): $text[$span.start:$span.end];
def surfaces($text; $spans): [$spans[]|surface($text; .)];
def count_where(f): map(select(f))|length;
($reference[0]|INDEX(.case_id)) as $panel |
[.[] as $r | $panel[$r.case_id] as $p | $p.turns[1] as $text |
 if ($text|utf8bytelength) != ($text|length) then error("byte slicing requires ASCII on this frozen panel") else . end |
 $r.arms.governor_only as $g |
 {id:$r.case_id,pair:$r.pair_id,admission:$r.admission, family:$r.family,distance:$r.distance,
  chosen_distance:$g.governor_distance,authored_distance:($p.governors.affirmative.gap+$p.governors.excluded.gap),
  read_a_governor:any($g.tokens[0:2][];
    .==$p.governors.affirmative.native.tokens or .==$p.governors.excluded.native.tokens),
  kept_signed_earlier:[$g.tokens[0]==$r.gold_tokens[0],$g.tokens[1]==$r.gold_tokens[1]],
  text:$text,gold:surfaces($text;$r.gold),gold_tokens:$r.gold_tokens,
  picked:(if $g.selection then surfaces($text;$g.selection) else null end),picked_tokens:$g.tokens,
  cues:[surface($text; $p.governors.affirmative.bytes),surface($text; $p.governors.excluded.bytes)],
  cue_tokens:[$p.governors.affirmative.native.tokens,$p.governors.excluded.native.tokens],
  neutral:surface($text;$r.neutral_rival),neutral_tokens:$r.neutral_rival_tokens}] |
{all:{rows:length,closer_than_authored:count_where(.chosen_distance<.authored_distance),
        same_gap_as_authored:count_where(.chosen_distance==.authored_distance),
        read_a_governor_as_noun:count_where(.read_a_governor),
        neither_original_in_its_correct_role:count_where(.kept_signed_earlier==[false,false])},
 distances:(group_by(.distance)|map({distance:.[0].distance,rows:length,
        chosen_total_bpe_gap:(map(.chosen_distance)|{min:min,max:max}),
        authored_total_bpe_gap:(map(.authored_distance)|{min:min,max:max}),
        closer_than_authored:count_where(.chosen_distance<.authored_distance)})),
 examples:(group_by([.distance,.admission])|map(first))}
