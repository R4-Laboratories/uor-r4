# Retrospective loss localization, before any fit. Counts joint score aliases
# within each individual prompt, where the query is fixed for every rival.
($keys|INDEX(.case_id)) as $truth |
[inputs as $r | $truth[$r.case_id] as $k | select($k.target!=null) |
 ($r.candidates|map(select(.pos==$k.target.pos))|first) as $g |
 [$r.candidates[]|select(.pos!=$g.pos and .score >= $g.score)] as $wrong |
 {case_id:$k.case_id, pair:$k.pair_id, class:$k.kind, source:$g.pos,
   query:$r.query_code, source_code:$g.recurrence_code, mask:$g.real_mask,
   rivals_outranking_or_tying:($wrong|length),
   same_mask:([$wrong[]|select(.real_mask==$g.real_mask)]|length),
   same_full_site_code:([$wrong[]|select(.recurrence_code==$g.recurrence_code)]|length),
   beyond_table_range:([$wrong[]|select(.score-$g.score>=8192)]|length)}]
