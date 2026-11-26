def fail($v;$msg): if $v then . else error($msg) end;
def surface($text;$span):
 if $span.start<0 or $span.end<$span.start or $span.end>($text|utf8bytelength) then error("source outside original text")
 elif ($text|utf8bytelength)==($text|length) then $text[$span.start:$span.end]
 else [range(($text|length)+1) as $i|{index:$i,byte:($text[:$i]|utf8bytelength)}] as $offsets |
    ([$offsets[]|select(.byte==$span.start)|.index]) as $starts | ([$offsets[]|select(.byte==$span.end)|.index]) as $ends |
    if ($starts|length)==1 and ($ends|length)==1 then $text[$starts[0]:$ends[0]] else error("source cuts a UTF-8 character") end
 end;
def words($text;$spans): if $spans==null then null else [$spans[]|surface($text;.)] end;
def mark($left;$right):
 if $left then (if $right then "kept" else "lost" end) else (if $right then "gained" else "neither" end) end;
($reference[0]|INDEX(.id)) as $refs |
[range(256) as $i | $parent[$i] as $b | $matched[$i] as $f | $retained[$i] as $z | $refs[$z.case_id] as $r |
 fail($b.case_id==$f.case_id and $f.case_id==$z.case_id and $z.prompt_sha256==$r.structural_prompt_sha256;"source mismatch") |
 {case_id:$z.case_id,pair:$z.pair_id,admission:$z.admission,family:$z.family,distance:$z.distance,gold:$z.gold,gold_tokens:$z.gold_tokens,
 parent:$b.arm,far:$f.arm,retained:$z.arm,text:$r.turns[1]}] as $rows |
def snapshot:
  . as $v | {case_id,admission,gold:words($v.text;.gold),gold_bpe:.gold_tokens,
  parent:{terms:words($v.text;.parent.selection),bpe:.parent.tokens,first:.parent.first_source_error},
  far_only:{terms:words($v.text;.far.selection),bpe:.far.tokens,first:.far.first_source_error},
  retained:{terms:words($v.text;.retained.selection),bpe:.retained.tokens,first:.retained.first_source_error}};
($rows|group_by(.pair)|map({pair:.[0].pair,distance:.[0].distance,family:.[0].family,
 vs_parent:mark(all(.[];.parent.exact);all(.[];.retained.exact)),
 vs_far:mark(all(.[];.far.exact);all(.[];.retained.exact)),rows:.})) as $pairs |
{moves:($pairs|group_by(.family,.distance,.vs_parent,.vs_far)|map({family:.[0].family,distance:.[0].distance,
 vs_parent:.[0].vs_parent,vs_far_only:.[0].vs_far,pairs:length})),
 first_errors:($rows|group_by(.distance,.retained.first_source_error)|map({distance:.[0].distance,first:.[0].retained.first_source_error,n:length})),
 examples:($pairs|group_by(.distance,.vs_parent,.vs_far)|map(sort_by(.pair)|first|
 {distance,family,pair,vs_parent,vs_far,rows:(.rows|map(snapshot))}))}
