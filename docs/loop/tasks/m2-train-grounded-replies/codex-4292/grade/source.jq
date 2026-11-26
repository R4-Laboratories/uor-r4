# Extract the earliest failing prompt of the chosen lost pair using the original user bytes.
def term($text;$bytes):
 if $bytes==null then null else
   def index_at($offset): if ($text|length)==($text|utf8bytelength) then $offset else
     [range(($text|length)+1) as $i|select(($text[:$i]|utf8bytelength)==$offset)|$i] |
       if length==1 then .[0] else error("invalid original UTF-8 boundary") end end;
   if $bytes.start<0 or $bytes.end<=$bytes.start or $bytes.end>($text|utf8bytelength) then error("source outside turn")
   else $text[index_at($bytes.start):index_at($bytes.end)] end end;
def sites($text;$b;$t):
 if $b==null then null else [range($b|length) as $i|{word:term($text;$b[$i]),bytes:$b[$i],bpe:$t[$i]}] end;
[.[]|select((.pair_id as $id|$lost|index($id))!=null and (.arm.full|not))]|sort_by(.ordinal)|first as $r |
if $r==null then null else ($reference[0][$r.ordinal]) as $ref | $ref.turns[1] as $text |
if $ref.id!=$r.case_id or $r.prompt_sha256!=$ref.structural_prompt_sha256 then error("witness identity changed") else
{pair:$r.pair_id,case_id:$r.case_id,ordinal:$r.ordinal,admission:$r.admission,first_source_error:$r.arm.first_source_error,
 gold:sites($text;$r.gold;$r.gold_tokens),chosen:sites($text;$r.arm.selection;$r.arm.tokens),
 governors:$r.arm.governors,expected_governors:[$ref.governors.affirmative,$ref.governors.excluded],
 ties:$r.arm.ties} end end
