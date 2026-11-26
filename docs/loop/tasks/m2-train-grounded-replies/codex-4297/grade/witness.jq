def term($text; $span):
    if $span==null then null
    elif ($text|length)!=( $text|utf8bytelength) then error("fresh witness needs UTF-8 coordinate conversion")
    else {word:$text[$span.start:$span.end],bytes:[$span.start,$span.end]} end;
def sites($text;$bytes;$tokens):
    if $bytes==null then null else [range(0;$bytes|length) as $i|
        term($text;$bytes[$i])+{bpe:$tokens[$i]}] end;
map(select(.arm.full|not)) | sort_by(.ordinal) | first as $row |
if $row==null then null else
    $reference[0][$row.ordinal] as $ref | $ref.turns[1] as $text |
    if $row.case_id!=$ref.id or $row.prompt_sha256!=$ref.structural_prompt_sha256 then error("source identity changed") else
      {case_id:$row.case_id,pair:$row.pair_id,admission:$row.admission,distance:$row.distance,
       family:$row.family,first_source_error:$row.arm.first_source_error,
       gold:sites($text;$row.gold;$row.gold_tokens),chosen:sites($text;$row.arm.selection;$row.arm.tokens),
       governors:$row.arm.governors,expected_governors:[$ref.governors.affirmative,$ref.governors.excluded],
       ties:$row.arm.ties}
    end
end
