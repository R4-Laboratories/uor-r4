($inputs | INDEX(.case_id)) as $keys
| if length == 36 and (map([.id,.arm])|unique|length) == 36 then . else error("missing transfer arm") end
| map(. as $row | $keys[.id] as $key | .client as $c
  | ($c.verified_span_release.steps) as $steps
  | if $c.prompt_ids == $key.prompt_ids and ($steps|map(.committed_id)) == $c.generated_ids then . else error("prompt or commit identity") end
  | ($steps|map(select(.commit=="copy") | .source_pos)) as $addresses
  | ($key.span != null) as $supported
  | (if $supported then [range($key.span.start;$key.span.end)] else [] end) as $gold
  | {arm:.arm, supported:$supported, verified:($steps|any(.event=="verified_end")), cap:($steps|any(.event=="cap")),
     delim:([16,1,0,201]|index($key.next_id)!=null), length:($gold|length),
     refused:($steps[0].event=="rejected"),
     exact_words:($supported and $addresses==$gold and $c.generated_ids[0:($gold|length)]==$key.expected[:-1]),
     exact:($supported and $c.stop=="eos" and $c.generated_ids==$key.expected),
     wrong:(if $supported then ([range(0;([($c.generated_ids|length),($key.expected|length)]|max)) as $i |
       select($c.generated_ids[$i]!=$key.expected[$i])|$i]|first) else null end),
     false_supplied:(($supported|not) and ($steps|any(.commit!="ordinary"))),
     actual_ordinary_copy:($c.authenticated_copy_steps|any(.copied))})
| group_by(.arm)
| map({arm:.[0].arm,
       supported:([.[]|select(.supported)]|length),
       exact_words:([.[]|select(.exact_words)]|length),
       exact_through_eos:([.[]|select(.exact)]|length),
       boundary_by_next_source_token:([.[]|select(.supported)]|group_by(.delim)|map({
           followed_by_legacy_delimiter:.[0].delim,rows:length,
           verified:([.[]|select(.verified)]|length),cap:([.[]|select(.cap)]|length),exact:([.[]|select(.exact)]|length)})),
       first_wrong_offsets:([.[]|select(.supported)|.wrong]|group_by(.)|map({offset:.[0],count:length})),
       unsupported:([.[]|select(.supported|not)]|length),
       unsupported_refusals:([.[]|select((.supported|not) and .refused)]|length),
       unsupported_supplied_commits:([.[]|select(.false_supplied)]|length),
       unsupported_ordinary_copies:([.[]|select((.supported|not) and .actual_ordinary_copy)]|length)})
