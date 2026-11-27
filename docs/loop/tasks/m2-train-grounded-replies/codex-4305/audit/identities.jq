def span: {bytes:[.bytes.start,.bytes.end],tokens:[.tokens.start,.tokens.end]};
def field($seen; $word; $noun):
    $seen.word == $word.index and $seen.governor == ($word | span)
    and $seen.site == $noun.index and $seen.noun == ($noun | span);
def at_site($tuple; $role; $noun):
    $tuple.sites[$role] == $noun.index and $tuple.addresses[$role] == ($noun | span);
def unique($r): $r.tuple.ties == 1 and $r.tuple.ordinal != null;
def pair($k): group_by(.pair) | map(select(all(.[]; .[$k])) | .[0].pair);
def requested:
    $requests | map(. as $q | $native[.row_index] as $r |
        (.sign | if . == 1 then 0 else 1 end) as $s |
        ($r.links[$s] | field(.;$q.governor;$q.requested)) as $focus |
        (unique($r) and $focus and at_site($r.tuple;$s;$q.requested)) as $signed |
        {pair:.pair_id, selector, original_row:.row_index,
         signed:$signed, strict_signed:($signed and $r.links[0].governor != $r.links[1].governor),
         full:($signed and field($r.links[1-$s];$q.opposite_governor;$q.opposite)
            and at_site($r.tuple;1-$s;$q.opposite) and at_site($r.tuple;2;$q.heading)),
         ties:$r.tuple.ties});
def near:
    $archived[0].members | map(. as $q | $native[.row_index] as $r |
        {pair:.pair_id, original_row:.row_index, admission,
         full:(unique($r) and $r.tuple.sites == $q.sites and $r.tuple.addresses == $q.site_addresses
              and all(range(0;2); . as $sign | $r.links[$sign] as $link |
                  $link.word == $q.governors[$sign] and $link.governor == $q.governor_addresses[$sign]
                  and $link.site == $q.sites[$sign] and $link.noun == $q.site_addresses[$sign])),
         ties:$r.tuple.ties});
(if $split == "near" then near else requested end) as $rows |
if ($rows | length) != ([8,48,256][if $split=="regression" then 0 elif $split=="near" then 1 else 2 end]) or
    any($rows | group_by(.pair)[]; length != 2)
then error("missing or repeated original pair member")
else $rows | {arm:$arm,rows:length,original_ids:(map(.pair)|unique),signed_pairs:pair("signed"),
    strict_signed_pairs:pair("strict_signed"),full_pairs:pair("full"),
    signed_rows:(map(select(.signed))|length),full_rows:(map(select(.full))|length),
    absent_rows:(map(select(.ties == 0))|length),tied_rows:(map(select(.ties > 1))|length),
    selector_outcomes:(if $split == "near" then null else
        group_by(.pair) | map(sort_by(.selector)|[.[].signed]|map(if . then "1" else "0" end)|join(""))
        | group_by(.) | map({key:.[0],value:length}) | from_entries
    end)}
end
