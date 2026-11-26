# Diagnose the already measured protected TRAIN trajectory by original address.
def require($ok; $message): if $ok then . else error($message) end;
def span: [.start,.end];
def address: {bytes:(.bytes|span),tokens:(.tokens|span)};
def histogram(f): group_by(f) | map({key:(.[0]|f|tostring),value:length}) | from_entries;
def surface($text; $a):
    def index_at($offset):
        if ($text|length)==($text|utf8bytelength) then $offset else
            [range(($text|length)+1) as $i | select(($text[:$i]|utf8bytelength)==$offset) | $i] |
            require(length==1; "address is not an original UTF-8 boundary") | .[0]
        end;
    if $a==null then null else
        $a.bytes as [$start,$end] |
        require(0 <= $start and $start < $end and $end <= ($text|utf8bytelength); "invalid byte address") |
        $a + {surface:$text[index_at($start):index_at($end)]}
    end;
def between($a; $b; $x):
    $x!=null and (($a.bytes[1]<=$x.bytes[0] and $x.bytes[1]<=$b.bytes[0]) or
                  ($b.bytes[1]<=$x.bytes[0] and $x.bytes[1]<=$a.bytes[0]));
def link_resolves($c):
    (if .word==null then .governor==null else .governor==($c.words[.word]|address) end) and
    (if .site==null then .noun==null else .noun==($c.sites[.site]|address) end);

def decorate($i; $stage; $ref; $c):
    $stage.rows[$i] as $row | $stage.ranks[$i] as $rank | $ref.turns[1] as $text |
    ([$ref.governors.affirmative,$ref.governors.excluded] | map({bytes:(.bytes|span),tokens})) as $expected_governors |
    ($ref.native_scope | map({bytes:[.bytes.start-1,.bytes.end-1],tokens:(.tokens|span)})) as $expected_sites |
    require($row.ordinal==$i and $rank.row==$i and $c.ordinal==$i and $row.case_id==$ref.id and $c.id==$ref.id and
            $row.prompt_sha256==$c.prompt_sha256 and $c.prompt_sha256==$ref.structural_prompt_sha256;
            "mixed TRAIN row identity differs") |
    [range(2) as $s | $stage.edges[2*$i+$s] as $edge | $edge.before as $m |
        $rank.links[$s] as $link | $rank.independent_links[$s] as $independent |
        ($c.words[$edge.governor_word] | address) as $governor |
        ($c.sites[$edge.gold_site] | address) as $gold |
        ($c.sites[$m.rival_site] | address) as $rival |
        require($edge.row==$i and $edge.sign==$s and $edge.eligible_for_fit and
                $governor==$expected_governors[$s] and $gold==$expected_sites[$s] and
                ($link|link_resolves($c)) and ($independent|link_resolves($c)) and
                $m.rival_site==$m.rivals[0] and $m.rival_ties==($m.rivals|length) and
                $m.margin==($m.rival_cost-$m.gold_cost) and $m.loss==([0,1-$m.margin]|max);
                "signed or joint source does not resolve to the recorded original address") |
        {sign:["affirmed","excluded"][$s], governor_correct:($link.governor==$governor),
         noun_correct:($link.noun==$gold), joint_uses_signed_rival:($link.noun==$rival),
         independent_uses_signed_rival:($independent.noun==$rival and $independent.governor==$governor),
         intervening:between($governor;$gold;$link.noun), governor:surface($text;$governor),
         chosen_governor:surface($text;$link.governor), gold_noun:surface($text;$gold),
         chosen_noun:surface($text;$link.noun), strongest_wrong:surface($text;$rival),
         joint_link_cost:$link.best, independent_noun:surface($text;$independent.noun),
         signed:($m|{gold_cost,gold_center,rival_cost,rival_center,rival_site,rival_ties,rivals,margin,loss,outcome})}
    ] as $links |
    {row:$i,case_id:$ref.id,pair:$ref.pair_id,admission:$ref.admission,family:$ref.family,
     strict_triplet:$row.arm.exact,full:$row.arm.full,both_governors:$row.arm.both_governors,
     first_error:$row.arm.first_source_error, correct_heading:($rank.tuple.addresses[2]==$expected_sites[2]),
     expected_triplet:($expected_sites|map(surface($text;.))),
     joint:{sites:($rank.tuple.addresses|map(surface($text;.))),best:$rank.tuple.best,
            rival:$rank.tuple.rival,ties:$rank.tuple.ties},links:$links};

def pair_counts(f):
    group_by(.pair) | map([.[] | select(f)]|length) |
    {zero:map(select(.==0))|length,one:map(select(.==1))|length,both:map(select(.==2))|length};
def sources:
    . as $rows | [$rows[].links[]|select(.noun_correct|not)] as $wrong |
    {prompts:length,both_governors:map(select(.both_governors))|length,
     correct_headings:map(select(.correct_heading))|length,
     first_errors:({none:0,signed_governor:0,no_eligible_tuple:0,joint_tie:0,affirmed_noun:0,excluded_noun:0,
                    heading:0,link_witness:0}+(histogram(.first_error))),
     first_errors_by_admission:(group_by(.admission)|map({key:.[0].admission,value:histogram(.first_error)})|from_entries),
     strict_triplet_pairs:pair_counts(.strict_triplet),full_pairs:pair_counts(.full),
     wrong_nouns:{count:($wrong|length),surfaces:($wrong|histogram(.chosen_noun.surface)),
       intervening:([$wrong[]|select(.intervening)]|length),
       same_surface_as_gold:([$wrong[]|select(.chosen_noun.surface==.gold_noun.surface)]|length),
       joint_matches_signed_rival:([$wrong[]|select(.joint_uses_signed_rival)]|length),
       independent_matches_signed_rival:([$wrong[]|select(.independent_uses_signed_rival)]|length),
       signed_margin:($wrong|histogram(.signed.margin)),hinge:([$wrong[].signed.loss]|add)}};
def edges($a; $b):
    [range($a|length) as $i | $a[$i] as $x | $b[$i] as $y |
     {before:$x.outcome,after:$y.outcome,margin_delta:($y.margin-$x.margin),
      hinge_delta:($y.loss-$x.loss),rival_changed:($x.rival_site!=$y.rival_site)}] |
    {count:length, transitions:histogram(.before+"->"+.after),
     flipped:map(select(.before!=.after))|length,
     margin_improved:map(select(.margin_delta>0))|length,
     margin_worsened:map(select(.margin_delta<0))|length,
     strongest_rival_changed:map(select(.rival_changed))|length,
     hinge_delta:map(.hinge_delta)|add};

[$parent[0],$first[0],$final[0]] as $stages |
require(all($stages[]; (.rows|length)==256 and (.ranks|length)==256 and (.edges|length)==512);
        "incomplete measured stages") |
[$stages[] as $stage | [range(128) as $i | decorate($i;$stage;$refs[0][$i];$coords[$i])]] as $all |
($all[0] | group_by(.pair) | require(length==64 and all(.[]; length==2 and (map(.admission)|sort)==["defer","take"]);
                                      "incomplete common far pairs") |
    map(select(any(.[];.full|not))|.[0].pair)) as $ids |
require(($ids|length)==24 and ($all|map([group_by(.pair)[]|select(all(.[];.full))|.[0].pair])) ==
        [$selection[0].parent.far_full,$selection[0].protected_first.far_full,$selection[0].protected.far_full];
        "joined full pairs differ from the sealed election") |
[$all[] | map(select(.pair as $id | $ids|index($id)!=null))] as $residual |
($residual[0] | map(select(.pair=="enamel-display/road/18") | .row)) as $witness |
require(($witness|length)==2; "witness must include both original prompts") |
{scope:"recorded mixed TRAIN, common far only",protected_moves:[$selection[0].protected_first.id,$selection[0].protected.id],
 residual_pairs:($ids|length),residual_pair_ids:$ids,
 stages:{parent:($residual[0]|sources),after_234:($residual[1]|sources),after_240:($residual[2]|sources)},
 joint_address_changes:([$all[0][] as $x | $all[2][$x.row] as $y |
     select($x.joint.sites!=$y.joint.sites or ($x.links|map(.chosen_governor))!=($y.links|map(.chosen_governor)))]|length),
 common_signed_edges: {
    first:edges([$all[0][].links[].signed];[$all[1][].links[].signed]),
    second:edges([$all[1][].links[].signed];[$all[2][].links[].signed]),
    total:edges([$all[0][].links[].signed];[$all[2][].links[].signed])},
 witness:{pair:"enamel-display/road/18",prompts:[$witness[] as $i |
    $all[0][$i] as $start | (if $start.admission=="defer" then 0 else 1 end) as $sign |
    $start | {row,case_id,admission,expected_triplet, chosen_triplet:.joint.sites,
              wrong_attachment:(.links[$sign]|{sign,governor,chosen_governor,gold_noun,chosen_noun,strongest_wrong,independent_noun}),
              trajectory:[$all[]|.[$i]|{signed:.links[$sign].signed,joint_cost:.joint.best,
                                       joint_runner_up:.joint.rival,joint_ties:.joint.ties}]}]},
 next_comparison:"In a controlled source-read diagnostic, mask only the winning intervening address and compare against masking a nonwinning intervening address. Freeze the governors, state, remaining original candidates and tuple bank. Record the new strongest signed rival and whether both opposed prompts recover the exact original triplet with both governors."}
