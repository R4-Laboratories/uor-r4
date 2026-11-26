def need($ok; $message): if $ok then . else error($message) end;
def extent: {minimum:min,maximum:max};

$premise[0].protected as $locked |
need(($locked|unique|length)==24 and length==2 and map(.round)==[1,2];
    "protected identity set or ordered round count changed") |
.[0].candidates[0].lost_protected as $missing |
need(($missing|unique|length)==8 and ($missing-$locked)==[];
    "ordered base did not lose the recorded eight original near identities") |
map(.round as $round | .candidates as $records |
    need($records[0].id==null and ($records[1:]|map(.id))==[range(68)];
        "ordered candidate IDs are incomplete or out of order") |
    need(all($records[]; (.lost_protected|sort)==(($locked-.near_full)|sort));
        "a recorded protected subset disagrees with the identities") |
    $records[0] as $base |
    ($records[1:] | map(. as $p | {
        id, protected_retained:($locked-($locked-$p.near_full)|length),
        ordered_base_misses_recovered:($missing-($missing-$p.near_full)),
        other_near_wins:($p.near_full-$locked), far_full:(.far_full|length), all_loss
    })) as $candidates |
    {round:$round,base:{protected_retained:(24-($base.lost_protected|length)),
        far_full:($base.far_full|length),all_loss:$base.all_loss},
     proposals:($candidates|length),
     protected_retained_extent:($candidates|map(.protected_retained)|extent),
     total_near_extent:($candidates|map(.protected_retained+(.other_near_wins|length))|extent),
     other_near_extent:($candidates|map(.other_near_wins|length)|extent),
     far_full_extent:($candidates|map(.far_full)|extent),
     all_loss_extent:($candidates|map(.all_loss)|extent),
     improved_loss:([$candidates[]|select(.all_loss<$base.all_loss)]|length),
     retained_every_protected:([$candidates[]|select(.protected_retained==24)]|length),
     recovered_original_misses:([$candidates[].ordered_base_misses_recovered[]]|unique),
     candidates:$candidates}) |
{original_protected:($locked|length), ordered_base_original_misses:$missing, rounds:.}
