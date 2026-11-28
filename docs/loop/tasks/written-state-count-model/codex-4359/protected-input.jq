# Apply to #4357 snapped-proofs.jsonl after verifying the archived member digests.
# The two supporting files are JSONL slurped via --slurpfile cert and captured.
def demand($ok; $error): if $ok then . else error($error) end;
def pack: to_entries | map((.value % 64) * pow(64;.key)) | add;
. as $proofs
| demand(length == 1334 and ($cert|length) == 1334 and ($captured|length) == 2668; "wrong proof domain")
| (["pair_id","ordinal","group","gold_id","rival_id","gold_packed","rival_packed","gold_prior_q48","rival_prior_q48","carry"] | @tsv),
(range(0;1334) as $p | range(0;2) as $k
 | $proofs[$p] as $pair | $pair.first[$k] as $row | $cert[$p][$k] as $old
 | $captured[2*$p+$k] as $cap
 | demand($pair.kind == "opposing" and $pair.first == $pair.reread
       and $row.donor == $row.ordinal and $row.carry == $cap[1].carries[0]
       and ([$row.ordinal,$row.group,$row.gold_id,$row.rival_id] == [$old.ordinal,$old.group,$old.gold_id,$old.rival_id])
       and $row.ordinal == $cap[0].ordinal and $row.packed_slots == $cap[1].addresses[0]
       and $row.packed_slots == [($old.gold_slots|pack),($old.rival_slots|pack)]
       and $row.q48_priors == [$old.gold_prior,$old.rival_prior]
       and $row.q48_priors == $cap[1].priors; "historical pair or GPU reread differs")
 | ([$p,$row.ordinal,$row.group,$row.gold_id,$row.rival_id] + $row.packed_slots + $row.q48_priors + [$row.carry]) | @tsv)
