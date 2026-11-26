def need($test;$why): if $test then . else error($why) end;
def gap($r;$key): $r.governor.tokens.start - $r[$key].tokens.end;
def expected_spacing:
    all(.[]; .governor_order=="after" and
       (if .selector=="first" then [gap(.;"requested"),gap(.;"rival")]==[25,18]
        elif .selector=="later" then [gap(.;"requested"),gap(.;"rival")]==[18,25]
        else false end));
def words: ascii_downcase|scan("[a-z0-9]+");
[$original[0][]|select(.family=="planning-review")|.turns[1]|words] + ["first","later"] |
unique as $vocabulary |
[$original[0][], $historical[0][], $derived[0][] | .structural_prompt_sha256] as $older |
need(($req|length)==8 and ($refs[0]|length)==8 and
     ([$req[]|[.fact,.sign,.selector]]|sort|unique|length)==8 and
     all($req[];.family=="planning-review" and (.fact=="school" or .fact=="road") and (.sign==1 or .sign== -1));
     "missing original TRAIN fact, sign or selector from the complete 2x2x2 roster") |
need(([$historic_requests[]|select(.historical_lost)]|length)==8 and
     ([$historic_requests[]|select(.historical_lost)]|expected_spacing) and ($req|expected_spacing);
     "expected historical BPE gaps 25 and 18 on original registered occurrences") |
need(([$refs[0][].structural_prompt_sha256]|unique|length)==8 and
     all($refs[0][]; .structural_prompt_sha256 as $digest | ($older|index($digest))==null);
     "supplement overlaps original TRAIN, historical or primary diagnostic prompts") |
need(all($refs[0][];.structural_history==$original[0][0].structural_history and
      (.turns[1] | [words] - $vocabulary | length)==0);
     "supplement imports new nonselector vocabulary or changes genuine first history") |
need($primary[0].train_derived.all.pairs==32 and
     ($primary[0].train_derived.matching_full_pair_signature|length)==0 and
     ($primary_before[0].classifications|group_by(.pair_id)|length)==32 and
     ($primary_before[0].classifications|group_by(.pair_id)|all(.[]; any(.[];.native!="requested")));
     "primary frozen trigger failed independent confirmation") |
{requests:8,pairs:4,eligible_geometry:"after governor, first 25 BPE and later 18 BPE",
 independent_primary_trigger:true,scope:"TRAIN-only matched historical distances"}
