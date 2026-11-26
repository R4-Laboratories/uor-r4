def check($ok; $why): if $ok then . else error($why) end;
def slim($w):
    if $w==null then null
    else $w | {ordinal,pair,case,family,fact,admission,first_changed_source,
        before:(.before|{first_source_error,tuple_best,tuple_rival,ties,eligible}),
        after:(.after|{first_source_error,tuple_best,tuple_rival,ties,eligible})} +
        (if .first_changed_source==null then {gold:.original_nouns,before_chosen:.before.nouns,after_chosen:.after.nouns} else {} end)
    end;
def stratum:
    {distance,address_changes,rows,pairs,pairs_with_any_tie,pairs_with_any_no_eligible,
     first_differing_source:slim(.first_differing_source),first_full_result_error:slim(.first_after_error)};
def selectors:
    {role,gold_changes,before,after,gained,lost};
def election($arm; $domain):
    if $arm.elected_reads_byte_identical_to_full and $arm.elected_candidate=="full" then
        {full_is_elected:true}
    elif $arm.elected_candidate==null and $arm.training_diagnostic_candidate=="base" and
        ($arm.deployment_fallback | .geometry=="endpoint" and .state=="protected" and .training_bundle_verified and
            ($domain=="mixed" or .held_reads_byte_identical_to_endpoint_baseline)) then
        {full_is_elected:false,elected_candidate:null,training_diagnostic_candidate:"base",
         deployment_fallback:$arm.deployment_fallback}
    else error("expected the full election or a verified protected endpoint fallback") end;
def study($report; $domain):
    ($report.domains|map(select(.domain==$domain))) as $domains |
    check(($domains|length)==1; "expected a single sealed domain") |
    $domains[0].arms | map(. as $arm |
        check(($arm.comparisons|length)==1 and $arm.comparisons[0].from=="frozen" and $arm.comparisons[0].to=="full";
              "expected the frozen-to-full comparison") |
        $arm.comparisons[0] as $delta |
        {key:$arm.geometry,value:(election($arm;$domain) +
         (if $domain=="mixed" then
             $delta.training | {original_protected,missed_after,newly_lost,recovered,first_protected_recovery:slim(.first_protected_recovery),
                 far_wins:([.far_wins[].pairs[]]|length),far_wins_gained,far_wins_beyond_protected_reference,
                 first_far_gain:slim(.first_far_gain),residual_far_example:slim(.residual_far_example)}
          else
             {strata:($delta.strata|map(stratum))} +
             (if $domain=="contrast" then {signed_selectors:($delta.signed_selectors|map(selectors))} else {} end)
          end))}) | from_entries;

{schema:"uor-r4.codex-4299.source-attribution-compact/1",
 source_reports:{training_sha256:$train_sha,held_sha256:$held_sha},
 training:study($train[0];"mixed"),held:study($held[0];"held"),contrast:study($held[0];"contrast")}
