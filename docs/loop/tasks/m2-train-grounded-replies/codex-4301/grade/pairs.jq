def tally: group_by(.)|map({key:.[0],value:length})|from_entries;
def summarize:
 sort_by(.pair_id,.selector)|group_by(.pair_id)|map(
   if length!=2 or .[0].selector!="first" or .[1].selector!="later" or .[0].sign!=.[1].sign or
      .[0].governor!=.[1].governor or .[0].requested!=.[1].rival or .[1].requested!=.[0].rival
   then error("pair identity or source relocation") else
   {id:.[0].pair_id,family:.[0].family,source:all(.[];.source),full:all(.[];.full),
    local:all(.[];.local.relation=="win"),joint:all(.[];.joint.relation=="win"),
    beats_third:all(.[];.gold_vs_third=="win" or .gold_vs_third=="unopposed"),
    strict_third:all(.[];.gold_vs_third=="win"),third_present:all(.[];.third_free.cost!=null),
    first_source:.[0].source,later_source:.[1].source,missing_margins:([.[]|select(.local.margin==null)]|length),
    worst_deficit:([.[]|select(.local.margin!=null)|[0,1-.local.margin]|max]|max//0),
    sum_deficit:([.[]|select(.local.margin!=null)|[0,1-.local.margin]|max]|add//0)}
   end);
. as $rows | summarize as $p |
{pairs:($p|length),rows:($rows|length),source_ids:[$p[]|select(.source)|.id],full_ids:[$p[]|select(.full)|.id],
 source:([$p[]|select(.source)]|length),full:([$p[]|select(.full)]|length),local:([$p[]|select(.local)]|length),
 joint:([$p[]|select(.joint)]|length),strict_third:([$p[]|select(.strict_third)]|length),
 beats_third:([$p[]|select(.beats_third)]|length),third_present:([$p[]|select(.third_present)]|length),
 missing_margins:([$p[].missing_margins]|add//0),worst_deficit:([$p[].worst_deficit]|add//0),
 sum_deficit:([$p[].sum_deficit]|add//0),source_rows:([$rows[]|select(.source)]|length),
 first_rows:([$p[]|select(.first_source)]|length),later_rows:([$p[]|select(.later_source)]|length),
 third_outcomes:([$rows[].gold_vs_third]|tally),native_choices:([$rows[].native.choice]|tally),
 third_masked_local_wins:([$rows[]|select(.local.relation=="win" and .native.choice=="third site")]|length),
 by_family:($p|group_by(.family)|map({family:.[0].family,pairs:length,source:([.[]|select(.source)]|length),
     full:([.[]|select(.full)]|length),local:([.[]|select(.local)]|length),third:([.[]|select(.strict_third)]|length)}))}
