def n(f): map(select(f))|length;
def positions($field):
  map(.[$field].wrong | select(.!=null)) | group_by([.position,.expected,.actual]) |
  map(.[0]+{count:length});
def summarize($rows;$base):
  $rows | {
   replies:length,returned_eos:n(.returned.exact),first_eos:n(.first.exact),
   pairs_both_eos:(group_by(.pair)|n(all(.[];.returned.exact))),
   correct_prefix_at_cap:n(.returned.cap_prefix),retry:n(.retry),
   first:positions("first"),returned:positions("returned"),
   moved_relative_to_native:([.[]|{old:$base[.id].returned.wrong.position,new:.returned.wrong.position}] |
       group_by([.old,.new]) | map(.[0]+{count:length})),
   before_after_retry_changed:n(.first.wrong!=.returned.wrong)
  };
. as $all | (map(select(.arm=="native"))|INDEX(.id)) as $base |
if length!=1280 or ($base|length)!=256 then error("ordinary independent census is incomplete") else
{arms:(group_by(.arm)|map({key:.[0].arm,value:summarize(.;$base)})|from_entries),
 by_distance:(group_by([.arm,.distance])|map({arm:.[0].arm,distance:.[0].distance,
   pairs_both_eos:(group_by(.pair)|n(all(.[];.returned.exact)))}))} end
