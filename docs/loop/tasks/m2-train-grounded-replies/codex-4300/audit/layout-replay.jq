def need($ok;$why):if $ok then . else error($why) end;
def join($before;$after):
    $before[0].classifications as $old | $after[0].classifications as $new |
    need(($old|length)==($new|length);"paired state request count changed") |
    [$old[] as $x | $new[$x.request] as $y |
        need([$x.request,$x.row,$x.pair_id,$x.selector,$x.sign]==[$y.request,$y.row,$y.pair_id,$y.selector,$y.sign];
           "paired state identity changed") |
        {pair_id:$x.pair_id,selector:$x.selector,sign:$x.sign,historical_lost:$x.historical_lost,
         row:$x.row,before:$x,after:$y}];
def signature:
    [.before,.after]|map([.local,.joint,.native,.native_governor_correct]);
def pairs:
    group_by(.pair_id) |
    map(sort_by(.selector) as [$f,$l] |
      need(length==2 and $f.selector=="first" and $l.selector=="later" and $f.sign==$l.sign;
          "independent first/later pair lacks either endpoint") |
      {pair_id:$f.pair_id,sign:$f.sign,
       before_correct:all(.[];.before.native=="requested"),
       after_correct:all(.[];.after.native=="requested"),
       before_with_governor:all(.[];.before.native=="requested" and .before.native_governor_correct),
       after_with_governor:all(.[];.after.native=="requested" and .after.native_governor_correct),
       signature:[$f,$l]|map(signature)});
(join($historic_before;$historic_after)|map(select(.historical_lost))) as $historical |
join($before;$after) as $layout |
need(($historical|length)==8 and ($layout|length)==8;"historical or layout denominator changed") |
($historical|pairs) as $groups |
($layout|pairs) as $current |
([$current[]|select(.before_correct and (.after_correct|not))]) as $lost |
([$lost[]|. as $p|select(any($groups[];.signature==$p.signature))]) as $full |
([$historical[]|select(.before.native=="requested" and .after.native!="requested")]) as $failed_rows |
{pairs:($current|length),rows:($layout|length),
 before_correct:([$current[]|select(.before_correct)]|length),after_correct:([$current[]|select(.after_correct)]|length),
 before_with_governor:([$current[]|select(.before_with_governor)]|length),
 after_with_governor:([$current[]|select(.after_with_governor)]|length),
 lost_pairs:($lost|length),matching_full_signatures:($full|length),
 matching_failed_later_rows:([$layout[] as $r|select(any($failed_rows[];signature==($r|signature)))]|length),
 results:$current}
