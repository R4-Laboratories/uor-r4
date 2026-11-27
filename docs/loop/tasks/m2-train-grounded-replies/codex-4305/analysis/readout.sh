#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
grade=$(realpath -e "$1"); raw=$(realpath -e "$2"); out=$3
here=$(cd "$(dirname "$0")" && pwd -P)
mkdir "$out"
arms=(baseline G N GN wrong-G wrong-N GN-wrong-G GN-wrong-N GN-wrong-both)
for arm in "${arms[@]}"; do
    jq -enc --slurpfile r "$grade/regression/$arm.json" --slurpfile n "$grade/near/$arm.json" \
        --slurpfile f "$grade/fresh/$arm.json" --slurpfile rows "$grade/fresh/$arm.rows.jsonl" '
      $r[0] as $r | $n[0] as $n | $f[0] as $f |
      {arm:$f.arm,regression:($r|{correct:.signed,count:.pairs,lost:(.ids-.wins)}),
       near:($n|{correct:.full,count:.pairs,lost:(.ids-.full_wins)}),
       all_guards:($r.signed==4 and $n.full==24),
       fresh:($f|{rows,pairs,signed,signed_rows,full,full_rows,distinct_signed,same_governor_rows,
                   first_later,tied,absent,source_kinds,governor_rows,noun_rows,opposite_rows,heading_rows}),
       signed_pair_ids:$f.wins, full_pair_ids:$f.full_wins,
       signed_rows_with_wrong_complement:([$rows[]|select(.signed and (.full|not))]|
          {count:length,opposite_correct:map(select(.opposite))|length,heading_correct:map(select(.heading))|length}),
       family:[$f.by_family[]|{family,pairs,signed,full,tied,absent}],
       sign:[$f.by_sign[]|{sign,pairs,signed,full}],
       governor_order:[$f.by_order[]|{order,pairs,signed,full}]}' >> "$out/outcomes.jsonl"
    jq -enc --slurpfile rows "$grade/fresh/$arm.rows.jsonl" --slurpfile routes "$raw/fresh/routes.$arm.jsonl" \
        -f "$here/trace.jq" > "$out/trace-$arm.json"
done
jq -sc 'map({arm:.arm,guards:[.regression.correct,.near.correct],signed:.fresh.signed,full:.fresh.full,
   strict:.fresh.distinct_signed,first_later:.fresh.first_later})' "$out/outcomes.jsonl" > "$out/overview.json"
cp "$grade/fresh/transitions.json" "$out/transitions.json"
cp "$grade/controls.json" "$out/controls.json"
(cd "$out" && sha256sum -- *.json *.jsonl > SHA256SUMS)
