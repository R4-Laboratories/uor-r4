#!/usr/bin/env bash
# Mix no research data into this test. Re-index the first hand-computed CUDA example for a whole join.
set -euo pipefail
umask 077
test "$(hostname -s)" = wick
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-text-governor-links
fixture="$root/learner/finite-v2"
out=${1:?new Wick fixture directory}
[[ $out == "$root"/join-finite-v* ]]
here=$(cd "$(dirname "$0")" && pwd)
(cd "$fixture" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$out"
jq -nc 'def addr: {bytes:{start:.[0],end:.[1]},tokens:{start:.[2],end:.[3]}};
  range(256) as $i | {ordinal:$i,id:($i|tostring),prompt_sha256:("prompt"+($i|tostring)),
  words:([[10,11,20,21],[40,41,50,51],[0,1,10,11],[80,81,90,91]]|map(addr)),
  sites:([[24,26,35,36],[43,45,53,54],[7,9,18,19],[65,67,72,73],[69,71,76,77]]|map(addr)),
  tuples:[[0,1,3],[1,0,3],[2,1,3]]}' > "$out/map"
jq -sc 'def framed: .bytes.start+=1 | .bytes.end+=1;
  def gov: {bytes,tokens:[.tokens.start,.tokens.end]};
  map(. as $r | {id,structural_prompt_sha256:.prompt_sha256,case_id:.id,pair_id:(.ordinal/2|floor|tostring),
    governors:{affirmative:(.words[0]|gov),excluded:(.words[1]|gov)},
    native_scope:([0,1,3]|map($r.sites[.]|framed)),neutral_rival:(.sites[2]|framed),
    gold_binding:{affirmed:.sites[0].bytes,excluded:.sites[1].bytes,referenced:.sites[3].bytes}})' "$out/map" > "$out/ref"
for names in 'trained learned' 'governor_gap gap' 'native native' 'unfitted unfitted' \
  'endpoints corrupt' 'swapped flipped' 'correct correct' 'exchanged exchange'; do
    read -r input arm <<<"$names"
    head -1 "$fixture/$input.jsonl" | jq -c '. as $r | range(256) as $i | $r | .row=$i' > "$out/$arm"
done
grade(){
  jq -nce --slurpfile coords "$out/map" --slurpfile reference "$out/ref" \
    --slurpfile learned "$1" --slurpfile gap "$out/gap" --slurpfile correct "$out/correct" --slurpfile exchange "$out/exchange" \
    --slurpfile corrupt "$out/corrupt" --slurpfile flipped "$out/flipped" --slurpfile native "$out/native" --slurpfile unfitted "$out/unfitted" \
    -f "$here/rows.jq"
}
grade "$out/learned" | jq -es 'length==256 and all(.[]; .arms.learned.full and .arms.correct_link.full and .arms.exchanged_link.swapped and (.arms.governor_gap.full|not))' > "$out/positive.ok"
jq -c 'if .row==0 then .links[1]=.links[0] else . end' "$out/learned" > "$out/wrong"
if grade "$out/wrong" > "$out/wrong.stdout" 2> "$out/wrong.stderr"; then exit 1; fi
grep -Fq 'winning links disagree' "$out/wrong.stderr"
(cd "$out" && sha256sum positive.ok wrong.stderr > SHA256SUMS)
