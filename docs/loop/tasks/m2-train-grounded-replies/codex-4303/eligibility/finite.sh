#!/usr/bin/env bash
set -euo pipefail
export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0 LC_ALL=C
lex=$(realpath -e "$1"); scorer=$(realpath -e "$2"); f=$(realpath -e "$3"); out=$4
mkdir "$out"; cd "$out"
printf '0\t0\n1\t0\n2\t0\n3\t0\n' > labels.tsv
"$lex" fixture-seed "$f/original.state" blocked.state
"$lex" learn "$f/text.bin" labels.tsv blocked.state learn
"$lex" learn "$f/text.bin" labels.tsv blocked.state repeat
diff -r learn repeat
for id in blocked learn/e1 learn/e2 learn/e3 learn/e4 learn/wrong; do
    "$lex" probe "$f/text.bin" "$id.state" "${id##*/}.jsonl"
    printf '%s\t%s.state\n' "${id##*/}" "$id" >> models.tsv
done
"$scorer" "$f/fixture.words" "$f/text.bin" "$f/requests.tsv" models.tsv observed
cmp observed/e4.native.jsonl "$f/original.native.jsonl"
jq -ens --slurpfile b blocked.jsonl --slurpfile a e1.jsonl --slurpfile m e2.jsonl --slurpfile t e3.jsonl \
    --slurpfile e e4.jsonl --slurpfile w wrong.jsonl --slurpfile fit learn/fit.json \
    --slurpfile raw observed/e4.costs.jsonl --slurpfile before observed/blocked.costs.jsonl '
    ($fit[0].median[0]==-16 and $fit[0].rows==4 and ([$b,$a,$m,$t,$e,$w]|all(.[];length==4))) and
    (all(range(4); . as $i|[$b,$a,$m,$t,$e,$w]|map(.[$i].words[0][:2])==[[32,32],[24,32],[16,32],[8,32],[0,32],[32,0]])) and
    (all($raw[]; . as $r|$r.sites[$r.requested].local!=null)) and
    (all($before[]|select(.sign==1); . as $r|$r.sites[$r.requested].local==null))' > passed.json
grep -qx true passed.json
if "$lex" cross blocked.state learn/e4.state learn/e4.state invalid.state > bad.log 2>&1; then exit 1; fi
grep -q 'factors overlap' bad.log
printf '0\t0\n0\t0\n' > repeated.tsv
if "$lex" learn "$f/text.bin" repeated.tsv blocked.state invalid-fit > invalid.log 2>&1; then exit 1; fi
grep -q 'invalid or repeated labeled row' invalid.log
