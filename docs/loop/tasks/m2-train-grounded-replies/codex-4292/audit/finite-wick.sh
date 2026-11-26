#!/usr/bin/env bash
# Read physical finite captures only. No CUDA launch occurs here.
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
words="$runs/20261126-codex-far-transport-margin/fit/finite-v4/fixture.words"
here=$(cd "$(dirname "$0")" && pwd)
finite=${1:?physical fixture version or restored provider finite directory}
build=${2:?independent build version}
version=${3:?new audit version}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
for id in "$build" "$version"; do [[ $id =~ ^v[1-9][0-9]*$ ]]; done
if [[ $finite =~ ^v[1-9][0-9]*$ ]]; then
    f="$s/fit/finite-$finite"
    inputs="$f/inputs.sha256"
else
    f=$(realpath "$finite")
    [[ $f == "$s/fit"/vast-*/finite ]]
    inputs=
fi
o="$s/audit/finite-$version"
v="$s/audit/build-$build"
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
verify "$f"; verify "$v"
if [[ -n $inputs ]]; then sha256sum --strict --quiet -c "$inputs"; fi
mkdir "$o" "$o/tampered"
auditor="$v/protected-transport-audit"
replay(){
    local arm=$1 base=$2 directory=$3 output=$4
    "$auditor" batch "$words" "$f/text.bin" "$f/labels.bin" "$f/parent.state" \
        "$f/expected-step-1.state" "$f/$base" "$directory" - "$o/$output"
}
replay first parent.state "$f/first" first.json
replay repeated parent.state "$f/repeated" repeated.json
replay second first/5.state "$f/second" second.json
cmp "$o/first.json" "$o/repeated.json"
for file in "$f/first/"*; do
    if [[ ${file##*/} != 0.edges.jsonl ]]; then ln -s "$file" "$o/tampered/"; fi
done
# Change reported gold distance without touching reported hinge, winners or state.
jq -c 'if .row==0 and .sign==0 then .before.gold_cost+=1 else . end' "$f/first/0.edges.jsonl" \
    > "$o/tampered/0.edges.jsonl"
if replay first parent.state "$o/tampered" false-pass.json 2> "$o/tampered-error.txt"; then
    exit 1
fi
test ! -e "$o/false-pass.json"
grep -Fq '0 signed edge 0 field before diverged' "$o/tampered-error.txt"
jq -ne --slurpfile first "$o/first.json" --slurpfile second "$o/second.json" '
    $first[0] as $f | $second[0] as $s |
    if $f.pass and $s.pass and $f.candidate_count==6 and $s.candidate_count==6 and
        $f.comparison_rows_per_state==6 and $s.comparison_rows_per_state==6 then
    {pass:true,batches:3,same_fresh_retry:true,rejected_tampered_edge:true,
      candidate_count_per_batch:6,checked_state_bytes:($f.complete_state_bytes_verified*2+$s.complete_state_bytes_verified),
      checked_signed_edges:($f.signed_edges_verified*2+$s.signed_edges_verified),
      checked_joint_rows:($f.joint_rows_verified*2+$s.joint_rows_verified)}
    else error("finite physical dimensions differ") end
' > "$o/checks.json"
sha256sum "$f/SHA256SUMS" "$v/SHA256SUMS" "$words" "$here/finite-wick.sh" > "$o/inputs.sha256"
(cd "$o" && sha256sum -- *.json *.txt inputs.sha256 tampered/0.edges.jsonl > SHA256SUMS)
