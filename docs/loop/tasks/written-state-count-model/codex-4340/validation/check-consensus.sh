#!/usr/bin/env bash
# Finite label-assembly checks. All roles, cases and passages here are synthetic.
set -euo pipefail
tmp=${1:?fresh test directory}
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$tmp"
cd "$tmp"
jq -nc 'range(67)|tostring as $k|{pair_id:$k,article:"fixture",passage_sha256:("paragraph-"+$k),
  family:(if .<23 then "two_answer" else "answerability" end),left:{case_id:("L"+$k)},right:{case_id:("R"+$k)}}' > src
jq -c '{pair_id,passage_sha256,left_case_id:.left.case_id,right_case_id:.right.case_id,left_grade:"supported",
  right_grade:(if .family=="two_answer" then "supported" else "unanswerable" end),
  left_reason:"fixture left",right_reason:"fixture right",distinct:true,distinct_reason:"fixture distinction"}' src > a
jq -c 'if .pair_id=="0" then .distinct=false elif .pair_id=="23" then .right_grade="false_premise" else . end' a > b
run() { jq -nc --slurpfile source src --slurpfile first a --slurpfile second "$1" -f "$here/consensus.jq"; }
run b | jq -se 'length==67 and ([.[]|select(.accepted)]|length)==65 and
  (.[0].reasons|index("contrast_not_uniquely_licensed")!=null) and
  (.[23].reasons|index("second_question_grade_disagreement")!=null)' >/dev/null
sed '1d' b > omitted
jq -c 'if .pair_id=="1" then .right_case_id="L1" else . end' b > wrong
for file in omitted wrong; do
    if run "$file" > "$file.stdout" 2> "$file.stderr"; then
        echo "accepted corrupted review $file" >&2
        exit 1
    fi
done
