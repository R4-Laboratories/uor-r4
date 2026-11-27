#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 4 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${4%/*}") == zfs ]]
cold=$(realpath -e "$1")
build=$(realpath -e "$2")
panel=$(realpath -e "$3")
out=$4
codec=$cold/inputs/tokenizer.json
for member in "$build" "$panel/fresh" "$panel/near" "$panel/regression"; do
    (cd "$member" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd "$panel" && sha256sum --strict --quiet -c PANELS.sha256 guards.sha256 authority.sha256)
printf '%s  %s\n' 048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d "$cold/MEMBERS.sha256" |
    sha256sum --strict --quiet -c -
(cd "$cold" && awk '
    $2 ~ /^\.\/panel\/(train|fresh|regression|near)\/reference\.json$/ ||
    $2 ~ /^\.\/panel\/(regression|near)\/(ordered\/(text\.bin|coordinates\.jsonl)|(gold|requests)\.jsonl)$/ ||
    $2 == "./inputs/tokenizer.json" {print}
' MEMBERS.sha256 | sha256sum --strict --quiet -c -)
mkdir "$out"
for split in regression near; do
    cmp "$cold/panel/$split/ordered/text.bin" "$panel/$split/ordered/text.bin"
    cmp "$cold/panel/$split/ordered/coordinates.jsonl" "$panel/$split/ordered/coordinates.jsonl"
    cmp "$cold/panel/$split/reference.json" "$panel/$split/reference.json"
done
cmp "$cold/panel/near/gold.jsonl" "$panel/near/gold.jsonl"
for suffix in jsonl tsv; do
    cmp "$cold/panel/regression/requests.$suffix" "$panel/regression/requests.$suffix"
done
"$build/argument-audit" argument-census "$panel/fresh" "$codec" "$out/geometry.json"
"$build/argument-audit" geometry "$panel/fresh" "$codec" "$panel/fresh/endpoint/text.bin" "$out/paired-geometry.json"
"$build/argument-audit" control-census "$panel/fresh" "$codec" "$out/controls.json"
jq -e '.nominated_tuple_eligible == [256,256]' "$out/controls.json" > /dev/null
jq -n --slurpfile fresh "$panel/fresh/reference.json" \
    --slurpfile train "$cold/panel/train/reference.json" \
    --slurpfile prev "$cold/panel/fresh/reference.json" \
    --slurpfile near "$cold/panel/near/reference.json" \
    --slurpfile reg "$cold/panel/regression/reference.json" '
    ($train[0]+$prev[0]+$near[0]+$reg[0]) as $history |
    def ids($set;$key): [$set[] | .[$key]] | unique;
    def facts($set): [$set[] | .fact_spans[] | [.role,.value]] | unique;
    def overlap($a;$b): [$b[] as $x | select(any($a[]; . == $x)) | $x] | unique;
    {rows:($fresh[0]|length),families:ids($fresh[0];"family"),facts:ids($fresh[0];"fact"),
     compared_history_rows:($history|length),
     historical_overlap:([ "family","fact","id","case_id","structural_prompt_sha256","pair_id" ] |
         map(. as $key | {key:$key,value:overlap(ids($fresh[0];$key); ids($history;$key))}) | from_entries),
     historical_fact_values:overlap(facts($fresh[0]); facts($history))}
    | if .rows == 256 and (.families|length) == 4 and (.facts|length) == 8
      and all(.historical_overlap[]; length == 0) and (.historical_fact_values|length) == 0
      then . else error("fresh family, exact original, fact or history split failed") end
' > "$out/split.json"
sha256sum "$build/SHA256SUMS" "$cold/MEMBERS.sha256" "$codec" "$panel/PANELS.sha256" \
    "$panel"/{fresh,near,regression}/SHA256SUMS "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
