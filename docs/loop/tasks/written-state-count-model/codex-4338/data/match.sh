#!/usr/bin/env bash
set -euo pipefail
umask 077
parent=$(realpath "${1:?original #4337 run or restored fixture}")
native=$(realpath "${2:?authenticated native-v2 TRAIN tape}")
fit=$(realpath "${3:?rebuilt interval-fit executable}")
out=$(realpath -m "${4:?fresh output directory}")
here=$(cd "$(dirname "$0")" && pwd)
cd "$parent"
sha256sum --quiet --strict -c - <<'PINS'
6bab351d9b84e225a0687158ace2595a27287750772f7f1cfc0176febbe047c8  fit-device-v1/features/train.jsonl
ba816f821a0a214dc34238b6410425dba06dfa08d8369424ef3f8b7208f17857  fit-device-v1/features/train.coverage.json
025314a90a9283b971bd7f91d7bb64dd81dd0ddebfbcf2c2b7572f86f1dca8ee  data/fit-v1/train-gold.jsonl
PINS
cd "$native"
sha256sum --quiet --strict -c - <<'PINS'
3aff549ec87ff0d8080f5aaa74804de39d0f81d8622d850f774b167e13718055  train.features.jsonl
9e5110e07c69a76aac6bcd0e2c48e2c04ef54158c92aae49e5153ad6ae8053b9  train-gold.jsonl
677f1672a05125b48bd15c86bf1e7bbfb864ebb2e43a74d56c93890d92cdb799  coverage.json
PINS
mkdir "$out"
cd "$out"
: > empty.jsonl
for mix in original natural; do
    mkdir "$mix"
    if [[ "$mix" == original ]]; then
        gold=$parent/data/fit-v1/train-gold.jsonl
        features=$parent/fit-device-v1/features/train.jsonl
        coverage=$parent/fit-device-v1/features/train.coverage.json
    else
        gold=$native/train-gold.jsonl
        features=$native/train.features.jsonl
        coverage=$native/coverage.json
    fi
    jq -rs --slurpfile coverage "$coverage" -f "$here/match-groups.jq" "$gold" > "$mix/groups.tsv"
    while IFS=$'\t' read -r kind ids; do
        # Case IDs have no whitespace. Hash the full ordered group, before scores.
        rank=$(printf 'uor-r4/4338/matched-train-v1\n%s\n' "$ids" | sha256sum | cut -d' ' -f1)
        printf '%s\t%s\t%s\n' "$rank" "$kind" "$ids"
    done < "$mix/groups.tsv" > "$mix/ranks.tsv"
    jq -Rn '[inputs|split("\t")|{rank:.[0],kind:.[1],ids:(.[2]|split(" "))}]' "$mix/ranks.tsv" |
      jq -r --arg mix "$mix" -f "$here/match-schedule.jq" > "$mix/case-ids"
    jq -cs --rawfile order "$mix/case-ids" 'INDEX(.case_id) as $by | $order|split("\n")[]| select(length>0)|$by[.]' \
        "$gold" > "$mix/train-gold.jsonl"
    jq -cs --rawfile order "$mix/case-ids" 'group_by(.case_id)|INDEX(.[0].case_id) as $by |
      $order|split("\n")[]|select(length>0)|$by[.][]' "$features" > "$mix/train.features.jsonl"
    jq -r '.start != null' "$mix/train-gold.jsonl" > "$mix/polarity"
    "$fit" coverage "$mix/train.features.jsonl" "$mix/train-gold.jsonl" > "$mix/coverage.json"
    jq -e '.positive==204 and .no_read==60 and (.cases|length)==264 and
        all(.cases[]; .gold_present and .swapped_gold_present!=false)' "$mix/coverage.json" >/dev/null
    for arm in geometry count; do
        "$fit" prepare-train "$mix/train.features.jsonl" "$mix/train-gold.jsonl" empty.jsonl \
          "$arm" "$mix/$arm.train.tape" > "$mix/$arm.inventory.json"
    done
    test "$(wc -c < "$mix/geometry.train.tape")" = "$(wc -c < "$mix/count.train.tape")"
done
cmp original/polarity natural/polarity
jq -n --slurpfile original original/coverage.json --slurpfile natural natural/coverage.json '
  def summary: .[0]|{prompts:(.cases|length),source:.positive,no_read,paired_rows:.paired,
    candidates_including_no_read:(.cases|map(.choices)|add)};
  {purpose:"Matched TRAIN comparison; optimizer not run",original:($original|summary),natural:($natural|summary),
   passes_if_registered:32,events_per_arm_if_registered:(32*264),model_weight_bytes_per_arm:8192,
   identical_polarity_order:true}' > evidence.json
sha256sum evidence.json */{case-ids,groups.tsv,ranks.tsv,polarity,train-gold.jsonl,coverage.json,train.features.jsonl,*.train.tape,*.inventory.json} > SEALED.sha256
