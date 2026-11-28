#!/usr/bin/env bash
# Rebuild pinned legacy containers and compare the ordinary client on TRAIN.
# Exercise v2 with fabricated state words and fabricated prompts only.
set -euo pipefail
umask 077

old=$(realpath "${1:?previous uor-chat}")
chat=$(realpath "${2:?new uor-chat}")
packer=$(realpath "${3:?new pack-native-interval}")
base=$(realpath "${4:?pinned balanced base}")
tokenizer=$(realpath "${5:?pinned tokenizer}")
prior=$(realpath "${6:?sealed previous cold TRAIN directory}")
out=$(realpath -m "${7:?unused output directory}")
source=$(realpath "${8:?source directory used for the build}")

digest() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { test "$(digest "$2")" = "$1"; }
bound_train_file() {
    local file=$1 hash
    hash=$(awk -v key="./$file" '$2 == key { print $1 }' "$prior/OUTPUTS.sha256")
    pin "$hash" "$prior/$file"
}
pin 21461b5af9c82d98f6f0c7f8780ae6f738bd601a52fe8a98edf25a7e07548381 "$old"
pin 5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$base"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer"
pin 4cdefd4a56276fce28cf489ed4373f51f3d697c9846df45c4a4323400340c0a7 "$prior/OUTPUTS.sha256"
train="$prior/natural.blind.jsonl"
bound_train_file natural.blind.jsonl
test "$(wc -l < "$train")" -eq 264

mkdir "$out"
cd "$out"
mkdir evidence
(cd "$source" && sha256sum Cargo.lock crates/uor-r4-integer/src/discrete_d11.rs \
    crates/uor-r4-integer/src/discrete_d11/native_interval.rs \
    crates/uor-r4-integer/src/bin/uor-chat.rs crates/uor-r4-integer/src/bin/pack-native-interval.rs \
    docs/loop/tasks/written-state-count-model/codex-4342/operator/replay.sh) > evidence/OPERATOR-SOURCE.sha256
cat > evidence/openings.jsonl <<'ROWS'
{"case_id":"both_neighborhoods","text":"Amber Keeper sent Falcon toward Green Heron. Which Keeper sent Falcon where?"}
{"case_id":"missing_question","text":"Keeper sent Falcon. Keeper?"}
{"case_id":"two_clauses","text":"Amber Keeper took Falcon south. Green Keeper took Heron north. Which Keeper took Heron where?"}
ROWS
jq -n --arg old "$(digest "$old")" --arg chat "$(digest "$chat")" --arg packer "$(digest "$packer")" \
    --arg base "$(digest "$base")" --arg tokenizer "$(digest "$tokenizer")" --arg train "$(digest "$train")" \
    '{previous_client_sha256:$old,client_sha256:$chat,packer_sha256:$packer,base_sha256:$base,
      tokenizer_sha256:$tokenizer,train_openings_sha256:$train,train_openings:264}' > evidence/inputs.json

say() {
    local client=$1 artifact=$2 row=$3 text
    shift 3
    text=$(jq -jr '.text, "."' <<< "$row")
    "$client" --discrete "$artifact" --tokenizer "$tokenizer" --say "${text%.}" \
        --protocol 2 --seed 0 --temperature 0 --max-tokens 3 --native-occurrence-trace "$@" |
        jq -cseS 'if length==1 and (.[0]|type)=="object" then .[0] else error("expected one trace") end'
}

for arm in geometry count; do
    bound_train_file "natural.$arm.actual.d11"
    bound_train_file "natural.$arm.actual.pack.json"
    model="$prior/natural.$arm.actual.d11"
    receipt="$prior/natural.$arm.actual.pack.json"
    "$old" --discrete "$model" --tokenizer "$tokenizer" --interval-features "$train" > "$arm.train.before.jsonl"
    "$chat" --discrete "$model" --tokenizer "$tokenizer" --interval-features "$train" > "$arm.train.after.jsonl"
    cmp "$arm.train.before.jsonl" "$arm.train.after.jsonl"
    jq -es --arg arm "$arm" 'length==528 and all(.[]; has("scores")==(.arm==$arm))' "$arm.train.after.jsonl" >/dev/null
    {
        head -n 1 "$train"
        cat evidence/openings.jsonl
    } | while IFS= read -r row; do
        say "$old" "$model" "$row" >> "$arm.say.before.jsonl"
        say "$chat" "$model" "$row" >> "$arm.say.after.jsonl"
    done
    cmp "$arm.say.before.jsonl" "$arm.say.after.jsonl"
    test "$(wc -l < "$arm.say.after.jsonl")" -eq 4

    header_bytes=$(od -An -tu8 -j8 -N8 "$model")
    dd if="$model" of="$arm.legacy.i16le" bs=1 skip=$((16 + header_bytes)) count=8192 status=none
    fit=$(jq -er .interval.fit_sha256 "$receipt")
    "$packer" "$arm" "$base" "$arm.legacy.i16le" "$arm.legacy.d11" "$fit" > "evidence/$arm.legacy.pack.json"
    cmp "$model" "$arm.legacy.d11"
    cmp "$receipt" "evidence/$arm.legacy.pack.json"
    sha256sum "$arm.train.before.jsonl" "$arm.train.after.jsonl" "$arm.say.before.jsonl" "$arm.say.after.jsonl" \
        > "evidence/$arm.legacy-replay.sha256"
done

"$chat" --discrete "$base" --tokenizer "$tokenizer" --interval-relation-features evidence/openings.jsonl \
    > evidence/base.diagnostic.jsonl
jq -es '
    length==15 and all(.[]; (has("scores")|not) and
        (if (.arm=="geometry" or .arm=="count") then (has("relation_endpoints")|not)
        else (.candidates|length)==(.relation_endpoints|length) end))
' evidence/base.diagnostic.jsonl >/dev/null
opening=$(head -n 1 evidence/openings.jsonl)
fixture=$(digest evidence/openings.jsonl)

for arm in directed_geometry directed_count swapped_geometry; do
    target=$(jq -cer --arg arm "$arm" '
      select(.arm==$arm and .case_id=="both_neighborhoods") |
      [.candidates[] | {span:{start,end},key:.keys[10]}] |
      group_by(.key) | map(select(length==1 and .[0].key>2560)) | first | first
    ' evidence/base.diagnostic.jsonl)
    key=$(jq -er .key <<< "$target")
    span=$(jq -cer .span <<< "$target")
    dd if=/dev/zero of="$arm.i16le" bs=8192 count=1 status=none
    printf '\144\000' | dd of="$arm.i16le" bs=1 seek=$((2 * key)) conv=notrunc status=none
    "$packer" "$arm" "$base" "$arm.i16le" "$arm.d11" "$fixture" > "evidence/$arm.pack.json"
    jq -e --arg arm "$arm" --arg fixture "$fixture" '
        .schema=="uor-r4.lut-discrete-native-interval/2" and .interval.arm==$arm and
        .interval.fit_sha256==$fixture and .interval.state_bytes==8192
    ' "evidence/$arm.pack.json" >/dev/null
    "$chat" --discrete "$arm.d11" --tokenizer "$tokenizer" --interval-relation-features evidence/openings.jsonl \
        > "$arm.cold.jsonl"
    jq -es --arg arm "$arm" --argjson span "$span" --argjson key "$key" '
        length==15 and all(.[]; has("scores")==(.arm==$arm)) and
        ([.[] | select(.arm==$arm and .case_id=="both_neighborhoods")] |
            length==1 and all(.[];
              .scores.selection.span==$span and .scores.selection.best_candidate_score==100 and
              .scores.selection.no_read_score==0 and
              ([.candidates[].keys[10] | if .==$key then 100 else 0 end]==.scores.candidates)))
    ' "$arm.cold.jsonl" >/dev/null
    jq -c --arg arm "$arm" 'select(.arm==$arm) | {case_id,scores}' "$arm.cold.jsonl" \
        > "evidence/$arm.cold-scores.jsonl"
    say "$chat" "$arm.d11" "$opening" > "evidence/$arm.say.json"
    jq -ce --argjson span "$span" --arg arm "$arm" '
      select(.model_schema=="uor-r4.lut-discrete-native-interval/2" and .decoding==.model_schema and
        .native_interval.identity.arm==$arm and .native_interval.selection.span==$span) |
      {prompt_ids, start:$span.start, end:$span.end, ids:.prompt_ids[$span.start:$span.end]}
    ' "evidence/$arm.say.json" > "evidence/$arm.supplied.json"
    say "$old" "$base" "$opening" --interval-supplied-source "evidence/$arm.supplied.json" > "evidence/$arm.writer.before.json"
    say "$chat" "$base" "$opening" --interval-supplied-source "evidence/$arm.supplied.json" > "$arm.writer.after.json"
    cmp "evidence/$arm.writer.before.json" "$arm.writer.after.json"
    field='{prompt_ids,generated_ids,stop, steps:.native_interval.steps, writer:.native_occurrence_steps}'
    jq -ceS "$field" "evidence/$arm.writer.before.json" > "$arm.writer.before.projection.json"
    jq -ceS "$field" "evidence/$arm.say.json" > "$arm.writer.v2.projection.json"
    cmp "$arm.writer.before.projection.json" "$arm.writer.v2.projection.json"
done

(cd evidence && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > REPLAY.sha256
mv REPLAY.sha256 evidence/REPLAY.sha256
