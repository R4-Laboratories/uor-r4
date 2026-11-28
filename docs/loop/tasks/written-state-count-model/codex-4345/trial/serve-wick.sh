#!/usr/bin/env bash
# Run only after the top session seals the complete fit and releases held serving.
set -euo pipefail
umask 077

worker() {
    local chat=$1 tok=$2 artifact=$3 sha=$4 rows=$5 kind=$6 mode=$7 dir=$8 index=$9
    local row prefix text out
    row=$(sed -n "$((index + 1))p" "$rows")
    prefix=$(jq -ce '.prefix | select(type == "array" and length > 0)' <<<"$row")
    local flags=(--discrete "$artifact" --protocol 2)
    case "$kind" in
        boundary)
            flags+=(--max-tokens 1 --prompt-ids "$(jq -r 'map(tostring) | join(",")' <<<"$prefix")" --ordinary-termination-trace)
            ;;
        open)
            jq -e --argjson index "$index" '.index == $index and (.user | type == "string")' <<<"$row" >/dev/null
            # Preserve every trailing newline in the original user message.
            text=$(jq -jr '.user,"."' <<<"$row")
            flags+=(--max-tokens 192 --tokenizer "$tok" --say "${text%.}")
            if [[ $mode == base ]]; then flags+=(--ordinary-termination-trace); fi
            ;;
        *) return 1 ;;
    esac
    if [[ $mode != base ]]; then flags+=(--ordinary-termination "$mode"); fi
    out=$dir/$index
    "$chat" "${flags[@]}" >"$out.partial" 2>"$out.stderr"
    jq -e --arg sha "$sha" --argjson prefix "$prefix" --arg kind "$kind" '
        .artifact_sha256 == $sha and .prompt_ids == $prefix
        and (.generated_ids | length > 0)
        and if $kind == "boundary" then
            (.generated_ids | length == 1) and (.ordinary_termination.steps | length == 1)
        else .stop == "eos" or .stop == "max_tokens" end
    ' "$out.partial" >/dev/null
    mv -- "$out.partial" "$out.json"
}

if [[ ${1-} == --worker ]]; then
    shift
    worker "$@"
    exit
fi

[[ $(hostname -s) == wick ]]
self=$(realpath "$0")
tools=$(dirname "$self")
r=$(realpath "${1:?registered run root}")
bin=$(realpath "${2:?sealed ordinary client and packer directory}")
stage=$(realpath "${3:?sealed stage-v1}")
trial=$(realpath "${4:?trial executable}")
fit=$(realpath "${5:?sealed GPU fit root}")
out=${6:?new result directory}
base=$(dirname "$r")/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
tok=$(dirname "$r")/20261127-codex-visible-answer/panel/inputs/tokenizer.json
base_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
tok_sha=a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
hash() { sha256sum "$1" | cut -d' ' -f1; }
pin "$base_sha" "$base"
pin "$tok_sha" "$tok"
pin a98c09c8a09cc897beff0406b46e720fdc57308d11c2344d71d723ee8f6172c3 "$stage/FILES.sha256"
pin a1e0e65482c642c03f30b83d7d23bad0b93ca82d5d5232342b779886f578879f "$fit/FIT-OUTPUTS.sha256"
(cd "$fit"; sha256sum --strict --quiet -c FIT-OUTPUTS.sha256)
(cd "$fit"; sha256sum --strict --quiet -c FIT-INPUTS.sha256)
for executable in "$bin/uor-chat" "$bin/pack-native-interval" "$trial"; do
    test -x "$executable"
    pin "$(awk -v name="$executable" '$2 == name {print $1}' "$fit/TOOLS.sha256")" "$executable"
done
mkdir -m 700 "$out"
out=$(realpath "$out")
cd "$out"
mkdir -m 700 blind responses grades
for name in held.open.jsonl held.prefix.jsonl; do
    # No labels or answer text are staged for generation.
    want=$(awk -v name="$name" '$2 == name {print $1}' "$stage/FILES.sha256")
    cp -- "$stage/$name" "blind/$name"
    pin "$want" "blind/$name"
done
[[ $(wc -l <blind/held.open.jsonl) -eq 90 && $(wc -l <blind/held.prefix.jsonl) -eq 94 ]]
sha256sum blind/*.jsonl > BLIND.sha256
sha256sum "$self" "$tools/summarize.jq" "$trial" "$bin/uor-chat" "$bin/pack-native-interval" "$fit/FIT-OUTPUTS.sha256" "$stage/FILES.sha256" > TOOLS.sha256

dd if=/dev/zero of=zero.words bs=8192 count=1 status=none
"$bin/pack-native-interval" ordinary-termination "$base" "$base_sha" geometry \
    zero.words zero.d11 "$(hash zero.words)" >zero.pack.json
zero_sha=$(jq -er .artifact_sha256 zero.pack.json)
pin "$zero_sha" zero.d11

runs=(geometry.1 geometry.2 ordered.1 ordered.2)
declare -A artifacts=([base]="$base" [off]="$fit/geometry.1.d11" [zero]="$out/zero.d11")
declare -A pins=([base]="$base_sha" [zero]="$zero_sha")
for run in "${runs[@]}"; do
    artifacts[$run]="$fit/$run.d11"
    pins[$run]=$(jq -er '.artifact_sha256' "$fit/$run.pack.json")
    # FIT-OUTPUTS already verified the artifact bytes and packer receipt.
    [[ $(awk -v name="$run.d11" '$2 == name {print $1}' "$fit/FIT-OUTPUTS.sha256") == "${pins[$run]}" ]]
done
pins[off]=${pins[geometry.1]}

capture() {
    local kind=$1 count=$2 rows=$3 run mode index
    shift 3
    for run; do mkdir -p "responses/$run/$kind"; done
    for ((index = 0; index < count; index++)); do
        for run; do
            mode=generic
            if [[ $run == base || $run == off ]]; then mode=$run; fi
            printf '%s\0' "$bin/uor-chat" "$tok" "${artifacts[$run]}" "${pins[$run]}" "$out/blind/$rows" \
                "$kind" "$mode" "$out/responses/$run/$kind" "$index"
        done
    done | xargs -0 -r -n 9 -P 4 "$self" --worker
    for run; do
        (cd "responses/$run/$kind"; sha256sum -- *.json > FILES.sha256)
    done
}

echo 'Serving 94 target-blind boundary prefixes per arm'
capture boundary 94 held.prefix.jsonl off zero "${runs[@]}"
echo 'Serving 90 target-blind ordinary openings per arm'
capture open 90 held.open.jsonl base off zero "${runs[@]}"
find responses -name FILES.sha256 -print0 | sort -z | xargs -0 sha256sum > RESPONSES.sha256
test "$(wc -l <RESPONSES.sha256)" -eq 13
sha256sum --strict --quiet -c BLIND.sha256

# All generation is sealed before opening the gold decisions and replies.
(cd "$stage"; sha256sum --strict --quiet -c FILES.sha256)
for run in off zero; do
    for ((i = 0; i < 90; i++)); do
        jq -e --slurpfile base "responses/base/open/$i.json" '
            .prompt_ids == $base[0].prompt_ids and .generated_ids == $base[0].generated_ids
            and .stop == $base[0].stop
        ' "responses/$run/open/$i.json" >/dev/null
    done
    jq -en --slurpfile captured "$fit/held.keys.jsonl" '$captured | length == 94' >/dev/null
    for ((i = 0; i < 94; i++)); do
        jq -e --slurpfile caps "$fit/held.keys.jsonl" --argjson i "$i" --arg run "$run" '
            $caps[$i] as $c | .ordinary_termination.steps[0] as $s
            | .generated_ids == [$c.ordinary_id] and $s.ordinary_id == $c.ordinary_id
            and $s.committed_id == $c.ordinary_id and ($s.promoted | not)
            and $s.geometry_keys == $c.geometry_keys and $s.ordered_keys == $c.ordered_keys
            and $s.prefix_start == 0 and $s.prefix_len == $c.prefix_len
            and $s.writer.read_vector == $c.read_vector and $s.writer.read_code == $c.read_code
            and $s.writer.recurrence_code == $c.recurrence_code
            and (if $run == "zero" then $s.score == 0 else $s.score != null end)
        ' "responses/$run/boundary/$i.json" >/dev/null
    done
done
for run in "${runs[@]}"; do
    "$trial" decisions "$stage" "$fit/held.keys.jsonl" "$fit/$run.scores" "$fit/$run.words" \
        "responses/$run/boundary" "${run%.*}" "${pins[$run]}" "$tok" "grades/$run.decisions.jsonl"
    jq -sc 'map([.index,.target,.native,.fixed,.native_correct,.fixed_correct])' \
        "grades/$run.decisions.jsonl" >"grades/$run.baselines.json"
    cmp "grades/geometry.1.baselines.json" "grades/$run.baselines.json"
done
for run in base off zero "${runs[@]}"; do
    "$trial" grade "$stage" "responses/$run/open" "${pins[$run]}" "$tok" native "grades/$run.writers.jsonl"
done
"$trial" grade "$stage" responses/base/open "$base_sha" "$tok" fixed grades/fixed.writers.jsonl

for run in "${runs[@]}"; do
    jq -sc -L "$tools" 'include "summarize"; learned' "grades/$run.decisions.jsonl" >"grades/$run.boundary.json"
done
for run in base fixed; do
    jq -sc -L "$tools" --arg mode "$run" 'include "summarize"; baseline($mode)' \
        grades/geometry.1.decisions.jsonl >"grades/$run.boundary.json"
done
for run in base fixed off zero "${runs[@]}"; do
    jq -sc -L "$tools" --slurpfile base grades/base.writers.jsonl 'include "summarize"; ordinary($base)' \
        "grades/$run.writers.jsonl" >"grades/$run.ordinary.json"
done
for run in base zero "${runs[@]}"; do
    jq -nc --arg name "$run" --arg sha "${pins[$run]}" --argjson bytes "$(stat -c %s "${artifacts[$run]}")" \
        --argjson tokenizer_bytes "$(stat -c %s "$tok")" '$ARGS.named' >>ARTIFACTS.jsonl
done
jq -ns --slurpfile artifacts ARTIFACTS.jsonl \
    --slurpfile base_b grades/base.boundary.json --slurpfile fixed_b grades/fixed.boundary.json \
    --slurpfile g1_b grades/geometry.1.boundary.json --slurpfile g2_b grades/geometry.2.boundary.json \
    --slurpfile o1_b grades/ordered.1.boundary.json --slurpfile o2_b grades/ordered.2.boundary.json \
    --slurpfile base grades/base.ordinary.json --slurpfile fixed grades/fixed.ordinary.json \
    --slurpfile off grades/off.ordinary.json --slurpfile zero grades/zero.ordinary.json \
    --slurpfile g1 grades/geometry.1.ordinary.json --slurpfile g2 grades/geometry.2.ordinary.json \
    --slurpfile o1 grades/ordered.1.ordinary.json --slurpfile o2 grades/ordered.2.ordinary.json '
        {schema:"uor-r4.ordinary-termination-held-summary/1", artifacts:$artifacts,
         parity:{device_client_one_step_per_fit:94, total_device_client_one_step:376,
                 off_boundary:94, zero_boundary:94, off_ordinary:90, zero_ordinary:90},
         teacher_forced:{base:$base_b[0],fixed:$fixed_b[0],"geometry.1":$g1_b[0],"geometry.2":$g2_b[0],
                         "ordered.1":$o1_b[0],"ordered.2":$o2_b[0]},
         self_fed:{base:$base[0],fixed_punctuation_reconstruction:$fixed[0],off:$off[0],zero:$zero[0],
                   "geometry.1":$g1[0],"geometry.2":$g2[0],"ordered.1":$o1[0],"ordered.2":$o2[0]}}
    ' > SUMMARY.json
sha256sum SUMMARY.json ARTIFACTS.jsonl TOOLS.sha256 BLIND.sha256 RESPONSES.sha256 zero.words zero.d11 zero.pack.json grades/*.json grades/*.jsonl > RESULTS.sha256
sha256sum --strict --quiet -c RESULTS.sha256
sha256sum SUMMARY.json RESULTS.sha256
