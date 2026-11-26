#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 2 ]]
root=$(realpath -e "$1")
out=$2
mkdir -- "$out"
out=$(realpath -e "$out")
here="$root/methods/storage"
panel="$root/panel"
grade="$root/methods/grade"
physical="$root/physical"
primary="$physical/train"
projections="$physical/backoff/projected"
held="$physical/held"
expected="$root/expected"
elected="$expected/election"
train="$root/inputs/train-input"
models="$train/models"
seed="$models/slot3.state"
bank="$models/primary.words"
tokenizer="$root/inputs/tokenizer.json"
independent="$root/independent/build/slot3-audit"
digest() { sha256sum -- "$1" | cut -d' ' -f1; }
verify() (cd "$1" && sha256sum --strict --quiet -c "${2:-SHA256SUMS}")
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
state_dir() {
    case "$1" in
    matrix) printf '%s\n' "$primary/matrix" ;;
    projected/matrix) printf '%s\n' "$projections/matrix" ;;
    *) echo "Unrecognized physical state directory: $1" >&2; return 1 ;;
    esac
}
score_dir() {
    case "$1" in
    matrix) printf '%s\n' "$primary/score" ;;
    projected/matrix) printf '%s\n' "$projections/score" ;;
    *) echo "Unrecognized physical score directory: $1" >&2; return 1 ;;
    esac
}

test -z "$(find "$root" ! -type f ! -type d -print -quit)"
verify "$root" MEMBERS.sha256
(
    cd "$root"
    find . -type f ! -path './MEMBERS.sha256' -print | sort > "$out/actual-members.txt"
    cut -d' ' -f3- MEMBERS.sha256 > "$out/declared-members.txt"
)
cmp "$out/actual-members.txt" "$out/declared-members.txt"
verify "$root/methods" grade-ops.sha256
verify "$root/methods/frozen"
verify "$root/methods/margin-fit"
verify "$root/methods/eligibility"
verify "$root/methods/scorer"
verify "$panel" training.sha256
verify "$panel" guards.sha256
verify "$panel/train"
verify "$panel/fresh"
verify "$train"
verify "$physical/fit-finite-e"
verify "$physical/fit-finite-m"
verify "$root/custody/device-records"
verify "$root/independent/build"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$root/inputs/ordered-all.d11"
pin e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977 "$seed"
for phase in regression near previous; do
    cmp "$train/$phase/reference.native.jsonl" "$primary/prefit/$phase/base.native.jsonl"
done
cmp "$seed" "$primary/fit-out/start.state"
cmp "$seed" "$primary/matrix/b.state"
cmp "$primary/e-out/wrong.state" "$primary/matrix/wrong.state"
cp -- "$primary/matrix/factors.tsv" "$out/factors.tsv"
if [[ -d $projections ]]; then
    tail -n +2 "$projections/matrix/factors.tsv" >> "$out/factors.tsv"
    verify "$root/inputs/backoff-input"
fi
cmp "$out/factors.tsv" "$elected/factors.tsv"

for kind in matrix projected/matrix; do
    if [[ $kind == projected/matrix && ! -d $projections ]]; then continue; fi
    dir=$(state_dir "$kind")
    while IFS=$'\t' read -r id m e mid eid src hash; do
        [[ $id =~ ^[a-z0-9-]+$ && $hash =~ ^[a-f0-9]{64}$ ]]
        pin "$hash" "$dir/$id.state"
        if [[ $kind == matrix && $e == 0 ]]; then
            cmp "$dir/$id.state" "$primary/fit-out/$src.state"
            cmp "$primary/fit-out/$src.native.jsonl" "$primary/score/train/$id.native.jsonl"
        elif [[ $kind == matrix && $m == 0 ]]; then
            cmp "$dir/$id.state" "$primary/e-out/$eid.state"
        elif [[ $kind == projected/matrix && $e == 0 ]]; then
            cmp "$dir/$id.state" "$projections/deltas/$src.state"
        fi
    done < <(tail -n +2 "$dir/factors.tsv")
done

bash "$grade/grid.sh" "$panel" "$primary/score" "$primary/train-probes" \
    "$primary/matrix/factors.tsv" "$out/train"
if [[ -d $projections && -s $projections/matrix/models.tsv ]]; then
    bash "$grade/grid.sh" "$panel" "$projections/score" "$primary/train-probes" \
        "$projections/matrix/factors.tsv" "$out/projected"
    for d in "$out/projected/"*; do [[ ! -e $out/train/${d##*/} ]]; mv -- "$d" "$out/train/"; done
fi
bash "$grade/election.sh" "$out/train" "$out/factors.tsv" "$out/election"
for name in election.json records.jsonl; do cmp "$out/election/$name" "$elected/$name"; done
printf 'Replayed all grid grades, election and candidate manifests\n'

# Verify archived arm identities and exact bridge state readback independently of reported grades.
mkdir "$out/bundle" "$out/guards" "$out/fresh" "$out/independent"
choice="$out/election/election.json"
jq -r '.choice | ["baseline","b"],["margin",.m],["eligibility",.e],["combined",.combined],["wrong","wrong"] | @tsv' \
    "$choice" > "$out/selected.tsv"
cut -f1,2 "$elected/arms.tsv" | cmp - "$out/selected.tsv"
cut -f1 "$elected/arms.tsv" | awk '{print $1 "\t" $1 ".state"}' | cmp - "$held/bundle/models.tsv"

while IFS=$'\t' read -r alias id kind hash; do
    dir=$(state_dir "$kind")
    score=$(score_dir "$kind")
    [[ $id =~ ^[a-z0-9-]+$ && $hash =~ ^[a-f0-9]{64}$ && $alias =~ ^[a-z]+$ ]]
    if [[ $alias == wrong ]]; then
        [[ $kind == matrix ]]
    else
        record_hash=$(jq -ers --arg id "$id" '[.[]|select(.id==$id)|.sha] |
            if length==1 then .[0] else error("nonunique elected state ID") end' "$out/election/records.jsonl")
        [[ $record_hash == "$hash" ]]
    fi
    pin "$hash" "$dir/$id.state"
    pin "$hash" "$held/bundle/$alias.state"
    [[ $("$models/bundle" unpack "$held/bundle/$alias.bridge" "$out/bundle/$alias.state") == ordered ]]
    cmp "$out/bundle/$alias.state" "$held/bundle/$alias.state"
    cmp "$out/bundle/$alias.state" "$dir/$id.state"
    bash "$grade/state.sh" "$panel" "$held/score" "$held/probe" "$alias" "$alias" baseline "$out/fresh/$alias" fresh
    for name in fresh.json fresh.rows.jsonl; do cmp "$out/fresh/$alias/$name" "$expected/held/$alias/$name"; done
    "$independent" slot3-replay "$panel/fresh" "$tokenizer" "$out/bundle/$alias.state" "$bank" \
        "$held/score/fresh/$alias.native.jsonl" "$held/score/fresh/$alias.costs.jsonl" "$seed" "$out/independent/$alias.json"
    "$independent" physical-words "$panel/fresh/ordered/text.bin" "$out/bundle/$alias.state" \
        "$held/probe/$alias.jsonl" "$out/independent/$alias.words.json"
    cmp "$out/independent/$alias.json" "$root/independent/physical-held-v1/$alias.json"
    cmp "$out/independent/$alias.words.json" "$root/independent/physical-held-v1/$alias-words.json"
    jq -en --slurpfile independent "$out/independent/$alias.json" --slurpfile count "$out/fresh/$alias/fresh.json" '
        ["fixed_pairs","fixed_ids","fixed_rows","blocked","worst","debt","retain_admit","other_losses",
         "source","source_ids","full","full_ids"]
        | all(.[]; . as $k | $independent[0].metrics[$k] == $count[0][$k])
    ' > "$out/independent/$alias.grade-agreement.json"
    if [[ $alias == wrong ]]; then continue; fi

    record=$(jq -ecs --arg id "$id" '.[]|select(.id==$id)' "$out/election/records.jsonl")
    jq -en --argjson grade "$record" --slurpfile reg "$panel/regression/requests.jsonl" \
        --slurpfile gold "$panel/near/gold.jsonl" --slurpfile coordinates "$panel/near/ordered/coordinates.jsonl" \
        --slurpfile r_native "$score/regression/$id.native.jsonl" --slurpfile n_native "$score/near/$id.native.jsonl" \
        -f "$here/guards.jq" > "$out/guards/$alias.json"
done < "$elected/arms.tsv"
jq -en --slurpfile election "$choice" --slurpfile baseline "$out/guards/baseline.json" \
    --slurpfile margin "$out/guards/margin.json" --slurpfile eligibility "$out/guards/eligibility.json" \
    --slurpfile combined "$out/guards/combined.json" '
    if $baseline[0].losses==0 and ([$margin[0].losses,$eligibility[0].losses,$combined[0].losses]|add)
        == $election[0].choice.lost_guard_reads
    then {baseline:$baseline[0].losses, margin:$margin[0].losses, eligibility:$eligibility[0].losses, combined:$combined[0].losses}
    else error("guard election disagrees with exact original address captures") end
' > "$out/guards/counts.json"

jq -en --slurpfile gold "$panel/fresh/gold.jsonl" --slurpfile pos "$held/controls/gold.jsonl" \
    --slurpfile neg "$held/controls/swapped.jsonl" -f "$grade/controls.jq" > "$out/fresh/controls.json"
cmp "$out/fresh/controls.json" "$expected/controls/controls.json"
"$independent" physical-controls "$panel/fresh" "$tokenizer" "$held/controls/gold.jsonl" \
    "$held/controls/swapped.jsonl" "$out/independent/controls.json"
"$independent" factorial "$out/bundle/baseline.state" "$out/bundle/margin.state" "$out/bundle/eligibility.state" \
    "$out/bundle/combined.state" "$out/independent/factorial.json"
for name in controls factorial; do
    cmp "$out/independent/$name.json" "$root/independent/physical-held-v1/$name.json"
done
"$independent" margin-capture "$panel/train/ordered/text.bin" "$panel/train/requests.tsv" "$bank" "$primary/fit-out" \
    "$out/independent/margin-fit.json"
"$independent" eligibility-fit "$panel/train/ordered/text.bin" "$panel/train/retain.tsv" "$seed" "$primary/e-out" \
    "$out/independent/eligibility-fit.json"
bash "$here/lineage.sh" "$root" "$out"

reduce="$root/methods/held-reduction"
fresh="$panel/fresh"
mkdir "$out/reduction"
jq -enc --slurpfile geometry "$fresh/rows.jsonl" --slurpfile gold "$fresh/gold.jsonl" \
    --slurpfile correct "$held/controls/gold.jsonl" --slurpfile swap "$held/controls/swapped.jsonl" \
    -f "$reduce/held-controls.jq" > "$out/reduction/controls.json"
for alias in baseline margin eligibility combined wrong; do
    jq -enc --slurpfile geometry "$fresh/rows.jsonl" --slurpfile meta "$fresh/requests.jsonl" \
        --slurpfile gold "$fresh/gold.jsonl" --slurpfile registered "$fresh/eligibility.jsonl" \
        --slurpfile reads "$held/score/fresh/$alias.native.jsonl" --slurpfile costs "$held/score/fresh/$alias.costs.jsonl" \
        --slurpfile probe "$held/probe/$alias.jsonl" --slurpfile baseline_probe "$held/probe/baseline.jsonl" \
        --slurpfile review "$out/independent/$alias.json" -f "$reduce/held-row.jq" > "$out/reduction/$alias.rows.jsonl"
    cmp "$out/reduction/$alias.rows.jsonl" "$expected/held-reduction/$alias.rows.jsonl"
done
jq -enc --slurpfile controls "$out/reduction/controls.json" --slurpfile baseline "$out/reduction/baseline.rows.jsonl" \
    --slurpfile margin "$out/reduction/margin.rows.jsonl" --slurpfile eligibility "$out/reduction/eligibility.rows.jsonl" \
    --slurpfile combined "$out/reduction/combined.rows.jsonl" --slurpfile wrong "$out/reduction/wrong.rows.jsonl" \
    -f "$reduce/held-summary.jq" > "$out/reduction/summary.json"
for name in summary controls; do cmp "$out/reduction/$name.json" "$expected/held-reduction/$name.json"; done

(cd "$out" && find . -type f -print0 | sort -z | xargs -0 sha256sum > ../verification.pending)
mv -- "$out/../verification.pending" "$out/SHA256SUMS"
printf 'Verified five archived arms, four-arm factor bytes, both controls and exact guard identities\n'
