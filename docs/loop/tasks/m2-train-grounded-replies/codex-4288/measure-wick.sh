#!/usr/bin/env bash
# The outer process verifies the study release before taking the existing Wick GPU lock.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-text-governor-links"
historical="$runs/20261125-codex-governing-source-pairs"
previous="$runs/20261126-codex-governor-noun-links"
fit="$historical/fit-v1/primary.words"
model="$historical/models/fold-8/ordered-all.d11"
lock="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "$0")" && pwd)
test "$(hostname -s)" = wick
test "$(findmnt -no FSTYPE --target "$root")" = zfs
[[ $here == "$root"/methods-v* ]]
mode=${1:?run, inner, or grade}
version=${2:?fresh measurement version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
release=$(realpath "${3:?release manifest}")
digest=${4:?release digest}
[[ $release == "$root"/release-v*/inputs.sha256 && $digest =~ ^[a-f0-9]{64}$ ]]
sha(){ sha256sum "$1" | cut -d' ' -f1; }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
seal(){ (cd "$1" && find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 sha256sum > .seal.pending && mv .seal.pending SHA256SUMS); }
test "$(sha "$release")" = "$digest"
sha256sum --strict --quiet -c "$release"
release_dir=$(dirname "$release")
learner=$(cat "$release_dir/learner-path.txt")
[[ $learner == "$root/learner/"*/textlinks-sm89 ]]
for needed in "$learner" "$lock" "$here/measure-wick.sh" "$release_dir/learner-path.txt"      "$release_dir/FIT.md" "$release_dir/DECISION.md" "$root/export/"{train,held}-v2/SHA256SUMS      "$root/panel/panel-v1/SHA256SUMS" "$fit" "$model"; do
    grep -Fxq "$(sha256sum "$needed")" "$release"
done
out="$root/measurement-$version"
train="$root/export/train-v2"
held="$root/export/held-v2"
old="$previous/device/build-v2"
if [[ $mode == run ]]; then
    bash "$lock" "$out-lease" bash "$here/measure-wick.sh" inner "$version" "$release" "$digest"
elif [[ $mode == inner ]]; then
    test -e /proc/$$/fd/9
    verify "$train"; verify "$held"; verify "$old"; verify "$root/panel/panel-v1"
    mkdir "$out" "$out/train" "$out/held" "$out/independent"
    # Only this first phase receives train labels. The primary fit forwards values to the same
    # predictor used by a separate restored process; compare complete JSONL records, including ties.
    for pair in 'train learned' 'train-swapped flipped_sign' 'train-endpoints corrupt_noun'; do
        read -r update arm <<<"$pair"
        "$learner" "$update" "$fit" "$train/blind/text.bin" "$train/links.bin" "$out/train/$arm.state" "$out/train/$arm.fit.jsonl"
        "$learner" learned "$fit" "$train/blind/text.bin" "$out/train/$arm.state" "$out/train/$arm.jsonl"
        cmp "$out/train/$arm.fit.jsonl" "$out/train/$arm.jsonl"
    done
    # Rerun primary training from zero with a new state file.
    "$learner" train "$fit" "$train/blind/text.bin" "$train/links.bin" "$out/train/repeat.state" "$out/train/repeat.jsonl"
    cmp "$out/train/learned.state" "$out/train/repeat.state"
    cmp "$out/train/learned.jsonl" "$out/train/repeat.jsonl"
    for split in train held; do
        if [[ $split == held ]]; then data=$held; else data=$train; fi
        dir="$out/$split"
        for arm in native unfitted; do "$learner" "$arm" "$fit" "$data/blind/text.bin" "$dir/$arm.jsonl"; done
        "$learner" governor_gap "$fit" "$data/blind/text.bin" "$out/train/learned.state" "$dir/governor_gap.jsonl"
        if [[ $split == held ]]; then
            for arm in learned corrupt_noun flipped_sign; do
                "$learner" learned "$fit" "$data/blind/text.bin" "$out/train/$arm.state" "$dir/$arm.jsonl"
            done
            "$learner" learned "$fit" "$data/blind/text.bin" "$out/train/repeat.state" "$dir/repeat.jsonl"
            cmp "$dir/learned.jsonl" "$dir/repeat.jsonl"
        fi
        # Only these two held commands receive an intervention.
        for arm in correct exchanged; do "$learner" "$arm" "$fit" "$data/blind/text.bin" "$data/aid.bin" "$dir/$arm.jsonl"; done
    done
    independent="$root/panel/panel-v1/structural-links"
    for arm in native governor_only nouns_only correct_link exchanged_link; do
        "$old/governor-noun-links-sm89" "$fit" "$independent/geometry.bin" "$independent/$arm.bin" "$out/independent/$arm.rank"
    done
    "$old/target/release/governor-noun-links" grade "$independent" "$model" "$fit" "$out/independent"
    sha256sum "$release" > "$out/inputs.sha256"
    seal "$out"
elif [[ $mode == grade ]]; then
    verify "$out"
    grade="$root/grade-${5:?new grade version}"
    mkdir "$grade"
    for split in train held; do
        if [[ $split == train ]]; then data=$train; ref="$previous/panel/panel-v1/fresh/reference.json"
        else data=$held; ref="$root/panel/panel-v1/fresh/reference.json"; fi
        dir="$out/$split"
        jq -nce --slurpfile reference "$ref" --slurpfile coords "$data/blind/coordinates.jsonl" \
          --slurpfile learned "$dir/learned.jsonl" --slurpfile native "$dir/native.jsonl" --slurpfile unfitted "$dir/unfitted.jsonl" \
          --slurpfile correct "$dir/correct.jsonl" --slurpfile exchange "$dir/exchanged.jsonl" --slurpfile gap "$dir/governor_gap.jsonl" \
          --slurpfile corrupt "$dir/corrupt_noun.jsonl" --slurpfile flipped "$dir/flipped_sign.jsonl" \
          -f "$here/grade/rows.jq" > "$grade/$split.jsonl"
        jq -es --arg ref_sha "$(sha "$ref")" --arg end_sha "$(sha "$grade/$split.jsonl")" -f "$here/grade/gate.jq" \
            "$grade/$split.jsonl" > "$grade/$split.summary.json"
    done
    jq -nce --slurpfile ranked "$grade/held.jsonl" --slurpfile other "$out/independent/endpoints.jsonl" '
        ($other|INDEX(.case_id)) as $by_id |
        [ $ranked[] as $r | $by_id[$r.case_id] as $o | ["native","correct_link","exchanged_link"][] as $arm |
          select($o.prompt_sha256!=$r.prompt_sha256 or
             ($o.arms[$arm]|[.selection,.tokens,.bank_cost,.ties,.eligible])!=
             ($r.arms[$arm]|[.selection,.tokens,.bank_cost,.ties,.eligible])) | {id:$r.case_id,arm:$arm}] as $bad |
        if ($ranked|length)==256 and ($other|length)==256 and $bad==[] then
           {rows:256,original_address_comparisons:768,disagreements:$bad} else error("independent native/assisted control differs: "+($bad[:2]|tostring)) end' \
        > "$grade/independent.json"
    # Ordinary-client adapter requires just the five arms which go through the writer.
    jq -c '.arms |= with_entries(select(.key|IN("native","learned","governor_gap","correct_link","exchanged_link")))' \
        "$grade/held.jsonl" > "$grade/endpoints.jsonl"
    jq -e --arg sha "$(sha "$grade/endpoints.jsonl")" '.full_source_endpoints_sha256=.endpoints_sha256 | .endpoints_sha256=$sha' \
       "$grade/held.summary.json" > "$grade/summary.json"
    jq -es --slurpfile reference "$root/panel/panel-v1/fresh/reference.json" -f "$here/grade/trace.jq" "$grade/held.jsonl" > "$grade/trace.json"
    sha256sum "$out/SHA256SUMS" "$release" "$here/grade/"*.jq > "$grade/inputs.sha256"
    seal "$grade"
else exit 64; fi
