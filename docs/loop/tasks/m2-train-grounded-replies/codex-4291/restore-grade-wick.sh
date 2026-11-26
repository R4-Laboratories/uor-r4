#!/usr/bin/env bash
# Regrade the archived physical reads and replay their integer training votes.
set -euo pipefail
umask 077
export LC_ALL=C
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport
storage="$s/storage"
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
sha(){ sha256sum "$1" | cut -d' ' -f1; }

if [[ ${1:?extracted archive or download} == download ]]; then
    receipt=$2
    restore=$3
    [[ $restore == "$storage"/download-v* && $# == 3 ]]
    repo=$(jq -er '.repo|select(.=="caseyallard/uor-r4-store")' "$receipt")
    prefix=$(jq -er '.path|select(test("^results/codex/mixed-transport-20261126-[0-9a-f]{12}$"))' "$receipt")
    revision=$(jq -er '.revision|select(test("^[0-9a-f]{40}$"))' "$receipt")
    archive=$(jq -er '.archive_sha256|select(test("^[0-9a-f]{64}$"))' "$receipt")
    mkdir "$restore" "$restore/extracted"
    "$HOME/.local/bin/hf" download "$repo" --repo-type dataset --revision "$revision" \
        --include "$prefix/*" --local-dir "$restore/download" --quiet > "$restore/download.log"
    packet="$restore/download/$prefix"
    [[ $(sha "$packet/measurement.tar.zst") == "$archive" ]]
    (cd "$packet" && sha256sum --strict --quiet -c SHA256SUMS)
    tar --zstd -xf "$packet/measurement.tar.zst" -C "$restore/extracted"
    (cd "$restore/extracted" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
    exec bash "$restore/extracted/archive-methods/restore-grade-wick.sh" "$restore/extracted" "$restore/replayed"
fi

root=$(realpath "$1")
out=${2:?new replay directory}
[[ $root == "$storage"/*/extracted && $out == "$(dirname "$root")"/replayed ]]
meta="$root/archive.json"
measurement=$(jq -er '.measurement|select(test("^measurement-v[1-9][0-9]*$"))' "$meta")
release=$(jq -er '.release|select(test("^release-v[1-9][0-9]*$"))' "$meta")
methods=$(jq -er '.methods|select(test("^methods-v[1-9][0-9]*$"))' "$meta")
panel=$(jq -er '.panel|select(test("^panel/panel-v[1-9][0-9]*$"))' "$meta")
premise=$(jq -er '.premise|select(type=="boolean")|tostring' "$meta")
m="$root/$measurement"; panel="$root/$panel"; prov="$root/provenance"
orig="$root/$methods"; replay="$out/$measurement"
released="$root/$release/inputs.sha256"
grep -Fxq "$(sha "$released")  $s/$release/inputs.sha256" "$m/inputs.sha256"
manifest_path=$(awk -v prefix="$s/methods-" 'index($2,prefix)==1 && $2~/\/SHA256SUMS$/ {print $2}' "$released")
[[ $manifest_path == "$s/$methods/SHA256SUMS" ]]
for file in SHA256SUMS grade-wick.sh common-wick.sh; do
    grep -Fxq "$(sha "$orig/$file")  $s/$methods/$file" "$released"
done
(cd "$orig" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$m" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$out" "$replay" "$out/geometry"
here="$orig"
# The sealed common helper definitions do not require the live command's path bindings.
# shellcheck disable=SC1090
source <(sed -n '/^seal(){ /,$p' "$orig/common-wick.sh")

sha256sum --strict --quiet -c "$prov/geometry-dependencies.sha256"
mapfile -t geometry_dependencies < <(cut -d' ' -f3- "$prov/geometry-dependencies.sha256")
[[ ${#geometry_dependencies[@]} == 3 ]]
codec=${geometry_dependencies[0]}; model=${geometry_dependencies[1]}; exporter=${geometry_dependencies[2]}
build_geometry() {
    local path=$1 label=$2 manifest=$3 original=$4
    local expected
    expected=$(awk -v name="./$original/blind/text.bin" '$2==name {print $1}' "$manifest")
    [[ $expected =~ ^[0-9a-f]{64}$ ]]
    "$exporter" pack "$path/user-only.json" "$prov/history.json" "$model" "$codec" "$out/geometry/$label"
    cmp "$path/blind/coordinates.jsonl" "$out/geometry/$label/coordinates.jsonl"
    [[ $(sha "$out/geometry/$label/text.bin") == "$expected" ]]
}
train_read() {
    local d=$1 target=$2 split=$3 state=$4 physical=$5 label=$6
    local geometry="$out/geometry/$split/text.bin"
    "$prov/bin/joint-audit" "$geometry" "$prov/primary.words" "$state" \
      "$d/$physical.raw.jsonl" "$d/gap.raw.jsonl" > "$target/$label.replay.json"
    cmp "$d/$physical.replay.json" "$target/$label.replay.json"
}
grade_raw() {
    local raw=$1 computed=$2 split=$3 p=$4 kind=$5 label=$6 original=$7
    arm_rows "$split" "$p" "$kind" "$raw/$original.raw.jsonl" "$computed/$label"
    cmp "$raw/$label.rows.jsonl" "$computed/$label.rows.jsonl"
    cmp "$raw/$label.pairs.json" "$computed/$label.pairs.json"
}
for split in mixed far; do
    p="$panel/$split"; d="$m/train/$split"; t="$replay/train/$split"
    build_geometry "$p" "$split" "$panel/SHA256SUMS" "$split"
    geom="$out/geometry/$split/text.bin"
    mkdir -p "$t"
    "$prov/bin/signed-edge-audit" diagnose "$geom" "$p/links.bin" "$prov/parent.state" \
        "$prov/parent.state" "$prov/parent.state" "$t/initial.oracle.jsonl" "$t/initial.reconstruction.json"
    cmp "$d/initial.oracle.jsonl" "$t/initial.oracle.jsonl"
    cmp "$d/initial.reconstruction.json" "$t/initial.reconstruction.json"
    jq -es --slurpfile physical "$d/initial.jsonl" -f "$prov/compare.jq" \
        "$t/initial.oracle.jsonl" > "$t/initial.checks.json"
    cmp "$d/initial.checks.json" "$t/initial.checks.json"
    train_read "$d" "$t" "$split" "$prov/parent.state" step-0 step-0
    grade_raw "$d" "$t" "$split" "$p" learned step-0 step-0
done
jq -ne --slurpfile m "$m/train/mixed/initial.jsonl" --slurpfile f "$m/train/far/initial.jsonl" \
    --slurpfile p "$replay/train/mixed/step-0.pairs.json" '
 {mixed_eligible:($m|map(select(.eligible_for_fit))|length),far_eligible:($f|map(select(.eligible_for_fit))|length),
 common_far_signed_losses:($m|map(select(.row<128 and .eligible_for_fit and .before.margin<0))|length),
 original_full_near:($p[0]|map(select(.distance=="near" and .full)|.pair)),
 common_far_losing_pairs:($p[0]|map(select(.block=="common" and .distance=="far" and (.exact|not))|.pair))} |
 .pass=(.mixed_eligible==512 and .far_eligible==512 and .common_far_signed_losses>0 and (.original_full_near|length)>0 and (.common_far_losing_pairs|length)>0)' \
    > "$replay/premise.json"
cmp "$m/premise.json" "$replay/premise.json"
[[ $(jq -r .pass "$replay/premise.json") == "$premise" ]]

if [[ $premise == true ]]; then
    for split in mixed far; do
        p="$panel/$split"; d="$m/train/$split"; t="$replay/train/$split"
        geom="$out/geometry/$split/text.bin"
        "$prov/bin/signed-edge-audit" contrastive "$geom" "$p/links.bin" "$prov/parent.state" \
            "$d/old.state" "$d/old.final" "$t/old.oracle.jsonl" "$t/old.reconstruction.json"
        cmp "$d/old.oracle.jsonl" "$t/old.oracle.jsonl"
        cmp "$d/old.reconstruction.json" "$t/old.reconstruction.json"
        jq -es --slurpfile physical "$d/old.trace" -f "$prov/compare.jq" "$t/old.oracle.jsonl" > "$t/old.checks.json"
        cmp "$d/old.checks.json" "$t/old.checks.json"
        "$prov/bin/coordinate-audit" "$geom" "$p/distance.txt" "$prov/parent.state" \
            "$t/old.oracle.jsonl" "$d/first" "$t/coordinate-checks.json"
        cmp "$d/coordinate-checks.json" "$t/coordinate-checks.json"
        for step in {1..8}; do
            train_read "$d" "$t" "$split" "$d/first/step-$step.state" "step-$step" "step-$step"
            grade_raw "$d" "$t" "$split" "$p" learned "step-$step" "step-$step"
        done
        jq -ne --slurpfile checkpoints <(cat "$t"/step-{0..8}.pairs.json) \
            --slurpfile quality "$d/first/quality.jsonl" -f "$orig/grade/elect.jq" > "$t/election.json"
        cmp "$d/election.json" "$t/election.json"
    done
    for item in far_only:far:unconstrained mixed_free:mixed:unconstrained retained:mixed:retained; do
        IFS=: read -r arm split pick <<< "$item"
        step=$(jq -er --arg pick "$pick" '.selections[$pick]' "$replay/train/$split/election.json")
        cmp "$m/train/$split/first/step-$step.state" "$m/states/$arm.state"
    done
    historical="$root/historical/panel/panel-v1"
    for split in held bridge; do
        p="$panel/held"; source_manifest="$panel/SHA256SUMS"
        if [[ $split == bridge ]]; then p="$historical/held"; source_manifest="$historical/SHA256SUMS"; fi
        build_geometry "$p" "$split" "$source_manifest" held
        d="$m/$split"; t="$replay/$split"; mkdir "$t"
        for arm in baseline far_only mixed_free retained last_far; do
            case $arm in
                baseline) state="$prov/parent.state" ;;
                last_far) state="$prov/previous-far.state" ;;
                *) state="$m/states/$arm.state" ;;
            esac
            train_read "$d" "$t" "$split" "$state" "$arm" "$arm"
            grade_raw "$d" "$t" "$split" "$p" learned "$arm" "$arm"
        done
        for item in native:native:native unfitted:unfitted:unfitted gap:governor_gap:gap \
            correct_link:correct_link:correct exchanged_link:exchanged_link:exchanged; do
            IFS=: read -r label kind raw <<< "$item"
            grade_raw "$d" "$t" "$split" "$p" "$kind" "$label" "$raw"
        done
    done
    cmp "$m/bridge/baseline.raw.jsonl" "$root/historical/measurement-v2/held/baseline.jsonl"
    cmp "$m/bridge/last_far.raw.jsonl" "$root/historical/measurement-v2/held/learned.jsonl"
    grade=$(jq -er '.grade|select(test("^grade-v[1-9][0-9]*$"))' "$meta")
    for item in "$measurement" "$methods"; do
        grep -Fxq "$(sha "$root/$item/SHA256SUMS")  $s/$item/SHA256SUMS" "$root/$grade/inputs.sha256"
    done
    (cd "$root/$grade" && sha256sum --strict --quiet -c SHA256SUMS)
    # Replay the authenticated scoring statements with restored inputs. Live path validation
    # above uses archived hashes because the original absolute paths need not still exist.
    awk '
        $0 == "mkdir \"$g\"" { starts++; next }
        $0 == "sha256sum \"$m/SHA256SUMS\" \"$here/SHA256SUMS\" > \"$g/inputs.sha256\"; seal \"$g\"" { ends++; next }
        starts == 1 && ends == 0 { print }
        END { if (starts != 1 || ends != 1) exit 1 }
    ' "$orig/grade-wick.sh" > "$out/grade-commands.sh"
    bash -n "$out/grade-commands.sh"
    seal "$replay"
    (
        m="$replay"; g="$out/$grade"; p="$panel"
        # The restored input paths are consumed by the authenticated scoring statements.
        # shellcheck disable=SC2034
        prev="$root/historical"
        mkdir "$g"
        # shellcheck disable=SC1091
        source "$out/grade-commands.sh"
        sha256sum "$m/SHA256SUMS" "$here/SHA256SUMS" > "$g/inputs.sha256"
        seal "$g"
    )
    for file in "$root/$grade/"*; do
        case $(basename "$file") in
            inputs.sha256|SHA256SUMS) ;;
            *) cmp "$file" "$out/$grade/$(basename "$file")" ;;
        esac
    done
    capture=$(jq -r '.client.capture // "-"' "$meta")
    if [[ $capture != - ]]; then
        token=$(jq -er '.client.token|select(test("^token-v[1-9][0-9]*$"))' "$meta")
        mkdir "$out/client"
        jq -ce -f "$root/client/audit-tokens.jq" "$root/client/$capture/grade/"*.jsonl > "$out/client/tokens.jsonl"
        cmp "$root/client/$token/tokens.jsonl" "$out/client/tokens.jsonl"
        jq -es 'if length!=1536 or (group_by(.arm)|length)!=6 then error("missing client read") else
          group_by(.arm)|map({arm:.[0].arm,events:length,first_eos:map(select(.first.exact_through_eos))|length,
          returned_eos:map(select(.returned.exact_through_eos))|length,returned_exact_pairs:group_by(.pair_id)|map(select(length==2 and all(.[];.returned.exact_through_eos)))|length,
          correct_prefix_at_cap:map(select(.returned.correct_prefix_at_cap))|length,
          first_divergences:(map(.first.first_differing_id)|group_by(.)|map({first:.[0],n:length})),
          returned_divergences:(map(.returned.first_differing_id)|group_by(.)|map({first:.[0],n:length}))}) end' \
          "$out/client/tokens.jsonl" > "$out/client/summary.json"
        cmp "$root/client/$token/summary.json" "$out/client/summary.json"
    fi
fi

jq -n --arg measurement "$measurement" --arg methods "$methods" --arg methods_sha256 "$(sha "$orig/SHA256SUMS")" \
  --argjson premise "$premise" \
  --argjson checkpoints "$([[ $premise == true ]] && echo 18 || echo 2)" \
  --arg grade "${grade:--}" --arg capture "${capture:--}" '{
      measurement:$measurement,methods:$methods,methods_sha256:$methods_sha256,
      premise:$premise,checkpoint_rank_replays:$checkpoints,
      elections_reproduced:$premise,coordinate_votes_reproduced:$premise,
      grade:(if $grade=="-" then null else $grade end),
      client_token_trace:(if $capture=="-" then null else $capture end),
      substantive_disagreements:0}' > "$out/receipt.json"
seal "$out"
cat "$out/receipt.json"
