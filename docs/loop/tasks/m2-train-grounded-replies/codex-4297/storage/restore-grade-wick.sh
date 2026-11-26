#!/usr/bin/env bash
set -euo pipefail
lib=$(dirname "$(realpath "$0")")/archive-lib.sh
source "$lib"
[[ $# == 3 ]]
root=$(realpath -e "${1:?extracted archive}")
meta=$(realpath -e "${2:?archived layout}")
out=${3:?new output directory}
[[ $meta == "$root"/storage/archive-v*/layout.json ]]
jq -e '.schema=="uor-r4.codex-4297.archive-layout/1"' "$meta" >/dev/null
from_layout() {
    local name
    name=$(jq -er --arg key "$1" --arg pattern "$2" '.[$key]|select(test($pattern))' "$meta")
    safe_member "$name"
    printf '%s/%s\n' "$root" "$name"
}
methods=$(from_layout methods '^methods-v[1-9][0-9]*$')
train_methods=$(from_layout train_methods '^methods-v[1-9][0-9]*$')
panel=$(from_layout panel '^panel/panel-v[1-9][0-9]*$')
m=$(from_layout measurement '^measurement-v[1-9][0-9]*$')
raw=$(from_layout held '^held-v[1-9][0-9]*$')
grade=$(from_layout grade '^grade-v[1-9][0-9]*$')
release=$(from_layout release '^release-v[1-9][0-9]*$')
permit=$(from_layout train_permit '^train-verified-v[1-9][0-9]*$')
archive_dir=${meta%/*}
ancestry="$archive_dir/ancestry"
archived_banks="$ancestry/20261125-codex-governing-source-pairs/fit-v1/primary.words"
archived_replay="$ancestry/20261126-codex-text-governor-links/parity/build-v2/release/textlinks-trace-audit"
packer=$(from_layout packer '^finite-core-v[1-9][0-9]*/bundle$')
audit=$(from_layout train_audit '^audit/train-v[1-9][0-9]*$')
audit_build=$(from_layout audit_build '^audit/build-v[1-9][0-9]*$')
held_audit=$(from_layout held_audit '^audit/held-v[1-9][0-9]*$')
held_build=$(from_layout held_audit_build '^audit/build-v[1-9][0-9]*$')
client=$(jq -er '.client.version' "$meta")
for dir in "$m" "$raw" "$grade" "$panel" "$methods" "$train_methods" "$audit" "$audit_build" "$held_audit" "$held_build"; do
    verify "$dir"
done
pin 786f22c1cebafbe7d7be8038b42ed229cd8b994c980c745fe058b65182bcc581 "$archived_replay"
verify "$(dirname "$packer")"
jq -e '.verified==true and .batches==6' "$permit/verified.json" >/dev/null
source "$methods/common-wick.sh"
mkdir "$out" "$out/joined" "$out/bundles" "$out/train"
cmp_file() {
    cmp "$1" "$2" || { printf 'Cold replay disagrees with %s\n' "$2" >&2; return 1; }
    printf '%s\n' "${2#"$root"/}" >> "$out/compared.txt"
}
arms=(native unfitted gap ordered_gap baseline ordered_base far_only strongest all ordered-all correct_link exchanged_link)
trained=(strongest all ordered-all)
frozen="$ancestry/20261126-codex-protected-transport/measurement-v1"
"$audit_build/bridge-audit" train "$frozen" "$archived_banks" "$panel" "$m" "$out/train/readback.json"
cmp_file "$out/train/readback.json" "$audit/readback.json"

for arm in "${trained[@]}"; do
    "$packer" unpack "$m/states/$arm.bridge" "$out/bundles/$arm.state" > "$out/bundles/$arm.mode"
    cmp_file "$out/bundles/$arm.state" "$m/states/$arm.state"
    cmp_file "$out/bundles/$arm.mode" "$raw/unpacked/$arm.mode"
    cmp_file "$out/bundles/$arm.state" "$raw/unpacked/$arm.state"
    for round in 1 2; do
        batch="$m/train/$arm/round-$round"
        jq -e --arg arm "$arm" '.mode==$arm and .rows==256 and .common_far_rows==128 and .proposals==68 and .state_bytes==13336' "$batch/device/complete.json" >/dev/null
        verify "$batch/device"; verify "$batch/grade"
        elected="$out/train/$arm-$round.election.json"
        jq -es --arg arm "$arm" --slurpfile menu "$batch/device/menu.jsonl" \
            -f "$train_methods/grade/elect.jq" "$batch/grade/candidates.jsonl" > "$elected"
        cmp_file "$elected" "$batch/grade/election.json"
        id=$(jq -er '.chosen.id // "base"' "$elected")
        state="$m/states/$arm.state"; if [[ $round == 1 ]]; then state="$m/train/$arm/first.state"; fi
        cmp_file "$batch/device/$id.state" "$state"
    done
    cmp_file "$m/train/$arm/first.state" "$m/train/$arm/round-2/device/base.state"
done

for domain in held contrast; do
    p="$panel/$domain"; d="$raw/$domain"; j="$out/joined/$domain"
    mkdir "$j"
    for arm in "${arms[@]}"; do
        kind=learned
        case $arm in
            native|unfitted|correct_link|exchanged_link) kind=$arm ;;
            gap|ordered_gap) kind=governor_gap ;;
        esac
        join_rows "$domain" "$p" "$kind" "$d/$arm.raw.jsonl" > "$j/$arm.rows.jsonl"
        cmp_file "$j/$arm.rows.jsonl" "$d/$arm.rows.jsonl"
        pair_rows "$domain" < "$j/$arm.rows.jsonl" > "$j/$arm.pairs.json"
        cmp_file "$j/$arm.pairs.json" "$d/$arm.pairs.json"

        case $arm in
            strongest|all|ordered-all)
                geometry="$p/blind/text.bin"; gap=gap
                if [[ $arm == ordered-all ]]; then geometry="$p/ordered/text.bin"; gap=ordered_gap; fi
                "$archived_replay" "$geometry" "$archived_banks" "$out/bundles/$arm.state" "$d/$arm.raw.jsonl" \
                    "$d/$gap.raw.jsonl" > "$j/$arm.replayed.json"
                cmp_file "$j/$arm.replayed.json" "$d/$arm.replayed.json"
                ;;
        esac
    done
done
seal "$out/joined"
bash "$methods/grade-outputs-wick.sh" "$m" "$out/joined" "$panel" "$out/grade"
g="$out/grade"
grade_paths() { (cd "$1" && find . -type f ! -name SHA256SUMS ! -name inputs.sha256 -printf '%P\n' | sort); }
cmp <(grade_paths "$g") <(grade_paths "$grade")
while IFS= read -r file; do cmp_file "$g/$file" "$grade/$file"; done < <(grade_paths "$g")
archived_far="$ancestry/20261126-codex-mixed-transport/measurement-v1/states/far_only.state"
"$held_build/bridge-audit" held "$archived_banks" "$frozen/states/protected.state" "$archived_far" \
    "$panel" "$m" "$raw" "$g" "$out/held-readback.json"
cmp_file "$out/held-readback.json" "$held_audit/readback.json"

case $(jq -er '.client.status' "$meta") in
NOT_RUN)
    refusal="$root/client/refusal-$client"; verify "$refusal"
    jq -e 'select(.pass==false and .passing==[] and (.arms|length)==3)|{status:"NOT_RUN",reason:"fresh source gate",passing,arms:(.arms|map_values([.gate|to_entries[]|select(.value==false)|.key]))}' \
        "$g/summary.json" > "$out/client.json"
    cmp_file "$out/client.json" "$refusal/receipt.json"
    ;;
CAPTURED)
    jq -e '.pass and (.passing|length)>0' "$g/summary.json" >/dev/null
    for dir in capture writer tokens; do verify "$root/client/$dir-$client"; done
    token_grade="$ancestry/20261126-codex-text-governor-links/client/adapter-v1/audit-tokens.jq"
    jq -ce -f "$token_grade" "$root/client/capture-$client/grade/"*.jsonl > "$out/tokens.jsonl"
    cmp_file "$out/tokens.jsonl" "$root/client/tokens-$client/tokens.jsonl"
    ;;
*) exit 65 ;;
esac

jq -n --arg measurement "$(digest "$m/SHA256SUMS")" --arg held "$(digest "$raw/SHA256SUMS")" \
    --arg grade "$(digest "$grade/SHA256SUMS")" --arg release "$(digest "$release/inputs.sha256")" \
    --arg permit "$(digest "$permit/inputs.sha256")" --arg summary "$(digest "$g/summary.json")" \
    --arg writer "$(jq -er '.client.status' "$meta")" --argjson compared "$(wc -l < "$out/compared.txt")" \
    '{schema:"uor-r4.codex-4297.cold-replay/1",pass:true,disagreements:0,
     measurement_sha256:$measurement,held_sha256:$held,grade_sha256:$grade,release_sha256:$release,
     train_permit_sha256:$permit,summary_sha256:$summary,held_sets:2,lanes_per_set:12,
     independent_held_readback:true,cpu_replay_lanes:6,train_audited_batches:6,train_elections:6,
     compared_files:$compared,writer:$writer}' > "$out/receipt.json"
