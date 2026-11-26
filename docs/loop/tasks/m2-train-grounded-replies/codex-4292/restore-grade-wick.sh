#!/usr/bin/env bash
# Reconstruct omitted geometry and verify recorded physical decisions without a GPU dispatch.
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
storage="$s/storage"
previous="$runs/20261126-codex-mixed-transport"
history="$runs/20261125-codex-governing-source-pairs"
source="$runs/20261126-codex-governor-noun-links"
signed="$runs/20261126-codex-text-governor-links"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$history/models/fold-8/ordered-all.d11"
banks="$history/fit-v1/primary.words"
exporter="$signed/export/build-v2/target/release/source-link-text"
legacy="$source/device/build-v2/target/release/governor-noun-links"
joint="$signed/parity/build-v2/release/textlinks-trace-audit"
train_panel="$previous/panel/panel-v1/mixed"
original="$previous/measurement-v1/train/mixed"
parent="$original/first/step-0.state"
first_step="$original/first/step-1.state"
far_only="$previous/measurement-v1/states/far_only.state"
prev="$previous/methods-v2"

[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
sha(){ sha256sum "$1"|cut -d' ' -f1; }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }

if [[ ${1:?extracted archive or download} == download ]]; then
    receipt=${2:?immutable archive receipt}; restore=${3:?new download directory}
    [[ $# == 3 && $restore == "$storage"/download-v* ]]
    repo=$(jq -er '.repo|select(.=="caseyallard/uor-r4-store")' "$receipt")
    prefix=$(jq -er '.path|select(test("^results/codex/protected-transport-20261126-[0-9a-f]{12}$"))' "$receipt")
    revision=$(jq -er '.revision|select(test("^[0-9a-f]{40}$"))' "$receipt")
    archive=$(jq -er '.archive_sha256|select(test("^[0-9a-f]{64}$"))' "$receipt")
    mkdir "$restore" "$restore/extracted"
    "$HOME/.local/bin/hf" download "$repo" --repo-type dataset --revision "$revision" \
        --include "$prefix/*" --local-dir "$restore/download" --quiet > "$restore/download.log"
    packet="$restore/download/$prefix"
    [[ $(sha "$packet/measurement.tar.zst") == "$archive" ]]; verify "$packet"
    tar --zstd -xf "$packet/measurement.tar.zst" -C "$restore/extracted"
    (cd "$restore/extracted" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
    exec bash "$0" "$restore/extracted" "$restore/replayed"
fi

root=$(realpath "$1")
out=${2:?new replay directory}
[[ $# == 2 && $root == "$storage"/*/extracted && $out == "${root%/extracted}/replayed" ]]
verify_inputs(){
    local manifest=$1 hash file
    while read -r hash file; do
        [[ $hash =~ ^[0-9a-f]{64}$ && $file == "$runs/"* ]]
        if [[ $file == "$s/"* ]]; then file="$root/${file#"$s"/}"; fi
        printf '%s  %s\n' "$hash" "$file" | sha256sum --strict --quiet -c -
    done < "$manifest"
}
rehydrate_panel(){
    local p=$1 geometry=$2 relative expected
    mkdir "$geometry"
    # Verify the inputs to both programs before rebuilding. The authority also binds the private grammar.
    verify_inputs "$p/authority.sha256"
    "$exporter" pack "$p/held/user-only.json" "$p/history.json" "$model" "$codec" "$geometry/blind"
    cmp "$p/held/blind/coordinates.jsonl" "$geometry/blind/coordinates.jsonl"
    "$exporter" supervise "$p/held/reference.json" "$geometry/blind/coordinates.jsonl" "$geometry/links.bin" "$geometry/aid.bin"
    cmp "$p/held/links.bin" "$geometry/links.bin"; cmp "$p/held/aid.bin" "$geometry/aid.bin"
    "$legacy" prepare "$p/held/reference.json" "$model" "$codec" "$banks" "$geometry/structural-links"
    for relative in "$p/held/structural-links/"*; do
        # The omitted geometry does not exist yet. Compare every supplied packet and its labels.
        cmp "$relative" "$geometry/structural-links/${relative##*/}"
    done
    for relative in blind/text.bin structural-links/geometry.bin; do
        expected=$(awk -v name="./held/$relative" '$2==name { print $1 }' "$p/SHA256SUMS")
        [[ $expected =~ ^[0-9a-f]{64}$ && $expected == "$(sha "$geometry/$relative")" ]]
        cp --reflink=auto "$geometry/$relative" "$p/held/$relative"
    done
    verify "$p"
}

# Panel-only mode exercises restoration from the compact member set before a TRAIN archive exists.
if [[ -f $root/panel-only.txt ]]; then
    mkdir "$out"
    rehydrate_panel "$root/panel/panel-v1" "$out/geometry"
    jq -n '{pass:true, geometry_members_reconstructed:2, substantive_disagreements:0}' > "$out/receipt.json"
    exit
fi

meta=$(find "$root/storage" -name archive.json -type f)
[[ $meta == "$root"/storage/archive-v*/archive.json && $meta != *$'\n'* ]]
rel() { jq -er --arg key "$1" --arg pattern "$2" '.[$key]|select(test($pattern))' "$meta"; }
m="$root/$(rel measurement '^measurement-v[1-9][0-9]*$')"
release="$root/$(rel release '^release-v[1-9][0-9]*$')"
here="$root/$(rel methods '^methods-v[1-9][0-9]*$')"
panel="$root/$(rel panel '^panel/panel-v[1-9][0-9]*$')"
audit_build="$root/$(rel audit_build '^audit/build-v[1-9][0-9]*$')"
premise=$(jq -er '.premise|select(type=="boolean")|tostring' "$meta")
[[ $(jq -er '.pass|tostring' "$m/premise.json") == "$premise" ]]
for dir in "$m" "$here" "$audit_build" "$prev"; do verify "$dir"; done
mkdir "$out"
rehydrate_panel "$panel" "$out/geometry"
antecedents="$root/panel/antecedents-v1"
verify "$antecedents"; verify_inputs "$antecedents/inputs.sha256"
for manifest in "$m/inputs.sha256" "$release/inputs.sha256" "$here/previous.sha256"; do verify_inputs "$manifest"; done
p="$panel/held"

rows(){
    local split=$1 panel=$2 kind=$3 physical=$4
    jq -nce --arg split "$split" --arg kind "$kind" --slurpfile reference "$panel/reference.json" \
        --slurpfile coords "$panel/blind/coordinates.jsonl" --slurpfile physical "$physical" -f "$prev/grade/join.jq"
}
initial="$m/train/initial"
rows mixed "$train_panel" learned "$initial/reader.jsonl" | jq -es -f "$prev/grade/pairs.jq" > "$out/premise-pairs.json"
cmp "$initial/pairs.json" "$out/premise-pairs.json"
cmp "$initial/pairs.json" "$original/step-0.pairs.json"
cmp "$initial/edges.jsonl" "$original/initial.jsonl"
cmp "$initial/reader.jsonl" "$original/step-0.raw.jsonl"
if [[ $premise == false ]]; then
    jq -n '{pass:true,premise:false,geometry_members_reconstructed:2,substantive_disagreements:0}' > "$out/receipt.json"
    exit
fi

verify_inputs "$m/selection.sha256"
t="$root/$(rel transfer '^transfer-v[1-9][0-9]*$')"
grade="$root/$(rel grade '^grade-v[1-9][0-9]*$')"
permit="$root/$(rel train_permit '^train-verified-v[1-9][0-9]*/inputs\.sha256$')"
for dir in "$t" "$grade"; do verify "$dir"; done
for manifest in "$t/inputs.sha256" "$grade/inputs.sha256" "$permit"; do verify_inputs "$manifest"; done
TRAIN_AUDIT=$(awk -v prefix="$s/audit/train-v" 'index($2,prefix)==1 && $2~/\/inputs.sha256$/ {print $2}' "$permit")
[[ $TRAIN_AUDIT == "$s"/audit/train-v*/inputs.sha256 && $TRAIN_AUDIT != *$'\n'* ]]
TRAIN_AUDIT="$root/${TRAIN_AUDIT#"$s"/}"
verify "${TRAIN_AUDIT%/*}"; verify_inputs "$TRAIN_AUDIT"
d="$t/held"
mkdir "$out/train" "$out/held" "$out/controls"
mapfile -t batches < <(jq -er '.batches[]' "${permit%/*}/verified.json")
for label in "${batches[@]}"; do
    [[ $label == root || $label == protected || $label == free ]]
    batch="$m/train/$label"
    input="$batch/device"; graded="$batch/grade"
    verify "$input"; verify "$graded"
    "$audit_build/protected-transport-audit" batch "$banks" "$train_panel/blind/text.bin" "$train_panel/links.bin" \
        "$parent" "$first_step" "$input/base.state" "$input" "$train_panel/reference.json" "$out/train/$label.readback.json"
    # The permit binds either direct batch audits or a sealed wrapper that binds them.
    mapfile -t recorded < <(find "$root/audit" -path "*/batch-*/readback.json" -type f |
        while IFS= read -r candidate; do
            if grep -Fxq "$(sha "$input/SHA256SUMS")  $s/${input#"$root"/}/SHA256SUMS" "${candidate%/*}/inputs.sha256"; then
                printf '%s\n' "$candidate"
            fi
        done)
    [[ ${#recorded[@]} == 1 ]]
    verify "${recorded[0]%/*}"
    verify_inputs "${recorded[0]%/*}/inputs.sha256"
    cmp "${recorded[0]}" "$out/train/$label.readback.json"
    jq -e --slurpfile origin "$out/train/root.readback.json" --slurpfile grade "$graded/candidates.jsonl" \
        --slurpfile device "$graded/election.json" -f "$audit_build/election.jq" "$out/train/$label.readback.json" \
        > "$out/train/$label.election.json"
    cmp "${TRAIN_AUDIT%/*}/$label.election.json" "$out/train/$label.election.json"
done
for arm in protected free; do
    label=$(cat "$m/train/$arm.batch"); [[ $label == root || $label == protected || $label == free ]]
    first_id=$(jq -r --arg arm "$arm" '.[$arm].id // "base"' "$out/train/root.election.json")
    id=$(jq -r --arg arm "$arm" '.[$arm].id // "base"' "$out/train/$label.election.json")
    cmp "$m/states/$arm-first.state" "$m/train/root/device/$first_id.state"
    cmp "$m/states/$arm-first.state" "$m/train/$label/device/base.state"
    cmp "$m/states/$arm.state" "$m/train/$label/device/$id.state"
    jq -e --arg arm "$arm" --arg state "$(sha "$m/states/$arm.state")" \
        --slurpfile select "$m/selection.json" '.[$arm]==$select[0][$arm] and $select[0].state_sha256[$arm]==$state' \
        "$out/train/$label.election.json" >/dev/null
done
for arm in native unfitted gap baseline far_only free protected correct_link exchanged_link; do
    kind=$arm
    case $arm in
        gap) kind=governor_gap ;;
        baseline) kind=learned; state=$parent ;;
        far_only) kind=learned; state=$far_only ;;
        free|protected) kind=learned; state="$m/states/$arm.state" ;;
    esac
    rows held "$p" "$kind" "$d/$arm.raw.jsonl" > "$out/held/$arm.rows.jsonl"
    cmp "$d/$arm.rows.jsonl" "$out/held/$arm.rows.jsonl"
    jq -es -f "$prev/grade/pairs.jq" "$out/held/$arm.rows.jsonl" > "$out/held/$arm.pairs.json"
    cmp "$d/$arm.pairs.json" "$out/held/$arm.pairs.json"
    if [[ $kind == learned ]]; then
        "$joint" "$p/blind/text.bin" "$banks" "$state" "$d/$arm.raw.jsonl" "$d/gap.raw.jsonl" > "$out/held/$arm.replay.json"
        cmp "$d/$arm.replay.json" "$out/held/$arm.replay.json"
    fi
done
for arm in native governor_only nouns_only correct_link exchanged_link; do
    cp --reflink=auto "$t/controls/$arm.rank" "$out/controls/"
done
"$legacy" grade "$p/structural-links" "$model" "$banks" "$out/controls"
cmp "$t/controls/endpoints.jsonl" "$out/controls/endpoints.jsonl"
cmp "$t/controls/parity.json" "$out/controls/parity.json"

# Replay the authenticated grade body with the restored paths and independently joined rows.
g="$out/grade"; mkdir "$g"; d="$out/held"
awk '
    $0=="mkdir \"$g\"" { starts++; next }
    /^sha256sum .*inputs\.sha256.*seal "\$g"/ { ends++; next }
    starts==1 && ends==0 { print }
    END { if (starts!=1 || ends!=1) exit 1 }
' "$here/grade-held-wick.sh" > "$out/grade-commands.sh"
# shellcheck disable=SC1091
source "$out/grade-commands.sh"
for name in arms.json ids.json scores.json endpoints.jsonl summary.json movements.json protected.witness.json free.witness.json; do
    cmp "$g/$name" "$grade/$name"
done

gate_pass=$(jq -er '.pass|select(type=="boolean")|tostring' "$grade/summary.json")
writer_rows=0
if [[ $gate_pass == true ]]; then
    client="$root/$(rel client '^client/capture-v[1-9][0-9]*$')"
    tokens="$root/$(rel tokens '^client/token-v[1-9][0-9]*$')"
    verify "$client"; verify "$tokens"; verify_inputs "$tokens/inputs.sha256"
    token_audit="$signed/client/adapter-v1/audit-tokens.jq"
    jq -ce -f "$token_audit" "$client/grade/"*.jsonl > "$out/tokens.jsonl"
    cmp "$tokens/tokens.jsonl" "$out/tokens.jsonl"
    writer_rows=$(wc -l < "$out/tokens.jsonl")
fi
jq -n --argjson batches "${#batches[@]}" --argjson writer "$writer_rows" \
    --arg measure "$(sha "$m/SHA256SUMS")" --arg transfer "$(sha "$t/SHA256SUMS")" \
    --arg grade_hash "$(sha "$grade/SHA256SUMS")" --arg gate "$(sha "$grade/summary.json")" '
    {pass:true,premise:true,geometry_members_reconstructed:2,audited_unique_batches:$batches,
    held_arms:9,writer_token_rows:$writer,measurement_sha256:$measure,transfer_sha256:$transfer,
    grade_sha256:$grade_hash,writer_gate_sha256:$gate,substantive_disagreements:0}' > "$out/receipt.json"
