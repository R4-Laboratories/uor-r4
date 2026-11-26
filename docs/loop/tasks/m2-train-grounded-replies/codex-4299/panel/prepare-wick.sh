#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-fixed-assignment-median/panel"
prior="$runs/20261125-codex-governing-source-pairs"
linked="$runs/20261126-codex-governor-noun-links"
signed="$runs/20261126-codex-text-governor-links"
far="$runs/20261126-codex-far-transport-margin/panel"
mixed="$runs/20261126-codex-mixed-transport/panel"
protected="$runs/20261126-codex-protected-transport"
predecessor="$runs/20261126-codex-all-outranker-interval/panel"
parent="$predecessor/panel-v1"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive panel version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
output="$root/panel-v$version"
partial="$output.partial"
[[ ! -e $output && ! -e $partial ]]

codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$prior/models/fold-8/ordered-all.d11"
fit="$prior/fit-v1/primary.words"
teacher="$prior/train-v1/teacher.jsonl"
capture="$prior/held/primary/capture-stock-v1/stock/fresh/rows.jsonl"
primary="$linked/panel/target/release/governor-links-panel"
enrollment="$linked/panel/source-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel/enrollment.jq"
legacy="$linked/device/build-v2/target/release/governor-noun-links"
exporter="$signed/export/build-v2/target/release/source-link-text"
ordered="$predecessor/build-v3/target/release/source-link-text"
selector="$predecessor/build-v3/target/release/governor-links-panel"
guard="$predecessor/methods-v5/guard.jq"
state="$protected/measurement-v1/states/protected.state"
reader="$signed/learner/build-v2/textlinks-sm89"
held_design="$root/private/held-v2.json"
contrast_design="$root/private/contrast-v4.json"
prior_designs=(
    "$linked/panel/grammar-v4.json" "$signed/panel/grammar-v1.json"
    "$far/private/"{train-a-v2,train-b-v2,held-v3}.json
    "$mixed/private/"{a-v2,b-v1,held-v2}.json
    "$protected/panel/private/held-v4.json"
    "$predecessor/private/"{held-v5,contrast-v2}.json
)
prior_rows=(
    "$linked/panel/panel-v1/fresh/reference.json" "$signed/panel/panel-v1/fresh/reference.json"
    "$far/panel-v1/train/"{full-a,full-b}/reference.json "$far/panel-v1/held/reference.json"
    "$mixed/panel-v1/layout/"{a,b}/reference.json "$mixed/panel-v1/held/reference.json"
    "$prior/reserve/panel-v"{1,2}/fresh/reference.json
    "$protected/panel/panel-v1/held/reference.json" "$mixed/panel-v1/mixed/reference.json"
    "$prior/held/primary/panel-v1/"{fresh,regress}/reference.json
    "$prior/held/primary/panel-design-v1/fresh/reference.json"
    "$prior/reserve/sandbox-v6/reference.json"
    "$parent/"{held,contrast}/reference.json
)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 92704b9d3ffb8201245a26fe05c5648e29360d0a383dd0db9ccb1389f845eb4b "$parent/SHA256SUMS"
pin 716cfbb3e281f87c6ad1b76d886b66f0fde45538d1eb10cfe9726ea306de43df "$parent/authority.sha256"
(cd "$parent" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256 blind.sha256 labels.sha256 operator.sha256)
(cd "$predecessor/build-v3" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
(cd "$signed/export/build-v2" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$linked/device/build-v2" && sha256sum --strict --quiet -c SHA256SUMS dependencies.sha256)
pin b6cadb9d62e18d3c22c57a264a08f7c616cc3e91b8fb1de987f0a83d87626691 "$held_design"
pin 70591bf6ae287a2e366fef24111070823667bcc28fc077ee92bce34c370946ad "$contrast_design"
pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$state"
pin 2bd85f7280e6ffec0026f686e64f4376ee6f74aa6d0924816ce56a6291430da0 "$reader"

mkdir "$partial"
mkdir "$partial/"{staging,held,contrast}
head -n 1 "$capture" | jq -ce '.record.turns[0] | {user,prompt_ids,reply_ids,stop,reply_text}' > "$partial/history.json"
cmp "$parent/history.json" "$partial/history.json"
"$ordered" bridge-fixture "$model" "$codec" > "$partial/bridge-finite.json"
cmp "$parent/bridge-finite.json" "$partial/bridge-finite.json"
"$primary" "$codec" "$model" "$partial/history.json" "$held_design" "$partial/staging/held" > "$partial/held/fixture.txt"
"$selector" --contrast "$codec" "$model" "$partial/history.json" "$contrast_design" "$partial/staging/contrast" > "$partial/contrast/fixture.txt"
cp -a "$parent/mixed" "$partial/"
diff -qr "$parent/mixed" "$partial/mixed"

for cohort in held contrast; do
    target="$partial/$cohort"
    jq -ce 'map(.turns[1]) | select(all(.[]; type=="string"))' \
        "$partial/staging/$cohort/reference.json" > "$target/user-only.json"
    "$exporter" pack "$target/user-only.json" "$partial/history.json" "$model" "$codec" "$target/blind"
    "$ordered" pack-ordered "$target/user-only.json" "$partial/history.json" "$model" "$codec" "$target/ordered"
    cmp "$target/blind/coordinates.jsonl" "$target/ordered/coordinates.jsonl"
    "$ordered" compare-bridge "$target/blind/text.bin" "$target/ordered/text.bin" > "$target/bridge.json"
done

# Freeze the physical packets and coordinate maps before adding held labels or controls.
(
    cd "$partial"
    sha256sum history.json {mixed,held,contrast}/user-only.json \
        {mixed,held,contrast}/{blind,ordered}/{text.bin,coordinates.jsonl} > blind.sha256
    sha256sum --strict --quiet -c blind.sha256
)

for cohort in held contrast; do
    target="$partial/$cohort"
    mv -T "$partial/staging/$cohort" "$target/fresh"
    ln "$target/fresh/reference.json" "$target/reference.json"
    jq -er '.[] | .distance | if .=="far" then 0 elif .=="near" then 1
        else error("unrecognized distance") end' "$target/reference.json" > "$target/distance.txt"
    jq -e 'to_entries | group_by(.value.pair_id) | map({
        pair_id:.[0].value.pair_id,distance:.[0].value.distance,
        endpoints:map({ordinal:.key,id:.value.id,admission:.value.admission}
            + if .value.contrast then {selector:.value.contrast.selector} else {} end)
    })' "$target/reference.json" > "$target/pairs.json"
    jq -e --arg cohort "$cohort" -f "$here/design.jq" "$target/reference.json" > "$target/design.json"
done
rmdir "$partial/staging"
jq -e -f "$enrollment" "$partial/held/reference.json" > "$partial/held/enrollment.json"

check_novelty() {
    local cohort=$1 expected=$2 grammar=$3
    shift 3
    jq -e --argjson expected "$expected" --slurpfile design "$grammar" \
        --slurpfile coordinates "$partial/$cohort/blind/coordinates.jsonl" \
        --slurpfile history "$partial/history.json" --slurpfile teacher "$teacher" \
        --slurpfile earlier_grammars <(jq -c '.' "${prior_designs[@]}" ${1:+"$1"}) \
        --slurpfile earlier_rows <(jq -c '.' "${prior_rows[@]}" ${2:+"$2"}) \
        -f "$guard" "$partial/$cohort/reference.json" > "$partial/$cohort/guard.json"
}
check_novelty held 256 "$held_design"
check_novelty contrast 64 "$contrast_design" "$held_design" "$partial/held/reference.json"
"$ordered" compare-contrast "$partial/contrast/reference.json" \
    "$partial/contrast/blind/text.bin" "$partial/contrast/ordered/text.bin" > "$partial/contrast/matched.json"

for cohort in held contrast; do
    target="$partial/$cohort"
    "$exporter" supervise "$target/reference.json" "$target/blind/coordinates.jsonl" "$target/links.bin" "$target/aid.bin"
    "$legacy" prepare "$target/reference.json" "$model" "$codec" "$fit" "$target/structural-links"
    cmp <(jq -r '.[] | [.fold,.id] | @tsv' "$target/reference.json") "$target/fresh/jobs.tsv"
    if cmp -s "$target/structural-links/"{correct_link,exchanged_link}.bin; then
        echo "Identical correct and exchanged supplied links for $cohort" >&2
        exit 1
    fi
done

cp "$parent/operator.sha256" "$partial/operator.sha256"
sha256sum "$here/"{prepare-wick.sh,design.jq} "$parent/"{SHA256SUMS,authority.sha256,operator.sha256} \
    "$guard" "$codec" "$model" "$fit" "$teacher" "$capture" "$enrollment" "$state" "$reader" \
    "$primary" "$legacy" "$exporter" "$ordered" "$selector" \
    "${prior_designs[@]}" "${prior_rows[@]}" "$held_design" "$contrast_design" > "$partial/authority.sha256"
(
    cd "$partial"
    sha256sum {mixed,held,contrast}/{reference.json,links.bin,aid.bin,distance.txt,pairs.json} \
        {held,contrast}/structural-links/{geometry.bin,correct_link.bin,exchanged_link.bin,labels.json} > labels.sha256
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS blind.sha256 labels.sha256 operator.sha256 authority.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$output"
test ! -e "$partial"
sha256sum "$output/"{SHA256SUMS,blind.sha256,labels.sha256,operator.sha256}
