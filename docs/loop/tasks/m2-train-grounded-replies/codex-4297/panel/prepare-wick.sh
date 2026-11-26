#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-all-outranker-interval/panel"
protected="$runs/20261126-codex-protected-transport/panel"
mixed="$runs/20261126-codex-mixed-transport/panel"
far="$runs/20261126-codex-far-transport-margin/panel"
signed="$runs/20261126-codex-text-governor-links"
previous="$runs/20261126-codex-governor-noun-links"
prior="$runs/20261125-codex-governing-source-pairs"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive panel version}
build_version=${2:?sealed positive build version}
[[ $# == 2 && $version =~ ^[1-9][0-9]*$ && $build_version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
build="$root/build-v$build_version"
output="$root/panel-v$version"
partial="$output.partial"
[[ ! -e $output && ! -e $partial ]]

codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$prior/models/fold-8/ordered-all.d11"
fit="$prior/fit-v1/primary.words"
teacher="$prior/train-v1/teacher.jsonl"
capture="$prior/held/primary/capture-stock-v1/stock/fresh/rows.jsonl"
history="$previous/panel/panel-v1/history.json"
frozen_generator="$previous/panel/target/release/governor-links-panel"
enrollment="$previous/panel/source-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel/enrollment.jq"
legacy_build="$previous/device/build-v2"
legacy="$legacy_build/target/release/governor-noun-links"
export_build="$signed/export/build-v2"
exporter="$export_build/target/release/source-link-text"
ordered="$build/target/release/source-link-text"
contrast_generator="$build/target/release/governor-links-panel"
held_design="$root/private/held-v5.json"
contrast_design="$root/private/contrast-v2.json"
prior_designs=(
    "$previous/panel/grammar-v4.json" "$signed/panel/grammar-v1.json"
    "$far/private/"{train-a-v2,train-b-v2,held-v3}.json
    "$mixed/private/"{a-v2,b-v1,held-v2}.json
    "$protected/private/held-v4.json"
)
prior_rows=(
    "$previous/panel/panel-v1/fresh/reference.json" "$signed/panel/panel-v1/fresh/reference.json"
    "$far/panel-v1/train/"{full-a,full-b}/reference.json "$far/panel-v1/held/reference.json"
    "$mixed/panel-v1/layout/"{a,b}/reference.json "$mixed/panel-v1/held/reference.json"
    "$prior/reserve/panel-v"{1,2}/fresh/reference.json
    "$protected/panel-v1/held/reference.json" "$mixed/panel-v1/mixed/reference.json"
    "$prior/held/primary/panel-v1/"{fresh,regress}/reference.json
    "$prior/held/primary/panel-design-v1/fresh/reference.json"
    "$prior/reserve/sandbox-v6/reference.json"
)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
prior_panel() {
    pin "$2" "$1/SHA256SUMS"
    pin "$3" "$1/authority.sha256"
    (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
}
prior_panel "$mixed/panel-v1" \
    72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f \
    1f373ac5bdb3b31b34970a57c261633a96cab78603fc25cb8ef77f2af0ded979
prior_panel "$far/panel-v1" \
    0241228fa9f93f7362852d6be551dee465cb94299135d9c277b24991d85d329a \
    65360b8d8376aacca9d51ee35f38c0d964ba33821449c8ec2028c1d5ea8e6d4d
prior_panel "$previous/panel/panel-v1" \
    70e1ec5547c86caf37bd5d0e8de53df349653d7dd33b7041f1e01eb6c11f4655 \
    9876dc230e54f7126f1545dc2f940b28c6112b27f04ef035d3703e8474635b7d
prior_panel "$signed/panel/panel-v1" \
    85f267274746f65f79f60a3c7e4f8c830c779de5258e5dc63ceb1c063042a6bb \
    c790e7c407cb9115db2da913de417f610206ab9a51f7090eeb2dbc3545655f81
prior_panel "$protected/panel-v1" \
    22213dcba2081598d51e08cb573ca5a721ad26dcb04c20d4fd906a5cbae511cc \
    bd42c8e91a83e5e310f87a8b40ecb518449e1edbf01b274f54734521592ff3b8
for revision in 1 2; do
    reserve="$prior/reserve/panel-v$revision"
    pin 46ebb9f5591ce9e5658846d8c3bd63af2528a8d5ff1959d1b9b5c705042f1f35 "$reserve/SHA256SUMS"
    (cd "$reserve" && sha256sum --strict --quiet -c SHA256SUMS)
done
pin 62067f2a1762ca85385910cb89d0b233cae8a1fa60c3e3c8aa58b395b0a50f8f "${prior_rows[12]}"
pin ef98f04867309e77558b0bfed5ee569727a94f51dcf37d33e8cf50136302848a "${prior_rows[13]}"
pin f96029e315babc6a65823eafc6dfd13034b0705b04147b183a955d625cb50bfd "${prior_rows[14]}"
pin 9229e98dc330b73eca6e6564c627c8149664ed3c87de913b127a8a5cd077b634 "${prior_rows[15]}"
pin 06154d7e8e763fcc710f9de27a46d001a69330e2727c3c8d730e10e64d2cd7b5 "$frozen_generator"
pin 548c16f9c0ec925dfcd30d017b9446ed78a96e7c9e5e3f33c8be2d8bb0424bd3 "$held_design"
pin e1081472c6d654a51ebfd0a4a430f64d6a31ae487c2702e58d637712332a3110 "$contrast_design"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$model"
pin 45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389 "$history"
pin 0d7db70bd8f89bf3a5ac634a543a077fb864c211211cbedc7091107328aa5902 "$capture"
(cd "$legacy_build" && sha256sum --strict --quiet -c SHA256SUMS dependencies.sha256)
(cd "$export_build" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$build" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)

mkdir "$partial"
mkdir "$partial/"{staging,held,contrast,mixed}
head -n 1 "$capture" | jq -ce '.record.turns[0] | {user,prompt_ids,reply_ids,stop,reply_text}' > "$partial/history.json"
cmp "$history" "$partial/history.json"
"$ordered" bridge-fixture "$model" "$codec" > "$partial/bridge-finite.json"
"$frozen_generator" "$codec" "$model" "$partial/history.json" "$held_design" "$partial/staging/held" > "$partial/held/fixture.txt"
"$contrast_generator" --contrast "$codec" "$model" "$partial/history.json" "$contrast_design" "$partial/staging/contrast" > "$partial/contrast/fixture.txt"
cp "$mixed/panel-v1/mixed/user-only.json" "$partial/mixed/"
cp -a "$mixed/panel-v1/mixed/blind" "$partial/mixed/"
for cohort in held contrast; do
    target="$partial/$cohort"
    jq -ce 'map(.turns[1]) | select(all(.[]; type=="string"))' \
        "$partial/staging/$cohort/reference.json" > "$target/user-only.json"
    "$exporter" pack "$target/user-only.json" "$partial/history.json" "$model" "$codec" "$target/blind"
done
for cohort in mixed held contrast; do
    target="$partial/$cohort"
    "$ordered" pack-ordered "$target/user-only.json" "$partial/history.json" "$model" "$codec" "$target/ordered"
    cmp "$target/blind/coordinates.jsonl" "$target/ordered/coordinates.jsonl"
    "$ordered" compare-bridge "$target/blind/text.bin" "$target/ordered/text.bin" > "$target/bridge.json"
done

# Seal the blind exporter outputs before admitting reference joins or labels.
(
    cd "$partial"
    sha256sum history.json {mixed,held,contrast}/user-only.json \
        {mixed,held,contrast}/{blind,ordered}/{text.bin,coordinates.jsonl} > blind.sha256
    sha256sum --strict --quiet -c blind.sha256
)
cp "$mixed/panel-v1/mixed/"{reference.json,links.bin,aid.bin,distance.txt,pairs.json} "$partial/mixed/"
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
done
rmdir "$partial/staging"
jq -e -f "$enrollment" "$partial/held/reference.json" > "$partial/held/enrollment.json"
guard() {
    local expected=$1 grammar=$2 reference=$3 coordinates=$4
    shift 4
    jq -e --argjson expected "$expected" --slurpfile design "$grammar" \
        --slurpfile coordinates "$coordinates" --slurpfile history "$partial/history.json" \
        --slurpfile teacher "$teacher" \
        --slurpfile earlier_grammars <(jq -c '.' "${prior_designs[@]}" ${1:+"$1"}) \
        --slurpfile earlier_rows <(jq -c '.' "${prior_rows[@]}" ${2:+"$2"}) \
        -f "$here/guard.jq" "$reference"
}
guard 256 "$held_design" "$partial/held/reference.json" "$partial/held/blind/coordinates.jsonl" > "$partial/held/guard.json"
guard 64 "$contrast_design" "$partial/contrast/reference.json" "$partial/contrast/blind/coordinates.jsonl" \
    "$held_design" "$partial/held/reference.json" > "$partial/contrast/guard.json"
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

method=docs/loop/tasks/m2-train-grounded-replies/codex-4288/export
native="$prior/artifact-build-v3/source/crates/uor-r4-integer/src"
sha256sum "$build/source/$method/src/"{main.rs,bridge.rs} \
    "$native/"{role_action.rs,role_action/instruction.rs,discrete_d11.rs} "$ordered" > "$partial/operator.sha256"
sha256sum "$here/"{guard.jq,prepare-wick.sh} "$codec" "$model" "$fit" "$teacher" "$history" "$capture" \
    "$frozen_generator" "$enrollment" "$legacy_build/"{SHA256SUMS,dependencies.sha256} \
    "$export_build/SHA256SUMS" "$build/"{SHA256SUMS,authority.sha256} \
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
sha256sum "$output/"{SHA256SUMS,blind.sha256,labels.sha256,operator.sha256,bridge-finite.json} \
    "$output/"{mixed,held,contrast}/ordered/text.bin
