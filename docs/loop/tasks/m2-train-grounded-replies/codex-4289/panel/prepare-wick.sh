#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-far-transport-margin/panel"
previous="$runs/20261126-codex-governor-noun-links"
signed="$runs/20261126-codex-text-governor-links"
prior="$runs/20261125-codex-governing-source-pairs"
here=$(cd "$(dirname "$0")" && pwd)
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$prior/models/fold-8/ordered-all.d11"
fit="$prior/fit-v1/primary.words"
teacher="$prior/train-v1/teacher.jsonl"
history="$previous/panel/panel-v1/history.json"
generator="$previous/panel/target/release/governor-links-panel"
enrollment="$previous/panel/source-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel/enrollment.jq"
legacy_build="$previous/device/build-v2"
legacy="$legacy_build/target/release/governor-noun-links"
export_build="$signed/export/build-v2"
exporter="$export_build/target/release/source-link-text"
old_a="$previous/panel/grammar-v4.json"
old_b="$signed/panel/grammar-v1.json"
old_rows_a="$previous/panel/panel-v1/fresh/reference.json"
old_rows_b="$signed/panel/panel-v1/fresh/reference.json"
train_a="$root/private/train-a-v2.json"
train_b="$root/private/train-b-v2.json"
held_design="$root/private/held-v3.json"

version=${1:?unused positive output version}
[[ $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root/"* ]]
out="$root/panel-v$version"
partial="$out.partial.$$"
[[ ! -e $out && ! -e $partial ]]

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$model"
pin ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$fit"
pin 66d43f34c7245b23fb3d23564a9d7beabfa00474cfa742618c0bd0edafe2c276 "$teacher"
pin 45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389 "$history"
pin 06154d7e8e763fcc710f9de27a46d001a69330e2727c3c8d730e10e64d2cd7b5 "$generator"
pin 222b35579b251883e79965a67f1367eac63b3ced4616362e4054a5a5d352ff0f "$enrollment"
pin 4c70da49f82dc11b09121d15b12791a6203fba9bafe8edd4063b7c893aaffba4 "$legacy_build/SHA256SUMS"
pin 039d9da586d5d8079043fde4803c00f74807e821449d25c8e09722b5d1228da4 "$legacy"
pin e3e473ff58cbfa1588dcc3293ed44a50054e0a841013c7916513759283e8c9a9 "$export_build/SHA256SUMS"
pin a4fe785d399000309b4807f55780bcff7e53accf30772d2a9bd015a7ebfc80bc "$exporter"
pin c5b9fdb9c514e9ce9a9a8310520611f8defdbe1b17bb36107d589b554a022fea "$old_a"
pin 7421efc8dd4a3fcb34e8f22c44d2f2ed8161229928ba59fe0aef855a79c1181f "$old_b"
pin b95b52f26debef4dae7197baa2557b228fe60d583c466c8d3d78353271fe88fb "$old_rows_a"
pin fcc8031daf05d66a47fda878cbe3015ead90b1b9990dd67a268f316e716edce3 "$old_rows_b"
pin 8988cb23c503ef030492d450ca3249f160327f36cffc9cac9a8e5bcbac3858ec "$train_a"
pin 17f81229522e2bbc351dd1e1b02833e019c53af0b6ff9891cf48633fdeaae7a6 "$train_b"
pin c3db4a006ea1c07d7d7dd18ff12c53124881e44befa840af2d1ba763c5d396de "$held_design"
(cd "$legacy_build" && sha256sum --strict --quiet -c SHA256SUMS dependencies.sha256)
(cd "$export_build" && sha256sum --strict --quiet -c SHA256SUMS)

mkdir -p "$partial/train" "$partial/held"
generate() {
    local design=$1 output=$2
    "$generator" "$codec" "$model" "$history" "$design" "$output" > "$output.fixture.txt"
    jq -e -f "$enrollment" "$output/reference.json" > "$output.enrollment.json"
}
generate "$train_a" "$partial/train/full-a"
generate "$train_b" "$partial/train/full-b"
generate "$held_design" "$partial/held/fresh"

# Use only the fixed far cells of both train designs.
jq -sce 'add | map(select(.distance == "far")) | select(length == 256)' \
    "$partial/train/"full-{a,b}/reference.json > "$partial/train/reference.json"
ln "$partial/held/fresh/reference.json" "$partial/held/reference.json"
jq -sce 'select(length == 512 and all(.[]; (.user | type) == "string")) |
    {users:(map(.user) | unique), words:([.[].user | ascii_downcase | scan("[a-z]+")] | unique)}' \
    "$teacher" > "$partial/historical-teacher-index.json"

guard() {
    local design=$1
    shift
    jq -e --slurpfile training "$partial/train/reference.json" \
        --slurpfile train_a "$train_a" --slurpfile train_b "$train_b" \
        --slurpfile held_design "$design" --slurpfile history "$history" \
        --slurpfile old_a "$old_a" --slurpfile old_b "$old_b" \
        --slurpfile old_rows_a "$old_rows_a" --slurpfile old_rows_b "$old_rows_b" \
        --slurpfile legacy_words "$partial/historical-teacher-index.json" \
        -f "$here/guard.jq" "$@"
}
guard "$held_design" "$partial/held/reference.json" > "$partial/guard.json"
if guard "$train_a" "$partial/held/reference.json" > "$partial/duplicate-family.stdout" 2> "$partial/duplicate-family.stderr"; then
    echo 'Duplicate training family passed the split guard' >&2
    exit 1
fi
rg -Fq 'wording family overlap' "$partial/duplicate-family.stderr"
digest=$(jq -r '.[0].structural_prompt_sha256' "$partial/train/reference.json")
if jq -c --arg digest "$digest" '.[0].structural_prompt_sha256 = $digest' "$partial/held/reference.json" |
   guard "$held_design" > "$partial/duplicate-digest.stdout" 2> "$partial/duplicate-digest.stderr"; then
    echo 'Held row carrying a training digest passed the split guard' >&2
    exit 1
fi
rg -Fq 'encoded prompt digest overlap' "$partial/duplicate-digest.stderr"

export_split() {
    local target=$1
    jq -ce '[.[] | .turns[1] | select(type == "string")] | select(length == 256)' \
        "$target/reference.json" > "$target/user-only.json"
    "$exporter" pack "$target/user-only.json" "$history" "$model" "$codec" "$target/blind"
    "$exporter" supervise "$target/reference.json" "$target/blind/coordinates.jsonl" \
        "$target/links.bin" "$target/aid.bin"
}
export_split "$partial/train"
export_split "$partial/held"

# The inherited preparation mode verifies exact candidate addresses and makes the legacy
# structural controls available to the ordinary client workflow.
"$legacy" prepare "$partial/held/reference.json" "$model" "$codec" "$fit" "$partial/held/structural-links"
test -s "$partial/held/structural-links/correct_link.bin"
test -s "$partial/held/structural-links/exchanged_link.bin"
if cmp -s "$partial/held/structural-links/"{correct_link,exchanged_link}.bin; then
    echo 'Correct and exchanged links serialize to identical bytes' >&2
    exit 1
fi
cmp <(jq -r '.[] | [.fold,.id] | @tsv' "$partial/held/reference.json") "$partial/held/fresh/jobs.tsv"

sha256sum "$here/"{prepare-wick.sh,guard.jq} "$codec" "$model" "$fit" "$teacher" "$history" \
    "$generator" "$enrollment" "$legacy" "$exporter" "$legacy_build/SHA256SUMS" "$export_build/SHA256SUMS" \
    "$old_a" "$old_b" "$old_rows_a" "$old_rows_b" "$train_a" "$train_b" "$held_design" > "$partial/authority.sha256"
(
    cd "$partial"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,authority.sha256,guard.json,{train,held}/reference.json}
