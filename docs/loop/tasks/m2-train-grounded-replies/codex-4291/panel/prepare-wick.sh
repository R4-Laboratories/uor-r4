#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-mixed-transport/panel"
previous="$runs/20261126-codex-governor-noun-links"
signed="$runs/20261126-codex-text-governor-links"
parent="$runs/20261126-codex-far-transport-margin/panel"
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
train_a="$root/private/a-v2.json"
train_b="$root/private/b-v1.json"
held="$root/private/held-v2.json"
grammar_manifest="$root/private/GRAMMARS-v2.sha256"
prior_designs=(
    "$previous/panel/grammar-v4.json"
    "$signed/panel/grammar-v1.json"
    "$parent/private/train-a-v2.json"
    "$parent/private/train-b-v2.json"
    "$parent/private/held-v3.json"
)
prior_rows=(
    "$previous/panel/panel-v1/fresh/reference.json"
    "$signed/panel/panel-v1/fresh/reference.json"
    "$parent/panel-v1/train/full-a/reference.json"
    "$parent/panel-v1/train/full-b/reference.json"
    "$parent/panel-v1/held/reference.json"
)

version=${1:?unused positive output version}
[[ $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root/"* ]]
out="$root/panel-v$version"
partial="$out.partial.$$"
[[ ! -e $out && ! -e $partial ]]

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 0241228fa9f93f7362852d6be551dee465cb94299135d9c277b24991d85d329a "$parent/panel-v1/SHA256SUMS"
pin 65360b8d8376aacca9d51ee35f38c0d964ba33821449c8ec2028c1d5ea8e6d4d "$parent/panel-v1/authority.sha256"
(cd "$parent/panel-v1" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
(cd "$legacy_build" && sha256sum --strict --quiet -c SHA256SUMS dependencies.sha256)
(cd "$export_build" && sha256sum --strict --quiet -c SHA256SUMS)
sha256sum --strict --quiet -c "$grammar_manifest"

mkdir -p "$partial/"{layout,mixed,far,held}
generate() {
    local design=$1 output=$2
    "$generator" "$codec" "$model" "$history" "$design" "$output" > "$output.fixture.txt"
    jq -e -f "$enrollment" "$output/reference.json" > "$output.enrollment.json"
}
generate "$train_a" "$partial/layout/a"
generate "$train_b" "$partial/layout/b"
generate "$held" "$partial/held/fresh"

jq -ce 'map(select(.distance == "far")) | select(length == 128)' \
    "$partial/layout/a/reference.json" > "$partial/layout/common-far.json"
compose() {
    local distance=$1 split=$2
    jq -sce --arg distance "$distance" \
        '.[0] + (.[1] | map(select(.distance == $distance))) | select(length == 256)' \
        "$partial/layout/common-far.json" "$partial/layout/b/reference.json" \
        > "$partial/$split/reference.json"
}
compose near mixed
compose far far
ln "$partial/held/fresh/reference.json" "$partial/held/reference.json"

jq -sce 'select(length == 512 and all(.[]; (.user | type) == "string")) |
    {users:(map(.user) | unique), words:([.[].user | ascii_downcase | scan("[a-z]+")] | unique)}' \
    "$teacher" > "$partial/layout/historical-teacher-index.json"
jq -sc '.' "${prior_designs[@]}" > "$partial/layout/prior-designs.json"
jq -sc '.' "${prior_rows[@]}" > "$partial/layout/prior-rows.json"
guard() {
    local mixed=$1 design=$2
    shift 2
    jq -e --slurpfile mixed_rows "$mixed" --slurpfile far_rows "$partial/far/reference.json" \
        --slurpfile a "$train_a" --slurpfile b "$train_b" --slurpfile held_design "$design" \
        --slurpfile teacher "$partial/layout/historical-teacher-index.json" \
        --slurpfile prior_designs "$partial/layout/prior-designs.json" \
        --slurpfile prior_rows "$partial/layout/prior-rows.json" \
        -f "$here/guard.jq" "$@"
}
guard "$partial/mixed/reference.json" "$held" "$partial/held/reference.json" > "$partial/layout/guard-with-matches.json"
jq -e 'del(.matches)' "$partial/layout/guard-with-matches.json" > "$partial/guard.json"
jq -e '.matches' "$partial/layout/guard-with-matches.json" > "$partial/layout/matched-b-endpoints.json"
if guard "$partial/mixed/reference.json" "$train_b" "$partial/held/reference.json" \
   > "$partial/layout/duplicate-family.stdout" 2> "$partial/layout/duplicate-family.stderr"; then
    echo 'Duplicate training family passed the split guard' >&2
    exit 1
fi
grep -Fq 'wording family overlap' "$partial/layout/duplicate-family.stderr"
digest=$(jq -er '.[0].structural_prompt_sha256' "$partial/mixed/reference.json")
if jq -ce --arg digest "$digest" '.[0].structural_prompt_sha256 = $digest' "$partial/held/reference.json" |
   guard "$partial/mixed/reference.json" "$held" \
   > "$partial/layout/duplicate-digest.stdout" 2> "$partial/layout/duplicate-digest.stderr"; then
    echo 'Held row carrying a training digest passed the split guard' >&2
    exit 1
fi
grep -Fq 'encoded prompt digest overlap' "$partial/layout/duplicate-digest.stderr"
if jq -ce '.[128].placement_name = "mismatched"' "$partial/mixed/reference.json" |
   guard /dev/stdin "$held" "$partial/held/reference.json" \
   > "$partial/layout/mismatched-b.stdout" 2> "$partial/layout/mismatched-b.stderr"; then
    echo 'Altered B partner passed the match guard' >&2
    exit 1
fi
grep -Fq 'unmatched B row 128' "$partial/layout/mismatched-b.stderr"

export_split() {
    local target=$1
    jq -ce '[.[] | .turns[1] | select(type == "string")] | select(length == 256)' \
        "$target/reference.json" > "$target/user-only.json"
    jq -er '.[] | .distance | if . == "far" then 0 elif . == "near" then 1
        else error("unrecognized distance") end' "$target/reference.json" > "$target/distance.txt"
    jq -e 'to_entries | group_by(.value.pair_id) | map({
        pair_id:.[0].value.pair_id, distance:.[0].value.distance,
        endpoints:map({ordinal:.key,id:.value.id,admission:.value.admission})
    }) | select(length == 128)' "$target/reference.json" > "$target/pairs.json"
    "$exporter" pack "$target/user-only.json" "$history" "$model" "$codec" "$target/blind"
    "$exporter" supervise "$target/reference.json" "$target/blind/coordinates.jsonl" \
        "$target/links.bin" "$target/aid.bin"
}
export_split "$partial/mixed"
export_split "$partial/far"
export_split "$partial/held"
cmp <(head -n 128 "$partial/mixed/blind/coordinates.jsonl") \
    <(head -n 128 "$partial/far/blind/coordinates.jsonl")

# Preparation verifies the exact candidate addresses without running the ranker.
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
    "$parent/panel-v1/"{SHA256SUMS,authority.sha256} "${prior_designs[@]}" "${prior_rows[@]}" \
    "$train_a" "$train_b" "$held" "$grammar_manifest" > "$partial/authority.sha256"
(
    cd "$partial"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,authority.sha256,guard.json,{mixed,far,held}/reference.json}
