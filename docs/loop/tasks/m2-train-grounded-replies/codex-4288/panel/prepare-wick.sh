#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-text-governor-links/panel"
previous="$runs/20261126-codex-governor-noun-links"
prior="$runs/20261125-codex-governing-source-pairs"
codec="$runs/20261125-codex-instruction-admission-phase-archive/payload/signed/tokenizer.json"
model="$prior/models/fold-8/ordered-all.d11"
fit="$prior/fit-v1/primary.words"
train_root="$previous/panel/panel-v1"
train="$train_root/fresh/reference.json"
history="$train_root/history.json"
training_grammar="$previous/panel/grammar-v4.json"
grammar="$root/grammar-v1.json"
generator="$previous/panel/target/release/governor-links-panel"
enrollment="$previous/panel/source-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel/enrollment.jq"
device="$previous/device/build-v2"
packet="$device/target/release/governor-noun-links"

version=${1:?unused positive output version}
[[ $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root/"* ]]
out="$root/panel-v$version"
partial="$out.partial.$$"
[[ ! -e $out && ! -e $partial ]]

check_hash() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -
}
check_hash 70e1ec5547c86caf37bd5d0e8de53df349653d7dd33b7041f1e01eb6c11f4655 "$train_root/SHA256SUMS"
check_hash 9876dc230e54f7126f1545dc2f940b28c6112b27f04ef035d3703e8474635b7d "$train_root/authority.sha256"
check_hash b95b52f26debef4dae7197baa2557b228fe60d583c466c8d3d78353271fe88fb "$train"
check_hash c5b9fdb9c514e9ce9a9a8310520611f8defdbe1b17bb36107d589b554a022fea "$training_grammar"
check_hash 7421efc8dd4a3fcb34e8f22c44d2f2ed8161229928ba59fe0aef855a79c1181f "$grammar"
check_hash 06154d7e8e763fcc710f9de27a46d001a69330e2727c3c8d730e10e64d2cd7b5 "$generator"
check_hash 4c70da49f82dc11b09121d15b12791a6203fba9bafe8edd4063b7c893aaffba4 "$device/SHA256SUMS"
check_hash 039d9da586d5d8079043fde4803c00f74807e821449d25c8e09722b5d1228da4 "$packet"
check_hash a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
check_hash 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$model"
check_hash ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$fit"
(cd "$train_root" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
(cd "$device" && sha256sum --strict --quiet -c SHA256SUMS dependencies.sha256)

mkdir "$partial"
"$generator" "$codec" "$model" "$history" "$grammar" "$partial/fresh" > "$partial/fixture.txt"
reference="$partial/fresh/reference.json"
check_hash fcc8031daf05d66a47fda878cbe3015ead90b1b9990dd67a268f316e716edce3 "$reference"
jq -e -f "$enrollment" "$reference" > "$partial/enrollment.json"

split() {
    jq -e --slurpfile train "$train" --slurpfile training_grammar "$training_grammar" \
        --slurpfile held_grammar "$1" --slurpfile history "$history" \
        -f "$here/split.jq" "${@:2}"
}
split "$grammar" "$reference" > "$partial/split.json"
if split "$training_grammar" "$train" > "$partial/duplicate-split.stdout" 2> "$partial/duplicate-split.stderr"; then
    echo 'Guard accepted the same historical panel on both sides' >&2
    exit 1
fi
grep -Fq 'wording family overlap' "$partial/duplicate-split.stderr"
if jq --arg digest "$(jq -r '.[0].structural_prompt_sha256' "$train")" \
        '.[0].structural_prompt_sha256 = $digest' "$reference" |
        split "$grammar" > "$partial/duplicate-prompt.stdout" 2> "$partial/duplicate-prompt.stderr"; then
    echo 'Guard accepted a held prompt with a training digest' >&2
    exit 1
fi
grep -Fq 'encoded prompt digest overlap' "$partial/duplicate-prompt.stderr"

# The old ABI's preparation mode resolves exact occurrences for every proposed
# tuple, validates the four registered tuples, and serializes both interventions.
"$packet" prepare "$reference" "$model" "$codec" "$fit" "$partial/structural-links"
test -s "$partial/structural-links/correct_link.bin"
test -s "$partial/structural-links/exchanged_link.bin"
if cmp -s "$partial/structural-links/correct_link.bin" "$partial/structural-links/exchanged_link.bin"; then
    echo 'Correct and exchanged links serialize to identical bytes' >&2
    exit 1
fi

sha256sum "$here/"{prepare-wick.sh,split.jq} "$train_root/"{SHA256SUMS,authority.sha256} \
    "$train" "$history" "$training_grammar" "$grammar" "$generator" "$enrollment" \
    "$device/SHA256SUMS" "$packet" "$codec" "$model" "$fit" > "$partial/authority.sha256"
(
    cd "$partial"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,fresh/reference.json,enrollment.json,split.json,authority.sha256}
