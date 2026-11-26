#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-protected-transport/panel"
mixed="$runs/20261126-codex-mixed-transport/panel"
far="$runs/20261126-codex-far-transport-margin/panel"
signed="$runs/20261126-codex-text-governor-links"
previous="$runs/20261126-codex-governor-noun-links"
prior="$runs/20261125-codex-governing-source-pairs"
here=$(cd "$(dirname "$0")" && pwd)
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$prior/models/fold-8/ordered-all.d11"
fit="$prior/fit-v1/primary.words"
teacher="$prior/train-v1/teacher.jsonl"
capture="$prior/held/primary/capture-stock-v1/stock/fresh/rows.jsonl"
history="$previous/panel/panel-v1/history.json"
generator="$previous/panel/target/release/governor-links-panel"
enrollment="$previous/panel/source-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel/enrollment.jq"
legacy_build="$previous/device/build-v2"
legacy="$legacy_build/target/release/governor-noun-links"
export_build="$signed/export/build-v2"
exporter="$export_build/target/release/source-link-text"
design="$root/private/held-v4.json"
prior_designs=(
    "$previous/panel/grammar-v4.json" "$signed/panel/grammar-v1.json"
    "$far/private/"{train-a-v2,train-b-v2,held-v3}.json
    "$mixed/private/"{a-v2,b-v1,held-v2}.json
)
prior_rows=(
    "$previous/panel/panel-v1/fresh/reference.json" "$signed/panel/panel-v1/fresh/reference.json"
    "$far/panel-v1/train/"{full-a,full-b}/reference.json "$far/panel-v1/held/reference.json"
    "$mixed/panel-v1/layout/"{a,b}/reference.json "$mixed/panel-v1/held/reference.json"
    "$prior/reserve/panel-v"{1,2}/fresh/reference.json
)

version=${1:?unused positive output version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
out="$root/panel-v$version"
partial="$out.partial.$$"
[[ ! -e $out && ! -e $partial ]]

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
prior_panel() {
    local tree=$1 manifest=$2 authority=$3
    pin "$manifest" "$tree/SHA256SUMS"
    pin "$authority" "$tree/authority.sha256"
    (cd "$tree" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
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
for revision in 1 2; do
    reserve="$prior/reserve/panel-v$revision"
    pin 46ebb9f5591ce9e5658846d8c3bd63af2528a8d5ff1959d1b9b5c705042f1f35 "$reserve/SHA256SUMS"
    (cd "$reserve" && sha256sum --strict --quiet -c SHA256SUMS)
done
pin 06154d7e8e763fcc710f9de27a46d001a69330e2727c3c8d730e10e64d2cd7b5 "$generator"
pin ff71ba32ec6f99a3d98de5d2ad4dd1bed60e0e71350d89f8404819e209bf2eb1 "$design"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$model"
pin 45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389 "$history"
pin 0d7db70bd8f89bf3a5ac634a543a077fb864c211211cbedc7091107328aa5902 "$capture"
(cd "$legacy_build" && sha256sum --strict --quiet -c SHA256SUMS dependencies.sha256)
(cd "$export_build" && sha256sum --strict --quiet -c SHA256SUMS)

mkdir -p "$partial/held"
target="$partial/held"
head -n 1 "$capture" | jq -ce '.record.turns[0] | {user,prompt_ids,reply_ids,stop,reply_text}' > "$partial/history.json"
cmp "$history" "$partial/history.json"
"$generator" "$codec" "$model" "$partial/history.json" "$design" "$target/fresh" > "$partial/fixture.txt"
ln "$target/fresh/reference.json" "$target/reference.json"
jq -e -f "$enrollment" "$target/reference.json" > "$partial/enrollment.json"

jq -ce 'map(.turns[1]) | select(length == 256 and all(.[]; type == "string"))' \
    "$target/reference.json" > "$target/user-only.json"
jq -er '.[] | .distance | if . == "far" then 0 elif . == "near" then 1
    else error("unrecognized distance") end' "$target/reference.json" > "$target/distance.txt"
jq -e 'to_entries | group_by(.value.pair_id) | map({
    pair_id:.[0].value.pair_id, distance:.[0].value.distance,
    endpoints:map({ordinal:.key,id:.value.id,admission:.value.admission})
}) | select(length == 128)' "$target/reference.json" > "$target/pairs.json"

# Complete the blind export before attaching any supervised coordinates.
"$exporter" pack "$target/user-only.json" "$partial/history.json" "$model" "$codec" "$target/blind"
guard() {
    local grammar=$1 reference=$2 coordinates=$3
    jq -e --slurpfile design "$grammar" --slurpfile coordinates "$coordinates" \
        --slurpfile history "$partial/history.json" --slurpfile teacher "$teacher" \
        --slurpfile past_designs <(jq -sc '.[]' "${prior_designs[@]}") \
        --slurpfile past_rows <(jq -c '.' "${prior_rows[@]}") \
        -f "$here/guard.jq" "$reference"
}
guard "$design" "$target/reference.json" "$target/blind/coordinates.jsonl" > "$partial/guard.json"

# Deliberately perturb a training digest and a blind noun address.
prior_digest=$(jq -er '.[0].structural_prompt_sha256' "${prior_rows[5]}")
if guard "$design" <(jq -c --arg digest "$prior_digest" '.[0].structural_prompt_sha256=$digest' "$target/reference.json") \
    "$target/blind/coordinates.jsonl" >"$partial/duplicate-digest.stdout" 2>"$partial/duplicate-digest.stderr"; then
    echo 'A duplicate prompt digest passed the guard' >&2
    exit 1
fi
grep -Fq 'encoded prompt digest overlap' "$partial/duplicate-digest.stderr"
if guard "$design" "$target/reference.json" \
    <(jq -c 'if .ordinal==0 then .sites |= map(.tokens.start += 1) else . end' "$target/blind/coordinates.jsonl") \
    >"$partial/moved-address.stdout" 2>"$partial/moved-address.stderr"; then
    echo 'A shifted BPE address passed the guard' >&2
    exit 1
fi
grep -Fq 'blind byte/BPE address' "$partial/moved-address.stderr"
sed -i 's/jq: error (at .*): /jq: error: /' "$partial/"{duplicate-digest,moved-address}.stderr

"$exporter" supervise "$target/reference.json" "$target/blind/coordinates.jsonl" "$target/links.bin" "$target/aid.bin"
"$legacy" prepare "$target/reference.json" "$model" "$codec" "$fit" "$target/structural-links"
test -s "$target/structural-links/correct_link.bin"
test -s "$target/structural-links/exchanged_link.bin"
if cmp -s "$target/structural-links/"{correct_link,exchanged_link}.bin; then
    echo 'Correct and exchanged links serialize to identical bytes' >&2
    exit 1
fi
cmp <(jq -r '.[] | [.fold,.id] | @tsv' "$target/reference.json") "$target/fresh/jobs.tsv"

sha256sum "$here/"{prepare-wick.sh,guard.jq} "$codec" "$model" "$fit" "$teacher" "$history" "$capture" \
    "$generator" "$enrollment" "$legacy" "$exporter" "$legacy_build/SHA256SUMS" "$export_build/SHA256SUMS" \
    "$mixed/panel-v1/"{SHA256SUMS,authority.sha256} "$far/panel-v1/"{SHA256SUMS,authority.sha256} \
    "$previous/panel/panel-v1/"{SHA256SUMS,authority.sha256} "$signed/panel/panel-v1/"{SHA256SUMS,authority.sha256} \
    "$prior/reserve/panel-v"{1,2}/SHA256SUMS "${prior_designs[@]}" "${prior_rows[@]}" "$design" > "$partial/authority.sha256"
(
    cd "$partial"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,authority.sha256,guard.json,held/reference.json,held/blind/text.bin,held/blind/coordinates.jsonl,held/aid.bin}
