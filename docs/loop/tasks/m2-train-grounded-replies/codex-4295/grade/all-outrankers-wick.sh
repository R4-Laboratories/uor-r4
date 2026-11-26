#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-intervening-source"
mixed="$runs/20261126-codex-mixed-transport/panel/panel-v1"
measurement="$study/measurement-v1"
panel="$study/panel/panel-v1"
grade="$study/grade-v1"
root="$study/supplemental"
[[ $# == 0 && $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs ]]
mkdir -p "$root"

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
member() (
    cd "$1"
    awk -v path="$2" '$2 == path {print; ++n} END {if (n!=1) exit 1}' SHA256SUMS |
        sha256sum --strict --quiet -c -
)
pin ae9471af842a490d39a9b114fd096e194cb61ba9158c495adbbcddae35278fb8 "$measurement/SHA256SUMS"
pin 6c8b8d3a98267078474b9b7167547cbf50bf54792f9e628afa158ca9753351c2 "$panel/SHA256SUMS"
pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$mixed/SHA256SUMS"
pin f3570c7fca46fd961183cc2985845928c02f04a94c45145ba409f58d0fdc9490 "$grade/SHA256SUMS"
member "$measurement" ./raw/neutral.ranks.jsonl
member "$panel" rows.jsonl
member "$mixed" ./mixed/reference.json
member "$mixed" ./mixed/blind/coordinates.jsonl
member "$grade" ./neutral.residuals.jsonl

partial=$(mktemp -d "$root/all-outrankers.partial.XXXXXXXX")
jq -nce --slurpfile panel "$panel/rows.jsonl" \
    --slurpfile rank "$measurement/raw/neutral.ranks.jsonl" \
    --slurpfile grade "$grade/neutral.residuals.jsonl" \
    --slurpfile reference "$mixed/mixed/reference.json" \
    --slurpfile coordinates "$mixed/mixed/blind/coordinates.jsonl" \
    -f "$here/all-outrankers.jq" >"$partial/diagnosis.json"
sha256sum "$measurement/"{SHA256SUMS,raw/neutral.ranks.jsonl} "$panel/"{SHA256SUMS,rows.jsonl} \
    "$mixed/"{SHA256SUMS,mixed/reference.json,mixed/blind/coordinates.jsonl} \
    "$grade/"{SHA256SUMS,neutral.residuals.jsonl} >"$partial/inputs.sha256"
(cd "$here" && sha256sum all-outrankers.jq all-outrankers-wick.sh >"$partial/method.sha256")
(cd "$partial" && sha256sum diagnosis.json inputs.sha256 method.sha256 >SHA256SUMS &&
    sha256sum --strict --quiet -c SHA256SUMS inputs.sha256)
read -r digest _ < <(sha256sum "$partial/SHA256SUMS")
out="$root/all-outrankers-$digest"
if [[ -e $out ]]; then
    (cd "$out" && sha256sum --strict --quiet -c SHA256SUMS inputs.sha256)
    cmp "$out/SHA256SUMS" "$partial/SHA256SUMS"
    chmod -R a-w "$partial"
else
    chmod -R a-w "$partial"
    mv -T --no-clobber "$partial" "$out"
    test ! -e "$partial"
fi
printf '%s\n' "$out"
jq -c '{census,by_family}' "$out/diagnosis.json"
