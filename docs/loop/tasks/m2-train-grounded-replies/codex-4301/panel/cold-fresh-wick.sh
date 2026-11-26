#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd -P)
panel=$(realpath -e "${1:?existing sealed fresh packet}")
grade=$(realpath -e "${2:?existing per-state grades}")
raw=$(realpath -e "${3:?existing physical costs directory}")
election=$(realpath -e "${4:?existing election JSON}")
map=$(realpath -e "${5:?existing independent replay index}")
frozen=$(realpath -e "${6:?existing final reduction}")
out=${7:?new archive and replay directory}
[[ $# == 7 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$(dirname "$out")") == zfs ]]
[[ $(grep -Fc 'all minimizing original tuple ordinals' "$frozen/receipt.txt") == 1 ]]
(cd "$frozen" && sha256sum --strict --quiet -c SHA256SUMS)

mkdir "$out"
out=$(realpath -e "$out")
bundle="$out/input"
mkdir -p "$bundle/"{source,grade,raw,audit} "$out/restored"
cp -R "$panel" "$bundle/panel"
cp "$election" "$bundle/election.json"
cp "$here/"{reduce-fresh-wick.sh,fresh-state.jq,fresh-comparison.jq,pairs.jq} "$bundle/source/"
while IFS= read -r id; do
    [[ $id =~ ^[a-zA-Z0-9_-]+$ ]]
    audit=$(awk -F '\t' -v id="$id" '$1==id {print $2}' "$map")
    [[ -f $audit ]]
    mkdir "$bundle/grade/$id"
    cp "$grade/$id/"{fresh.json,fresh.rows.jsonl} "$bundle/grade/$id/"
    cp "$raw/$id.costs.jsonl" "$bundle/raw/"
    cp "$audit" "$bundle/audit/$id.json"
done < "$frozen/order.txt"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -C "$bundle" -cf - . | \
    zstd -q -T1 -o "$out/inputs.tar.zst"
zstd -q -t "$out/inputs.tar.zst"
zstd -qdc "$out/inputs.tar.zst" | tar -C "$out/restored" -xf -

restore="$out/restored"
while IFS= read -r id; do
    printf '%s\t%s\n' "$id" "$restore/audit/$id.json"
done < "$frozen/order.txt" > "$out/audit.tsv"
bash "$restore/source/reduce-fresh-wick.sh" "$restore/panel" "$restore/grade" "$restore/raw" \
    "$restore/election.json" "$out/audit.tsv" "$out/replay"
cmp "$frozen/SHA256SUMS" "$out/replay/SHA256SUMS"
(
    cd "$out"
    sha256sum inputs.tar.zst replay/SHA256SUMS > cold.sha256
    sha256sum --strict --quiet -c cold.sha256
)
chmod a-w "$out/"{inputs.tar.zst,cold.sha256,audit.tsv}
cat "$out/cold.sha256"
