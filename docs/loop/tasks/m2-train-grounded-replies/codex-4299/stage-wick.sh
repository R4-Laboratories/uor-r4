#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
v=${1:?new methods version}; version "$v"; [[ $here == "$s"/incoming-v* ]]
dest="$s/methods-$v"; mkdir "$dest" "$dest/inputs" "$dest/grade"
cp "$here/"*.sh "$here/"{ORDER.md,PRE-RUN.md,DECISION.md} "$dest/"
cp "$here/grade/"*.jq "$dest/grade/"
old="$parent/methods-v4"
pin c48af0d0234836fbbd4cdf9550b38b74927814dc462c388f28648bca4f8c0f82 "$old/SHA256SUMS"
verify "$old"
cp "$old/"original-{join,pairs,summary}.jq "$old/grade/"{gate,contrast,contrast-pairs,witness}.jq "$dest/"
cp "$protected/states/protected.state" "$protected/premise.json" "$mixed/measurement-v1/states/far_only.state" "$dest/inputs/"
cp "$mixed/measurement-v1/train/mixed/step-0.pairs.json" "$dest/inputs/mixed-pairs.json"
cp "$parent/finite-core-v3/bundle" "$banks" "$replay" "$reader" "$dest/inputs/"
cp "$parent/measurement-v1/train/strongest/round-1/device/base.reader.jsonl" "$dest/inputs/endpoint.original.jsonl"
cp "$parent/measurement-v1/train/ordered-all/round-1/device/base.reader.jsonl" "$dest/inputs/ordered.original.jsonl"
sha256sum "$old/SHA256SUMS" "$parent/measurement-v1/SHA256SUMS" "$parent/held-v1/SHA256SUMS" \
    "$parent/panel/panel-v1/SHA256SUMS" "$parent/storage/archive-v2/published-v1/receipt.json" \
    "$reader" "$banks" "$replay" "$lock" "$parent/client/methods-v4/SHA256SUMS" > "$dest/ancestor.sha256"
(cd "$dest" && source ./common-wick.sh && physical_pins)
seal "$dest"; sha256sum "$dest/SHA256SUMS"

