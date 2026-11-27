#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 5 ]]; then
  echo 'usage: readback.sh NEW_ROOT PACKET.tar.zst EXPECTED_SHA256 BASE.d11 ORDINARY_PROBE' >&2
  exit 2
fi
out=$(realpath "$1")
packet=$(realpath "$2")
expected_packet=$3
base=$(realpath "$4")
probe=$(realpath "$5")
printf '%s  %s\n' "$expected_packet" "$packet" | sha256sum --strict --quiet -c -
mkdir "$out/payload" "$out/rebuilt"
tar --zstd -C "$out/payload" -xf "$packet"
cd "$out/payload"
sha256sum --strict --quiet -c PAYLOAD.sha256
(cd measured-v1 && sha256sum --strict --quiet -c OUTPUTS.sha256)
(cd panels/held && sha256sum --strict --quiet -c SHA256SUMS)
(cd product-ordinary-v1 && sha256sum --strict --quiet -c SHA256SUMS)
for arm in rowwise paired; do
  cmp "measured-v1/raw/fit/$arm.i16le" "measured-v1/raw/repeated/$arm.i16le"
done
for file in schedule.jsonl authenticated.jsonl rowwise-slots.jsonl paired-slots.jsonl; do
  cmp "measured-v1/raw/fit/$file" "measured-v1/raw/repeated/$file"
done
jq -n --slurpfile prior prior-schedule.jsonl --slurpfile groups measured-v1/schedule.jsonl \
  -f source/accounting/matched-schedule.jq > "$out/schedule-check.json"
cmp schedule-check.json "$out/schedule-check.json"
bash source/accounting/check-fit.sh measured-v1/raw/fit prior-schedule.jsonl "$out/accounting"
cmp schedule-check.json "$out/accounting/exposures.json"
for file in exposures.json rowwise-last.json paired-last.json; do
  cmp "update-accounting-v1/$file" "$out/accounting/$file"
done
for arm in rowwise paired; do
  case "$arm" in
    rowwise) slot=first-period ;;
    paired) slot=joint ;;
  esac
  jq -n --slurpfile rows "measured-v1/raw/held/trained/$slot.jsonl" \
    --slurpfile keys panels/held/keys.jsonl -f source/accounting/holdout-scores.jq > "$out/accounting/$arm-held.json"
  cmp "update-accounting-v1/$arm-held.json" "$out/accounting/$arm-held.json"
done
(cd grade-v1 && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$out/regraded"
for split in held train joint-held protect prior-dev scope-dev scope-held quartet; do
  dest="$out/regraded/$split"
  mkdir "$dest"
  (cd "grade-v1/$split" && sha256sum --strict --quiet -c SHA256SUMS)
  if [[ $split == held ]]; then
    controls=measured-v1/raw/held/frozen
    old_joint=/dev/null
  else
    controls="frozen/$split"
    old_joint="$controls/joint.jsonl"
  fi
  jq -nc --slurpfile panel "panels/$split/rows.jsonl" --slurpfile keys "panels/$split/keys.jsonl" \
    --slurpfile balanced "$controls/balanced.jsonl" --slurpfile linked "$controls/candidate.jsonl" \
    --slurpfile pointwise "measured-v1/raw/$split/trained/first-period.jsonl" \
    --slurpfile paired "measured-v1/raw/$split/trained/joint.jsonl" \
    --slurpfile old_joint "$old_joint" -f source/analysis/rows.jq > "$dest/rows.jsonl"
  jq -sc -f source/analysis/pairs.jq "$dest/rows.jsonl" > "$dest/pairs.jsonl"
  jq -n --arg split "$split" --slurpfile rows "$dest/rows.jsonl" --slurpfile pairs "$dest/pairs.jsonl" \
    -f source/analysis/summary.jq > "$dest/summary.json"
  for file in rows.jsonl pairs.jsonl summary.json; do
    cmp "grade-v1/$split/$file" "$dest/$file"
  done
done
bash source/product/pack.sh "$out/rebuilt" measured-v1/raw/fit "$base"
bash source/product/replay.sh "$out/rebuilt" panels/held/rows.jsonl measured-v1/raw/held/trained
bash source/product/ordinary.sh "$out/rebuilt" "$probe" tokenizer.json "$base"
for arm in rowwise paired; do
  cmp "product-v1/$arm/artifact.json" "$out/rebuilt/product-v1/$arm/artifact.json"
  for file in requests.jsonl client.jsonl cold.jsonl cold-receipt.json; do
    cmp "product-replay-v1/$arm/$file" "$out/rebuilt/product-replay-v1/$arm/$file"
  done
done
for file in base.jsonl rowwise.jsonl paired.jsonl summary.json; do
  cmp "product-ordinary-v1/$file" "$out/rebuilt/product-ordinary-v1/$file"
done
cd "$out"
{
  printf '%s  %s\n' "$expected_packet" "$packet"
  sha256sum schedule-check.json accounting/*.json regraded/*/summary.json rebuilt/product-v1/*/{artifact.d11,artifact.json} \
    rebuilt/product-replay-v1/*/{requests.jsonl,client.jsonl,cold.jsonl,cold-receipt.json} \
    rebuilt/product-ordinary-v1/{base.jsonl,rowwise.jsonl,paired.jsonl,summary.json}
} > readback.sha256
sha256sum --strict --quiet -c readback.sha256
