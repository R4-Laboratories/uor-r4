#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 2 ]]; then
  echo 'usage: stage.sh NEW_PRODUCT_INPUTS_DIR WICK_RUNS_DIR' >&2
  exit 2
fi
out=$1
runs=$(realpath "$2")
joint="$runs/20261127-codex-joint-release"
factorial="$runs/20261127-codex-release-factorial"
scope="$runs/20261127-codex-scope-trace"
prior="$runs/20261127-codex-period-margin"

for dir in "$joint/measured-v1" "$factorial/measured-v1"; do
  (cd "$dir" && sha256sum --strict --quiet -c OUTPUTS.sha256)
done
mkdir "$out"
mkdir "$out/panels" "$out/historical" "$out/trajectory" "$out/previous-grade"
cp "$joint/measured-v1/raw/fit/fit.json" "$out/original-fit.json"
cp "$joint/product-v1/joint/artifact.json" "$out/original-artifact.json"
cp "$scope/fit-v1/fit-out/weights/weights.i16le" "$out/trajectory/linked.i16le"
for name in joint.i16le joint-slots.jsonl schedule.jsonl authenticated.jsonl; do
  cp "$joint/measured-v1/raw/fit/$name" "$out/trajectory/$name"
done
for split in held train quartet scope-dev scope-held prior-dev protect; do
  previous="$joint/grade-v1/$split"
  (cd "$previous" && sha256sum --strict --quiet -c SHA256SUMS)
  mkdir "$out/previous-grade/$split"
  cp "$previous/"{balanced,linked}.rows.jsonl "$out/previous-grade/$split/"
  case "$split" in
    held|train)
      panel="$joint/panel-v1/$split/rows.jsonl"
      zero="$joint/measured-v1/raw/$split/frozen/candidate"
      ;;
    quartet)
      panel="$factorial/panel-v1/fresh/rows.jsonl"
      zero="$factorial/measured-v1/raw/fresh/candidate"
      ;;
    scope-dev|scope-held)
      panel="$scope/panel-v1/${split#scope-}/rows.jsonl"
      zero="$factorial/measured-v1/raw/${split#scope-}/candidate"
      ;;
    prior-dev|protect)
      old=${split#prior-}
      panel="$prior/panel-v1/$old/rows.jsonl"
      zero="$factorial/measured-v1/raw/$split/candidate"
      ;;
  esac
  full="$joint/measured-v1/raw/$split/trained/joint"
  panel_sha=$(sed -n '2s/^\([0-9a-f]\{64\}\)  .*/\1/p' "$joint/product-replay-v1/joint/$split/INPUTS.sha256")
  printf '%s  %s\n' "$panel_sha" "$panel" | sha256sum --strict --quiet -c -
  cp "$panel" "$out/panels/$split.jsonl"
  case "$split" in
    held|train) keys="$joint/panel-v1/$split/keys.jsonl" ;;
    quartet) keys="$factorial/panel-v1/fresh/keys.jsonl" ;;
    scope-dev|scope-held) keys="$scope/panel-v1/${split#scope-}/keys.jsonl" ;;
    prior-dev|protect) keys="$prior/panel-v1/${split#prior-}/keys.jsonl" ;;
  esac
  if [[ -f $keys ]]; then cp "$keys" "$out/panels/$split.keys.jsonl"; fi
  mkdir "$out/historical/$split"
  for suffix in .jsonl -reverse.jsonl; do
    cp "$zero$suffix" "$out/historical/$split/zero$suffix"
    cp "$full$suffix" "$out/historical/$split/full$suffix"
  done
done
(
  cd "$out"
  manifest=$(find . -type f -print0 | sort -z | xargs -0 sha256sum)
  printf '%s\n' "$manifest" > SHA256SUMS
  sha256sum --strict --quiet -c SHA256SUMS
)
