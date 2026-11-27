#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then
  echo 'usage: archive.sh STUDY_ROOT STAGED_UNIT_SOURCE TOKENIZER.json' >&2
  exit 2
fi
root=$(realpath "$1")
source=$(realpath "$2")
tokenizer=$(realpath "$3")
runs=$(dirname "$root")
joint="$runs/20261127-codex-joint-release"
scope="$runs/20261127-codex-scope-trace"
prior="$runs/20261127-codex-period-margin"
factorial="$runs/20261127-codex-release-factorial"
components="$runs/20261127-codex-release-components"
measured="$root/measured-v1"
archive="$root/product-archive-v1"
payload="$archive/payload"
check_sha() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }

(cd "$measured" && sha256sum --strict --quiet -c OUTPUTS.sha256)
(cd "$root/panel-v1/held" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$root/product-ordinary-v1" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$root/grade-v1" && sha256sum --strict --quiet -c SHA256SUMS)
fit_sha=$(sha256sum < "$measured/raw/fit/fit.json" | cut -d' ' -f1)
for arm in rowwise paired; do
  (cd "$root/product-v1/$arm" && sha256sum --strict --quiet -c SHA256SUMS)
  (cd "$root/product-replay-v1/$arm" && sha256sum --strict --quiet -c SHA256SUMS)
  cmp "$measured/raw/fit/$arm.i16le" "$measured/raw/repeated/$arm.i16le"
  read -r state_sha _ < <(sha256sum "$measured/raw/fit/$arm.i16le")
  jq -e --arg fit "$fit_sha" --arg state "$state_sha" '
    .source_choice.fit_sha256 == $fit and .source_choice.state_sha256 == $state
  ' "$root/product-v1/$arm/artifact.json" >/dev/null
done
mkdir "$archive"
mkdir "$payload" "$payload/panels" "$payload/product-v1" "$payload/frozen"
cp -a "$root/"{measured-v1,product-replay-v1,product-ordinary-v1,grade-v1,update-accounting-v1} "$payload/"
cp -a "$source" "$payload/source"
cp "$tokenizer" "$payload/tokenizer.json"
cp "$root/staging/schedule-check.json" "$payload/"
cp "$joint/measured-v1/raw/fit/schedule.jsonl" "$payload/prior-schedule.jsonl"
cp -a "$root/panel-v1/held" "$payload/panels/held"
check_sha "$(cat "$measured/held.digest")" "$payload/panels/held/rows.jsonl"
for split in train joint-held quartet scope-dev scope-held prior-dev protect; do
  case "$split" in
    train) dir="$joint/panel-v1/train" ;;
    joint-held) dir="$joint/panel-v1/held" ;;
    quartet) dir="$factorial/panel-v1/fresh" ;;
    scope-*) dir="$scope/panel-v1/${split#scope-}" ;;
    prior-dev) dir="$prior/panel-v1/dev" ;;
    protect) dir="$prior/panel-v1/protect" ;;
  esac
  mkdir "$payload/panels/$split"
  cp "$dir/"{rows,keys}.jsonl "$payload/panels/$split/"
  panel_sha=$(awk -v path="$split.jsonl" '$2 == path {print $1}' "$measured/static.sha256")
  check_sha "$panel_sha" "$payload/panels/$split/rows.jsonl"
  (cd "$root/grade-v1/$split" && sha256sum --strict --quiet -c SHA256SUMS && sha256sum --strict --quiet -c INPUTS.sha256)
  mkdir "$payload/frozen/$split"
  mapfile -t traces < <(awk 'NR == 3 || NR == 4 || NR == 7 {print $2}' "$root/grade-v1/$split/INPUTS.sha256")
  [[ ${#traces[@]} == 3 ]]
  for index in 0 1 2; do
    cp "${traces[$index]}" "$payload/frozen/$split/"
  done
done
check_sha "$(awk '$2 == "train-keys.jsonl" {print $1}' "$measured/static.sha256")" "$payload/panels/train/keys.jsonl"
cp "$components/components-source-v1/base-full-v1/roots" "$payload/frozen-roots.u32le"
check_sha "$(awk '$2 == "base-full-v1/roots" {print $1}' "$measured/static.sha256")" "$payload/frozen-roots.u32le"
for arm in rowwise paired; do
  mkdir "$payload/product-v1/$arm"
  cp "$root/product-v1/$arm/"{artifact.json,INPUTS.sha256,SHA256SUMS} "$payload/product-v1/$arm/"
done
(
  cd "$payload"
  manifest=$(find . -type f -print0 | sort -z | xargs -0 sha256sum)
  printf '%s\n' "$manifest" > PAYLOAD.sha256
  sha256sum --strict --quiet -c PAYLOAD.sha256
)
tar -C "$payload" --zstd -cf "$archive/paired-release.tar.zst" .
(
  cd "$archive"
  sha256sum paired-release.tar.zst > SHA256SUMS
)
