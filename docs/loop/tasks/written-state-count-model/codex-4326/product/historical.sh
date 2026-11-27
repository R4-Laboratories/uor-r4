#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 2 ]]; then
  echo 'usage: historical.sh MEASURED_DIR PRODUCT_INPUTS_DIR' >&2
  exit 2
fi
measured=$(realpath "$1")
inputs=$(realpath "$2")
(cd "$measured" && sha256sum --strict --quiet -c OUTPUTS.sha256)
(cd "$inputs" && sha256sum --strict --quiet -c SHA256SUMS)

printf 'split\tstate\tdirection\tmatched_rows\tsha256\n'
for split in held train quartet scope-dev scope-held prior-dev protect; do
  rows=$(wc -l < "$inputs/panels/$split.jsonl")
  for state in zero full; do
    case "$state" in
      zero) stem=zero-period/first-period ;;
      full) stem=post-full/joint ;;
    esac
    for direction in forward reverse; do
      suffix=
      if [[ $direction == reverse ]]; then suffix=-reverse; fi
      current="$measured/raw/$split/$stem$suffix.jsonl"
      historical="$inputs/historical/$split/$state$suffix.jsonl"
      cmp "$historical" "$current" >&2
      [[ $(wc -l < "$current") == "$rows" ]]
      sha=$(sha256sum < "$current" | cut -d' ' -f1)
      printf '%s\t%s\t%s\t%s\t%s\n' "$split" "$state" "$direction" "$rows" "$sha"
    done
  done
done
