#!/usr/bin/env bash
set -euo pipefail

analysis=$(cd -- "$(dirname -- "$0")" && pwd)
root=${PERIOD_ROOT:-/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-period-margin}
grade="$root/analysis-v1"
cold="$root/cold-v1"
for config in baseline balanced weak both; do
  for split in train dev protect; do
    (cd "$grade/$config/$split" && sha256sum --quiet --strict -c SHA256SUMS)
    (cd "$cold/$config/$split" && sha256sum --quiet --strict -c SHA256SUMS)
  done
  jq -en --slurpfile original "$grade/$config/protect/protected-original.json" \
    --slurpfile protect "$grade/$config/protect/grade.json" \
    '$original[0].original_exact_pairs==50 and
     $original[0].retained_exact_pairs==$protect[0].overall.exact' > /dev/null
done
for config in baseline balanced weak both; do
  jq -n --arg config "$config" --slurpfile train "$grade/$config/train/grade.json" \
    --slurpfile dev "$grade/$config/dev/grade.json" --slurpfile protect "$grade/$config/protect/grade.json" \
    '{($config):{train:$train[0],dev:$dev[0],protect:$protect[0]}}'
done | jq -s 'add' | jq -f "$analysis/select.jq"
