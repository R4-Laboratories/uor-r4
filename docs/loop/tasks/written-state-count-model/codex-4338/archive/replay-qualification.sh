#!/usr/bin/env bash
set -euo pipefail
umask 077
restore=$(realpath "${1:?authenticated extracted packet}")
replay=$(realpath -m "${2:?fresh replay directory}")
source=$restore/source/docs/loop/tasks/written-state-count-model/codex-4338
data=$restore/unit
parent=$restore/fixtures/20261127-codex-grounded-interval
bash "$source/data/verify-qualification-ledgers.sh" "$data"
bash "$source/data/qualify.sh" "$data" "$parent" "$replay"
(cd "$replay" && sha256sum --strict -c "$data/qualified-v1/SEALED.sha256")
cmp "$data/qualified-v1/SEALED.sha256" "$replay/SEALED.sha256"
