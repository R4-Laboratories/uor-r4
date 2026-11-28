#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 ]]; then
    echo 'usage: restore-intake.sh VERIFIED_4361_PACKET VERIFIED_4363_PACKET FRESH_INTAKE' >&2
    exit 2
fi
parent=$(cd "$1" && pwd)
packet=$(cd "$2" && pwd)
unit=$(cd "$(dirname "$0")" && pwd)
mkdir -m 700 "$3"
cd "$3"
mkdir inputs bin
for name in base.uorbfy03 historical.json train.jsonl; do
    ln -s "$parent/inputs/$name" "inputs/$name"
done
for name in reset.uorbfy03 full.uorbfy03; do
    ln -s "$parent/received/attempt02/checkpoints/$name" "inputs/$name"
done
ln -s "$parent/bin/probe-final.sm89.cubin" inputs/probe-final.sm89.cubin
ln -s "$packet/bin/margin" bin/margin
ln -s "$packet/capture.sh" capture.sh
cp "$unit/evidence/intake.sha256" intake.sha256
cmp "$packet/intake.sha256" intake.sha256
sha256sum --strict -c intake.sha256
