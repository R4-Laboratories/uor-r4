#!/bin/bash
set -euo pipefail
umask 077
cd /root/j4359/census
echo '0f78801e825f10aa21ce4dc10901dc81f1e6ae1e03630268a14bd37156e08c48  approved.sha256' | sha256sum --strict -c -
sha256sum --strict -c approved.sha256
(cd input && sha256sum --strict -c MEMBERS.sha256)
out=/workspace/attempt02
mkdir "$out" scratch
./capture prior-tape p4.d11 input protected frozen-sparse.jsonl prior-rows.jsonl snapped-proofs.jsonl trunk.bin "$out/tape" > "$out/capture.log"
for arm in baseline product unary; do
    ./signs native-tables.bin "$out/tape/roots.bin" "$out/tape/rows.bin" "$arm" "scratch/$arm.constraints" > "$out/$arm.masks"
    ./match native-tables.bin "$out/tape/roots.bin" "$out/tape/rows.bin" "$arm" "scratch/$arm.constraints" "$out/$arm.witnesses.tsv" > "$out/$arm.summary"
done
sha256sum scratch/*.constraints > "$out/constraint-hashes.txt"
cd "$out"
sha256sum tape/* *.log *.tsv *.masks *.summary constraint-hashes.txt > SHA256
touch done
