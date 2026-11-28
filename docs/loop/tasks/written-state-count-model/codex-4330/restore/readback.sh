#!/usr/bin/env bash
set -euo pipefail
umask 077

packet=${1:?absolute path to the downloaded packet}
replay=${2:?absolute path to a fresh replay directory}
balanced=${3:?absolute path to the exact balanced checkpoint}
packet_sha=${4:?expected packet SHA-256}

printf '%s  %s\n' "$packet_sha" "$packet" | sha256sum --quiet --strict -c -
printf '%s  %s\n' \
    5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$balanced" \
    | sha256sum --quiet --strict -c -
mkdir "$replay"
tar --zstd -xf "$packet" -C "$replay"
cd "$replay"
sha256sum --quiet --strict -c SHA256SUMS
mkdir reconstructed

packer=tools/pack-whole-turn-occurrence
verifier=tools/verify-whole-turn-occurrence
"$packer" extract "$balanced" reconstructed/initial.i16le > reconstructed/initial.sha256
cmp reconstructed/initial.i16le product-v1/initial.i16le
cmp reconstructed/initial.sha256 product-v1/initial.sha256

for arm in frozen fitted; do
    if [[ "$arm" == frozen ]]; then
        words=product-v1/initial.i16le
    else
        words=measured-v1/fit1/weights.i16le
    fi
    receipt="product-v1/$arm.receipt.json"
    provenance=$(jq -er '.occurrence.fit_sha256' "$receipt")
    "$packer" pack "$balanced" "$words" "reconstructed/$arm.d11" "$provenance" \
        > "reconstructed/$arm.receipt.json"
    cmp "$receipt" "reconstructed/$arm.receipt.json"
    expected=$(jq -er .artifact_sha256 "$receipt")
    printf '%s  %s\n' "$expected" "reconstructed/$arm.d11" | sha256sum --quiet --strict -c -
done

"$verifier" "$balanced" reconstructed/frozen.d11 reconstructed/fitted.d11 \
    tools/tokenizer.json panel-v1/prompts.jsonl > reconstructed/cold-candidates.jsonl
bash analysis-program-v1/check-parity.sh measured-v1/scored/decisions.jsonl \
    reconstructed/cold-candidates.jsonl reconstructed/parity
cmp analysis-v1/parity/parity.json reconstructed/parity/parity.json
jq -nc --slurpfile gold panel-v1/gold.jsonl -f analysis-program-v1/source-grade.jq \
    measured-v1/scored/decisions.jsonl > reconstructed/source.json
cmp analysis-v1/source.json reconstructed/source.json

(
    cd reconstructed
    sha256sum initial.i16le initial.sha256 frozen.d11 fitted.d11 frozen.receipt.json \
        fitted.receipt.json cold-candidates.jsonl parity/parity.json source.json > SHA256SUMS
    sha256sum --quiet --strict -c SHA256SUMS
)
