#!/usr/bin/env bash
set -euo pipefail
umask 077

packet=${1:?absolute path to the downloaded ordinary packet}
replay=${2:?absolute path to a fresh ordinary replay directory}
core=${3:?absolute path to the restored core packet}
balanced=${4:?absolute path to the exact balanced checkpoint}
packet_sha=${5:?expected ordinary packet SHA-256}

printf '%s  %s\n' "$packet_sha" "$packet" | sha256sum --quiet --strict -c -
printf '%s  %s\n' \
    5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$balanced" \
    | sha256sum --quiet --strict -c -
(cd "$core/reconstructed" && sha256sum --quiet --strict -c SHA256SUMS)
mkdir "$replay"
tar --zstd -xf "$packet" -C "$replay"
cd "$replay"
sha256sum --quiet --strict -c SHA256SUMS
mkdir representative

while IFS=$'\t' read -r arm case_id; do
    [[ "$case_id" =~ ^[A-Za-z0-9_-]+$ ]]
    case "$arm" in
        old) artifact=$balanced ;;
        frozen|fitted) artifact="$core/reconstructed/$arm.d11" ;;
        *) exit 2 ;;
    esac
    IFS= read -r -d '' text < <(
        jq -erj --arg id "$case_id" \
            'select(.case_id == $id) | .prompt, "\u0000"' "$core/panel-v1/prompts.jsonl"
    )
    "$core/tools/uor-chat" --discrete "$artifact" --tokenizer "$core/tools/tokenizer.json" \
        --protocol 2 --say "$text" --temperature 0 --max-tokens 16 \
        --authenticated-copy --native-occurrence-trace \
        | jq -ceS 'del(.artifact)' > "representative/$arm-$case_id.json"
done < ordinary-replay.tsv

sha256sum --quiet --strict -c representative.sha256
