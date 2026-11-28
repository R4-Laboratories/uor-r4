#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 4 ]; then
    echo "usage: restore.sh WICK_RUNS NEW_RESTORE PACKET EXPECTED_SHA256" >&2
    exit 2
fi
runs=$(realpath "$1")
restore=$(realpath -m "$2")
packet=$(realpath "$3")
digest=$4
test "${#digest}" -eq 64
printf '%s  %s\n' "$digest" "$packet" | sha256sum --strict -c
mkdir "$restore"
mkdir "$restore/payload"
tar -I zstd -xf "$packet" -C "$restore/payload"
payload="$restore/payload"
(cd "$payload" && sha256sum --strict -c MEMBERS.sha256)
(cd "$runs" && sha256sum --strict -c "$payload/PRIVATE-WICK.sha256")

study=docs/loop/tasks/written-state-count-model/codex-4329
source="$payload/checkout/$study"
target="$restore/target"
tokenizer="$payload/tokenizer/tokenizer.json"
squad="$runs/20261127-codex-whole-turn-census/supply-inputs"
if [ ! -f "$squad/squad-train-v2.0.json" ] || [ ! -f "$squad/squad-dev-v2.0.json" ]; then
    squad="$restore/official"
    mkdir "$squad"
    for split in train dev; do
        curl --fail --silent --show-error --location \
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/$split-v2.0.json" \
            -o "$squad/squad-$split-v2.0.json"
    done
fi

for instrument in supply contrasts census; do
    cargo build --offline --locked --release --manifest-path "$source/$instrument/Cargo.toml" \
        --target-dir "$target"
done
"$target/release/whole-turn-supervision" "$runs" "$squad" "$restore/natural"
for tape in census.json train-prompts.jsonl train-truth.jsonl reserve-prompts.jsonl \
    reserve-truth.jsonl dev-prompts.jsonl dev-truth.jsonl; do
    cmp "$payload/natural/$tape" "$restore/natural/$tape"
done
(cd "$restore/natural" && sha256sum --strict -c "$payload/natural/SEALED.sha256")

"$target/release/whole-turn-contrasts" "$tokenizer" "$restore/authored"
for split in train dev; do
    cmp "$payload/authored/$split.jsonl" "$restore/authored/$split.jsonl"
    jq -sS -f "$source/contrasts/check.jq" "$restore/authored/$split.jsonl" \
        > "$restore/authored/$split.json"
    cmp "$payload/authored/$split.json" "$restore/authored/$split.json"
done

"$target/release/audit-whole-turn" \
    "$runs/20261127-codex-marker-question/donor-v2" \
    "$runs/20261127-codex-visible-answer/donor/pair-01" \
    "$tokenizer" \
    "$runs/20261127-codex-shared-source/product-inputs-v1/token-classes.u8" \
    "$payload/checkout/docs/loop/tasks/written-state-count-model/codex-4315/evidence/donor-admission.json" \
    "$restore/census"
cmp "$payload/receipts/census.json" "$restore/census/summary.json"
cmp "$runs/20261127-codex-whole-turn-census/census-v2/rows.jsonl" "$restore/census/rows.jsonl"
(
    cd "$restore"
    sha256sum natural/census.json natural/*-prompts.jsonl natural/*-truth.jsonl \
        authored/train.jsonl authored/dev.jsonl authored/train.json authored/dev.json \
        census/rows.jsonl census/summary.json > READBACK.sha256
    sha256sum --strict -c READBACK.sha256
)
