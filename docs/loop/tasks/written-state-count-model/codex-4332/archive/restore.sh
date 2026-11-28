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
printf '%s  %s\n' "$4" "$packet" | sha256sum --strict -c
mkdir "$restore"
mkdir "$restore/payload"
tar -I zstd -xf "$packet" -C "$restore/payload"
payload="$restore/payload"
(cd "$payload" && sha256sum --strict -c MEMBERS.sha256)
(cd "$runs" && sha256sum --strict -c "$payload/WICK-ARTIFACTS.sha256")

unit="$payload/checkout/docs/loop/tasks/written-state-count-model/codex-4332"
for instrument in panel probe gate; do
    cargo build --offline --locked --release --manifest-path "$unit/$instrument/Cargo.toml" \
        --target-dir "$restore/target"
done
"$restore/target/release/whole-turn-recurrence-panel" \
    "$runs" "$payload/spec-v2.json" "$restore/panel"
for stream in blind truth; do
    diff -qr "$payload/panel-v2/$stream" "$restore/panel/$stream"
done

balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
frozen="$runs/20261127-codex-whole-turn-fit/archive-readback-v1/reconstructed/frozen.d11"
"$restore/target/release/finite-native-recurrence" \
    "$balanced" "$frozen" "$payload/tools/flat-trunk" "$payload/tools/tokenizer.json" \
    "$restore/panel/blind/input.jsonl" "$restore/panel/blind/prompts.jsonl" \
    > "$restore/capture.jsonl"
cmp "$payload/gate-v1/capture.jsonl" "$restore/capture.jsonl"
"$restore/target/release/request-site-gate" "$restore/capture.jsonl" \
    "$restore/panel/truth/key.jsonl" > "$restore/verdict.json"
cmp "$payload/gate-v1/verdict.json" "$restore/verdict.json"
jq -c --slurpfile caps "$restore/capture.jsonl" --slurpfile keys "$restore/panel/truth/key.jsonl" \
    -f "$unit/gate/locate.jq" "$restore/verdict.json" > "$restore/sites.json"
cmp "$payload/gate-v1/sites.json" "$restore/sites.json"
(cd "$restore" && sha256sum --strict -c "$payload/gate-v1/SHA256SUMS")
jq -nc --slurpfile keys "$restore/panel/truth/key.jsonl" \
    -f "$unit/gate/alias.jq" "$restore/capture.jsonl" > "$restore/alias.json"
cmp "$payload/alias.json" "$restore/alias.json"

jq -en --slurpfile keys "$restore/panel/truth/key.jsonl" \
    --slurpfile grades "$payload/panel-v2/independent-adjudication/grades.jsonl" '
    ($keys | INDEX(.case_id)) as $key |
    ($grades | INDEX(.case_id)) as $review |
    ($key | keys) == ($review | keys) and
    ($keys | length) == ($grades | length) and
    ($key | length) == ($keys | length) and
    all($keys[]; .target.bytes == $review[.case_id].span)
'
