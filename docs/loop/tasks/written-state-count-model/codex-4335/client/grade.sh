#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 4 ]; then
    echo "usage: grade.sh COLD_4333 COLD_4332 COLD_4330 CAPTURE_OUTPUT_PARENT" >&2
    exit 2
fi
signed=$(realpath "$1")
request=$(realpath "$2")
core=$(realpath "$3")
out=$(realpath "$4")
src=$(cd "$(dirname "$0")" && pwd)
study=docs/loop/tasks/written-state-count-model/codex-4335

sha256sum --strict --quiet -c <<SUMS
641e9ed7c21ed0d7f12da5601615bc0f38dc07bbec859bd706462d38cc5ed693  $signed/payload/MEMBERS.sha256
b5b192da4e77b12b78327c95176400107c8f361c3c66562b6521e425896f8121  $request/payload/MEMBERS.sha256
SUMS
(cd "$signed/payload" && awk '$2 ~ /^\.\/panel-v1\//' MEMBERS.sha256 | sha256sum --strict --quiet -c)
(cd "$request/payload" && awk '$2 ~ /^checkout\//' MEMBERS.sha256 | sha256sum --strict --quiet -c)
(cd "$out/client" && sha256sum --strict --quiet -c SHA256SUMS)

checkout="$out/checkout"
mkdir "$checkout"
cp -a "$request/payload/checkout/." "$checkout/"
mkdir -p "$checkout/$study"
cp -a "$src" "$checkout/$study/client"
cargo build --offline --locked --release --manifest-path "$checkout/$study/client/Cargo.toml" --target-dir "$out/target"
panel="$signed/payload/panel-v1"
"$out/target/release/extractive-client-grade" "$core/tools/tokenizer.json" \
    "$panel/blind/input.jsonl" "$panel/blind/prompts.jsonl" "$panel/truth/key.jsonl" "$out/client" > "$out/scores.jsonl"
jq -escS -f "$src/summary.jq" "$out/scores.jsonl" > "$out/summary.json"
(cd "$out" && sha256sum client/{balanced,frozen,contextual}.jsonl client/SHA256SUMS scores.jsonl summary.json > SHA256SUMS)
