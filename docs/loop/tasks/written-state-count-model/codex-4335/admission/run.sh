#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 6 ]] || { echo 'usage: run.sh WICK_RUNS COLD_4334 COLD_4333 COLD_4332 D30_RAW NEW_OUTPUT' >&2; exit 2; }
runs=$(realpath "$1"); parent=$(realpath "$2"); older=$(realpath "$3"); base=$(realpath "$4")
historic=$(realpath "$5"); out=$(realpath -m "$6")
src=$(cd "$(dirname "$0")" && pwd)
balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
frozen="$runs/20261127-codex-whole-turn-fit/archive-readback-v1/reconstructed/frozen.d11"
carrier="$historic/inputs/local/codex/pre-gate-source-26/serve-native-01/recurrence.d11"
control="$historic/ordinary/d26/30006.json"
tokenizer="$base/payload/tools/tokenizer.json"
sha256sum --strict -c <<SUMS
5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa  $balanced
6415f6cb9fbf4694b449c08170d17c7e821c168d28e811cfa4e7a3448a333422  $frozen
b17352679c64a96bf657fad7640d4bb05ea3de0ecd08dadd13e9d2756e979bc4  $carrier
4e71786896da1dea980995bc753a7a7d58fad7a1c4551dba1e1883caa716ab63  $control
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
SUMS
(cd "$parent/transport" && sha256sum --strict --quiet -c "$parent/payload/transport/SCAN.sha256")
mkdir "$out"
cp -a "$base/payload/checkout" "$out/checkout"
unit="$out/checkout/docs/loop/tasks/written-state-count-model/codex-4335"
mkdir -p "$unit"
cp -a "$src" "$unit/admission"
cargo build --offline --locked --release --manifest-path "$unit/admission/Cargo.toml" --target-dir "$out/target"
probe=("$out/target/release/source-interval-admission" "$balanced" "$frozen" "$carrier" "$tokenizer")
"${probe[@]}" control "$control" > "$out/control.json"
for panel in fresh diagnostic; do
    "${probe[@]}" capture "$parent/transport/$panel-tail32.jsonl" > "$out/$panel.jsonl"
    case "$panel" in
         fresh) truth="$older/panel/truth" ;;
         diagnostic) truth="$base/panel/truth" ;;
    esac
    (cd "$truth" && sha256sum --strict --quiet -c SEALED.sha256)
    jq -es --slurpfile truth "$truth/key.jsonl" -f "$src/grade.jq" "$out/$panel.jsonl" > "$out/summary.$panel.json"
    "${probe[@]}" endpoints "$parent/transport/$panel-tail32.jsonl" "$truth/key.jsonl" > "$out/endpoints.$panel.jsonl"
done
(cd "$out" && sha256sum {fresh,diagnostic}.jsonl summary.{fresh,diagnostic}.json endpoints.{fresh,diagnostic}.jsonl control.json > ADMISSION.sha256)
