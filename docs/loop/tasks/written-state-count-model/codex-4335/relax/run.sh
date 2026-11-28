#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 5 ]] || { echo 'usage: run.sh COLD_4334 COLD_4333 COLD_4332 D30_RAW NEW_OUTPUT' >&2; exit 2; }
parent=$(realpath "$1")
signed=$(realpath "$2")
base=$(realpath "$3")
historic=$(realpath "$4")
out=$(realpath -m "$5")
src=$(cd "$(dirname "$0")" && pwd)
carrier="$historic/inputs/local/codex/pre-gate-source-26/serve-native-01/recurrence.d11"
control="$historic/ordinary/d26/30006.json"
sha256sum --strict --quiet -c <<SUMS
b17352679c64a96bf657fad7640d4bb05ea3de0ecd08dadd13e9d2756e979bc4  $carrier
4e71786896da1dea980995bc753a7a7d58fad7a1c4551dba1e1883caa716ab63  $control
SUMS
(cd "$parent/transport" && sha256sum --strict --quiet -c "$parent/payload/transport/SCAN.sha256")
mkdir "$out"
cp -a "$base/payload/checkout" "$out/checkout"
unit="$out/checkout/docs/loop/tasks/written-state-count-model/codex-4335"
mkdir -p "$unit"
cp -a "$src" "$unit/relax"
cat "$src/study.rs" >> "$out/checkout/crates/uor-r4-integer/src/contextual_source.rs"
cargo build --offline --locked --release --manifest-path "$unit/relax/Cargo.toml" --target-dir "$out/target"
probe="$out/target/release/delimiter-relaxation-study"
"$probe" "$carrier" control "$control" > "$out/control.json"
for panel in diagnostic fresh; do
    tape="$parent/transport/$panel-tail32.jsonl"
    "$probe" "$carrier" capture "$tape" > "$out/$panel.jsonl"
    "$probe" "$carrier" capture <(tac "$tape") > "$out/reverse-$panel.jsonl"
    cmp "$out/$panel.jsonl" <(tac "$out/reverse-$panel.jsonl")
    case "$panel" in
        diagnostic) truth="$base/panel/truth" ;;
        fresh) truth="$signed/panel/truth" ;;
    esac
    (cd "$truth" && sha256sum --strict --quiet -c SEALED.sha256)
    jq -es --slurpfile truth "$truth/key.jsonl" -f "$src/grade.jq" "$out/$panel.jsonl" > "$out/summary.$panel.json"
done
(cd "$out" && sha256sum {diagnostic,fresh}.jsonl summary.{diagnostic,fresh}.json control.json > RELAX.sha256)
