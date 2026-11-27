#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
src=$(realpath -e "${1:?source closure}")
fixture=$(realpath -e "${2:?finite fixture directory}")
build=$(realpath -e "${3:?physical build directory}")
out=${4:?unused output directory}
(cd "$src" && sha256sum --strict -c SHA256SUMS)
(cd "$fixture" && sha256sum --strict -c SHA256SUMS)
(cd "$build" && sha256sum --strict -c SHA256SUMS)
"$build/argument-score" "$fixture/fixture.bank" "$fixture/fixture.text" "$fixture/fixture.state" \
    "$fixture/masks.tsv" "$out" "$fixture/gold.aid" "$fixture/swapped.aid"
cmp "$out/baseline.jsonl" "$out/native-reference.jsonl"
cmp "$out/state.readback" "$fixture/fixture.state"
jq -ne \
    --slurpfile baseline "$out/baseline.jsonl" --slurpfile rb "$out/routes.baseline.jsonl" \
    --slurpfile g "$out/G.jsonl" --slurpfile rg "$out/routes.G.jsonl" \
    --slurpfile n "$out/N.jsonl" --slurpfile rn "$out/routes.N.jsonl" \
    --slurpfile gn "$out/GN.jsonl" --slurpfile rgn "$out/routes.GN.jsonl" \
    --slurpfile wrong_g "$out/wrong-G.jsonl" --slurpfile wrong_n "$out/wrong-N.jsonl" \
    --slurpfile gn_wrong_g "$out/GN-wrong-G.jsonl" --slurpfile gn_wrong_n "$out/GN-wrong-N.jsonl" \
    --slurpfile gn_wrong_both "$out/GN-wrong-both.jsonl" \
    --slurpfile gold "$out/gold.jsonl" --slurpfile swap "$out/swapped.jsonl" \
    -f "$src/finite.jq" | tee "$out/finite.json"
(cd "$out" && sha256sum *.json *.jsonl state.readback > SHA256SUMS)
