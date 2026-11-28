# Finite signed transport readback

Each captured site supplies a packed four-root word. The probe takes the canonical D11 action from each request root to root zero and applies it to the corresponding site root. The resulting block has two nonzero coordinates of magnitude two. Exact division by two gives four contiguous eight-coordinate blocks with eight signed unit contributions in total. A 32-word i16 table would use 64 deployed bytes.

The augmented capture preserves every #4332 field and original candidate. In `candidates[]`, `real_signed` uses the captured query. `fixed_signed` uses the fixed code zero. `swapped_signed` uses the opposing case's query and the current case's site. `no_question_signed` uses the captured passage-only recurrence where the blind tape records the separator. A null value preserves the absence of that control. An ablation with no added score leaves the frozen words and NoRead alone. The extractor consumes case pairs without source labels.

The startup check compares the public raw root and signed axis tables against the product implementation for all 1,024 action-root cells. It compares all 256 BFS cells with a frozen copy of the raw transition table, verifies each destination, and checks the 256 normalized query-site cells and all 16 query-to-anchor identities. From the raw axes it independently derives the relative signed permutation for each pair of request roots and checks it on each of the 16 site roots. There are 4,096 such comparisons. The permutation is the same for all sites in a row, so its inverse weight substitution keeps the signed integer box and every row-local score, including when frozen offsets are present. Pooling rows can force different substitutions even when each row is individually feasible.

## Reproduce on Wick

Start with the locked, checksum-verified [#4332 cold restore](../../codex-4332/RESTORE.md). Set `OLD` to its root, `RUNS` to the Wick runs directory and `PANEL` to the independently sealed fresh panel root. Set `UNIT` to this transport source and `OUT` to a new directory. Use the unmodified #4332 binary to capture model state. The transport build exposes three existing product functions in a disposable copy with a visibility-only patch.

```sh
set -euo pipefail
umask 077
mkdir "$OUT"
cp -a "$OLD/payload/checkout" "$OUT/checkout"
study=docs/loop/tasks/written-state-count-model/codex-4333
mkdir "$OUT/checkout/$study"
cp -a "$UNIT" "$OUT/checkout/$study/transport"
transport="$OUT/checkout/$study/transport"
(cd "$OUT/checkout" && patch -p1 < "$transport/operator-visibility.patch")
cargo build --offline --locked --release --manifest-path "$transport/Cargo.toml" \
  --target-dir "$OUT/target"
extract="$OUT/target/release/check-signed-site-transport"
"$extract" --check > "$OUT/operator-check.json"

jq -es '
  [.[] | {case_id, pair_id}] | group_by(.pair_id) |
  if all(.[]; length == 2) then
    map([{key: .[0].case_id, value: .[1].case_id},
         {key: .[1].case_id, value: .[0].case_id}]) | add | from_entries
  else error("invalid diagnostic pairs") end
' "$OLD/panel/truth/key.jsonl" > "$OUT/diagnostic-opposing.json"
"$extract" "$OLD/capture.jsonl" "$OUT/diagnostic-opposing.json" > "$OUT/diagnostic-signed.jsonl"

(cd "$RUNS" && sha256sum --strict -c "$OLD/payload/WICK-ARTIFACTS.sha256")
(cd "$PANEL/blind" && sha256sum --strict -c SEALED.sha256)
balanced="$RUNS/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
frozen="$RUNS/20261127-codex-whole-turn-fit/archive-readback-v1/reconstructed/frozen.d11"
"$OLD/target/release/finite-native-recurrence" \
  "$balanced" "$frozen" "$OLD/payload/tools/flat-trunk" "$OLD/payload/tools/tokenizer.json" \
  "$PANEL/blind/input.jsonl" "$PANEL/blind/prompts.jsonl" > "$OUT/fresh-capture.jsonl"
"$extract" "$OUT/fresh-capture.jsonl" "$PANEL/blind/opposing.json" > "$OUT/fresh-signed.jsonl"
```

The #4332 native probe authenticates the exact user opening and uses its independent recurrence oracle to verify the D11 product query and site words before recording them. The transport checker has no reader for span grades, learned weights or emitted prose. Keep raw rows and prompt material inside the private packet.

For the retained-field comparison, remove the four added fields from each augmented candidate with `jq -cS 'del(.candidates[] | .real_signed,.fixed_signed,.swapped_signed,.no_question_signed)'` and compare with the raw capture after `jq -cS .`. Repeating fresh capture and augmentation with both input streams reversed reproduced the original bytes when the resulting JSONL line order was restored.

The exact inverse control `m6_exact_arithmetic::tests::control_orbit_digit_and_cl06_inverse` uses the original D11 operators. In the restored source, the library test build failed on a pre-existing import in `stack/tests/pair_rows.rs`. The named control passed on Wick after replacing `use super::super as stack;` with `use crate::stack;` in a separate disposable test checkout.

## Post-hoc source ranking attribution

This adjunct uses the sealed fresh capture, key and primary solver output. It copies the primary per-case classifications and recomputes two disjoint pooled systems on the supported case IDs. One requires the labeled original site to beat each eligible source rival. The other requires each labeled site to beat NoRead. Both use the gate's original inequality construction and certificate verifier. Its scope is finite attribution after the primary gate.

Copy the sibling `solver` source into a new private directory and run the following from that copy. `Z3`, `SIGNED_GATE_HIGHS` and `SIGNED_GATE_SCRATCH` use the primary solver configuration. `FRESH_SIGNED`, `FRESH_KEY` and `PRIMARY_VERDICT` are the sealed fresh signed capture, case key and full primary JSONL verdict.

```sh
sed 's/^fn main()/#[allow(dead_code)]\nfn original_main()/' src/main.rs > src/main.attribution
cat "$UNIT/attribution.rs" >> src/main.attribution
mv src/main.attribution src/main.rs
cargo build --offline --locked --release
target/release/signed-site-gate "$Z3" "$FRESH_SIGNED" "$FRESH_KEY" "$PRIMARY_VERDICT" > attribution.jsonl
```
