# RTX 4090 native occurrence fit

The Rust host validates the released #4312 D11 parent, extracts the scorer's full packed writer, and checks the canonical 4,096-byte token classification table. The CUDA image uses the [native feature contract](../product/FEATURES.md) to derive every source candidate and feature from causal prefix IDs. The candidate word addresses and score parameters are distinct. The suffix and feature logic are implemented on device for both fitting and observation. All five arms call the existing D11 writer.

Build the host with `cargo build --locked --release --manifest-path compute/Cargo.toml` on Linux with the CUDA driver library available. Run `compute/build-image.sh` on the leased RTX 4090 with its compiled scorer `OUT_DIR` and a new image directory. The script audits the three AOT entry points and rejects floating-point and tensor instructions in the generated PTX and SASS. The host loads the sm_89 cubin.

```bash
fit-native-occurrence extract answer-first/artifact.d11 base-full-v1
fit-native-occurrence fixture image-v1/occurrence.cubin base-full-v1 token-classes.u8 finite-v1

# Under the registered GO and lease:
fit-native-occurrence fit image-v1/occurrence.cubin base-full-v1 token-classes.u8 \
  donor-v2/colon-targets.jsonl panel-v1/train/rows.jsonl panel-v1/train-absent/rows.jsonl fit-v1

# Repeat these commands for each independently sealed held tape and output root.
fit-native-opening score opening.cubin base-full-v1 held.jsonl marker-weights-v1 marker-held-v1
fit-native-occurrence score image-v1/occurrence.cubin base-full-v1 token-classes.u8 \
  held.jsonl marker-held-v1/marker.jsonl fit-v1 occurrence-held-v1
```

Use the pinned #4314 marker TSV and cubin for the baseline. The base directory contains the same fixed sections consumed by that binary plus the full sparse trunk needed when a real space reaches the next lexical position. The scorer checks the unchanged marker gate and emitted ID on the immediate opening for every held row. Syntax failures, a candidate scope mismatch, an output from another baseline row, a writer disagreement or a changed reversed observation stop scoring.

The native fit uses 20 batches and two raw integer counter lanes. Each batch receives one contiguous part of the 118,915-row donor tape, ten complete positive contrast groups and one matched absent prompt per group. The group ordering is ascending group ID, then assignment, then question side. The complete batch sees its previous published state. A positive contrast adds six units to the gold occurrence, subtracts four at its rival and subtracts two at NoRead. It also adds one to gold and subtracts one at the strongest eligible alternative besides gold and the named rival, when available. On the matched absent prompts, the strongest eligible candidate loses eight units and NoRead gains eight. Exact hard-negative ties choose the smaller position. A causal donor opening contributes one unit to an admissible, uniquely matched factual target occurrence, and subtracts one at NoRead. Ordinary colon, unsupported prompt envelopes, nonlexical openings, absent targets and ambiguous occurrences receive separate census counts.

After each batch, each signed i16 word is `clamp(contrast + clamp(donor, -128, 128), -4096, 4096)`. Fit ends with 513 raw little-endian words in `weights.i16le`. `fit.json` preserves the two raw count lanes, training census, fixed update bounds and every published snapshot digest. The held scorer opens the saved state afresh for the reverse pass and checks exact equality of candidates, scores, source choices and writer outcomes. `observations.jsonl` records every factual position, its exact ID, packed root code, eligibility, score, competition rank and active features, along with each arm's source address, gate ID, actual immediate emission and first lexical continuation where a standalone space was emitted.
