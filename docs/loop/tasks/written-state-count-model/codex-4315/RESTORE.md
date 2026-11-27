# Restore the native model and its measured evidence

Wick ZFS holds the full model at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-shared-source/product-validation-v1/fitted/native-occurrence-product.tar.zst`. Its [receipt](product/FITTED.json) gives the artifact and archive byte lengths and hashes. Its `PAYLOAD.sha256` binds the complete checkpoint, tokenizer, fitted state and decision. Extract into a new empty directory using `tar -I zstd -xf` and check `sha256sum -c PAYLOAD.sha256`.

The private Hub stores a compressed 2.69 MB delta and research packet. This permits exact reconstruction while avoiding another full copy of the parent. Download `results/codex/shared-source-opening-20261127/measurement-and-native-delta.tar.zst` from revision `79b1778c18571e9abcf2192c11cf581a5429fa75` and check its [SHA-256](evidence/storage.sha256). Its `text-report-v2/controlled-openings.csv` has the 800 controlled question outcomes, decoded original words and emitted tokens. `human-openings.csv` records the 29 human references at their actual token offsets. `analysis-v1/opening-scores.csv` covers all 1,461 inputs. `measurement-v1/held/observations.jsonl` contains every admitted candidate and each of the five writer interventions. `finite-v1` includes the compiled AOT image, planted native weights and controls. [Pinned private archive](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/79b1778c18571e9abcf2192c11cf581a5429fa75/results/codex/shared-source-opening-20261127)

Recover the unchanged #4314 marker artifact `d782af466c2ebdfba1304158a3f995b50937c434de6ee5a505f8c49cc944f143` using the [parent restore instructions](../codex-4314/compute/README.md). Its own frozen answer-first base, tokenizer and corrected causal tape are pinned there. Build the two product binaries from this repository, and use the unpacked delta as the working directory.

```sh
# REPO is the absolute repository path. Run from the unpacked delta directory.
cargo build --manifest-path "$REPO/Cargo.toml" --release --locked -p uor-r4-integer \
  --features p1-count-serving --bin pack-native-occurrence --bin verify-native-occurrence
# PACK and VERIFY are the absolute paths to those release binaries.
FIT_SHA=$(sha256sum measurement-v1/fit/fit.json | cut -d' ' -f1)
"$PACK" pack "$MARKER" measurement-v1/fit/weights.i16le product-inputs-v1/token-classes.u8 artifact.d11 "$FIT_SHA"
sha256sum artifact.d11
"$VERIFY" artifact.d11 qualified-panel-v1/rows.jsonl measurement-v1/held/observations.jsonl > cold.jsonl
```

The downloaded delta reconstructed SHA-256 `591ea4a21e39942ac86625a8903ff90e2e6055b382966e740787211edcfe73c8` and matched the independently archived Wick artifact byte for byte. The [readback receipt](evidence/storage-readback.sha256) pins that reconstruction. The old checkpoint is also a complete language artifact in its own right. `product-validation-v1/fitted/payload/PAYLOAD.sha256` in the delta includes the digest of the reconstructed full model. The complete file itself is in the Wick archive.

Use the [GPU build and fit instructions](compute/README.md) for the trained reader. Its native CUDA qualification used source `8097abc4b0d7cc573aca819396f2bc303d05ac0e`, the [checked inputs](evidence/approved.sha256), [fit](evidence/fit.json) and [registered invocation](dispatch.sh). For static audits, run `audit-shared-source` on the saved panel, observations and weights, then `show-pairs` on the tokenizer, panel directory and observations. Extract the full packed original writer from the #4312 answer-first checkpoint with `fit-native-occurrence extract`, then run `writer-identity measurement-v1/held/observations.jsonl base-full-v1`. The [saved census](evidence/writer-identity.json) pins its four inputs. The tabulators need no training or inference.
