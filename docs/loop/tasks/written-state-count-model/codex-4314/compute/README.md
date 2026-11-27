# Restore the fitted model and reproduce the scores

The [private experiment archive](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/caca4c57636d80d15793cb5b56389d3211ee5949/results/codex/marker-question-fit-20261127) and Wick ZFS hold the same two compressed files. Check both against the [archive SHA-256 manifest](../evidence/storage.sha256). The complete common parent and tokenizer are stored in the [earlier pinned revision](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/36bb47220ba0d60e07f0a6cb177da16b10902b84/results/codex/visible-answer-20261127). The parent file is `checkpoints/answer-first.tar.zst`. The tokenizer is the `tokenizer.json` member of `input/paired-inputs.tar.zst`. The earlier [store manifest](../../codex-4312/evidence/store.sha256) identifies both source archives.

Within `training-and-evidence.tar.zst`, open `analysis-v1/controlled-matrix.csv` for one row per group with both answers under both assignments and both questions in both models. `analysis-v1/openings.csv` has one row per question. `analysis-v1/natural-openings.csv` gives the human-passage observations. `analysis-v1/summary.json` contains full subgroup, address and score statistics. `measurement-v1/held` and `measurement-v1/weights` contain the raw CUDA scores and sparse integer states. `donor-v2` and `panel-v2` hold the corrected UORT tape, audits and exact fitted and held inputs. `product-replay-v3.tar.zst` holds the ordinary-product readbacks and finite comparisons from the final consumer. Open `delivery-loader-v3/fresh-question-swaps.html` inside it to browse all four prompts in any of the 200 groups. The score viewer works offline. Both expanded fitted artifacts are also on Wick under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-marker-question/product-validation-v3`. The private package stores the learned state once per arm and reuses the pinned parent for recovery.

From the repository root build the ordinary serving binaries. Set `bundle` to the directory holding the extracted experiment and `parent` to the extracted original `answer-first/artifact.d11`. Choose an unused output directory. The fit digest binds the approved inputs from both training arms.

```sh
cargo build --release --locked -p uor-r4-integer --features p1-count-serving \
  --bin pack-native-opening --bin verify-native-opening --bin uor-chat
fit=1e93e4c64cb4bdd49ef6c35c1ec08f9f17fa105b979bb9b9a7fabf5a9398f728
mkdir "$bundle/restored"
for arm in colon marker
do
  target/release/pack-native-opening "$parent" "$arm" \
    "$bundle/measurement-v1/weights/$arm.tsv" "$bundle/restored/$arm.d11" "$fit"
  target/release/verify-native-opening "$bundle/restored/$arm.d11" \
    "$bundle/panel-v2/held/rows.jsonl" > "$bundle/restored/$arm.jsonl"
done
sha256sum "$bundle/restored/"*.d11
```

Expected SHA-256 is `180e291b71e5de6badc3e1564f27193bb69a21cf95f940fb9b05978e39ad7531` for colon and `d782af466c2ebdfba1304158a3f995b50937c434de6ee5a505f8c49cc944f143` for marker. Wick independently reconstructed both byte strings using the final regular-file and stream validation. The [fitted product receipt](../product/FITTED.json) pins ordinary CLI, bulk replay, row projections and finite controls.

For raw-score reproduction use the Rust panel analyzer described in the [panel notes](../panel/README.md) and the two device JSONL files. For actual GPU refitting use the [frozen measurement source](https://github.com/UOR-Foundation/uor-r4/tree/85d2f4e9fee20f97ab555e8bde231821ee76e6e5/docs/loop/tasks/written-state-count-model/codex-4314), the registered `sm_89` image and the immutable input digests. The [original dispatch](../dispatch.sh) records the exact `fit`, `score` and after-space commands. A new GPU run needs its own lease, pre-run record, output root and hardware binding. The product's subsequent input validation produced the same artifacts and deployed predictions. The GPU qualification used no floating arithmetic or tensor instructions in the opening image. [PTX, cubin and SASS digests](../evidence/device.sha256)
