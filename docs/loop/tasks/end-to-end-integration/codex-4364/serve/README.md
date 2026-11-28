# Physical writer and cold client readout

This executable evaluates four registered QuAC TRAIN conversations from two source pairs with the frozen, sham and active UORBFY03 checkpoints. The roster is pinned at `700c0bd3e78bd2dd8c0644d0abaea3e265538205eec20f69b9da0ccddca1ba92`. It uses the deployed [CUDA replay harness](../../codex-4360/runner/src/bin/device.rs) and [planted controls](../../codex-4363/src/finite.rs), with the original sm_89 serving cubin and separate training cubin identity. All real generation uses the complete causal prefix, cold state and the original 64×64 greedy election. It self-feeds actual IDs until natural EOS or a fixed cap of 96, above the longest gold response of 61 IDs including EOS.

The retained native Q15 state contains carry, shift history and token count. The shared integer Rust transition checks the physical replay step by step, including carry, shift history, token count, greedy ID and route, coarse score words, hidden words and PKM selections. The rollout and repeat use EOS placeholders regardless of references. Only a separate one-step diagnostic read addresses the gold fine cell and the actually elected fine cell. The physical election, hidden words, coarse scores, PKM and first-step state must agree with the blind launch. Rust recomputes both diagnostic fine score vectors. Rank uses the stable smallest eligible slot for ties. Strict gold wins require a positive gap. A pair's source comparison uses coarse scores when the reference cells differ and fine scores within the shared cell otherwise.

## Build and dispatch

Run `cargo build --locked --release` and `cargo test --locked --release` from this directory. This exact CUDA replay program requires `CUDA_DISABLE_PTX_JIT=1` and the authorized RTX 4090 environment. Verify the readout ELF against the run's startup pin before invoking it. Supply trusted receipt SHA-256 values from the sealed training and finite evidence.

```sh
grounded-opposed-opening-readout finite-host
grounded-opposed-opening-readout check-inputs TRAIN ROLES TOKENIZER ORIGINAL_CLI
grounded-opposed-opening-readout check-receipt TRAIN_REPORT TRAIN_REPORT_SHA TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA

CUDA_DISABLE_PTX_JIT=1 grounded-opposed-opening-readout gpu-calibration \
  SERVING_CUBIN HISTORICAL_2048_FIXTURE FROZEN

CUDA_DISABLE_PTX_JIT=1 grounded-opposed-opening-readout gpu \
  SERVING_CUBIN HISTORICAL_2048_FIXTURE TRAIN TRAIN_REPORT TRAIN_REPORT_SHA \
  TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA FROZEN SHAM ACTIVE NEW_GPU_OUTPUT
```

The real stage repeats calibration. It verifies the historical 2,048-ID prompt and 128 greedy IDs and Q48 fixture values, the batched historical rollout and four planted tie, one-word direction and mask controls. Real artifact checks bind the frozen step 3333, both step 3334 checkpoints and one update with 167 labels per arm to the original trainer receipt. They also check the same native configuration and permutation and byte-identical serialization. The raw output contains 12 private rows, per-checkpoint ranks, elected and referenced scores, paired source gaps, emitted IDs, first wrong offsets and exact through EOS. The summary records six pair results and authenticating hashes. First wrong offsets are zero-based. Early EOS is a mismatch against a content ID. A content ID when gold expects EOS marks late or missing EOS. `calibration.json`, `rows.jsonl` and `report.json` are retained in the fresh output directory.

## Cold ordinary CLI

After verified GPU results and checkpoint restore, use the original #4361 `uor-chat` on Wick.

```sh
./replay-cold.sh READOUT READOUT_SHA ORIGINAL_CLI INPUTS CHECKPOINTS \
  TRAIN_REPORT_SHA TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA GPU_ROWS GPU_ROWS_SHA NEW_COLD_OUTPUT
```

The original CLI resides at `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-reset/ordinary/uor-chat` on Wick. Its pinned SHA-256 is `15d5856ebf19200021700f77166a4caa641896c5f54ef0dbec279e0cc70a77e2`. `INPUTS` contains `base.uorbfy03`, `train.jsonl`, `role-and-source.jsonl` and `tokenizer.json`. `CHECKPOINTS` contains `report.json`, `sham.uorbfy03` and `active.uorbfy03`. The wrapper invokes the readout's `cold` command with the same pins:

```sh
grounded-opposed-opening-readout cold \
  ORIGINAL_CLI TOKENIZER ROLES TRAIN TRAIN_REPORT TRAIN_REPORT_SHA \
  TRAINING_CUBIN_SHA FINITE_RECEIPT_SHA FROZEN SHAM ACTIVE GPU_ROWS GPU_ROWS_SHA NEW_COLD_OUTPUT
```

Each cold process reopens the pinned checkpoint and tokenizer, rebuilds the literal protocol-2 prefix from the five authenticated role messages, and runs ordinary greedy serving. All 12 comparisons require exact tape prefix IDs and prefix digests, emitted IDs, stop conditions and stdout bytes. Only the five causal role messages reach the CLI. The original target text and source support fields stay in the private source record. Keep the per-row outputs private.
