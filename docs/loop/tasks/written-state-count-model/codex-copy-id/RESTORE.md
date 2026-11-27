# Reproduce the two-step comparison

The private Hub packet is 2,698,447 bytes. Download the [pinned packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/3ca768cb6cf3094399f4c059ab016a6dbbe12459/results/codex/copy-two-step-20261127/copy-two-step.tar.zst), verify its SHA-256 against the [readback receipt](evidence/storage-replay.json), and unpack with `tar --zstd -xf copy-two-step.tar.zst` into an empty directory. The restored panel and observations independently reproduced the original 1,416 cold replay rows byte for byte on Wick. The model has its own restore path below.

The product uses the unmodified #4315 artifact SHA-256 `591ea4a21e39942ac86625a8903ff90e2e6055b382966e740787211edcfe73c8`. Restore it and the pinned tokenizer using the [previous model restore recipe](../codex-4315/RESTORE.md). Bind the tokenizer and learned state using [frozen digests](evidence/frozen.sha256). The one-shot action is an explicit ordinary client option. Its implementation adds zero learned words. The [client instructions](product/README.md) describe the raw-ID, text and verification paths.

Wick ZFS keeps the private research packet at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-copy-id`. The `qualified-v1` directory holds the full aligned questions, token IDs and answer keys. The `measurement-v1/measured` directory holds every raw arm and the independently reloaded reverse-order pass. In `gpu-finite-v1` the previously exposed calibration rows, accepted cubin, integer-only instruction scan, and cold native comparison are separately recoverable. The unsuccessful build with compiler-generated reciprocal instructions is preserved in `aot-rejected-v1`. The registered runner checked the program, every model input, compiled image, panel and FIT before measuring the frozen two steps. [GO inputs](evidence/approved.sha256)

To verify the actual writer and self-fed transition, build the Rust tools with the feature set in the [product notes](product/README.md). Invoke `verify-native-copy ARTIFACT qualified-v1/qualified.jsonl measurement-v1/measured/observations.jsonl`. For the matched answer and source counts, use the panel package's `report` subcommand and the [panel instructions](panel/README.md). An independent static check follows.

```sh
jq -n --slurpfile p qualified-v1/qualified.jsonl \
  --slurpfile t measurement-v1/measured/observations.jsonl \
  -f "$REPO/docs/loop/tasks/written-state-count-model/codex-copy-id/audit-transitions.jq"
```

For a new GPU run, use the [CUDA host and kernel recipe](device/README.md) and a new output directory. The original `dispatch.sh` uses the immutable input manifest, original hardware UUID and original GO; a new allocation needs its own hardware binding. Real successor fields cover exactly one ordinary transition following the committed opening, including a real standalone space. An EOS commit has no continuation.
