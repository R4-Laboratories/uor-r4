# Restore the Wick fixed-assignment median measurement

The [publication receipt](storage/receipt.json) pins the private dataset, immutable revision, compressed object, member manifest and expected semantic replay. The physical originals remain in `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median` on Wick. The [layout](storage/layout-v1.json) identifies the sealed versions. The accepted TRAIN and fresh executions bind their methods through the [measurement seals](evidence/measured.sha256). [Private archive at the verified revision](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/ee6864137f0d4bcc5e2da2baf059068c90a13345/results/codex/fixed-assignment-median-20261126-cd2c44d79d3f)

On Wick with the existing authorized private-store client, use an unused download version. This reads the immutable archive, verifies each member in a new directory, re-elects the two states, uniquely rejoins the original addresses, and regrades the captured decisions and writer admission.

```bash
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
receipt="$study/storage/archive-v1/published-v1/receipt.json"
bash "$study/storage/methods-v2/preserve-wick.sh" download v2 "$receipt"
cmp "$study/storage/download-v2/replayed/receipt.json" \
    "$study/storage/download-v1/replayed/receipt.json"
```

The expected cold-output digest is `d18845aedca51f56639ff92258b63f700165741b46b6060dff9dc30b18d82aff`. Both forced downloads reproduced the original local cold replay byte for byte. The [cold receipt](evidence/restored.json) records 184 comparisons and both auditor results. For the two full post-run source summaries, the [attribution reducer](analysis/README.md) includes the exact compact projection recipe and [source hashes](evidence/native-state.sha256). The public rendering of the initial FIT corrects its relative link to the roadmap. The archive preserves its original issued bytes at `audit/FIT-v1/FIT.md` under SHA-256 `df2d25577766e47f50999e465bd9d94905f3ebf040a24f6dcbead89df49ee814`. The second FIT is the one bound to the measured release. [FIT addendum](audit/FIT-v2.md) · [Archive and restore tools](storage/README.md)

## Reproduce the post-review fallback checks

The measured run used `methods-v2`. When an election has no feasible candidate the corrected packer selects the original endpoint state and operator. The two measured selections qualified. On Wick the following commands use the separately sealed post-review methods and the original immutable extraction. Supply unused result directory versions.

```bash
bash "$study/methods-v3/packing-fixture-wick.sh" v2
extracted="$study/storage/download-v1/extracted"
patched="$study/storage/restore-fallback-v1"
bash "$patched/restore-grade-wick.sh" "$extracted" \
    "$extracted/storage/archive-v1/layout.json" "$patched/replayed-v2"
cmp "$patched/replayed-v2/receipt.json" "$study/storage/download-v1/replayed/receipt.json"
```

The [packing fixture](evidence/packing-fixture.json) records all four qualification combinations. The [post-review seals](evidence/fallback.sha256) also bind the rerun audits, source-attribution reports, protected TRAIN report, and corrected cold replay. The first supplemental TRAIN report manifest included its own temporary filename. Its successor sealed the identical verified report bytes and is the retained receipt.
