# Restore the matched reciprocal transfer experiment

The #4341 packet preserves the fixed validation complement, both source reviews, all qualified and excluded pairs, the label-blind device inputs, the recovered GPU states and scores, and raw ordinary-client captures and grading records. It also retains the early extraction and ordering diagnostics and the first GPU launch failure. Original training tapes, the official materials, protected sources, tokenizer and balanced writer come from their earlier revision-pinned packets. The twelve complete D11 artifacts can be reproduced byte for byte from the recovered signed state and one balanced artifact.

The original #4340 packet describes the earlier insufficient instrument. #4341 is the expanded development study. Scores in the #4341 trial are for development and the four protected originals. The seven and twenty locked article partitions stay in authenticated upstream custody.

The completed packet contains 46,911,015 compressed bytes and 964 verified members. Wick force-downloaded the private revision, verified every member, reproduced both reports and all three derived stage manifests byte for byte, and rejected a same-length damaged packet before extraction. Reassembling the natural-geometry actual artifact from this restore reproduced its registered SHA-256 `6b7cb196d036b0ec223390ff10dbbaecc46c120468f863f9092818125a689443`. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/54f4c3c8fe9546bf1b73f0b188cca7693ebd51d3/results/codex/natural-reciprocal-repair-20261127/matched-transfer-result.tar.zst) · [Archive receipt](evidence/archive.json)

## Download and restore

Use a fresh private Wick directory, GNU coreutils, GNU tar with zstd, jq, and the authenticated `hf` command. Set `UNIT` to the directory containing this document and `WORK` to an unused absolute path. The published result receipt is `evidence/archive.json`. The archive helper can also restore a local repair checkpoint whose `packet.json` has `kind: repair`. The upload helper accepts a completed result packet.

```bash
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
umask 077
mkdir "$WORK"

download_restore() {
    local receipt=$1 restore=$2 dest=$3 repository revision path
    repository=$(jq -er .repository "$receipt") || return
    revision=$(jq -er .revision "$receipt") || return
    path=$(jq -er .path "$receipt") || return
    hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
        jq -e --arg revision "$revision" '.private == true and .sha == $revision' || return
    hf download "$repository" "$path" --repo-type dataset --revision "$revision" \
        --force-download --local-dir "$WORK/download" || return
    bash "$restore" "$WORK/download/$path" "$receipt" "$dest"
}

restore_parent="$UNIT/../codex-4338/archive/restore.sh"
download_restore "$UNIT/evidence/archive.json" "$UNIT/archive/restore.sh" "$WORK/result" &&
    download_restore "$UNIT/../codex-4340/evidence/archive.json" "$restore_parent" "$WORK/prior" &&
    download_restore "$UNIT/../codex-4338/evidence/archive.json" "$restore_parent" "$WORK/supply" &&
    download_restore "$UNIT/../codex-4337/evidence/archive.json" "$restore_parent" "$WORK/protected"
```

Restore the balanced D11 by following the [first-period reconstruction](../codex-4322/RESTORE.md), then set `BASE` to its verified file. The expected SHA-256 is `5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa`. The #4337 [original protected and packer receipt](../codex-4337/evidence/archive.json) provides the packer at `archive-context-v1/collection-bin/pack-native-interval`. The #4338 [natural supervision receipt](../codex-4338/evidence/archive.json) supplies the verified client, native tokenizer binder, matched TRAIN recording and official source files. The #4340 [prior transfer receipt](../codex-4340/evidence/archive.json) supplies both old judgments, excluded pairs and the exact 264-row TRAIN capsule.

## Reproduce the instrument and grades

The source replay runs the frozen selector, official-byte extraction, two-ledger join and label-blind ordinary feature capture. It compares each original disposition and every regenerated relative output with the archive, then verifies the scorer roster, the four device score tapes and the writer stage byte for byte. The commands use the source embedded in the packet.

```bash
bash "$UNIT/archive/replay-repair.sh" "$WORK/result" "$WORK/prior" "$WORK/supply" "$BASE" "$WORK/replay"
tool="$WORK/replay/cargo-target/debug/validation-transfer-score"
bash "$UNIT/archive/regrade.sh" "$WORK/result" "$tool" "$WORK/regrade"
```

The source grader reuses the recovered signed CUDA scores, frozen-artifact captures and separately replayed mate requests. The writer grader reuses the verified original-position commits, first-token and EOS traces, and visibility projection for the recorded disposition. The resulting report bytes must match the sealed reports. These are cold rechecks of the recorded experiment.

Use at least 16 GB free for full reconstruction of the twelve artifacts. Rehydration reads the preserved eight actual/permuted signed states and the prior capsule's zero payload. It checks each new artifact's SHA-256 and exact packing receipt against the original cold manifest.

```bash
packer="$WORK/protected/archive-context-v1/collection-bin/pack-native-interval"
bash "$UNIT/archive/rehydrate.sh" "$WORK/result" "$WORK/prior" "$BASE" "$packer" "$WORK/artifacts"
```

Append a registered model name such as `natural.geometry.actual` to reconstruct only that artifact.

## Build the private packet on Wick

Prepare a small exact source tree with `git archive` from the reviewed experiment head. Include `codex-4341`, `codex-4340/validation`, `codex-4340/fit`, `codex-4337/client`, `codex-4337/data/client-grade.jq`, `codex-4338/data/native.jq` and `codex-4338/archive/restore.sh`, under `docs/loop/tasks/written-state-count-model`. Verify the embedded scorer and writer source manifests. Set `REPAIR`, `CONTROLS`, `SOURCE_TREE`, `SCORER_RUN` and `WRITER_BUILD` to their authenticated Wick roots. Use an unused private `STAGE`. The child directories supplying the roster and writer stage are `roster-v2`, `eval-v2` and `stage-v2`.

```bash
bash "$UNIT/archive/stage-repair.sh" "$REPAIR" "$CONTROLS" "$SOURCE_TREE" "$SCORER_RUN" "$WRITER_BUILD" "$STAGE"
bash "$UNIT/archive/stage-result.sh" "$STAGE" "$FINAL_RUN"
bash "$UNIT/archive/pack.sh" result "$STAGE" "$PACKET"
bash "$UNIT/archive/publish.sh" "$PACKET" "$VERIFICATION"
```

`FINAL_RUN` is the top-session terminal receipt. Its schema is `uor-r4-natural-reciprocal-repair-final-run/1`, its `trial` is `finished`, and it carries the final `source_status`, `writer_status`, `locked_evaluation` and, for a graded writer, the `writer_projection` directory name. The `roots` and `pins` bind the following measured directories and their independently checked manifests. Include the failed device return and any superseded grading receipts in the `diagnostics` array as `{name,path}` entries. Include `manifest` and `sha256` for each sealed supplemental analysis to verify the nested manifest before staging.

| Role | Source tree | Manifest |
|---|---|---|
| `device` | Successful return's `job/measured` | `OUTPUTS.sha256` |
| `replay` | Separate return's `scored` directory | `OUTPUTS.sha256` |
| `cold` | Cold TRAIN verification directory | `OUTPUTS.sha256` |
| `source` | Cold validation and protected source evidence | `EVIDENCE.sha256` |
| `source_grade` | Preserved source decision | `SOURCE-GRADE.sha256` |
| `writer` | All ordinary captures, diagnostics, validation and visibility ledgers | `WRITER-MATRIX.sha256`, with per-arm capture, trace and visibility seals, projected `WRITER.sha256`, and the operations receipt |
| `writer_grade` | Preserved writer decision | `WRITER-GRADE.sha256` |
| `lease` | Sanitized lifecycle and off-pod receipt | `LEASE.sha256` |

The Wick packet keeps the original frozen-state, image, replay, cold artifact and trace manifests. The final packet receipt binds the exact compressed bytes and all member hashes to the verified private revision. Publication force-downloads the pinned revision, compares the result to the staged packet, and confirms a same-length corrupted packet fails before extraction. The returned `archive.json` is the repository receipt. Retain any failed checkpoint in its original private directory.
