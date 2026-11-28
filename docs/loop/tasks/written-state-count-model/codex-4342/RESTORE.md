# Restore the directed relation screen

[codex] The small private packet on Wick ZFS preserves the entire registered TRAIN source inventory, the two review requests and separate returns, rejected groups and consensus, the ordinary client's first over-capacity refusal, admitted and refused cases, old and new captures, and the native structural grader. The operator evidence includes checks on the existing trained states and finite fabricated weights in the new arm formats. The previous checkpoint, tokenizer and two large fitted artifacts already have verified restoration procedures. [Packet identity](evidence/archive.json) · [Balanced artifact and prior states](../codex-4341/RESTORE.md)

Use a fresh private Wick directory, GNU coreutils, tar with zstd, jq and an authenticated Hugging Face CLI. Set `UNIT` to this directory, `WORK` to a new private scratch directory and `BASE` and `TOKENIZER` to the restored balanced D11 and its original tokenizer. The archive accepts the balanced artifact with SHA-256 `5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa` and the tokenizer with SHA-256 `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`.

```bash
set -eu
umask 077
mkdir "$WORK"
receipt="$UNIT/evidence/archive.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
path=$(jq -er .path "$receipt")
hf download "$repository" "$path" --repo-type dataset --revision "$revision" --force-download --local-dir "$WORK/download"
bash "$UNIT/../codex-4338/archive/restore.sh" "$WORK/download/$path" "$receipt" "$WORK/restored"
bash "$UNIT/archive/regrade.sh" "$WORK/restored" "$BASE" "$TOKENIZER" "$WORK/regraded"
```

For source-only reproduction, follow the separately pinned [inventory and qualification procedure](supply/REPLAY.md). For the previous ordinary client's raw feature and self-fed replay, use [the operator replay](operator/replay.sh) with the restored #4341 natural geometry and count states. The packet preserves the exact current Wick executables. Executable file hashes can depend on the compiler and build environment. The independent operator and grader compare the values, candidate addresses and emitted integer IDs at the model boundary.

The [initial receipt](evidence/archive-initial.json) retains the first v2 client and the same TRAIN screen before exact-head review found a clipped-word defect on an unseparated synthetic question. Restore it in a different directory to run the [before-and-after witness](operator/check-window.sh) with both client executables. The final packet binds the corrected verifier and client. Existing v1 fitted artifacts are unaffected by that correction.
