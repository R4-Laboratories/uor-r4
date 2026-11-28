# Restore the matched-transfer validation qualification

The private packet preserves the validation-only selection and exact source
rows, two 67-pair semantic judgment ledgers, pre-fit pins, ordinary-client
qualification files, the staged TRAIN capsule and the committed source used
to rebuild both outputs. The authenticated #4338 packet supplies the official
source files, original partitions, native client and tools, tokenizer and
matched TRAIN fixture. The balanced artifact is restored separately.
[Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/9bd98bef04d00fd88d01d1a14305bbc449f4046b/results/codex/natural-reciprocal-transfer-20261127/transfer.tar.zst) ·
[Packet receipt](evidence/archive.json) · [#4338 source custody](../codex-4338/RESTORE.md) ·
[Balanced artifact restore](../codex-4322/RESTORE.md)

On Wick, set `UNIT` to the absolute directory containing this file and
`WORK` to a fresh absolute private directory. Use the authenticated Hugging
Face client, GNU coreutils, GNU tar with zstd, jq 1.8.1 and the offline Cargo
cache.

```sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
umask 077
mkdir "$WORK"
BASE=/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
RESTORE="$UNIT/../codex-4338/archive/restore.sh"

download_restore() {
    local receipt=$1 dest=$2 repository revision path
    repository=$(jq -er .repository "$receipt") || return
    revision=$(jq -er .revision "$receipt") || return
    path=$(jq -er .path "$receipt") || return
    hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
        jq -e --arg revision "$revision" '.private == true and .sha == $revision' || return
    hf download "$repository" "$path" --repo-type dataset --revision "$revision" \
        --force-download --local-dir "$WORK/download" || return
    bash "$RESTORE" "$WORK/download/$path" "$receipt" "$dest"
}

download_restore "$UNIT/evidence/archive.json" "$WORK/transfer" &&
    download_restore "$UNIT/../codex-4338/evidence/archive.json" "$WORK/parent" &&
    bash "$UNIT/archive/replay.sh" "$WORK/transfer" "$WORK/parent" "$BASE" "$WORK/replay"
```

The cold replay checks every manifest member, then rebuilds the fixed
selection from the pinned #4338 fixtures and compares every selection and
source row byte for byte. The original extraction input manifest uses
absolute Wick paths. Replay compares the ordered source digests and produces
a private relocation-only extraction pin for the qualification driver. The
ordinary client reproduces the consensus assembly, feature capture, bound
openings and qualified-pair evidence byte for byte. TRAIN staging reproduces
every input tape, source file, manifest and zero-weight payload byte for
byte. The seven locked two-answer articles and 20 locked answerability
articles stay in the authenticated parent seal, outside the validation
and TRAIN capture paths.

The archived ordinary-client qualification is instrument-inconclusive. The
qualified validation sample contains 15 reciprocal pairs and 24
answerability pairs, and it misses the registered per-title coverage floor.
Optimizer updates, GPU dispatch, fit scoring and locked evaluation are
`NOT_RUN`. The forced packet restore rejects a same-length corrupted
packet before extraction. Replay rejects a modified review ledger before
ordinary-client capture.
