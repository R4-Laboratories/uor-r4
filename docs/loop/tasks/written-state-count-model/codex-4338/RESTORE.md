# Restore the natural reciprocal and answerability source qualification

The private packet retains the frozen SQuAD 2.0 answerability roster, the 24/7/7
reciprocal positive partition, source and adjudication ledgers, qualified TRAIN
rows, native ordinary-client feature captures, and compact device tapes. It
contains the original official files, exact tokenizer, parent source fixtures,
and the executable ordinary client and interval tools. The large balanced D11
artifact comes from the earlier pinned restore. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/0bc863b1414a6cbe7aeaf9a7253e9431132a3aa3/results/codex/natural-reciprocal-20261127/natural-reciprocal.tar.zst) ·
[Packet receipt](evidence/archive.json) ·
[Balanced D11 restore](../codex-4322/RESTORE.md) ·
[Parent client provenance](../codex-4337/RESTORE.md) ·
[Dataset attribution](archive/ATTRIBUTION.md)

The archived roster holds private assessment text and labels. Keep them in
custody until their registered gate. The replay tools reopen the corpus
mechanically to check identity and bounds and emit only aggregate roster
results. The semantic assembly and native replay consume only TRAIN rows.
[Account of the seal and exposure](holdout/README.md)

Use Wick with the populated offline Cargo cache and the authenticated
Hugging Face client. Set `UNIT` to the absolute path of this source directory.
Choose a fresh absolute `DOWNLOAD`, `RESTORE`, and `REPLAY` in the unit's
private Wick directory.

```sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
RUNS=/mnt/plexpool/uor-r4-corpus/runs
BALANCED="$RUNS/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
receipt="$UNIT/evidence/archive.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
packet=$(jq -er .path "$receipt")
hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private == true and .sha == $revision'
hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$DOWNLOAD/$packet" "$receipt" "$RESTORE"

umask 077
mkdir "$REPLAY"
drivers="$RESTORE/source/docs/loop/tasks/written-state-count-model/codex-4338/archive"
bash "$drivers/replay-roster.sh" "$RESTORE" "$REPLAY/roster"
bash "$drivers/replay-sources.sh" "$RESTORE" "$REPLAY/source"
bash "$drivers/replay-qualification.sh" "$RESTORE" "$REPLAY/qualification"
bash "$drivers/replay-native.sh" "$RESTORE" "$BALANCED" "$REPLAY/native"
```

The restore checks the compressed packet before extraction and verifies every
member. Wick's revision-pinned forced download matched all 39,604,897 bytes,
and a fresh extraction verified all 281 members. The restore rejected a
same-length corrupted copy before extraction. Source replay regenerates the
149 prospective reciprocal pairs and
130 prospective answerability contrasts. Qualification replay reassembles
the private, adjudicated TRAIN tape. Native replay verifies the original
client and fit binaries against the #4337 receipt, builds the current fit and
opening tools from archived source, and checks the raw label-blind ordinary
client candidates, original opening IDs, coverage, inventories and TRAIN tapes
byte for byte. The replay compares manifest entries for the captured content
because source manifests retain execution paths. Its final native record is
`native-v2`. The earlier `native-v1` ledger omitted an independently identified
locked-copy-stop exclusion reason and remains in custody.

The roster replay reproduces 1,394 exact official questions and checks the
original UTF-8 offsets, common source windows, prefix and positive copy limits.
Six mutated roster controls and a truncated assessment seal are rejected.
The native replay regenerates three adversarial controls for wrong original
bytes, forged NoRead source IDs and a missing native NoRead choice. It
checks their private output digests against the archived receipt. The same
replay rebuilds the matched original and natural TRAIN groups and tapes. Each
mix has 264 rows with 204 source and 60 NoRead targets in the same polarity
order. The packet retains the prior TRAIN features and coverage used for that
control, together with the paired natural features and coverage.

The archived matched-schedule filters ran on Wick with jq 1.8.1. The source
tree also explicitly parenthesizes the group and permutation bindings for
jq 1.7. The portable filters regenerated the same complete matched manifest
on Wick, including both case orders and four packed tapes.
